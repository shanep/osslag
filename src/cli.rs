//! The `osslag` command line.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use chrono::{DateTime, FixedOffset, NaiveDate};
use clap::{Args, Parser, Subcommand};

use crate::engine::{Config, Engine, Files, Step, score_dataset};
use crate::github::{GitHub, metadata_to_batch, pull_requests_to_batch};
use crate::io::write_parquet;
use crate::progress::ProgressObserver;
use crate::vcs::url::construct_repo_local_path;

#[derive(Parser)]
#[command(name = "osslag", version, about = "OSS Lag - Technical Lag tools for Open Source Software Projects.")]
pub struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Clone a single Git repository into DEST_DIR/{owner}--{repo}.
    Clone {
        repo_url: String,
        #[arg(long, default_value = "./cache/repos")]
        dest_dir: PathBuf,
    },
    /// Fetch GitHub repository metadata and save it to {cache}/metadata.parquet.
    GetMetadata {
        repo_url: String,
        #[arg(long, env = "CACHE_DIR", default_value = "./cache")]
        cache: PathBuf,
    },
    /// Show the GitHub API rate limit.
    RateLimit,
    /// Fetch every pull request of a repository and save them to {cache}/pull_requests.parquet.
    PullRequests {
        repo_url: String,
        #[arg(long, env = "CACHE_DIR", default_value = "./cache")]
        cache: PathBuf,
    },
    /// Show the current cache directory.
    ShowCache,
    /// Score every merged package with MALTA and write {distro}_malta_scores.parquet.
    Score {
        #[arg(long, default_value = "debian")]
        distro: String,
        /// End of the evaluation window (YYYY-MM-DD, UTC).
        #[arg(long, value_parser = parse_eval_end)]
        eval_end: DateTime<FixedOffset>,
        #[arg(long, env = "CACHE_DIR", default_value = "./cache")]
        cache: PathBuf,
    },
    /// Dataset pipeline commands for building package analysis datasets.
    #[command(subcommand)]
    Dataset(Dataset),
}

#[derive(Args, Clone)]
struct Common {
    /// Cache directory.
    #[arg(long, env = "CACHE_DIR", default_value = "./cache")]
    cache: PathBuf,
    /// Force re-processing even if cache exists.
    #[arg(long, short)]
    force: bool,
}

#[derive(Args, Clone)]
struct Workers {
    /// Cache directory for cloned repositories (default: {cache}/repos).
    #[arg(long, env = "REPOS_CACHE_DIR")]
    repos_cache: Option<PathBuf>,
    /// Number of parallel workers.
    #[arg(long, env = "MAX_WORKERS", default_value_t = 4)]
    max_workers: usize,
    /// Only use the first N merged packages (for testing).
    #[arg(long, hide = true)]
    limit: Option<usize>,
}

#[derive(Args, Clone)]
struct RunArgs {
    /// The Linux distribution to process.
    #[arg(long, default_value = "debian")]
    distro: String,
    /// One or more releases. Repeat the flag or separate them with commas.
    #[arg(long = "release", required = true)]
    releases: Vec<String>,
    #[command(flatten)]
    common: Common,
    #[command(flatten)]
    workers: Workers,
    /// Only load commits on or after this date (YYYY-MM-DD, UTC). Default: full history.
    /// For MALTA scoring, use a date no later than the baseline window start
    /// (eval_end minus 42 months with the default windows).
    #[arg(long, value_parser = parse_since_date)]
    commits_since: Option<i64>,
}

impl RunArgs {
    fn config(&self) -> Config {
        let mut config =
            config(&self.distro, &self.releases, &self.common.cache, self.common.force, Some(&self.workers));
        config.commits_since = self.commits_since;
        config
    }
}

#[derive(Subcommand)]
enum Dataset {
    /// Run the full pipeline: fetch packages, filter repos, extract versions, merge releases,
    /// clone repos, load commits, pull GitHub data. Uses cached data when available.
    Run(RunArgs),
    /// Run the pipeline in a full-screen view: r runs every step, 1 to 8 run one, s stops, q quits.
    Tui(RunArgs),
    /// Step 1: fetch and cache distribution package data for the given releases.
    FetchPackages {
        distro: String,
        #[arg(required = true)]
        releases: Vec<String>,
        #[arg(long, env = "CACHE_DIR", default_value = "./cache")]
        cache: PathBuf,
    },
    /// Step 2: keep only packages whose homepage is a GitHub repository.
    FilterDebianGithubRepos {
        distro: String,
        #[arg(required = true)]
        releases: Vec<String>,
        #[command(flatten)]
        common: Common,
    },
    /// Step 3: add the upstream version taken from each package version.
    ExtractUpstreamVersions {
        distro: String,
        #[arg(required = true)]
        releases: Vec<String>,
        #[command(flatten)]
        common: Common,
    },
    /// Step 4: merge the releases into one table.
    MergeReleases {
        distro: String,
        #[arg(required = true)]
        releases: Vec<String>,
        #[command(flatten)]
        common: Common,
    },
    /// Step 5: clone every upstream GitHub repository.
    CloneUpstreamRepos {
        distro: String,
        #[arg(long, env = "CACHE_DIR", default_value = "./cache")]
        cache: PathBuf,
        #[command(flatten)]
        workers: Workers,
    },
    /// Step 6: load every upstream commit into one table.
    LoadCommitsIntoDataframe {
        distro: String,
        #[command(flatten)]
        common: Common,
        #[command(flatten)]
        workers: Workers,
        /// Only load commits on or after this date (YYYY-MM-DD, UTC). Default: full history.
        /// Changing it requires --force, since checkpoints are reused.
        #[arg(long, value_parser = parse_since_date)]
        since: Option<i64>,
    },
    /// Step 7: fetch GitHub metadata for every upstream repository.
    AllGithubMetadata {
        #[arg(long, default_value = "debian")]
        distro: String,
        #[command(flatten)]
        common: Common,
        #[command(flatten)]
        workers: Workers,
    },
    /// Step 8: fetch GitHub pull requests for every upstream repository.
    AllGithubPullRequests {
        #[arg(long, default_value = "debian")]
        distro: String,
        #[command(flatten)]
        common: Common,
        #[command(flatten)]
        workers: Workers,
    },
}

/// A YYYY-MM-DD date as midnight UTC, in Unix seconds.
pub fn parse_since_date(s: &str) -> Result<i64, String> {
    NaiveDate::parse_from_str(s, "%Y-%m-%d")
        .map(|d| d.and_hms_opt(0, 0, 0).expect("midnight").and_utc().timestamp())
        .map_err(|_| format!("--since must be YYYY-MM-DD, got {s:?}"))
}

fn parse_eval_end(s: &str) -> Result<DateTime<FixedOffset>, String> {
    let secs = parse_since_date(s).map_err(|_| format!("--eval-end must be YYYY-MM-DD, got {s:?}"))?;
    DateTime::from_timestamp(secs, 0).map(|t| t.fixed_offset()).ok_or_else(|| format!("date out of range: {s}"))
}

/// Releases given as repeated values, comma-separated values or both.
pub fn split_releases(values: &[String]) -> Vec<String> {
    values.iter().flat_map(|v| v.split(',')).map(str::trim).filter(|r| !r.is_empty()).map(String::from).collect()
}

fn config(distro: &str, releases: &[String], cache: &Path, force: bool, workers: Option<&Workers>) -> Config {
    let repos_cache = workers.and_then(|w| w.repos_cache.clone()).unwrap_or_else(|| cache.join("repos"));
    Config {
        distro: distro.to_string(),
        releases: split_releases(releases),
        cache_dir: cache.to_path_buf(),
        repos_cache,
        force,
        commits_since: None,
        max_workers: workers.map_or(4, |w| w.max_workers.max(1)),
        limit: workers.and_then(|w| w.limit),
    }
}

fn run_steps(config: Config, steps: &[Step]) -> Result<()> {
    let engine = Engine::new(config, Arc::new(GitHub::from_env()?), Arc::new(ProgressObserver::new()))?;
    for step in steps {
        engine.run_step(*step)?;
    }
    Ok(())
}

fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Clone { repo_url, dest_dir } => {
            let target = construct_repo_local_path(&repo_url, &dest_dir, false)
                .ok_or_else(|| anyhow!("Not a GitHub repository URL: {repo_url}"))?;
            println!("Cloning repository {repo_url} into directory {}", target.display());
            std::fs::create_dir_all(&dest_dir)?;
            match crate::vcs::git::clone_repo(&GitHub::from_env()?, &repo_url, &target, None) {
                Ok(()) => println!("Successfully cloned or updated repository {repo_url} into {}", target.display()),
                Err(e) => return Err(anyhow!("Failed to clone repository {repo_url}: {e}")),
            }
        }
        Command::GetMetadata { repo_url, cache } => {
            std::fs::create_dir_all(&cache)?;
            println!("Fetching metadata for repository {repo_url}");
            let meta = GitHub::from_env()?.repo_metadata(&repo_url)?;
            let path = cache.join("metadata.parquet");
            write_parquet(&path, &metadata_to_batch(&[meta])?)?;
            println!("Metadata saved to {}", path.display());
        }
        Command::RateLimit => {
            let Some(token) = std::env::var("GITHUB_TOKEN").ok().filter(|t| !t.is_empty()) else {
                println!("GITHUB_TOKEN is not set in environment variables.");
                return Ok(());
            };
            let masked = token.get(token.len().saturating_sub(4)..).filter(|_| token.len() >= 4);
            println!("Using token: {}", masked.map_or("<not set>".into(), |m| format!("...{m}")));
            let r = GitHub::new(Some(token))?.rate_limit()?;
            println!("GitHub API Rate Limit: {}/{} remaining (resets at {})", r.remaining, r.limit, r.reset_local());
        }
        Command::PullRequests { repo_url, cache } => {
            std::fs::create_dir_all(&cache)?;
            println!("Fetching pull requests for repository {repo_url}");
            let rows = GitHub::from_env()?.pull_requests(&repo_url, None)?;
            let path = cache.join("pull_requests.parquet");
            write_parquet(&path, &pull_requests_to_batch(&rows)?)?;
            println!("Pull requests saved to {}", path.display());
        }
        Command::ShowCache => {
            let cache = std::env::var("CACHE_DIR").unwrap_or_else(|_| "./cache".into());
            println!("Current cache directory: {cache}");
        }
        Command::Score { distro, eval_end, cache } => {
            let (path, n) = score_dataset(&Files::new(&cache, &distro), eval_end)?;
            println!("Scored {n} packages, saved to {}", path.display());
        }
        Command::Dataset(dataset) => run_dataset(dataset)?,
    }
    Ok(())
}

fn run_dataset(dataset: Dataset) -> Result<()> {
    match dataset {
        Dataset::Run(args) => {
            let config = args.config();
            println!(
                "Dataset pipeline: distro {}, releases {}, cache {}{}",
                config.distro,
                config.releases.join(", "),
                config.cache_dir.display(),
                if config.force { ", --force" } else { "" }
            );
            run_steps(config, &Step::ALL).context("Pipeline failed")?;
            println!("Pipeline completed successfully!");
        }
        Dataset::Tui(args) => crate::tui::run(args.config(), Arc::new(GitHub::from_env()?))?,
        Dataset::FetchPackages { distro, releases, cache } => {
            run_steps(config(&distro, &releases, &cache, false, None), &[Step::FetchPackages])?;
        }
        Dataset::FilterDebianGithubRepos { distro, releases, common } => {
            run_steps(config(&distro, &releases, &common.cache, common.force, None), &[Step::FilterRepos])?;
        }
        Dataset::ExtractUpstreamVersions { distro, releases, common } => {
            run_steps(config(&distro, &releases, &common.cache, common.force, None), &[Step::ExtractVersions])?;
        }
        Dataset::MergeReleases { distro, releases, common } => {
            run_steps(config(&distro, &releases, &common.cache, common.force, None), &[Step::MergeReleases])?;
        }
        Dataset::CloneUpstreamRepos { distro, cache, workers } => {
            run_steps(config(&distro, &[], &cache, false, Some(&workers)), &[Step::CloneRepos])?;
        }
        Dataset::LoadCommitsIntoDataframe { distro, common, workers, since } => {
            let mut config = config(&distro, &[], &common.cache, common.force, Some(&workers));
            config.commits_since = since;
            run_steps(config, &[Step::LoadCommits])?;
        }
        Dataset::AllGithubMetadata { distro, common, workers } => {
            run_steps(config(&distro, &[], &common.cache, common.force, Some(&workers)), &[Step::GitHubMetadata])?;
        }
        Dataset::AllGithubPullRequests { distro, common, workers } => {
            run_steps(config(&distro, &[], &common.cache, common.force, Some(&workers)), &[Step::PullRequests])?;
        }
    }
    Ok(())
}

/// Load `.env`, set up logging, then run the command.
pub fn main() -> ExitCode {
    // A missing .env is fine. Any other problem with it is worth a warning.
    if let Err(e) = dotenvy::dotenv()
        && !e.not_found()
    {
        eprintln!("warning: could not load .env: {e}");
    }
    if let Err(e) = crate::logging::init() {
        eprintln!("warning: could not open the log file: {e}");
    }
    let cli = Cli::parse();
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            tracing::error!("{e:#}");
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use clap::CommandFactory;

    use super::*;

    #[test]
    fn since_is_midnight_utc() {
        assert_eq!(parse_since_date("2021-07-31"), Ok(1_627_689_600));
        assert!(parse_since_date("07/31/2021").is_err());
    }

    #[test]
    fn releases_split_on_commas() {
        let given = vec!["bookworm, trixie".to_string(), "sid".to_string()];
        assert_eq!(split_releases(&given), vec!["bookworm", "trixie", "sid"]);
    }

    #[test]
    fn command_tree_is_valid() {
        Cli::command().debug_assert();
        let cli = Cli::try_parse_from(["osslag", "dataset", "run", "--release", "bookworm,trixie", "--limit", "5"]);
        assert!(cli.is_ok());
    }
}
