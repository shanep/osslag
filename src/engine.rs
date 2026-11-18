//! The dataset pipeline: eight steps from a distribution's package lists to upstream commits,
//! GitHub metadata and pull requests.
//!
//! Every step skips work that is already on disk, so a stopped or failed run picks up where it
//! left off. Steps 6 to 8 save one checkpoint per package in `{cache}/*_checkpoints/{source}.parquet`
//! and combine them into one file at the end. A task that fails leaves `{source}.failed` next to
//! the checkpoints, so later runs skip it until `--force`.

use std::collections::{HashMap, HashSet};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use anyhow::{Context, Result, bail};
use arrow::array::{ArrayRef, LargeStringArray, RecordBatch, RecordBatchOptions, new_null_array};
use arrow::compute::{cast, concat_batches};
use arrow::datatypes::{Field, Schema, SchemaRef};
use chrono::{DateTime, FixedOffset};
use tracing::{error, info, warn};

use crate::distro::{DistroHandler, get_handler};
use crate::github::{GitHubApi, metadata_to_batch, pull_requests_to_batch};
use crate::io::table::StringTable;
use crate::io::{read_parquet, write_parquet};
use crate::metrics::frames::{RepoTables, results_to_batch};
use crate::metrics::malta::{Constants, Package, score_repos};
use crate::vcs::git::{clone_repo, commits_to_batch, load_commits};
use crate::vcs::url::construct_repo_local_path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Step {
    FetchPackages,
    FilterRepos,
    ExtractVersions,
    MergeReleases,
    CloneRepos,
    LoadCommits,
    GitHubMetadata,
    PullRequests,
}

impl Step {
    pub const ALL: [Step; 8] = [
        Step::FetchPackages,
        Step::FilterRepos,
        Step::ExtractVersions,
        Step::MergeReleases,
        Step::CloneRepos,
        Step::LoadCommits,
        Step::GitHubMetadata,
        Step::PullRequests,
    ];

    /// 1 to 8, as the CLI and the docs number them.
    pub fn number(self) -> usize {
        Self::ALL.iter().position(|s| *s == self).expect("listed") + 1
    }

    pub fn from_number(n: usize) -> Option<Step> {
        n.checked_sub(1).and_then(|i| Self::ALL.get(i).copied())
    }

    pub fn title(self) -> &'static str {
        match self {
            Step::FetchPackages => "Fetch packages",
            Step::FilterRepos => "Filter GitHub repos",
            Step::ExtractVersions => "Extract upstream versions",
            Step::MergeReleases => "Merge releases",
            Step::CloneRepos => "Clone repos",
            Step::LoadCommits => "Load commits",
            Step::GitHubMetadata => "GitHub metadata",
            Step::PullRequests => "Pull requests",
        }
    }
}

/// Where each pipeline file lives.
#[derive(Debug, Clone)]
pub struct Files {
    pub cache: PathBuf,
    pub distro: String,
}

impl Files {
    pub fn new(cache: &Path, distro: &str) -> Self {
        Self { cache: cache.to_path_buf(), distro: distro.to_string() }
    }

    fn file(&self, name: &str) -> PathBuf {
        self.cache.join(format!("{}_{name}.parquet", self.distro))
    }

    pub fn all_packages(&self, release: &str) -> PathBuf {
        self.file(&format!("{release}_all_packages"))
    }

    pub fn filtered_packages(&self, release: &str) -> PathBuf {
        self.file(&format!("{release}_filtered_packages"))
    }

    pub fn packages_with_versions(&self, release: &str) -> PathBuf {
        self.file(&format!("{release}_packages_with_upstream_versions"))
    }

    pub fn merged_packages(&self) -> PathBuf {
        self.file("merged_releases_packages")
    }

    pub fn dropped_after_merge(&self) -> PathBuf {
        self.file("dropped_after_merge")
    }

    pub fn all_commits(&self) -> PathBuf {
        self.file("all_upstream_commits")
    }

    pub fn all_metadata(&self) -> PathBuf {
        self.file("all_upstream_metadata")
    }

    pub fn all_pull_requests(&self) -> PathBuf {
        self.file("all_upstream_pull_requests")
    }

    pub fn malta_scores(&self) -> PathBuf {
        self.file("malta_scores")
    }

    pub fn commit_checkpoints(&self) -> PathBuf {
        self.cache.join("commit_checkpoints")
    }

    pub fn metadata_checkpoints(&self) -> PathBuf {
        self.cache.join("github_metadata_checkpoints")
    }

    pub fn pr_checkpoints(&self) -> PathBuf {
        self.cache.join("github_pr_checkpoints")
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub distro: String,
    pub releases: Vec<String>,
    pub cache_dir: PathBuf,
    pub repos_cache: PathBuf,
    pub force: bool,
    /// Only load commits at or after this Unix time.
    pub commits_since: Option<i64>,
    pub max_workers: usize,
    /// Only work on the first N merged packages in steps 5 to 8 (for testing).
    pub limit: Option<usize>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            distro: "debian".into(),
            releases: Vec::new(),
            cache_dir: PathBuf::from("./cache"),
            repos_cache: PathBuf::from("./cache/repos"),
            force: false,
            commits_since: None,
            max_workers: 4,
            limit: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    Info,
    Warn,
    Error,
}

/// What happened in one step.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct StepSummary {
    /// Tasks run plus tasks skipped.
    pub total: usize,
    pub skipped: usize,
    pub completed: usize,
    pub failed: usize,
    /// The stop flag ended the step before every task ran.
    pub stopped: bool,
}

/// Receives pipeline events, to draw progress. Called from worker threads.
pub trait Observer: Send + Sync {
    /// A step begins, before its tasks are counted.
    fn step_begin(&self, _step: Step) {}
    /// The step's tasks are counted and about to run.
    fn step_start(&self, _step: Step, _tasks: usize, _skipped: usize, _workers: usize) {}
    fn task_start(&self, _worker: usize, _task: &str) {}
    fn task_done(&self, _worker: usize, _task: &str, _result: &Result<(), String>) {}
    fn step_done(&self, _step: Step, _summary: &StepSummary) {}
    fn log(&self, _level: Level, _message: &str) {}
}

/// An observer that ignores everything.
pub struct NullObserver;

impl Observer for NullObserver {}

pub struct Engine {
    pub config: Config,
    files: Files,
    handler: &'static dyn DistroHandler,
    github: Arc<dyn GitHubApi + Send>,
    observer: Arc<dyn Observer>,
    stop: Arc<AtomicBool>,
}

fn require_file(path: PathBuf, hint: &str) -> Result<PathBuf> {
    if !path.exists() {
        bail!("Required file {} does not exist. {hint}", path.display());
    }
    Ok(path)
}

fn read_table(path: &Path) -> Result<StringTable> {
    StringTable::from_batch(&read_parquet(path)?)
}

/// Delete the checkpoints and failure markers in a directory.
fn clear_checkpoints(dir: &Path) -> Result<usize> {
    let mut removed = 0;
    if !dir.exists() {
        return Ok(0);
    }
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e == "parquet" || e == "failed") {
            std::fs::remove_file(&path)?;
            removed += 1;
        }
    }
    Ok(removed)
}

/// Add `repo_url` and `source` columns holding one value.
fn with_repo_columns(batch: &RecordBatch, repo_url: &str, source: &str) -> Result<RecordBatch> {
    let n = batch.num_rows();
    let mut fields: Vec<Field> = batch.schema().fields().iter().map(|f| f.as_ref().clone()).collect();
    let mut columns: Vec<ArrayRef> = batch.columns().to_vec();
    for (name, value) in [("repo_url", repo_url), ("source", source)] {
        fields.push(Field::new(name, arrow::datatypes::DataType::LargeUtf8, true));
        columns.push(Arc::new(LargeStringArray::from(vec![value; n])));
    }
    Ok(RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)?)
}

/// Cast a checkpoint to the schema the step writes today. Files written by the Python version
/// may lack columns (an empty table has none, and pull request files had no `repo_url` or
/// `source`), so missing columns are filled in from the file name or left null.
fn conform(batch: &RecordBatch, schema: &SchemaRef, source: &str, repo_url: Option<&str>) -> Result<RecordBatch> {
    let n = batch.num_rows();
    let columns = schema
        .fields()
        .iter()
        .map(|f| -> Result<ArrayRef> {
            Ok(match batch.column_by_name(f.name()) {
                Some(c) => cast(c, f.data_type())?,
                None if f.name() == "source" => Arc::new(LargeStringArray::from(vec![source; n])),
                None if f.name() == "repo_url" => Arc::new(LargeStringArray::from(vec![repo_url; n])),
                None => new_null_array(f.data_type(), n),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let options = RecordBatchOptions::new().with_row_count(Some(n));
    Ok(RecordBatch::try_new_with_options(schema.clone(), columns, &options)?)
}

/// Every checkpoint in a directory, sorted by file name, as one batch. None when there are none.
pub fn combine_checkpoints(
    dir: &Path,
    schema: &SchemaRef,
    urls: &HashMap<String, String>,
) -> Result<Option<RecordBatch>> {
    if !dir.exists() {
        return Ok(None);
    }
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "parquet"))
        .collect();
    paths.sort();
    if paths.is_empty() {
        return Ok(None);
    }
    let mut batches = Vec::with_capacity(paths.len());
    for path in &paths {
        let source = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let batch = read_parquet(path)?;
        let url = urls.get(&source).map(String::as_str);
        batches.push(conform(&batch, schema, &source, url).with_context(|| format!("reading {}", path.display()))?);
    }
    Ok(Some(concat_batches(schema, &batches)?))
}

fn commits_schema() -> SchemaRef {
    commits_to_batch(&[], "", "").expect("empty commits batch").schema()
}

fn metadata_schema() -> SchemaRef {
    metadata_to_batch(&[]).expect("empty metadata batch").schema()
}

fn pull_requests_schema() -> SchemaRef {
    let empty = pull_requests_to_batch(&[]).expect("empty PR batch");
    with_repo_columns(&empty, "", "").expect("PR columns").schema()
}

/// One package to fetch or load for steps 6 to 8.
struct PackageTask {
    source: String,
    repo_url: String,
    local_path: Option<PathBuf>,
}

impl Engine {
    pub fn new(config: Config, github: Arc<dyn GitHubApi + Send>, observer: Arc<dyn Observer>) -> Result<Self> {
        let handler = get_handler(&config.distro)?;
        let files = Files::new(&config.cache_dir, &config.distro);
        Ok(Self { config, files, handler, github, observer, stop: Arc::new(AtomicBool::new(false)) })
    }

    pub fn files(&self) -> &Files {
        &self.files
    }

    /// Set this to stop after the tasks that are running finish.
    pub fn stop_flag(&self) -> Arc<AtomicBool> {
        self.stop.clone()
    }

    fn stopped(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    fn info(&self, message: String) {
        info!("{message}");
        self.observer.log(Level::Info, &message);
    }

    fn warn(&self, message: String) {
        warn!("{message}");
        self.observer.log(Level::Warn, &message);
    }

    /// Run every step in order, stopping early if the stop flag is set.
    pub fn run_all(&self) -> Result<Vec<StepSummary>> {
        let mut summaries = Vec::new();
        for step in Step::ALL {
            if self.stopped() {
                break;
            }
            summaries.push(self.run_step(step)?);
        }
        Ok(summaries)
    }

    pub fn run_step(&self, step: Step) -> Result<StepSummary> {
        self.observer.step_begin(step);
        let summary = match step {
            Step::FetchPackages => self.fetch_packages(),
            Step::FilterRepos => self.filter_repos(),
            Step::ExtractVersions => self.extract_versions(),
            Step::MergeReleases => self.merge_releases(),
            Step::CloneRepos => self.clone_repos(),
            Step::LoadCommits => self.load_commits(),
            Step::GitHubMetadata => self.github_metadata(),
            Step::PullRequests => self.pull_requests(),
        };
        match summary {
            Ok(summary) => {
                self.observer.step_done(step, &summary);
                Ok(summary)
            }
            Err(e) => {
                let message = format!("Step {} ({}) failed: {e:#}", step.number(), step.title());
                error!("{message}");
                self.observer.log(Level::Error, &message);
                Err(e)
            }
        }
    }

    /// Run `work` once per release, one at a time. `work` returns false when it skipped.
    fn per_release(&self, step: Step, mut work: impl FnMut(&str) -> Result<bool>) -> Result<StepSummary> {
        let releases = &self.config.releases;
        if releases.is_empty() {
            bail!("No releases given");
        }
        self.observer.step_start(step, releases.len(), 0, 1);
        let mut summary = StepSummary { total: releases.len(), ..Default::default() };
        for release in releases {
            self.observer.task_start(0, release);
            match work(release) {
                Ok(true) => summary.completed += 1,
                Ok(false) => summary.skipped += 1,
                Err(e) => {
                    self.observer.task_done(0, release, &Err(format!("{e:#}")));
                    return Err(e);
                }
            }
            self.observer.task_done(0, release, &Ok(()));
        }
        Ok(summary)
    }

    fn fetch_packages(&self) -> Result<StepSummary> {
        std::fs::create_dir_all(&self.config.cache_dir)?;
        self.per_release(Step::FetchPackages, |rel| {
            let path = self.files.all_packages(rel);
            if path.exists() {
                self.info(format!("Using cached {rel} packages from {}", path.display()));
                return Ok(false);
            }
            self.info(format!("Fetching and caching {rel} packages to {}", path.display()));
            let packages = self
                .handler
                .fetch_packages(rel)
                .with_context(|| format!("Failed to fetch {rel} packages from upstream."))?;
            write_parquet(&path, &packages.to_batch()?)?;
            self.info(format!("Fetched {} {rel} packages", packages.len()));
            Ok(true)
        })
    }

    fn filter_repos(&self) -> Result<StepSummary> {
        self.per_release(Step::FilterRepos, |rel| {
            let out = self.files.filtered_packages(rel);
            if out.exists() && !self.config.force {
                self.info(format!("Using cached filtered packages from {}", out.display()));
                return Ok(false);
            }
            let input = require_file(self.files.all_packages(rel), "Run 'fetch-packages' first.")?;
            self.info(format!("Filtering GitHub repositories for {} release '{rel}'", self.config.distro));
            let packages = read_table(&input)?;
            packages.require("homepage", &format!("Step 2 ({rel})"))?;
            let mut filtered = self.handler.filter_github_repos(&packages)?;
            self.info(format!("Dropped {} packages due to non-GitHub '{rel}'.", packages.len() - filtered.len()));
            self.handler.add_local_repo_cache_path_column(&mut filtered, &self.config.cache_dir)?;
            write_parquet(&out, &filtered.to_batch()?)?;
            self.info(format!("Filtered GitHub repositories saved to {}", out.display()));
            Ok(true)
        })
    }

    fn extract_versions(&self) -> Result<StepSummary> {
        self.per_release(Step::ExtractVersions, |rel| {
            let out = self.files.packages_with_versions(rel);
            if out.exists() && !self.config.force {
                self.info(format!("Using cached upstream versions from {}", out.display()));
                return Ok(false);
            }
            let input = require_file(self.files.filtered_packages(rel), "Run 'filter-debian-github-repos' first.")?;
            self.info(format!("Extracting upstream versions for {} release '{rel}'", self.config.distro));
            let mut packages = read_table(&input)?;
            let version_column = format!("{rel}_version");
            packages.require(&version_column, &format!("Step 3 ({rel})"))?;
            let new_column = format!("{rel}_upstream_version");
            self.handler.add_upstream_version_column(&mut packages, &version_column, &new_column)?;
            let keep: Vec<bool> = packages.require(&new_column, "Step 3")?.iter().map(Option::is_some).collect();
            let kept = packages.filter(&keep);
            self.info(format!(
                "Dropped {} rows with missing upstream versions for release '{rel}'.",
                packages.len() - kept.len()
            ));
            write_parquet(&out, &kept.to_batch()?)?;
            self.info(format!("Upstream versions extracted and saved to {}", out.display()));
            Ok(true)
        })
    }

    fn merge_releases(&self) -> Result<StepSummary> {
        let step = Step::MergeReleases;
        self.observer.step_start(step, 1, 0, 1);
        let out = self.files.merged_packages();
        if out.exists() && !self.config.force {
            self.info(format!("Using cached merged releases from {}", out.display()));
            return Ok(StepSummary { total: 1, skipped: 1, ..Default::default() });
        }
        let task = self.config.releases.join(",");
        self.observer.task_start(0, &task);
        let merged = (|| -> Result<()> {
            let mut tables = Vec::new();
            for rel in &self.config.releases {
                let path =
                    require_file(self.files.packages_with_versions(rel), "Run 'extract-upstream-versions' first.")?;
                self.info(format!(
                    "Loading packages with upstream versions for {} release '{rel}'",
                    self.config.distro
                ));
                let table = read_table(&path)?;
                table.require("source", &format!("Step 4 ({rel})"))?;
                tables.push(table);
            }
            let merged = self.handler.merge_release_packages(&tables)?;
            self.info(format!(
                "Merged releases {:?}. Dropped {} rows that were not present in all releases.",
                self.config.releases,
                merged.unmatched.len()
            ));
            write_parquet(&out, &merged.matched.to_batch()?)?;
            self.info(format!("Merged release packages saved to {}", out.display()));
            let dropped = self.files.dropped_after_merge();
            write_parquet(&dropped, &merged.unmatched.to_batch()?)?;
            self.info(format!("Dropped rows after merge saved to {}", dropped.display()));
            Ok(())
        })();
        self.observer.task_done(0, &task, &merged.as_ref().map(|_| ()).map_err(|e| format!("{e:#}")));
        merged?;
        Ok(StepSummary { total: 1, completed: 1, ..Default::default() })
    }

    /// The merged packages, cut to `--limit`, checked for the columns a step needs.
    fn merged_packages(&self, context: &str, hint: &str, columns: &[&str]) -> Result<StringTable> {
        let path = require_file(self.files.merged_packages(), hint)?;
        let mut table = read_table(&path)?;
        for column in columns {
            table.require(column, context)?;
        }
        if let Some(n) = self.config.limit {
            table = table.head(n);
        }
        Ok(table)
    }

    /// Run tasks on a pool of worker threads, checking the stop flag before each one.
    fn run_pool<T: Sync>(
        &self,
        step: Step,
        tasks: &[T],
        skipped: usize,
        workers: usize,
        id: impl Fn(&T) -> String + Sync,
        work: impl Fn(&T) -> Result<(), String> + Sync,
    ) -> StepSummary {
        let workers = workers.clamp(1, tasks.len().max(1));
        self.observer.step_start(step, tasks.len(), skipped, workers);
        let next = AtomicUsize::new(0);
        let completed = AtomicUsize::new(0);
        let failed = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for worker in 0..workers {
                let (next, completed, failed, id, work) = (&next, &completed, &failed, &id, &work);
                scope.spawn(move || {
                    loop {
                        if self.stopped() {
                            break;
                        }
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(task) = tasks.get(i) else { break };
                        let task_id = id(task);
                        self.observer.task_start(worker, &task_id);
                        let result = catch_unwind(AssertUnwindSafe(|| work(task))).unwrap_or_else(|panic| {
                            let reason = panic
                                .downcast_ref::<String>()
                                .cloned()
                                .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                                .unwrap_or_else(|| "unknown panic".into());
                            Err(format!("panicked: {reason}"))
                        });
                        match &result {
                            Ok(()) => completed.fetch_add(1, Ordering::Relaxed),
                            Err(e) => {
                                warn!("{task_id}: {e}");
                                failed.fetch_add(1, Ordering::Relaxed)
                            }
                        };
                        self.observer.task_done(worker, &task_id, &result);
                    }
                });
            }
        });
        let completed = completed.into_inner();
        let failed = failed.into_inner();
        StepSummary {
            total: tasks.len() + skipped,
            skipped,
            completed,
            failed,
            stopped: completed + failed < tasks.len(),
        }
    }

    fn clone_repos(&self) -> Result<StepSummary> {
        let packages = self.merged_packages("Step 5 (clone)", "Run 'merge-releases' first.", &["upstream_repo_url"])?;
        let repos_cache = &self.config.repos_cache;
        std::fs::create_dir_all(repos_cache)?;
        let mut tasks: Vec<(String, PathBuf)> = Vec::new();
        let mut seen = HashSet::new();
        let (mut skipped, mut invalid) = (0, 0);
        for url in packages.require("upstream_repo_url", "Step 5 (clone)")? {
            let Some(target) = url.as_deref().and_then(|u| construct_repo_local_path(u, repos_cache, false)) else {
                invalid += 1;
                continue;
            };
            let marker =
                repos_cache.join(format!("{}.failed", target.file_name().unwrap_or_default().to_string_lossy()));
            if target.exists() || marker.exists() || !seen.insert(target.clone()) {
                skipped += 1;
                continue;
            }
            tasks.push((url.clone().unwrap_or_default(), target));
        }
        if invalid > 0 {
            self.warn(format!("Skipped {invalid} invalid repository URLs"));
        }
        if tasks.is_empty() {
            self.info(format!("All {skipped} repositories already cloned."));
        }
        Ok(self.run_pool(
            Step::CloneRepos,
            &tasks,
            skipped,
            self.config.max_workers,
            |(url, _)| url.clone(),
            |(url, target)| clone_repo(self.github.as_ref(), url, target, None),
        ))
    }

    /// Tasks for steps 6 to 8: one per source, leaving out sources with a checkpoint or a
    /// failure marker. Returns the tasks, the skipped count and every source's repo URL.
    fn package_tasks(
        &self,
        context: &str,
        hint: &str,
        checkpoints: &Path,
        need_clone: bool,
    ) -> Result<(Vec<PackageTask>, usize, HashMap<String, String>)> {
        let packages = self.merged_packages(context, hint, &["upstream_repo_url", "source"])?;
        let urls = packages.require("upstream_repo_url", context)?;
        let sources = packages.require("source", context)?;
        let mut tasks = Vec::new();
        let mut url_by_source = HashMap::new();
        let mut skipped = 0;
        for (url, source) in urls.iter().zip(sources) {
            let (Some(url), Some(source)) = (url, source) else {
                skipped += 1;
                continue;
            };
            let local_path = construct_repo_local_path(url, &self.config.repos_cache, true);
            let done = checkpoints.join(format!("{source}.parquet")).exists()
                || checkpoints.join(format!("{source}.failed")).exists();
            let duplicate = url_by_source.insert(source.clone(), url.clone()).is_some();
            if done || duplicate || (need_clone && local_path.is_none()) {
                skipped += 1;
                continue;
            }
            tasks.push(PackageTask { source: source.clone(), repo_url: url.clone(), local_path });
        }
        Ok((tasks, skipped, url_by_source))
    }

    /// Run one checkpointed task: write `{source}.parquet`, or `{source}.failed` with the error.
    fn checkpoint(dir: &Path, source: &str, batch: Result<RecordBatch>) -> Result<(), String> {
        let result = batch.and_then(|b| write_parquet(&dir.join(format!("{source}.parquet")), &b));
        result.map_err(|e| {
            let error = format!("{e:#}");
            if let Err(w) = std::fs::write(dir.join(format!("{source}.failed")), &error) {
                warn!("could not write the failure marker for {source}: {w}");
            }
            error
        })
    }

    /// Prepare a checkpoint directory. Returns false when the step's output is already cached.
    fn prepare_checkpoints(&self, dir: &Path, output: &Path, what: &str) -> Result<bool> {
        std::fs::create_dir_all(dir)?;
        if self.config.force {
            let removed = clear_checkpoints(dir)?;
            if removed > 0 {
                self.warn(format!("Removed {removed} checkpoint files from {}", dir.display()));
            }
        }
        if output.exists() && !self.config.force {
            self.info(format!("Using cached {what} from {}", output.display()));
            return Ok(false);
        }
        Ok(true)
    }

    /// Combine a step's checkpoints into its output file, unless the step was stopped.
    fn finish_checkpoints(
        &self,
        summary: &StepSummary,
        dir: &Path,
        schema: &SchemaRef,
        urls: &HashMap<String, String>,
        output: &Path,
        what: &str,
    ) -> Result<()> {
        if summary.stopped {
            self.warn(format!(
                "Stopped before every task ran, so {} was not written. Run again to resume.",
                output.display()
            ));
            return Ok(());
        }
        match combine_checkpoints(dir, schema, urls)? {
            Some(batch) => {
                write_parquet(output, &batch)?;
                self.info(format!("Saved {} {what} to {}", batch.num_rows(), output.display()));
            }
            None => self.warn(format!("No {what} were loaded from any repositories.")),
        }
        Ok(())
    }

    /// The summary for a step whose output is already cached.
    fn cached(&self, step: Step) -> StepSummary {
        self.observer.step_start(step, 0, 0, 1);
        StepSummary::default()
    }

    fn load_commits(&self) -> Result<StepSummary> {
        let dir = self.files.commit_checkpoints();
        let output = self.files.all_commits();
        if !self.prepare_checkpoints(&dir, &output, "commits")? {
            return Ok(self.cached(Step::LoadCommits));
        }
        let hint = "Run 'merge-releases' and 'clone-upstream-repos' first.";
        let (tasks, skipped, urls) = self.package_tasks("Step 6 (commits)", hint, &dir, true)?;
        let since = self.config.commits_since;
        let summary = self.run_pool(
            Step::LoadCommits,
            &tasks,
            skipped,
            self.config.max_workers,
            |t| t.repo_url.clone(),
            |t| {
                let path = t.local_path.as_deref().expect("need_clone keeps only cloned repos");
                let batch =
                    load_commits(path, None, since).and_then(|rows| commits_to_batch(&rows, &t.repo_url, &t.source));
                Self::checkpoint(&dir, &t.source, batch)
            },
        );
        self.finish_checkpoints(&summary, &dir, &commits_schema(), &urls, &output, "commits")?;
        Ok(summary)
    }

    fn log_rate_limit(&self) {
        match self.github.rate_limit() {
            Ok(r) => self.info(format!(
                "GitHub API Rate Limit: {}/{} remaining (resets at {})",
                r.remaining,
                r.limit,
                r.reset_local()
            )),
            Err(e) => self.warn(format!("Could not fetch rate limit info: {e:#}")),
        }
    }

    fn github_metadata(&self) -> Result<StepSummary> {
        let dir = self.files.metadata_checkpoints();
        let output = self.files.all_metadata();
        if !self.prepare_checkpoints(&dir, &output, "GitHub metadata")? {
            return Ok(self.cached(Step::GitHubMetadata));
        }
        let (tasks, skipped, urls) =
            self.package_tasks("Step 7 (metadata)", "Run 'merge-releases' first.", &dir, false)?;
        self.log_rate_limit();
        self.info(format!("Fetching GitHub metadata for {} repositories...", tasks.len()));
        let summary = self.run_pool(
            Step::GitHubMetadata,
            &tasks,
            skipped,
            self.config.max_workers,
            |t| t.repo_url.clone(),
            |t| {
                let batch = self.github.repo_metadata(&t.repo_url).and_then(|m| metadata_to_batch(&[m]));
                Self::checkpoint(&dir, &t.source, batch)
            },
        );
        self.finish_checkpoints(&summary, &dir, &metadata_schema(), &urls, &output, "metadata entries")?;
        Ok(summary)
    }

    fn pull_requests(&self) -> Result<StepSummary> {
        let dir = self.files.pr_checkpoints();
        let output = self.files.all_pull_requests();
        if !self.prepare_checkpoints(&dir, &output, "GitHub pull requests")? {
            return Ok(self.cached(Step::PullRequests));
        }
        let (tasks, skipped, urls) =
            self.package_tasks("Step 8 (pull requests)", "Run 'merge-releases' first.", &dir, false)?;
        self.log_rate_limit();
        self.info(format!("Fetching GitHub pull requests for {} repositories...", tasks.len()));
        let summary = self.run_pool(
            Step::PullRequests,
            &tasks,
            skipped,
            self.config.max_workers,
            |t| t.repo_url.clone(),
            |t| {
                let batch = self
                    .github
                    .pull_requests(&t.repo_url, None)
                    .and_then(|rows| pull_requests_to_batch(&rows))
                    .and_then(|b| with_repo_columns(&b, &t.repo_url, &t.source));
                Self::checkpoint(&dir, &t.source, batch)
            },
        );
        self.finish_checkpoints(&summary, &dir, &pull_requests_schema(), &urls, &output, "pull request entries")?;
        Ok(summary)
    }
}

/// Score every merged package with MALTA and write `{distro}_malta_scores.parquet`.
/// Returns the output path and the number of packages scored.
pub fn score_dataset(files: &Files, eval_end: DateTime<FixedOffset>) -> Result<(PathBuf, usize)> {
    let merged = read_table(&require_file(files.merged_packages(), "Run 'dataset run' first.")?)?;
    let sources = merged.require("source", "score")?;
    let urls = merged.require("upstream_repo_url", "score")?;
    let mut seen = HashSet::new();
    let packages: Vec<Package> = sources
        .iter()
        .zip(urls)
        .filter_map(|(s, u)| Some(Package { source: s.clone()?, repo_url: u.clone()? }))
        .filter(|p| seen.insert((p.source.clone(), p.repo_url.clone())))
        .collect();
    let commits = read_parquet(&require_file(files.all_commits(), "Run 'load-commits-into-dataframe' first.")?)?;
    let optional =
        |p: PathBuf| -> Result<Option<RecordBatch>> { if p.exists() { Ok(Some(read_parquet(&p)?)) } else { Ok(None) } };
    let prs = optional(files.all_pull_requests())?;
    let meta = optional(files.all_metadata())?;
    if prs.is_none() {
        warn!("No pull request file, so every responsiveness score is missing");
    }
    let tables = RepoTables::new(&commits, prs.as_ref(), meta.as_ref())?;
    let results = score_repos(&packages, |url| tables.repo_data(url), eval_end, Constants::default())?;
    let out = files.malta_scores();
    write_parquet(&out, &results_to_batch(&results)?)?;
    Ok((out, results.len()))
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::github::{PullRequestRow, RepoMetadata};

    struct FakeGitHub;

    impl GitHubApi for FakeGitHub {
        fn check_repo_exists(&self, _: &str, _: &str) -> Result<(), String> {
            Err("offline".into())
        }

        fn repo_metadata(&self, repo_url: &str) -> Result<RepoMetadata> {
            if repo_url.ends_with("/bad") {
                bail!("Not Found");
            }
            Ok(RepoMetadata {
                repo_url: repo_url.into(),
                full_name: None,
                description: None,
                stargazers_count: 5,
                forks_count: 1,
                open_issues_count: 0,
                watchers_count: 5,
                created_at: Some(0),
                updated_at: None,
                pushed_at: None,
                archived: false,
                license: None,
                topics: String::new(),
            })
        }

        fn pull_requests(&self, _: &str, _: Option<u32>) -> Result<Vec<PullRequestRow>> {
            Ok(vec![PullRequestRow {
                number: 1,
                title: None,
                state: Some("open".into()),
                user: None,
                created_at: Some("2024-01-02T03:04:05+00:00".into()),
                updated_at: None,
                closed_at: None,
                merged_at: None,
                html_url: None,
            }])
        }
    }

    #[derive(Default)]
    struct Recorder(Mutex<Vec<String>>);

    impl Observer for Recorder {
        fn step_start(&self, step: Step, tasks: usize, skipped: usize, _: usize) {
            self.0.lock().unwrap().push(format!("start {} {tasks} {skipped}", step.number()));
        }
        fn task_done(&self, _: usize, task: &str, result: &Result<(), String>) {
            self.0.lock().unwrap().push(format!("done {task} {}", result.is_ok()));
        }
        fn step_done(&self, step: Step, s: &StepSummary) {
            self.0.lock().unwrap().push(format!("end {} {} {}", step.number(), s.completed, s.failed));
        }
    }

    fn engine(cache: &Path, observer: Arc<dyn Observer>) -> Engine {
        let config = Config {
            releases: vec!["a".into(), "b".into()],
            cache_dir: cache.to_path_buf(),
            repos_cache: cache.join("repos"),
            max_workers: 2,
            ..Config::default()
        };
        Engine::new(config, Arc::new(FakeGitHub), observer).unwrap()
    }

    fn write_merged(cache: &Path) {
        let mut t = StringTable::with_columns(&["source", "upstream_repo_url"]);
        t.push_row(vec![Some("good".into()), Some("https://github.com/o/good".into())]);
        t.push_row(vec![Some("bad".into()), Some("https://github.com/o/bad".into())]);
        write_parquet(&Files::new(cache, "debian").merged_packages(), &t.to_batch().unwrap()).unwrap();
    }

    #[test]
    fn step_numbers() {
        assert_eq!(Step::PullRequests.number(), 8);
        assert_eq!(Step::from_number(5), Some(Step::CloneRepos));
        assert_eq!(Step::from_number(0), None);
    }

    #[test]
    fn metadata_checkpoints_markers_and_resume() {
        let dir = tempfile::tempdir().unwrap();
        write_merged(dir.path());
        let recorder = Arc::new(Recorder::default());
        let e = engine(dir.path(), recorder.clone());
        let summary = e.run_step(Step::GitHubMetadata).unwrap();
        assert_eq!((summary.completed, summary.failed, summary.stopped), (1, 1, false));
        let ck = e.files().metadata_checkpoints();
        assert!(ck.join("good.parquet").exists());
        assert_eq!(std::fs::read_to_string(ck.join("bad.failed")).unwrap(), "Not Found");
        let out = read_parquet(&e.files().all_metadata()).unwrap();
        assert_eq!(out.num_rows(), 1);

        // The output is cached, and without it the checkpoint and marker are both skipped.
        std::fs::remove_file(e.files().all_metadata()).unwrap();
        let again = e.run_step(Step::GitHubMetadata).unwrap();
        assert_eq!((again.skipped, again.completed, again.failed), (2, 0, 0));
        assert!(recorder.0.lock().unwrap().contains(&"start 7 2 0".to_string()));
    }

    #[test]
    fn pull_requests_carry_repo_url_and_source() {
        let dir = tempfile::tempdir().unwrap();
        write_merged(dir.path());
        let e = engine(dir.path(), Arc::new(NullObserver));
        e.run_step(Step::PullRequests).unwrap();
        let t = StringTable::from_batch(&read_parquet(&e.files().all_pull_requests()).unwrap()).unwrap();
        assert_eq!(t.column("source").unwrap(), &[Some("bad".to_string()), Some("good".to_string())]);
        assert_eq!(t.column("repo_url").unwrap()[1].as_deref(), Some("https://github.com/o/good"));
    }

    #[test]
    fn python_checkpoints_without_repo_columns_still_combine() {
        let dir = tempfile::tempdir().unwrap();
        let schema = pull_requests_schema();
        let old = pull_requests_to_batch(&[]).unwrap();
        write_parquet(&dir.path().join("pkg.parquet"), &old).unwrap();
        let urls = HashMap::from([("pkg".to_string(), "https://github.com/o/r".to_string())]);
        let combined = combine_checkpoints(dir.path(), &schema, &urls).unwrap().unwrap();
        assert_eq!(combined.schema(), schema);
    }

    #[test]
    fn stop_flag_stops_between_tasks() {
        let dir = tempfile::tempdir().unwrap();
        write_merged(dir.path());
        let e = engine(dir.path(), Arc::new(NullObserver));
        e.stop_flag().store(true, Ordering::Relaxed);
        let summary = e.run_step(Step::GitHubMetadata).unwrap();
        assert!(summary.stopped);
        assert_eq!(summary.completed + summary.failed, 0);
        assert!(!e.files().all_metadata().exists());
    }

    #[test]
    fn clone_failures_are_marked_and_skipped() {
        let dir = tempfile::tempdir().unwrap();
        write_merged(dir.path());
        let e = engine(dir.path(), Arc::new(NullObserver));
        let first = e.run_step(Step::CloneRepos).unwrap();
        assert_eq!(first.failed, 2);
        assert!(dir.path().join("repos/o--good.failed").exists());
        let second = e.run_step(Step::CloneRepos).unwrap();
        assert_eq!((second.skipped, second.failed), (2, 0));
    }

    #[test]
    fn missing_input_names_the_step_to_run() {
        let dir = tempfile::tempdir().unwrap();
        let e = engine(dir.path(), Arc::new(NullObserver));
        let err = e.run_step(Step::FilterRepos).unwrap_err().to_string();
        assert!(err.contains("Run 'fetch-packages' first."), "{err}");
    }
}
