//! GitHub REST API: repo existence, metadata, pull requests and rate limits.
//!
//! Requests send `GITHUB_TOKEN` as a bearer token when it is set. When the rate limit runs out,
//! requests wait for the reset and try again instead of failing.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use arrow::array::{ArrayRef, BooleanArray, Int64Array, LargeStringArray, RecordBatch, TimestampMicrosecondArray};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use chrono::{DateTime, Local, Utc};
use reqwest::StatusCode;
use reqwest::blocking::{Client, Response};
use serde::Deserialize;
use tracing::{debug, warn};

use crate::io::columns::parse_datetime_us;
use crate::vcs::url::extract_owner_name_repo;

const API: &str = "https://api.github.com";
/// Longest wait for a rate limit reset. GitHub's window is an hour.
const MAX_RATE_WAIT: Duration = Duration::from_secs(3700);

/// The calls the pipeline makes, so tests can swap in a fake.
pub trait GitHubApi: Sync {
    /// `Ok` when the repo exists, otherwise the reason it could not be found.
    fn check_repo_exists(&self, owner: &str, repo: &str) -> Result<(), String>;

    fn repo_metadata(&self, repo_url: &str) -> Result<RepoMetadata> {
        bail!("repo_metadata is not available for {repo_url}")
    }

    fn pull_requests(&self, repo_url: &str, _months: Option<u32>) -> Result<Vec<PullRequestRow>> {
        bail!("pull_requests is not available for {repo_url}")
    }

    fn rate_limit(&self) -> Result<RateLimit> {
        bail!("rate_limit is not available")
    }
}

#[derive(Debug, Clone)]
pub struct GitHub {
    client: Client,
    token: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RateLimit {
    pub limit: i64,
    pub remaining: i64,
    /// Unix seconds when the limit resets.
    pub reset: i64,
    pub authenticated: bool,
}

impl RateLimit {
    /// Reset time in local time, like `03:15:42 PM`.
    pub fn reset_local(&self) -> String {
        DateTime::<Utc>::from_timestamp(self.reset, 0)
            .map(|t| t.with_timezone(&Local).format("%I:%M:%S %p").to_string())
            .unwrap_or_default()
    }
}

/// One row of `{distro}_all_upstream_metadata.parquet`.
#[derive(Debug, Clone, PartialEq)]
pub struct RepoMetadata {
    pub repo_url: String,
    pub full_name: Option<String>,
    pub description: Option<String>,
    pub stargazers_count: i64,
    pub forks_count: i64,
    pub open_issues_count: i64,
    pub watchers_count: i64,
    /// Microseconds since the epoch, UTC.
    pub created_at: Option<i64>,
    pub updated_at: Option<i64>,
    pub pushed_at: Option<i64>,
    pub archived: bool,
    pub license: Option<String>,
    pub topics: String,
}

/// One row of `{distro}_all_upstream_pull_requests.parquet`. Dates are ISO-8601 strings
/// ending in `+00:00`, as Python's `isoformat()` wrote them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PullRequestRow {
    pub number: i64,
    pub title: Option<String>,
    pub state: Option<String>,
    pub user: Option<String>,
    pub created_at: Option<String>,
    pub updated_at: Option<String>,
    pub closed_at: Option<String>,
    pub merged_at: Option<String>,
    pub html_url: Option<String>,
}

#[derive(Deserialize)]
struct ApiLicense {
    spdx_id: Option<String>,
}

#[derive(Deserialize)]
struct ApiRepo {
    full_name: Option<String>,
    description: Option<String>,
    stargazers_count: i64,
    forks_count: i64,
    open_issues_count: i64,
    watchers_count: i64,
    created_at: Option<String>,
    updated_at: Option<String>,
    pushed_at: Option<String>,
    archived: bool,
    license: Option<ApiLicense>,
    #[serde(default)]
    topics: Vec<String>,
}

#[derive(Deserialize)]
struct ApiUser {
    login: String,
}

#[derive(Deserialize)]
struct ApiPull {
    number: i64,
    title: Option<String>,
    state: Option<String>,
    user: Option<ApiUser>,
    created_at: Option<String>,
    updated_at: Option<String>,
    closed_at: Option<String>,
    merged_at: Option<String>,
    html_url: Option<String>,
}

#[derive(Deserialize)]
struct ApiMessage {
    message: Option<String>,
}

#[derive(Deserialize)]
struct ApiRateCore {
    limit: i64,
    remaining: i64,
    reset: i64,
}

#[derive(Deserialize)]
struct ApiRateResources {
    core: ApiRateCore,
}

#[derive(Deserialize)]
struct ApiRateLimit {
    resources: ApiRateResources,
}

/// `2024-01-02T03:04:05Z` as Python's `isoformat()` prints it: `2024-01-02T03:04:05+00:00`.
pub fn iso_utc(s: &str) -> Result<String> {
    let us = parse_datetime_us(s)?;
    let t = DateTime::<Utc>::from_timestamp_micros(us).ok_or_else(|| anyhow!("date out of range: {s}"))?;
    Ok(if us % 1_000_000 == 0 {
        t.format("%Y-%m-%dT%H:%M:%S+00:00").to_string()
    } else {
        t.format("%Y-%m-%dT%H:%M:%S%.6f+00:00").to_string()
    })
}

/// The URL of the `rel="next"` page in a `Link` header.
fn next_link(link: &str) -> Option<String> {
    link.split(',').find_map(|part| {
        let (url, params) = part.split_once(';')?;
        params
            .split(';')
            .any(|p| p.trim() == r#"rel="next""#)
            .then(|| url.trim().trim_start_matches('<').trim_end_matches('>').to_string())
    })
}

fn header_i64(response: &Response, name: &str) -> Option<i64> {
    response.headers().get(name)?.to_str().ok()?.parse().ok()
}

impl GitHub {
    pub fn new(token: Option<String>) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent(concat!("osslag/", env!("CARGO_PKG_VERSION")))
            .build()?;
        Ok(Self { client, token: token.filter(|t| !t.is_empty()) })
    }

    /// A client using `GITHUB_TOKEN` when it is set.
    pub fn from_env() -> Result<Self> {
        Self::new(std::env::var("GITHUB_TOKEN").ok())
    }

    pub fn authenticated(&self) -> bool {
        self.token.is_some()
    }

    /// GET a URL, waiting out the rate limit when it runs out.
    fn get(&self, url: &str) -> Result<Response> {
        loop {
            let mut request = self.client.get(url).header("Accept", "application/vnd.github+json");
            if let Some(token) = &self.token {
                request = request.bearer_auth(token);
            }
            let response = request.send().with_context(|| format!("GET {url}"))?;
            let limited = matches!(response.status(), StatusCode::FORBIDDEN | StatusCode::TOO_MANY_REQUESTS)
                && header_i64(&response, "x-ratelimit-remaining") == Some(0);
            if !limited {
                return Ok(response);
            }
            let reset = header_i64(&response, "x-ratelimit-reset").unwrap_or(0);
            let wait = Duration::from_secs(u64::try_from(reset - Utc::now().timestamp()).unwrap_or(0) + 1);
            if wait > MAX_RATE_WAIT {
                bail!("Rate limited (remaining: 0, resets: {reset})");
            }
            warn!("GitHub rate limit reached, waiting {}s for the reset", wait.as_secs());
            std::thread::sleep(wait);
        }
    }

    /// The API's error message for a failed response, or `HTTP {status}`.
    fn error_message(response: Response) -> String {
        let status = response.status();
        response
            .json::<ApiMessage>()
            .ok()
            .and_then(|m| m.message)
            .unwrap_or_else(|| format!("HTTP {}", status.as_u16()))
    }

    pub fn rate_limit(&self) -> Result<RateLimit> {
        let response = self.get(&format!("{API}/rate_limit"))?;
        if !response.status().is_success() {
            bail!("GitHub API error: {}", Self::error_message(response));
        }
        let core = response.json::<ApiRateLimit>()?.resources.core;
        Ok(RateLimit {
            limit: core.limit,
            remaining: core.remaining,
            reset: core.reset,
            authenticated: self.authenticated(),
        })
    }

    pub fn repo_metadata(&self, repo_url: &str) -> Result<RepoMetadata> {
        let repo = extract_owner_name_repo(repo_url).ok_or_else(|| anyhow!("Invalid repository URL: {repo_url}"))?;
        let response = self.get(&format!("{API}/repos/{}/{}", repo.owner, repo.name))?;
        if !response.status().is_success() {
            bail!("GitHub API error: {}", Self::error_message(response));
        }
        let api: ApiRepo = response.json().context("parsing repository JSON")?;
        let time = |t: &Option<String>| t.as_deref().map(parse_datetime_us).transpose();
        Ok(RepoMetadata {
            repo_url: repo_url.to_string(),
            full_name: api.full_name,
            description: api.description,
            stargazers_count: api.stargazers_count,
            forks_count: api.forks_count,
            open_issues_count: api.open_issues_count,
            watchers_count: api.watchers_count,
            created_at: time(&api.created_at)?,
            updated_at: time(&api.updated_at)?,
            pushed_at: time(&api.pushed_at)?,
            archived: api.archived,
            // Python wrote str(spdx_id), so a license without an SPDX id is "None".
            license: api.license.map(|l| l.spdx_id.unwrap_or_else(|| "None".into())),
            topics: api.topics.join(","),
        })
    }

    /// Pull requests, newest first. With `months`, stop at PRs created more than
    /// `months * 30` days ago.
    pub fn pull_requests(&self, repo_url: &str, months: Option<u32>) -> Result<Vec<PullRequestRow>> {
        if months == Some(0) {
            bail!("months parameter must be a positive integer or None");
        }
        let repo =
            extract_owner_name_repo(repo_url).ok_or_else(|| anyhow!("Invalid GitHub repository URL: {repo_url}"))?;
        let cutoff = months.map(|m| (Utc::now() - chrono::Duration::days(i64::from(m) * 30)).timestamp_micros());
        let mut url = Some(format!(
            "{API}/repos/{}/{}/pulls?state=all&sort=created&direction=desc&per_page=100",
            repo.owner, repo.name
        ));
        let mut rows = Vec::new();
        while let Some(page) = url.take() {
            let response = self.get(&page)?;
            if !response.status().is_success() {
                bail!("GitHub API error: {}", Self::error_message(response));
            }
            url = response.headers().get("link").and_then(|l| l.to_str().ok()).and_then(next_link);
            let pulls: Vec<ApiPull> = response.json().context("parsing pull request JSON")?;
            debug!("{} pull requests on {page}", pulls.len());
            for pr in pulls {
                if let (Some(cutoff), Some(created)) = (cutoff, &pr.created_at)
                    && parse_datetime_us(created)? < cutoff
                {
                    return Ok(rows);
                }
                let iso = |t: &Option<String>| t.as_deref().map(iso_utc).transpose();
                rows.push(PullRequestRow {
                    number: pr.number,
                    title: pr.title,
                    state: pr.state,
                    user: pr.user.map(|u| u.login),
                    created_at: iso(&pr.created_at)?,
                    updated_at: iso(&pr.updated_at)?,
                    closed_at: iso(&pr.closed_at)?,
                    merged_at: iso(&pr.merged_at)?,
                    html_url: pr.html_url,
                });
            }
        }
        Ok(rows)
    }
}

impl GitHubApi for GitHub {
    fn repo_metadata(&self, repo_url: &str) -> Result<RepoMetadata> {
        GitHub::repo_metadata(self, repo_url)
    }

    fn pull_requests(&self, repo_url: &str, months: Option<u32>) -> Result<Vec<PullRequestRow>> {
        GitHub::pull_requests(self, repo_url, months)
    }

    fn rate_limit(&self) -> Result<RateLimit> {
        GitHub::rate_limit(self)
    }

    fn check_repo_exists(&self, owner: &str, repo: &str) -> Result<(), String> {
        let url = format!("{API}/repos/{owner}/{repo}");
        let response = self.get(&url).map_err(|e| format!("{e:#}"))?;
        match response.status() {
            s if s.is_success() => Ok(()),
            StatusCode::NOT_FOUND => {
                let fallback = format!("404 Not Found: {url}");
                Err(response.json::<ApiMessage>().ok().and_then(|m| m.message).unwrap_or(fallback))
            }
            StatusCode::FORBIDDEN => {
                let remaining = response
                    .headers()
                    .get("x-ratelimit-remaining")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("?")
                    .to_string();
                let reset =
                    response.headers().get("x-ratelimit-reset").and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
                Err(format!("Rate limited (remaining: {remaining}, resets: {reset})"))
            }
            _ => {
                let msg = Self::error_message(response);
                warn!("GitHub API error for {owner}/{repo}: {msg}");
                Err(msg)
            }
        }
    }
}

fn utc_micros_type() -> DataType {
    DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()))
}

/// Metadata rows as a record batch with the columns Python wrote.
pub fn metadata_to_batch(rows: &[RepoMetadata]) -> Result<RecordBatch> {
    let s = |f: &dyn Fn(&RepoMetadata) -> Option<&str>| -> ArrayRef {
        Arc::new(rows.iter().map(f).collect::<LargeStringArray>())
    };
    let i = |f: &dyn Fn(&RepoMetadata) -> i64| -> ArrayRef {
        Arc::new(rows.iter().map(|r| Some(f(r))).collect::<Int64Array>())
    };
    let t = |f: &dyn Fn(&RepoMetadata) -> Option<i64>| -> ArrayRef {
        Arc::new(rows.iter().map(f).collect::<TimestampMicrosecondArray>().with_timezone("UTC"))
    };
    let columns: Vec<(&str, ArrayRef)> = vec![
        ("repo_url", s(&|r| Some(&r.repo_url))),
        ("full_name", s(&|r| r.full_name.as_deref())),
        ("description", s(&|r| r.description.as_deref())),
        ("stargazers_count", i(&|r| r.stargazers_count)),
        ("forks_count", i(&|r| r.forks_count)),
        ("open_issues_count", i(&|r| r.open_issues_count)),
        ("watchers_count", i(&|r| r.watchers_count)),
        ("created_at", t(&|r| r.created_at)),
        ("updated_at", t(&|r| r.updated_at)),
        ("pushed_at", t(&|r| r.pushed_at)),
        ("archived", Arc::new(rows.iter().map(|r| Some(r.archived)).collect::<BooleanArray>())),
        ("license", s(&|r| r.license.as_deref())),
        ("topics", s(&|r| Some(&r.topics))),
    ];
    batch(columns)
}

/// Pull request rows as a record batch with the columns Python wrote.
pub fn pull_requests_to_batch(rows: &[PullRequestRow]) -> Result<RecordBatch> {
    let s = |f: &dyn Fn(&PullRequestRow) -> Option<&str>| -> ArrayRef {
        Arc::new(rows.iter().map(f).collect::<LargeStringArray>())
    };
    let columns: Vec<(&str, ArrayRef)> = vec![
        ("number", Arc::new(rows.iter().map(|r| Some(r.number)).collect::<Int64Array>())),
        ("title", s(&|r| r.title.as_deref())),
        ("state", s(&|r| r.state.as_deref())),
        ("user", s(&|r| r.user.as_deref())),
        ("created_at", s(&|r| r.created_at.as_deref())),
        ("updated_at", s(&|r| r.updated_at.as_deref())),
        ("closed_at", s(&|r| r.closed_at.as_deref())),
        ("merged_at", s(&|r| r.merged_at.as_deref())),
        ("html_url", s(&|r| r.html_url.as_deref())),
    ];
    batch(columns)
}

fn batch(columns: Vec<(&str, ArrayRef)>) -> Result<RecordBatch> {
    let fields: Vec<Field> = columns
        .iter()
        .map(|(name, col)| {
            let dt = col.data_type().clone();
            debug_assert!(!matches!(dt, DataType::Timestamp(..)) || dt == utc_micros_type());
            Field::new(*name, dt, true)
        })
        .collect();
    Ok(RecordBatch::try_new(Arc::new(Schema::new(fields)), columns.into_iter().map(|(_, c)| c).collect())?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn link_header() {
        let link = r#"<https://api.github.com/x?page=2>; rel="next", <https://api.github.com/x?page=9>; rel="last""#;
        assert_eq!(next_link(link).as_deref(), Some("https://api.github.com/x?page=2"));
        assert_eq!(next_link(r#"<https://a/x?page=1>; rel="prev""#), None);
    }

    #[test]
    fn isoformat() {
        assert_eq!(iso_utc("2024-01-02T03:04:05Z").unwrap(), "2024-01-02T03:04:05+00:00");
    }

    #[test]
    fn batches() {
        let pr = PullRequestRow {
            number: 1,
            title: Some("t".into()),
            state: Some("open".into()),
            user: None,
            created_at: Some("2024-01-02T03:04:05+00:00".into()),
            updated_at: None,
            closed_at: None,
            merged_at: None,
            html_url: None,
        };
        assert_eq!(pull_requests_to_batch(&[pr]).unwrap().num_columns(), 9);
        assert_eq!(pull_requests_to_batch(&[]).unwrap().num_rows(), 0);
        let meta = RepoMetadata {
            repo_url: "https://github.com/o/r".into(),
            full_name: None,
            description: None,
            stargazers_count: 1,
            forks_count: 2,
            open_issues_count: 3,
            watchers_count: 4,
            created_at: Some(0),
            updated_at: None,
            pushed_at: None,
            archived: false,
            license: None,
            topics: String::new(),
        };
        let b = metadata_to_batch(&[meta]).unwrap();
        assert_eq!(b.schema().field_with_name("created_at").unwrap().data_type(), &utc_micros_type());
    }
}
