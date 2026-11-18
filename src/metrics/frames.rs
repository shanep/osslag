//! Turn the pipeline's parquet tables into per-repo MALTA inputs, and MALTA results back into a table.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use arrow::array::{ArrayRef, BooleanArray, Float64Array, Int64Array, LargeStringArray, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};

use crate::io::columns::{bools, floats, has_column, strings, timestamps_us};
use crate::metrics::malta::{Commit, MaltaResult, PullRequest, RepoData, RepoMeta};
use crate::vcs::trivial::is_trivial_commit;

/// Metadata columns MALTA reads, with the pipeline's names as fallbacks.
/// The metadata step writes `stargazers_count` and friends, while MALTA's own tables say `stars`.
const META_COLUMNS: [(&str, &str); 4] = [
    ("stars", "stargazers_count"),
    ("forks", "forks_count"),
    ("watchers", "watchers_count"),
    ("open_issues", "open_issues_count"),
];

/// Per-repo inputs, keyed by repo URL. Rows with a missing repo URL are dropped, as pandas'
/// `groupby` does.
#[derive(Debug, Default)]
pub struct RepoTables {
    commits: HashMap<String, Vec<Commit>>,
    prs: HashMap<String, Vec<PullRequest>>,
    meta: HashMap<String, Result<RepoMeta, String>>,
}

impl RepoTables {
    pub fn new(commits: &RecordBatch, prs: Option<&RecordBatch>, meta: Option<&RecordBatch>) -> Result<Self> {
        let mut tables = Self::default();
        tables.load_commits(commits)?;
        if let Some(prs) = prs {
            tables.load_prs(prs)?;
        }
        if let Some(meta) = meta {
            tables.load_meta(meta)?;
        }
        Ok(tables)
    }

    fn load_commits(&mut self, batch: &RecordBatch) -> Result<()> {
        if batch.num_rows() == 0 {
            return Ok(());
        }
        let urls = strings(batch, "repo_url")?;
        let dates = timestamps_us(batch, "date")?;
        let trivial: Vec<bool> = if has_column(batch, "is_trivial") {
            bools(batch, "is_trivial")?.into_iter().map(|t| t.unwrap_or(false)).collect()
        } else if has_column(batch, "files") {
            crate::io::columns::string_lists(batch, "files")?
                .iter()
                .map(|files| is_trivial_commit(files.as_deref().unwrap_or(&[])))
                .collect()
        } else {
            bail!("commits table needs an 'is_trivial' or a 'files' column");
        };
        for ((url, date), is_trivial) in urls.into_iter().zip(dates).zip(trivial) {
            if let Some(url) = url {
                self.commits.entry(url).or_default().push(Commit { date, is_trivial });
            }
        }
        Ok(())
    }

    fn load_prs(&mut self, batch: &RecordBatch) -> Result<()> {
        if batch.num_rows() == 0 || !has_column(batch, "repo_url") {
            return Ok(());
        }
        let urls = strings(batch, "repo_url")?;
        let created = timestamps_us(batch, "created_at")?;
        let closed = timestamps_us(batch, "closed_at")?;
        let merged = timestamps_us(batch, "merged_at")?;
        let states = strings(batch, "state")?;
        for i in 0..batch.num_rows() {
            if let Some(url) = &urls[i] {
                self.prs.entry(url.clone()).or_default().push(PullRequest {
                    created_at: created[i],
                    closed_at: closed[i],
                    merged_at: merged[i],
                    // Python's str(None) is "None", which is an unknown state.
                    state: states[i].clone().unwrap_or_else(|| "None".into()),
                });
            }
        }
        Ok(())
    }

    fn load_meta(&mut self, batch: &RecordBatch) -> Result<()> {
        if batch.num_rows() == 0 {
            return Ok(());
        }
        let urls = strings(batch, "repo_url")?;
        let mut counts: Vec<Option<Vec<Option<f64>>>> = Vec::new();
        for (name, fallback) in META_COLUMNS {
            counts.push(if has_column(batch, name) {
                Some(floats(batch, name)?)
            } else if has_column(batch, fallback) {
                Some(floats(batch, fallback)?)
            } else {
                None
            });
        }
        let archived = if has_column(batch, "archived") { Some(bools(batch, "archived")?) } else { None };
        for (i, url) in urls.into_iter().enumerate() {
            let Some(url) = url else { continue };
            // Only the first row for a repo counts, like `iloc[0]`.
            self.meta.entry(url).or_insert_with(|| {
                let count = |k: usize| -> Result<i64, String> {
                    match &counts[k] {
                        None => Ok(0),
                        Some(col) => match col[i] {
                            Some(v) if v.is_finite() => Ok(v.trunc() as i64),
                            Some(v) if v.is_infinite() => Err("cannot convert float infinity to integer".into()),
                            _ => Err("cannot convert float NaN to integer".into()),
                        },
                    }
                };
                Ok(RepoMeta {
                    stars: count(0)?,
                    forks: count(1)?,
                    watchers: count(2)?,
                    open_issues: count(3)?,
                    archived: archived.as_ref().and_then(|a| a[i]).unwrap_or(false),
                })
            });
        }
        Ok(())
    }

    /// Inputs for one repo. Repos with no rows get empty commits, no PRs and default metadata.
    pub fn repo_data(&self, repo_url: &str) -> RepoData {
        let commits = self.commits.get(repo_url).cloned().unwrap_or_default();
        let pull_requests = self.prs.get(repo_url).cloned().unwrap_or_default();
        RepoData {
            n_commits_total: commits.len(),
            n_prs_total: pull_requests.len(),
            commits,
            pull_requests,
            meta: self.meta.get(repo_url).cloned().unwrap_or(Ok(RepoMeta::default())),
        }
    }
}

/// MALTA results as a record batch with the 29 Python columns.
pub fn results_to_batch(results: &[MaltaResult]) -> Result<RecordBatch> {
    fn f(results: &[MaltaResult], get: impl Fn(&MaltaResult) -> Option<f64>) -> ArrayRef {
        Arc::new(results.iter().map(get).collect::<Float64Array>())
    }
    fn i(results: &[MaltaResult], get: impl Fn(&MaltaResult) -> Option<i64>) -> ArrayRef {
        Arc::new(results.iter().map(get).collect::<Int64Array>())
    }
    fn b(results: &[MaltaResult], get: impl Fn(&MaltaResult) -> Option<bool>) -> ArrayRef {
        Arc::new(results.iter().map(get).collect::<BooleanArray>())
    }
    fn s(results: &[MaltaResult], get: impl Fn(&MaltaResult) -> Option<&str>) -> ArrayRef {
        Arc::new(results.iter().map(get).collect::<LargeStringArray>())
    }
    let r = results;
    let columns: Vec<ArrayRef> = vec![
        s(r, |x| Some(&x.source)),
        s(r, |x| Some(&x.repo_url)),
        f(r, |x| x.das_score),
        f(r, |x| x.das_dc),
        f(r, |x| x.das_rc),
        f(r, |x| x.mrs_score),
        f(r, |x| x.mrs_rdec),
        f(r, |x| x.mrs_ddec),
        f(r, |x| x.mrs_popen),
        i(r, |x| x.mrs_n_prs),
        i(r, |x| x.mrs_n_terminated),
        i(r, |x| x.mrs_n_open),
        f(r, |x| x.rmvs_score),
        b(r, |x| x.rmvs_archived),
        f(r, |x| x.rmvs_stars_phi),
        f(r, |x| x.rmvs_forks_phi),
        f(r, |x| x.rmvs_issues_penalty),
        f(r, |x| x.final_score),
        f(r, |x| x.final_score_100),
        i(r, |x| x.n_commits_total),
        i(r, |x| x.n_commits_window),
        i(r, |x| x.n_prs_total),
        i(r, |x| x.n_prs_window),
        i(r, |x| x.stars),
        i(r, |x| x.forks),
        i(r, |x| x.watchers),
        i(r, |x| x.open_issues),
        b(r, |x| x.archived),
        s(r, |x| x.error.as_deref()),
    ];
    let fields: Vec<Field> = MaltaResult::COLUMNS
        .iter()
        .zip(&columns)
        .map(|(name, col)| Field::new(*name, col.data_type().clone(), !matches!(*name, "source" | "repo_url")))
        .collect();
    debug_assert!(fields.iter().all(|f| f.data_type() != &DataType::Null));
    Ok(RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)?)
}
