//! Turn the pipeline's parquet tables into per-repo MALTA inputs, and MALTA results back into a table.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::{Result, bail};
use arrow::array::{ArrayRef, BooleanArray, Float64Array, Int64Array, LargeStringArray, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};

use crate::io::columns::{bools, floats, has_column, strings, timestamps_us};
use crate::metrics::malta::{Commit, MaltaResult, PullRequest, RepoData, RepoMeta, RmvsConstants};
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
        if batch.num_rows() == 0 {
            return Ok(());
        }
        let urls = if has_column(batch, "repo_url") {
            strings(batch, "repo_url")?
        } else if has_column(batch, "html_url") {
            // PR files from the Python version have no repo_url, so take it from the PR link the way
            // the paper's notebooks do: the first five parts of https://github.com/{owner}/{repo}/pull/{n}.
            strings(batch, "html_url")?.into_iter().map(|u| u.map(|u| pr_repo_url(&u))).collect()
        } else {
            return Ok(());
        };
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

/// The repo URL of a pull request link, as Python's `"/".join(url.split("/")[:5])`.
fn pr_repo_url(html_url: &str) -> String {
    html_url.split('/').take(5).collect::<Vec<_>>().join("/")
}

/// numpy's default (linear) quantile, which pandas' `Series.quantile` uses, skipping NaNs.
fn numpy_quantile(mut values: Vec<f64>, q: f64) -> Option<f64> {
    values.retain(|v| !v.is_nan());
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let last = values.len() - 1;
    let index = last as f64 * q;
    let lower = (index.floor() as usize).min(last);
    let gamma = index - index.floor();
    let (a, b) = (values[lower], values[(lower + 1).min(last)]);
    let diff = b - a;
    // numpy's _lerp, which works from the upper value when gamma is at least 0.5.
    Some(if gamma >= 0.5 { b - diff * (1.0 - gamma) } else { a + diff * gamma })
}

/// RMVS saturation constants taken from the data, as the paper's `compute_rmvs_constants` does:
/// each K is the `q` quantile of that count over the metadata rows of the scored repos,
/// truncated to an integer. The other constants come from `base`.
pub fn quantile_rmvs_constants(
    meta: &RecordBatch,
    repo_urls: &HashSet<String>,
    q: f64,
    base: RmvsConstants,
) -> Result<RmvsConstants> {
    if !(0.0..=1.0).contains(&q) {
        bail!("the RMVS quantile must be between 0 and 1, got {q}");
    }
    let urls = strings(meta, "repo_url")?;
    let keep: Vec<bool> = urls.iter().map(|u| u.as_ref().is_some_and(|u| repo_urls.contains(u))).collect();
    let mut ks = [0i64; 4];
    for (k, (name, fallback)) in ks.iter_mut().zip(META_COLUMNS) {
        let column = if has_column(meta, name) { name } else { fallback };
        let values: Vec<f64> =
            floats(meta, column)?.into_iter().zip(&keep).filter(|(_, k)| **k).filter_map(|(v, _)| v).collect();
        let Some(value) = numpy_quantile(values, q) else {
            bail!("no {name} values for the scored repos, so the RMVS quantile is undefined");
        };
        *k = value.trunc() as i64;
    }
    let [k_stars, k_forks, k_watchers, k_issues] = ks;
    Ok(RmvsConstants { k_stars, k_forks, k_watchers, k_issues, ..base })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantile_matches_numpy() {
        // np.quantile([1, 2, 3, 4], q) for each q, and a single value.
        assert_eq!(numpy_quantile(vec![4.0, 1.0, 3.0, 2.0], 0.95), Some(3.8499999999999996));
        assert_eq!(numpy_quantile(vec![1.0, 2.0, 3.0, 4.0], 0.5), Some(2.5));
        assert_eq!(numpy_quantile(vec![1.0, 2.0, 3.0, 4.0], 1.0), Some(4.0));
        assert_eq!(numpy_quantile(vec![7.0, f64::NAN], 0.95), Some(7.0));
        assert_eq!(numpy_quantile(vec![f64::NAN], 0.95), None);
    }

    #[test]
    fn pr_links_give_repo_urls() {
        assert_eq!(pr_repo_url("https://github.com/osrg/gobgp/pull/12"), "https://github.com/osrg/gobgp");
        assert_eq!(pr_repo_url("https://github.com/a"), "https://github.com/a");
    }

    #[test]
    fn quantile_constants_use_only_scored_repos() {
        let meta = RecordBatch::try_from_iter([
            ("repo_url", Arc::new(LargeStringArray::from(vec!["a", "b", "c"])) as ArrayRef),
            ("stargazers_count", Arc::new(Int64Array::from(vec![10, 20, 1000])) as ArrayRef),
            ("forks_count", Arc::new(Int64Array::from(vec![1, 2, 1000])) as ArrayRef),
            ("watchers_count", Arc::new(Int64Array::from(vec![10, 20, 1000])) as ArrayRef),
            ("open_issues_count", Arc::new(Int64Array::from(vec![0, 5, 1000])) as ArrayRef),
        ])
        .unwrap();
        let urls: HashSet<String> = ["a".to_string(), "b".to_string()].into();
        let k = quantile_rmvs_constants(&meta, &urls, 0.95, RmvsConstants::default()).unwrap();
        // 10 + 10 * 0.95 = 19.5, truncated.
        assert_eq!((k.k_stars, k.k_forks, k.k_watchers, k.k_issues), (19, 1, 19, 4));
        assert_eq!(k.alpha_archived, RmvsConstants::default().alpha_archived);
        assert!(quantile_rmvs_constants(&meta, &urls, 1.5, RmvsConstants::default()).is_err());
    }
}
