//! Maintenance-Aware Lag and Technical Abandonment (MALTA) scoring.
//!
//! Implements the metric from "MALTA: Maintenance-Aware Technical Lag Estimation to Address
//! Software Abandonment" (Panter & Eisty). Three maintenance signals are combined:
//!
//! - Development Activity Score: `S_dev = min(1, D_c) * R_c` with
//!   `D_c = (C_e/|W_e|) / (C_b/|W_b|)` and `R_c = exp(-t_last/tau)`.
//! - Maintainer Responsiveness Score: `S_resp = R_dec * (1 - D_dec) * (1 - P_open)`.
//! - Repository Metadata Viability Score: log-saturated stars, forks, watchers and an
//!   open-issue penalty, scaled down for archived repos.
//!
//! `S_final = 0.55*S_dev + 0.35*S_resp + 0.10*S_meta`, renormalized over the components present.
//!
//! The arithmetic follows the Python implementation operation for operation so scores match it
//! exactly. Times are microseconds since the Unix epoch in UTC, the resolution Python used.

use chrono::{DateTime, Datelike, FixedOffset, Months};
use rayon::prelude::*;
use thiserror::Error;

use crate::metrics::pysum::py_sum;

const MICROS_PER_DAY: i64 = 86_400_000_000;

/// Microseconds since the Unix epoch, UTC.
pub type Micros = i64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Commit {
    /// `None` when the source date was missing. Such commits fall outside every window.
    pub date: Option<Micros>,
    pub is_trivial: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PullRequest {
    pub created_at: Option<Micros>,
    pub closed_at: Option<Micros>,
    pub merged_at: Option<Micros>,
    /// "open" or "closed". Any other state is skipped but still counts toward |P|.
    pub state: String,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RepoMeta {
    pub stars: i64,
    pub forks: i64,
    pub watchers: i64,
    pub open_issues: i64,
    pub archived: bool,
}

/// Everything MALTA needs about one repository.
#[derive(Debug, Clone)]
pub struct RepoData {
    pub commits: Vec<Commit>,
    pub pull_requests: Vec<PullRequest>,
    /// `Err` when the metadata row could not be read (for example a missing count), which makes
    /// the whole repo an error row, as in Python.
    pub meta: Result<RepoMeta, String>,
    /// All commit rows for the repo, including trivial ones and rows outside the windows.
    pub n_commits_total: usize,
    /// All PR rows for the repo.
    pub n_prs_total: usize,
}

impl Default for RepoData {
    fn default() -> Self {
        Self {
            commits: Vec::new(),
            pull_requests: Vec::new(),
            meta: Ok(RepoMeta::default()),
            n_commits_total: 0,
            n_prs_total: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaltaConstants {
    pub eval_months: u32,
    pub baseline_months: u32,
}

impl Default for MaltaConstants {
    fn default() -> Self {
        Self { eval_months: 18, baseline_months: 24 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DasConstants {
    /// Exponential decay time constant in days (1/e life).
    pub tau_days: f64,
}

impl Default for DasConstants {
    fn default() -> Self {
        Self { tau_days: 180.0 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MrsConstants {
    /// Time reference for decision timeliness in days.
    pub tref_days: i64,
}

impl Default for MrsConstants {
    fn default() -> Self {
        Self { tref_days: 180 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RmvsConstants {
    pub k_stars: i64,
    pub k_forks: i64,
    pub k_watchers: i64,
    pub k_issues: i64,
    pub alpha_archived: f64,
    pub beta_stars: f64,
    pub beta_forks: f64,
    pub beta_watchers: f64,
    pub beta_issues: f64,
}

impl Default for RmvsConstants {
    fn default() -> Self {
        Self {
            k_stars: 10_000,
            k_forks: 10_000,
            k_watchers: 10_000,
            k_issues: 10_000,
            alpha_archived: 0.7,
            beta_stars: 0.25,
            beta_forks: 0.25,
            beta_watchers: 0.25,
            beta_issues: 0.25,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AggregateConstants {
    pub w_dev: f64,
    pub w_resp: f64,
    pub w_meta: f64,
}

impl Default for AggregateConstants {
    fn default() -> Self {
        Self { w_dev: 0.55, w_resp: 0.35, w_meta: 0.10 }
    }
}

/// All tunable constants in one place.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Constants {
    pub malta: MaltaConstants,
    pub das: DasConstants,
    pub mrs: MrsConstants,
    pub rmvs: RmvsConstants,
    pub aggregate: AggregateConstants,
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum MaltaError {
    #[error("window arithmetic overflowed")]
    WindowOverflow,
    #[error("window_baseline_days and window_eval_days must be > 0")]
    EmptyWindow,
    #[error("K must be positive.")]
    NonPositiveK,
    #[error("alpha_archived must be in [0,1].")]
    AlphaOutOfRange,
    #[error("beta weights must sum to 1.0 exactly.")]
    BetaSum,
    #[error("Sum of active weights must be > 0.")]
    NoWeights,
    #[error("closed pull request has neither merged_at nor closed_at")]
    ClosedPrWithoutDate,
    #[error("{0}")]
    Meta(String),
}

/// A half-open time window `[start, end)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Window {
    pub start: Micros,
    pub end: Micros,
    /// Calendar days between the start and end dates in the evaluation time zone.
    pub ordinal_days: i64,
}

impl Window {
    pub fn contains(&self, t: Micros) -> bool {
        self.start <= t && t < self.end
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DasComponents {
    pub s_dev: f64,
    pub d_c: f64,
    pub r_c: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MrsComponents {
    /// `None` when no PRs were created in the evaluation window.
    pub s_resp: Option<f64>,
    pub r_dec: f64,
    pub d_dec: f64,
    pub p_open: f64,
    pub n_prs: usize,
    pub n_terminated: usize,
    pub n_open: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RmvsComponents {
    pub s_meta: f64,
    pub stars_phi: f64,
    pub forks_phi: f64,
    pub watchers_phi: f64,
    pub open_issues_penalty: f64,
    pub archived: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AggregateComponents {
    pub s_final: f64,
    pub s_final_100: f64,
    pub s_dev: f64,
    pub s_resp: Option<f64>,
    pub s_meta: f64,
}

/// MALTA scorer for one evaluation date.
#[derive(Debug, Clone, Copy)]
pub struct Malta {
    pub eval_window: Window,
    pub baseline_window: Window,
    pub constants: Constants,
}

/// Python's `max(lo, min(hi, x))`.
fn clamp01(x: f64) -> f64 {
    let m = if x < 1.0 { x } else { 1.0 };
    if m > 0.0 { m } else { 0.0 }
}

/// Python's `min(1.0, x)`: returns 1.0 unless `x` is strictly smaller.
fn min1(x: f64) -> f64 {
    if x < 1.0 { x } else { 1.0 }
}

/// `statistics.median`: the middle value, or the mean of the two middle values.
fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.total_cmp(b));
    let n = xs.len();
    if n % 2 == 1 { xs[n / 2] } else { (xs[n / 2 - 1] + xs[n / 2]) / 2.0 }
}

/// Whole days in a duration, floored like Python's `timedelta.days`.
fn floor_days(us: i64) -> i64 {
    us.div_euclid(MICROS_PER_DAY)
}

/// Log-saturating normalization `min(1, log(1+x)/log(1+K))`.
fn phi(x: i64, k: i64) -> Result<f64, MaltaError> {
    let x = x.max(0);
    if k <= 0 {
        return Err(MaltaError::NonPositiveK);
    }
    Ok(min1((x as f64).ln_1p() / (k as f64).ln_1p()))
}

impl Malta {
    /// Build the evaluation and baseline windows ending at `eval_end`.
    ///
    /// Months are subtracted on the calendar in `eval_end`'s time zone, clamping the day of the
    /// month (Mar 31 minus one month is Feb 28 or 29), exactly like `dateutil.relativedelta`.
    pub fn new(eval_end: DateTime<FixedOffset>, constants: Constants) -> Result<Self, MaltaError> {
        let eval_end = eval_end.with_nanosecond_truncated();
        let eval_start =
            eval_end.checked_sub_months(Months::new(constants.malta.eval_months)).ok_or(MaltaError::WindowOverflow)?;
        let base_start = eval_start
            .checked_sub_months(Months::new(constants.malta.baseline_months))
            .ok_or(MaltaError::WindowOverflow)?;
        let ordinal = |d: &DateTime<FixedOffset>| i64::from(d.date_naive().num_days_from_ce());
        Ok(Self {
            eval_window: Window {
                start: eval_start.timestamp_micros(),
                end: eval_end.timestamp_micros(),
                ordinal_days: ordinal(&eval_end) - ordinal(&eval_start),
            },
            baseline_window: Window {
                start: base_start.timestamp_micros(),
                end: eval_start.timestamp_micros(),
                ordinal_days: ordinal(&eval_start) - ordinal(&base_start),
            },
            constants,
        })
    }

    /// Development Activity Score. Trivial commits are excluded unless `include_trivial`.
    pub fn development_activity_score(
        &self,
        commits: &[Commit],
        include_trivial: bool,
    ) -> Result<DasComponents, MaltaError> {
        let (bw, ew) = (self.baseline_window, self.eval_window);
        if bw.ordinal_days <= 0 || ew.ordinal_days <= 0 {
            return Err(MaltaError::EmptyWindow);
        }
        let mut baseline: Vec<Micros> = Vec::new();
        let mut eval: Vec<Micros> = Vec::new();
        for c in commits {
            let Some(d) = c.date else { continue };
            if !include_trivial && c.is_trivial {
                continue;
            }
            if bw.contains(d) {
                baseline.push(d);
            } else if ew.contains(d) {
                eval.push(d);
            }
        }

        let rate_b = baseline.len() as f64 / bw.ordinal_days as f64;
        let rate_e = eval.len() as f64 / ew.ordinal_days as f64;
        let d_c = if rate_b == 0.0 { if rate_e > 0.0 { 1.0 } else { 0.0 } } else { rate_e / rate_b };

        let candidates = if eval.is_empty() { &baseline } else { &eval };
        let r_c = match candidates.iter().max() {
            Some(&last) => {
                let diff_us = ew.end - last;
                let t_last = (diff_us as f64 / 1e6) / 86400.0;
                let t_last = if 0.0 > t_last { 0.0 } else { t_last };
                (-t_last / self.constants.das.tau_days).exp()
            }
            None => 0.0,
        };
        Ok(DasComponents { s_dev: clamp01(min1(d_c) * r_c), d_c, r_c })
    }

    /// Maintainer Responsiveness Score over PRs created in the evaluation window.
    pub fn maintainer_responsiveness_score(&self, prs: &[PullRequest]) -> Result<MrsComponents, MaltaError> {
        let none =
            MrsComponents { s_resp: None, r_dec: 0.0, d_dec: 0.0, p_open: 0.0, n_prs: 0, n_terminated: 0, n_open: 0 };
        let ew = self.eval_window;
        let in_window: Vec<&PullRequest> =
            prs.iter().filter(|pr| pr.created_at.is_some_and(|c| ew.contains(c))).collect();
        if in_window.is_empty() {
            return Ok(none);
        }
        let mut terminated: Vec<&PullRequest> = Vec::new();
        let mut open: Vec<&PullRequest> = Vec::new();
        for pr in &in_window {
            match pr.state.as_str() {
                "closed" => terminated.push(pr),
                "open" => open.push(pr),
                other => tracing::warn!("Unknown PR state '{other}'; skipping PR"),
            }
        }
        if terminated.is_empty() {
            return Ok(MrsComponents { s_resp: Some(0.0), n_prs: in_window.len(), n_open: open.len(), ..none });
        }

        let r_dec = terminated.len() as f64 / in_window.len() as f64;
        let tref = self.constants.mrs.tref_days as f64;
        let delays = terminated
            .iter()
            .map(|pr| {
                let created = pr.created_at.expect("filtered to PRs with a creation date");
                let decided = pr.merged_at.or(pr.closed_at).ok_or(MaltaError::ClosedPrWithoutDate)?;
                Ok(min1(floor_days(decided - created) as f64 / tref))
            })
            .collect::<Result<Vec<f64>, MaltaError>>()?;
        let d_dec = median(delays);

        let p_open = if open.is_empty() {
            0.0
        } else {
            let ages = open
                .iter()
                .map(|pr| min1(floor_days(ew.end - pr.created_at.expect("filtered")) as f64 / tref))
                .collect();
            median(ages)
        };

        let s_resp = r_dec * (1.0 - d_dec) * (1.0 - p_open);
        Ok(MrsComponents {
            s_resp: Some(clamp01(s_resp)),
            r_dec,
            d_dec,
            p_open,
            n_prs: in_window.len(),
            n_terminated: terminated.len(),
            n_open: open.len(),
        })
    }

    /// Repository Metadata Viability Score.
    pub fn repo_metadata_viability_score(&self, meta: &RepoMeta) -> Result<RmvsComponents, MaltaError> {
        let c = &self.constants.rmvs;
        if !(0.0..=1.0).contains(&c.alpha_archived) {
            return Err(MaltaError::AlphaOutOfRange);
        }
        let beta_sum = py_sum([c.beta_stars, c.beta_forks, c.beta_watchers, c.beta_issues]);
        if (beta_sum - 1.0).abs() > 1e-9 {
            return Err(MaltaError::BetaSum);
        }
        let s = phi(meta.stars, c.k_stars)?;
        let f = phi(meta.forks, c.k_forks)?;
        let w = phi(meta.watchers, c.k_watchers)?;
        let i = phi(meta.open_issues, c.k_issues)?;
        let i_pen = 1.0 - i;
        let linear = c.beta_stars * s + c.beta_forks * f + c.beta_watchers * w + c.beta_issues * i_pen;
        let a = if meta.archived { 1.0 } else { 0.0 };
        let a_pen = 1.0 - c.alpha_archived * a;
        Ok(RmvsComponents {
            s_meta: clamp01(a_pen * linear),
            stars_phi: s,
            forks_phi: f,
            watchers_phi: w,
            open_issues_penalty: i_pen,
            archived: meta.archived,
        })
    }

    /// Weighted final score. An archived repo has its responsiveness forced to 0, and the
    /// weights are renormalized over the components that are present.
    pub fn final_aggregation_score(
        &self,
        das: &DasComponents,
        mrs: &MrsComponents,
        rmvs: &RmvsComponents,
    ) -> Result<AggregateComponents, MaltaError> {
        let w = &self.constants.aggregate;
        let s_resp = if rmvs.archived { Some(0.0) } else { mrs.s_resp };
        let mut terms = vec![(w.w_dev, das.s_dev)];
        if let Some(r) = s_resp {
            terms.push((w.w_resp, r));
        }
        terms.push((w.w_meta, rmvs.s_meta));
        let wsum = py_sum(terms.iter().map(|(w, _)| *w));
        if wsum <= 0.0 {
            return Err(MaltaError::NoWeights);
        }
        let s_final = clamp01(py_sum(terms.iter().map(|(w, v)| (w / wsum) * v)));
        Ok(AggregateComponents { s_final, s_final_100: 100.0 * s_final, s_dev: das.s_dev, s_resp, s_meta: rmvs.s_meta })
    }

    /// Score one repository end to end.
    pub fn score(&self, source: &str, repo_url: &str, data: &RepoData) -> MaltaResult {
        self.try_score(source, repo_url, data).unwrap_or_else(|e| MaltaResult::error(source, repo_url, e.to_string()))
    }

    fn try_score(&self, source: &str, repo_url: &str, data: &RepoData) -> Result<MaltaResult, MaltaError> {
        let meta = data.meta.clone().map_err(MaltaError::Meta)?;
        let das = self.development_activity_score(&data.commits, false)?;
        let mrs = self.maintainer_responsiveness_score(&data.pull_requests)?;
        let rmvs = self.repo_metadata_viability_score(&meta)?;
        let fin = self.final_aggregation_score(&das, &mrs, &rmvs)?;
        let ew = self.eval_window;
        Ok(MaltaResult {
            source: source.to_string(),
            repo_url: repo_url.to_string(),
            das_score: Some(das.s_dev),
            das_dc: Some(das.d_c),
            das_rc: Some(das.r_c),
            mrs_score: mrs.s_resp,
            mrs_rdec: Some(mrs.r_dec),
            mrs_ddec: Some(mrs.d_dec),
            mrs_popen: Some(mrs.p_open),
            mrs_n_prs: Some(mrs.n_prs as i64),
            mrs_n_terminated: Some(mrs.n_terminated as i64),
            mrs_n_open: Some(mrs.n_open as i64),
            rmvs_score: Some(rmvs.s_meta),
            rmvs_archived: Some(rmvs.archived),
            rmvs_stars_phi: Some(rmvs.stars_phi),
            rmvs_forks_phi: Some(rmvs.forks_phi),
            rmvs_issues_penalty: Some(rmvs.open_issues_penalty),
            final_score: Some(fin.s_final),
            final_score_100: Some(fin.s_final_100),
            n_commits_total: Some(data.n_commits_total as i64),
            n_commits_window: Some(
                data.commits.iter().filter(|c| c.date.is_some_and(|d| ew.contains(d))).count() as i64
            ),
            n_prs_total: Some(data.n_prs_total as i64),
            n_prs_window: Some(
                data.pull_requests.iter().filter(|p| p.created_at.is_some_and(|d| ew.contains(d))).count() as i64,
            ),
            stars: Some(meta.stars),
            forks: Some(meta.forks),
            watchers: Some(meta.watchers),
            open_issues: Some(meta.open_issues),
            archived: Some(meta.archived),
            error: None,
        })
    }
}

trait TruncateNanos {
    fn with_nanosecond_truncated(self) -> Self;
}

impl TruncateNanos for DateTime<FixedOffset> {
    /// Python datetimes stop at microseconds.
    fn with_nanosecond_truncated(self) -> Self {
        use chrono::Timelike;
        let ns = self.nanosecond();
        self.with_nanosecond(ns - ns % 1_000).expect("truncating nanoseconds stays valid")
    }
}

/// One row of MALTA output. Field order is the column order of the output parquet file.
#[derive(Debug, Clone, PartialEq)]
pub struct MaltaResult {
    pub source: String,
    pub repo_url: String,
    pub das_score: Option<f64>,
    pub das_dc: Option<f64>,
    pub das_rc: Option<f64>,
    pub mrs_score: Option<f64>,
    pub mrs_rdec: Option<f64>,
    pub mrs_ddec: Option<f64>,
    pub mrs_popen: Option<f64>,
    pub mrs_n_prs: Option<i64>,
    pub mrs_n_terminated: Option<i64>,
    pub mrs_n_open: Option<i64>,
    pub rmvs_score: Option<f64>,
    pub rmvs_archived: Option<bool>,
    pub rmvs_stars_phi: Option<f64>,
    pub rmvs_forks_phi: Option<f64>,
    pub rmvs_issues_penalty: Option<f64>,
    pub final_score: Option<f64>,
    pub final_score_100: Option<f64>,
    pub n_commits_total: Option<i64>,
    pub n_commits_window: Option<i64>,
    pub n_prs_total: Option<i64>,
    pub n_prs_window: Option<i64>,
    pub stars: Option<i64>,
    pub forks: Option<i64>,
    pub watchers: Option<i64>,
    pub open_issues: Option<i64>,
    pub archived: Option<bool>,
    pub error: Option<String>,
}

impl MaltaResult {
    /// Output columns, in order. Pinned to the Python `MaltaResult` field order.
    pub const COLUMNS: [&'static str; 29] = [
        "source",
        "repo_url",
        "das_score",
        "das_dc",
        "das_rc",
        "mrs_score",
        "mrs_rdec",
        "mrs_ddec",
        "mrs_popen",
        "mrs_n_prs",
        "mrs_n_terminated",
        "mrs_n_open",
        "rmvs_score",
        "rmvs_archived",
        "rmvs_stars_phi",
        "rmvs_forks_phi",
        "rmvs_issues_penalty",
        "final_score",
        "final_score_100",
        "n_commits_total",
        "n_commits_window",
        "n_prs_total",
        "n_prs_window",
        "stars",
        "forks",
        "watchers",
        "open_issues",
        "archived",
        "error",
    ];

    fn error(source: &str, repo_url: &str, error: String) -> Self {
        Self {
            source: source.to_string(),
            repo_url: repo_url.to_string(),
            das_score: None,
            das_dc: None,
            das_rc: None,
            mrs_score: None,
            mrs_rdec: None,
            mrs_ddec: None,
            mrs_popen: None,
            mrs_n_prs: None,
            mrs_n_terminated: None,
            mrs_n_open: None,
            rmvs_score: None,
            rmvs_archived: None,
            rmvs_stars_phi: None,
            rmvs_forks_phi: None,
            rmvs_issues_penalty: None,
            final_score: None,
            final_score_100: None,
            n_commits_total: None,
            n_commits_window: None,
            n_prs_total: None,
            n_prs_window: None,
            stars: None,
            forks: None,
            watchers: None,
            open_issues: None,
            archived: None,
            error: Some(error),
        }
    }
}

/// A package to score: its distro source name and upstream repo URL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Package {
    pub source: String,
    pub repo_url: String,
}

/// Score many packages in parallel. Results are sorted by source (stable), unlike Python,
/// which returned them in completion order.
pub fn score_repos(
    packages: &[Package],
    data_for: impl Fn(&str) -> RepoData + Sync,
    eval_end: DateTime<FixedOffset>,
    constants: Constants,
) -> Result<Vec<MaltaResult>, MaltaError> {
    let malta = Malta::new(eval_end, constants)?;
    let mut results: Vec<MaltaResult> =
        packages.par_iter().map(|p| malta.score(&p.source, &p.repo_url, &data_for(&p.repo_url))).collect();
    results.sort_by(|a, b| a.source.cmp(&b.source));
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    fn eval_end() -> DateTime<FixedOffset> {
        Utc.with_ymd_and_hms(2024, 1, 15, 12, 0, 0).unwrap().fixed_offset()
    }

    fn malta(eval_months: u32) -> Malta {
        let constants = Constants { malta: MaltaConstants { eval_months, baseline_months: 24 }, ..Default::default() };
        Malta::new(eval_end(), constants).unwrap()
    }

    fn days_before(days: i64) -> Micros {
        eval_end().timestamp_micros() - days * MICROS_PER_DAY
    }

    fn commit(days: i64, is_trivial: bool) -> Commit {
        Commit { date: Some(days_before(days)), is_trivial }
    }

    #[test]
    fn windows_use_calendar_months() {
        let m = malta(12);
        let start = Utc.with_ymd_and_hms(2023, 1, 15, 12, 0, 0).unwrap().timestamp_micros();
        let base = Utc.with_ymd_and_hms(2021, 1, 15, 12, 0, 0).unwrap().timestamp_micros();
        assert_eq!(m.eval_window.start, start);
        assert_eq!(m.baseline_window, Window { start: base, end: start, ordinal_days: 730 });
        assert_eq!(m.eval_window.ordinal_days, 365);
        let mar31 = Utc.with_ymd_and_hms(2024, 3, 31, 0, 0, 0).unwrap().fixed_offset();
        let constants =
            Constants { malta: MaltaConstants { eval_months: 1, baseline_months: 1 }, ..Default::default() };
        let m = Malta::new(mar31, constants).unwrap();
        assert_eq!(m.eval_window.start, Utc.with_ymd_and_hms(2024, 2, 29, 0, 0, 0).unwrap().timestamp_micros());
    }

    #[test]
    fn das_no_commits_is_zero() {
        let das = malta(12).development_activity_score(&[], false).unwrap();
        assert_eq!(das, DasComponents { s_dev: 0.0, d_c: 0.0, r_c: 0.0 });
    }

    #[test]
    fn das_no_baseline_but_eval_active() {
        let das = malta(12).development_activity_score(&[commit(10, false)], false).unwrap();
        assert_eq!(das.d_c, 1.0);
        assert_eq!(das.r_c, (-10.0f64 / 180.0).exp());
    }

    #[test]
    fn das_trivial_commits_excluded_unless_asked() {
        let m = malta(12);
        let commits = [commit(10, true)];
        assert_eq!(m.development_activity_score(&commits, false).unwrap().s_dev, 0.0);
        assert!(m.development_activity_score(&commits, true).unwrap().s_dev > 0.0);
    }

    #[test]
    fn das_falls_back_to_baseline_recency() {
        let das = malta(12).development_activity_score(&[commit(400, false)], false).unwrap();
        assert_eq!(das.d_c, 0.0);
        assert_eq!(das.s_dev, 0.0);
        assert!(das.r_c > 0.0);
    }

    fn pr(created_days: i64, decided_days: Option<i64>, state: &str) -> PullRequest {
        PullRequest {
            created_at: Some(days_before(created_days)),
            closed_at: decided_days.map(days_before),
            merged_at: None,
            state: state.into(),
        }
    }

    #[test]
    fn mrs_cases() {
        let m = malta(12);
        assert_eq!(m.maintainer_responsiveness_score(&[]).unwrap().s_resp, None);
        let open_only = m.maintainer_responsiveness_score(&[pr(10, None, "open")]).unwrap();
        assert_eq!(open_only.s_resp, Some(0.0));
        assert_eq!(open_only.n_open, 1);
        let quick =
            m.maintainer_responsiveness_score(&[pr(10, Some(10), "closed"), pr(20, Some(20), "closed")]).unwrap();
        assert_eq!(quick.s_resp, Some(1.0));
        // Unknown states still count toward |P|.
        let draft = m.maintainer_responsiveness_score(&[pr(10, Some(10), "closed"), pr(20, None, "draft")]).unwrap();
        assert_eq!(draft.r_dec, 0.5);
        // Even counts average the two middle delays: 18 and 54 days -> (0.1 + 0.3) / 2.
        let slow =
            m.maintainer_responsiveness_score(&[pr(100, Some(82), "closed"), pr(100, Some(46), "closed")]).unwrap();
        assert_eq!(slow.d_dec, (18.0 / 180.0 + 54.0 / 180.0) / 2.0);
        let bad = PullRequest { closed_at: None, ..pr(10, None, "closed") };
        assert_eq!(m.maintainer_responsiveness_score(&[bad]), Err(MaltaError::ClosedPrWithoutDate));
    }

    #[test]
    fn rmvs_cases() {
        let m = malta(12);
        assert_eq!(m.repo_metadata_viability_score(&RepoMeta::default()).unwrap().s_meta, 0.25);
        let popular = RepoMeta { stars: 10_000, forks: 10_000, watchers: 10_000, open_issues: 0, archived: false };
        assert_eq!(m.repo_metadata_viability_score(&popular).unwrap().s_meta, 1.0);
        let archived = RepoMeta { archived: true, ..popular };
        assert!((m.repo_metadata_viability_score(&archived).unwrap().s_meta - 0.3).abs() < 1e-12);
    }

    #[test]
    fn archived_forces_responsiveness_to_zero() {
        let m = malta(12);
        let data = RepoData {
            commits: vec![commit(5, false)],
            meta: Ok(RepoMeta { archived: true, ..Default::default() }),
            ..Default::default()
        };
        let r = m.score("pkg", "https://github.com/a/b", &data);
        assert_eq!(r.mrs_score, None);
        let das = m.development_activity_score(&data.commits, false).unwrap();
        let expected =
            clamp01((0.55 / (0.55 + 0.35 + 0.10)) * das.s_dev + (0.35 / 1.0) * 0.0 + (0.10 / 1.0) * 0.25 * 0.3);
        assert!((r.final_score.unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    fn meta_error_becomes_error_row() {
        let data = RepoData { meta: Err("cannot convert float NaN to integer".into()), ..Default::default() };
        let r = malta(12).score("pkg", "u", &data);
        assert_eq!(r.error.as_deref(), Some("cannot convert float NaN to integer"));
        assert_eq!(r.final_score, None);
    }
}
