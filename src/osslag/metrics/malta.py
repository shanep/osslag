# Maintenance-Aware Lag and Technical Abandonment (MALTA) metrics
from __future__ import annotations

import logging
import math
from statistics import median
from datetime import datetime
from typing import TYPE_CHECKING, Any, NamedTuple, Sequence
import pandas as pd
from dateutil.relativedelta import relativedelta

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from tqdm import tqdm as TqdmType


class EvaluationWindow(NamedTuple):
    """Represents a time window for evaluation."""

    start: datetime
    end: datetime
    days: int


class Commit(NamedTuple):
    """Minimal commit record."""

    date: datetime
    is_trivial: bool = False


class PullRequest(NamedTuple):
    """Minimal pull request record."""

    created_at: datetime
    closed_at: datetime | None
    merged_at: datetime | None
    state: str  # 'open' or 'closed'


class RepoMeta(NamedTuple):
    """Repository Metadata record."""

    stars: int = 0
    forks: int = 0
    watchers: int = 0
    open_issues: int = 0
    archived: bool = False


class DASComponents(NamedTuple):
    """Development Activity Score and its components"""

    s_dev: float  # Development Activity Score
    d_c: float  # Decay Ratio
    r_c: float  # Recency Factor


class MRSComponents(NamedTuple):
    """Maintenance Responsiveness Score and its components"""

    s_resp: float | None  # Responsiveness score (None when no PR data available)
    r_dec: float  # Decision rate
    d_dec: float  # Decision delay (normalized median)
    p_open: float  # Open PR staleness penalty
    n_prs: int  # Total PRs in window
    n_terminated: int  # Closed/merged PRs
    n_open: int  # Still open PRs


class RMVSComponents(NamedTuple):
    """Repository Metadata Viability Score and its components"""

    s_meta: float  # Metadata score
    stars_phi: float  # Normalized stars
    forks_phi: float  # Normalized forks
    watchers_phi: float  # Normalized watchers
    open_issues_penalty: float  # Open issues penalty
    archived: bool  # Archived status


class AggregateScoreComponents(NamedTuple):
    """Final maintenance score and its components"""

    s_final: float  # Final S_dev
    s_final_100: float  # in [0,100]
    s_dev: float  # Development activity score
    s_resp: float | None  # Responsiveness score (None when no PR data available)
    s_meta: float  # Metadata score


class MaltaConstants(NamedTuple):
    """Constants for MALTA metric computations."""

    eval_months: int = 18
    baseline_months: int = 24
    repo_url_column: str = "repo_url"
    repo_dates_column: str = "date"
    repo_is_trivial_column: str = "is_trivial"
    pr_created_at_column: str = "created_at"
    pr_closed_at_column: str = "closed_at"
    pr_merged_at_column: str = "merged_at"
    pr_state_column: str = "state"


class DevelopmentActivityScoreConstants(NamedTuple):
    """Constants for development activity score computation."""

    tau_days: float = 180.0  # Exponential decay time constant in days (1/e life)


class MaintainerResponsivenessScoreConstants(NamedTuple):
    """Constants for maintainer responsiveness score computation."""

    tref_days: int = 180  # Time reference for decision timeliness in days


class RepoViabilityScoreConstants(NamedTuple):
    """Constants for repository viability score computation."""

    K_stars: int = 10_000
    K_forks: int = 10_000
    K_watchers: int = 10_000
    K_issues: int = 10_000
    alpha_archived: float = 0.7
    beta_stars: float = 0.25
    beta_forks: float = 0.25
    beta_watchers: float = 0.25
    beta_issues: float = 0.25


class AggregateScoreConstants(NamedTuple):
    """Weights for final maintenance score aggregation."""

    w_dev: float = 0.55
    w_resp: float = 0.35
    w_meta: float = 0.10


class ExtractedRepoData(NamedTuple):
    """Pre-extracted lightweight data for a single repository.

    Used to pass data to worker processes without pickling full DataFrames.
    """

    commits: tuple[Commit, ...]
    pull_requests: tuple[PullRequest, ...]
    repo_meta: RepoMeta
    n_commits_total: int
    n_prs_total: int


class Malta:
    def __init__(
        self,
        package: str,
        github_repo_url: str,
        eval_end: datetime,
        commits_df: pd.DataFrame,
        pull_requests_df: pd.DataFrame,
        repo_meta_df: pd.DataFrame,
        malta_constants: MaltaConstants | None = None,
        das_constants: DevelopmentActivityScoreConstants | None = None,
        mrs_constants: MaintainerResponsivenessScoreConstants | None = None,
        repo_meta_constants: RepoViabilityScoreConstants | None = None,
        final_agg_constants: AggregateScoreConstants | None = None,
    ):
        self.package = package
        self.github_repo_url = github_repo_url
        self.commits_df = commits_df
        self.pull_requests_df = pull_requests_df
        self.repo_meta_df = repo_meta_df

        # Score components (populated by calling the respective methods)
        self.das: DASComponents | None = None
        self.mrs: MRSComponents | None = None
        self.rmvs: RMVSComponents | None = None
        self.final: AggregateScoreComponents | None = None

        # Cached extracted data (lazily populated)
        self._commits_cache: Sequence[Commit] | None = None
        self._prs_cache: Sequence[PullRequest] | None = None
        self._meta_cache: RepoMeta | None = None

        self.malta_constants = malta_constants if malta_constants is not None else MaltaConstants()
        self.das_constants = das_constants if das_constants is not None else DevelopmentActivityScoreConstants()
        self.mrs_constants = mrs_constants if mrs_constants is not None else MaintainerResponsivenessScoreConstants()
        self.rmv_constants = repo_meta_constants if repo_meta_constants is not None else RepoViabilityScoreConstants()
        self.final_constants = final_agg_constants if final_agg_constants is not None else AggregateScoreConstants()
        # Precompute evaluation and baseline windows
        if eval_end.tzinfo is None:
            raise ValueError("Datetime object must be timezone-aware")
        eval_window_start = eval_end - relativedelta(months=self.malta_constants.eval_months)
        # Evaluation window
        self.eval_window = EvaluationWindow(
            start=eval_window_start,
            end=eval_end,
            days=(eval_end - eval_window_start).days,
        )
        baseline_window_start = eval_window_start - relativedelta(months=self.malta_constants.baseline_months)
        baseline_window_end = eval_window_start
        # Baseline window
        self.baseline_window = EvaluationWindow(
            start=baseline_window_start,
            end=baseline_window_end,
            days=(baseline_window_end - baseline_window_start).days,
        )

    @classmethod
    def from_extracted(
        cls,
        package: str,
        github_repo_url: str,
        eval_end: datetime,
        extracted: ExtractedRepoData,
        malta_constants: MaltaConstants | None = None,
        das_constants: DevelopmentActivityScoreConstants | None = None,
        mrs_constants: MaintainerResponsivenessScoreConstants | None = None,
        repo_meta_constants: RepoViabilityScoreConstants | None = None,
        final_agg_constants: AggregateScoreConstants | None = None,
    ) -> Malta:
        """Create a Malta instance from pre-extracted data.

        Bypasses DataFrame extraction by injecting NamedTuples directly into
        internal caches. The DataFrames stored on the instance are empty
        placeholders — the extraction methods will return the cached data
        without ever touching them.

        This reduces cross-process pickle size by ~85-95% when used with
        ProcessPoolExecutor.
        """
        empty = pd.DataFrame()
        instance = cls(
            package=package,
            github_repo_url=github_repo_url,
            eval_end=eval_end,
            commits_df=empty,
            pull_requests_df=empty,
            repo_meta_df=empty,
            malta_constants=malta_constants,
            das_constants=das_constants,
            mrs_constants=mrs_constants,
            repo_meta_constants=repo_meta_constants,
            final_agg_constants=final_agg_constants,
        )
        # Inject pre-extracted data into caches so extraction methods
        # return immediately without touching the empty DataFrames.
        instance._commits_cache = extracted.commits
        instance._prs_cache = extracted.pull_requests
        instance._meta_cache = extracted.repo_meta
        return instance

    @staticmethod
    def __clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, x))

    @staticmethod
    def __phi_count(x: int, K: int) -> float:
        """Log-saturating normalization: phi(x)=min(1, log(1+x)/log(1+K))."""
        x = max(0, x)
        if K <= 0:
            raise ValueError("K must be positive.")
        return min(1.0, math.log1p(x) / math.log1p(K))

    def get_commits_for_package(self) -> Sequence[Commit]:
        """Extract commits for a given repository URL. Results are cached."""
        if self._commits_cache is not None:
            return self._commits_cache

        repo_url_column = self.malta_constants.repo_url_column
        repo_dates_column = self.malta_constants.repo_dates_column
        repo_is_trivial_column = self.malta_constants.repo_is_trivial_column

        mask = self.commits_df[repo_url_column] == self.github_repo_url
        repo_commits_df = self.commits_df.loc[mask]
        if len(repo_commits_df) == 0:
            self._commits_cache = []
            return self._commits_cache

        # Extract columns as arrays (vectorized datetime conversion)
        dates = pd.to_datetime(repo_commits_df[repo_dates_column], utc=True).dt.to_pydatetime()
        trivials = repo_commits_df[repo_is_trivial_column].to_numpy()

        # Build Commit objects from arrays (faster than iterrows/itertuples)
        self._commits_cache = [Commit(date=d, is_trivial=bool(t)) for d, t in zip(dates, trivials)]
        return self._commits_cache

    def get_pull_requests_for_package(self) -> Sequence[PullRequest]:
        """Extract pull requests for a given repository URL. Results are cached."""
        if self._prs_cache is not None:
            return self._prs_cache

        repo_url_column = self.malta_constants.repo_url_column
        pr_created_at_column = self.malta_constants.pr_created_at_column
        pr_closed_at_column = self.malta_constants.pr_closed_at_column
        pr_merged_at_column = self.malta_constants.pr_merged_at_column
        pr_state_column = self.malta_constants.pr_state_column

        mask = self.pull_requests_df[repo_url_column] == self.github_repo_url
        repo_prs_df = self.pull_requests_df.loc[mask]
        if len(repo_prs_df) == 0:
            self._prs_cache = []
            return self._prs_cache

        # Extract columns as arrays (vectorized datetime conversion)
        created = pd.to_datetime(repo_prs_df[pr_created_at_column], utc=True).dt.to_pydatetime()
        closed = pd.to_datetime(repo_prs_df[pr_closed_at_column], utc=True).dt.to_pydatetime()
        merged = pd.to_datetime(repo_prs_df[pr_merged_at_column], utc=True).dt.to_pydatetime()
        states = repo_prs_df[pr_state_column].to_numpy()

        # Build PullRequest objects from arrays
        self._prs_cache = [
            PullRequest(created_at=c, closed_at=cl, merged_at=m, state=str(s))
            for c, cl, m, s in zip(created, closed, merged, states)
        ]
        return self._prs_cache

    def get_repo_meta_for_package(self) -> RepoMeta:
        """Extract repository metadata for a given repository URL. Results are cached."""
        if self._meta_cache is not None:
            return self._meta_cache

        repo_url_column = self.malta_constants.repo_url_column
        mask = self.repo_meta_df[repo_url_column] == self.github_repo_url
        repo_meta_row = self.repo_meta_df.loc[mask]
        if len(repo_meta_row) == 0:
            self._meta_cache = RepoMeta()
            return self._meta_cache

        row = repo_meta_row.iloc[0]
        self._meta_cache = RepoMeta(
            stars=int(row.get("stars", 0)),
            forks=int(row.get("forks", 0)),
            watchers=int(row.get("watchers", 0)),
            open_issues=int(row.get("open_issues", 0)),
            archived=bool(row.get("archived", False)),
        )
        return self._meta_cache

    def development_activity_score(
        self,
        include_trivial: bool = False,
    ) -> DASComponents:
        """Computes S_dev in [0,1]:

            D_c = (C_e / |W_e|) / (C_b / |W_b|)
            R_c = exp(-t_last / tau)
            S_dev = min(1, D_c) * R_c

        Parameters
        ----------
        include_trivial : bool
            If False, exclude trivial commits from counts and recency calculations.

        Returns
        -------
        DAComponents
            Components of the development activity score.

        Notes
        -----
        - Uses commits from self.commits_df via get_commits_for_package().
        - If baseline rate is 0, we treat D_c as 1 when eval also has activity,
        else 0. This avoids division-by-zero while remaining conservative.
        - t_last is measured since most recent non-trivial commit (unless include_trivial=True).

        """
        commits = self.get_commits_for_package()
        # Partition commits into the baseline and evaluation sets, filtering trivial if needed.
        commits_baseline: list[Commit] = []
        commits_eval: list[Commit] = []
        window_baseline_days: int = self.baseline_window.end.toordinal() - self.baseline_window.start.toordinal()
        window_eval_days: int = self.eval_window.end.toordinal() - self.eval_window.start.toordinal()

        for c in commits:
            if self.baseline_window.start <= c.date < self.baseline_window.end:
                commits_baseline.append(c)
            elif self.eval_window.start <= c.date < self.eval_window.end:
                commits_eval.append(c)

        if window_baseline_days <= 0 or window_eval_days <= 0:
            raise ValueError("window_baseline_days and window_eval_days must be > 0")
        if self.baseline_window.end > self.eval_window.start:
            raise ValueError("Baseline window must end before evaluation window starts.")

        def _filter(cs: Sequence[Commit]) -> list[Commit]:
            if include_trivial:
                return list(cs)
            return [c for c in cs if not c.is_trivial]

        b = _filter(commits_baseline)
        e = _filter(commits_eval)

        C_b = len(b)
        C_e = len(e)

        # Rates per day
        rate_b = C_b / float(window_baseline_days)
        rate_e = C_e / float(window_eval_days)

        # Velocity decay with careful handling when baseline has no commits.
        if rate_b == 0.0:
            D_c = 1.0 if rate_e > 0.0 else 0.0
        else:
            D_c = rate_e / rate_b

        # Recency term based on last non-trivial commit in eval, else fallback to baseline.
        candidates = e if e else b
        if candidates:
            last_commit_time = max(c.date for c in candidates)
            if last_commit_time.tzinfo is None:
                raise ValueError("Commit.authored_at must be timezone-aware (UTC recommended).")
            t_last = max(0.0, (self.eval_window.end - last_commit_time).total_seconds() / 86400.0)
            R_c = math.exp(-t_last / self.das_constants.tau_days)
        else:
            # No commits at all -> fully inactive.
            R_c = 0.0

        S_dev = self.__clamp(min(1.0, D_c) * R_c)

        self.das = DASComponents(d_c=D_c, r_c=R_c, s_dev=S_dev)
        return self.das

    def maintainer_responsiveness_score(
        self,
    ) -> MRSComponents:
        """Compute the PR-outcome-bound Maintainer Responsiveness Score (S_resp).

        Returns
        -------
        MRSComponents
            Components of the maintainer responsiveness score.

        Notes
        -----
        - Uses pull requests from self.pull_requests_df via get_pull_requests_for_package().

        """
        pull_requests = self.get_pull_requests_for_package()
        if not pull_requests:
            # No external contribution signal - Sresp is undefined per paper
            self.mrs = MRSComponents(
                s_resp=None,
                r_dec=0.0,
                d_dec=0.0,
                p_open=0.0,
                n_prs=0,
                n_terminated=0,
                n_open=0,
            )
            return self.mrs

        # Filter PRs to those created within the evaluation window
        P = [pr for pr in pull_requests if self.eval_window.start <= pr.created_at < self.eval_window.end]
        if not P:
            # No PRs in evaluation window
            self.mrs = MRSComponents(
                s_resp=None,
                r_dec=0.0,
                d_dec=0.0,
                p_open=0.0,
                n_prs=0,
                n_terminated=0,
                n_open=0,
            )
            return self.mrs

        # Partition PRs
        P_term = []
        P_open = []

        for pr in P:
            if pr.state == "closed":
                P_term.append(pr)
            elif pr.state == "open":
                P_open.append(pr)
            else:
                logger.warning("Unknown PR state '%s'; skipping PR", pr.state)
                continue

        # If PRs exist but none are terminated, Sresp = 0 per paper
        if not P_term:
            self.mrs = MRSComponents(
                s_resp=0.0,
                r_dec=0.0,
                d_dec=0.0,
                p_open=0.0,
                n_prs=len(P),
                n_terminated=0,
                n_open=len(P_open),
            )
            return self.mrs

        R_dec = len(P_term) / len(P)

        # ---- Decision Timeliness (D_dec) ----
        decision_delays = []
        for pr in P_term:
            closed_time = pr.merged_at if pd.notna(pr.merged_at) else pr.closed_at
            delta_days = (closed_time - pr.created_at).days
            decision_delays.append(min(1.0, delta_days / self.mrs_constants.tref_days))

        D_dec = median(decision_delays)

        # ---- Open PR Staleness Penalty (P_open) ----
        if P_open:
            open_ages = []
            for pr in P_open:
                age_days = (self.eval_window.end - pr.created_at).days
                open_ages.append(min(1.0, age_days / self.mrs_constants.tref_days))
            P_open_penalty = median(open_ages)
        else:
            P_open_penalty = 0.0

        # ---- Responsiveness Aggregation ----
        S_resp = R_dec * (1.0 - D_dec) * (1.0 - P_open_penalty)

        self.mrs = MRSComponents(
            s_resp=self.__clamp(S_resp),
            r_dec=R_dec,
            d_dec=D_dec,
            p_open=P_open_penalty,
            n_prs=len(P),
            n_terminated=len(P_term),
            n_open=len(P_open),
        )

        return self.mrs

    def repo_metadata_viability_score(
        self,
    ) -> RMVSComponents:
        """Compute repository metadata viability score S_meta.

        phi(x) = min(1, log(1+x)/log(1+K))
        S* = phi(stars), F* = phi(forks), W* = phi(watchers), I* = phi(open_issues)
        I_pen = 1 - I*
        A_pen = 1 - alpha * A  (A=1 if archived else 0)

        S_meta = A_pen * (beta_s*S* + beta_f*F* + beta_w*W* + beta_i*I_pen)

        Missing-data handling:
        - If all counts are None: return None.
        - If some counts are missing: renormalize betas over observed fields.

        Notes
        -----
        - Uses repo metadata from self.repo_meta_df via __get_repo_meta_for_package().
        """
        meta = self.get_repo_meta_for_package()
        if not (0.0 <= self.rmv_constants.alpha_archived <= 1.0):
            raise ValueError("alpha_archived must be in [0,1].")
        betas = {
            "stars": self.rmv_constants.beta_stars,
            "forks": self.rmv_constants.beta_forks,
            "watchers": self.rmv_constants.beta_watchers,
            "issues": self.rmv_constants.beta_issues,
        }
        beta_sum = sum(betas.values())
        if abs(beta_sum - 1.0) > 1e-9:
            # Keep strict for reproducibility
            raise ValueError("beta weights must sum to 1.0 exactly.")
        s = self.__phi_count(meta.stars, self.rmv_constants.K_stars)
        f = self.__phi_count(meta.forks, self.rmv_constants.K_forks)
        w = self.__phi_count(meta.watchers, self.rmv_constants.K_watchers)
        i = self.__phi_count(meta.open_issues, self.rmv_constants.K_issues)
        i_pen = 1.0 - i

        linear = (
            betas["stars"] * s
            + betas["forks"] * f
            + betas["watchers"] * w
            + betas["issues"] * i_pen
        )

        A = 1.0 if meta.archived else 0.0
        A_pen = 1.0 - self.rmv_constants.alpha_archived * A
        self.rmvs = RMVSComponents(
            s_meta=self.__clamp(A_pen * linear),
            stars_phi=s,
            forks_phi=f,
            watchers_phi=w,
            open_issues_penalty=i_pen,
            archived=meta.archived,
        )

        return self.rmvs

    def final_aggregation_score(
        self,
    ) -> AggregateScoreComponents:
        """Aggregate S_dev, S_resp, S_meta into S_final with missing-data handling.

        Base:
        S_final = w_dev*S_dev + w_resp*S_resp + w_meta*S_meta

        Missing responsiveness:
        - If S_resp is None: renormalize over {S_dev, S_meta} that exist.

        Archived override:
        - If archived and S_resp is None: set S_resp = 0.0 (explicit cessation)
            before aggregation (still renormalizes if S_meta is None).

        Raises
        ------
        ValueError
            If component scores have not been computed yet.
        """
        if self.das is None or self.mrs is None or self.rmvs is None:
            raise ValueError(
                "Component scores must be computed before final aggregation. "
                "Call development_activity_score(), maintainer_responsiveness_score(), "
                "and repo_metadata_viability_score() first."
            )

        s_dev = self.das.s_dev
        s_resp = self.mrs.s_resp
        s_meta = self.rmvs.s_meta

        # If the repo is archived, treat as 0.0
        if self.rmvs.archived:
            s_resp = 0.0

        terms: list[tuple[float, float]] = [(self.final_constants.w_dev, s_dev)]

        if s_resp is not None:
            terms.append((self.final_constants.w_resp, s_resp))
        if s_meta is not None:
            terms.append((self.final_constants.w_meta, s_meta))

        wsum = sum(w for w, _ in terms)
        if wsum <= 0:
            raise ValueError("Sum of active weights must be > 0.")
        s_final = self.__clamp(sum((w / wsum) * v for w, v in terms))

        self.final = AggregateScoreComponents(
            s_final=s_final,
            s_dev=s_dev,
            s_resp=s_resp,
            s_meta=s_meta,
            s_final_100=100.0 * s_final,
        )

        return self.final


class MaltaResult(NamedTuple):
    """Result from scoring a single package with MALTA metrics."""

    source: str
    repo_url: str
    # DAS components
    das_score: float | None
    das_dc: float | None
    das_rc: float | None
    # MRS components
    mrs_score: float | None
    mrs_rdec: float | None
    mrs_ddec: float | None
    mrs_popen: float | None
    mrs_n_prs: int | None
    mrs_n_terminated: int | None
    mrs_n_open: int | None
    # RMVS components
    rmvs_score: float | None
    rmvs_archived: bool | None
    rmvs_stars_phi: float | None
    rmvs_forks_phi: float | None
    rmvs_issues_penalty: float | None
    # Final score
    final_score: float | None
    final_score_100: float | None
    # Counts
    n_commits_total: int | None
    n_commits_window: int | None
    n_prs_total: int | None
    n_prs_window: int | None
    # Metadata
    stars: int | None
    forks: int | None
    watchers: int | None
    open_issues: int | None
    archived: bool | None
    # Error tracking
    error: str | None


def _score_single_repo(
    source: str,
    repo_url: str,
    repo_commits_df: pd.DataFrame,
    repo_prs_df: pd.DataFrame,
    repo_meta_df: pd.DataFrame,
    eval_end: datetime,
    malta_constants: MaltaConstants | None,
    das_constants: DevelopmentActivityScoreConstants | None,
    mrs_constants: MaintainerResponsivenessScoreConstants | None,
    repo_meta_constants: RepoViabilityScoreConstants | None,
    final_agg_constants: AggregateScoreConstants | None,
) -> MaltaResult:
    """Score a single repository. Internal function used by score_repos."""
    try:
        m = Malta(
            package=source,
            github_repo_url=repo_url,
            eval_end=eval_end,
            commits_df=repo_commits_df,
            pull_requests_df=repo_prs_df,
            repo_meta_df=repo_meta_df,
            malta_constants=malta_constants,
            das_constants=das_constants,
            mrs_constants=mrs_constants,
            repo_meta_constants=repo_meta_constants,
            final_agg_constants=final_agg_constants,
        )

        das = m.development_activity_score()
        mrs = m.maintainer_responsiveness_score()
        rmvs = m.repo_metadata_viability_score()
        final = m.final_aggregation_score()

        commits = m.get_commits_for_package()
        prs = m.get_pull_requests_for_package()
        meta = m.get_repo_meta_for_package()

        # Count commits/PRs within the evaluation window (not total for repo)
        commits_in_eval = [c for c in commits if m.eval_window.start <= c.date < m.eval_window.end]
        prs_in_eval = [pr for pr in prs if m.eval_window.start <= pr.created_at < m.eval_window.end]

        return MaltaResult(
            source=source,
            repo_url=repo_url,
            das_score=das.s_dev,
            das_dc=das.d_c,
            das_rc=das.r_c,
            mrs_score=mrs.s_resp,
            mrs_rdec=mrs.r_dec,
            mrs_ddec=mrs.d_dec,
            mrs_popen=mrs.p_open,
            mrs_n_prs=mrs.n_prs,
            mrs_n_terminated=mrs.n_terminated,
            mrs_n_open=mrs.n_open,
            rmvs_score=rmvs.s_meta,
            rmvs_archived=rmvs.archived,
            rmvs_stars_phi=rmvs.stars_phi,
            rmvs_forks_phi=rmvs.forks_phi,
            rmvs_issues_penalty=rmvs.open_issues_penalty,
            final_score=final.s_final,
            final_score_100=final.s_final_100,
            n_commits_total=len(repo_commits_df),
            n_commits_window=len(commits_in_eval),
            n_prs_total=len(repo_prs_df),
            n_prs_window=len(prs_in_eval),
            stars=meta.stars,
            forks=meta.forks,
            watchers=meta.watchers,
            open_issues=meta.open_issues,
            archived=meta.archived,
            error=None,
        )
    except Exception as e:
        return MaltaResult(
            source=source,
            repo_url=repo_url,
            das_score=None,
            das_dc=None,
            das_rc=None,
            mrs_score=None,
            mrs_rdec=None,
            mrs_ddec=None,
            mrs_popen=None,
            mrs_n_prs=None,
            mrs_n_terminated=None,
            mrs_n_open=None,
            rmvs_score=None,
            rmvs_archived=None,
            rmvs_stars_phi=None,
            rmvs_forks_phi=None,
            rmvs_issues_penalty=None,
            final_score=None,
            final_score_100=None,
            n_commits_total=None,
            n_commits_window=None,
            n_prs_total=None,
            n_prs_window=None,
            stars=None,
            forks=None,
            watchers=None,
            open_issues=None,
            archived=None,
            error=str(e),
        )


def _extract_repo_data(
    repo_url: str,
    commits_df: pd.DataFrame,
    prs_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    malta_constants: MaltaConstants | None = None,
) -> ExtractedRepoData:
    """Pre-extract repo data from DataFrames into lightweight NamedTuples.

    Called once in the main process so that worker processes receive
    compact tuples instead of full DataFrames.
    """
    mc = malta_constants or MaltaConstants()

    # Extract commits
    mask = commits_df[mc.repo_url_column] == repo_url
    repo_commits = commits_df.loc[mask]
    if len(repo_commits) > 0:
        dates = pd.to_datetime(repo_commits[mc.repo_dates_column], utc=True).dt.to_pydatetime()
        trivials = repo_commits[mc.repo_is_trivial_column].to_numpy()
        commits = tuple(Commit(date=d, is_trivial=bool(t)) for d, t in zip(dates, trivials))
    else:
        commits = ()

    # Extract PRs
    mask = prs_df[mc.repo_url_column] == repo_url
    repo_prs = prs_df.loc[mask]
    if len(repo_prs) > 0:
        created = pd.to_datetime(repo_prs[mc.pr_created_at_column], utc=True).dt.to_pydatetime()
        closed = pd.to_datetime(repo_prs[mc.pr_closed_at_column], utc=True).dt.to_pydatetime()
        merged = pd.to_datetime(repo_prs[mc.pr_merged_at_column], utc=True).dt.to_pydatetime()
        states = repo_prs[mc.pr_state_column].to_numpy()
        pull_requests = tuple(
            PullRequest(created_at=c, closed_at=cl, merged_at=m, state=str(s))
            for c, cl, m, s in zip(created, closed, merged, states)
        )
    else:
        pull_requests = ()

    # Extract metadata
    mask = meta_df[mc.repo_url_column] == repo_url
    repo_meta_row = meta_df.loc[mask]
    if len(repo_meta_row) > 0:
        row = repo_meta_row.iloc[0]
        repo_meta = RepoMeta(
            stars=int(row.get("stars", 0)),
            forks=int(row.get("forks", 0)),
            watchers=int(row.get("watchers", 0)),
            open_issues=int(row.get("open_issues", 0)),
            archived=bool(row.get("archived", False)),
        )
    else:
        repo_meta = RepoMeta()

    return ExtractedRepoData(
        commits=commits,
        pull_requests=pull_requests,
        repo_meta=repo_meta,
        n_commits_total=len(repo_commits),
        n_prs_total=len(repo_prs),
    )


def _score_single_repo_fast(
    source: str,
    repo_url: str,
    extracted: ExtractedRepoData,
    eval_end: datetime,
    malta_constants: MaltaConstants | None,
    das_constants: DevelopmentActivityScoreConstants | None,
    mrs_constants: MaintainerResponsivenessScoreConstants | None,
    repo_meta_constants: RepoViabilityScoreConstants | None,
    final_agg_constants: AggregateScoreConstants | None,
) -> MaltaResult:
    """Score a single repository from pre-extracted lightweight data.

    Same logic as _score_single_repo but accepts an ExtractedRepoData tuple
    instead of DataFrames, drastically reducing pickle overhead for
    cross-process transfer.
    """
    try:
        m = Malta.from_extracted(
            package=source,
            github_repo_url=repo_url,
            eval_end=eval_end,
            extracted=extracted,
            malta_constants=malta_constants,
            das_constants=das_constants,
            mrs_constants=mrs_constants,
            repo_meta_constants=repo_meta_constants,
            final_agg_constants=final_agg_constants,
        )

        das = m.development_activity_score()
        mrs = m.maintainer_responsiveness_score()
        rmvs = m.repo_metadata_viability_score()
        final = m.final_aggregation_score()

        commits = m.get_commits_for_package()
        prs = m.get_pull_requests_for_package()
        meta = m.get_repo_meta_for_package()

        commits_in_eval = [c for c in commits if m.eval_window.start <= c.date < m.eval_window.end]
        prs_in_eval = [pr for pr in prs if m.eval_window.start <= pr.created_at < m.eval_window.end]

        return MaltaResult(
            source=source,
            repo_url=repo_url,
            das_score=das.s_dev,
            das_dc=das.d_c,
            das_rc=das.r_c,
            mrs_score=mrs.s_resp,
            mrs_rdec=mrs.r_dec,
            mrs_ddec=mrs.d_dec,
            mrs_popen=mrs.p_open,
            mrs_n_prs=mrs.n_prs,
            mrs_n_terminated=mrs.n_terminated,
            mrs_n_open=mrs.n_open,
            rmvs_score=rmvs.s_meta,
            rmvs_archived=rmvs.archived,
            rmvs_stars_phi=rmvs.stars_phi,
            rmvs_forks_phi=rmvs.forks_phi,
            rmvs_issues_penalty=rmvs.open_issues_penalty,
            final_score=final.s_final,
            final_score_100=final.s_final_100,
            n_commits_total=extracted.n_commits_total,
            n_commits_window=len(commits_in_eval),
            n_prs_total=extracted.n_prs_total,
            n_prs_window=len(prs_in_eval),
            stars=meta.stars,
            forks=meta.forks,
            watchers=meta.watchers,
            open_issues=meta.open_issues,
            archived=meta.archived,
            error=None,
        )
    except Exception as e:
        return MaltaResult(
            source=source,
            repo_url=repo_url,
            das_score=None,
            das_dc=None,
            das_rc=None,
            mrs_score=None,
            mrs_rdec=None,
            mrs_ddec=None,
            mrs_popen=None,
            mrs_n_prs=None,
            mrs_n_terminated=None,
            mrs_n_open=None,
            rmvs_score=None,
            rmvs_archived=None,
            rmvs_stars_phi=None,
            rmvs_forks_phi=None,
            rmvs_issues_penalty=None,
            final_score=None,
            final_score_100=None,
            n_commits_total=None,
            n_commits_window=None,
            n_prs_total=None,
            n_prs_window=None,
            stars=None,
            forks=None,
            watchers=None,
            open_issues=None,
            archived=None,
            error=str(e),
        )


def score_repos(
    packages: Sequence[tuple[str, str]],
    commits_df: pd.DataFrame,
    pull_requests_df: pd.DataFrame,
    repo_meta_df: pd.DataFrame,
    eval_end: datetime,
    n_workers: int | None = None,
    malta_constants: MaltaConstants | None = None,
    das_constants: DevelopmentActivityScoreConstants | None = None,
    mrs_constants: MaintainerResponsivenessScoreConstants | None = None,
    repo_meta_constants: RepoViabilityScoreConstants | None = None,
    final_agg_constants: AggregateScoreConstants | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Score multiple repositories concurrently using MALTA metrics.

    Parameters
    ----------
    packages : Sequence[tuple[str, str]]
        List of (source, repo_url) tuples identifying packages to score.
    commits_df : pd.DataFrame
        DataFrame containing commit data with 'repo_url' column.
    pull_requests_df : pd.DataFrame
        DataFrame containing PR data with 'repo_url' column.
    repo_meta_df : pd.DataFrame
        DataFrame containing repository metadata with 'repo_url' column.
    eval_end : datetime
        End of evaluation window (must be timezone-aware).
    n_workers : int | None
        Number of worker processes. None for auto (CPU count).
    malta_constants : MaltaConstants | None
        Custom MALTA constants.
    das_constants : DevelopmentActivityScoreConstants | None
        Custom DAS constants.
    mrs_constants : MaintainerResponsivenessScoreConstants | None
        Custom MRS constants.
    repo_meta_constants : RepoViabilityScoreConstants | None
        Custom RMVS constants.
    final_agg_constants : AggregateScoreConstants | None
        Custom aggregation constants.
    show_progress : bool
        Whether to show a progress bar (requires tqdm).

    Returns
    -------
    pd.DataFrame
        DataFrame with MALTA scores for each package. Columns include:
        - source, repo_url: Package identifiers
        - das_score, das_dc, das_rc: Development Activity Score components
        - mrs_score, mrs_rdec, mrs_ddec, mrs_popen, mrs_n_*: MRS components
        - rmvs_score, rmvs_archived, rmvs_*_phi, rmvs_issues_penalty: RMVS components
        - final_score, final_score_100: Aggregated scores
        - n_commits_total, n_commits_window, n_prs_total, n_prs_window: Counts
        - stars, forks, watchers, open_issues, archived: Repository metadata
        - error: Error message if scoring failed

    Example
    -------
    >>> packages = [("pkg1", "https://github.com/owner/repo1"), ...]
    >>> results_df = score_repos(
    ...     packages=packages,
    ...     commits_df=commits_df,
    ...     pull_requests_df=prs_df,
    ...     repo_meta_df=meta_df,
    ...     eval_end=datetime(2026, 1, 1, tzinfo=timezone.utc),
    ...     n_workers=8,
    ... )
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import os

    if n_workers is None:
        n_workers = os.cpu_count() or 4

    # Get column name for repo_url (default or from constants)
    repo_url_col = (malta_constants or MaltaConstants()).repo_url_column

    # Pre-group DataFrames by repo_url for efficient lookup
    commits_grouped = {url: group for url, group in commits_df.groupby(repo_url_col)}
    prs_grouped = {url: group for url, group in pull_requests_df.groupby(repo_url_col)}
    meta_grouped = {url: group for url, group in repo_meta_df.groupby(repo_url_col)}

    # Empty DataFrames for repos with no data
    empty_commits = commits_df.iloc[:0]
    empty_prs = pull_requests_df.iloc[:0]
    empty_meta = repo_meta_df.iloc[:0]

    # Pre-extract data in the main process so workers receive lightweight
    # NamedTuples instead of full DataFrames (~85-95% pickle size reduction).
    work_items = []
    for source, repo_url in packages:
        extracted = _extract_repo_data(
            repo_url,
            commits_grouped.get(repo_url, empty_commits),
            prs_grouped.get(repo_url, empty_prs),
            meta_grouped.get(repo_url, empty_meta),
            malta_constants,
        )
        work_items.append(
            (
                source,
                repo_url,
                extracted,
                eval_end,
                malta_constants,
                das_constants,
                mrs_constants,
                repo_meta_constants,
                final_agg_constants,
            )
        )

    results: list[MaltaResult] = []

    # Set up progress bar if requested
    progress: Any = None
    if show_progress:
        try:
            from tqdm import tqdm

            progress = tqdm(total=len(work_items), desc="Scoring repos")
        except ImportError:
            show_progress = False

    # Process in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_score_single_repo_fast, *item): i for i, item in enumerate(work_items)}

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if progress:
                progress.update(1)

    if progress:
        progress.close()

    # Convert to DataFrame
    return pd.DataFrame([r._asdict() for r in results])
