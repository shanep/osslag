# Maintenance-Aware Lag and Technical Abandonment (MALTA) metrics
from __future__ import annotations

import math
from statistics import median
from datetime import datetime
from typing import NamedTuple, Optional, Sequence
import pandas as pd
from dateutil.relativedelta import relativedelta


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

    s_resp: float  # Responsiveness score
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
    s_dev: float  # Development activity score
    s_resp: float  # Responsiveness score
    s_meta: float  # Metadata score
    s_final: float  # in [0,1]
    s_final_100: float  # in [0,100]


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

    tau_days: float = 180.0  # Decay half-life in days


class MaintainerResponsivenessScoreConstants(NamedTuple):
    """Constants for maintainer responsiveness score computation."""

    tref_days: int = 180  # Time reference for decision timeliness in days


class RepoViabilityScoreConstants(NamedTuple):
    """Constants for repository viability score computation."""

    K: int = 10_000
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


class Malta:
    def __init__(
        self,
        package: str,
        github_repo_url: str,
        eval_end: datetime,
        commits_df: pd.DataFrame,
        pull_requests_df: pd.DataFrame,
        repo_meta_df: pd.DataFrame,
        malta_constants: Optional[MaltaConstants] = None,
        das_constants: Optional[DevelopmentActivityScoreConstants] = None,
        mrs_constants: Optional[MaintainerResponsivenessScoreConstants] = None,
        repo_meta_constants: Optional[RepoViabilityScoreConstants] = None,
        final_agg_constants: Optional[AggregateScoreConstants] = None,
    ):
        self.package = package
        self.github_repo_url = github_repo_url
        self.commits_df = commits_df
        self.pull_requests_df = pull_requests_df
        self.repo_meta_df = repo_meta_df

        self.das: DASComponents
        self.mrs: MRSComponents
        self.rmvs: RMVSComponents
        self.final: AggregateScoreComponents

        self.malta_constants = (
            malta_constants if malta_constants is not None else MaltaConstants()
        )
        self.das_constants = (
            das_constants
            if das_constants is not None
            else DevelopmentActivityScoreConstants()
        )
        self.mrs_constants = (
            mrs_constants
            if mrs_constants is not None
            else MaintainerResponsivenessScoreConstants()
        )
        self.rmv_constants = (
            repo_meta_constants
            if repo_meta_constants is not None
            else RepoViabilityScoreConstants()
        )
        self.final_constants = (
            final_agg_constants
            if final_agg_constants is not None
            else AggregateScoreConstants()
        )
        # Precompute evaluation and baseline windows
        if eval_end.tzinfo is None:
            raise ValueError("Datetime object must be timezone-aware")
        eval_window_start = eval_end - relativedelta(
            months=self.malta_constants.eval_months
        )
        # Evaluation window
        self.eval_window = EvaluationWindow(
            start=eval_window_start,
            end=eval_end,
            days=(eval_end - eval_window_start).days,
        )
        baseline_window_start = eval_window_start - relativedelta(
            months=self.malta_constants.baseline_months
        )
        baseline_window_end = eval_window_start
        # Baseline window
        self.baseline_window = EvaluationWindow(
            start=baseline_window_start,
            end=baseline_window_end,
            days=(baseline_window_end - baseline_window_start).days,
        )

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
        """Extract commits for a given repository URL."""
        repo_url_column = self.malta_constants.repo_url_column
        repo_dates_column = self.malta_constants.repo_dates_column
        repo_is_trivial_column = self.malta_constants.repo_is_trivial_column
        repo_commits_df = self.commits_df[
            self.commits_df[repo_url_column] == self.github_repo_url
        ].copy()
        if len(repo_commits_df) == 0:
            return []
        # Ensure datetime is timezone-aware UTC
        repo_commits_df[repo_dates_column] = pd.to_datetime(
            repo_commits_df[repo_dates_column], utc=True
        )
        commits = [
            Commit(date=row[repo_dates_column], is_trivial=row[repo_is_trivial_column])
            for _, row in repo_commits_df.iterrows()
        ]
        return commits

    def get_pull_requests_for_package(self) -> Sequence[PullRequest]:
        """Extract pull requests for a given repository URL."""
        repo_url_column = self.malta_constants.repo_url_column
        pr_created_at_column = self.malta_constants.pr_created_at_column
        pr_closed_at_column = self.malta_constants.pr_closed_at_column
        pr_merged_at_column = self.malta_constants.pr_merged_at_column
        pr_state_column = self.malta_constants.pr_state_column
        repo_prs_df = self.pull_requests_df[
            self.pull_requests_df[repo_url_column] == self.github_repo_url
        ].copy()
        if len(repo_prs_df) == 0:
            return []
        # Ensure datetime is timezone-aware UTC
        repo_prs_df[pr_created_at_column] = pd.to_datetime(
            repo_prs_df[pr_created_at_column], utc=True
        )
        repo_prs_df[pr_closed_at_column] = pd.to_datetime(
            repo_prs_df[pr_closed_at_column], utc=True
        )
        repo_prs_df[pr_merged_at_column] = pd.to_datetime(
            repo_prs_df[pr_merged_at_column], utc=True
        )
        prs = [
            PullRequest(
                created_at=row[pr_created_at_column],
                closed_at=row[pr_closed_at_column],
                merged_at=row[pr_merged_at_column],
                state=row[pr_state_column],
            )
            for _, row in repo_prs_df.iterrows()
        ]
        return prs

    def development_activity_score(
        self,
        commits: Sequence[Commit],
        include_trivial: bool = False,
    ) -> DASComponents:
        """Computes S_dev in [0,1]:

            D_c = (C_e / |W_e|) / (C_b / |W_b|)
            R_c = exp(-t_last / tau)
            S_dev = min(1, D_c) * R_c

        Parameters
        ----------
        commits : Sequence[Commit]
            All commits in both baseline and evaluation windows.
        tau_days : float
            Decay half-life in days (default 180).
        include_trivial : bool
            If False, exclude trivial commits from counts and recency calculations.

        Returns
        -------
        DAComponents
            Components of the development activity score.

        Notes
        -----
        - If baseline rate is 0, we treat D_c as 1 when eval also has activity,
        else 0. This avoids division-by-zero while remaining conservative.
        - t_last is measured since most recent non-trivial commit (unless include_trivial=True).

        """
        # Partition commits into the baseline and evaluation sets, filtering trivial if needed.
        commits_baseline: Sequence[Commit] = []
        commits_eval: Sequence[Commit] = []
        window_baseline_days: int = (
            self.baseline_window.end.toordinal()
            - self.baseline_window.start.toordinal()
        )
        window_eval_days: int = (
            self.eval_window.end.toordinal() - self.eval_window.start.toordinal()
        )

        for c in commits:
            if self.baseline_window.start <= c.date < self.baseline_window.end:
                commits_baseline.append(c)
            elif self.eval_window.start <= c.date < self.eval_window.end:
                commits_eval.append(c)

        if window_baseline_days <= 0 or window_eval_days <= 0:
            raise ValueError("window_baseline_days and window_eval_days must be > 0")
        if self.baseline_window.end > self.eval_window.start:
            raise ValueError(
                "Baseline window must end before evaluation window starts."
            )
        if self.eval_window.start < self.baseline_window.end:
            raise ValueError("Evaluation window must start after baseline window ends.")

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
                raise ValueError(
                    "Commit.authored_at must be timezone-aware (UTC recommended)."
                )
            t_last = max(
                0.0, (self.eval_window.end - last_commit_time).total_seconds() / 86400.0
            )
            R_c = math.exp(-t_last / self.das_constants.tau_days)
        else:
            # No commits at all -> fully inactive.
            R_c = 0.0

        S_dev = self.__clamp(min(1.0, D_c) * R_c)

        self.das = DASComponents(d_c=D_c, r_c=R_c, s_dev=S_dev)
        return self.das

    def maintainer_responsiveness_score(
        self,
        pull_requests: Sequence[PullRequest],
    ) -> MRSComponents:
        """Compute the PR-outcome-bound Maintainer Responsiveness Score (S_resp).

        Parameters
        ----------
        pull_requests : Sequence[PullRequest]

        Returns
        -------
        MRSComponents
            Components of the maintainer responsiveness score.

        """
        if not pull_requests:
            # No external contribution signal
            return MRSComponents(
                s_resp=0.0,
                r_dec=0.0,
                d_dec=0.0,
                p_open=0.0,
                n_prs=0,
                n_terminated=0,
                n_open=0,
            )
        # Filter PRs to those created within the evaluation window
        P = [
            pr
            for pr in pull_requests
            if self.eval_window.start <= pr.created_at < self.eval_window.end
        ]
        if not P:
            # No PRs in evaluation window
            return MRSComponents(
                s_resp=0.0,
                r_dec=0.0,
                d_dec=0.0,
                p_open=0.0,
                n_prs=0,
                n_terminated=0,
                n_open=0,
            )
        # Partition PRs
        P_term = []
        P_open = []

        for pr in P:
            if pr.state == "closed":
                P_term.append(pr)
            elif pr.state == "open":
                P_open.append(pr)
            else:
                raise ValueError(f"Unknown PR state: {pr.state}")

        # If PRs exist but none are handled, score is 0
        if P_term:
            R_dec = len(P_term) / len(P)
        else:
            return MRSComponents(
                s_resp=0.0,
                r_dec=0.0,
                d_dec=0.0,
                p_open=0.0,
                n_prs=len(P),
                n_terminated=0,
                n_open=len(P_open),
            )

        # ---- Decision Timeliness (D_dec) ----
        decision_delays = []
        for pr in P_term:
            closed_time = pr.merged_at or pr.closed_at
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

        self.mrsc = MRSComponents(
            s_resp=self.__clamp(S_resp),
            r_dec=R_dec,
            d_dec=D_dec,
            p_open=P_open_penalty,
            n_prs=len(P),
            n_terminated=len(P_term),
            n_open=len(P_open),
        )

        return self.mrsc

    def repo_metadata_viability_score(
        self,
        meta: RepoMeta,
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
        """
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
        K = self.rmv_constants.K
        s = self.__phi_count(meta.stars, K)
        f = self.__phi_count(meta.forks, K)
        w = self.__phi_count(meta.watchers, K)
        i = self.__phi_count(meta.open_issues, K)
        i_pen = 0 if i == 0 else (1.0 - i)

        parts = {"stars": s, "forks": f, "watchers": w, "issues": i_pen}
        observed = {k: v for k, v in parts.items() if v is not None}

        if not observed:
            # All fields missing
            self.rmvs = RMVSComponents(
                s_meta=0.0,
                stars_phi=s,
                forks_phi=f,
                watchers_phi=w,
                open_issues_penalty=i_pen,
                archived=meta.archived,
            )
            return self.rmvs

        # Renormalize betas over observed fields
        wsum = sum(betas[k] for k in observed.keys())
        linear = sum((betas[k] / wsum) * observed[k] for k in observed.keys())

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
        """
        s_dev = self.das.s_dev
        s_resp = self.mrsc.s_resp
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
