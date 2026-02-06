import pytest
from datetime import datetime, timezone, timedelta
import math
import pandas as pd

from osslag.metrics.malta import (
    Malta,
    MaltaConstants,
    DevelopmentActivityScoreConstants,
    Commit,
    PullRequest,
    RepoMeta,
    DASComponents,
    MRSComponents,
    RMVSComponents,
    AggregateScoreComponents,
    MaltaResult,
    score_repos,
)


# Fixture for common test time
@pytest.fixture
def eval_end():
    return datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def empty_prs_df():
    return pd.DataFrame(columns=["repo_url", "created_at", "closed_at", "merged_at", "state"])


@pytest.fixture
def empty_meta_df():
    return pd.DataFrame(columns=["repo_url", "stars", "forks", "watchers", "open_issues", "archived"])


def make_commits_df(commits: list[Commit], repo_url: str = "https://github.com/test/repo") -> pd.DataFrame:
    """Helper to create a commits DataFrame from Commit objects."""
    return pd.DataFrame(
        {
            "repo_url": [repo_url] * len(commits),
            "date": [c.date for c in commits],
            "is_trivial": [c.is_trivial for c in commits],
        }
    )


def make_prs_df(prs: list[PullRequest], repo_url: str = "https://github.com/test/repo") -> pd.DataFrame:
    """Helper to create a PRs DataFrame from PullRequest objects."""
    return pd.DataFrame(
        {
            "repo_url": [repo_url] * len(prs),
            "created_at": [pr.created_at for pr in prs],
            "closed_at": [pr.closed_at for pr in prs],
            "merged_at": [pr.merged_at for pr in prs],
            "state": [pr.state for pr in prs],
        }
    )


def make_meta_df(meta: RepoMeta, repo_url: str = "https://github.com/test/repo") -> pd.DataFrame:
    """Helper to create a metadata DataFrame from RepoMeta."""
    return pd.DataFrame(
        {
            "repo_url": [repo_url],
            "stars": [meta.stars],
            "forks": [meta.forks],
            "watchers": [meta.watchers],
            "open_issues": [meta.open_issues],
            "archived": [meta.archived],
        }
    )


def create_malta(
    eval_end: datetime,
    commits: list[Commit],
    prs: list[PullRequest] | None = None,
    meta: RepoMeta | None = None,
    repo_url: str = "https://github.com/test/repo",
    baseline_months: int = 24,
    eval_months: int = 12,
    tau_days: float = 180.0,
) -> Malta:
    """Helper to create a Malta instance with test data."""
    commits_df = make_commits_df(commits, repo_url)
    prs_df = make_prs_df(prs or [], repo_url)
    meta_df = make_meta_df(meta or RepoMeta(), repo_url)

    malta_constants = MaltaConstants(
        eval_months=eval_months,
        baseline_months=baseline_months,
    )
    das_constants = DevelopmentActivityScoreConstants(tau_days=tau_days)

    return Malta(
        package="test-package",
        github_repo_url=repo_url,
        eval_end=eval_end,
        commits_df=commits_df,
        pull_requests_df=prs_df,
        repo_meta_df=meta_df,
        malta_constants=malta_constants,
        das_constants=das_constants,
    )


# =============================================================================
# Tests for Development Activity Score
# =============================================================================


class TestDevelopmentActivityScore:
    def test_equal_rates_recent_eval(self, eval_end):
        """Equal commit rates with recent eval commits should score high."""
        # Create Malta instance - windows are computed from eval_end
        malta_for_windows = create_malta(eval_end, [], baseline_months=24, eval_months=12)

        # Baseline: 1 commit per month for 24 months
        baseline_commits = [
            Commit(
                date=malta_for_windows.baseline_window.start + timedelta(days=30 * i),
                is_trivial=False,
            )
            for i in range(24)
        ]

        # Eval: 1 commit per month for 12 months
        eval_commits = [
            Commit(date=malta_for_windows.eval_window.start + timedelta(days=30 * i), is_trivial=False)
            for i in range(12)
        ]

        all_commits = baseline_commits + eval_commits
        malta = create_malta(eval_end, all_commits, baseline_months=24, eval_months=12)

        result = malta.development_activity_score(include_trivial=False)

        assert isinstance(result, DASComponents)
        assert result.s_dev > 0.50
        assert result.s_dev < 1.0

    def test_faster_eval_than_baseline(self, eval_end):
        """Eval with more commits per day than baseline should have D_c > 1 clamped."""
        malta_for_windows = create_malta(eval_end, [], baseline_months=12, eval_months=12)

        # One commit in baseline
        baseline_commits = [
            Commit(date=malta_for_windows.baseline_window.start + timedelta(days=100), is_trivial=False)
        ]

        # 10 commits in eval, one per day from the end
        eval_commits = [
            Commit(date=malta_for_windows.eval_window.end - timedelta(days=i + 1), is_trivial=False) for i in range(10)
        ]

        all_commits = baseline_commits + eval_commits
        malta = create_malta(eval_end, all_commits, baseline_months=12, eval_months=12)

        result = malta.development_activity_score(include_trivial=False)

        # D_c = (10/days) / (1/days) = 10 -> clamped to 1 in S_dev
        assert result.d_c == 10.0  # Unclamped D_c
        assert result.s_dev == pytest.approx(math.exp(-1 / 180.0), abs=0.01)

    def test_no_eval_commits_zero_score(self, eval_end):
        """If eval has no commits, score should be 0."""
        malta_for_windows = create_malta(eval_end, [], baseline_months=12, eval_months=12)

        # One commit in baseline only
        commits = [Commit(date=malta_for_windows.baseline_window.start + timedelta(days=100), is_trivial=False)]
        malta = create_malta(eval_end, commits, baseline_months=12, eval_months=12)

        result = malta.development_activity_score(include_trivial=False)

        # No eval commits -> D_c = 0
        assert result.d_c == 0.0
        assert result.s_dev == 0.0

    def test_no_baseline_commits_eval_active(self, eval_end):
        """If baseline has no commits but eval is active, D_c = 1."""
        malta_for_windows = create_malta(eval_end, [], baseline_months=12, eval_months=12)

        # Only commits in eval
        eval_commits = [
            Commit(date=malta_for_windows.eval_window.end - timedelta(days=i + 1), is_trivial=False) for i in range(5)
        ]
        malta = create_malta(eval_end, eval_commits, baseline_months=12, eval_months=12)

        result = malta.development_activity_score(include_trivial=False)

        # D_c = 1 (baseline is zero, eval is nonzero)
        assert result.d_c == 1.0
        assert result.s_dev == pytest.approx(math.exp(-1 / 180.0), abs=0.01)

    def test_no_commits_at_all_zero_score(self, eval_end):
        """If neither baseline nor eval has commits, score should be 0."""
        malta = create_malta(eval_end, [], baseline_months=12, eval_months=12)

        result = malta.development_activity_score(include_trivial=False)

        assert result.d_c == 0.0
        assert result.r_c == 0.0
        assert result.s_dev == 0.0

    def test_include_trivial_commits(self, eval_end):
        """When include_trivial=True, trivial commits should be counted."""
        malta_for_windows = create_malta(eval_end, [], baseline_months=12, eval_months=12)

        # Baseline: one non-trivial, one trivial
        baseline_commits = [
            Commit(date=malta_for_windows.baseline_window.start + timedelta(days=100), is_trivial=False),
            Commit(date=malta_for_windows.baseline_window.start + timedelta(days=150), is_trivial=True),
        ]
        # Eval: one non-trivial
        eval_commits = [Commit(date=malta_for_windows.eval_window.end - timedelta(days=100), is_trivial=False)]

        all_commits = baseline_commits + eval_commits
        malta = create_malta(eval_end, all_commits, baseline_months=12, eval_months=12)

        result_with_trivial = malta.development_activity_score(include_trivial=True)
        result_without_trivial = malta.development_activity_score(include_trivial=False)

        # Trivial commit increases baseline count, decreasing D_c and thus the score
        assert result_with_trivial.s_dev < result_without_trivial.s_dev

    def test_recency_decay_tau(self, eval_end):
        """Older eval commits should decay more with smaller tau."""
        # Create Malta instance to get window bounds
        malta_for_windows = create_malta(eval_end, [], baseline_months=12, eval_months=12, tau_days=180.0)

        # One commit in baseline
        baseline_commits = [
            Commit(
                date=malta_for_windows.baseline_window.start + timedelta(days=100),
                is_trivial=False,
            )
        ]
        # Last commit 180 days before eval_end
        eval_commits = [
            Commit(
                date=malta_for_windows.eval_window.end - timedelta(days=180),
                is_trivial=False,
            )
        ]

        all_commits = baseline_commits + eval_commits

        # Create two Malta instances with different tau values
        malta_tau_180 = create_malta(eval_end, all_commits, baseline_months=12, eval_months=12, tau_days=180.0)
        malta_tau_90 = create_malta(eval_end, all_commits, baseline_months=12, eval_months=12, tau_days=90.0)

        result_tau_180 = malta_tau_180.development_activity_score(include_trivial=False)
        result_tau_90 = malta_tau_90.development_activity_score(include_trivial=False)

        # exp(-180/90) < exp(-180/180)
        # So tau=90 should decay faster -> lower score
        assert result_tau_90.s_dev < result_tau_180.s_dev

    def test_very_recent_eval_high_recency(self, eval_end):
        """Very recent eval commits should have minimal recency decay."""
        malta_for_windows = create_malta(eval_end, [], baseline_months=12, eval_months=12)

        baseline_commits = [
            Commit(date=malta_for_windows.baseline_window.start + timedelta(days=100), is_trivial=False)
        ]
        # Commit 1 day before eval_end
        eval_commits = [Commit(date=malta_for_windows.eval_window.end - timedelta(days=1), is_trivial=False)]

        all_commits = baseline_commits + eval_commits
        malta = create_malta(eval_end, all_commits, baseline_months=12, eval_months=12)

        result = malta.development_activity_score(include_trivial=False)

        # R_c = exp(-1/180) ≈ 0.9944
        expected_approx = math.exp(-1 / 180.0)
        assert result.s_dev == pytest.approx(expected_approx, abs=0.001)

    def test_very_old_eval_low_recency(self, eval_end):
        """Very old eval commits should have strong recency decay."""
        malta_for_windows = create_malta(eval_end, [], baseline_months=12, eval_months=12)

        baseline_commits = [
            Commit(date=malta_for_windows.baseline_window.start + timedelta(days=100), is_trivial=False)
        ]
        # Commit at start of eval window
        eval_commits = [Commit(date=malta_for_windows.eval_window.start + timedelta(days=1), is_trivial=False)]

        all_commits = baseline_commits + eval_commits
        malta = create_malta(eval_end, all_commits, baseline_months=12, eval_months=12)

        result = malta.development_activity_score(include_trivial=False)

        # R_c = exp(-t_last/180) where t_last is close to eval window length
        t_last = (malta.eval_window.end - (malta.eval_window.start + timedelta(days=1))).days
        expected_approx = math.exp(-t_last / 180.0)
        assert result.s_dev == pytest.approx(expected_approx, abs=0.01)

    def test_velocity_ratio_clamp(self, eval_end):
        """D_c > 1 should be clamped to 1 in final score."""
        malta_for_windows = create_malta(eval_end, [], baseline_months=12, eval_months=12)

        baseline_commits = [
            Commit(date=malta_for_windows.baseline_window.start + timedelta(days=100), is_trivial=False)
        ]
        # 100 commits in eval
        eval_commits = [
            Commit(date=malta_for_windows.eval_window.end - timedelta(days=i + 1), is_trivial=False) for i in range(100)
        ]

        all_commits = baseline_commits + eval_commits
        malta = create_malta(eval_end, all_commits, baseline_months=12, eval_months=12)

        result = malta.development_activity_score(include_trivial=False)

        # D_c = 100 (unclamped)
        # S_dev = min(1, 100) * R_c = 1 * exp(-1/180)
        assert result.d_c == 100.0
        assert result.s_dev == pytest.approx(math.exp(-1 / 180.0), abs=0.01)

    def test_only_trivial_eval_commits(self, eval_end):
        """All trivial eval commits should result in 0 when include_trivial=False."""
        malta_for_windows = create_malta(eval_end, [], baseline_months=12, eval_months=12)

        baseline_commits = [
            Commit(date=malta_for_windows.baseline_window.start + timedelta(days=100), is_trivial=False)
        ]
        eval_commits = [
            Commit(date=malta_for_windows.eval_window.end - timedelta(days=100), is_trivial=True),
            Commit(date=malta_for_windows.eval_window.end - timedelta(days=50), is_trivial=True),
        ]

        all_commits = baseline_commits + eval_commits
        malta = create_malta(eval_end, all_commits, baseline_months=12, eval_months=12)

        result = malta.development_activity_score(include_trivial=False)

        assert result.d_c == 0.0
        assert result.s_dev == 0.0

    def test_only_trivial_baseline_commits(self, eval_end):
        """All trivial baseline commits should allow D_c=1 when eval is active."""
        malta_for_windows = create_malta(eval_end, [], baseline_months=12, eval_months=12)

        baseline_commits = [
            Commit(date=malta_for_windows.baseline_window.start + timedelta(days=100), is_trivial=True),
        ]
        eval_commits = [Commit(date=malta_for_windows.eval_window.end - timedelta(days=100), is_trivial=False)]

        all_commits = baseline_commits + eval_commits
        malta = create_malta(eval_end, all_commits, baseline_months=12, eval_months=12)

        result = malta.development_activity_score(include_trivial=False)

        assert result.d_c == 1.0
        assert result.s_dev > 0.0


# =============================================================================
# Tests for Maintainer Responsiveness Score
# =============================================================================


class TestMaintainerResponsivenessScore:
    def test_no_prs_returns_zero(self, eval_end):
        """No PRs should return zero scores."""
        malta = create_malta(eval_end, [])

        result = malta.maintainer_responsiveness_score()

        assert isinstance(result, MRSComponents)
        assert result.s_resp == 0.0
        assert result.n_prs == 0

    def test_all_prs_closed_quickly(self, eval_end):
        """All PRs closed quickly should have high responsiveness."""
        malta_for_windows = create_malta(eval_end, [])

        # PRs created and closed within eval window
        prs = [
            PullRequest(
                created_at=malta_for_windows.eval_window.start + timedelta(days=10),
                closed_at=malta_for_windows.eval_window.start + timedelta(days=11),
                merged_at=malta_for_windows.eval_window.start + timedelta(days=11),
                state="closed",
            ),
            PullRequest(
                created_at=malta_for_windows.eval_window.start + timedelta(days=20),
                closed_at=malta_for_windows.eval_window.start + timedelta(days=22),
                merged_at=malta_for_windows.eval_window.start + timedelta(days=22),
                state="closed",
            ),
        ]
        malta = create_malta(eval_end, [], prs=prs)

        result = malta.maintainer_responsiveness_score()

        assert result.s_resp > 0.9  # High responsiveness
        assert result.r_dec == 1.0  # All PRs decided
        assert result.n_terminated == 2
        assert result.n_open == 0

    def test_all_prs_open(self, eval_end):
        """All PRs still open should have zero responsiveness."""
        malta_for_windows = create_malta(eval_end, [])

        prs = [
            PullRequest(
                created_at=malta_for_windows.eval_window.start + timedelta(days=10),
                closed_at=None,
                merged_at=None,
                state="open",
            ),
        ]
        malta = create_malta(eval_end, [], prs=prs)

        result = malta.maintainer_responsiveness_score()

        assert result.s_resp == 0.0
        assert result.r_dec == 0.0
        assert result.n_open == 1
        assert result.n_terminated == 0

    def test_mixed_prs_partial_responsiveness(self, eval_end):
        """Mix of open and closed PRs should have partial responsiveness."""
        malta_for_windows = create_malta(eval_end, [])

        # Use PRs close to eval_end to avoid high staleness penalty
        prs = [
            PullRequest(
                created_at=malta_for_windows.eval_window.end - timedelta(days=30),
                closed_at=malta_for_windows.eval_window.end - timedelta(days=25),
                merged_at=None,
                state="closed",
            ),
            PullRequest(
                created_at=malta_for_windows.eval_window.end - timedelta(days=10),  # Recent open PR
                closed_at=None,
                merged_at=None,
                state="open",
            ),
        ]
        malta = create_malta(eval_end, [], prs=prs)

        result = malta.maintainer_responsiveness_score()

        assert 0.0 < result.s_resp < 1.0
        assert result.r_dec == 0.5  # Half decided
        assert result.n_terminated == 1
        assert result.n_open == 1

    def test_slow_decisions_penalized(self, eval_end):
        """Slow PR decisions should reduce the score."""
        malta_for_windows = create_malta(eval_end, [])

        # PR that takes a long time to close
        prs = [
            PullRequest(
                created_at=malta_for_windows.eval_window.start + timedelta(days=10),
                closed_at=malta_for_windows.eval_window.start + timedelta(days=190),  # 180 days to close
                merged_at=None,
                state="closed",
            ),
        ]
        malta = create_malta(eval_end, [], prs=prs)

        result = malta.maintainer_responsiveness_score()

        # Slow decision should reduce score via D_dec
        assert result.d_dec == 1.0  # Capped at 1.0 (180/180)
        assert result.s_resp == 0.0  # (1 - 1.0) factor makes it 0


# =============================================================================
# Tests for Repository Metadata Viability Score
# =============================================================================


class TestRepoMetadataViabilityScore:
    def test_popular_repo_high_score(self, eval_end):
        """Popular non-archived repo should have high viability."""
        meta = RepoMeta(stars=10000, forks=5000, watchers=1000, open_issues=10, archived=False)
        malta = create_malta(eval_end, [], meta=meta)

        result = malta.repo_metadata_viability_score()

        assert isinstance(result, RMVSComponents)
        assert result.s_meta > 0.7

    def test_archived_repo_penalized(self, eval_end):
        """Archived repo should have reduced viability."""
        meta_active = RepoMeta(stars=1000, forks=500, watchers=100, open_issues=10, archived=False)
        meta_archived = RepoMeta(stars=1000, forks=500, watchers=100, open_issues=10, archived=True)

        malta_active = create_malta(eval_end, [], meta=meta_active)
        malta_archived = create_malta(eval_end, [], meta=meta_archived)

        result_active = malta_active.repo_metadata_viability_score()
        result_archived = malta_archived.repo_metadata_viability_score()

        assert result_archived.s_meta < result_active.s_meta
        assert result_archived.archived is True

    def test_high_open_issues_penalized(self, eval_end):
        """Many open issues should reduce viability."""
        meta_few_issues = RepoMeta(stars=1000, forks=500, watchers=100, open_issues=10, archived=False)
        meta_many_issues = RepoMeta(stars=1000, forks=500, watchers=100, open_issues=5000, archived=False)

        malta_few = create_malta(eval_end, [], meta=meta_few_issues)
        malta_many = create_malta(eval_end, [], meta=meta_many_issues)

        result_few = malta_few.repo_metadata_viability_score()
        result_many = malta_many.repo_metadata_viability_score()

        assert result_many.s_meta < result_few.s_meta

    def test_zero_metrics_partial_score(self, eval_end):
        """Repo with zero metrics still gets credit for having no open issues.

        Per the paper: Ipen = 1 - I* where I* = phi(open_issues).
        When open_issues=0, I*=0, so Ipen=1.0 (full credit for no backlog).
        With all other metrics at 0: s_meta = 0.25*0 + 0.25*0 + 0.25*0 + 0.25*1.0 = 0.25
        """
        meta = RepoMeta(stars=0, forks=0, watchers=0, open_issues=0, archived=False)
        malta = create_malta(eval_end, [], meta=meta)

        result = malta.repo_metadata_viability_score()

        # Zero stars/forks/watchers contribute 0, but zero open_issues contributes 0.25
        assert result.s_meta == 0.25
        assert result.open_issues_penalty == 1.0  # Full credit for no backlog


# =============================================================================
# Tests for Final Aggregation Score
# =============================================================================


class TestFinalAggregationScore:
    def test_all_scores_combined(self, eval_end):
        """Final score should combine all component scores."""
        malta_for_windows = create_malta(eval_end, [])

        # Set up component scores by calling the methods
        commits = [
            Commit(date=malta_for_windows.baseline_window.start + timedelta(days=100), is_trivial=False),
            Commit(date=malta_for_windows.eval_window.end - timedelta(days=10), is_trivial=False),
        ]

        prs = [
            PullRequest(
                created_at=malta_for_windows.eval_window.start + timedelta(days=10),
                closed_at=malta_for_windows.eval_window.start + timedelta(days=15),
                merged_at=malta_for_windows.eval_window.start + timedelta(days=15),
                state="closed",
            ),
        ]

        meta = RepoMeta(stars=1000, forks=500, watchers=100, open_issues=10, archived=False)
        malta = create_malta(eval_end, commits, prs=prs, meta=meta)

        malta.development_activity_score()
        malta.maintainer_responsiveness_score()
        malta.repo_metadata_viability_score()

        result = malta.final_aggregation_score()

        assert isinstance(result, AggregateScoreComponents)
        assert 0.0 <= result.s_final <= 1.0
        assert result.s_final_100 == result.s_final * 100.0

    def test_archived_repo_affects_final(self, eval_end):
        """Archived repo should reduce final score via responsiveness."""
        malta_for_windows = create_malta(eval_end, [])

        commits = [
            Commit(date=malta_for_windows.eval_window.end - timedelta(days=10), is_trivial=False),
        ]

        # Need PRs that go through the full processing path to set self.mrs
        prs = [
            PullRequest(
                created_at=malta_for_windows.eval_window.end - timedelta(days=30),
                closed_at=malta_for_windows.eval_window.end - timedelta(days=25),
                merged_at=malta_for_windows.eval_window.end - timedelta(days=25),
                state="closed",
            ),
        ]

        meta_active = RepoMeta(stars=1000, forks=500, watchers=100, open_issues=10, archived=False)
        meta_archived = RepoMeta(stars=1000, forks=500, watchers=100, open_issues=10, archived=True)

        # Test with non-archived repo
        malta_active = create_malta(eval_end, commits, prs=prs, meta=meta_active)
        malta_active.development_activity_score()
        malta_active.maintainer_responsiveness_score()
        malta_active.repo_metadata_viability_score()
        result_active = malta_active.final_aggregation_score()

        # Test with archived repo
        malta_archived = create_malta(eval_end, commits, prs=prs, meta=meta_archived)
        malta_archived.development_activity_score()
        malta_archived.maintainer_responsiveness_score()
        malta_archived.repo_metadata_viability_score()
        result_archived = malta_archived.final_aggregation_score()

        # Archived repo should have lower score
        assert result_archived.s_final < result_active.s_final


# =============================================================================
# Tests for Malta initialization and window computation
# =============================================================================


class TestMaltaInitialization:
    def test_naive_datetime_raises(self):
        """Naive datetime should raise ValueError."""
        naive_dt = datetime(2024, 1, 15, 12, 0, 0)  # No tzinfo

        with pytest.raises(ValueError, match="timezone-aware"):
            create_malta(naive_dt, [])

    def test_windows_computed_correctly(self, eval_end):
        """Windows should be computed based on constants."""
        malta = create_malta(eval_end, [], baseline_months=24, eval_months=18)

        # Eval window ends at eval_end
        assert malta.eval_window.end == eval_end

        # Windows should be contiguous (baseline ends where eval starts)
        assert malta.baseline_window.end == malta.eval_window.start


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Performance tests for Malta class with large datasets."""

    @pytest.fixture
    def large_commits_df(self, eval_end):
        """Generate a large commits DataFrame with 10,000 commits across 100 repos."""
        import numpy as np

        n_repos = 100
        n_commits_per_repo = 100
        total_commits = n_repos * n_commits_per_repo

        repo_urls = [f"https://github.com/test/repo-{i}" for i in range(n_repos)]

        # Generate random dates within a 4-year window
        base_date = eval_end - timedelta(days=4 * 365)
        random_days = np.random.randint(0, 4 * 365, total_commits)
        dates = [base_date + timedelta(days=int(d)) for d in random_days]

        # 10% trivial commits
        is_trivial = np.random.random(total_commits) < 0.1

        return pd.DataFrame(
            {
                "repo_url": [repo_urls[i // n_commits_per_repo] for i in range(total_commits)],
                "date": dates,
                "is_trivial": is_trivial,
            }
        )

    @pytest.fixture
    def large_prs_df(self, eval_end):
        """Generate a large PRs DataFrame with 5,000 PRs across 100 repos."""
        import numpy as np

        n_repos = 100
        n_prs_per_repo = 50
        total_prs = n_repos * n_prs_per_repo

        repo_urls = [f"https://github.com/test/repo-{i}" for i in range(n_repos)]

        # Generate random created dates
        base_date = eval_end - timedelta(days=2 * 365)
        random_days = np.random.randint(0, 2 * 365, total_prs)
        created_dates = [base_date + timedelta(days=int(d)) for d in random_days]

        # 70% closed, 30% open
        states = np.where(np.random.random(total_prs) < 0.7, "closed", "open")

        # Closed PRs have closed_at and possibly merged_at
        closed_dates = []
        merged_dates = []
        for i, (created, state) in enumerate(zip(created_dates, states)):
            if state == "closed":
                days_to_close = np.random.randint(1, 60)
                closed_at = created + timedelta(days=days_to_close)
                closed_dates.append(closed_at)
                # 60% of closed PRs are merged
                merged_dates.append(closed_at if np.random.random() < 0.6 else None)
            else:
                closed_dates.append(None)
                merged_dates.append(None)

        return pd.DataFrame(
            {
                "repo_url": [repo_urls[i // n_prs_per_repo] for i in range(total_prs)],
                "created_at": created_dates,
                "closed_at": closed_dates,
                "merged_at": merged_dates,
                "state": states,
            }
        )

    @pytest.fixture
    def large_meta_df(self):
        """Generate metadata for 100 repos."""
        import numpy as np

        n_repos = 100
        repo_urls = [f"https://github.com/test/repo-{i}" for i in range(n_repos)]

        return pd.DataFrame(
            {
                "repo_url": repo_urls,
                "stars": np.random.randint(0, 50000, n_repos),
                "forks": np.random.randint(0, 10000, n_repos),
                "watchers": np.random.randint(0, 5000, n_repos),
                "open_issues": np.random.randint(0, 1000, n_repos),
                "archived": np.random.random(n_repos) < 0.05,  # 5% archived
            }
        )

    def test_caching_commits(self, eval_end, large_commits_df, large_prs_df, large_meta_df):
        """Verify that commit extraction is cached - second call should be instant."""
        import time

        malta = Malta(
            package="test-package",
            github_repo_url="https://github.com/test/repo-0",
            eval_end=eval_end,
            commits_df=large_commits_df,
            pull_requests_df=large_prs_df,
            repo_meta_df=large_meta_df,
        )

        # First call - extracts and caches
        start = time.perf_counter()
        commits1 = malta.get_commits_for_package()
        first_call_time = time.perf_counter() - start

        # Second call - should return cached result
        start = time.perf_counter()
        commits2 = malta.get_commits_for_package()
        second_call_time = time.perf_counter() - start

        assert commits1 is commits2  # Same object (cached)
        assert second_call_time < first_call_time * 0.1  # At least 10x faster

    def test_caching_prs(self, eval_end, large_commits_df, large_prs_df, large_meta_df):
        """Verify that PR extraction is cached - second call should be instant."""
        import time

        malta = Malta(
            package="test-package",
            github_repo_url="https://github.com/test/repo-0",
            eval_end=eval_end,
            commits_df=large_commits_df,
            pull_requests_df=large_prs_df,
            repo_meta_df=large_meta_df,
        )

        # First call - extracts and caches
        start = time.perf_counter()
        prs1 = malta.get_pull_requests_for_package()
        first_call_time = time.perf_counter() - start

        # Second call - should return cached result
        start = time.perf_counter()
        prs2 = malta.get_pull_requests_for_package()
        second_call_time = time.perf_counter() - start

        assert prs1 is prs2  # Same object (cached)
        assert second_call_time < first_call_time * 0.1  # At least 10x faster

    def test_caching_meta(self, eval_end, large_commits_df, large_prs_df, large_meta_df):
        """Verify that metadata extraction is cached - second call should be instant."""
        import time

        malta = Malta(
            package="test-package",
            github_repo_url="https://github.com/test/repo-0",
            eval_end=eval_end,
            commits_df=large_commits_df,
            pull_requests_df=large_prs_df,
            repo_meta_df=large_meta_df,
        )

        # First call - extracts and caches
        start = time.perf_counter()
        meta1 = malta.get_repo_meta_for_package()
        first_call_time = time.perf_counter() - start

        # Second call - should return cached result
        start = time.perf_counter()
        meta2 = malta.get_repo_meta_for_package()
        second_call_time = time.perf_counter() - start

        assert meta1 is meta2  # Same object (cached)
        assert second_call_time < first_call_time * 0.1  # At least 10x faster

    def test_full_scoring_pipeline(self, eval_end, large_commits_df, large_prs_df, large_meta_df):
        """Test full scoring pipeline completes in reasonable time."""
        import time

        malta = Malta(
            package="test-package",
            github_repo_url="https://github.com/test/repo-0",
            eval_end=eval_end,
            commits_df=large_commits_df,
            pull_requests_df=large_prs_df,
            repo_meta_df=large_meta_df,
        )

        start = time.perf_counter()
        malta.development_activity_score()
        malta.maintainer_responsiveness_score()
        malta.repo_metadata_viability_score()
        result = malta.final_aggregation_score()
        elapsed = time.perf_counter() - start

        # Full pipeline should complete in under 100ms for a single repo
        assert elapsed < 0.1, f"Full scoring took {elapsed:.3f}s, expected < 0.1s"
        assert 0.0 <= result.s_final <= 1.0

    def test_multiple_repos_performance(self, eval_end, large_commits_df, large_prs_df, large_meta_df):
        """Test scoring multiple repos from the same DataFrames."""
        import time

        n_repos_to_score = 10
        repo_urls = [f"https://github.com/test/repo-{i}" for i in range(n_repos_to_score)]

        start = time.perf_counter()
        results = []
        for repo_url in repo_urls:
            malta = Malta(
                package=f"package-{repo_url}",
                github_repo_url=repo_url,
                eval_end=eval_end,
                commits_df=large_commits_df,
                pull_requests_df=large_prs_df,
                repo_meta_df=large_meta_df,
            )
            malta.development_activity_score()
            malta.maintainer_responsiveness_score()
            malta.repo_metadata_viability_score()
            results.append(malta.final_aggregation_score())
        elapsed = time.perf_counter() - start

        # 10 repos should complete in under 1 second
        assert elapsed < 1.0, f"Scoring 10 repos took {elapsed:.3f}s, expected < 1.0s"
        assert len(results) == n_repos_to_score
        for result in results:
            assert 0.0 <= result.s_final <= 1.0


# =============================================================================
# Tests for Batch Processing (score_repos)
# =============================================================================


class TestScoreRepos:
    """Tests for the score_repos batch processing function."""

    def test_score_repos_basic(self, eval_end):
        """Test basic batch scoring functionality."""
        # Create test data for 3 repos
        repo_urls = [
            "https://github.com/test/repo-a",
            "https://github.com/test/repo-b",
            "https://github.com/test/repo-c",
        ]

        # Commits: repo-a has commits, repo-b has none, repo-c has some
        commits_data = []
        for i in range(10):
            commits_data.append(
                {
                    "repo_url": repo_urls[0],
                    "date": eval_end - timedelta(days=30 + i * 10),
                    "is_trivial": False,
                }
            )
        for i in range(5):
            commits_data.append(
                {
                    "repo_url": repo_urls[2],
                    "date": eval_end - timedelta(days=60 + i * 20),
                    "is_trivial": False,
                }
            )
        commits_df = pd.DataFrame(commits_data)

        # PRs: only repo-a has PRs
        prs_data = [
            {
                "repo_url": repo_urls[0],
                "created_at": eval_end - timedelta(days=100),
                "closed_at": eval_end - timedelta(days=95),
                "merged_at": eval_end - timedelta(days=95),
                "state": "closed",
            }
        ]
        prs_df = pd.DataFrame(prs_data)

        # Metadata for all repos
        meta_df = pd.DataFrame(
            [
                {
                    "repo_url": repo_urls[0],
                    "stars": 1000,
                    "forks": 100,
                    "watchers": 50,
                    "open_issues": 10,
                    "archived": False,
                },
                {
                    "repo_url": repo_urls[1],
                    "stars": 500,
                    "forks": 50,
                    "watchers": 25,
                    "open_issues": 5,
                    "archived": False,
                },
                {
                    "repo_url": repo_urls[2],
                    "stars": 100,
                    "forks": 10,
                    "watchers": 5,
                    "open_issues": 1,
                    "archived": True,
                },
            ]
        )

        packages = [
            ("pkg-a", repo_urls[0]),
            ("pkg-b", repo_urls[1]),
            ("pkg-c", repo_urls[2]),
        ]

        result_df = score_repos(
            packages=packages,
            commits_df=commits_df,
            pull_requests_df=prs_df,
            repo_meta_df=meta_df,
            eval_end=eval_end,
            n_workers=2,
            show_progress=False,
        )

        # Check DataFrame structure
        assert len(result_df) == 3
        assert "source" in result_df.columns
        assert "repo_url" in result_df.columns
        assert "das_score" in result_df.columns
        assert "mrs_score" in result_df.columns
        assert "rmvs_score" in result_df.columns
        assert "final_score" in result_df.columns
        assert "error" in result_df.columns

        # All should have scores (no errors)
        assert bool(result_df["error"].isna().all())

        # Check that all expected columns from notebook are present
        expected_columns = [
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
        ]
        for col in expected_columns:
            assert col in result_df.columns, f"Missing column: {col}"

    def test_score_repos_handles_errors(self, eval_end):
        """Test that errors are captured and don't crash the batch."""
        # Create minimal valid data
        commits_df = pd.DataFrame(columns=["repo_url", "date", "is_trivial"])
        prs_df = pd.DataFrame(columns=["repo_url", "created_at", "closed_at", "merged_at", "state"])
        meta_df = pd.DataFrame(columns=["repo_url", "stars", "forks", "watchers", "open_issues", "archived"])

        packages = [
            ("pkg-good", "https://github.com/test/good"),
            ("pkg-missing", "https://github.com/test/missing"),
        ]

        result_df = score_repos(
            packages=packages,
            commits_df=commits_df,
            pull_requests_df=prs_df,
            repo_meta_df=meta_df,
            eval_end=eval_end,
            n_workers=1,
            show_progress=False,
        )

        # Should return results for all packages
        assert len(result_df) == 2

    def test_score_repos_parallel_performance(self, eval_end):
        """Test that parallel processing improves performance."""
        import time
        import numpy as np

        # Generate data for 20 repos
        n_repos = 20
        repo_urls = [f"https://github.com/test/repo-{i}" for i in range(n_repos)]

        # Generate commits
        commits_data = []
        base_date = eval_end - timedelta(days=365)
        for repo_url in repo_urls:
            for j in range(50):
                commits_data.append(
                    {
                        "repo_url": repo_url,
                        "date": base_date + timedelta(days=np.random.randint(0, 365)),
                        "is_trivial": np.random.random() < 0.1,
                    }
                )
        commits_df = pd.DataFrame(commits_data)

        # Generate PRs
        prs_data = []
        for repo_url in repo_urls:
            for j in range(10):
                created = base_date + timedelta(days=np.random.randint(0, 300))
                prs_data.append(
                    {
                        "repo_url": repo_url,
                        "created_at": created,
                        "closed_at": created + timedelta(days=5),
                        "merged_at": created + timedelta(days=5),
                        "state": "closed",
                    }
                )
        prs_df = pd.DataFrame(prs_data)

        # Generate metadata
        meta_df = pd.DataFrame(
            [
                {
                    "repo_url": url,
                    "stars": np.random.randint(100, 10000),
                    "forks": np.random.randint(10, 1000),
                    "watchers": np.random.randint(10, 500),
                    "open_issues": np.random.randint(0, 100),
                    "archived": False,
                }
                for url in repo_urls
            ]
        )

        packages = [(f"pkg-{i}", url) for i, url in enumerate(repo_urls)]

        # Run with multiple workers
        start = time.perf_counter()
        result_df = score_repos(
            packages=packages,
            commits_df=commits_df,
            pull_requests_df=prs_df,
            repo_meta_df=meta_df,
            eval_end=eval_end,
            n_workers=4,
            show_progress=False,
        )
        elapsed = time.perf_counter() - start

        # Should complete reasonably fast and return all results
        assert len(result_df) == n_repos
        assert elapsed < 10.0, f"Batch scoring took {elapsed:.2f}s, expected < 10s"

    def test_malta_result_columns_match_notebook(self):
        """Verify MaltaResult has all columns expected by the notebook."""
        # These are the columns used in the RQ1.ipynb notebook
        notebook_columns = {
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
        }

        malta_result_fields = set(MaltaResult._fields)

        assert notebook_columns == malta_result_fields, (
            f"Column mismatch!\n"
            f"Missing from MaltaResult: {notebook_columns - malta_result_fields}\n"
            f"Extra in MaltaResult: {malta_result_fields - notebook_columns}"
        )
