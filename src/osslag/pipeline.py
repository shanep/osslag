"""Pipeline infrastructure: file paths, validation, and error handling.

Centralises all parquet artifact names so producers and consumers share a
single source of truth.  Also provides lightweight schema validation and
a custom exception that the CLI pipeline runner can catch to abort cleanly.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


class PipelineStepError(Exception):
    """Raised when a pipeline step fails due to missing files or invalid data."""


class PipelineFiles:
    """Single source of truth for every pipeline artifact path.

    Each static method returns the canonical ``Path`` for one artifact.
    The naming convention is ``{distro}_{release}_<step>.parquet`` for
    per-release files and ``{distro}_<kind>.parquet`` for merged/global
    files.
    """

    # ------------------------------------------------------------------
    # Per-release files (steps 1-3)
    # ------------------------------------------------------------------

    @staticmethod
    def all_packages(cache_dir: str | os.PathLike, distro: str, release: str) -> Path:
        """Step 1 output — raw package list for one release."""
        return Path(cache_dir, f"{distro}_{release}_all_packages.parquet")

    @staticmethod
    def filtered_packages(cache_dir: str | os.PathLike, distro: str, release: str) -> Path:
        """Step 2 output — packages filtered to GitHub repos."""
        return Path(cache_dir, f"{distro}_{release}_filtered_packages.parquet")

    @staticmethod
    def packages_with_versions(cache_dir: str | os.PathLike, distro: str, release: str) -> Path:
        """Step 3 output — filtered packages with upstream version column."""
        return Path(cache_dir, f"{distro}_{release}_packages_with_upstream_versions.parquet")

    # ------------------------------------------------------------------
    # Merged / global files (steps 4-8)
    # ------------------------------------------------------------------

    @staticmethod
    def merged_packages(cache_dir: str | os.PathLike, distro: str) -> Path:
        """Step 4 output — releases merged into one DataFrame."""
        return Path(cache_dir, f"{distro}_merged_releases_packages.parquet")

    @staticmethod
    def dropped_after_merge(cache_dir: str | os.PathLike, distro: str) -> Path:
        """Step 4 side-output — rows dropped during merge."""
        return Path(cache_dir, f"{distro}_dropped_after_merge.parquet")

    @staticmethod
    def all_commits(cache_dir: str | os.PathLike, distro: str) -> Path:
        """Step 6 output — combined upstream commits."""
        return Path(cache_dir, f"{distro}_all_upstream_commits.parquet")

    @staticmethod
    def all_metadata(cache_dir: str | os.PathLike, distro: str) -> Path:
        """Step 7 output — combined GitHub metadata."""
        return Path(cache_dir, f"{distro}_all_upstream_metadata.parquet")

    @staticmethod
    def all_pull_requests(cache_dir: str | os.PathLike, distro: str) -> Path:
        """Step 8 output — combined GitHub pull requests."""
        return Path(cache_dir, f"{distro}_all_upstream_pull_requests.parquet")

    # ------------------------------------------------------------------
    # Checkpoint directories
    # ------------------------------------------------------------------

    @staticmethod
    def commit_checkpoints_dir(cache_dir: str | os.PathLike) -> Path:
        return Path(cache_dir, "commit_checkpoints")

    @staticmethod
    def metadata_checkpoints_dir(cache_dir: str | os.PathLike) -> Path:
        return Path(cache_dir, "github_metadata_checkpoints")

    @staticmethod
    def pr_checkpoints_dir(cache_dir: str | os.PathLike) -> Path:
        return Path(cache_dir, "github_pr_checkpoints")


# ------------------------------------------------------------------
# Validation helpers
# ------------------------------------------------------------------


def require_file(path: Path, hint: str) -> Path:
    """Return *path* if it exists, otherwise raise ``PipelineStepError``.

    Parameters
    ----------
    path:
        File path to check.
    hint:
        Human-readable suggestion appended to the error message
        (e.g. ``"Run 'fetch-packages' first."``).
    """
    if not path.exists():
        raise PipelineStepError(f"Required file {path} does not exist. {hint}")
    return path


def validate_columns(df: pd.DataFrame, required: list[str], context: str) -> None:
    """Raise ``PipelineStepError`` if *df* is missing any *required* columns.

    Parameters
    ----------
    df:
        DataFrame to inspect.
    required:
        Column names that must be present.
    context:
        Label for error messages (e.g. ``"Step 5 (clone)"``).
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise PipelineStepError(
            f"{context}: missing required columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )
