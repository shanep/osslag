"""Worker task functions for parallel execution.

These functions are designed to run in separate processes via
ProcessPoolExecutor. Each accepts a tuple of arguments, performs
a single unit of work, and returns a TaskResult.
"""

from __future__ import annotations

from pathlib import Path

from osslag.executor import TaskResult
from osslag.utils import github_helper as gh
from osslag.utils import vcs


def _fetch_github_repo_metadata_task(args: tuple[str, str, str]) -> TaskResult:
    """Worker function to fetch GitHub repo metadata with proper timeout and rate limit handling."""
    repo_url, source, cache_path = args
    checkpoint_path = Path(cache_path, f"{source}.parquet")
    try:
        metadata_df = gh.fetch_github_repo_metadata(repo_url)
        # Save checkpoint
        metadata_df.to_parquet(checkpoint_path)
        return TaskResult(
            task_id=repo_url,
            success=True,
            data=metadata_df,
        )

    except Exception as e:
        failed_marker = checkpoint_path.parent / f"{checkpoint_path.name}.failed"
        return TaskResult(
            task_id=repo_url,
            success=False,
            error=str(e),
            failed_marker_path=failed_marker,
        )


def _fetch_github_repo_pull_requests_task(args: tuple[str, str, str]) -> TaskResult:
    """Worker function to fetch GitHub repo pull requests with proper timeout and rate limit handling."""
    repo_url, source, cache_path = args
    checkpoint_path = Path(cache_path, f"{source}.parquet")
    try:
        pull_requests_df = gh.fetch_pull_requests(repo_url)
        # Save checkpoint
        pull_requests_df.to_parquet(checkpoint_path)
        return TaskResult(
            task_id=repo_url,
            success=True,
            data=pull_requests_df,
        )

    except Exception as e:
        failed_marker = checkpoint_path.parent / f"{checkpoint_path.name}.failed"
        return TaskResult(
            task_id=repo_url,
            success=False,
            error=str(e),
            failed_marker_path=failed_marker,
        )


def _clone_task(args: tuple[str, str]) -> TaskResult:
    """Worker function for parallel cloning. Returns TaskResult."""
    repo_url, target_dir = args
    try:
        result = vcs.clone_repo(repo_url, target_dir)
        return TaskResult(
            task_id=repo_url,
            success=result.success,
            error=result.error,
        )
    except Exception as e:
        return TaskResult(task_id=repo_url, success=False, error=str(e))


def _load_commits_task(args: tuple[str, str, str, str]) -> TaskResult:
    """Worker function for parallel commit loading. Returns TaskResult with DataFrame."""
    local_repo_path, repo_url, source, cache_path = args
    checkpoint_path = Path(cache_path, f"{source}.parquet")
    try:
        repo_commits_df = vcs.load_commits(local_repo_path, include_files=True)
        repo_commits_df["repo_url"] = repo_url
        repo_commits_df["source"] = source
        # Save checkpoint
        repo_commits_df.to_parquet(checkpoint_path)
        return TaskResult(
            task_id=repo_url,
            success=True,
            data=repo_commits_df,
        )
    except Exception as e:
        error_detail = f"{type(e).__name__}: {str(e)}"
        return TaskResult(task_id=repo_url, success=False, error=error_detail)
