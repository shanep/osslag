from __future__ import annotations

import logging
import logging.config
import os
import pathlib
import sys
from pathlib import Path

import pandas as pd
import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.status import Status

from osslag import __version__
from osslag.distro import get_handler
from osslag.executor import ParallelExecutor, SuppressConsoleLogging, TaskResult
from osslag.pipeline import PipelineFiles, PipelineStepError, require_file, validate_columns
from osslag.tasks import (
    _clone_task,
    _fetch_github_repo_metadata_task,
    _fetch_github_repo_pull_requests_task,
    _load_commits_task,
)
from osslag.utils import github_helper as gh
from osslag.utils import vcs

load_dotenv()


def _load_checkpoints(checkpoint_dir: Path, console: Console) -> list[TaskResult]:
    """Load all parquet checkpoint files from a directory into TaskResult objects.

    Args:
        checkpoint_dir: Directory containing .parquet checkpoint files.
        console: Rich console for status messages.

    Returns:
        List of TaskResult objects, one per successfully loaded checkpoint file.
    """
    results: list[TaskResult] = []
    if not checkpoint_dir.exists():
        return results
    try:
        console.print(f"[green]Loading checkpointed data from {checkpoint_dir}[/]")
        for ck in checkpoint_dir.iterdir():
            if not ck.name.endswith(".parquet"):
                continue
            checkpoint_df: pd.DataFrame = pd.read_parquet(ck)
            results.append(
                TaskResult(
                    task_id="checkpoint",
                    success=True,
                    data=checkpoint_df,
                )
            )
    except Exception as e:
        console.print(f"[yellow]Warning:[/] Failed to load checkpointed data: {e}")
    return results


def version_callback(value: bool):
    if value:
        print(f"osslag {__version__}")
        raise typer.Exit()


app = typer.Typer()


@app.callback()
def main_callback(
    version: bool = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True, help="Show version"
    ),
):
    """OSS Lag - Technical Lag tools for Open Source Software Projects."""
    pass


dataset_app = typer.Typer(help="Dataset pipeline commands for building package analysis datasets.")
app.add_typer(dataset_app, name="dataset")
logger = logging.getLogger(__name__)
console = Console()


def setup_logging() -> None:
    log_level = os.getenv("LOG_LEVEL", "INFO")
    log_file = os.getenv("LOG_FILE", "osslag.log")

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s | %(name)-20s | %(funcName)-20s:%(lineno)-4d | %(levelname)-8s | %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": log_level,
                    "formatter": "standard",
                    "stream": "ext://sys.stdout",
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": log_level,
                    "formatter": "standard",
                    "filename": log_file,
                    "maxBytes": 5 * 1024 * 1024,
                    "backupCount": 3,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "osslag": {
                    "handlers": ["console", "file"],
                    "level": log_level,
                    "propagate": False,
                },
                "__main__": {
                    "handlers": ["console", "file"],
                    "level": log_level,
                    "propagate": False,
                },
            },
            "root": {
                "handlers": ["console"],
                "level": log_level,
            },
        }
    )


@app.command()
def clone(repo_url: str, dest_dir: str = "./cache/repos"):
    """Clone a single Git repository to the specified destination directory."""
    dest_dir = os.path.abspath(dest_dir)
    print(f"Cloning repository {repo_url} into directory {dest_dir}")
    success = vcs.clone_repo(repo_url, dest_dir)
    if success:
        print(f"Successfully cloned or updated repository {repo_url} into {dest_dir}")
    else:
        print(f"Failed to clone repository {repo_url} into {dest_dir}")


@app.command()
def get_metadata(
    repo_url: str,
    cache: str = typer.Option("./cache", help="Output path for metadata parquet file"),
):
    """Fetch GitHub repository metadata and save to a parquet file."""
    github_token = os.getenv("GITHUB_TOKEN")
    cache_path = os.getenv("CACHE_DIR") or cache
    pathlib.Path(cache_path).mkdir(parents=True, exist_ok=True)
    print(f"Fetching metadata for repository {repo_url}")
    try:
        metadata_df = gh.fetch_github_repo_metadata(repo_url, github_token)
        parquet_path = Path(cache_path, "metadata.parquet")
        metadata_df.to_parquet(parquet_path)
        print(f"Metadata saved to {parquet_path}")
    except Exception as e:
        print(f"Failed to fetch metadata for {repo_url}: {e}")


@app.command()
def rate_limit():
    """Fetch and display GitHub API rate limit information."""
    github_token = os.getenv("GITHUB_TOKEN")
    if github_token is None:
        print("GITHUB_TOKEN is not set in environment variables.")
        return
    masked = f"...{github_token[-4:]}" if github_token and len(github_token) >= 4 else "<not set>"
    print(f"Using token: {masked}")
    rate_info = gh.gh_get_rate_limit_info(github_token)
    if rate_info is not None:
        print(
            f"GitHub API Rate Limit: {rate_info['limit']}/{rate_info['remaining']} remaining (resets at {rate_info['reset_datetime']})"
        )
    else:
        print("Failed to fetch rate limit info from GitHub.")


@app.command()
def pull_requests(
    repo_url: str = typer.Argument(..., help="The GitHub repository URL to fetch pull requests for"),
    cache: str = typer.Option("./cache", help="Cache directory"),
):
    """Fetch GitHub pull requests for a specified repository and save to a parquet file."""
    github_token = os.getenv("GITHUB_TOKEN")
    cache_path = os.getenv("CACHE_DIR") or cache
    pathlib.Path(cache_path).mkdir(parents=True, exist_ok=True)
    output_path = Path(cache_path, "pull_requests.parquet")
    print(f"Fetching pull requests for repository {repo_url}")
    try:
        pr_df = gh.fetch_pull_requests(repo_url, github_token)
        pr_df.to_parquet(output_path)
        print(f"Pull requests saved to {output_path}")
    except Exception as e:
        print(f"Failed to fetch pull requests for {repo_url}: {e}")


@dataset_app.command(name="run", rich_help_panel="Full Pipeline")
def run_dataset_pipeline(
    distro: str = typer.Option("debian", help="The Linux distribution to process (e.g., 'debian' 'fedora')"),
    releases: list[str] = typer.Option(
        ...,
        "--release",
        help="One or more distro releases to process (e.g., 'trixie', 'bookworm', '40'). Can repeat flag or use comma-separated.",
    ),
    cache: str = typer.Option("./cache", help="Cache directory (EV: CACHE_DIR)"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-processing even if cache exists"),
):
    """Run the full pipeline: fetch packages, filter repos, extract versions,
    merge releases, clone repos, load commits, pull GitHub data.

    Uses cached data when available. Use --force to re-process all steps.
    """
    # Flatten: handle both repeated flags and comma-separated values
    to_process = []
    for r in releases:
        to_process.extend([item.strip() for item in r.split(",")])

    cache_dir = os.getenv("CACHE_DIR") or cache

    console.print(
        Panel(
            f"[bold]Distro:[/] {distro}  [bold]Releases:[/] {', '.join(to_process)}  [bold]Cache:[/] {cache_dir}"
            + ("  [bold yellow]--force[/]" if force else ""),
            title="[bold blue]🚀 Dataset Pipeline[/]",
            border_style="blue",
        )
    )

    try:
        # Suppress console logging for steps 1-4 (non-parallel steps)
        with SuppressConsoleLogging():
            # Step 1: Get and cache package data for each release
            with Status("[bold cyan]Step 1/8:[/] Fetching packages...", console=console):
                fetch_packages(distro=distro, releases=to_process, cache=cache_dir)
            console.print("[green]✓[/] Step 1/8: Fetched packages")

            # Step 2: Filter GitHub repos
            with Status("[bold cyan]Step 2/8:[/] Filtering GitHub repos...", console=console):
                filter_debian_github_repos(distro=distro, release=to_process, cache=cache_dir, force=force)
            console.print("[green]✓[/] Step 2/8: Filtered GitHub repos")

            # Step 3: Extract the version string and add upstream version columns
            with Status("[bold cyan]Step 3/8:[/] Extracting upstream versions...", console=console):
                extract_upstream_versions(distro=distro, release=to_process, cache=cache_dir, force=force)
            console.print("[green]✓[/] Step 3/8: Extracted upstream versions")

            # Step 4: Merge releases into a single DataFrame with all required columns
            with Status("[bold cyan]Step 4/8:[/] Merging releases...", console=console):
                merge_releases(distro=distro, releases=to_process, cache=cache_dir, force=force)
            console.print("[green]✓[/] Step 4/8: Merged releases")

        # Step 5: Clone all upstream GitHub repos (has its own UI)
        console.print("\n[bold cyan]Step 5/8:[/] Cloning repositories...")
        clone_upstream_repos(distro=distro, cache=cache_dir)

        # Step 6: Extract all commits into a single DataFrame (has its own UI)
        console.print("\n[bold cyan]Step 6/8:[/] Loading commits...")
        load_commits_into_dataframe(distro=distro, cache=cache_dir, force=force)

        # Step 7: Fetch GitHub metadata for all repos (has its own UI)
        console.print("\n[bold cyan]Step 7/8:[/] Fetching GitHub metadata...")
        all_github_metadata(distro=distro, cache=cache_dir, force=force)

        # Step 8: Fetch GitHub pull requests for all repos (has its own UI)
        console.print("\n[bold cyan]Step 8/8:[/] Fetching GitHub pull requests...")
        all_github_pull_requests(distro=distro, cache=cache_dir, force=force)

    except PipelineStepError as e:
        console.print(f"\n[red bold]Pipeline failed:[/] {e}")
        raise typer.Exit(code=1)

    console.print(
        Panel(
            "[bold green]Pipeline completed successfully![/]",
            border_style="green",
        )
    )


@dataset_app.command(rich_help_panel="Step 1: Fetch Data")
def fetch_packages(
    distro: str = typer.Argument(
        ...,
        help="The Linux distribution to fetch packages for (e.g., 'debian' 'fedora')",
    ),
    releases: list[str] = typer.Argument(
        ...,
        help="The release(s) to fetch packages for (e.g., 'trixie', 'bookworm', '40')",
    ),
    cache: str = typer.Option("./cache", help="Cache directory"),
):
    """Fetch and cache distribution package data for specified releases."""
    cache_dir = os.getenv("CACHE_DIR") or cache

    # Ensure cache directory exists
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    handler = get_handler(distro)
    for rel in releases:
        parquet_path = PipelineFiles.all_packages(cache_dir, distro, rel)
        if parquet_path.exists():
            logger.info(f"Using cached {rel} packages from {parquet_path}")
            continue

        # Show status since this can take a while (large file download + parsing)
        with Status(
            f"[bold cyan]Fetching {rel} packages (this may take a minute)...[/]",
            console=console,
        ):
            logger.info(f"Fetching and caching {rel} packages to {parquet_path}")
            df: pd.DataFrame | None = handler.fetch_packages(rel)
            if df is None:
                raise PipelineStepError(f"Failed to fetch {rel} packages from upstream.")
            df.to_parquet(parquet_path)
            console.print(f"[green]✓ Fetched {len(df):,} {rel} packages[/]")


@dataset_app.command(rich_help_panel="Step 2: Filter Repos")
def filter_debian_github_repos(
    distro: str = typer.Argument(..., help="The Linux distribution to process (e.g., 'debian' 'fedora')"),
    release: list[str] = typer.Argument(
        ...,
        help="One or more distro releases to process (e.g., 'trixie', 'bookworm', '40'). Can repeat flag or use comma-separated.",
    ),
    cache: str = typer.Option("./cache", help="Cache directory"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-processing even if cache exists"),
):
    """Filter distro package DataFrames to only include GitHub repositories."""
    cache_dir = os.getenv("CACHE_DIR") or cache

    handler = get_handler(distro)
    for rel in release:
        filtered_parquet_path = PipelineFiles.filtered_packages(cache_dir, distro, rel)
        if filtered_parquet_path.exists() and not force:
            logger.info(f"Using cached filtered packages from {filtered_parquet_path}")
            continue

        parquet_path = require_file(
            PipelineFiles.all_packages(cache_dir, distro, rel),
            "Run 'fetch-packages' first.",
        )

        logger.info(f"Filtering GitHub repositories for {distro} release '{rel}'")
        df: pd.DataFrame = pd.read_parquet(parquet_path)
        validate_columns(df, ["homepage"], f"Step 2 ({rel})")
        size_before = df.shape[0]
        filtered_df = handler.filter_github_repos(df)
        size_after = filtered_df.shape[0]
        logger.info(f"Dropped {size_before - size_after} packages due to non-GitHub '{rel}'.")
        filtered_df = handler.add_local_repo_cache_path_column(filtered_df, cache_dir=cache_dir)
        filtered_df.reset_index(drop=True, inplace=True)
        filtered_df.to_parquet(filtered_parquet_path)
        logger.info(f"Filtered GitHub repositories saved to {filtered_parquet_path}")


@dataset_app.command(rich_help_panel="Step 3: Extract Versions")
def extract_upstream_versions(
    distro: str = typer.Argument(..., help="The Linux distribution to process (e.g., 'debian' 'fedora')"),
    release: list[str] = typer.Argument(
        ...,
        help="One or more distro releases to process (e.g., 'trixie', 'bookworm', '40'). Can repeat flag or use comma-separated.",
    ),
    cache: str = typer.Option("./cache", help="Cache directory"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-processing even if cache exists"),
):
    """Extract upstream version strings from Debian package versions and add as a new column."""
    cache_dir = os.getenv("CACHE_DIR") or cache

    handler = get_handler(distro)
    for rel in release:
        versions_parquet_path = PipelineFiles.packages_with_versions(cache_dir, distro, rel)
        if versions_parquet_path.exists() and not force:
            logger.info(f"Using cached upstream versions from {versions_parquet_path}")
            continue

        filtered_parquet_path = require_file(
            PipelineFiles.filtered_packages(cache_dir, distro, rel),
            "Run 'filter-debian-github-repos' first.",
        )

        logger.info(f"Extracting upstream versions for {distro} release '{rel}'")
        df: pd.DataFrame = pd.read_parquet(filtered_parquet_path)
        validate_columns(df, [f"{rel}_version"], f"Step 3 ({rel})")
        version_column = f"{rel}_upstream_version"
        df_with_versions = handler.add_upstream_version_column(df, f"{rel}_version", new_column_name=version_column)
        drop_before = df_with_versions.shape[0]
        df_with_versions.dropna(subset=[version_column], inplace=True)
        drop_after = df_with_versions.shape[0]
        logger.info(f"Dropped {drop_before - drop_after} rows with missing upstream versions for release '{rel}'.")
        df_with_versions.reset_index(drop=True, inplace=True)
        df_with_versions.to_parquet(versions_parquet_path)
        logger.info(f"Upstream versions extracted and saved to {versions_parquet_path}")


@dataset_app.command(rich_help_panel="Step 4: Merge Releases")
def merge_releases(
    distro: str = typer.Argument(..., help="The Linux distribution to process (e.g., 'debian' 'fedora')"),
    releases: list[str] = typer.Argument(
        ...,
        help="One or more distro releases to merge (e.g., 'trixie', 'bookworm', '40'). Can repeat flag or use comma-separated.",
    ),
    cache: str = typer.Option("./cache", help="Cache directory"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-processing even if cache exists"),
):
    """Merge multiple release DataFrames into a single DataFrame with all required columns."""
    cache_dir = os.getenv("CACHE_DIR") or cache

    handler = get_handler(distro)
    merged_parquet_path = PipelineFiles.merged_packages(cache_dir, distro)
    if merged_parquet_path.exists() and not force:
        logger.info(f"Using cached merged releases from {merged_parquet_path}")
        return

    dfs = []
    for rel in releases:
        versions_parquet_path = require_file(
            PipelineFiles.packages_with_versions(cache_dir, distro, rel),
            "Run 'extract-upstream-versions' first.",
        )

        logger.info(f"Loading packages with upstream versions for {distro} release '{rel}'")
        df: pd.DataFrame = pd.read_parquet(versions_parquet_path)
        validate_columns(df, ["source"], f"Step 4 ({rel})")
        dfs.append(df)
    merged_df, dropped_after_merge = handler.merge_release_packages(dfs)
    logger.info(
        f"Merged releases {releases}. Dropped {dropped_after_merge.shape[0]} rows that were not present in all releases."
    )
    merged_df.reset_index(drop=True, inplace=True)
    merged_df.to_parquet(merged_parquet_path)
    logger.info(f"Merged release packages saved to {merged_parquet_path}")
    dropped_path = PipelineFiles.dropped_after_merge(cache_dir, distro)
    dropped_after_merge.to_parquet(dropped_path)
    logger.info(f"Dropped rows after merge saved to {dropped_path}")


@dataset_app.command(rich_help_panel="Step 5: Clone Repos")
def clone_upstream_repos(
    distro: str = typer.Argument(..., help="The distro for (e.g., 'debian' 'fedora', etc.)"),
    repos_cache: str = typer.Option("./cache/repos", help="Cache directory for cloned repositories"),
    cache: str = typer.Option("./cache", help="Cache directory"),
    max_workers: int = typer.Option(4, help="Maximum number of parallel clone processes (env: MAX_WORKERS)"),
):
    """Clone all upstream GitHub repositories in the filtered package DataFrame."""
    cache_dir = os.getenv("CACHE_DIR") or cache
    repos_cache = os.getenv("REPOS_CACHE_DIR") or repos_cache
    max_workers = int(os.getenv("MAX_WORKERS", str(max_workers)))
    get_handler(distro)  # validate distro name
    parquet_path = require_file(
        PipelineFiles.merged_packages(cache_dir, distro),
        "Run 'merge-releases' first.",
    )

    # Suppress logging during setup
    with SuppressConsoleLogging():
        df: pd.DataFrame = pd.read_parquet(parquet_path)
        validate_columns(df, ["upstream_repo_url"], "Step 5 (clone)")
        repos_cache_path = pathlib.Path(repos_cache)

        vcs.ensure_dir(repos_cache_path)

        # Build list of repos to clone (skip already cloned)
        clone_tasks: list[tuple[str, str]] = []
        skipped = 0
        invalid = 0
        for row in df.itertuples():
            repo_url = str(row.upstream_repo_url)
            target_dir = vcs.construct_repo_local_path(repo_url, cache_dir=repos_cache_path, must_exist=False)
            if target_dir is None:
                invalid += 1
                continue
            if target_dir.exists():
                skipped += 1
                continue
            # Skip if there is a .failed marker file
            failed_marker = repos_cache_path / f"{target_dir.name}.failed"
            if failed_marker.exists():
                skipped += 1
                continue

            clone_tasks.append((repo_url, str(target_dir)))

    if invalid > 0:
        console.print(f"[yellow]Skipped {invalid} invalid repository URLs[/]")

    if len(clone_tasks) == 0:
        console.print(f"[green]All {skipped} repositories already cloned.[/]")
        return

    # Use the parallel executor with fancy UI
    executor = ParallelExecutor(
        task_name=f"Cloning {distro.title()} Repositories",
        max_workers=max_workers,
    )
    executor.run(
        tasks=clone_tasks,
        worker_fn=_clone_task,
        task_id_fn=lambda t: t[0],  # repo_url
        skipped=skipped,
    )


@dataset_app.command(rich_help_panel="Step 6: Load Commits")
def load_commits_into_dataframe(
    distro: str = typer.Argument(..., help="The Linux distribution to process (e.g., 'debian' 'fedora')"),
    cache: str = typer.Option("./cache", help="Cache directory"),
    repos_cache: str = typer.Option("./cache/repos", help="Cache directory for cloned repositories (env: REPOS_CACHE_DIR)"),
    max_workers: int = typer.Option(4, help="Maximum number of parallel worker processes (env: MAX_WORKERS)"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-processing even if cache exists"),
):
    """Load all GitHub commits for the upstream repositories into a single DataFrame."""
    cache_dir = os.getenv("CACHE_DIR") or cache
    repo_cache_dir = os.getenv("REPOS_CACHE_DIR") or repos_cache
    checkpoint_dir = PipelineFiles.commit_checkpoints_dir(cache_dir)
    max_workers = int(os.getenv("MAX_WORKERS", str(max_workers)))

    commits_parquet_path = PipelineFiles.all_commits(cache_dir, distro)
    if commits_parquet_path.exists() and not force:
        console.print(f"[green]Using cached commits from {commits_parquet_path}[/]")
        return

    all_packages_parquet_path = require_file(
        PipelineFiles.merged_packages(cache_dir, distro),
        "Run 'merge-releases' and 'clone-upstream-repos' first.",
    )
    # Create checkpoint directory
    vcs.ensure_dir(checkpoint_dir)

    if force and checkpoint_dir.exists():
        console.print(f"[yellow]Removing existing checkpoint at {checkpoint_dir}[/]")
        for ck in checkpoint_dir.iterdir():
            if ck.name.endswith(".parquet"):
                ck.unlink()

    df: pd.DataFrame = pd.read_parquet(all_packages_parquet_path)
    validate_columns(df, ["upstream_repo_url", "source"], "Step 6 (commits)")

    # Build list of tasks (skip repos without local paths)
    tasks: list[tuple[str, str, str, str]] = []
    skipped = 0
    for row in df.itertuples():
        repo_url = str(row.upstream_repo_url)
        local_repo_path = vcs.construct_repo_local_path(repo_url, cache_dir=Path(repo_cache_dir), must_exist=True)
        if local_repo_path is None or not local_repo_path.exists():
            skipped += 1
            continue
        source = str(row.source)
        if Path(checkpoint_dir, f"{source}.parquet").exists():
            skipped += 1
            continue
        tasks.append((str(local_repo_path), repo_url, source, str(checkpoint_dir)))
    results: list[TaskResult] = []
    # Run tasks if any
    if len(tasks) > 0:
        # Use the parallel executor with fancy UI
        executor = ParallelExecutor(
            task_name=f"Loading {distro.title()} Commits",
            max_workers=max_workers,
        )
        results = executor.run(
            tasks=tasks,
            worker_fn=_load_commits_task,
            task_id_fn=lambda t: t[1],  # repo_url
            skipped=skipped,
        )
    # Collect all the checkpointed DataFrames
    results = _load_checkpoints(checkpoint_dir, console)

    # Collect successful DataFrames
    all_commits = [r.data for r in results if r.success and r.data is not None]

    if all_commits:
        console.print(f"[green]Loaded commits from {len(all_commits)} repositories.[/]")
        combined_commits_df = pd.concat(all_commits, ignore_index=True)
        combined_commits_df.to_parquet(commits_parquet_path)
        console.print(f"[green]Saved {len(combined_commits_df):,} commits to {commits_parquet_path}[/]")
    else:
        console.print("[yellow]No commits were loaded from any repositories.[/]")


@dataset_app.command(rich_help_panel="Step 7: GitHub Metadata")
def all_github_metadata(
    distro: str = typer.Option("debian", help="The Linux distribution to process (default: debian)"),
    cache: str = typer.Option("./cache", help="Cache directory"),
    max_workers: int = typer.Option(4, help="Maximum number of parallel GitHub API workers (env: MAX_WORKERS)"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-processing even if cache exists"),
):
    """Fetch GitHub repository metadata for all unique repos in the commits parquet file."""
    cache_dir = os.getenv("CACHE_DIR") or cache
    max_workers = int(os.getenv("MAX_WORKERS", str(max_workers)))
    all_packages_parquet_path = PipelineFiles.merged_packages(cache_dir, distro)
    output_parquet_path = PipelineFiles.all_metadata(cache_dir, distro)
    checkpoint_dir = PipelineFiles.metadata_checkpoints_dir(cache_dir)
    # Create checkpoint directory
    vcs.ensure_dir(checkpoint_dir)

    if force and checkpoint_dir.exists():
        console.print(f"[yellow]Removing existing GitHub metadata checkpoint at {checkpoint_dir}[/]")
        for ck in checkpoint_dir.iterdir():
            if ck.name.endswith(".parquet"):
                ck.unlink()

    if output_parquet_path.exists() and not force:
        console.print(f"[green]Using cached GitHub metadata from {output_parquet_path}[/]")
        return

    require_file(all_packages_parquet_path, "Run 'merge-releases' first.")

    df: pd.DataFrame = pd.read_parquet(all_packages_parquet_path)
    validate_columns(df, ["upstream_repo_url", "source"], "Step 7 (metadata)")

    # Build list of tasks (skip repos without local paths)
    tasks: list[tuple[str, str, str]] = []
    skipped = 0
    for row in df.itertuples():
        repo_url = str(row.upstream_repo_url)
        source = str(row.source)
        if Path(checkpoint_dir, f"{source}.parquet").exists():
            skipped += 1
            continue
        # Skip if there is a .failed marker file
        failed_marker = checkpoint_dir / f"{source}.failed"
        if failed_marker.exists():
            skipped += 1
            continue
        tasks.append((repo_url, source, str(checkpoint_dir)))
    results: list[TaskResult] = []

    # Display rate limit info
    github_token = os.getenv("GITHUB_TOKEN")
    rate_info = gh.gh_get_rate_limit_info(github_token)
    rate_limit = rate_info["limit"] if rate_info else None
    rate_remaining = rate_info["remaining"] if rate_info else None
    if rate_info:
        console.print(
            f"[cyan]GitHub API Rate Limit:[/] {rate_info['remaining']}/{rate_info['limit']} remaining (resets at {rate_info['reset_datetime']})"
        )
    else:
        console.print("[yellow]Warning:[/] Could not fetch rate limit info")

    console.print(f"[cyan]Fetching GitHub metadata for {len(tasks)} repositories...[/]")
    executor = ParallelExecutor(
        task_name="GitHub Metadata Fetch",
        max_workers=min(max_workers, len(tasks)),
        rate_limit=rate_limit,
        rate_remaining=rate_remaining,
    )
    results = executor.run(
        tasks=tasks,
        worker_fn=_fetch_github_repo_metadata_task,
        task_id_fn=lambda t: t[0],
        skipped=skipped,
    )
    # Collect all the checkpointed DataFrames
    results = _load_checkpoints(checkpoint_dir, console)
    # Collect successful DataFrames
    all_metadata = [r.data for r in results if r.success and r.data is not None]

    if all_metadata:
        console.print(f"[green]Loaded metadata from {len(all_metadata)} repositories.[/]")
        combined_metadata_df = pd.concat(all_metadata, ignore_index=True)
        combined_metadata_df.to_parquet(output_parquet_path)
        console.print(f"[green]Saved {len(combined_metadata_df):,} metadata entries to {output_parquet_path}[/]")
    else:
        console.print("[yellow]No metadata entries were loaded from any repositories.[/]")


@dataset_app.command(rich_help_panel="Step 8: GitHub Metadata")
def all_github_pull_requests(
    distro: str = typer.Option("debian", help="The Linux distribution to process (default: debian)"),
    cache: str = typer.Option("./cache", help="Cache directory"),
    max_workers: int = typer.Option(4, help="Maximum number of parallel GitHub API workers (env: MAX_WORKERS)"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-processing even if cache exists"),
):
    """Fetch GitHub repository pull requests for all unique repos in the commits parquet file."""
    cache_dir = os.getenv("CACHE_DIR") or cache
    max_workers = int(os.getenv("MAX_WORKERS", str(max_workers)))
    all_packages_parquet_path = PipelineFiles.merged_packages(cache_dir, distro)
    output_parquet_path = PipelineFiles.all_pull_requests(cache_dir, distro)
    checkpoint_dir = PipelineFiles.pr_checkpoints_dir(cache_dir)
    # Create checkpoint directory
    vcs.ensure_dir(checkpoint_dir)

    if output_parquet_path.exists() and not force:
        console.print(f"[green]Using cached GitHub pull requests from {output_parquet_path}[/]")
        return

    require_file(all_packages_parquet_path, "Run 'merge-releases' first.")

    if force and checkpoint_dir.exists():
        console.print(f"[yellow]Removing existing GitHub pull requests checkpoint at {checkpoint_dir}[/]")
        for ck in checkpoint_dir.iterdir():
            if ck.name.endswith(".parquet"):
                ck.unlink()

    df: pd.DataFrame = pd.read_parquet(all_packages_parquet_path)
    validate_columns(df, ["upstream_repo_url", "source"], "Step 8 (pull requests)")

    # Build list of tasks (skip repos without local paths)
    tasks: list[tuple[str, str, str]] = []
    skipped = 0
    for row in df.itertuples():
        repo_url = str(row.upstream_repo_url)
        source = str(row.source)
        if Path(checkpoint_dir, f"{source}.parquet").exists():
            skipped += 1
            continue
        # Skip if there is a .failed marker file
        failed_marker = checkpoint_dir / f"{source}.failed"
        if failed_marker.exists():
            skipped += 1
            continue
        tasks.append((repo_url, source, str(checkpoint_dir)))
    results: list[TaskResult] = []

    # Display rate limit info
    github_token = os.getenv("GITHUB_TOKEN")
    rate_info = gh.gh_get_rate_limit_info(github_token)
    if rate_info:
        console.print(
            f"[cyan]GitHub API Rate Limit:[/] {rate_info['remaining']}/{rate_info['limit']} remaining (resets at {rate_info['reset_datetime']})"
        )
    else:
        console.print("[yellow]Warning:[/] Could not fetch rate limit info")

    console.print(f"[cyan]Fetching GitHub pull requests for {len(tasks)} repositories...[/]")
    executor = ParallelExecutor(
        task_name="GitHub Pull Requests Fetch",
        max_workers=min(max_workers, len(tasks)),
    )
    results = executor.run(
        tasks=tasks,
        worker_fn=_fetch_github_repo_pull_requests_task,
        task_id_fn=lambda t: t[0],
        skipped=skipped,
    )
    # Collect all the checkpointed DataFrames
    results = _load_checkpoints(checkpoint_dir, console)
    # Collect successful DataFrames
    all_metadata = [r.data for r in results if r.success and r.data is not None]

    if all_metadata:
        console.print(f"[green]Loaded pull requests from {len(all_metadata)} repositories.[/]")
        combined_metadata_df = pd.concat(all_metadata, ignore_index=True)
        combined_metadata_df.to_parquet(output_parquet_path)
        console.print(f"[green]Saved {len(combined_metadata_df):,} pull request entries to {output_parquet_path}[/]")
    else:
        console.print("[yellow]No pull request entries were loaded from any repositories.[/]")


@app.command()
def show_cache():
    """Show the current cache directory."""
    cache_dir = os.getenv("CACHE_DIR", "./cache")
    console.print(f"[blue]Current cache directory:[/] {cache_dir}")


def main():
    """Main entry point for the CLI application."""
    setup_logging()
    # Show help menu if no arguments provided
    if len(sys.argv) == 1:
        app(["--help"])
    else:
        app()
