from __future__ import annotations

import logging
import logging.config
import os
import pathlib
import sys
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import typer
from dotenv import load_dotenv
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)
from rich.status import Status
from rich.table import Table
from rich.text import Text

from osslag.distro import debian as deb
from osslag.utils import github_helper as gh
from osslag.utils import vcs

load_dotenv()
app = typer.Typer()
dataset_app = typer.Typer(
    help="Dataset pipeline commands for building package analysis datasets."
)
app.add_typer(dataset_app, name="dataset")
logger = logging.getLogger(__name__)
console = Console()


# region Tasks
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


# endregion


class SuppressConsoleLogging:
    """Context manager to temporarily suppress console logging output."""

    def __enter__(self):
        # Find and disable all console/stream handlers, saving their original levels
        self._disabled_handlers: list[tuple[logging.Handler, int]] = []
        for name in list(logging.Logger.manager.loggerDict.keys()) + ["", "root"]:
            log = logging.getLogger(name) if name else logging.getLogger()
            for handler in log.handlers[:]:
                if isinstance(handler, logging.StreamHandler) and not isinstance(
                    handler, logging.FileHandler
                ):
                    original_level = handler.level
                    handler.setLevel(logging.CRITICAL + 1)  # Effectively disable
                    self._disabled_handlers.append((handler, original_level))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original levels
        for handler, original_level in self._disabled_handlers:
            handler.setLevel(original_level)
        return False


@dataclass
class TaskResult:
    """Result from a parallel task execution."""

    task_id: str
    success: bool
    error: str | None = None
    data: Any = None
    failed_marker_path: pathlib.Path | None = None


@dataclass
class WorkerStatus:
    """Tracks the status of a single worker."""

    worker_id: int
    current_task: str | None = None
    tasks_completed: int = 0
    tasks_failed: int = 0


class ParallelExecutor:
    """Generic parallel task executor with a fancy Rich CLI UI.

    Usage:
        executor = ParallelExecutor(
            task_name="Cloning repositories",
            max_workers=4,
        )
        results = executor.run(
            tasks=[(url, path), ...],
            worker_fn=clone_task,
            task_id_fn=lambda t: t[0],  # Extract task ID for display
        )
    """

    def __init__(
        self,
        task_name: str,
        max_workers: int = 4,
        show_recent_completed: int = 5,
        rate_limit: int | None = None,
        rate_remaining: int | None = None,
    ):
        self.task_name = task_name
        self.max_workers = max_workers
        self.show_recent_completed = show_recent_completed

        # Rate limit info
        self.rate_limit = rate_limit
        self.rate_remaining = rate_remaining

        # Tracking state
        self.workers: dict[int, WorkerStatus] = {}
        self.completed_tasks: list[TaskResult] = []
        self.failed_tasks: list[TaskResult] = []
        self.recent_completed: list[tuple[str, bool]] = []  # (task_id, success)
        self.total_tasks = 0
        self.skipped_tasks = 0

        # Timing
        self.start_time: float | None = None

    def create_display(self, progress: Progress) -> Panel:
        """Create the rich display panel."""
        # Stats section
        completed = len(self.completed_tasks)
        failed = len(self.failed_tasks)
        in_progress = sum(1 for w in self.workers.values() if w.current_task)

        stats_table = Table.grid(padding=(0, 2))
        stats_table.add_column(style="cyan", justify="right")
        stats_table.add_column(style="white")
        stats_table.add_column(style="cyan", justify="right")
        stats_table.add_column(style="white")

        stats_table.add_row(
            "Total:",
            f"{self.total_tasks}",
            "Skipped:",
            f"{self.skipped_tasks}",
        )
        stats_table.add_row(
            "✓ Completed:",
            f"[green]{completed}[/]",
            "✗ Failed:",
            f"[red]{failed}[/]",
        )
        stats_table.add_row(
            "⟳ In Progress:",
            f"[yellow]{in_progress}[/]",
            "Workers:",
            f"{self.max_workers}",
        )

        # Add rate limit info if available
        if self.rate_limit is not None and self.rate_remaining is not None:
            stats_table.add_row(
                "Rate Limit:",
                f"[cyan]{self.rate_remaining}[/]/[white]{self.rate_limit}[/]",
                "",
                "",
            )

        # Workers section
        workers_table = Table(
            title="[bold]Active Workers[/]",
            show_header=True,
            header_style="bold magenta",
            border_style="dim",
            expand=False,
        )
        workers_table.add_column("Worker", style="cyan", width=8)
        workers_table.add_column("Status", style="white", width=12)
        workers_table.add_column(
            "Current Task", style="yellow", overflow="ellipsis", no_wrap=True, width=60
        )
        workers_table.add_column("Done", style="green", justify="right", width=6)
        workers_table.add_column("Fail", style="red", justify="right", width=6)

        for wid in sorted(self.workers.keys()):
            w = self.workers[wid]
            status = "[green]●[/] Working" if w.current_task else "[dim]○ Idle[/]"
            task_display = (
                w.current_task[:58] + "…"
                if w.current_task and len(w.current_task) > 58
                else (w.current_task or "-")
            )
            workers_table.add_row(
                f"#{wid}",
                status,
                task_display,
                str(w.tasks_completed),
                str(w.tasks_failed),
            )

        # Recent completions
        recent_text = Text()
        for task_id, success in self.recent_completed[-self.show_recent_completed :]:
            short_id = task_id[:70] + "…" if len(task_id) > 70 else task_id
            recent_text.append("  ")
            recent_text.append(
                "✓ " if success else "✗ ", style="bold green" if success else "bold red"
            )
            recent_text.append(f"{short_id}\n")

        components = [
            stats_table,
            Text(),
            progress,
            Text(),
            workers_table,
            Text(),
            Panel(
                recent_text
                if recent_text.plain
                else Text("  Waiting for tasks...", style="dim italic"),
                title="[bold]Recent Completions[/]",
                border_style="dim",
            ),
        ]

        group = Group(*components)

        return Panel(
            group,
            title=f"[bold blue]⚡ {self.task_name}[/]",
            border_style="blue",
        )

    def run(
        self,
        tasks: list[Any],
        worker_fn: Callable[[Any], TaskResult],
        task_id_fn: Callable[[Any], str],
        skipped: int = 0,
    ) -> list[TaskResult]:
        """Execute tasks in parallel with a live UI.

        Args:
            tasks: List of task arguments to pass to worker_fn
            worker_fn: Function that processes a single task and returns TaskResult
            task_id_fn: Function to extract a display ID from a task
            skipped: Number of tasks that were skipped before execution

        Returns:
            List of TaskResult objects

        """
        import time as time_module

        self.total_tasks = len(tasks)
        self.skipped_tasks = skipped
        self.workers = {i: WorkerStatus(worker_id=i) for i in range(self.max_workers)}
        self.completed_tasks = []
        self.failed_tasks = []
        self.recent_completed = []
        self.start_time = time_module.time()

        if self.total_tasks == 0:
            console.print(
                Panel(
                    "[dim]No tasks to process[/]",
                    title=f"[bold]{self.task_name}[/]",
                    border_style="dim",
                )
            )
            return []

        results: list[TaskResult] = []

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        )
        progress_task_id = progress.add_task(self.task_name, total=self.total_tasks)

        with (
            SuppressConsoleLogging(),
            Live(
                self.create_display(progress), refresh_per_second=4, console=console
            ) as live,
        ):
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Map futures to (task, worker_id)
                future_to_task: dict[Future, tuple[Any, int]] = {}
                available_workers = list(range(self.max_workers))
                pending_tasks = list(tasks)

                # Submit initial batch
                while available_workers and pending_tasks:
                    worker_id = available_workers.pop(0)
                    task = pending_tasks.pop(0)
                    task_id = task_id_fn(task)

                    self.workers[worker_id].current_task = task_id
                    future = executor.submit(worker_fn, task)
                    future_to_task[future] = (task, worker_id)

                live.update(self.create_display(progress))

                # Process as futures complete
                while future_to_task:
                    # Wait for at least one to complete
                    done_futures = []
                    for future in list(future_to_task.keys()):
                        if future.done():
                            done_futures.append(future)

                    if not done_futures:
                        # Small sleep to avoid busy waiting
                        time_module.sleep(0.05)
                        continue

                    for future in done_futures:
                        task, worker_id = future_to_task.pop(future)
                        task_id = task_id_fn(task)
                        try:
                            result = future.result()
                            if result.success:
                                results.append(result)
                                self.completed_tasks.append(result)
                                self.workers[worker_id].tasks_completed += 1
                                self.recent_completed.append((task_id, result.success))
                            else:
                                results.append(result)
                                self.failed_tasks.append(result)
                                self.workers[worker_id].tasks_failed += 1
                                self.recent_completed.append((task_id, result.success))
                                marker = result.failed_marker_path
                                marker.write_text(
                                    f"Task failed: {result.error}\n"
                                ) if marker else None

                        except Exception as e:
                            error_result = TaskResult(
                                task_id=task_id, success=False, error=str(e)
                            )
                            results.append(error_result)
                            self.failed_tasks.append(error_result)
                            self.workers[worker_id].tasks_failed += 1
                            self.recent_completed.append((task_id, False))
                            result = error_result  # Assign for progress check below
                            marker = result.failed_marker_path
                            marker.write_text(
                                f"Task failed: {result.error}\n"
                            ) if marker else None
                        # Update progress
                        progress.advance(progress_task_id)

                        # Mark worker as available
                        self.workers[worker_id].current_task = None
                        available_workers.append(worker_id)

                        # Submit next task if available and not rate limited
                        if pending_tasks and available_workers:
                            next_worker = available_workers.pop(0)
                            next_task = pending_tasks.pop(0)
                            next_task_id = task_id_fn(next_task)

                            self.workers[next_worker].current_task = next_task_id
                            next_future = executor.submit(worker_fn, next_task)
                            future_to_task[next_future] = (next_task, next_worker)

                    if done_futures:
                        live.update(self.create_display(progress))

        # Final summary with elapsed time
        elapsed = time_module.time() - self.start_time
        if elapsed >= 60:
            elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
        else:
            elapsed_str = f"{elapsed:.1f}s"

        console.print()
        console.print(
            Panel(
                f"[green]✓ Completed:[/] {len(self.completed_tasks)}  "
                f"[red]✗ Failed:[/] {len(self.failed_tasks)}  "
                f"[dim]Skipped:[/] {self.skipped_tasks}  "
                f"[cyan]⏱ Time:[/] {elapsed_str}",
                title=f"[bold]{self.task_name} Complete[/]",
                border_style="green" if not self.failed_tasks else "yellow",
            )
        )

        return results


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
    print(f"Using token: {github_token}")
    rate_info = gh.gh_get_rate_limit_info(github_token)
    if rate_info is not None:
        print(
            f"GitHub API Rate Limit: {rate_info['limit']}/{rate_info['remaining']} remaining (resets at {rate_info['reset_datetime']})"
        )
    else:
        print("Failed to fetch rate limit info from GitHub.")


@app.command()
def pull_requests(
    repo_url: str = typer.Argument(
        ..., help="The GitHub repository URL to fetch pull requests for"
    ),
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
    distro: str = typer.Option(
        "debian", help="The Linux distribution to process (e.g., 'debian' 'fedora')"
    ),
    releases: list[str] = typer.Option(
        ...,
        "--release",
        help="One or more distro releases to process (e.g., 'trixie', 'bookworm', '40'). Can repeat flag or use comma-separated.",
    ),
    cache: str = typer.Option("./cache", help="Cache directory (EV: CACHE_DIR)"),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force re-processing even if cache exists"
    ),
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

    # Suppress console logging for steps 1-4 (non-parallel steps)
    with SuppressConsoleLogging():
        # Step 1: Get and cache package data for each release
        with Status("[bold cyan]Step 1/6:[/] Fetching packages...", console=console):
            fetch_packages(distro=distro, releases=to_process, cache=cache_dir)
        console.print("[green]✓[/] Step 1/6: Fetched packages")

        # Step 2: Filter GitHub repos
        with Status(
            "[bold cyan]Step 2/6:[/] Filtering GitHub repos...", console=console
        ):
            filter_debian_github_repos(
                distro=distro, release=to_process, cache=cache_dir, force=force
            )
        console.print("[green]✓[/] Step 2/6: Filtered GitHub repos")

        # Step 3: Extract the version string and add upstream version columns
        with Status(
            "[bold cyan]Step 3/6:[/] Extracting upstream versions...", console=console
        ):
            extract_upstream_versions(
                distro=distro, release=to_process, cache=cache_dir, force=force
            )
        console.print("[green]✓[/] Step 3/6: Extracted upstream versions")

        # Step 4: Merge releases into a single DataFrame with all required columns
        with Status("[bold cyan]Step 4/6:[/] Merging releases...", console=console):
            merge_releases(
                distro=distro, releases=to_process, cache=cache_dir, force=force
            )
        console.print("[green]✓[/] Step 4/6: Merged releases")

    # Step 5: Clone all upstream GitHub repos (has its own UI)
    console.print("\n[bold cyan]Step 5/6:[/] Cloning repositories...")
    clone_upstream_repos(distro=distro, cache=cache_dir)

    # Step 6: Extract all commits into a single DataFrame (has its own UI)
    console.print("\n[bold cyan]Step 6/6:[/] Loading commits...")
    load_commits_into_dataframe(distro=distro, cache=cache_dir, force=force)

    # Step 7: Fetch GitHub metadata for all repos (has its own UI)
    console.print("\n[bold cyan]Step 7/7:[/] Fetching GitHub metadata...")
    all_github_metadata(distro=distro, cache=cache_dir, force=force)

    # Step 8: Fetch GitHub pull requests for all repos (has its own UI)
    console.print("\n[bold cyan]Step 8/8:[/] Fetching GitHub pull requests...")
    all_github_pull_requests(distro=distro, cache=cache_dir, force=force)

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

    if distro.lower() == "debian":
        for rel in releases:
            parquet_path = Path(cache_dir, f"{distro}_{rel}_all_packages.parquet")
            if parquet_path.exists():
                logger.info(f"Using cached {rel} packages from {parquet_path}")
                continue

            # Show status since this can take a while (large file download + parsing)
            with Status(
                f"[bold cyan]Fetching {rel} packages (this may take a minute)...[/]",
                console=console,
            ):
                logger.info(f"Fetching and caching {rel} packages to {parquet_path}")
                df: pd.DataFrame | None = deb.fetch_packages(rel)
                if df is None:
                    logger.error(f"Failed to fetch {rel} packages.")
                    console.print(f"[red]✗ Failed to fetch {rel} packages[/]")
                else:
                    df.to_parquet(parquet_path)
                    console.print(f"[green]✓ Fetched {len(df):,} {rel} packages[/]")
    else:
        logger.error(f"Distro '{distro}' is not supported for fetching packages.")


@dataset_app.command(rich_help_panel="Step 2: Filter Repos")
def filter_debian_github_repos(
    distro: str = typer.Argument(
        ..., help="The Linux distribution to process (e.g., 'debian' 'fedora')"
    ),
    release: list[str] = typer.Argument(
        ...,
        help="One or more distro releases to process (e.g., 'trixie', 'bookworm', '40'). Can repeat flag or use comma-separated.",
    ),
    cache: str = typer.Option("./cache", help="Cache directory"),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force re-processing even if cache exists"
    ),
):
    """Filter distro package DataFrames to only include GitHub repositories."""
    cache_dir = os.getenv("CACHE_DIR") or cache

    if distro.lower() == "debian":
        for rel in release:
            filtered_parquet_path = Path(
                cache_dir, f"{distro}_{rel}_filtered_packages.parquet"
            )
            if filtered_parquet_path.exists() and not force:
                logger.info(
                    f"Using cached filtered packages from {filtered_parquet_path}"
                )
                continue

            parquet_path = Path(cache_dir, f"{distro}_{rel}_all_packages.parquet")
            if not parquet_path.exists():
                logger.error(
                    f"Required parquet file {parquet_path} does not exist. Please run the 'fetch-packages' command first."
                )
                continue

            logger.info(f"Filtering GitHub repositories for Debian release '{rel}'")
            df: pd.DataFrame = pd.read_parquet(parquet_path)
            size_before = df.shape[0]
            filtered_df = deb.filter_github_repos(df)
            size_after = filtered_df.shape[0]
            logger.info(
                f"Dropped {size_before - size_after} packages due to non-GitHub '{rel}'."
            )
            filtered_df = deb.add_local_repo_cache_path_column(
                filtered_df, cache_dir=cache_dir
            )
            filtered_df.reset_index(drop=True, inplace=True)
            filtered_df.to_parquet(filtered_parquet_path)
            logger.info(
                f"Filtered GitHub repositories saved to {filtered_parquet_path}"
            )
    else:
        logger.error(
            f"Distro '{distro}' is not supported for filtering GitHub repositories."
        )


@dataset_app.command(rich_help_panel="Step 3: Extract Versions")
def extract_upstream_versions(
    distro: str = typer.Argument(
        ..., help="The Linux distribution to process (e.g., 'debian' 'fedora')"
    ),
    release: list[str] = typer.Argument(
        ...,
        help="One or more distro releases to process (e.g., 'trixie', 'bookworm', '40'). Can repeat flag or use comma-separated.",
    ),
    cache: str = typer.Option("./cache", help="Cache directory"),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force re-processing even if cache exists"
    ),
):
    """Extract upstream version strings from Debian package versions and add as a new column."""
    cache_dir = os.getenv("CACHE_DIR") or cache

    if distro.lower() == "debian":
        for rel in release:
            versions_parquet_path = Path(
                cache_dir, f"{distro}_{rel}_packages_with_upstream_versions.parquet"
            )
            if versions_parquet_path.exists() and not force:
                logger.info(
                    f"Using cached upstream versions from {versions_parquet_path}"
                )
                continue

            filtered_parquet_path = Path(
                cache_dir, f"{distro}_{rel}_filtered_packages.parquet"
            )
            if not filtered_parquet_path.exists():
                logger.error(
                    f"Required parquet file {filtered_parquet_path} does not exist. Please run the 'filter-debian-github-repos' command first."
                )
                continue

            logger.info(f"Extracting upstream versions for Debian release '{rel}'")
            df: pd.DataFrame = pd.read_parquet(filtered_parquet_path)
            version_column = f"{rel}_upstream_version"
            df_with_versions = deb.add_upstream_version_column(
                df, f"{rel}_version", new_column_name=version_column
            )
            drop_before = df_with_versions.shape[0]
            df_with_versions.dropna(subset=[version_column], inplace=True)
            drop_after = df_with_versions.shape[0]
            logger.info(
                f"Dropped {drop_before - drop_after} rows with missing upstream versions for release '{rel}'."
            )
            df_with_versions.reset_index(drop=True, inplace=True)
            df_with_versions.to_parquet(versions_parquet_path)
            logger.info(
                f"Upstream versions extracted and saved to {versions_parquet_path}"
            )
    else:
        logger.error(
            f"Distro '{distro}' is not supported for extracting upstream versions."
        )


@dataset_app.command(rich_help_panel="Step 4: Merge Releases")
def merge_releases(
    distro: str = typer.Argument(
        ..., help="The Linux distribution to process (e.g., 'debian' 'fedora')"
    ),
    releases: list[str] = typer.Argument(
        ...,
        help="One or more distro releases to merge (e.g., 'trixie', 'bookworm', '40'). Can repeat flag or use comma-separated.",
    ),
    cache: str = typer.Option("./cache", help="Cache directory"),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force re-processing even if cache exists"
    ),
):
    """Merge multiple release DataFrames into a single DataFrame with all required columns."""
    cache_dir = os.getenv("CACHE_DIR") or cache

    if distro.lower() == "debian":
        merged_parquet_path = Path(
            cache_dir, f"{distro}_merged_releases_packages.parquet"
        )
        if merged_parquet_path.exists() and not force:
            logger.info(f"Using cached merged releases from {merged_parquet_path}")
            return

        dfs = []
        for rel in releases:
            versions_parquet_path = Path(
                cache_dir, f"{distro}_{rel}_packages_with_upstream_versions.parquet"
            )
            if not versions_parquet_path.exists():
                logger.error(
                    f"Required parquet file {versions_parquet_path} does not exist. Please run the 'extract-upstream-versions' command first."
                )
                continue

            logger.info(
                f"Loading packages with upstream versions for Debian release '{rel}'"
            )
            df: pd.DataFrame = pd.read_parquet(versions_parquet_path)
            dfs.append(df)
        deb_merged_df, deb_dropped_after_merge = deb.merge_release_packages(dfs)
        logger.info(
            f"Merged releases {releases}. Dropped {deb_dropped_after_merge.shape[0]} rows that were not present in all releases."
        )
        deb_merged_df.reset_index(drop=True, inplace=True)
        deb_merged_df.to_parquet(merged_parquet_path)
        logger.info(f"Merged release packages saved to {merged_parquet_path}")
        deb_dropped_after_merge.to_parquet(
            Path(cache_dir, f"{distro}_dropped_after_merge.parquet")
        )
        logger.info(
            f"Dropped rows after merge saved to {Path(cache_dir, f'{distro}_dropped_after_merge.parquet')}"
        )

    else:
        logger.error(f"Distro '{distro}' is not supported for merging releases.")


@dataset_app.command(rich_help_panel="Step 5: Clone Repos")
def clone_upstream_repos(
    distro: str = typer.Argument(
        ..., help="The distro for (e.g., 'debian' 'fedora', etc.)"
    ),
    repos_cache: str = typer.Option(
        "./cache/repos", help="Cache directory for cloned repositories"
    ),
    cache: str = typer.Option("./cache", help="Cache directory"),
    max_workers: int = typer.Option(
        4, help="Maximum number of parallel clone processes (env: MAX_WORKERS)"
    ),
):
    """Clone all upstream GitHub repositories in the filtered package DataFrame."""
    cache_dir = os.getenv("CACHE_DIR") or cache
    repos_cache = os.getenv("REPOS_CACHE_DIR") or repos_cache
    max_workers = int(os.getenv("MAX_WORKERS", str(max_workers)))
    if distro.lower() == "debian":
        parquet_path = Path(cache_dir, f"{distro}_merged_releases_packages.parquet")
        if not parquet_path.exists():
            console.print(
                f"[red]Error:[/] Required parquet file {parquet_path} does not exist. Please run the 'merge-releases' command first."
            )
            return

        # Suppress logging during setup
        with SuppressConsoleLogging():
            df: pd.DataFrame = pd.read_parquet(parquet_path)
            repos_cache_path = pathlib.Path(repos_cache)

            vcs.ensure_dir(repos_cache_path)

            # Build list of repos to clone (skip already cloned)
            clone_tasks: list[tuple[str, str]] = []
            skipped = 0
            invalid = 0
            for _, row in df.iterrows():
                repo_url = str(row["upstream_repo_url"])
                target_dir = vcs.construct_repo_local_path(
                    repo_url, cache_dir=repos_cache_path, must_exist=False
                )
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
    else:
        console.print(
            f"[red]Error:[/] Distro '{distro}' is not supported for cloning repositories."
        )


@dataset_app.command(rich_help_panel="Step 6: Load Commits")
def load_commits_into_dataframe(
    distro: str = typer.Argument(
        ..., help="The Linux distribution to process (e.g., 'debian' 'fedora')"
    ),
    cache: str = typer.Option("./cache", help="Cache directory"),
    repo_cache: str = typer.Option(
        "./cache/repos", help="Cache directory for cloned repositories"
    ),
    max_workers: int = typer.Option(
        4, help="Maximum number of parallel worker processes (env: MAX_WORKERS)"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force re-processing even if cache exists"
    ),
):
    """Load all GitHub commits for the upstream repositories into a single DataFrame."""
    cache_dir = os.getenv("CACHE_DIR") or cache
    repo_cache_dir = os.getenv("REPOS_CACHE_DIR") or repo_cache
    checkpoint_dir = Path(cache_dir, "commit_checkpoints")
    max_workers = int(os.getenv("MAX_WORKERS", str(max_workers)))

    commits_parquet_path = Path(cache_dir, f"{distro}_all_upstream_commits.parquet")
    if commits_parquet_path.exists() and not force:
        console.print(f"[green]Using cached commits from {commits_parquet_path}[/]")
        return

    all_packages_parquet_path = Path(
        cache_dir, f"{distro}_merged_releases_packages.parquet"
    )
    if not all_packages_parquet_path.exists():
        console.print(
            f"[red]Error:[/] Required parquet file {all_packages_parquet_path} does not exist. Please run the 'merge-releases' and 'clone-upstream-repos' commands first."
        )
        return
    # Create checkpoint directory
    vcs.ensure_dir(checkpoint_dir)

    if force and checkpoint_dir.exists():
        console.print(f"[yellow]Removing existing checkpoint at {checkpoint_dir}[/]")
        for ck in checkpoint_dir.iterdir():
            if ck.name.endswith(".parquet"):
                ck.unlink()

    df: pd.DataFrame = pd.read_parquet(all_packages_parquet_path)

    # Build list of tasks (skip repos without local paths)
    tasks: list[tuple[str, str, str, str]] = []
    skipped = 0
    for _, row in df.iterrows():
        repo_url = str(row["upstream_repo_url"])
        local_repo_path = vcs.construct_repo_local_path(
            repo_url, cache_dir=Path(repo_cache_dir), must_exist=True
        )
        if local_repo_path is None or not local_repo_path.exists():
            skipped += 1
            continue
        source = str(row["source"])
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
    if checkpoint_dir.exists():
        try:
            console.print(
                f"[green]Loading checkpointed commits from {checkpoint_dir}[/]"
            )
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
            console.print(
                f"[yellow]Warning:[/] Failed to load checkpointed commits: {e}[/]"
            )

    # Collect successful DataFrames
    all_commits = [r.data for r in results if r.success and r.data is not None]

    if all_commits:
        console.print(f"[green]Loaded commits from {len(all_commits)} repositories.[/]")
        combined_commits_df = pd.concat(all_commits, ignore_index=True)
        commits_parquet_path = Path(cache_dir, f"{distro}_all_upstream_commits.parquet")
        combined_commits_df.to_parquet(commits_parquet_path)
        console.print(
            f"[green]Saved {len(combined_commits_df):,} commits to {commits_parquet_path}[/]"
        )
    else:
        console.print("[yellow]No commits were loaded from any repositories.[/]")


@dataset_app.command(rich_help_panel="Step 7: GitHub Metadata")
def all_github_metadata(
    distro: str = typer.Option(
        "debian", help="The Linux distribution to process (default: debian)"
    ),
    cache: str = typer.Option("./cache", help="Cache directory"),
    max_workers: int = typer.Option(
        4, help="Maximum number of parallel GitHub API workers (env: MAX_WORKERS)"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force re-processing even if cache exists"
    ),
):
    """Fetch GitHub repository metadata for all unique repos in the commits parquet file."""
    cache_dir = os.getenv("CACHE_DIR") or cache
    max_workers = int(os.getenv("MAX_WORKERS", str(max_workers)))
    all_packages_parquet_path = Path(
        cache_dir, f"{distro}_merged_releases_packages.parquet"
    )
    output_parquet_path = Path(cache_dir, f"{distro}_github_repo_metadata.parquet")
    checkpoint_dir = Path(cache_dir, "github_metadata_checkpoints")
    # Create checkpoint directory
    vcs.ensure_dir(checkpoint_dir)

    if force and checkpoint_dir.exists():
        console.print(
            f"[yellow]Removing existing GitHub metadata checkpoint at {checkpoint_dir}[/]"
        )
        for ck in checkpoint_dir.iterdir():
            if ck.name.endswith(".parquet"):
                ck.unlink()

    if output_parquet_path.exists() and not force:
        console.print(
            f"[green]Using cached GitHub metadata from {output_parquet_path}[/]"
        )
        return

    if not all_packages_parquet_path.exists():
        console.print(
            f"[red]Error:[/] Required parquet file {all_packages_parquet_path} does not exist. Please run the 'merge-releases' command first."
        )
        return

    df: pd.DataFrame = pd.read_parquet(all_packages_parquet_path)

    # Build list of tasks (skip repos without local paths)
    tasks: list[tuple[str, str, str]] = []
    skipped = 0
    for _, row in df.iterrows():
        repo_url = str(row["upstream_repo_url"])
        source = str(row["source"])
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
    if checkpoint_dir.exists():
        try:
            console.print(
                f"[green]Loading checkpointed commits from {checkpoint_dir}[/]"
            )
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
            console.print(
                f"[yellow]Warning:[/] Failed to load checkpointed commits: {e}[/]"
            )
    # Collect successful DataFrames
    all_metadata = [r.data for r in results if r.success and r.data is not None]

    if all_metadata:
        console.print(
            f"[green]Loaded metadata from {len(all_metadata)} repositories.[/]"
        )
        combined_metadata_df = pd.concat(all_metadata, ignore_index=True)
        metadata_parquet_path = Path(
            cache_dir, f"{distro}_all_upstream_metadata.parquet"
        )
        combined_metadata_df.to_parquet(metadata_parquet_path)
        console.print(
            f"[green]Saved {len(combined_metadata_df):,} metadata entries to {metadata_parquet_path}[/]"
        )
    else:
        console.print(
            "[yellow]No metadata entries were loaded from any repositories.[/]"
        )


@dataset_app.command(rich_help_panel="Step 8: GitHub Metadata")
def all_github_pull_requests(
    distro: str = typer.Option(
        "debian", help="The Linux distribution to process (default: debian)"
    ),
    cache: str = typer.Option("./cache", help="Cache directory"),
    max_workers: int = typer.Option(
        4, help="Maximum number of parallel GitHub API workers (env: MAX_WORKERS)"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force re-processing even if cache exists"
    ),
):
    """Fetch GitHub repository pull requests for all unique repos in the commits parquet file."""
    cache_dir = os.getenv("CACHE_DIR") or cache
    max_workers = int(os.getenv("MAX_WORKERS", str(max_workers)))
    all_packages_parquet_path = Path(
        cache_dir, f"{distro}_merged_releases_packages.parquet"
    )
    output_parquet_path = Path(cache_dir, f"{distro}_github_repo_pull_requests.parquet")
    checkpoint_dir = Path(cache_dir, "github_pr_checkpoints")
    # Create checkpoint directory
    vcs.ensure_dir(checkpoint_dir)

    if output_parquet_path.exists() and not force:
        console.print(
            f"[green]Using cached GitHub pull requests from {output_parquet_path}[/]"
        )
        return

    if not all_packages_parquet_path.exists():
        console.print(
            f"[red]Error:[/] Required parquet file {all_packages_parquet_path} does not exist. Please run the 'merge-releases' command first."
        )
        return

    if force and checkpoint_dir.exists():
        console.print(
            f"[yellow]Removing existing GitHub pull requests checkpoint at {checkpoint_dir}[/]"
        )
        for ck in checkpoint_dir.iterdir():
            if ck.name.endswith(".parquet"):
                ck.unlink()

    df: pd.DataFrame = pd.read_parquet(all_packages_parquet_path)

    # Build list of tasks (skip repos without local paths)
    tasks: list[tuple[str, str, str]] = []
    skipped = 0
    for _, row in df.iterrows():
        repo_url = str(row["upstream_repo_url"])
        source = str(row["source"])
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

    console.print(
        f"[cyan]Fetching GitHub pull requests for {len(tasks)} repositories...[/]"
    )
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
    if checkpoint_dir.exists():
        try:
            console.print(
                f"[green]Loading checkpointed commits from {checkpoint_dir}[/]"
            )
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
            console.print(
                f"[yellow]Warning:[/] Failed to load checkpointed commits: {e}[/]"
            )
    # Collect successful DataFrames
    all_metadata = [r.data for r in results if r.success and r.data is not None]

    if all_metadata:
        console.print(
            f"[green]Loaded pull requests from {len(all_metadata)} repositories.[/]"
        )
        combined_metadata_df = pd.concat(all_metadata, ignore_index=True)
        metadata_parquet_path = Path(
            cache_dir, f"{distro}_all_upstream_pull_requests.parquet"
        )
        combined_metadata_df.to_parquet(metadata_parquet_path)
        console.print(
            f"[green]Saved {len(combined_metadata_df):,} pull request entries to {metadata_parquet_path}[/]"
        )
    else:
        console.print(
            "[yellow]No pull request entries were loaded from any repositories.[/]"
        )


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
