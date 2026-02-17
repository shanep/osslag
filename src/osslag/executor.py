"""Parallel task executor with Rich CLI UI.

This module provides a generic parallel execution framework using
ProcessPoolExecutor with a live Rich dashboard for monitoring progress.
"""

from __future__ import annotations

import logging
import pathlib
import time as time_module
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Callable

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)
from rich.table import Table
from rich.text import Text


class SuppressConsoleLogging:
    """Context manager to temporarily suppress console logging output."""

    def __enter__(self):
        # Find and disable all console/stream handlers, saving their original levels
        self._disabled_handlers: list[tuple[logging.Handler, int]] = []
        for name in list(logging.Logger.manager.loggerDict.keys()) + ["", "root"]:
            log = logging.getLogger(name) if name else logging.getLogger()
            for handler in log.handlers[:]:
                if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
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

        # Console for output
        self.console = Console()

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
        workers_table.add_column("Current Task", style="yellow", overflow="ellipsis", no_wrap=True, width=60)
        workers_table.add_column("Done", style="green", justify="right", width=6)
        workers_table.add_column("Fail", style="red", justify="right", width=6)

        for wid in sorted(self.workers.keys()):
            w = self.workers[wid]
            status = "[green]●[/] Working" if w.current_task else "[dim]○ Idle[/]"
            task_display = (
                w.current_task[:58] + "…" if w.current_task and len(w.current_task) > 58 else (w.current_task or "-")
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
            recent_text.append("✓ " if success else "✗ ", style="bold green" if success else "bold red")
            recent_text.append(f"{short_id}\n")

        components = [
            stats_table,
            Text(),
            progress,
            Text(),
            workers_table,
            Text(),
            Panel(
                recent_text if recent_text.plain else Text("  Waiting for tasks...", style="dim italic"),
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
        self.total_tasks = len(tasks)
        self.skipped_tasks = skipped
        self.workers = {i: WorkerStatus(worker_id=i) for i in range(self.max_workers)}
        self.completed_tasks = []
        self.failed_tasks = []
        self.recent_completed = []
        self.start_time = time_module.time()

        if self.total_tasks == 0:
            self.console.print(
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
            Live(self.create_display(progress), refresh_per_second=4, console=self.console) as live,
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

                # Process as futures complete (blocks efficiently until one finishes)
                while future_to_task:
                    done_futures, _ = wait(
                        list(future_to_task.keys()),
                        timeout=1.0,
                        return_when=FIRST_COMPLETED,
                    )

                    if not done_futures:
                        # Timeout with no completions — just refresh the UI
                        live.update(self.create_display(progress))
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
                                marker.write_text(f"Task failed: {result.error}\n") if marker else None

                        except Exception as e:
                            error_result = TaskResult(task_id=task_id, success=False, error=str(e))
                            results.append(error_result)
                            self.failed_tasks.append(error_result)
                            self.workers[worker_id].tasks_failed += 1
                            self.recent_completed.append((task_id, False))
                            result = error_result  # Assign for progress check below
                            marker = result.failed_marker_path
                            marker.write_text(f"Task failed: {result.error}\n") if marker else None
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

        self.console.print()
        self.console.print(
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
