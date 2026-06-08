from __future__ import annotations

import sys

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from migration.core.types import ProgressCallback, ProgressEvent


def emit_progress(
    callback: ProgressCallback | None,
    *,
    phase: str,
    done: int,
    total: int,
    message: str,
) -> None:
    if callback is None:
        return
    callback(ProgressEvent(phase=phase, done=done, total=total, message=message))


class RichProgressReporter:
    def __init__(self) -> None:
        self._console = Console(stderr=True)
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self._console,
            transient=False,
        )
        self._task_id: TaskID | None = None
        self._last_done = 0

    def __call__(self, event: ProgressEvent) -> None:
        if event.phase == "start":
            self._last_done = 0
            self._progress.start()
            self._task_id = self._progress.add_task(event.message, total=event.total)
            return

        if self._task_id is None:
            self._progress.start()
            self._task_id = self._progress.add_task(event.message, total=event.total)

        increment = max(0, event.done - self._last_done)
        if increment:
            self._progress.update(self._task_id, advance=increment, description=event.message)
            self._last_done = event.done

        if event.phase == "finish":
            self._progress.update(self._task_id, completed=event.done, description=event.message)
            self._progress.stop()
            self._task_id = None
            sys.stdout.write(f"{event.message}\n")
