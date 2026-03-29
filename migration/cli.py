from __future__ import annotations

import argparse
import sys
from pathlib import Path

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

from .pipeline import DEFAULT_MIN_TRACK_LENGTH, Nd2Selection, ProgressEvent, run_pipeline


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="migration-track",
        description="Segment and track one ND2 position/channel/z plane across all frames.",
    )
    parser.add_argument("nd2_path", type=Path, help="Path to the ND2 file.")
    parser.add_argument("--position", type=int, required=True, help="Zero-based ND2 position index.")
    parser.add_argument("--channel", type=int, required=True, help="Zero-based ND2 channel index.")
    parser.add_argument("--z", type=int, required=True, help="Zero-based ND2 z-slice index.")
    parser.add_argument(
        "--output",
        "--out-dir",
        dest="output",
        type=Path,
        help="Output directory for cached segmentations, the overlay PNG, and the trajectories CSV.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Execution device for Cellpose and Trackastra.",
    )
    parser.add_argument("--diameter", type=float, help="Optional Cellpose diameter hint in pixels.")
    parser.add_argument(
        "--min-track-length",
        type=int,
        default=DEFAULT_MIN_TRACK_LENGTH,
        help="Minimum number of frames a trajectory must span to be kept. Use 0 to disable filtering.",
    )
    parser.add_argument(
        "--tracking-mode",
        choices=["greedy", "greedy_nodiv"],
        default="greedy",
        help="Trackastra linking mode.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.diameter is not None and args.diameter <= 0:
        parser.error("--diameter must be greater than 0")
    if args.min_track_length < 0:
        parser.error("--min-track-length must be greater than or equal to 0")

    progress = RichProgressReporter()
    try:
        outputs = run_pipeline(
            nd2_path=args.nd2_path,
            selection=Nd2Selection(position=args.position, channel=args.channel, z=args.z),
            out_dir=args.output,
            device_name=args.device,
            diameter=args.diameter,
            min_track_length=args.min_track_length,
            tracking_mode=args.tracking_mode,
            on_progress=progress,
        )
    except Exception as exc:
        sys.stderr.write("\n")
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    sys.stderr.write("\n")
    print(f"Segmentation: {outputs.segmentation_path}")
    print(f"Overlay: {outputs.overlay_path}")
    print(f"Trajectories: {outputs.trajectories_path}")
    print(f"Rows: {outputs.row_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
