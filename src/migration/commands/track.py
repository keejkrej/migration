from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Literal

import typer

from migration.app import app
from migration.core.nd2 import parse_channel_option
from migration.core.types import DEFAULT_MIN_TRACK_LENGTH, Nd2Selection
from migration.services.track import run_track
from migration.utils.progress import RichProgressReporter

TrackingMode = Literal["greedy", "greedy_nodiv"]


@app.command()
def track(
    nd2_path: Annotated[Path, typer.Argument(help="Path to the ND2 file.")],
    position: Annotated[int, typer.Option(help="Zero-based ND2 position index.")],
    channel: Annotated[
        str,
        typer.Option(
            help=(
                "Zero-based ND2 channel index, comma-separated list like '0,1', or 'all'. "
                "Must match the segmentation run."
            ),
        ),
    ],
    z: Annotated[int, typer.Option(help="Zero-based ND2 z-slice index.")],
    output: Annotated[
        Path,
        typer.Option(help="Directory containing cached segmentations and trajectory outputs."),
    ],
    track_weights: Annotated[
        str | None,
        typer.Option(
            help="Comma-separated fusion weights for selected channels. Defaults to equal weights.",
        ),
    ] = None,
    min_track_length: Annotated[
        int,
        typer.Option(
            "--min-track-length",
            help="Minimum number of frames a trajectory must span to be kept. Use 0 to disable filtering.",
        ),
    ] = DEFAULT_MIN_TRACK_LENGTH,
    tracking_mode: Annotated[
        TrackingMode,
        typer.Option(help="Trackastra linking mode."),
    ] = "greedy",
    delta_t: Annotated[
        int,
        typer.Option(
            "--delta-t",
            help="Maximum frame gap allowed when linking tracks in Trackastra.",
        ),
    ] = 1,
) -> None:
    if min_track_length < 0:
        raise typer.BadParameter(
            "--min-track-length must be greater than or equal to 0",
            param_hint="--min-track-length",
        )
    if delta_t < 1:
        raise typer.BadParameter("--delta-t must be greater than or equal to 1", param_hint="--delta-t")

    try:
        selected_channel = parse_channel_option(channel)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--channel") from exc

    progress = RichProgressReporter()
    try:
        outputs = run_track(
            nd2_path=nd2_path,
            selection=Nd2Selection(position=position, channel=selected_channel, z=z),
            output=output,
            min_track_length=min_track_length,
            tracking_mode=tracking_mode,
            delta_t=delta_t,
            track_weights=track_weights,
            on_progress=progress,
        )
    except Exception as exc:
        sys.stderr.write("\n")
        print(f"Error: {exc}", file=sys.stderr)
        raise typer.Exit(code=1) from exc

    sys.stderr.write("\n")
    print(f"Overlay: {outputs.overlay_path}")
    print(f"Trajectories: {outputs.trajectories_path}")
    print(f"Rows: {outputs.row_count}")
