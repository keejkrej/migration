from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

from migration.app import app
from migration.core.types import Nd2Selection
from migration.services.segment import run_segment
from migration.utils.progress import RichProgressReporter


@app.command()
def segment(
    nd2_path: Annotated[Path, typer.Argument(help="Path to the ND2 file.")],
    position: Annotated[int, typer.Option(help="Zero-based ND2 position index.")],
    channel: Annotated[int, typer.Option(help="Zero-based ND2 channel index.")],
    z: Annotated[int, typer.Option(help="Zero-based ND2 z-slice index.")],
    output: Annotated[
        Path,
        typer.Option(help="Output directory for cached segmentation masks."),
    ],
    diameter: Annotated[
        float | None,
        typer.Option(help="Optional Cellpose diameter hint in pixels."),
    ] = None,
) -> None:
    if diameter is not None and diameter <= 0:
        raise typer.BadParameter("--diameter must be greater than 0", param_hint="--diameter")

    progress = RichProgressReporter()
    try:
        outputs = run_segment(
            nd2_path=nd2_path,
            selection=Nd2Selection(position=position, channel=channel, z=z),
            output=output,
            diameter=diameter,
            on_progress=progress,
        )
    except Exception as exc:
        sys.stderr.write("\n")
        print(f"Error: {exc}", file=sys.stderr)
        raise typer.Exit(code=1) from exc

    sys.stderr.write("\n")
    print(f"Segmentation: {outputs.segmentation_path}")
    print(f"Frames: {outputs.frame_count}")
