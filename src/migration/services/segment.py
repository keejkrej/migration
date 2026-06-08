from __future__ import annotations

from pathlib import Path

from migration.core.device import resolve_device
from migration.core.nd2 import load_nd2_timeseries, selection_channel_count
from migration.core.outputs import load_or_create_segmentation_masks
from migration.core.types import Nd2Selection, ProgressCallback, SegmentOutputs
from migration.utils.progress import emit_progress


def run_segment(
    nd2_path: str | Path,
    selection: Nd2Selection,
    output: str | Path,
    diameter: float | None,
    on_progress: ProgressCallback | None = None,
) -> SegmentOutputs:
    resolved_path = Path(nd2_path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"ND2 file does not exist: {resolved_path}")

    output_dir = Path(output).expanduser().resolve()
    device = resolve_device()
    scan, frames = load_nd2_timeseries(resolved_path, selection)
    channel_count = selection_channel_count(selection, scan)
    total_steps = len(scan.times) + 1

    emit_progress(
        on_progress,
        phase="start",
        done=0,
        total=total_steps,
        message=(
            f"Selected 1 position, {len(scan.times)} timepoints, "
            f"{channel_count} channel(s), 1 z-slice. Total steps: {total_steps}"
        ),
    )

    segmentation_path, _masks = load_or_create_segmentation_masks(
        frames,
        output_dir,
        selection,
        device,
        diameter,
        on_progress=on_progress,
        total_steps=total_steps,
    )
    emit_progress(
        on_progress,
        phase="finish",
        done=total_steps,
        total=total_steps,
        message=f"Wrote {segmentation_path}",
    )

    return SegmentOutputs(
        segmentation_path=segmentation_path,
        frame_count=len(scan.times),
    )
