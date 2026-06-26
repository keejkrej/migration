from __future__ import annotations

from pathlib import Path

from migration.core.device import resolve_device
from migration.core.fusion import fuse_frames_for_tracking
from migration.core.nd2 import load_nd2_timeseries, selection_channel_count
from migration.core.outputs import (
    build_output_stem,
    load_segmentation_masks,
)
from migration.core.overlay import render_trajectory_overlay
from migration.core.tracking import run_trackastra_tracking
from migration.core.trajectories import (
    build_trajectory_rows,
    filter_short_trajectories,
    write_trajectories_csv,
)
from migration.core.types import Nd2Selection, ProgressCallback, TrackOutputs
from migration.utils.progress import emit_progress


def run_track(
    nd2_path: str | Path,
    selection: Nd2Selection,
    output: str | Path,
    min_track_length: int,
    tracking_mode: str,
    delta_t: int,
    track_weights: str | None = None,
    on_progress: ProgressCallback | None = None,
) -> TrackOutputs:
    resolved_path = Path(nd2_path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"ND2 file does not exist: {resolved_path}")

    output_dir = Path(output).expanduser().resolve()
    device = resolve_device()
    scan, frames = load_nd2_timeseries(resolved_path, selection)
    tracking_frames = fuse_frames_for_tracking(frames, track_weights)
    output_stem = build_output_stem(resolved_path, selection)
    channel_count = selection_channel_count(selection, scan)
    total_steps = len(scan.times) + 2

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

    masks = load_segmentation_masks(
        frames,
        output_dir,
        selection,
        on_progress=on_progress,
        total_steps=total_steps,
    )

    tracks, parent_map = run_trackastra_tracking(
        tracking_frames,
        masks,
        device,
        tracking_mode,
        delta_t,
    )
    emit_progress(
        on_progress,
        phase="advance",
        done=len(scan.times) + 1,
        total=total_steps,
        message="Tracking trajectories",
    )
    rows = filter_short_trajectories(build_trajectory_rows(tracks, parent_map), min_track_length=min_track_length)

    overlay_path = render_trajectory_overlay(
        output_dir / f"{output_stem}_overlay.png",
        tracking_frames[0],
        rows,
        masks[0],
    )
    trajectories_path = write_trajectories_csv(output_dir / f"{output_stem}_trajectories.csv", rows)
    emit_progress(
        on_progress,
        phase="finish",
        done=total_steps,
        total=total_steps,
        message=f"Wrote {output_dir}",
    )

    return TrackOutputs(
        overlay_path=overlay_path,
        trajectories_path=trajectories_path,
        row_count=len(rows),
    )
