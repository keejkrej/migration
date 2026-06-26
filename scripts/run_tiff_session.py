#!/usr/bin/env python3
"""Run migration segment + track on mdat-converted TIFF session directories."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import tifffile

from migration.core.device import resolve_device
from migration.core.fusion import fuse_frames_for_tracking
from migration.core.nd2 import convert_frame_to_uint16, parse_channel_option, selection_channel_count
from migration.core.outputs import (
    build_output_stem,
    load_or_create_segmentation_masks,
    load_segmentation_masks,
)
from migration.core.overlay import render_trajectory_overlay
from migration.core.tracking import run_trackastra_tracking
from migration.core.trajectories import (
    build_trajectory_rows,
    filter_short_trajectories,
    write_trajectories_csv,
)
from migration.core.types import (
    DEFAULT_CELLPOSE_BATCH_SIZE,
    ChannelSelection,
    Nd2Scan,
    Nd2Selection,
)
from migration.utils.progress import RichProgressReporter, emit_progress

TIFF_RE = re.compile(r"img_channel(\d+)_position(\d+)_time(\d+)_z(\d+)\.tif$")


def discover_positions(data_dir: Path) -> list[int]:
    positions: list[int] = []
    for path in sorted(data_dir.iterdir()):
        if path.is_dir() and path.name.startswith("Pos"):
            positions.append(int(path.name.removeprefix("Pos")))
    if not positions:
        raise FileNotFoundError(f"No Pos* directories found in {data_dir}")
    return positions


def load_tiff_timeseries(
    data_dir: Path,
    position: int,
    channel: ChannelSelection,
    z: int,
) -> tuple[Nd2Scan, np.ndarray]:
    pos_dir = data_dir / f"Pos{position}"
    if not pos_dir.is_dir():
        raise FileNotFoundError(f"Position directory not found: {pos_dir}")

    index: dict[int, dict[int, Path]] = {}
    channels_found: set[int] = set()
    for path in pos_dir.glob("img_channel*_position*_time*_z*.tif"):
        match = TIFF_RE.match(path.name)
        if match is None:
            continue
        ch, pos, time_index, z_index = (int(match.group(i)) for i in range(1, 5))
        if pos != position or z_index != z:
            continue
        channels_found.add(ch)
        index.setdefault(time_index, {})[ch] = path

    if not index:
        raise FileNotFoundError(f"No TIFF frames found in {pos_dir}")

    if channel == "all":
        channel_indices = sorted(channels_found)
    elif isinstance(channel, tuple):
        channel_indices = list(channel)
    else:
        channel_indices = [channel]

    times = sorted(index)
    frames: list[np.ndarray] = []
    for time_index in times:
        planes: list[np.ndarray] = []
        for ch in channel_indices:
            frame_path = index[time_index].get(ch)
            if frame_path is None:
                raise FileNotFoundError(f"Missing channel {ch} at time {time_index} in {pos_dir}")
            planes.append(np.asarray(tifffile.imread(frame_path)))
        if len(planes) == 1:
            frame = planes[0]
        else:
            for plane in planes:
                if plane.ndim != 2:
                    raise ValueError(f"Expected 2D channel plane, got shape {plane.shape}")
            frame = np.stack(planes, axis=0)
        frames.append(convert_frame_to_uint16(frame))

    scan = Nd2Scan(
        positions=[position],
        channels=list(range(max(channels_found) + 1)),
        times=list(range(len(times))),
        z_slices=[z],
    )
    return scan, np.stack(frames, axis=0)


def position_is_complete(data_dir: Path, position: int, channel: str, z: int) -> bool:
    selection = Nd2Selection(position=position, channel=parse_channel_option(channel), z=z)
    stem = build_output_stem(data_dir, selection)
    trajectories = data_dir / f"{stem}_trajectories.csv"
    overlay = data_dir / f"{stem}_overlay.png"
    return trajectories.is_file() and overlay.is_file()


def run_position(
    data_dir: Path,
    position: int,
    channel: str,
    z: int,
    min_track_length: int,
    tracking_mode: str,
    delta_t: int,
    track_weights: str | None,
    cellpose_batch_size: int,
    diameter: float | None,
) -> None:
    selection = Nd2Selection(position=position, channel=parse_channel_option(channel), z=z)
    progress = RichProgressReporter()
    device = resolve_device()
    scan, frames = load_tiff_timeseries(data_dir, position, selection.channel, z)
    channel_count = selection_channel_count(selection, scan)
    total_steps = len(scan.times) + 2

    emit_progress(
        progress,
        phase="start",
        done=0,
        total=total_steps,
        message=(
            f"Position {position}: {len(scan.times)} timepoints, "
            f"{channel_count} channel(s), z={z}. Total steps: {total_steps}"
        ),
    )

    segmentation_path, _masks = load_or_create_segmentation_masks(
        frames,
        data_dir,
        selection,
        device,
        diameter,
        cellpose_batch_size=cellpose_batch_size,
        on_progress=progress,
        total_steps=len(scan.times) + 1,
    )
    print(f"Segmentation: {segmentation_path}")

    tracking_frames = fuse_frames_for_tracking(frames, track_weights)
    masks = load_segmentation_masks(
        frames,
        data_dir,
        selection,
        on_progress=progress,
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
        progress,
        phase="advance",
        done=len(scan.times) + 1,
        total=total_steps,
        message="Tracking trajectories",
    )
    rows = filter_short_trajectories(
        build_trajectory_rows(tracks, parent_map),
        min_track_length=min_track_length,
    )
    output_stem = build_output_stem(data_dir, selection)
    overlay_path = render_trajectory_overlay(
        data_dir / f"{output_stem}_overlay.png",
        tracking_frames[0],
        rows,
        masks[0],
    )
    trajectories_path = write_trajectories_csv(data_dir / f"{output_stem}_trajectories.csv", rows)
    emit_progress(
        progress,
        phase="finish",
        done=total_steps,
        total=total_steps,
        message=f"Wrote {data_dir}",
    )
    print(f"Overlay: {overlay_path}")
    print(f"Trajectories: {trajectories_path}")
    print(f"Rows: {len(rows)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="mdat-converted session directory")
    parser.add_argument("--channel", default="all")
    parser.add_argument("--z", type=int, default=0)
    parser.add_argument("--position", type=int, action="append", help="Position index (repeatable)")
    parser.add_argument("--min-track-length", type=int, default=0)
    parser.add_argument("--tracking-mode", default="greedy")
    parser.add_argument("--delta-t", type=int, default=1)
    parser.add_argument("--track-weights", default=None)
    parser.add_argument("--cellpose-batch-size", type=int, default=DEFAULT_CELLPOSE_BATCH_SIZE)
    parser.add_argument("--diameter", type=float, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run positions even when trajectories and overlay already exist",
    )
    args = parser.parse_args(argv)

    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        return 1

    positions = args.position if args.position else discover_positions(data_dir)
    failed = False
    skipped = 0
    for position in positions:
        print(f"=== Position {position} ===")
        if not args.force and position_is_complete(data_dir, position, args.channel, args.z):
            print(
                f"Skip position {position}: trajectories and overlay already exist "
                "(pass --force to re-run)"
            )
            skipped += 1
            print()
            continue
        try:
            run_position(
                data_dir=data_dir,
                position=position,
                channel=args.channel,
                z=args.z,
                min_track_length=args.min_track_length,
                tracking_mode=args.tracking_mode,
                delta_t=args.delta_t,
                track_weights=args.track_weights,
                cellpose_batch_size=args.cellpose_batch_size,
                diameter=args.diameter,
            )
        except Exception as exc:
            failed = True
            print(f"Error on position {position}: {exc}", file=sys.stderr)
        print()

    if skipped:
        print(f"Skipped {skipped} complete position(s).")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())