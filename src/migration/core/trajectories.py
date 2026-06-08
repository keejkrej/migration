from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from migration.core.types import MIN_TRACK_LENGTH, TrajectoryRow


def build_trajectory_rows(tracks: np.ndarray, parent_map: dict[int, int]) -> list[TrajectoryRow]:
    if tracks.size == 0:
        return []
    if tracks.ndim != 2 or tracks.shape[1] < 4:
        raise ValueError("Expected 2D trajectory array with columns [track_id, frame, y, x]")

    rows = [
        TrajectoryRow(
            track_id=int(track[0]),
            parent_track_id=parent_map.get(int(track[0])),
            frame=int(track[1]),
            y=float(track[-2]),
            x=float(track[-1]),
        )
        for track in tracks
    ]
    return sorted(rows, key=lambda row: (row.track_id, row.frame))


def filter_short_trajectories(rows: list[TrajectoryRow], min_track_length: int = MIN_TRACK_LENGTH) -> list[TrajectoryRow]:
    if min_track_length <= 1 or not rows:
        return rows

    counts_by_track: dict[int, int] = {}
    for row in rows:
        counts_by_track[row.track_id] = counts_by_track.get(row.track_id, 0) + 1

    return [row for row in rows if counts_by_track[row.track_id] >= min_track_length]


def write_trajectories_csv(path: str | Path, rows: list[TrajectoryRow]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["track_id", "parent_track_id", "frame", "y", "x"])
        for row in rows:
            writer.writerow(
                [
                    row.track_id,
                    "" if row.parent_track_id is None else row.parent_track_id,
                    row.frame,
                    f"{row.y:.6f}",
                    f"{row.x:.6f}",
                ]
            )
    return output_path
