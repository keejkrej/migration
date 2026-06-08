from __future__ import annotations

from pathlib import Path

import numpy as np

from migration.core.types import TrajectoryRow


def normalize_frame_for_display(frame: np.ndarray) -> np.ndarray:
    image = np.asarray(frame, dtype=np.float32)
    if image.size == 0:
        return image
    low = float(np.percentile(image, 1))
    high = float(np.percentile(image, 99))
    if not np.isfinite(low):
        low = 0.0
    if not np.isfinite(high):
        high = low + 1.0
    if high <= low:
        high = low + 1.0
    return np.clip((image - low) / (high - low), 0.0, 1.0)


def normalize_track_lengths(track_lengths: dict[int, int]) -> dict[int, float]:
    if not track_lengths:
        return {}

    min_length = min(track_lengths.values())
    max_length = max(track_lengths.values())
    if min_length == max_length:
        return {track_id: 0.5 for track_id in track_lengths}

    scale = float(max_length - min_length)
    return {
        track_id: (length - min_length) / scale
        for track_id, length in track_lengths.items()
    }


def render_trajectory_overlay(path: str | Path, first_frame: np.ndarray, rows: list[TrajectoryRow]) -> Path:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    height, width = first_frame.shape
    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, frameon=False)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_facecolor((0.0, 0.0, 0.0, 0.0))
    fig.patch.set_alpha(0.0)

    tracks_by_id: dict[int, list[TrajectoryRow]] = {}
    for row in rows:
        tracks_by_id.setdefault(row.track_id, []).append(row)

    track_lengths = {track_id: len(points) for track_id, points in tracks_by_id.items()}
    color_values = normalize_track_lengths(track_lengths)
    cmap = plt.get_cmap("viridis")
    for track_id in sorted(tracks_by_id):
        points = sorted(tracks_by_id[track_id], key=lambda row: row.frame)
        xs = [point.x for point in points]
        ys = [point.y for point in points]
        color = cmap(color_values[track_id])
        ax.plot(xs, ys, color=color, linewidth=1.5, alpha=0.9)
        ax.scatter(xs[:1], ys[:1], color=[color], s=10, alpha=0.9)

    fig.savefig(output_path, dpi=dpi, transparent=True)
    plt.close(fig)
    return output_path
