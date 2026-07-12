#!/usr/bin/env python3
"""Compare patterned vs unpatterned positions in a six-panel ROI figure.

Headless by default; pass --interactive to pick ROI and cells in a matplotlib UI.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

from migration.core.fusion import fuse_frames_for_tracking
from migration.core.nd2 import parse_channel_option
from migration.core.outputs import build_output_stem, load_segmentation_masks
from migration.core.overlay import normalize_frame_for_display
from migration.core.types import Nd2Selection, TrajectoryRow

CELL_COLORS = ("#4daf4a", "#80b1d3", "#fb9a99")  # green, light blue, pink
PANEL_LABEL_FONT = 18
COLUMN_TITLE_FONT = 14
CELL_LABEL_FONT = 14


@dataclass(frozen=True)
class PositionPanelData:
    position: int
    frame: np.ndarray
    mask0: np.ndarray
    tracks: dict[int, list[TrajectoryRow]]
    roi: tuple[int, int, int, int]
    selected_ids: list[int]


@dataclass(frozen=True)
class CellInsetCrop:
    frame_crop: np.ndarray
    mask_crop: np.ndarray
    width: int
    height: int
    color_index: int
    origin: tuple[int, int]  # (y0, x0) of the crop within the ROI-local frame


def load_run_tiff_module():
    script = Path(__file__).resolve().parent / "run_tiff_session.py"
    spec = importlib.util.spec_from_file_location("run_tiff_session", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_trajectory_rows(path: Path) -> list[TrajectoryRow]:
    rows: list[TrajectoryRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                TrajectoryRow(
                    track_id=int(row["track_id"]),
                    parent_track_id=int(row["parent_track_id"]) if row["parent_track_id"] else None,
                    frame=int(row["frame"]),
                    y=float(row["y"]),
                    x=float(row["x"]),
                )
            )
    return rows


def group_rows(rows: list[TrajectoryRow]) -> dict[int, list[TrajectoryRow]]:
    grouped: dict[int, list[TrajectoryRow]] = {}
    for row in rows:
        grouped.setdefault(row.track_id, []).append(row)
    for track_id in grouped:
        grouped[track_id].sort(key=lambda item: item.frame)
    return grouped


def track_centroid(points: list[TrajectoryRow], frame: int = 0) -> tuple[float, float] | None:
    frame_points = [point for point in points if point.frame == frame]
    if not frame_points:
        frame_points = points[:1]
    if not frame_points:
        return None
    point = frame_points[0]
    return point.y, point.x


def resolve_roi_shape(
    image_shape: tuple[int, int],
    roi_fraction: float,
    roi_size: int | None,
) -> tuple[int, int]:
    height, width = image_shape
    if roi_size is not None:
        return roi_size, roi_size
    return max(1, int(round(height * roi_fraction))), max(1, int(round(width * roi_fraction)))


def points_in_roi(points: list[TrajectoryRow], roi: tuple[int, int, int, int]) -> bool:
    y0, x0, height, width = roi
    return any(y0 <= point.y < y0 + height and x0 <= point.x < x0 + width for point in points)


def tracks_in_roi(
    tracks: dict[int, list[TrajectoryRow]],
    roi: tuple[int, int, int, int],
    min_track_length: int,
) -> list[int]:
    return [
        track_id
        for track_id, points in tracks.items()
        if len(points) >= min_track_length and points_in_roi(points, roi)
    ]


def resolve_inner_margins(
    roi: tuple[int, int, int, int],
    inset_padding: int,
    margin_fraction: float,
    margin_px: int | None,
) -> tuple[int, int]:
    _, _, height, width = roi
    if margin_px is not None:
        margin_y = margin_x = margin_px
    else:
        margin_y = int(round(height * margin_fraction))
        margin_x = int(round(width * margin_fraction))

    inset_margin = inset_padding + 16
    margin_y = max(margin_y, inset_margin)
    margin_x = max(margin_x, inset_margin)
    margin_y = min(margin_y, max(0, (height - 1) // 2))
    margin_x = min(margin_x, max(0, (width - 1) // 2))
    return margin_y, margin_x


def centroid_in_roi_interior(
    centroid: tuple[float, float],
    roi: tuple[int, int, int, int],
    margin_y: int,
    margin_x: int,
) -> bool:
    y0, x0, height, width = roi
    y, x = centroid
    return (
        y0 + margin_y <= y < y0 + height - margin_y
        and x0 + margin_x <= x < x0 + width - margin_x
    )


def ranked_roi_origins(
    tracks: dict[int, list[TrajectoryRow]],
    min_track_length: int,
    roi_shape: tuple[int, int],
    image_shape: tuple[int, int],
) -> list[tuple[int, int, int]]:
    height, width = image_shape
    roi_height, roi_width = roi_shape
    step_y = max(roi_height // 4, 100)
    step_x = max(roi_width // 4, 100)
    candidates: list[tuple[int, int, int]] = []
    for y0 in range(0, max(1, height - roi_height + 1), step_y):
        for x0 in range(0, max(1, width - roi_width + 1), step_x):
            roi = (y0, x0, roi_height, roi_width)
            count = sum(
                1
                for points in tracks.values()
                if len(points) >= min_track_length and points_in_roi(points, roi)
            )
            if count >= 3:
                candidates.append((count, y0, x0))
    candidates.sort(reverse=True)
    return candidates


def densest_roi_origin(
    tracks: dict[int, list[TrajectoryRow]],
    min_track_length: int,
    roi_shape: tuple[int, int],
    image_shape: tuple[int, int],
    rank: int = 0,
) -> tuple[int, int]:
    candidates = ranked_roi_origins(tracks, min_track_length, roi_shape, image_shape)
    if not candidates:
        raise ValueError("No ROI contains at least three long tracks")
    if rank < 0 or rank >= len(candidates):
        raise ValueError(f"ROI rank {rank} is out of range (found {len(candidates)} candidate ROIs)")
    _, y0, x0 = candidates[rank]
    return y0, x0


def _pick_three_from_pool(
    pool: list[tuple[int, tuple[float, float] | None]],
    tracks: dict[int, list[TrajectoryRow]],
    seed_index: int,
) -> list[int] | None:
    if seed_index >= len(pool):
        return None
    seed_id, seed_centroid = pool[seed_index]
    if seed_centroid is None:
        return None

    selected = [seed_id]
    selected_centroids = [seed_centroid]
    remaining = [item for index, item in enumerate(pool) if index != seed_index]
    while len(selected) < 3 and remaining:
        best_index = -1
        best_score = -1.0
        for index, (track_id, centroid) in enumerate(remaining):
            if track_id in selected or centroid is None:
                continue
            min_dist = min(
                np.hypot(centroid[0] - other[0], centroid[1] - other[1])
                for other in selected_centroids
                if other is not None
            )
            score = min_dist + 0.03 * len(tracks[track_id])
            if score > best_score:
                best_score = score
                best_index = index
        if best_index < 0:
            break
        track_id, centroid = remaining.pop(best_index)
        selected.append(track_id)
        selected_centroids.append(centroid)

    if len(selected) < 3:
        return None
    return selected[:3]


def select_three_tracks(
    tracks: dict[int, list[TrajectoryRow]],
    min_track_length: int,
    roi: tuple[int, int, int, int],
    cell_rank: int = 0,
    *,
    inset_padding: int = 24,
    inner_margin_fraction: float = 0.12,
    inner_margin_px: int | None = None,
) -> list[int]:
    margin_y, margin_x = resolve_inner_margins(roi, inset_padding, inner_margin_fraction, inner_margin_px)
    pool = [
        (track_id, centroid)
        for track_id, points in tracks.items()
        if len(points) >= min_track_length
        and (centroid := track_centroid(points)) is not None
        and centroid_in_roi_interior(centroid, roi, margin_y, margin_x)
    ]
    if len(pool) < 3:
        raise ValueError(
            "ROI interior does not contain three tracked cells "
            f"(inner margins: y={margin_y}px, x={margin_x}px; "
            "try --roi-inner-margin-fraction or --cell-rank)"
        )
    pool.sort(key=lambda item: (len(tracks[item[0]]), item[0]), reverse=True)

    valid_sets: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for seed_index in range(len(pool)):
        picked = _pick_three_from_pool(pool, tracks, seed_index)
        if picked is None:
            continue
        key = tuple(sorted(picked))
        if key in seen:
            continue
        seen.add(key)
        valid_sets.append(picked)

    if not valid_sets:
        raise ValueError("Unable to select three separated cells in ROI")
    if cell_rank < 0 or cell_rank >= len(valid_sets):
        raise ValueError(f"Cell rank {cell_rank} is out of range (found {len(valid_sets)} cell sets)")
    return valid_sets[cell_rank]


def mask_label_for_track(mask: np.ndarray, track_rows: list[TrajectoryRow], frame: int = 0) -> int:
    frame_rows = [row for row in track_rows if row.frame == frame]
    if not frame_rows:
        # Tracks can appear after frame 0; use their first observed point.
        frame_rows = track_rows[:1]
    if not frame_rows:
        return 0
    row = frame_rows[0]
    y = int(np.clip(round(row.y), 0, mask.shape[0] - 1))
    x = int(np.clip(round(row.x), 0, mask.shape[1] - 1))
    label = int(mask[y, x])
    if label > 0:
        return label

    # Track centroids can fall between mask pixels; recover the nearest label.
    radius = 8
    y0 = max(0, y - radius)
    y1 = min(mask.shape[0], y + radius + 1)
    x0 = max(0, x - radius)
    x1 = min(mask.shape[1], x + radius + 1)
    window = mask[y0:y1, x0:x1]
    labels = window[window > 0]
    if labels.size == 0:
        return 0
    unique, counts = np.unique(labels, return_counts=True)
    return int(unique[np.argmax(counts)])


def mask_for_track(mask: np.ndarray, track_rows: list[TrajectoryRow], frame: int = 0) -> np.ndarray:
    label = mask_label_for_track(mask, track_rows, frame)
    if label <= 0:
        return np.zeros(mask.shape, dtype=bool)
    return mask == label


def crop(array: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    y0, x0, height, width = roi
    return array[y0 : y0 + height, x0 : x0 + width]


def shift_points(points: list[TrajectoryRow], roi: tuple[int, int, int, int]) -> list[TrajectoryRow]:
    y0, x0, _, _ = roi
    return [
        TrajectoryRow(
            track_id=point.track_id,
            parent_track_id=point.parent_track_id,
            frame=point.frame,
            y=point.y - y0,
            x=point.x - x0,
        )
        for point in points
    ]


def parse_track_ids(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    ids = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if len(ids) != 3:
        raise ValueError(f"Expected exactly three track IDs, got {len(ids)}")
    return ids


def load_position_data(
    data_dir: Path,
    position: int,
    channel: str,
    z: int,
    roi_fraction: float,
    roi_size: int | None,
    min_track_length: int,
    roi_origin: tuple[int, int] | None,
    roi_rank: int,
    cell_rank: int,
    inset_padding: int,
    inner_margin_fraction: float,
    inner_margin_px: int | None,
    track_ids: list[int] | None = None,
) -> PositionPanelData:
    run_tiff = load_run_tiff_module()
    selection = Nd2Selection(position=position, channel=parse_channel_option(channel), z=z)
    _, frames = run_tiff.load_tiff_timeseries(data_dir, position, selection.channel, z)
    tracking_frames = fuse_frames_for_tracking(frames, None)
    masks = load_segmentation_masks(frames, data_dir, selection)
    stem = build_output_stem(data_dir, selection)
    tracks = group_rows(load_trajectory_rows(data_dir / f"{stem}_trajectories.csv"))

    roi_shape = resolve_roi_shape(tracking_frames[0].shape, roi_fraction, roi_size)
    if roi_origin is None:
        roi_origin = densest_roi_origin(
            tracks,
            min_track_length,
            roi_shape,
            tracking_frames[0].shape,
            rank=roi_rank,
        )

    roi_height, roi_width = roi_shape
    image_height, image_width = tracking_frames[0].shape
    y0, x0 = roi_origin
    roi = (
        int(np.clip(y0, 0, image_height - roi_height)),
        int(np.clip(x0, 0, image_width - roi_width)),
        roi_height,
        roi_width,
    )
    if track_ids is None:
        selected_ids = select_three_tracks(
            tracks,
            min_track_length,
            roi,
            cell_rank,
            inset_padding=inset_padding,
            inner_margin_fraction=inner_margin_fraction,
            inner_margin_px=inner_margin_px,
        )
    else:
        missing = [track_id for track_id in track_ids if track_id not in tracks]
        if missing:
            raise ValueError(f"Unknown track IDs for Pos{position}: {missing}")
        selected_ids = track_ids
    return PositionPanelData(
        position=position,
        frame=normalize_frame_for_display(tracking_frames[0]),
        mask0=masks[0],
        tracks=tracks,
        roi=roi,
        selected_ids=selected_ids,
    )


def configure_panel_axes(ax: plt.Axes, width: int, height: int, *, facecolor: str) -> None:
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor(facecolor)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")


def plot_trajectories(
    ax: plt.Axes,
    panel: PositionPanelData,
    display_shape: tuple[int, int],
    min_track_length: int,
) -> None:
    display_height, display_width = display_shape
    configure_panel_axes(ax, display_width, display_height, facecolor="white")

    track_ids = tracks_in_roi(panel.tracks, panel.roi, min_track_length)
    highlight_colors = {track_id: CELL_COLORS[index] for index, track_id in enumerate(panel.selected_ids)}
    highlight_ids = set(panel.selected_ids)

    for track_id in sorted(track_ids):
        if track_id in highlight_ids:
            continue
        points = shift_points(panel.tracks[track_id], panel.roi)
        xs = [point.x for point in points]
        ys = [point.y for point in points]
        ax.plot(xs, ys, color="black", linewidth=1.2, alpha=0.85)
        ax.scatter(xs[:1], ys[:1], color="black", s=10, alpha=0.85)

    for index, track_id in enumerate(panel.selected_ids):
        if track_id not in panel.tracks:
            continue
        points = shift_points(panel.tracks[track_id], panel.roi)
        xs = [point.x for point in points]
        ys = [point.y for point in points]
        color = highlight_colors[track_id]
        ax.plot(xs, ys, color=color, linewidth=2.0, alpha=0.95, zorder=5)
        ax.scatter(xs[:1], ys[:1], color=[color], s=18, alpha=0.95, zorder=6)


def draw_mask_overlay(
    ax: plt.Axes,
    panel: PositionPanelData,
    track_id: int,
    color: str,
    roi_extent: tuple[float, float, float, float],
) -> tuple[float, float] | None:
    cell_mask = mask_for_track(panel.mask0, panel.tracks[track_id])
    if not cell_mask.any():
        return None
    y0, x0, _, _ = panel.roi
    overlay = np.zeros((*cell_mask.shape, 4), dtype=np.float32)
    overlay[cell_mask] = np.array(plt.matplotlib.colors.to_rgb(color) + (0.42,))
    ax.imshow(crop(overlay, panel.roi), origin="upper", extent=roi_extent)
    ys, xs = np.where(crop(cell_mask, panel.roi))
    if ys.size == 0:
        return None
    return float(np.mean(ys)), float(np.mean(xs))


def draw_roi_panel(
    ax: plt.Axes,
    panel: PositionPanelData,
    display_shape: tuple[int, int],
    insets: list[CellInsetCrop] | None = None,
) -> None:
    y0, x0, height, width = panel.roi
    display_height, display_width = display_shape
    frame_roi = crop(panel.frame, panel.roi)
    configure_panel_axes(ax, display_width, display_height, facecolor="black")
    extent = (0, width, height, 0)
    ax.imshow(frame_roi, cmap="gray", vmin=0.0, vmax=1.0, origin="upper", extent=extent)

    for index, track_id in enumerate(panel.selected_ids):
        centroid = draw_mask_overlay(ax, panel, track_id, CELL_COLORS[index], extent)
        if centroid is None:
            point = track_centroid(panel.tracks[track_id])
            if point is None:
                continue
            cy, cx = point
            label_y, label_x = cy - y0, cx - x0
        else:
            label_y, label_x = centroid
        ax.text(
            label_x,
            label_y,
            str(index + 1),
            color="white",
            fontsize=CELL_LABEL_FONT,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "black", "alpha": 0.45, "edgecolor": "none"},
        )

    # Dashed outline of the (equal-size) crop shown for this cell in the C/D
    # inset panels, so the reader can see exactly which sub-region is zoomed.
    if insets is not None:
        for inset in insets:
            oy, ox = inset.origin
            ax.add_patch(
                Rectangle(
                    (ox, oy), inset.width, inset.height,
                    fill=False, linestyle="--", linewidth=1.4,
                    edgecolor=CELL_COLORS[inset.color_index], zorder=8,
                )
            )


def cell_center_and_natural_size(
    panel: PositionPanelData,
    track_id: int,
    *,
    fallback_size: int,
) -> tuple[tuple[float, float], int]:
    """ROI-local (cy, cx) center and natural (unpadded) bounding-box size of a cell."""
    y0, x0, roi_height, roi_width = panel.roi
    cell_mask = crop(mask_for_track(panel.mask0, panel.tracks[track_id]), panel.roi)
    if cell_mask.any():
        ys, xs = np.where(cell_mask)
        cy = (float(ys.min()) + float(ys.max())) / 2.0
        cx = (float(xs.min()) + float(xs.max())) / 2.0
        natural_size = int(max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1))
        return (cy, cx), natural_size

    centroid = track_centroid(panel.tracks[track_id])
    if centroid is not None:
        return (centroid[0] - y0, centroid[1] - x0), fallback_size
    return (roi_height / 2.0, roi_width / 2.0), fallback_size


def uniform_inset_crop_size(
    panels: list[PositionPanelData],
    inset_padding: int,
    *,
    fallback_size: int = 40,
) -> int:
    """Largest natural cell bounding-box size across all selected cells (both sides).

    Used so every C/D inset panel shows an equal-size, directly comparable
    crop rather than one tightly fit to each individual cell.
    """
    max_natural = 0
    for panel in panels:
        for track_id in panel.selected_ids:
            _, natural_size = cell_center_and_natural_size(panel, track_id, fallback_size=fallback_size)
            max_natural = max(max_natural, natural_size)
    return max_natural + 2 * inset_padding


def fixed_square_crop(
    frame_roi: np.ndarray,
    cell_mask: np.ndarray,
    center: tuple[float, float],
    crop_size: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    roi_height, roi_width = frame_roi.shape
    size = min(crop_size, roi_height, roi_width)
    half = size // 2
    cy, cx = center
    y0 = int(round(cy)) - half
    x0 = int(round(cx)) - half
    y0 = max(0, min(y0, roi_height - size))
    x0 = max(0, min(x0, roi_width - size))
    frame_crop = frame_roi[y0 : y0 + size, x0 : x0 + size]
    mask_crop = cell_mask[y0 : y0 + size, x0 : x0 + size]
    return frame_crop, mask_crop, y0, x0


def prepare_cell_inset_crops(panel: PositionPanelData, crop_size: int) -> list[CellInsetCrop]:
    frame_roi = crop(panel.frame, panel.roi)
    crops: list[CellInsetCrop] = []
    for index, track_id in enumerate(panel.selected_ids):
        cell_mask = crop(mask_for_track(panel.mask0, panel.tracks[track_id]), panel.roi)
        center, _ = cell_center_and_natural_size(panel, track_id, fallback_size=crop_size)
        frame_crop_arr, mask_crop_arr, oy, ox = fixed_square_crop(frame_roi, cell_mask, center, crop_size)
        crops.append(
            CellInsetCrop(
                frame_crop=frame_crop_arr,
                mask_crop=mask_crop_arr,
                width=frame_crop_arr.shape[1],
                height=frame_crop_arr.shape[0],
                color_index=index,
                origin=(oy, ox),
            )
        )
    return crops


def inset_row_height_ratio(
    left_crops: list[CellInsetCrop],
    right_crops: list[CellInsetCrop],
    *,
    n_insets: int = 3,
) -> float:
    """Middle-row height ratio so the C/D strip matches A/B/E/F panel width.

    Three square insets in a row need row height ≈ column_width / 3 (same
    display units as the square ROI panels above/below).  Using a taller
    middle row shrinks each inset to stay square, so the morphology row ends
    up narrower than the trajectory panels.
    """
    _ = (left_crops, right_crops)  # kept for call-site compatibility
    return 1.0 / n_insets


def add_cell_inset_axes(
    fig: plt.Figure,
    col_spec,
    crops: list[CellInsetCrop],
) -> list[plt.Axes]:
    width_ratios = [max(crop.width, 1) for crop in crops]
    inset_grid = col_spec[1].subgridspec(1, 3, width_ratios=width_ratios, wspace=0.04)
    axes = [fig.add_subplot(inset_grid[0, index]) for index in range(3)]
    draw_cell_insets(axes, crops)
    return axes


def draw_cell_insets(ax_row: list[plt.Axes], crops: list[CellInsetCrop]) -> None:
    for ax, crop in zip(ax_row, crops, strict=True):
        ax.imshow(crop.frame_crop, cmap="gray", vmin=0.0, vmax=1.0, origin="upper")
        overlay = np.zeros((*crop.mask_crop.shape, 4), dtype=np.float32)
        color = CELL_COLORS[crop.color_index]
        overlay[crop.mask_crop] = np.array(plt.matplotlib.colors.to_rgb(color) + (0.45,))
        ax.imshow(overlay, origin="upper")
        ax.set_xlim(0, crop.width)
        ax.set_ylim(crop.height, 0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(str(crop.color_index + 1), fontsize=CELL_LABEL_FONT, color="black", pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)


def render_comparison_figure(
    data_dir: Path,
    left_position: int,
    right_position: int,
    channel: str,
    z: int,
    output_path: Path,
    roi_fraction: float,
    roi_size: int | None,
    min_track_length: int,
    inset_padding: int,
    left_roi_origin: tuple[int, int] | None,
    right_roi_origin: tuple[int, int] | None,
    roi_rank: int,
    cell_rank: int,
    inner_margin_fraction: float,
    inner_margin_px: int | None,
    left_track_ids: list[int] | None = None,
    right_track_ids: list[int] | None = None,
    trajectory_min_track_length: int | None = None,
) -> Path:
    left = load_position_data(
        data_dir,
        left_position,
        channel,
        z,
        roi_fraction,
        roi_size,
        min_track_length,
        left_roi_origin,
        roi_rank,
        cell_rank,
        inset_padding,
        inner_margin_fraction,
        inner_margin_px,
        track_ids=left_track_ids,
    )
    right = load_position_data(
        data_dir,
        right_position,
        channel,
        z,
        roi_fraction,
        roi_size,
        min_track_length,
        right_roi_origin,
        roi_rank,
        cell_rank,
        inset_padding,
        inner_margin_fraction,
        inner_margin_px,
        track_ids=right_track_ids,
    )

    crop_size = uniform_inset_crop_size([left, right], inset_padding)
    left_crops = prepare_cell_inset_crops(left, crop_size)
    right_crops = prepare_cell_inset_crops(right, crop_size)

    _, _, left_height, left_width = left.roi
    _, _, right_height, right_width = right.roi
    display_shape = (max(left_height, right_height), max(left_width, right_width))
    roi_aspect = display_shape[1] / display_shape[0]

    # Size rows so each lettered panel (A–F) gets the same display width while
    # keeping native aspect ratios: square ROI rows share height = column width;
    # the three-square morphology row needs height = column_width / 3.
    panel_width_in = 4.25
    top_row_in = panel_width_in / roi_aspect
    mid_row_in = panel_width_in * inset_row_height_ratio(left_crops, right_crops)
    fig_w = 2 * panel_width_in + 1.0
    fig_h = 2 * top_row_in + mid_row_in + 1.1
    row_height_ratios = [top_row_in, mid_row_in, top_row_in]

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.12)
    left_col = gs[0].subgridspec(3, 1, height_ratios=row_height_ratios, hspace=0.22)
    right_col = gs[1].subgridspec(3, 1, height_ratios=row_height_ratios, hspace=0.22)

    ax_a = fig.add_subplot(left_col[0])
    ax_e = fig.add_subplot(left_col[2])
    ax_b = fig.add_subplot(right_col[0])
    ax_f = fig.add_subplot(right_col[2])

    draw_roi_panel(ax_a, left, display_shape, insets=left_crops)
    draw_roi_panel(ax_b, right, display_shape, insets=right_crops)

    inset_left = add_cell_inset_axes(fig, left_col, left_crops)
    inset_right = add_cell_inset_axes(fig, right_col, right_crops)

    # E/F use a separate, looser minimum-track-length filter than the rest of
    # the pipeline (A-D rely on the fixed min_track_length for reproducible
    # ROI/cell selection): a lower bar here surfaces moderately long-lived
    # tracks the stricter global threshold would otherwise drop, sharpening
    # the mobility contrast between the two conditions.
    traj_min_track_length = (
        trajectory_min_track_length if trajectory_min_track_length is not None else min_track_length
    )
    plot_trajectories(ax_e, left, display_shape, traj_min_track_length)
    plot_trajectories(ax_f, right, display_shape, traj_min_track_length)

    ax_a.text(-0.08, 1.03, "A", transform=ax_a.transAxes, fontsize=PANEL_LABEL_FONT, fontweight="bold", va="bottom")
    ax_b.text(-0.08, 1.03, "B", transform=ax_b.transAxes, fontsize=PANEL_LABEL_FONT, fontweight="bold", va="bottom")
    inset_left[0].text(-0.15, 1.18, "C", transform=inset_left[0].transAxes, fontsize=PANEL_LABEL_FONT, fontweight="bold", va="bottom")
    inset_right[0].text(-0.15, 1.18, "D", transform=inset_right[0].transAxes, fontsize=PANEL_LABEL_FONT, fontweight="bold", va="bottom")
    ax_e.text(-0.08, 1.03, "E", transform=ax_e.transAxes, fontsize=PANEL_LABEL_FONT, fontweight="bold", va="bottom")
    ax_f.text(-0.08, 1.03, "F", transform=ax_f.transAxes, fontsize=PANEL_LABEL_FONT, fontweight="bold", va="bottom")

    ax_a.text(0.5, 1.02, "patterned", transform=ax_a.transAxes, ha="center", va="bottom", fontsize=COLUMN_TITLE_FONT)
    ax_b.text(0.5, 1.02, "unpatterned", transform=ax_b.transAxes, ha="center", va="bottom", fontsize=COLUMN_TITLE_FONT)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".svg":
        fig.savefig(output_path, format="svg", facecolor="white")
    else:
        fig.savefig(output_path, dpi=200, facecolor="white")
        fig.savefig(output_path.with_suffix(".svg"), format="svg", facecolor="white")
    plt.close(fig)

    for label, panel in (("Left", left), ("Right", right)):
        y0, x0, height, width = panel.roi
        print(f"{label} Pos{panel.position}: ROI y={y0}, x={x0}, {height}x{width}, tracks={panel.selected_ids}")
    return output_path


def _render_from_selection_payload(
    payload: dict,
    *,
    data_dir: Path,
    left_position: int,
    right_position: int,
    channel: str,
    z: int,
    output_path: Path,
    roi_fraction: float,
    roi_size: int | None,
    min_track_length: int,
    inset_padding: int,
    roi_rank: int,
    cell_rank: int,
    inner_margin_fraction: float,
    inner_margin_px: int | None,
    trajectory_min_track_length: int | None = None,
) -> Path:
    left_side = payload["left"]
    right_side = payload["right"]
    left_track_ids = [int(track_id) for track_id in left_side.get("track_ids", [])]
    right_track_ids = [int(track_id) for track_id in right_side.get("track_ids", [])]
    if len(left_track_ids) != 3 or len(right_track_ids) != 3:
        raise ValueError("Pick exactly three cells on each side before saving")

    return render_comparison_figure(
        data_dir=data_dir,
        left_position=left_position,
        right_position=right_position,
        channel=channel,
        z=z,
        output_path=output_path,
        roi_fraction=roi_fraction,
        roi_size=roi_size,
        min_track_length=min_track_length,
        inset_padding=inset_padding,
        left_roi_origin=(left_side["roi_y"], left_side["roi_x"]),
        right_roi_origin=(right_side["roi_y"], right_side["roi_x"]),
        roi_rank=roi_rank,
        cell_rank=cell_rank,
        inner_margin_fraction=inner_margin_fraction,
        inner_margin_px=inner_margin_px,
        left_track_ids=left_track_ids,
        right_track_ids=right_track_ids,
        trajectory_min_track_length=trajectory_min_track_length,
    )


def run_interactive_mode(
    args: argparse.Namespace,
    data_dir: Path,
    output: Path,
    inset_padding: int,
) -> int:
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    from interactive_roi_picker import run_interactive_picker

    selection_path = args.selection_json or data_dir / "roi_cell_selection.json"

    def on_save(payload: dict) -> None:
        figure_path = _render_from_selection_payload(
            payload,
            data_dir=data_dir,
            left_position=args.left_position,
            right_position=args.right_position,
            channel=args.channel,
            z=args.z,
            output_path=output,
            roi_fraction=args.roi_fraction,
            roi_size=args.roi_size,
            min_track_length=args.min_track_length,
            inset_padding=inset_padding,
            roi_rank=args.roi_rank,
            cell_rank=args.cell_rank,
            inner_margin_fraction=args.roi_inner_margin_fraction,
            inner_margin_px=args.roi_inner_margin_px,
            trajectory_min_track_length=args.trajectory_min_track_length,
        )
        print(f"Wrote {figure_path}")

    run_interactive_picker(
        data_dir=data_dir,
        left_position=args.left_position,
        right_position=args.right_position,
        channel=args.channel,
        z=args.z,
        roi_fraction=args.roi_fraction,
        roi_size=args.roi_size,
        output_path=selection_path.expanduser().resolve(),
        on_save=on_save,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--left-position", type=int, default=26, help="Patterned position (left column)")
    parser.add_argument("--right-position", type=int, default=37, help="Unpatterned position (right column)")
    parser.add_argument("--channel", default="all")
    parser.add_argument("--z", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open an interactive ROI/cell picker instead of auto-selecting headlessly",
    )
    parser.add_argument("--roi-fraction", type=float, default=0.5)
    parser.add_argument("--roi-size", type=int, default=None)
    parser.add_argument(
        "--inset-padding",
        type=int,
        default=24,
        help="Pixels of padding around each selected cell in the inset panels",
    )
    parser.add_argument(
        "--inset-size",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--min-track-length", type=int, default=50)
    parser.add_argument(
        "--trajectory-min-track-length",
        type=int,
        default=None,
        help="Looser min-track-length override for the E/F trajectory panels only "
        "(defaults to --min-track-length if not given)",
    )
    parser.add_argument(
        "--roi-rank",
        type=int,
        default=1,
        help="Pick the Nth densest ROI tile (0=best, 1=second best, ...)",
    )
    parser.add_argument(
        "--cell-rank",
        type=int,
        default=2,
        help="Pick the Nth separated three-cell set inside the ROI (0=default, 1=alternate, ...)",
    )
    parser.add_argument(
        "--roi-inner-margin-fraction",
        type=float,
        default=0.12,
        help="Exclude cells whose centroids are within this fraction of the ROI border",
    )
    parser.add_argument(
        "--roi-inner-margin-px",
        type=int,
        default=None,
        help="Fixed inner ROI margin in pixels (overrides --roi-inner-margin-fraction)",
    )
    parser.add_argument("--left-roi-y", type=int, default=None)
    parser.add_argument("--left-roi-x", type=int, default=None)
    parser.add_argument("--right-roi-y", type=int, default=None)
    parser.add_argument("--right-roi-x", type=int, default=None)
    parser.add_argument("--left-track-ids", default=None, help="Comma-separated three track IDs for the left column")
    parser.add_argument("--right-track-ids", default=None, help="Comma-separated three track IDs for the right column")
    parser.add_argument(
        "--selection-json",
        type=Path,
        default=None,
        help="JSON from a prior interactive session (headless) or write path when using --interactive",
    )
    args = parser.parse_args(argv)

    data_dir = args.data_dir.expanduser().resolve()
    output = args.output or data_dir / "20260519_pos26_vs_pos37_comparison.png"
    inset_padding = args.inset_padding if args.inset_size is None else args.inset_size

    if args.interactive:
        try:
            return run_interactive_mode(args, data_dir, output, inset_padding)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    def parse_origin(y_arg: int | None, x_arg: int | None, side: str) -> tuple[int, int] | None:
        if y_arg is None and x_arg is None:
            return None
        if y_arg is None or x_arg is None:
            print(f"Error: --{side}-roi-y and --{side}-roi-x must be provided together.", file=sys.stderr)
            raise SystemExit(2)
        return y_arg, x_arg

    try:
        left_origin = parse_origin(args.left_roi_y, args.left_roi_x, "left")
        right_origin = parse_origin(args.right_roi_y, args.right_roi_x, "right")
        left_track_ids = parse_track_ids(args.left_track_ids)
        right_track_ids = parse_track_ids(args.right_track_ids)
        if args.selection_json is not None:
            selection = json.loads(args.selection_json.expanduser().read_text(encoding="utf-8"))
            left_side = selection["left"]
            right_side = selection["right"]
            left_origin = left_side["roi_y"], left_side["roi_x"]
            right_origin = right_side["roi_y"], right_side["roi_x"]
            if left_track_ids is None and left_side.get("track_ids"):
                left_track_ids = [int(track_id) for track_id in left_side["track_ids"]]
            if right_track_ids is None and right_side.get("track_ids"):
                right_track_ids = [int(track_id) for track_id in right_side["track_ids"]]
        path = render_comparison_figure(
            data_dir=data_dir,
            left_position=args.left_position,
            right_position=args.right_position,
            channel=args.channel,
            z=args.z,
            output_path=output,
            roi_fraction=args.roi_fraction,
            roi_size=args.roi_size,
            min_track_length=args.min_track_length,
            inset_padding=inset_padding,
            left_roi_origin=left_origin,
            right_roi_origin=right_origin,
            roi_rank=args.roi_rank,
            cell_rank=args.cell_rank,
            inner_margin_fraction=args.roi_inner_margin_fraction,
            inner_margin_px=args.roi_inner_margin_px,
            left_track_ids=left_track_ids,
            right_track_ids=right_track_ids,
            trajectory_min_track_length=args.trajectory_min_track_length,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())