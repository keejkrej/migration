#!/usr/bin/env python3
"""Regenerate Figure 2 for the LISCA review with longer / more trajectories.

Standalone script: reads the trajectory CSVs and overlay PNGs directly, plus
the frame-0 segmentation masks for morphology contours. Produces a 2x3 panel
figure (rows = unpatterned/patterned, cols = trajectories/morphology/displacement)
matching the layout of the previous fig2.png, but with many more and longer
trajectories in panels A and D, and exports SVG for each configured cell-line
variant.

Run with:
    /home/jack/workspace/migration/.venv/bin/python \
        /home/jack/workspace/migration/scripts/plot_fig2_long_trajectories.py
"""

from __future__ import annotations

import json
from pathlib import Path

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from scipy import ndimage
from scipy.spatial import cKDTree

# --- Constants (matched to plot_roi_figure.py) ---------------------------
DATA_DIR = Path("/home/jack/data/lisca_review/fig2/20260519")
OUT_DIR = Path("/home/jack/workspace/lisca-paper/figs")

PANEL_LABEL_FONT = 20
CELL_LABEL_FONT = 14
COLUMN_TITLE_FONT = 16
TICK_LABEL_FONT = 14
AXIS_LABEL_FONT = 16
LEGEND_FONT = 12

CELL_COLORS = ("#4daf4a", "#80b1d3", "#fb9a99")  # green, light blue, pink

# Fixed square crop (px) shared by all 6 morphology sub-panels (C+D), sized to
# comfortably contain the largest of the 6 selected cell masks (~235 px) with
# padding.
MORPH_CROP_SIZE = 260

# Dashed island outline overlay (patterned side only): square side is a
# fraction of the empirically estimated island-to-island pitch.
PATTERN_SQUARE_COLOR = "yellow"
PATTERN_SQUARE_FRAC = 0.78

# Displacement panel E keeps full-field, temporally persistent tracks.
DISPLACEMENT_MIN_FRAMES = 110

# Panel F (patterned): pick an ROI and tracks by spatial path length, not duration.
SPATIAL_DISPLACEMENT_MIN_FRAMES = 50
SPATIAL_PATH_TOP_N = 10
SPATIAL_ROI_SCAN_STEP = 256
SPATIAL_ROI_MIN_TRACKS = 8

# Side definition: unpatterned on top row, patterned on bottom row.
@dataclass(frozen=True)
class Fig2Variant:
    name: str
    output_name: str
    selection: dict


FIG2_VARIANTS = (
    Fig2Variant(
        name="A549",
        output_name="fig2.svg",
        selection=json.loads((DATA_DIR / "roi_cell_selection.json").read_text()),
    ),
    Fig2Variant(
        name="MDA-231",
        output_name="fig2_mda231.svg",
        selection={
            "left": {
                "position": 2,
                "roi_y": 765,
                "roi_x": 1024,
                "roi_height": 1022,
                "roi_width": 1024,
                "track_ids": [55, 11, 12],
            },
            "right": {
                "position": 44,
                "roi_y": 1020,
                "roi_x": 1024,
                "roi_height": 1022,
                "roi_width": 1024,
                "track_ids": [97, 94, 34],
            },
        },
    ),
)


def sides_for(selection: dict) -> list[dict]:
    return [
        {
            "name": "unpatterned",
            "position": selection["right"]["position"],
            "sel_key": "right",
        },
        {
            "name": "patterned",
            "position": selection["left"]["position"],
            "sel_key": "left",
        },
    ]

MIN_TRACK_FRAMES = 50  # only plot tracks with at least this many frames


# --- Helpers --------------------------------------------------------------


def load_trajectories(position: int) -> pd.DataFrame:
    path = DATA_DIR / f"20260519_pos{position}_chall_z0_trajectories.csv"
    return pd.read_csv(path)


def load_overlay(position: int) -> np.ndarray:
    path = DATA_DIR / f"20260519_pos{position}_chall_z0_overlay.png"
    im = Image.open(path).convert("RGB")
    return np.array(im)


def load_mask0(position: int) -> np.ndarray:
    seg_dir = DATA_DIR / "segmentation" / f"Pos{position}"
    candidate = seg_dir / f"img_channelall_position{position:03d}_time000000000_z000_mask.tif"
    if not candidate.exists():
        masks = sorted(seg_dir.glob("img_*_mask.tif"))
        if not masks:
            raise FileNotFoundError(f"No segmentation masks found for Pos{position} in {seg_dir}")
        candidate = masks[0]
    return np.array(Image.open(candidate))


def roi_for(side: dict, selection: dict) -> tuple[int, int, int, int]:
    s = selection[side["sel_key"]]
    return (s["roi_y"], s["roi_x"], s["roi_height"], s["roi_width"])


def selected_track_ids(side: dict, selection: dict) -> list[int]:
    return list(selection[side["sel_key"]]["track_ids"])


def crop(array: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    y0, x0, h, w = roi
    return array[y0 : y0 + h, x0 : x0 + w]


def segmentation_centroids(mask: np.ndarray) -> np.ndarray:
    """Centroids (y, x) of every labelled cell in a frame-0 mask, full field."""
    labels = np.unique(mask)
    labels = labels[labels > 0]
    if labels.size == 0:
        return np.empty((0, 2))
    return np.array(ndimage.center_of_mass(mask > 0, mask, labels))


def estimate_island_pitch(centroids: np.ndarray) -> float:
    """Median nearest-neighbour spacing between cell centroids (px).

    Used as a proxy for the micropatterned-island grid pitch: on a
    single-cell array, adjacent occupied islands are ~one pitch apart.
    """
    tree = cKDTree(centroids)
    dist, _ = tree.query(centroids, k=2)
    return float(np.median(dist[:, 1]))


def draw_pattern_squares(
    ax: plt.Axes,
    centroids: np.ndarray,
    origin: tuple[float, float],
    width: int,
    height: int,
    side_px: float,
    color: str = PATTERN_SQUARE_COLOR,
) -> None:
    """Overlay dashed square outlines marking micropatterned islands.

    `centroids` are full-field (y, x) island/cell centroids; `origin` is the
    (y0, x0) offset of the panel's crop within the full image.
    """
    y0, x0 = origin
    half = side_px / 2.0
    for cy, cx in centroids:
        ly, lx = cy - y0, cx - x0
        # Require the whole square to lie inside the panel; a centroid just
        # outside the crop (e.g. an island row bordering the ROI edge) would
        # otherwise still pass a lenient "near the edge" check and get drawn,
        # but gets clipped by the axes limits down to a stray sliver of
        # dashes at the panel border.
        if half <= lx <= width - half and half <= ly <= height - half:
            ax.add_patch(
                Rectangle(
                    (lx - half, ly - half), side_px, side_px,
                    fill=False, linestyle="--", linewidth=1.4,
                    edgecolor=color, zorder=7,
                )
            )


def tracks_in_roi(
    df: pd.DataFrame, roi: tuple[int, int, int, int], min_frames: int
) -> list[int]:
    y0, x0, h, w = roi
    grouped = df.groupby("track_id")
    ids = []
    for tid, g in grouped:
        if len(g) < min_frames:
            continue
        inside = ((g.y >= y0) & (g.y < y0 + h) & (g.x >= x0) & (g.x < x0 + w)).any()
        if inside:
            ids.append(int(tid))
    return sorted(ids)


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


def mask_label_for_track(mask: np.ndarray, g: pd.DataFrame, frame: int = 0) -> int:
    sub = g[g.frame == frame]
    if sub.empty:
        sub = g.iloc[:1]
    y = int(np.clip(round(sub.y.iloc[0]), 0, mask.shape[0] - 1))
    x = int(np.clip(round(sub.x.iloc[0]), 0, mask.shape[1] - 1))
    label = int(mask[y, x])
    if label > 0:
        return label
    radius = 8
    y0 = max(0, y - radius)
    y1 = min(mask.shape[0], y + radius + 1)
    x0 = max(0, x - radius)
    x1 = min(mask.shape[1], x + radius + 1)
    window = mask[y0:y1, x0:x1]
    labels = window[window > 0]
    if labels.size == 0:
        return 0
    uniq, counts = np.unique(labels, return_counts=True)
    return int(uniq[np.argmax(counts)])


# --- Panel drawers --------------------------------------------------------


def draw_trajectory_panel(
    ax: plt.Axes,
    overlay: np.ndarray,
    df: pd.DataFrame,
    roi: tuple[int, int, int, int],
    min_frames: int,
    pattern_centroids: np.ndarray | None = None,
    pattern_side_px: float | None = None,
) -> int:
    y0, x0, h, w = roi
    bg = crop(overlay, roi)
    height, width = bg.shape[:2]
    configure_panel_axes(ax, width, height, facecolor="black")
    extent = (0, width, height, 0)
    ax.imshow(bg, origin="upper", extent=extent, aspect="auto")

    track_ids = tracks_in_roi(df, roi, min_frames)
    # distinct colors via a qualitative colormap
    cmap = matplotlib.colormaps.get_cmap("tab20").resampled(max(len(track_ids), 1))
    n_plotted = 0
    for idx, tid in enumerate(track_ids):
        g = df[df.track_id == tid].sort_values("frame")
        xs = g.x.values - x0
        ys = g.y.values - y0
        # only keep points inside the crop for clean drawing
        ax.plot(xs, ys, color=cmap(idx % cmap.N), linewidth=1.4, alpha=0.9)
        ax.scatter(xs[:1], ys[:1], color=cmap(idx % cmap.N), s=12, alpha=0.9, zorder=5)
        n_plotted += 1

    if pattern_centroids is not None and pattern_side_px is not None:
        draw_pattern_squares(ax, pattern_centroids, (y0, x0), width, height, pattern_side_px)
    return n_plotted


def fixed_window_for_centroid(
    cy: float, cx: float, crop_size: int, img_height: int, img_width: int
) -> tuple[int, int, int, int]:
    """Clamp a crop_size x crop_size window centered on (cy, cx) to image bounds."""
    half = crop_size // 2
    y0 = int(round(cy)) - half
    x0 = int(round(cx)) - half
    y0 = max(0, min(y0, img_height - crop_size))
    x0 = max(0, min(x0, img_width - crop_size))
    return y0, x0, crop_size, crop_size


def draw_morphology_subpanels(
    sub_axes: list[plt.Axes],
    overlay_full: np.ndarray,
    mask: np.ndarray,
    df: pd.DataFrame,
    selected_ids: list[int],
    crop_size: int,
    pattern_centroids: np.ndarray | None = None,
    pattern_side_px: float | None = None,
) -> None:
    img_h, img_w = mask.shape[:2]

    for i, (ax, tid) in enumerate(zip(sub_axes, selected_ids)):
        g = df[df.track_id == tid]
        color = CELL_COLORS[i]
        label = mask_label_for_track(mask, g.sort_values("frame")) if not g.empty else 0
        cell_mask = mask == label if label > 0 else np.zeros_like(mask, dtype=bool)

        if label > 0:
            ys, xs = np.where(cell_mask)
            cy, cx = float(np.mean(ys)), float(np.mean(xs))
        else:
            cy = float(g.y.iloc[0]) if not g.empty else img_h / 2
            cx = float(g.x.iloc[0]) if not g.empty else img_w / 2

        win = fixed_window_for_centroid(cy, cx, crop_size, img_h, img_w)
        y0, x0, h, w = win
        bg = crop(overlay_full, win)
        configure_panel_axes(ax, w, h, facecolor="black")
        extent = (0, w, h, 0)
        ax.imshow(bg, origin="upper", extent=extent, aspect="auto")

        if label > 0:
            overlay_rgba = np.zeros((*cell_mask.shape, 4), dtype=np.float32)
            overlay_rgba[cell_mask] = np.array(plt.matplotlib.colors.to_rgb(color) + (0.42,))
            ax.imshow(crop(overlay_rgba, win), origin="upper", extent=extent, aspect="auto")
            label_y, label_x = cy - y0, cx - x0
        else:
            label_y, label_x = cy - y0, cx - x0
            ax.scatter([label_x], [label_y], color=color, s=40, zorder=6)

        ax.text(
            label_x, label_y, str(i + 1), color="white", fontsize=CELL_LABEL_FONT,
            ha="center", va="center",
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "black", "alpha": 0.45, "edgecolor": "none"},
        )

        if pattern_centroids is not None and pattern_side_px is not None:
            draw_pattern_squares(ax, pattern_centroids, (y0, x0), w, h, pattern_side_px)


def full_field_track_ids(df: pd.DataFrame, min_frames: int) -> list[int]:
    grouped = df.groupby("track_id")
    return sorted(int(tid) for tid, g in grouped if len(g) >= min_frames)


def net_displacements(df: pd.DataFrame, track_ids: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts, nets, mags = [], [], []
    for tid in track_ids:
        g = df[df.track_id == tid].sort_values("frame")
        sy, sx = g.y.iloc[0], g.x.iloc[0]
        ey, ex = g.y.iloc[-1], g.x.iloc[-1]
        dy, dx = ey - sy, ex - sx
        starts.append((sy, sx))
        nets.append((dy, dx))
        mags.append(float(np.hypot(dy, dx)))
    return np.array(starts), np.array(nets), np.array(mags)


def track_path_length(track_df: pd.DataFrame) -> float:
    """Total distance travelled along a track (spatial extent, not net displacement)."""
    if len(track_df) < 2:
        return 0.0
    ys = track_df.y.values
    xs = track_df.x.values
    return float(np.hypot(np.diff(ys), np.diff(xs)).sum())


def select_spatial_displacement_roi(
    df: pd.DataFrame,
    image_shape: tuple[int, int],
    roi_shape: tuple[int, int],
    *,
    min_frames: int = SPATIAL_DISPLACEMENT_MIN_FRAMES,
    scan_step: int = SPATIAL_ROI_SCAN_STEP,
    min_tracks: int = SPATIAL_ROI_MIN_TRACKS,
) -> tuple[tuple[int, int, int, int], list[int]]:
    """Pick the ROI whose in-field tracks have the longest spatial trajectories."""
    img_h, img_w = image_shape
    roi_h, roi_w = roi_shape
    best_score = -1.0
    best_roi: tuple[int, int, int, int] | None = None
    best_track_ids: list[int] = []

    for y0 in range(0, img_h - roi_h + 1, scan_step):
        for x0 in range(0, img_w - roi_w + 1, scan_step):
            roi = (y0, x0, roi_h, roi_w)
            track_ids = tracks_in_roi(df, roi, min_frames)
            if len(track_ids) < min_tracks:
                continue
            path_lengths = [
                track_path_length(df[df.track_id == track_id].sort_values("frame"))
                for track_id in track_ids
            ]
            score = float(np.percentile(path_lengths, 90))
            if score > best_score:
                best_score = score
                best_roi = roi
                best_track_ids = track_ids

    if best_roi is None:
        raise ValueError("No ROI contains enough tracks for spatial displacement panel F")

    ranked = sorted(
        best_track_ids,
        key=lambda track_id: track_path_length(df[df.track_id == track_id].sort_values("frame")),
        reverse=True,
    )
    return best_roi, ranked[: min(SPATIAL_PATH_TOP_N, len(ranked))]


def draw_spatial_displacement_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    roi: tuple[int, int, int, int],
    track_ids: list[int],
) -> None:
    """Panel F: ROI zoom on spatially long trajectory paths (path length, not duration)."""
    y0, x0, h, w = roi
    configure_panel_axes(ax, w, h, facecolor="white")
    path_lengths = []
    for track_id in track_ids:
        track_df = df[df.track_id == track_id].sort_values("frame")
        path_len = track_path_length(track_df)
        path_lengths.append(path_len)
        xs = track_df.x.values - x0
        ys = track_df.y.values - y0
        ax.plot(xs, ys, color="#e41a1c", linewidth=1.5, alpha=0.9, zorder=4)
        ax.scatter(xs[:1], ys[:1], color="#666666", s=10, alpha=0.7, zorder=3)

    ax.set_title(
        f"n={len(track_ids)} cells, median path={np.median(path_lengths):.1f} px, "
        f"mean path={np.mean(path_lengths):.1f} px",
        fontsize=LEGEND_FONT,
        pad=2,
    )


def draw_displacement_panel(
    ax: plt.Axes,
    starts: np.ndarray,
    nets: np.ndarray,
    mags: np.ndarray,
    img_height: int,
    img_width: int,
) -> None:
    # Full field-of-view footprint (not just the 3-cell ROI) for a
    # statistically robust comparison of confined vs. free migration.
    configure_panel_axes(ax, img_width, img_height, facecolor="white")
    ax.set_xlim(0, img_width)
    ax.set_ylim(img_height, 0)
    # draw faint dots at start positions
    ax.scatter(starts[:, 1], starts[:, 0], color="#666666", s=8, alpha=0.5, zorder=2)
    # quiver: U=dx, V=dy (in image coords y down), scale in px
    ax.quiver(
        starts[:, 1], starts[:, 0], nets[:, 1], nets[:, 0],
        angles="xy", scale_units="xy", scale=1.0,
        color="#e41a1c", alpha=0.85, width=0.003, zorder=4,
    )
    # Compact single-line stats caption: the row side-label ("unpatterned" /
    # "patterned") already identifies the sample, and a single line keeps the
    # title from reaching up into the bold panel-label letter above the axes.
    ax.set_title(
        f"n={len(mags)} cells, median={np.median(mags):.1f} px, mean={mags.mean():.1f} px",
        fontsize=LEGEND_FONT, pad=2,
    )


# --- Main figure ----------------------------------------------------------


def render_variant(variant: Fig2Variant) -> None:
    selection = variant.selection
    sides = sides_for(selection)

    # Preload per-side data
    side_data = []
    for side in sides:
        pos = side["position"]
        side_data.append({
            "side": side,
            "df": load_trajectories(pos),
            "overlay": load_overlay(pos),
            "mask0": load_mask0(pos),
            "roi": roi_for(side, selection),
            "selected_ids": selected_track_ids(side, selection),
        })

    # Micropatterned-island grid outline (patterned side only): estimate the
    # pitch from full-field segmentation centroids.
    patterned_idx = next(i for i, sd in enumerate(side_data) if sd["side"]["name"] == "patterned")
    patterned_centroids = segmentation_centroids(side_data[patterned_idx]["mask0"])
    island_pitch = estimate_island_pitch(patterned_centroids)
    pattern_side_px = island_pitch * PATTERN_SQUARE_FRAC
    print(f"Estimated island pitch (patterned side, full field, n={len(patterned_centroids)} "
          f"cells): median NN spacing = {island_pitch:.1f} px -> square side = {pattern_side_px:.1f} px")

    roi = side_data[0]["roi"]
    _, _, roi_h, roi_w = roi
    roi_aspect = roi_w / roi_h

    # Size columns so A/B/E/F panels are wider than the C/D morphology strip while
    # keeping native aspect ratios. Trajectory and displacement columns get more
    # gridspec width; morphology stays at the width of three square insets.
    panel_width_in = 4.25
    row_height_in = panel_width_in / roi_aspect
    morph_col_ratio = 0.78
    traj_col_ratio = 1.28
    disp_col_ratio = 1.28

    fig_w = panel_width_in * (traj_col_ratio + morph_col_ratio + disp_col_ratio) + 1.6
    fig_h = 2 * row_height_in + 1.35

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs = GridSpec(
        2,
        3,
        figure=fig,
        hspace=0.28,
        wspace=0.18,
        width_ratios=[traj_col_ratio, morph_col_ratio, disp_col_ratio],
        height_ratios=[1, 1],
    )

    axes = {}
    morph_axes = {}
    # row 0 = unpatterned (side_data[0]), row 1 = patterned (side_data[1])
    # col 0 = trajectories, col 1 = morphology (3 sub-panels), col 2 = displacement
    for r in range(2):
        axes[(r, 0)] = fig.add_subplot(gs[r, 0])
        morph_gs = gs[r, 1].subgridspec(1, 3, wspace=0.04)
        morph_axes[r] = [fig.add_subplot(morph_gs[0, i]) for i in range(3)]
        axes[(r, 2)] = fig.add_subplot(gs[r, 2])

    n_traj = []
    disp_stats = {}
    for r, sd in enumerate(side_data):
        side = sd["side"]
        is_patterned = r == patterned_idx
        squares = (patterned_centroids, pattern_side_px) if is_patterned else (None, None)

        # Trajectories panel (col 0)
        n = draw_trajectory_panel(
            axes[(r, 0)], sd["overlay"], sd["df"], sd["roi"], MIN_TRACK_FRAMES,
            pattern_centroids=squares[0], pattern_side_px=squares[1],
        )
        n_traj.append((side["name"], n))

        # Morphology sub-panels (col 1): 3 fixed-size, centroid-centered crops
        draw_morphology_subpanels(
            morph_axes[r], sd["overlay"], sd["mask0"], sd["df"], sd["selected_ids"],
            MORPH_CROP_SIZE, pattern_centroids=squares[0], pattern_side_px=squares[1],
        )

        # Displacement panels: E = full-field temporal persistence; F = patterned
        # ROI chosen for spatially long trajectory paths.
        img_h, img_w = sd["overlay"].shape[:2]
        if is_patterned:
            disp_roi, spatial_track_ids = select_spatial_displacement_roi(
                sd["df"],
                (img_h, img_w),
                sd["roi"][2:4],
            )
            y0, x0, _, _ = disp_roi
            print(
                f"{variant.name} panel F ROI y={y0}, x={x0}, "
                f"tracks={len(spatial_track_ids)} (top spatial path length)"
            )
            draw_spatial_displacement_panel(axes[(r, 2)], sd["df"], disp_roi, spatial_track_ids)
            disp_stats[side["name"]] = np.array([
                track_path_length(sd["df"][sd["df"].track_id == track_id].sort_values("frame"))
                for track_id in spatial_track_ids
            ])
        else:
            track_ids = full_field_track_ids(sd["df"], DISPLACEMENT_MIN_FRAMES)
            starts, nets, mags = net_displacements(sd["df"], track_ids)
            draw_displacement_panel(axes[(r, 2)], starts, nets, mags, img_h, img_w)
            disp_stats[side["name"]] = mags

        # Column title on top row only
        if r == 0:
            axes[(r, 0)].set_title("trajectories", fontsize=COLUMN_TITLE_FONT, pad=4)
            morph_axes[r][1].set_title("morphology", fontsize=COLUMN_TITLE_FONT, pad=4)
            # E already carries its own compact stats title (set inside
            # draw_displacement_panel); stack the column header further above
            # the axes (offset in points, not axes-fraction, so it scales
            # with font metrics rather than the small displacement axes'
            # physical size) instead of overwriting that title.
            axes[(r, 2)].annotate(
                "displacement", xy=(0.5, 1.0), xytext=(0, 34),
                xycoords="axes fraction", textcoords="offset points",
                fontsize=COLUMN_TITLE_FONT, ha="center", va="bottom",
            )
        # side label on left edge
        axes[(r, 0)].text(
            -0.10, 0.5, side["name"], transform=axes[(r, 0)].transAxes,
            ha="right", va="center", fontsize=COLUMN_TITLE_FONT, rotation=90,
        )

    # Panel labels A-F, column-major so they follow the caption order
    # (1=unpat-traj, 2=pat-traj, 3=unpat-morph, 4=pat-morph, 5=unpat-disp, 6=pat-disp)
    label_map = {
        (0, 0): "A",  # unpatterned trajectories
        (1, 0): "B",  # patterned trajectories
        (0, 1): "C",  # unpatterned morphology
        (1, 1): "D",  # patterned morphology
        (0, 2): "E",  # unpatterned displacement
        (1, 2): "F",  # patterned displacement
    }
    for (r, c), lab in label_map.items():
        if c == 1:
            ax = morph_axes[r][0]
        else:
            ax = axes[(r, c)]
        if c == 2:
            # E/F carry their own compact stats title right above the axes;
            # push the label up by a point-based offset (scales with font
            # metrics, unlike an axes-fraction offset on these smaller axes)
            # so it clears that title instead of colliding with it.
            ax.annotate(
                lab, xy=(-0.08, 1.0), xytext=(0, 22),
                xycoords="axes fraction", textcoords="offset points",
                fontsize=PANEL_LABEL_FONT, fontweight="bold", va="bottom",
            )
        else:
            ax.text(
                -0.08, 1.03, lab, transform=ax.transAxes,
                fontsize=PANEL_LABEL_FONT, fontweight="bold", va="bottom",
            )

    fig.supxlabel("x (px)", fontsize=AXIS_LABEL_FONT, y=0.02)
    fig.supylabel("y (px)", fontsize=AXIS_LABEL_FONT, x=0.02)

    svg_path = OUT_DIR / variant.output_name
    fig.savefig(svg_path, format="svg", facecolor="white", bbox_inches="tight")
    plt.close(fig)

    for name, n in n_traj:
        print(f"{variant.name} {name}: plotted {n} trajectories (>= {MIN_TRACK_FRAMES} frames)")

    unpatterned_mags = disp_stats.get("unpatterned")
    patterned_paths = disp_stats.get("patterned")
    if unpatterned_mags is not None and patterned_paths is not None:
        print(
            f"{variant.name} panel E (full field, net displacement): "
            f"n={len(unpatterned_mags)} median={np.median(unpatterned_mags):.2f} px "
            f"mean={unpatterned_mags.mean():.2f} px"
        )
        print(
            f"{variant.name} panel F (spatial ROI, path length): "
            f"n={len(patterned_paths)} median={np.median(patterned_paths):.2f} px "
            f"mean={patterned_paths.mean():.2f} px"
        )

    print(f"Wrote {svg_path} ({svg_path.stat().st_size} bytes)")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for variant in FIG2_VARIANTS:
        render_variant(variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
