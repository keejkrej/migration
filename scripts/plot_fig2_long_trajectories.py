#!/usr/bin/env python3
"""Regenerate Figure 2 for the LISCA review with longer / more trajectories.

Standalone script: reads the trajectory CSVs and overlay PNGs directly, plus
the frame-0 segmentation masks for morphology contours. Produces a 2x3 panel
figure (rows = unpatterned/patterned, cols = trajectories/morphology/displacement)
matching the layout of the previous fig2.png, but with many more and longer
trajectories in panels A and D, and exports both PNG and SVG.

Run with:
    /home/jack/workspace/migration/.venv/bin/python \
        /home/jack/workspace/migration/scripts/plot_fig2_long_trajectories.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.gridspec import GridSpec

# --- Constants (matched to plot_roi_figure.py) ---------------------------
DATA_DIR = Path("/home/jack/data/lisca_review/fig2/20260519")
SELECTION = json.loads((DATA_DIR / "roi_cell_selection.json").read_text())
OUT_DIR = Path("/home/jack/workspace/lisca-paper/figs")

PANEL_LABEL_FONT = 20
CELL_LABEL_FONT = 14
COLUMN_TITLE_FONT = 16
TICK_LABEL_FONT = 14
AXIS_LABEL_FONT = 16
LEGEND_FONT = 12

CELL_COLORS = ("#4daf4a", "#80b1d3", "#fb9a99")  # green, light blue, pink

# Side definition: pos26 = unpatterned (top row), pos37 = patterned (bottom row)
SIDES = [
    {
        "name": "unpatterned",
        "position": 26,
        "sel_key": "left",
    },
    {
        "name": "patterned",
        "position": 37,
        "sel_key": "right",
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
    seg_dir = DATA_DIR / "segmentation" / f"Pos{position:02d}"
    # find frame 0 mask
    candidate = seg_dir / f"img_channelall_position{position:03d}_time000000000_z000_mask.tif"
    if not candidate.exists():
        # fall back to first sorted file
        candidate = sorted(seg_dir.glob("img_*_time000000000_*.tif"))[0]
    return np.array(Image.open(candidate))


def roi_for(side: dict) -> tuple[int, int, int, int]:
    s = SELECTION[side["sel_key"]]
    return (s["roi_y"], s["roi_x"], s["roi_height"], s["roi_width"])


def selected_track_ids(side: dict) -> list[int]:
    return list(SELECTION[side["sel_key"]]["track_ids"])


def crop(array: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    y0, x0, h, w = roi
    return array[y0 : y0 + h, x0 : x0 + w]


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
    return n_plotted


def draw_morphology_panel(
    ax: plt.Axes,
    overlay: np.ndarray,
    mask: np.ndarray,
    df: pd.DataFrame,
    roi: tuple[int, int, int, int],
    selected_ids: list[int],
) -> None:
    y0, x0, h, w = roi
    bg = crop(overlay, roi)
    height, width = bg.shape[:2]
    configure_panel_axes(ax, width, height, facecolor="black")
    extent = (0, width, height, 0)
    ax.imshow(bg, origin="upper", extent=extent, aspect="auto")

    for i, tid in enumerate(selected_ids):
        g = df[df.track_id == tid]
        if g.empty:
            continue
        label = mask_label_for_track(mask, g.sort_values("frame"))
        if label <= 0:
            # fall back to a small marker at the centroid
            cy = g.y.iloc[0] - y0
            cx = g.x.iloc[0] - x0
            ax.scatter([cx], [cy], color=CELL_COLORS[i], s=40, zorder=6)
            ax.text(
                cx, cy, str(i + 1), color="white", fontsize=CELL_LABEL_FONT,
                ha="center", va="center",
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "black", "alpha": 0.45, "edgecolor": "none"},
            )
            continue
        cell_mask = mask == label
        overlay_rgba = np.zeros((*cell_mask.shape, 4), dtype=np.float32)
        color = CELL_COLORS[i]
        overlay_rgba[cell_mask] = np.array(plt.matplotlib.colors.to_rgb(color) + (0.42,))
        ax.imshow(crop(overlay_rgba, roi), origin="upper", extent=extent, aspect="auto")
        # label at mask centroid within ROI
        ys, xs = np.where(crop(cell_mask, roi))
        if ys.size:
            ax.text(
                float(np.mean(xs)), float(np.mean(ys)), str(i + 1),
                color="white", fontsize=CELL_LABEL_FONT, ha="center", va="center",
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "black", "alpha": 0.45, "edgecolor": "none"},
            )


def draw_displacement_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    roi: tuple[int, int, int, int],
    min_frames: int,
    side_name: str,
) -> None:
    y0, x0, h, w = roi
    track_ids = tracks_in_roi(df, roi, min_frames)
    starts = []
    nets = []
    mags = []
    for tid in track_ids:
        g = df[df.track_id == tid].sort_values("frame")
        sy, sx = g.y.iloc[0], g.x.iloc[0]
        ey, ex = g.y.iloc[-1], g.x.iloc[-1]
        dy, dx = ey - sy, ex - sx
        mag = float(np.hypot(dy, dx))
        starts.append((sy - y0, sx - x0))
        nets.append((dy, dx))
        mags.append(mag)
    starts = np.array(starts)  # (N,2) y,x
    nets = np.array(nets)
    mags = np.array(mags)

    # left half: quiver of net displacement vectors over the ROI footprint
    configure_panel_axes(ax, w, h, facecolor="white")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    # draw faint dots at start positions
    ax.scatter(starts[:, 1], starts[:, 0], color="#666666", s=10, alpha=0.5, zorder=2)
    # quiver: U=dx, V=dy (in image coords y down), scale in px
    ax.quiver(
        starts[:, 1], starts[:, 0], nets[:, 1], nets[:, 0],
        angles="xy", scale_units="xy", scale=1.0,
        color="#e41a1c", alpha=0.85, width=0.004, zorder=4,
    )
    ax.set_title(
        f"{side_name} — net displacement\n"
        f"n={len(track_ids)} cells, median={np.median(mags):.1f} px, max={mags.max():.0f} px",
        fontsize=LEGEND_FONT,
    )
    ax.set_xlabel("x (px)", fontsize=AXIS_LABEL_FONT)
    ax.set_ylabel("y (px)", fontsize=AXIS_LABEL_FONT)


# --- Main figure ----------------------------------------------------------


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Preload per-side data
    side_data = []
    for side in SIDES:
        pos = side["position"]
        side_data.append({
            "side": side,
            "df": load_trajectories(pos),
            "overlay": load_overlay(pos),
            "mask0": load_mask0(pos),
            "roi": roi_for(side),
            "selected_ids": selected_track_ids(side),
        })

    # Display shape: use the larger ROI dimension so both columns share a scale
    display_h = max(sd["roi"][2] for sd in side_data)
    display_w = max(sd["roi"][3] for sd in side_data)

    fig = plt.figure(figsize=(13.0, 9.5), facecolor="white")
    gs = GridSpec(2, 3, figure=fig, hspace=0.28, wspace=0.18,
                  width_ratios=[1, 1, 1], height_ratios=[1, 1])

    axes = {}
    # row 0 = unpatterned (side_data[0]), row 1 = patterned (side_data[1])
    # col 0 = trajectories, col 1 = morphology, col 2 = displacement
    for r in range(2):
        for c in range(3):
            axes[(r, c)] = fig.add_subplot(gs[r, c])

    n_traj = []
    for r, sd in enumerate(side_data):
        side = sd["side"]
        # Trajectories panel (col 0)
        n = draw_trajectory_panel(
            axes[(r, 0)], sd["overlay"], sd["df"], sd["roi"], MIN_TRACK_FRAMES
        )
        n_traj.append((side["name"], n))
        # Morphology panel (col 1)
        draw_morphology_panel(
            axes[(r, 1)], sd["overlay"], sd["mask0"], sd["df"], sd["roi"], sd["selected_ids"]
        )
        # Displacement panel (col 2)
        draw_displacement_panel(
            axes[(r, 2)], sd["df"], sd["roi"], MIN_TRACK_FRAMES, side["name"]
        )
        # Column title on top row only for the trajectory panel
        if r == 0:
            axes[(r, 0)].set_title("trajectories", fontsize=COLUMN_TITLE_FONT, pad=4)
            axes[(r, 1)].set_title("morphology", fontsize=COLUMN_TITLE_FONT, pad=4)
            axes[(r, 2)].set_title("displacement", fontsize=COLUMN_TITLE_FONT, pad=4)
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
        ax = axes[(r, c)]
        ax.text(
            -0.08, 1.03, lab, transform=ax.transAxes,
            fontsize=PANEL_LABEL_FONT, fontweight="bold", va="bottom",
        )

    png_path = OUT_DIR / "fig2.png"
    svg_path = OUT_DIR / "fig2.svg"
    fig.savefig(png_path, dpi=200, facecolor="white", bbox_inches="tight")
    fig.savefig(svg_path, format="svg", facecolor="white", bbox_inches="tight")
    plt.close(fig)

    for name, n in n_traj:
        print(f"{name}: plotted {n} trajectories (>= {MIN_TRACK_FRAMES} frames)")
    print(f"Wrote {png_path} ({png_path.stat().st_size} bytes)")
    print(f"Wrote {svg_path} ({svg_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
