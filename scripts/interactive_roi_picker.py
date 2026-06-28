#!/usr/bin/env python3
"""Interactively pick ROI squares and three cells for the comparison figure."""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, RadioButtons

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from migration.core.fusion import fuse_frames_for_tracking
from migration.core.nd2 import parse_channel_option
from migration.core.outputs import build_output_stem, load_segmentation_masks
from migration.core.overlay import normalize_frame_for_display, normalize_track_lengths
from migration.core.types import Nd2Selection, TrajectoryRow

from plot_roi_figure import (
    CELL_COLORS,
    group_rows,
    load_run_tiff_module,
    load_trajectory_rows,
    mask_label_for_track,
    resolve_roi_shape,
    track_centroid,
)

PICK_RADIUS_PX = 30


@dataclass
class PanelState:
    position: int
    frame: np.ndarray
    mask0: np.ndarray
    tracks: dict[int, list[TrajectoryRow]]
    label_to_track: dict[int, int]
    roi_height: int
    roi_width: int
    roi_y0: int
    roi_x0: int
    selected_ids: deque[int] = field(default_factory=lambda: deque(maxlen=3))
    rect: Rectangle | None = None
    marker_artists: list = field(default_factory=list)
    dragging: bool = False
    drag_offset_y: float = 0.0
    drag_offset_x: float = 0.0


def build_label_to_track(mask: np.ndarray, tracks: dict[int, list[TrajectoryRow]]) -> dict[int, int]:
    best: dict[int, tuple[int, int]] = {}
    for track_id, rows in tracks.items():
        label = mask_label_for_track(mask, rows)
        if label <= 0:
            continue
        length = len(rows)
        current = best.get(label)
        if current is None or length > current[0]:
            best[label] = (length, track_id)
    return {label: track_id for label, (_, track_id) in best.items()}


def load_panel_state(
    data_dir: Path,
    position: int,
    channel: str,
    z: int,
    roi_fraction: float,
    roi_size: int | None,
) -> PanelState:
    run_tiff = load_run_tiff_module()
    selection = Nd2Selection(position=position, channel=parse_channel_option(channel), z=z)
    _, frames = run_tiff.load_tiff_timeseries(data_dir, position, selection.channel, z)
    tracking_frames = fuse_frames_for_tracking(frames, None)
    masks = load_segmentation_masks(frames, data_dir, selection)
    stem = build_output_stem(data_dir, selection)
    tracks = group_rows(load_trajectory_rows(data_dir / f"{stem}_trajectories.csv"))

    image = normalize_frame_for_display(tracking_frames[0])
    roi_height, roi_width = resolve_roi_shape(image.shape, roi_fraction, roi_size)
    frame_height, frame_width = image.shape
    roi_y0 = max(0, (frame_height - roi_height) // 2)
    roi_x0 = max(0, (frame_width - roi_width) // 2)

    return PanelState(
        position=position,
        frame=image,
        mask0=masks[0],
        tracks=tracks,
        label_to_track=build_label_to_track(masks[0], tracks),
        roi_height=roi_height,
        roi_width=roi_width,
        roi_y0=roi_y0,
        roi_x0=roi_x0,
    )


def pick_track_at_click(panel: PanelState, y: float, x: float) -> int | None:
    yi = int(np.clip(round(y), 0, panel.mask0.shape[0] - 1))
    xi = int(np.clip(round(x), 0, panel.mask0.shape[1] - 1))
    label = int(panel.mask0[yi, xi])
    if label > 0 and label in panel.label_to_track:
        return panel.label_to_track[label]

    best_id: int | None = None
    best_dist = PICK_RADIUS_PX
    for track_id, rows in panel.tracks.items():
        centroid = track_centroid(rows)
        if centroid is None:
            continue
        dist = float(np.hypot(centroid[0] - y, centroid[1] - x))
        if dist < best_dist:
            best_dist = dist
            best_id = track_id
    return best_id


def clamp_roi_origin(panel: PanelState, y0: float, x0: float) -> tuple[int, int]:
    frame_height, frame_width = panel.frame.shape
    roi_y0 = int(np.clip(round(y0), 0, frame_height - panel.roi_height))
    roi_x0 = int(np.clip(round(x0), 0, frame_width - panel.roi_width))
    return roi_y0, roi_x0


def panel_roi_tuple(panel: PanelState) -> tuple[int, int, int, int]:
    return panel.roi_y0, panel.roi_x0, panel.roi_height, panel.roi_width


def draw_panel_background(ax: plt.Axes, panel: PanelState) -> None:
    ax.imshow(panel.frame, cmap="gray", vmin=0.0, vmax=1.0, origin="upper")
    mask = np.asarray(panel.mask0, dtype=np.int32)
    if mask.shape == panel.frame.shape and mask.max() > 0:
        ax.contour(
            mask,
            levels=np.arange(0.5, mask.max() + 0.5, 1.0),
            colors="cyan",
            linewidths=0.35,
            alpha=0.8,
        )

    track_lengths = {track_id: len(rows) for track_id, rows in panel.tracks.items()}
    color_values = normalize_track_lengths(track_lengths)
    cmap = plt.get_cmap("viridis")
    for track_id in sorted(panel.tracks):
        rows = panel.tracks[track_id]
        xs = [row.x for row in rows]
        ys = [row.y for row in rows]
        color = cmap(color_values[track_id])
        ax.plot(xs, ys, color=color, linewidth=0.8, alpha=0.55)


def render_panel_background_png(panel: PanelState, *, dpi: int = 100) -> bytes:
    import matplotlib

    matplotlib.use("Agg")

    height, width = panel.frame.shape
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="white")
    draw_panel_background(ax, panel)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, facecolor="white", pad_inches=0)
    plt.close(fig)
    return buffer.getvalue()


def panel_selection_payload(panel: PanelState) -> dict:
    y0, x0, height, width = panel_roi_tuple(panel)
    return {
        "position": panel.position,
        "roi_y": y0,
        "roi_x": x0,
        "roi_height": height,
        "roi_width": width,
        "track_ids": list(panel.selected_ids),
    }


def build_selection_payload(data_dir: Path, left: PanelState, right: PanelState) -> dict:
    return {
        "data_dir": str(data_dir),
        "left": panel_selection_payload(left),
        "right": panel_selection_payload(right),
    }


def panel_marker_payload(panel: PanelState) -> list[dict]:
    markers: list[dict] = []
    for index, track_id in enumerate(panel.selected_ids):
        centroid = track_centroid(panel.tracks[track_id])
        if centroid is None:
            continue
        y, x = centroid
        markers.append(
            {
                "track_id": track_id,
                "x": x,
                "y": y,
                "color": CELL_COLORS[index % len(CELL_COLORS)],
                "label": str(index + 1),
            }
        )
    return markers


class InteractiveROIPicker:
    def __init__(
        self,
        data_dir: Path,
        left_position: int,
        right_position: int,
        channel: str,
        z: int,
        roi_fraction: float,
        roi_size: int | None,
        output_path: Path,
        on_save: Callable[[dict], None] | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.output_path = output_path
        self.on_save = on_save
        self.mode = "roi"
        self.active_drag_panel: PanelState | None = None
        self.left = load_panel_state(data_dir, left_position, channel, z, roi_fraction, roi_size)
        self.right = load_panel_state(data_dir, right_position, channel, z, roi_fraction, roi_size)
        self.axes_by_panel = {}

        self.fig, (self.ax_left, self.ax_right) = plt.subplots(
            1,
            2,
            figsize=(14, 7),
            facecolor="white",
        )
        self.fig.subplots_adjust(bottom=0.16, wspace=0.05)
        try:
            self.fig.canvas.manager.set_window_title("Interactive ROI and cell picker")
        except Exception:
            pass

        self._draw_panel(self.ax_left, self.left, "patterned")
        self._draw_panel(self.ax_right, self.right, "unpatterned")
        self.axes_by_panel[self.ax_left] = self.left
        self.axes_by_panel[self.ax_right] = self.right

        self._add_controls()
        self._update_status()
        self._connect_events()

    def _draw_panel(self, ax: plt.Axes, panel: PanelState, title: str) -> None:
        draw_panel_background(ax, panel)
        ax.set_title(title, fontsize=12)
        ax.set_xlim(0, panel.frame.shape[1])
        ax.set_ylim(panel.frame.shape[0], 0)
        ax.set_xticks([])
        ax.set_yticks([])

        panel.rect = Rectangle(
            (panel.roi_x0, panel.roi_y0),
            panel.roi_width,
            panel.roi_height,
            fill=False,
            edgecolor="#ffd700",
            linewidth=2.0,
            linestyle="--",
            zorder=5,
        )
        ax.add_patch(panel.rect)
        self._refresh_markers(panel)

    def _refresh_markers(self, panel: PanelState) -> None:
        ax = self._axis_for_panel(panel)
        for artist in panel.marker_artists:
            artist.remove()
        panel.marker_artists.clear()

        for index, track_id in enumerate(panel.selected_ids):
            centroid = track_centroid(panel.tracks[track_id])
            if centroid is None:
                continue
            y, x = centroid
            color = CELL_COLORS[index % len(CELL_COLORS)]
            marker = ax.scatter([x], [y], s=120, facecolors=[color], edgecolors="white", linewidths=1.2, zorder=6)
            label = ax.text(
                x,
                y,
                str(index + 1),
                color="white",
                fontsize=11,
                ha="center",
                va="center",
                fontweight="bold",
                zorder=7,
            )
            panel.marker_artists.extend([marker, label])

    def _axis_for_panel(self, panel: PanelState) -> plt.Axes:
        if panel is self.left:
            return self.ax_left
        return self.ax_right

    def _panel_for_axis(self, ax: plt.Axes) -> PanelState | None:
        return self.axes_by_panel.get(ax)

    def _set_roi(self, panel: PanelState, y0: float, x0: float) -> None:
        panel.roi_y0, panel.roi_x0 = clamp_roi_origin(panel, y0, x0)
        if panel.rect is not None:
            panel.rect.set_xy((panel.roi_x0, panel.roi_y0))
        self._update_status()

    def _add_selection(self, panel: PanelState, track_id: int | None) -> None:
        if track_id is None:
            return
        panel.selected_ids.append(track_id)
        self._refresh_markers(panel)
        self._update_status()
        self.fig.canvas.draw_idle()

    def _add_controls(self) -> None:
        tool_ax = self.fig.add_axes([0.08, 0.03, 0.18, 0.08])
        self.tool_selector = RadioButtons(tool_ax, ("Move ROI", "Pick cells"), active=0)

        save_ax = self.fig.add_axes([0.78, 0.03, 0.14, 0.06])
        save_label = "Save & render" if self.on_save is not None else "Save JSON"
        self.save_button = Button(save_ax, save_label)
        self.save_button.on_clicked(self._on_save_clicked)

        self.status_text = self.fig.text(
            0.30,
            0.04,
            "",
            fontsize=10,
            family="monospace",
            va="center",
        )
        save_hint = "save & render figure" if self.on_save is not None else "save JSON"
        help_text = self.fig.text(
            0.30,
            0.085,
            f"Drag the yellow ROI square. In Pick cells mode, click three cells per side (4th replaces 1st). "
            f"Keys: r=ROI, p=pick, s={save_hint}.",
            fontsize=9,
            va="center",
        )
        help_text.set_alpha(0.75)

    def _connect_events(self) -> None:
        self.tool_selector.on_clicked(self._on_tool_changed)
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _on_tool_changed(self, label: str) -> None:
        self.mode = "roi" if label == "Move ROI" else "pick"
        self._update_status()

    def _on_key(self, event) -> None:
        if event.key == "r":
            self.tool_selector.set_active(0)
            self.mode = "roi"
        elif event.key == "p":
            self.tool_selector.set_active(1)
            self.mode = "pick"
        elif event.key == "s":
            self.save_selection()
        self._update_status()

    def _point_in_roi(self, panel: PanelState, x: float, y: float) -> bool:
        return (
            panel.roi_x0 <= x < panel.roi_x0 + panel.roi_width
            and panel.roi_y0 <= y < panel.roi_y0 + panel.roi_height
        )

    def _on_press(self, event) -> None:
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        panel = self._panel_for_axis(event.inaxes)
        if panel is None:
            return

        if self.mode == "pick":
            track_id = pick_track_at_click(panel, event.ydata, event.xdata)
            self._add_selection(panel, track_id)
            return

        if self._point_in_roi(panel, event.xdata, event.ydata):
            panel.dragging = True
            self.active_drag_panel = panel
            panel.drag_offset_y = event.ydata - panel.roi_y0
            panel.drag_offset_x = event.xdata - panel.roi_x0

    def _on_motion(self, event) -> None:
        panel = self.active_drag_panel
        if panel is None or not panel.dragging or self.mode != "roi":
            return
        if event.xdata is None or event.ydata is None:
            return
        self._set_roi(
            panel,
            event.ydata - panel.drag_offset_y,
            event.xdata - panel.drag_offset_x,
        )
        self.fig.canvas.draw_idle()

    def _on_release(self, _event) -> None:
        for panel in (self.left, self.right):
            panel.dragging = False
        self.active_drag_panel = None

    def _selection_payload(self) -> dict:
        return build_selection_payload(self.data_dir, self.left, self.right)

    def _update_status(self) -> None:
        left_ids = list(self.left.selected_ids)
        right_ids = list(self.right.selected_ids)
        self.status_text.set_text(
            f"Mode: {self.mode.upper()} | "
            f"Left ROI=({self.left.roi_y0},{self.left.roi_x0}) cells={left_ids} | "
            f"Right ROI=({self.right.roi_y0},{self.right.roi_x0}) cells={right_ids}"
        )
        self.fig.canvas.draw_idle()

    def save_selection(self, *_args) -> None:
        payload = self._selection_payload()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {self.output_path}")
        if self.on_save is not None:
            try:
                self.on_save(payload)
            except Exception as exc:
                print(f"Error rendering figure: {exc}", file=sys.stderr)
        else:
            print(
                "Run comparison figure with:\n"
                f"  uv run python scripts/plot_roi_figure.py {self.data_dir} "
                f"--selection-json {self.output_path}"
            )

    def _on_save_clicked(self, _event) -> None:
        self.save_selection()

    def run(self) -> None:
        plt.show()


def run_interactive_picker(
    data_dir: Path,
    left_position: int,
    right_position: int,
    channel: str,
    z: int,
    roi_fraction: float,
    roi_size: int | None,
    output_path: Path,
    on_save: Callable[[dict], None] | None = None,
) -> None:
    picker = InteractiveROIPicker(
        data_dir=data_dir,
        left_position=left_position,
        right_position=right_position,
        channel=channel,
        z=z,
        roi_fraction=roi_fraction,
        roi_size=roi_size,
        output_path=output_path,
        on_save=on_save,
    )
    picker.run()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--left-position", type=int, default=26)
    parser.add_argument("--right-position", type=int, default=37)
    parser.add_argument("--channel", default="all")
    parser.add_argument("--z", type=int, default=0)
    parser.add_argument("--roi-fraction", type=float, default=0.5)
    parser.add_argument("--roi-size", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON path for saved ROI and cell selections",
    )
    args = parser.parse_args(argv)

    data_dir = args.data_dir.expanduser().resolve()
    output = args.output or data_dir / "roi_cell_selection.json"

    try:
        run_interactive_picker(
            data_dir=data_dir,
            left_position=args.left_position,
            right_position=args.right_position,
            channel=args.channel,
            z=args.z,
            roi_fraction=args.roi_fraction,
            roi_size=args.roi_size,
            output_path=output,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())