from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class Nd2Selection:
    position: int
    channel: int
    z: int


@dataclass(frozen=True)
class Nd2Scan:
    positions: list[int]
    channels: list[int]
    times: list[int]
    z_slices: list[int]


@dataclass(frozen=True)
class DeviceSpec:
    name: str


@dataclass(frozen=True)
class TrajectoryRow:
    track_id: int
    parent_track_id: int | None
    frame: int
    y: float
    x: float


@dataclass(frozen=True)
class PipelineOutputs:
    overlay_path: Path
    trajectories_path: Path
    segmentation_path: Path
    row_count: int


@dataclass(frozen=True)
class ProgressEvent:
    phase: str
    done: int
    total: int
    message: str


ProgressCallback = Callable[[ProgressEvent], None]


DEFAULT_MIN_TRACK_LENGTH = 0
MIN_TRACK_LENGTH = 50


def emit_progress(
    callback: ProgressCallback | None,
    *,
    phase: str,
    done: int,
    total: int,
    message: str,
) -> None:
    if callback is None:
        return
    callback(ProgressEvent(phase=phase, done=done, total=total, message=message))


def nd2_dimension_size(sizes: dict[str, int], key: str) -> int:
    return int(sizes.get(key, 1))


def nd2_dimension_values(sizes: dict[str, int], key: str) -> list[int]:
    size = nd2_dimension_size(sizes, key)
    return list(range(size)) if size > 0 else []


def nd2_loop_index(handle: Any, p: int, t: int, z: int) -> int:
    loop_indices = tuple(getattr(handle, "loop_indices", ()) or ())
    if not loop_indices:
        return 0
    for seq_index, indices in enumerate(loop_indices):
        if (
            int(indices.get("P", 0)) == p
            and int(indices.get("T", 0)) == t
            and int(indices.get("Z", 0)) == z
        ):
            return seq_index
    raise ValueError("Requested ND2 frame not found")


def nd2_frame_axes(sizes: dict[str, int]) -> list[str]:
    return [dimension for dimension in sizes.keys() if dimension in {"C", "Y", "X", "S"}]


def nd2_frame_to_grayscale(frame: np.ndarray, sizes: dict[str, int], channel: int) -> np.ndarray:
    grayscale = np.asarray(frame)
    active_axes = [axis for axis in nd2_frame_axes(sizes) if nd2_dimension_size(sizes, axis) > 1]

    if grayscale.ndim != len(active_axes):
        if grayscale.ndim == 2:
            active_axes = ["Y", "X"]
        else:
            raise ValueError("Unsupported ND2 frame layout")

    if "C" in active_axes:
        channel_axis = active_axes.index("C")
        if channel < 0 or channel >= grayscale.shape[channel_axis]:
            raise ValueError(f"Channel index {channel} is out of range")
        grayscale = np.take(grayscale, channel, axis=channel_axis)
        active_axes.pop(channel_axis)
    elif channel != 0:
        raise ValueError(f"Channel index {channel} is out of range")

    if "S" in active_axes:
        rgb_axis = active_axes.index("S")
        grayscale = np.rint(np.asarray(grayscale, dtype=np.float32).mean(axis=rgb_axis))
        active_axes.pop(rgb_axis)

    if active_axes != ["Y", "X"] or grayscale.ndim != 2:
        raise ValueError("Unsupported ND2 frame layout")

    return np.array(grayscale, copy=True)


def read_nd2_frame_2d(handle: Any, p: int, t: int, c: int, z: int) -> np.ndarray:
    sizes = {str(key): int(value) for key, value in handle.sizes.items()}
    seq_index = nd2_loop_index(handle, p, t, z)
    frame = handle.read_frame(seq_index)
    return nd2_frame_to_grayscale(frame, sizes, c)


def validate_nd2_index(label: str, value: int, size: int) -> int:
    if value < 0 or value >= max(1, size):
        raise ValueError(f"{label} index {value} is out of range")
    return value


def scan_nd2(path: str | Path) -> Nd2Scan:
    import nd2

    with nd2.ND2File(path) as handle:
        sizes = {str(key): int(value) for key, value in handle.sizes.items()}

    return Nd2Scan(
        positions=nd2_dimension_values(sizes, "P"),
        channels=nd2_dimension_values(sizes, "C"),
        times=nd2_dimension_values(sizes, "T"),
        z_slices=nd2_dimension_values(sizes, "Z"),
    )


def validate_selection(scan: Nd2Scan, selection: Nd2Selection) -> Nd2Selection:
    if not scan.times:
        raise ValueError("ND2 file contains no timepoints")
    validate_nd2_index("Position", selection.position, len(scan.positions))
    validate_nd2_index("Channel", selection.channel, len(scan.channels))
    validate_nd2_index("Z", selection.z, len(scan.z_slices))
    return selection


def convert_frame_to_uint16(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame, dtype=np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=np.iinfo(np.uint16).max, neginf=0.0)
    array = np.clip(np.rint(array), 0, np.iinfo(np.uint16).max)
    return array.astype(np.uint16, copy=False)


def load_nd2_timeseries(path: str | Path, selection: Nd2Selection) -> tuple[Nd2Scan, np.ndarray]:
    import nd2

    with nd2.ND2File(path) as handle:
        sizes = {str(key): int(value) for key, value in handle.sizes.items()}
        scan = Nd2Scan(
            positions=nd2_dimension_values(sizes, "P"),
            channels=nd2_dimension_values(sizes, "C"),
            times=nd2_dimension_values(sizes, "T"),
            z_slices=nd2_dimension_values(sizes, "Z"),
        )
        validate_selection(scan, selection)
        frames = [
            convert_frame_to_uint16(
                read_nd2_frame_2d(handle, selection.position, time_index, selection.channel, selection.z)
            )
            for time_index in scan.times
        ]
    if not frames:
        raise ValueError("ND2 file contains no timepoints")
    return scan, np.stack(frames, axis=0)


def resolve_device(name: str) -> DeviceSpec:
    import torch

    requested = name.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return DeviceSpec("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return DeviceSpec("mps")
        return DeviceSpec("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        return DeviceSpec("cuda")
    if requested == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise ValueError("MPS was requested but is not available")
        return DeviceSpec("mps")
    if requested == "cpu":
        return DeviceSpec("cpu")
    raise ValueError(f"Unsupported device '{name}'")


def run_cellpose_segmentation(frames: np.ndarray, device: DeviceSpec, diameter: float | None) -> np.ndarray:
    import torch
    from cellpose import models

    model = models.CellposeModel(
        device=torch.device(device.name),
        pretrained_model="cpsam",
        use_bfloat16=device.name != "cpu",
    )
    eval_kwargs: dict[str, Any] = {}
    if diameter is not None:
        eval_kwargs["diameter"] = diameter
    masks, _flows, _styles = model.eval([frame.astype(np.float32, copy=False) for frame in frames], **eval_kwargs)
    if isinstance(masks, list):
        return np.stack([np.asarray(mask, dtype=np.int32) for mask in masks], axis=0)
    return np.asarray(masks, dtype=np.int32)


def create_cellpose_model(device: DeviceSpec) -> Any:
    import torch
    from cellpose import models

    return models.CellposeModel(
        device=torch.device(device.name),
        pretrained_model="cpsam",
        use_bfloat16=device.name != "cpu",
    )


def run_cellpose_segmentation_frame(frame: np.ndarray, model: Any, diameter: float | None) -> np.ndarray:
    eval_kwargs: dict[str, Any] = {}
    if diameter is not None:
        eval_kwargs["diameter"] = diameter
    masks, _flows, _styles = model.eval([frame.astype(np.float32, copy=False)], **eval_kwargs)
    if isinstance(masks, list):
        return np.asarray(masks[0], dtype=np.int32)
    array = np.asarray(masks, dtype=np.int32)
    if array.ndim == 3:
        return np.asarray(array[0], dtype=np.int32)
    return array


def read_segmentation_frame(path: str | Path) -> np.ndarray:
    import tifffile

    return np.asarray(tifffile.imread(Path(path)), dtype=np.int32)


def write_segmentation_frame(path: str | Path, mask: np.ndarray) -> Path:
    import tifffile

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(output_path, np.asarray(mask, dtype=np.int32))
    return output_path


def segmentation_frame_cache_is_usable(frame: np.ndarray, mask: np.ndarray) -> bool:
    return mask.ndim == 2 and mask.shape == frame.shape


def run_trackastra_tracking(
    frames: np.ndarray,
    masks: np.ndarray,
    device: DeviceSpec,
    tracking_mode: str,
) -> tuple[np.ndarray, dict[int, int]]:
    if not np.any(masks):
        return np.empty((0, 4), dtype=np.float32), {}

    from trackastra.model import Trackastra
    from trackastra.tracking import graph_to_napari_tracks

    model = Trackastra.from_pretrained("general_2d", device=device.name)
    track_graph, _tracked_masks = model.track(frames, masks, mode=tracking_mode)
    tracks, track_graph_map, _track_props = graph_to_napari_tracks(track_graph)
    return np.asarray(tracks, dtype=np.float32), {int(k): int(v) for k, v in track_graph_map.items()}


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


def build_output_stem(nd2_path: str | Path, selection: Nd2Selection) -> str:
    stem = Path(nd2_path).stem
    return f"{stem}_pos{selection.position}_ch{selection.channel}_z{selection.z}"


def default_output_dir(nd2_path: str | Path, selection: Nd2Selection) -> Path:
    return Path(nd2_path).resolve().parent / build_output_stem(nd2_path, selection)


def segmentation_position_dir(output_dir: str | Path, position: int) -> Path:
    return Path(output_dir) / "segmentation" / f"Pos{position}"


def segmentation_frame_path(output_dir: str | Path, selection: Nd2Selection, time_index: int) -> Path:
    return segmentation_position_dir(output_dir, selection.position) / (
        f"img_channel{selection.channel:03d}"
        f"_position{selection.position:03d}"
        f"_time{time_index:09d}"
        f"_z{selection.z:03d}_mask.tif"
    )


def load_or_create_segmentation_masks(
    frames: np.ndarray,
    output_dir: str | Path,
    selection: Nd2Selection,
    device: DeviceSpec,
    diameter: float | None,
    on_progress: ProgressCallback | None = None,
    total_steps: int = 0,
) -> tuple[Path, np.ndarray]:
    position_dir = segmentation_position_dir(output_dir, selection.position)
    masks: list[np.ndarray] = []
    model: Any | None = None

    for time_index, frame in enumerate(frames):
        output_path = segmentation_frame_path(output_dir, selection, time_index)
        mask: np.ndarray | None = None

        if output_path.exists():
            try:
                cached_mask = read_segmentation_frame(output_path)
            except Exception:
                cached_mask = None
            if cached_mask is not None and segmentation_frame_cache_is_usable(frame, cached_mask):
                mask = cached_mask

        if mask is None:
            if model is None:
                model = create_cellpose_model(device)
            mask = run_cellpose_segmentation_frame(frame, model, diameter)
            write_segmentation_frame(output_path, mask)
            progress_message = "Segmenting frames"
        else:
            progress_message = "Loading cached segmentations"

        masks.append(np.asarray(mask, dtype=np.int32))
        emit_progress(
            on_progress,
            phase="advance",
            done=time_index + 1,
            total=total_steps,
            message=progress_message,
        )

    return position_dir, np.stack(masks, axis=0)


def run_pipeline(
    nd2_path: str | Path,
    selection: Nd2Selection,
    out_dir: str | Path | None,
    device_name: str,
    diameter: float | None,
    min_track_length: int,
    tracking_mode: str,
    on_progress: ProgressCallback | None = None,
) -> PipelineOutputs:
    resolved_path = Path(nd2_path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"ND2 file does not exist: {resolved_path}")

    output_dir = Path(out_dir).expanduser().resolve() if out_dir is not None else default_output_dir(resolved_path, selection)
    device = resolve_device(device_name)
    scan, frames = load_nd2_timeseries(resolved_path, selection)
    output_stem = build_output_stem(resolved_path, selection)
    total_steps = len(scan.times) + 2

    emit_progress(
        on_progress,
        phase="start",
        done=0,
        total=total_steps,
        message=(
            f"Selected 1 position, {len(scan.times)} timepoints, "
            f"1 channel, 1 z-slice. Total steps: {total_steps}"
        ),
    )

    segmentation_path, masks = load_or_create_segmentation_masks(
        frames,
        output_dir,
        selection,
        device,
        diameter,
        on_progress=on_progress,
        total_steps=total_steps,
    )

    tracks, parent_map = run_trackastra_tracking(frames, masks, device, tracking_mode)
    emit_progress(
        on_progress,
        phase="advance",
        done=len(scan.times) + 1,
        total=total_steps,
        message="Tracking trajectories",
    )
    rows = filter_short_trajectories(build_trajectory_rows(tracks, parent_map), min_track_length=min_track_length)

    overlay_path = render_trajectory_overlay(output_dir / f"{output_stem}_overlay.png", frames[0], rows)
    trajectories_path = write_trajectories_csv(output_dir / f"{output_stem}_trajectories.csv", rows)
    emit_progress(
        on_progress,
        phase="finish",
        done=total_steps,
        total=total_steps,
        message=f"Wrote {output_dir}",
    )

    return PipelineOutputs(
        overlay_path=overlay_path,
        trajectories_path=trajectories_path,
        segmentation_path=segmentation_path,
        row_count=len(rows),
    )
