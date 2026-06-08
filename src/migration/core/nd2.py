from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from migration.core.types import Nd2Scan, Nd2Selection


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
