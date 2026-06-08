from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from mdat.core.formats.input.base import ImageInfo
from mdat.core.formats.input.session import inspect_input, open_reader

from migration.core.types import ChannelSelection, Nd2Scan, Nd2Selection


def parse_channel_option(value: str) -> ChannelSelection:
    lowered = value.strip().lower()
    if lowered == "all":
        return "all"
    if "," in value:
        channels: list[int] = []
        for part in value.split(","):
            part = part.strip()
            if not part:
                raise ValueError("Channel list must not contain empty entries")
            try:
                channel = int(part)
            except ValueError as exc:
                raise ValueError("Channel must be a zero-based index, a comma-separated list, or 'all'") from exc
            if channel < 0:
                raise ValueError("Channel index must be greater than or equal to 0")
            channels.append(channel)
        if not channels:
            raise ValueError("At least one channel index is required")
        deduped = tuple(dict.fromkeys(channels))
        if len(deduped) == 1:
            return deduped[0]
        return deduped
    try:
        channel = int(value)
    except ValueError as exc:
        raise ValueError("Channel must be a zero-based index, a comma-separated list, or 'all'") from exc
    if channel < 0:
        raise ValueError("Channel index must be greater than or equal to 0")
    return channel


def channel_selection_stem(channel: ChannelSelection) -> str:
    if channel == "all":
        return "all"
    if isinstance(channel, tuple):
        return "-".join(str(index) for index in channel)
    return str(channel)


def channel_selection_label(channel: ChannelSelection) -> str:
    if channel == "all":
        return "all"
    if isinstance(channel, tuple):
        return "-".join(f"{index:03d}" for index in channel)
    return f"{channel:03d}"


def selection_channel_count(selection: Nd2Selection, scan: Nd2Scan) -> int:
    if selection.channel == "all":
        return max(1, len(scan.channels))
    if isinstance(selection.channel, tuple):
        return len(selection.channel)
    return 1


def frame_spatial_shape(frame: np.ndarray) -> tuple[int, int]:
    if frame.ndim == 2:
        return int(frame.shape[0]), int(frame.shape[1])
    if frame.ndim == 3:
        return int(frame.shape[-2]), int(frame.shape[-1])
    raise ValueError(f"Unsupported frame shape for spatial dimensions: {frame.shape}")


def image_info_to_scan(info: ImageInfo) -> Nd2Scan:
    return Nd2Scan(
        positions=list(range(info.n_pos)),
        channels=list(range(info.n_chan)),
        times=list(range(info.n_time)),
        z_slices=list(range(info.n_z)),
    )


def _channel_indices(channel: ChannelSelection, channel_count: int) -> list[int]:
    if channel == "all":
        indices = list(range(channel_count))
    elif isinstance(channel, tuple):
        indices = list(channel)
    else:
        indices = [channel]
    for index in indices:
        if index < 0 or index >= channel_count:
            raise ValueError(f"Channel index {index} is out of range")
    return indices


def _assemble_channels(planes: list[np.ndarray], channel: ChannelSelection) -> np.ndarray:
    if not planes:
        raise ValueError("At least one channel plane is required")

    if len(planes) == 1:
        plane = np.asarray(planes[0])
        if plane.ndim == 3:
            if channel == "all":
                return np.moveaxis(plane, -1, 0)
            return np.rint(np.asarray(plane, dtype=np.float32).mean(axis=-1)).astype(plane.dtype, copy=False)
        if plane.ndim != 2:
            raise ValueError(f"Unsupported frame shape: {plane.shape}")
        return np.asarray(plane, copy=True)

    stacked = [np.asarray(plane) for plane in planes]
    for plane in stacked:
        if plane.ndim != 2:
            raise ValueError(f"Expected 2D channel plane, got shape {plane.shape}")
    return np.stack(stacked, axis=0)


def read_selection_frame(
    read_frame: Callable[[int, int, int, int], np.ndarray],
    *,
    p: int,
    t: int,
    z: int,
    channel: ChannelSelection,
    channel_count: int,
) -> np.ndarray:
    indices = _channel_indices(channel, channel_count)
    planes = [read_frame(p, t, channel_index, z) for channel_index in indices]
    return _assemble_channels(planes, channel)


def validate_nd2_index(label: str, value: int, size: int) -> int:
    if value < 0 or value >= max(1, size):
        raise ValueError(f"{label} index {value} is out of range")
    return value


def scan_nd2(path: str | Path) -> Nd2Scan:
    return image_info_to_scan(inspect_input(path))


def validate_selection(scan: Nd2Scan, selection: Nd2Selection) -> Nd2Selection:
    if not scan.times:
        raise ValueError("ND2 file contains no timepoints")
    validate_nd2_index("Position", selection.position, len(scan.positions))
    if selection.channel == "all":
        pass
    elif isinstance(selection.channel, tuple):
        for index in selection.channel:
            validate_nd2_index("Channel", index, len(scan.channels))
    else:
        validate_nd2_index("Channel", selection.channel, len(scan.channels))
    validate_nd2_index("Z", selection.z, len(scan.z_slices))
    return selection


def convert_frame_to_uint16(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame, dtype=np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=np.iinfo(np.uint16).max, neginf=0.0)
    array = np.clip(np.rint(array), 0, np.iinfo(np.uint16).max)
    return array.astype(np.uint16, copy=False)


def load_nd2_timeseries(path: str | Path, selection: Nd2Selection) -> tuple[Nd2Scan, np.ndarray]:
    info, read_frame, close = open_reader(path)
    try:
        scan = image_info_to_scan(info)
        validate_selection(scan, selection)
        channel_count = max(1, info.n_chan)
        frames = [
            convert_frame_to_uint16(
                read_selection_frame(
                    read_frame,
                    p=selection.position,
                    t=time_index,
                    z=selection.z,
                    channel=selection.channel,
                    channel_count=channel_count,
                )
            )
            for time_index in scan.times
        ]
    finally:
        close()

    if not frames:
        raise ValueError("ND2 file contains no timepoints")
    return scan, np.stack(frames, axis=0)
