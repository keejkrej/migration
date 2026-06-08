from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


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
class SegmentOutputs:
    segmentation_path: Path
    frame_count: int


@dataclass(frozen=True)
class TrackOutputs:
    overlay_path: Path
    trajectories_path: Path
    row_count: int


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
