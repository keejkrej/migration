from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from migration.core.segmentation import (
    create_cellpose_model,
    read_segmentation_frame,
    run_cellpose_segmentation_frame,
    segmentation_frame_cache_is_usable,
    write_segmentation_frame,
)
from migration.core.nd2 import channel_selection_label, channel_selection_stem, frame_spatial_shape
from migration.core.types import DeviceSpec, Nd2Selection, ProgressCallback
from migration.utils.progress import emit_progress


def build_output_stem(nd2_path: str | Path, selection: Nd2Selection) -> str:
    stem = Path(nd2_path).stem
    return f"{stem}_pos{selection.position}_ch{channel_selection_stem(selection.channel)}_z{selection.z}"


def default_output_dir(nd2_path: str | Path, selection: Nd2Selection) -> Path:
    return Path(nd2_path).resolve().parent / build_output_stem(nd2_path, selection)


def segmentation_position_dir(output_dir: str | Path, position: int) -> Path:
    return Path(output_dir) / "segmentation" / f"Pos{position}"


def segmentation_frame_path(output_dir: str | Path, selection: Nd2Selection, time_index: int) -> Path:
    return segmentation_position_dir(output_dir, selection.position) / (
        f"img_channel{channel_selection_label(selection.channel)}"
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


def load_segmentation_masks(
    frames: np.ndarray,
    output_dir: str | Path,
    selection: Nd2Selection,
    on_progress: ProgressCallback | None = None,
    total_steps: int = 0,
) -> np.ndarray:
    masks: list[np.ndarray] = []

    for time_index, frame in enumerate(frames):
        output_path = segmentation_frame_path(output_dir, selection, time_index)
        if not output_path.exists():
            raise FileNotFoundError(f"Missing segmentation mask: {output_path}")

        cached_mask = read_segmentation_frame(output_path)
        if not segmentation_frame_cache_is_usable(frame, cached_mask):
            raise ValueError(f"Cached segmentation mask has incompatible shape: {output_path}")

        masks.append(np.asarray(cached_mask, dtype=np.int32))
        emit_progress(
            on_progress,
            phase="advance",
            done=time_index + 1,
            total=total_steps,
            message="Loading cached segmentations",
        )

    return np.stack(masks, axis=0)
