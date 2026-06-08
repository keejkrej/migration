from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from migration.core.nd2 import frame_spatial_shape
from migration.core.types import DeviceSpec


def run_cellpose_segmentation(
    frames: np.ndarray,
    device: DeviceSpec,
    diameter: float | None,
    batch_size: int,
) -> np.ndarray:
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
    eval_kwargs["batch_size"] = batch_size
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


def run_cellpose_segmentation_frame(
    frame: np.ndarray,
    model: Any,
    diameter: float | None,
    batch_size: int,
) -> np.ndarray:
    eval_kwargs: dict[str, Any] = {"batch_size": batch_size}
    if diameter is not None:
        eval_kwargs["diameter"] = diameter
    if frame.ndim == 3:
        eval_kwargs["channel_axis"] = 0
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
    return mask.ndim == 2 and mask.shape == frame_spatial_shape(frame)
