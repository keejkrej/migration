from __future__ import annotations

from migration.core.types import DeviceSpec


def resolve_device() -> DeviceSpec:
    import torch

    if torch.cuda.is_available():
        return DeviceSpec("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return DeviceSpec("mps")
    return DeviceSpec("cpu")
