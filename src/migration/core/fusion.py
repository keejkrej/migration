from __future__ import annotations

import numpy as np


def parse_track_weights(value: str | None, channel_count: int) -> tuple[float, ...]:
    if channel_count < 1:
        raise ValueError("At least one channel is required for track weights")
    if value is None:
        return tuple(1.0 for _ in range(channel_count))
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != channel_count:
        raise ValueError(f"Expected {channel_count} track weights, got {len(parts)}")
    if any(not part for part in parts):
        raise ValueError("Track weights must not contain empty entries")
    try:
        weights = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise ValueError("Track weights must be numbers separated by commas") from exc
    if any(weight < 0 for weight in weights):
        raise ValueError("Track weights must be greater than or equal to 0")
    if sum(weights) <= 0:
        raise ValueError("Track weights must sum to a positive value")
    return weights


def normalize_channel_across_time(channel_stack: np.ndarray) -> np.ndarray:
    array = np.asarray(channel_stack, dtype=np.float32)
    if array.size == 0:
        return array
    low = float(np.percentile(array, 1))
    high = float(np.percentile(array, 99))
    if not np.isfinite(low):
        low = 0.0
    if not np.isfinite(high):
        high = low + 1.0
    if high <= low:
        high = low + 1.0
    return np.clip((array - low) / (high - low), 0.0, 1.0)


def fuse_frames_for_tracking(
    frames: np.ndarray,
    weights: str | tuple[float, ...] | None = None,
) -> np.ndarray:
    array = np.asarray(frames)
    if array.ndim == 3:
        channel_count = 1
    elif array.ndim == 4:
        channel_count = int(array.shape[1])
    else:
        raise ValueError(f"Unsupported frame stack shape for tracking fusion: {array.shape}")

    if isinstance(weights, tuple):
        resolved_weights = weights
        if len(resolved_weights) != channel_count:
            raise ValueError(f"Expected {channel_count} track weights, got {len(resolved_weights)}")
    else:
        resolved_weights = parse_track_weights(weights, channel_count)

    if array.ndim == 3:
        return normalize_channel_across_time(array)

    weight_sum = float(sum(resolved_weights))
    fused = np.zeros(array.shape[0:1] + array.shape[2:], dtype=np.float32)
    for channel_index, weight in enumerate(resolved_weights):
        fused += weight * normalize_channel_across_time(array[:, channel_index])
    return fused / weight_sum
