from __future__ import annotations

import numpy as np

from migration.core.types import DeviceSpec


def run_trackastra_tracking(
    frames: np.ndarray,
    masks: np.ndarray,
    device: DeviceSpec,
    tracking_mode: str,
    delta_t: int,
) -> tuple[np.ndarray, dict[int, int]]:
    if not np.any(masks):
        return np.empty((0, 4), dtype=np.float32), {}

    from trackastra.model import Trackastra
    from trackastra.tracking import graph_to_napari_tracks

    model = Trackastra.from_pretrained("general_2d", device=device.name)
    track_graph, _tracked_masks = model.track(frames, masks, mode=tracking_mode, delta_t=delta_t)
    tracks, track_graph_map, _track_props = graph_to_napari_tracks(track_graph)
    return np.asarray(tracks, dtype=np.float32), {int(k): int(v) for k, v in track_graph_map.items()}
