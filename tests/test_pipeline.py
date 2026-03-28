from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pytest

from migration.cli import main
from migration.pipeline import (
    Nd2Scan,
    Nd2Selection,
    TrajectoryRow,
    build_trajectory_rows,
    nd2_dimension_size,
    read_nd2_frame_2d,
    render_trajectory_overlay,
    validate_selection,
    write_trajectories_csv,
)


class FakeND2Handle:
    def __init__(self, sizes: dict[str, int], frames: list[np.ndarray], loop_axes: tuple[str, ...]) -> None:
        self.sizes = sizes
        self._frames = frames
        self.loop_indices = tuple(
            dict(zip(loop_axes, coords))
            for coords in product(*(range(int(sizes[axis])) for axis in loop_axes))
        )
        self.last_seq_index: int | None = None

    def read_frame(self, seq_index: int) -> np.ndarray:
        self.last_seq_index = seq_index
        return self._frames[seq_index]


def test_reads_channel_from_frame_axes_without_using_channel_in_sequence_index() -> None:
    sizes = {"P": 2, "T": 2, "Z": 2, "C": 2, "Y": 3, "X": 4}
    frames = [
        np.stack(
            [
                np.full((3, 4), fill_value=seq_index, dtype=np.uint16),
                np.full((3, 4), fill_value=seq_index + 100, dtype=np.uint16),
            ]
        )
        for seq_index in range(8)
    ]
    handle = FakeND2Handle(sizes, frames, ("P", "T", "Z"))

    image = read_nd2_frame_2d(handle, p=1, t=0, c=1, z=1)

    expected_seq_index = handle.loop_indices.index({"P": 1, "T": 0, "Z": 1})
    assert handle.last_seq_index == expected_seq_index
    assert image.shape == (3, 4)
    assert np.all(image == expected_seq_index + 100)


def test_rgb_frame_is_converted_to_grayscale() -> None:
    sizes = {"T": 2, "Y": 2, "X": 3, "S": 3}
    frames = [
        np.dstack(
            [
                np.full((2, 3), fill_value=5 + seq_index, dtype=np.uint16),
                np.full((2, 3), fill_value=15 + seq_index, dtype=np.uint16),
                np.full((2, 3), fill_value=25 + seq_index, dtype=np.uint16),
            ]
        )
        for seq_index in range(2)
    ]
    handle = FakeND2Handle(sizes, frames, ("T",))

    image = read_nd2_frame_2d(handle, p=0, t=1, c=0, z=0)

    assert handle.last_seq_index == 1
    assert image.shape == (2, 3)
    assert np.all(image == 16)


def test_missing_axes_default_to_singleton_size() -> None:
    assert nd2_dimension_size({"Y": 8, "X": 9}, "P") == 1
    assert nd2_dimension_size({"Y": 8, "X": 9}, "Z") == 1


def test_validate_selection_rejects_out_of_range_indices() -> None:
    scan = Nd2Scan(positions=[0, 1], channels=[0], times=[0, 1], z_slices=[0, 1])

    try:
        validate_selection(scan, Nd2Selection(position=2, channel=0, z=0))
    except ValueError as exc:
        assert "Position index 2 is out of range" in str(exc)
    else:
        raise AssertionError("Expected out-of-range position to fail")


def test_validate_selection_rejects_missing_timepoints() -> None:
    scan = Nd2Scan(positions=[0], channels=[0], times=[], z_slices=[0])

    try:
        validate_selection(scan, Nd2Selection(position=0, channel=0, z=0))
    except ValueError as exc:
        assert "contains no timepoints" in str(exc)
    else:
        raise AssertionError("Expected empty times to fail")


def test_build_trajectory_rows_preserves_parent_mapping() -> None:
    tracks = np.array(
        [
            [2, 1, 11.0, 21.0],
            [1, 0, 10.0, 20.0],
            [2, 2, 12.0, 22.0],
        ],
        dtype=np.float32,
    )

    rows = build_trajectory_rows(tracks, {2: 1})

    assert rows == [
        TrajectoryRow(track_id=1, parent_track_id=None, frame=0, y=10.0, x=20.0),
        TrajectoryRow(track_id=2, parent_track_id=1, frame=1, y=11.0, x=21.0),
        TrajectoryRow(track_id=2, parent_track_id=1, frame=2, y=12.0, x=22.0),
    ]


def test_write_trajectories_csv_writes_header_and_rows(tmp_path: Path) -> None:
    output_path = tmp_path / "trajectories.csv"
    rows = [
        TrajectoryRow(track_id=7, parent_track_id=None, frame=0, y=1.25, x=2.5),
        TrajectoryRow(track_id=8, parent_track_id=7, frame=1, y=3.5, x=4.75),
    ]

    write_trajectories_csv(output_path, rows)

    assert output_path.read_text(encoding="utf-8").splitlines() == [
        "track_id,parent_track_id,frame,y,x",
        "7,,0,1.250000,2.500000",
        "8,7,1,3.500000,4.750000",
    ]


def test_render_trajectory_overlay_writes_png(tmp_path: Path) -> None:
    output_path = tmp_path / "overlay.png"
    frame = np.arange(64, dtype=np.uint16).reshape(8, 8)
    rows = [
        TrajectoryRow(track_id=1, parent_track_id=None, frame=0, y=1.0, x=1.0),
        TrajectoryRow(track_id=1, parent_track_id=None, frame=1, y=2.0, x=2.0),
        TrajectoryRow(track_id=2, parent_track_id=None, frame=0, y=5.0, x=5.0),
        TrajectoryRow(track_id=2, parent_track_id=None, frame=1, y=6.0, x=6.0),
    ]

    render_trajectory_overlay(output_path, frame, rows)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_cli_rejects_nonpositive_diameter() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["sample.nd2", "--position", "0", "--channel", "0", "--z", "0", "--diameter", "0"])

    assert exc_info.value.code == 2
