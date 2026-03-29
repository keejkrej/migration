from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pytest

from migration.cli import main
from migration.pipeline import (
    DEFAULT_MIN_TRACK_LENGTH,
    MIN_TRACK_LENGTH,
    Nd2Scan,
    Nd2Selection,
    ProgressEvent,
    TrajectoryRow,
    build_trajectory_rows,
    filter_short_trajectories,
    nd2_dimension_size,
    read_nd2_frame_2d,
    read_segmentation_frame,
    render_trajectory_overlay,
    run_pipeline,
    segmentation_frame_cache_is_usable,
    segmentation_frame_path,
    segmentation_position_dir,
    write_segmentation_frame,
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


def test_filter_short_trajectories_discards_tracks_shorter_than_minimum() -> None:
    rows = [
        TrajectoryRow(track_id=1, parent_track_id=None, frame=frame, y=float(frame), x=float(frame))
        for frame in range(MIN_TRACK_LENGTH)
    ] + [
        TrajectoryRow(track_id=2, parent_track_id=None, frame=frame, y=float(frame), x=float(frame))
        for frame in range(MIN_TRACK_LENGTH - 1)
    ]

    filtered = filter_short_trajectories(rows)

    assert len(filtered) == MIN_TRACK_LENGTH
    assert {row.track_id for row in filtered} == {1}


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


def test_segmentation_frame_cache_is_usable_requires_matching_2d_shape() -> None:
    frame = np.zeros((3, 4), dtype=np.uint16)

    assert segmentation_frame_cache_is_usable(frame, np.zeros((3, 4), dtype=np.int32))
    assert not segmentation_frame_cache_is_usable(frame, np.zeros((2, 3, 4), dtype=np.int32))
    assert not segmentation_frame_cache_is_usable(frame, np.zeros((3, 5), dtype=np.int32))


def test_run_pipeline_emits_convert_style_progress_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nd2_path = tmp_path / "sample.nd2"
    nd2_path.touch()
    frames = np.arange(2 * 4 * 5, dtype=np.uint16).reshape(2, 4, 5)
    output_dir = tmp_path / "results"
    events: list[ProgressEvent] = []

    monkeypatch.setattr(
        "migration.pipeline.resolve_device",
        lambda name: type("Device", (), {"name": name})(),
    )
    monkeypatch.setattr("migration.pipeline.load_nd2_timeseries", lambda path, selection: (Nd2Scan([0], [0], [0, 1], [0]), frames))
    monkeypatch.setattr("migration.pipeline.create_cellpose_model", lambda device: object())
    monkeypatch.setattr(
        "migration.pipeline.run_cellpose_segmentation_frame",
        lambda frame, model, diameter: np.zeros_like(frame, dtype=np.int32),
    )
    monkeypatch.setattr(
        "migration.pipeline.run_trackastra_tracking",
        lambda frames, masks, device, tracking_mode: (np.empty((0, 4), dtype=np.float32), {}),
    )

    run_pipeline(
        nd2_path=nd2_path,
        selection=Nd2Selection(position=0, channel=0, z=0),
        out_dir=output_dir,
        device_name="cpu",
        diameter=None,
        min_track_length=MIN_TRACK_LENGTH,
        tracking_mode="greedy",
        on_progress=events.append,
    )

    assert [event.phase for event in events] == ["start", "advance", "advance", "advance", "finish"]
    assert events[0].total == len(frames) + 2
    assert events[1].message == "Segmenting frames"
    assert events[2].message == "Segmenting frames"
    assert events[3].message == "Tracking trajectories"
    assert events[4].message == f"Wrote {output_dir.resolve()}"


def test_run_pipeline_writes_outputs_and_segmentation_cache_to_output_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nd2_path = tmp_path / "sample.nd2"
    nd2_path.touch()
    frames = np.arange(2 * 4 * 5, dtype=np.uint16).reshape(2, 4, 5)
    masks = np.stack(
        [
            np.full((4, 5), fill_value=11, dtype=np.int32),
            np.full((4, 5), fill_value=22, dtype=np.int32),
        ],
        axis=0,
    )
    long_track = np.array(
        [[1, frame, 1.0 + frame, 2.0 + frame] for frame in range(MIN_TRACK_LENGTH)],
        dtype=np.float32,
    )
    short_track = np.array(
        [[2, frame, 10.0 + frame, 20.0 + frame] for frame in range(MIN_TRACK_LENGTH - 1)],
        dtype=np.float32,
    )
    tracks = np.vstack([long_track, short_track])
    output_dir = tmp_path / "results"
    calls = {"count": 0}

    monkeypatch.setattr(
        "migration.pipeline.resolve_device",
        lambda name: type("Device", (), {"name": name})(),
    )
    monkeypatch.setattr("migration.pipeline.load_nd2_timeseries", lambda path, selection: (Nd2Scan([0], [0], [0, 1], [0]), frames))
    monkeypatch.setattr("migration.pipeline.create_cellpose_model", lambda device: object())

    def fake_segmentation_frame(frame: np.ndarray, model: object, diameter: float | None) -> np.ndarray:
        mask = masks[calls["count"]]
        calls["count"] += 1
        return mask

    monkeypatch.setattr("migration.pipeline.run_cellpose_segmentation_frame", fake_segmentation_frame)
    monkeypatch.setattr("migration.pipeline.run_trackastra_tracking", lambda frames, masks, device, tracking_mode: (tracks, {}))

    outputs = run_pipeline(
        nd2_path=nd2_path,
        selection=Nd2Selection(position=0, channel=0, z=0),
        out_dir=output_dir,
        device_name="cpu",
        diameter=None,
        min_track_length=MIN_TRACK_LENGTH,
        tracking_mode="greedy",
    )

    assert outputs.overlay_path == output_dir / "sample_pos0_ch0_z0_overlay.png"
    assert outputs.trajectories_path == output_dir / "sample_pos0_ch0_z0_trajectories.csv"
    assert outputs.segmentation_path == segmentation_position_dir(output_dir, 0)
    assert outputs.overlay_path.exists()
    assert outputs.trajectories_path.exists()
    assert outputs.segmentation_path.exists()
    assert outputs.row_count == MIN_TRACK_LENGTH
    assert calls["count"] == 2
    assert np.array_equal(read_segmentation_frame(segmentation_frame_path(output_dir, Nd2Selection(0, 0, 0), 0)), masks[0])
    assert np.array_equal(read_segmentation_frame(segmentation_frame_path(output_dir, Nd2Selection(0, 0, 0), 1)), masks[1])
    csv_lines = outputs.trajectories_path.read_text(encoding="utf-8").splitlines()
    assert len(csv_lines) == MIN_TRACK_LENGTH + 1
    assert all(line.startswith("1,") for line in csv_lines[1:])


def test_run_pipeline_reuses_cached_segmentations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nd2_path = tmp_path / "sample.nd2"
    nd2_path.touch()
    frames = np.arange(2 * 4 * 5, dtype=np.uint16).reshape(2, 4, 5)
    cached_mask = np.full((4, 5), fill_value=7, dtype=np.int32)
    computed_mask = np.full((4, 5), fill_value=9, dtype=np.int32)
    output_dir = tmp_path / "results"
    selection = Nd2Selection(position=0, channel=0, z=0)
    write_segmentation_frame(segmentation_frame_path(output_dir, selection, 0), cached_mask)
    calls = {"segmentation": 0}

    monkeypatch.setattr(
        "migration.pipeline.resolve_device",
        lambda name: type("Device", (), {"name": name})(),
    )
    monkeypatch.setattr("migration.pipeline.load_nd2_timeseries", lambda path, selection: (Nd2Scan([0], [0], [0, 1], [0]), frames))
    monkeypatch.setattr("migration.pipeline.create_cellpose_model", lambda device: object())

    def fake_segmentation_frame(frame: np.ndarray, model: object, diameter: float | None) -> np.ndarray:
        calls["segmentation"] += 1
        return computed_mask

    captured_masks: dict[str, np.ndarray] = {}

    def fake_tracking(frames: np.ndarray, masks_arg: np.ndarray, device: object, tracking_mode: str) -> tuple[np.ndarray, dict[int, int]]:
        captured_masks["value"] = np.array(masks_arg, copy=True)
        return np.empty((0, 4), dtype=np.float32), {}

    monkeypatch.setattr("migration.pipeline.run_cellpose_segmentation_frame", fake_segmentation_frame)
    monkeypatch.setattr("migration.pipeline.run_trackastra_tracking", fake_tracking)

    outputs = run_pipeline(
        nd2_path=nd2_path,
        selection=selection,
        out_dir=output_dir,
        device_name="cpu",
        diameter=None,
        min_track_length=MIN_TRACK_LENGTH,
        tracking_mode="greedy",
    )

    assert calls["segmentation"] == 1
    assert np.array_equal(captured_masks["value"][0], cached_mask)
    assert np.array_equal(captured_masks["value"][1], computed_mask)
    assert outputs.segmentation_path == segmentation_position_dir(output_dir, 0)
    assert np.array_equal(read_segmentation_frame(segmentation_frame_path(output_dir, selection, 1)), computed_mask)


def test_cli_rejects_nonpositive_diameter() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["sample.nd2", "--position", "0", "--channel", "0", "--z", "0", "--diameter", "0"])

    assert exc_info.value.code == 2


def test_cli_rejects_negative_min_track_length() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["sample.nd2", "--position", "0", "--channel", "0", "--z", "0", "--min-track-length", "-1"])

    assert exc_info.value.code == 2


def test_cli_accepts_output_alias(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    recorded: dict[str, object] = {}

    def fake_run_pipeline(
        nd2_path: Path,
        selection: Nd2Selection,
        out_dir: Path | None,
        device_name: str,
        diameter: float | None,
        min_track_length: int,
        tracking_mode: str,
        on_progress: object | None = None,
    ) -> object:
        recorded["out_dir"] = out_dir
        recorded["min_track_length"] = min_track_length
        recorded["on_progress"] = on_progress

        class Outputs:
            segmentation_path = tmp_path / "out" / "segmentation" / "Pos0"
            overlay_path = tmp_path / "out" / "sample_overlay.png"
            trajectories_path = tmp_path / "out" / "sample_trajectories.csv"
            row_count = 0

        return Outputs()

    monkeypatch.setattr("migration.cli.run_pipeline", fake_run_pipeline)

    exit_code = main(["sample.nd2", "--position", "0", "--channel", "0", "--z", "0", "--output", str(tmp_path / "out")])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert recorded["out_dir"] == tmp_path / "out"
    assert recorded["min_track_length"] == DEFAULT_MIN_TRACK_LENGTH
    assert recorded["on_progress"] is not None
    assert "Segmentation:" in captured.out


def test_cli_passes_min_track_length(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recorded: dict[str, object] = {}

    def fake_run_pipeline(
        nd2_path: Path,
        selection: Nd2Selection,
        out_dir: Path | None,
        device_name: str,
        diameter: float | None,
        min_track_length: int,
        tracking_mode: str,
        on_progress: object | None = None,
    ) -> object:
        recorded["min_track_length"] = min_track_length

        class Outputs:
            segmentation_path = tmp_path / "out" / "segmentation" / "Pos0"
            overlay_path = tmp_path / "out" / "sample_overlay.png"
            trajectories_path = tmp_path / "out" / "sample_trajectories.csv"
            row_count = 0

        return Outputs()

    monkeypatch.setattr("migration.cli.run_pipeline", fake_run_pipeline)

    exit_code = main(
        [
            "sample.nd2",
            "--position",
            "0",
            "--channel",
            "0",
            "--z",
            "0",
            "--min-track-length",
            "75",
        ]
    )

    assert exit_code == 0
    assert recorded["min_track_length"] == 75
