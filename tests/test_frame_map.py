import json
from dataclasses import replace

import numpy as np
import pytest

from reveil.frame_map import Alignment, load_alignment, save_alignment


@pytest.fixture
def alignment():
    return Alignment(
        retail="retail.mp4",
        leak="leak.mp4",
        retail_frames=14,
        leak_frames=50,
        retail_fps=29.97,
        leak_fps=30.0,
        retail_size=(1920, 1080),
        leak_size=(1280, 720),
        runs=((2, 10, 3), (5, 30, 2), (9, 10, 3)),
        segments=({"retail": [2, 7], "confidence": 0.98},),
        stats={"matched": 8},
    )


def test_alignment_round_trip_through_json(alignment, tmp_path):
    path = tmp_path / "alignment.json"
    path.write_text("old alignment", encoding="utf-8")
    save_alignment(alignment, path)
    loaded = load_alignment(path)
    assert loaded == alignment
    assert loaded.leak_index(9) == 10
    assert loaded.matched_frames() == 8
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert data["retail_size"] == [1920, 1080]
    assert data["runs"] == [[2, 10, 3], [5, 30, 2], [9, 10, 3]]
    assert ": " not in path.read_text(encoding="utf-8")
    assert not path.with_name(path.name + ".part").exists()


def test_alignment_write_preserves_previous_file_until_replace(alignment, tmp_path, monkeypatch):
    path = tmp_path / "alignment.json"
    path.write_text("old alignment", encoding="utf-8")

    def interrupted_replace(source, destination):
        assert destination == path
        assert source == path.with_name(path.name + ".part")
        assert json.loads(source.read_text(encoding="utf-8"))["runs"] == [[2, 10, 3], [5, 30, 2], [9, 10, 3]]
        raise OSError("interrupted replacement")

    monkeypatch.setattr("reveil.frame_map.os.replace", interrupted_replace)
    with pytest.raises(OSError, match="interrupted replacement"):
        save_alignment(alignment, path)
    assert path.read_text(encoding="utf-8") == "old alignment"


def test_leak_lookup_at_run_edges_and_gaps(alignment):
    expected = [-1, -1, 10, 11, 12, 30, 31, -1, -1, 10, 11, 12, -1, -1]
    for index, leak_index in enumerate(expected):
        assert alignment.leak_index(index) == (None if leak_index == -1 else leak_index)
    assert alignment.leak_index(-1) is None
    assert alignment.leak_index(14) is None
    indices = alignment.leak_indices(0, 14)
    assert indices.dtype == np.int64
    np.testing.assert_array_equal(indices, expected)
    np.testing.assert_array_equal(alignment.leak_indices(4, 10), [12, 30, 31, -1, -1, 10])
    np.testing.assert_array_equal(alignment.leak_indices(-1, 3), [-1, -1, -1, 10])
    np.testing.assert_array_equal(alignment.leak_indices(11, 16), [12, -1, -1, -1, -1])
    assert alignment.leak_indices(5, 5).shape == (0,)


def test_window_ok_follows_retail_across_forward_and_backward_jumps(alignment):
    alignment = replace(alignment, runs=((0, 10, 3), (3, 30, 3), (6, 10, 3), (10, 40, 2)))
    assert alignment.window_ok(1, 4)  # forward jump
    assert alignment.window_ok(4, 4)  # backward jump and reuse of a leak shot
    assert alignment.window_ok(0, 9)
    assert alignment.window_ok(10, 2)
    assert not alignment.window_ok(8, 3)  # unmatched frame 9
    assert not alignment.window_ok(12, 1)
    assert not alignment.window_ok(-1, 2)
    assert not alignment.window_ok(13, 2)
    assert not alignment.window_ok(0, -1)
    assert alignment.window_ok(14, 0)


def test_leak_jumps_only_counts_discontinuous_boundaries_inside_range(alignment):
    alignment = replace(alignment, runs=((0, 10, 3), (3, 13, 2), (5, 16, 2), (7, 40, 2), (9, 5, 2)))
    assert alignment.leak_jumps(0, 11) == 2
    assert alignment.leak_jumps(0, 7) == 0  # +1 and +2 are smooth; stop is excluded
    assert alignment.leak_jumps(6, 8) == 1
    assert alignment.leak_jumps(7, 9) == 0  # the jump before start is excluded
    assert alignment.leak_jumps(7, 10) == 1
    assert alignment.leak_jumps(9, 11) == 0
    assert alignment.leak_jumps(9, 9) == 0
    assert alignment.leak_jumps(10, 7) == 0
    assert alignment.leak_jumps(-5, 20) == 2
    gapped = replace(alignment, runs=((0, 10, 2), (3, 40, 2)))
    assert gapped.leak_jumps(0, 5) == 0  # no adjacent matched frames at a gap
    repeated = replace(alignment, runs=((0, 10, 2), (2, 11, 2)))
    assert repeated.leak_jumps(0, 4) == 1  # repeating the last leak frame is a jump


def test_empty_alignment(alignment):
    alignment = replace(alignment, runs=(), retail_frames=0, leak_frames=0)
    assert alignment.leak_index(0) is None
    np.testing.assert_array_equal(alignment.leak_indices(0, 3), [-1, -1, -1])
    assert alignment.leak_indices(0, 0).dtype == np.int64
    assert alignment.matched_frames() == 0
    assert alignment.leak_jumps(0, 10) == 0
    assert not alignment.window_ok(0, 1)


@pytest.mark.parametrize(
    ("runs", "message"),
    [
        (((2, 10, 0),), "length must be at least 1"),
        (((2, 10, -1),), "length must be at least 1"),
        (((-1, 10, 2),), "retail indices are outside"),
        (((14, 10, 1),), "retail indices are outside"),
        (((13, 10, 2),), "retail indices are outside"),
        (((2, -1, 2),), "leak indices are outside"),
        (((2, 50, 1),), "leak indices are outside"),
        (((2, 49, 2),), "leak indices are outside"),
        (((5, 10, 2), (2, 20, 2)), "sorted by retail_start"),
        (((2, 10, 3), (4, 20, 2)), "overlap on the retail side"),
        (((2, 10, 3), (2, 20, 2)), "overlap on the retail side"),
    ],
)
def test_invalid_runs_are_rejected(alignment, runs, message):
    with pytest.raises(ValueError, match=message):
        replace(alignment, runs=runs)


@pytest.mark.parametrize("field", ["retail_frames", "leak_frames"])
def test_negative_frame_counts_are_rejected(alignment, field):
    with pytest.raises(ValueError, match="frame counts must be non-negative"):
        replace(alignment, **{field: -1})


def test_runs_can_end_at_video_bounds(alignment):
    alignment = replace(alignment, runs=((0, 36, 14),))
    assert alignment.window_ok(0, 14)
    assert alignment.leak_index(13) == 49
    assert alignment.matched_frames() == 14


@pytest.mark.parametrize("version", [0, 2, None])
def test_unsupported_alignment_version_is_rejected(alignment, tmp_path, version):
    path = tmp_path / "alignment.json"
    save_alignment(alignment, path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if version is None:
        del data["version"]
    else:
        data["version"] = version
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported alignment version"):
        load_alignment(path)


def test_alignment_metadata_defaults_when_omitted(alignment, tmp_path):
    path = tmp_path / "alignment.json"
    save_alignment(alignment, path)
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["segments"], data["stats"]
    path.write_text(json.dumps(data), encoding="utf-8")
    loaded = load_alignment(path)
    assert loaded.segments == ()
    assert loaded.stats == {}
