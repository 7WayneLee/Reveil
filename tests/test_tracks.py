import pytest

from reveil.regions import Region, RegionSet
from reveil.tracks import box_iou, build_tracks, group_regions, window_starts


def box(x0, y0, x1, y1):
    return Region((x0, y0, x1, y1))


def test_two_moving_regions_make_two_tracks():
    frames = {f: [box(10 + f, 10, 50 + f, 50), box(200 - f, 100, 240 - f, 140)] for f in range(30)}
    tracks = build_tracks(RegionSet(frames))
    assert len(tracks) == 2
    assert all(t.start == 0 and len(t.boxes) == 30 for t in tracks)
    assert {t.boxes[29][0] for t in tracks} == {39, 171}


def test_short_gaps_are_interpolated_and_long_gaps_split():
    frames = {f: [box(f, 0, f + 40, 40)] for f in range(20) if f not in (5, 6, 12, 13, 14, 15, 16)}
    tracks = build_tracks(RegionSet(frames), max_gap=3)
    assert [(t.start, len(t.boxes)) for t in tracks] == [(0, 12), (17, 3)]
    assert tracks[0].boxes[5] == pytest.approx((5, 0, 45, 40))
    assert tracks[0].regions[5] is tracks[0].regions[4]


def test_scene_cuts_split_tracks():
    frames = {f: [box(0, 0, 40, 40)] for f in range(10)}
    tracks = build_tracks(RegionSet(frames, frozenset({4})))
    assert [(t.start, len(t.boxes)) for t in tracks] == [(0, 4), (4, 6)]


def test_overlapping_regions_group_and_iou():
    a, b, c = box(0, 0, 10, 10), box(5, 5, 15, 15), box(50, 50, 60, 60)
    groups = group_regions([a, c, b])
    assert sorted(len(g) for g in groups) == [1, 2]
    assert box_iou(a.box, b.box) == pytest.approx(25 / 175)
    assert box_iou(a.box, c.box) == 0.0


def test_window_starts_cover_whole_windows_only():
    assert window_starts(300, 121, 121) == [0, 121]
    assert window_starts(300, 121, 104, offset=10) == [10, 114]
    assert window_starts(100, 121, 121) == []
