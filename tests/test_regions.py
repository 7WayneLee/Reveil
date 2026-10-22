import json

import numpy as np
import pytest

from reveil.regions import Overlay, Region, RegionSet, clip_box, load_regions, random_regions, rasterize, save_regions


def test_regions_round_trip_through_json(tmp_path):
    regions = RegionSet(
        {
            0: [Region.from_json({"box": [10, 20, 50, 60]})],
            5: [Region.from_json({"polygon": [[0, 0], [30, 0], [30, 40]]})],
        },
        frozenset({3}),
    )
    path = tmp_path / "regions.json"
    save_regions(regions, path)
    loaded = load_regions(path)
    assert loaded.cuts == {3}
    assert loaded.overlays == ()
    assert loaded.frames[0][0].box == (10, 20, 50, 60) and loaded.frames[0][0].polygon is None
    assert loaded.frames[5][0].box == (0, 0, 30, 40)
    np.testing.assert_allclose(loaded.frames[5][0].polygon, [[0, 0], [30, 0], [30, 40]])


def test_overlay_round_trip_with_and_without_frame_range(tmp_path):
    overlays = (
        Overlay((10, 20, 50, 60), "retail", 0.86, (1200, 54000)),
        Overlay((100.5, 200, 150, 260), "leak", 1.0),
    )
    regions = RegionSet({0: [Region((5, 10, 20, 30))]}, frozenset({100}), overlays)
    path = tmp_path / "regions.json"
    save_regions(regions, path)
    loaded = load_regions(path)
    assert loaded.overlays == overlays
    assert loaded.cuts == {100}
    assert loaded.frames[0][0].box == (5, 10, 20, 30)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["overlays"] == [
        {"box": [10, 20, 50, 60], "side": "retail", "frequency": 0.86, "frames": [1200, 54000]},
        {"box": [100.5, 200, 150, 260], "side": "leak", "frequency": 1.0},
    ]
    assert not path.with_name(path.name + ".part").exists()


def test_overlay_active_at_half_open_range_edges():
    overlay = Overlay((0, 0, 10, 10), "retail", 0.5, (10, 20))
    assert not overlay.active(9)
    assert overlay.active(10)
    assert overlay.active(19)
    assert not overlay.active(20)
    assert not overlay.active_between(0, 10)
    assert overlay.active_between(9, 11)
    assert overlay.active_between(10, 11)
    assert overlay.active_between(19, 20)
    assert overlay.active_between(19, 21)
    assert overlay.active_between(0, 30)
    assert not overlay.active_between(20, 30)
    assert not overlay.active_between(15, 15)
    assert not overlay.active_between(20, 10)


def test_overlay_without_frame_range_is_active_throughout():
    overlay = Overlay((0, 0, 10, 10), "leak", 0.5)
    assert overlay.active(0)
    assert overlay.active(100000)
    assert overlay.active_between(0, 1)
    assert overlay.active_between(100000, 100001)
    assert not overlay.active_between(5, 5)
    assert not overlay.active_between(6, 5)


def test_regions_extra_keys_are_written_and_ignored_by_loader(tmp_path):
    regions = RegionSet(overlays=(Overlay((0, 0, 10, 10), "retail", 0.86),))
    extra = {"settings": {"threshold": 0.5}, "stats": {"analysed_frames": 100}}
    path = tmp_path / "regions.json"
    save_regions(regions, path, dropped={3: [Region((1, 2, 3, 4))]}, extra=extra)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["settings"] == extra["settings"]
    assert data["stats"] == extra["stats"]
    assert data["dropped"] == {"3": [{"box": [1, 2, 3, 4]}]}
    loaded = load_regions(path)
    assert loaded.frames == {}
    assert loaded.overlays == regions.overlays


def test_regions_without_overlays_keep_existing_json_bytes(tmp_path):
    regions = RegionSet(
        {5: [Region((10, 20, 50, 60))], 0: [Region((1, 2, 3, 4))]},
        frozenset({7, 3}),
    )
    path = tmp_path / "regions.json"
    save_regions(regions, path, dropped={2: [Region((20, 30, 40, 50))]})
    assert path.read_bytes() == (
        b'{"version":1,"frames":{"0":[{"box":[1,2,3,4]}],"5":[{"box":[10,20,50,60]}]},'
        b'"cuts":[3,7],"dropped":{"2":[{"box":[20,30,40,50]}]}}'
    )
    assert load_regions(path).overlays == ()


def test_file_without_overlays_loads_as_empty_tuple(tmp_path):
    path = tmp_path / "regions.json"
    path.write_text('{"version":1,"frames":{},"cuts":[]}', encoding="utf-8")
    assert load_regions(path).overlays == ()


def test_bad_regions_are_rejected():
    with pytest.raises(ValueError):
        Region.from_json({"box": [10, 10, 5, 20]})
    with pytest.raises(ValueError):
        Region.from_json({"polygon": [[0, 0], [1, 1]]})


def test_rasterize_box_ellipse_and_polygon_in_local_coordinates():
    box = Region.from_json({"box": [10, 10, 20, 30]})
    rect = rasterize([box], (5, 5), (40, 40), ellipse=False)
    assert rect.sum() == 10 * 20
    assert rect[5:25, 5:15].all()
    oval = rasterize([box], (5, 5), (40, 40), ellipse=True)
    assert 0 < oval.sum() < rect.sum() and not (oval & ~rect).any()
    triangle = Region.from_json({"polygon": [[0, 0], [20, 0], [0, 20]]})
    mask = rasterize([triangle], (0, 0), (30, 30), ellipse=False)
    assert mask[2, 2] and not mask[18, 18]


def test_rasterize_clips_regions_reaching_past_the_grid():
    region = Region.from_json({"box": [-10, -10, 30, 30]})
    for ellipse in (False, True):
        mask = rasterize([region], (0, 0), (20, 20), ellipse=ellipse)
        assert mask.shape == (20, 20) and mask[10, 10]


def test_clip_box_keeps_the_inside_part():
    assert clip_box((-10, 5, 30, 50), 20, 40) == (0, 5, 20, 40)
    assert clip_box((25, 5, 30, 10), 20, 40) is None


def test_random_regions_stay_inside_the_frame():
    regions = random_regions(300, 320, 240, np.random.default_rng(0))
    assert sorted(regions.frames) == list(range(300))
    for (region,) in regions.frames.values():
        x0, y0, x1, y1 = region.box
        assert 0 <= x0 < x1 <= 320 and 0 <= y0 < y1 <= 240
