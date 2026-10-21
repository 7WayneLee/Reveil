import numpy as np
import pytest

from reveil.regions import Region, RegionSet, clip_box, load_regions, random_regions, rasterize, save_regions


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
    assert loaded.frames[0][0].box == (10, 20, 50, 60) and loaded.frames[0][0].polygon is None
    assert loaded.frames[5][0].box == (0, 0, 30, 40)
    np.testing.assert_allclose(loaded.frames[5][0].polygon, [[0, 0], [30, 0], [30, 40]])


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
