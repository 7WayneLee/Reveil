import math

import numpy as np

from reveil.camera import CameraConfig, CropFrame, crop_to_canvas, gaussian_smooth, plan_crops


def contains(crop: CropFrame, box) -> bool:
    x0, y0, x1, y1 = crop.box
    return x0 <= math.floor(box[0]) and y0 <= math.floor(box[1]) and math.ceil(box[2]) <= x1 and math.ceil(box[3]) <= y1


def test_views_contain_the_region_and_stay_in_the_frame():
    boxes = []
    for f in range(240):
        size = 40 + 160 * (f > 120)  # the region jumps from small to large
        x = 50 + 3 * f
        boxes.append((x, 300, x + size, 300 + size))
    crops = plan_crops(boxes, 1280, 720, 30.0)
    for crop, b in zip(crops, boxes):
        x0, y0, x1, y1 = crop.box
        assert 0 <= x0 < x1 <= 1280 and 0 <= y0 < y1 <= 720
        assert contains(crop, b)


def test_small_regions_get_context_and_zoom_is_rate_limited():
    config = CameraConfig()
    boxes = [(600, 300, 640, 340)] * 60 + [(500, 200, 800, 500)] * 60
    crops = plan_crops(boxes, 1920, 1080, 30.0, config)
    first = crops[0].size[0]
    assert first >= min(config.context, 40 * config.max_context_ratio)
    sides = np.array([c.size[0] for c in crops], dtype=float)
    step = np.abs(np.diff(np.log(sides)))
    limit = math.log(config.max_zoom_per_second) / 30.0
    # ceil() of a rate-limited smooth size moves by at most one extra pixel per frame.
    assert (step <= limit + np.log1p(1 / sides[1:]) + 1e-9).all()


def test_view_is_letterboxed_with_reflection_at_the_frame_edge():
    image = np.random.default_rng(0).integers(0, 256, size=(100, 400, 3), dtype=np.uint8)
    crop = CropFrame((0, 0, 400, 100), 64)
    canvas = crop_to_canvas(image, crop)
    assert canvas.shape == (64, 64, 3)
    left, top, right, bottom = crop.pad
    assert (left, right) == (0, 0) and top + bottom == 64 - crop.resized[1]
    # reflect-101: the row above the content mirrors the row below its first row
    np.testing.assert_array_equal(canvas[top - 1], canvas[top + 1])


def test_view_at_canvas_size_is_copied_exactly():
    image = np.random.default_rng(1).integers(0, 256, size=(600, 800, 3), dtype=np.uint8)
    crop = CropFrame((100, 50, 612, 562), 512)
    np.testing.assert_array_equal(crop_to_canvas(image, crop), image[50:562, 100:612])
    shifted = crop.shifted(100, 50)
    assert shifted.box == (0, 0, 512, 512)


def test_gaussian_smooth_keeps_constants_and_length():
    values = np.full(50, 7.0)
    np.testing.assert_allclose(gaussian_smooth(values, 3.0), values)
    assert len(gaussian_smooth(np.arange(10.0), 2.0)) == 10
