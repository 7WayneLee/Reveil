import numpy as np

from reveil.mosaic import MosaicConfig, MosaicStyle, blur, censor, grow, pixelate, sample_style


def style(**overrides) -> MosaicStyle:
    values = dict(
        kind="pixelate", block=8, grid_origin=(0, 0), moving_grid=False,
        center_sample=False, dilate=0, ellipse=False, blur_sigma=4.0,
    )
    values.update(overrides)
    return MosaicStyle(**values)


def image(seed=0, shape=(64, 96)):
    return np.random.default_rng(seed).integers(0, 256, size=(*shape, 3), dtype=np.uint8)


def test_pixelate_fills_whole_grid_blocks_with_their_mean():
    src = image()
    mask = np.zeros(src.shape[:2], bool)
    mask[20:21, 30:31] = True  # one pixel, inside the block at rows 19..26, cols 27..34
    out = pixelate(src, mask, 8, (3, 3))
    block = out[19:27, 27:35]
    assert (block == block[0, 0]).all()
    np.testing.assert_allclose(block[0, 0], src[19:27, 27:35].reshape(-1, 3).mean(axis=0), atol=0.5)
    untouched = np.ones(src.shape[:2], bool)
    untouched[19:27, 27:35] = False
    np.testing.assert_array_equal(out[untouched], src[untouched])


def test_pixelate_border_blocks_average_only_pixels_inside_the_image():
    src = image(1)
    mask = np.zeros(src.shape[:2], bool)
    mask[0, 0] = True
    out = pixelate(src, mask, 8, (4, 4))  # the corner block spans rows/cols -4..3
    np.testing.assert_allclose(out[0, 0], src[0:4, 0:4].reshape(-1, 3).mean(axis=0), atol=0.5)
    assert (out[0:4, 0:4] == out[0, 0]).all()


def test_center_sample_takes_the_block_centre_pixel():
    src = image(2)
    mask = np.zeros(src.shape[:2], bool)
    mask[10, 10] = True
    out = pixelate(src, mask, 8, (0, 0), center_sample=True)
    assert (out[8:16, 8:16] == src[12, 12]).all()


def test_censor_dilates_before_snapping_and_blur_stays_local():
    src = image(3, (80, 80))
    mask = np.zeros(src.shape[:2], bool)
    mask[40, 40] = True
    plain = censor(src, mask, style(), (0, 0))
    grown = censor(src, mask, style(dilate=9), (0, 0))
    changed = lambda out: int((out != src).any(axis=2).sum())  # noqa: E731
    assert changed(grown) > changed(plain) > 0
    assert grow(mask, 3).sum() > 1
    soft = blur(src, grow(mask, 6), 3.0)
    assert (soft[40, 40] != src[40, 40]).any()
    np.testing.assert_array_equal(soft[:10], src[:10])


def test_sampled_styles_are_seeded_and_in_range():
    a = sample_style(np.random.default_rng(5), 1920, 1080)
    b = sample_style(np.random.default_rng(5), 1920, 1080)
    assert a == b
    config = MosaicConfig()
    for seed in range(50):
        s = sample_style(np.random.default_rng(seed), 640, 360, config)
        assert s.block >= config.min_block and 0 <= s.grid_origin[0] < s.block
        assert s.kind in ("pixelate", "blur")
