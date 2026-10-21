"""Synthetic censoring that looks like real releases: pixelation mosaics, sometimes blur.

A mosaic is a grid of solid blocks. The grid is anchored in frame coordinates (or moves
with the region when ``moving_grid``), every block touching the dilated region is filled,
and a block's colour is the mean of its pixels (or its centre pixel). One style is drawn
per training window from the ``MosaicConfig`` ranges.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MosaicConfig:
    block_fraction: tuple[float, float] = (0.007, 0.02)  # block side over the frame's long side
    min_block: int = 4
    dilate_blocks: tuple[float, float] = (0.0, 1.0)  # region growth before snapping to blocks
    center_sample_probability: float = 0.2
    moving_grid_probability: float = 0.3
    ellipse_probability: float = 0.5  # box regions drawn as their inscribed ellipse
    blur_probability: float = 0.05
    blur_sigma_fraction: tuple[float, float] = (0.004, 0.012)  # of the frame's long side


@dataclass(frozen=True)
class MosaicStyle:
    kind: str  # "pixelate" or "blur"
    block: int  # block side in source pixels
    grid_origin: tuple[int, int]  # a block corner (x, y) in frame coordinates
    moving_grid: bool  # the grid follows the region's top-left corner instead
    center_sample: bool  # block colour from its centre pixel instead of the mean
    dilate: int  # region growth in source pixels
    ellipse: bool
    blur_sigma: float


def sample_style(
    rng: np.random.Generator, frame_w: int, frame_h: int, config: MosaicConfig = MosaicConfig()
) -> MosaicStyle:
    long_side = max(frame_w, frame_h)
    block = max(config.min_block, round(long_side * rng.uniform(*config.block_fraction)))
    return MosaicStyle(
        kind="blur" if rng.random() < config.blur_probability else "pixelate",
        block=block,
        grid_origin=(int(rng.integers(block)), int(rng.integers(block))),
        moving_grid=bool(rng.random() < config.moving_grid_probability),
        center_sample=bool(rng.random() < config.center_sample_probability),
        dilate=round(block * rng.uniform(*config.dilate_blocks)),
        ellipse=bool(rng.random() < config.ellipse_probability),
        blur_sigma=float(long_side * rng.uniform(*config.blur_sigma_fraction)),
    )


def grow(mask: np.ndarray, radius: int) -> np.ndarray:
    """``mask`` dilated by an elliptical kernel of ``radius`` pixels."""
    if radius <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def _bounds(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows = np.flatnonzero(mask.any(axis=1))
    if not len(rows):
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def pixelate(
    image: np.ndarray, mask: np.ndarray, block: int, grid_origin: tuple[int, int], *, center_sample: bool = False
) -> np.ndarray:
    """``image`` (H, W, 3) uint8 with every grid block that touches ``mask`` filled with one
    colour. Block corners sit at ``grid_origin`` (x, y) plus multiples of ``block``, in image
    coordinates; blocks cut by the image border use only their pixels inside it."""
    bounds = _bounds(mask)
    if bounds is None:
        return image
    height, width = mask.shape
    gx, gy = grid_origin[0] % block, grid_origin[1] % block
    # The part of the grid that covers the mask, in whole blocks, clipped to the image.
    x0 = gx + (bounds[0] - gx) // block * block
    y0 = gy + (bounds[1] - gy) // block * block
    x1 = gx - (-(bounds[2] - gx) // block) * block
    y1 = gy - (-(bounds[3] - gy) // block) * block
    rows, cols = (y1 - y0) // block, (x1 - x0) // block
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)

    pixels = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.float32)
    valid = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
    hit = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    inner = (slice(cy0 - y0, cy1 - y0), slice(cx0 - x0, cx1 - x0))
    pixels[inner] = image[cy0:cy1, cx0:cx1]
    valid[inner] = 1.0
    hit[inner] = mask[cy0:cy1, cx0:cx1]

    hit_blocks = hit.reshape(rows, block, cols, block).any(axis=(1, 3))
    if center_sample:
        # The centre pixel of each block, moved inside the image for border blocks.
        ys = np.clip(y0 + np.arange(rows) * block + block // 2, cy0, cy1 - 1)
        xs = np.clip(x0 + np.arange(cols) * block + block // 2, cx0, cx1 - 1)
        colours = image[ys[:, None], xs[None, :]].astype(np.float32)
    else:
        sums = pixels.reshape(rows, block, cols, block, 3).sum(axis=(1, 3))
        counts = valid.reshape(rows, block, cols, block).sum(axis=(1, 3))
        colours = sums / np.maximum(counts, 1.0)[..., None]
    filled = np.repeat(np.repeat(np.rint(colours).astype(np.uint8), block, axis=0), block, axis=1)
    where = np.repeat(np.repeat(hit_blocks, block, axis=0), block, axis=1)

    out = image.copy()
    region = out[cy0:cy1, cx0:cx1]
    take = where[inner]
    region[take] = filled[inner][take]
    return out


def blur(image: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """``image`` blurred inside ``mask``, with a short soft edge."""
    bounds = _bounds(mask)
    if bounds is None:
        return image
    height, width = mask.shape
    reach = int(3 * sigma) + 2
    x0, y0 = max(bounds[0] - reach, 0), max(bounds[1] - reach, 0)
    x1, y1 = min(bounds[2] + reach, width), min(bounds[3] + reach, height)
    patch = image[y0:y1, x0:x1].astype(np.float32)
    blurred = cv2.GaussianBlur(patch, (0, 0), sigmaX=sigma)
    alpha = mask[y0:y1, x0:x1].astype(np.float32)
    alpha = np.maximum(cv2.GaussianBlur(alpha, (0, 0), sigmaX=max(1.0, sigma / 4)), alpha)[..., None]
    out = image.copy()
    out[y0:y1, x0:x1] = np.rint(patch * (1 - alpha) + blurred * alpha).clip(0, 255).astype(np.uint8)
    return out


def censor(image: np.ndarray, mask: np.ndarray, style: MosaicStyle, grid_origin: tuple[int, int]) -> np.ndarray:
    """``image`` censored inside ``mask`` in ``style``; ``grid_origin`` is a block corner in
    image coordinates (the caller resolves ``style.grid_origin`` or the moving grid)."""
    mask = grow(mask, style.dilate)
    if style.kind == "blur":
        return blur(image, mask, style.blur_sigma)
    return pixelate(image, mask, style.block, grid_origin, center_sample=style.center_sample)
