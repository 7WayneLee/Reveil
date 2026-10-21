"""The crop camera: a smoothed, slowly zooming view that follows one region track.

Training pairs and restoration both frame windows with this module, so the model always
sees the framing it was trained on. A small region gets extra context (up to ``context``
source pixels, at most ``max_context_ratio`` times its size); the view size holds its
recent maximum, zooms at most ``max_zoom_per_second``, and is Gaussian-smoothed like the
view centre. Each view is letterboxed into a square canvas with reflect padding.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from reveil.regions import Box


@dataclass(frozen=True)
class CameraConfig:
    canvas: int = 512  # side of the square model input, in pixels
    margin: float = 0.25  # free space on each side of the region, as a fraction of its long side
    context: int = 512  # small regions get this many source pixels of view ...
    max_context_ratio: float = 3.0  # ... but at most this multiple of their long side
    smooth_seconds: float = 0.5  # Gaussian sigma of the view centre and size
    hold_seconds: float = 1.0  # the view size never drops below its max over this span
    max_zoom_per_second: float = 1.1


@dataclass(frozen=True)
class CropFrame:
    """One frame's view: a source-pixel box letterboxed into a ``canvas`` square."""

    box: tuple[int, int, int, int]  # x0, y0, x1, y1 in source pixels, half-open
    canvas: int

    @property
    def size(self) -> tuple[int, int]:
        x0, y0, x1, y1 = self.box
        return x1 - x0, y1 - y0

    @property
    def resized(self) -> tuple[int, int]:
        """(w, h) of the view once scaled onto the canvas."""
        w, h = self.size
        scale = self.canvas / max(w, h)
        return min(self.canvas, max(1, round(w * scale))), min(self.canvas, max(1, round(h * scale)))

    @property
    def pad(self) -> tuple[int, int, int, int]:
        """Canvas padding (left, top, right, bottom) around the scaled view."""
        w, h = self.resized
        left, top = (self.canvas - w) // 2, (self.canvas - h) // 2
        return left, top, self.canvas - w - left, self.canvas - h - top

    def shifted(self, dx: int, dy: int) -> CropFrame:
        """The same view in coordinates whose origin is source pixel (dx, dy)."""
        x0, y0, x1, y1 = self.box
        return CropFrame((x0 - dx, y0 - dy, x1 - dx, y1 - dy), self.canvas)


def gaussian_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    """Zero-phase Gaussian smoothing with edge replication."""
    values = np.asarray(values, dtype=np.float64)
    if sigma <= 0 or len(values) < 2:
        return values.copy()
    radius = max(1, math.ceil(3 * sigma))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    padded = np.pad(values, radius, mode="edge")
    return np.convolve(padded, kernel / kernel.sum(), mode="valid")


def _running_max(values: np.ndarray, radius: int) -> np.ndarray:
    padded = np.pad(values, radius, mode="edge")
    return np.lib.stride_tricks.sliding_window_view(padded, 2 * radius + 1).max(axis=1)


def _rate_limit(values: np.ndarray, max_step: float) -> np.ndarray:
    """Raise values until neighbours differ by at most ``max_step`` (both directions)."""
    out = values.copy()
    for i in range(1, len(out)):
        out[i] = max(out[i], out[i - 1] - max_step)
    for i in range(len(out) - 2, -1, -1):
        out[i] = max(out[i], out[i + 1] - max_step)
    return out


def _place(center: float, size: int, low: float, high: float, extent: int) -> int:
    """Start of a ``size`` span around ``center`` that keeps [low, high) inside and stays
    within [0, extent)."""
    start = round(center - size / 2)
    start = min(start, math.floor(low))
    start = max(start, math.ceil(high) - size)
    return int(min(max(start, 0), extent - size))


def plan_crops(
    boxes: Sequence[Box], frame_w: int, frame_h: int, fps: float, config: CameraConfig = CameraConfig()
) -> list[CropFrame]:
    """The view on every frame of a gap-free track, from its per-frame region boxes
    (already clipped to the frame)."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    long_side = np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]).clip(min=1.0)
    wanted = np.maximum(
        long_side * (1 + 2 * config.margin),
        np.minimum(config.context, long_side * config.max_context_ratio),
    )
    log_side = _running_max(np.log(wanted), max(1, round(config.hold_seconds * fps / 2)))
    log_side = _rate_limit(log_side, math.log(config.max_zoom_per_second) / fps)
    sigma = config.smooth_seconds * fps
    side = np.maximum(np.ceil(np.exp(gaussian_smooth(log_side, sigma))), np.ceil(long_side))
    center_x = gaussian_smooth((b[:, 0] + b[:, 2]) / 2, sigma)
    center_y = gaussian_smooth((b[:, 1] + b[:, 3]) / 2, sigma)
    crops = []
    for i in range(len(b)):
        w, h = int(min(side[i], frame_w)), int(min(side[i], frame_h))
        x0 = _place(center_x[i], w, b[i, 0], b[i, 2], frame_w)
        y0 = _place(center_y[i], h, b[i, 1], b[i, 3], frame_h)
        crops.append(CropFrame((x0, y0, x0 + w, y0 + h), config.canvas))
    return crops


def crop_to_canvas(image: np.ndarray, crop: CropFrame) -> np.ndarray:
    """The ``crop`` view of an (H, W, 3) uint8 image, scaled (area when shrinking, cubic
    when enlarging) and reflect-padded to (canvas, canvas, 3)."""
    x0, y0, x1, y1 = crop.box
    view = image[y0:y1, x0:x1]
    if view.shape[:2] != (y1 - y0, x1 - x0) or min(x0, y0) < 0:
        raise ValueError(f"view {crop.box} is outside the {image.shape[1]}x{image.shape[0]} image")
    w, h = crop.resized
    if (w, h) != (x1 - x0, y1 - y0):
        interpolation = cv2.INTER_AREA if w < x1 - x0 else cv2.INTER_CUBIC
        view = cv2.resize(view, (w, h), interpolation=interpolation)
    left, top, right, bottom = crop.pad
    if left or top or right or bottom:
        mode = "reflect" if min(w, h) > 1 else "edge"
        view = np.pad(view, ((top, bottom), (left, right), (0, 0)), mode=mode)
    return np.ascontiguousarray(view)
