"""Region annotations: where a synthetic mosaic goes in each source frame.

A regions file is JSON written by any detector or annotation tool::

    {
      "version": 1,
      "frames": {
        "0": [{"box": [x0, y0, x1, y1]}],
        "1": [{"polygon": [[x, y], [x, y], [x, y]]}]
      },
      "cuts": [240, 1800]
    }

Frame keys are 0-based decode indices and coordinates are source pixels. ``cuts`` is
optional: the first frame after each scene cut, so no track or window spans a cut.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

Box = tuple[float, float, float, float]  # x0, y0, x1, y1 in source pixels, half-open


@dataclass(frozen=True, eq=False)
class Region:
    """One area to censor: its bounds and, when known, its outline."""

    box: Box
    polygon: np.ndarray | None = None  # (N, 2) float32 points in source pixels

    @classmethod
    def from_json(cls, item: dict) -> Region:
        if "polygon" in item:
            points = np.asarray(item["polygon"], dtype=np.float32).reshape(-1, 2)
            if len(points) < 3:
                raise ValueError("a polygon region needs at least 3 points")
            x0, y0 = points.min(axis=0)
            x1, y1 = points.max(axis=0)
            return cls((float(x0), float(y0), float(x1), float(y1)), points)
        x0, y0, x1, y1 = (float(v) for v in item["box"])
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"empty box {item['box']}")
        return cls((x0, y0, x1, y1))

    def to_json(self) -> dict:
        if self.polygon is not None:
            return {"polygon": [[_compact(x), _compact(y)] for x, y in self.polygon.tolist()]}
        return {"box": [_compact(v) for v in self.box]}


def _compact(value: float) -> float | int:
    """Whole pixels as ints, the rest to 0.1 px, to keep long region files small."""
    value = round(float(value), 1)
    return int(value) if value.is_integer() else value


@dataclass
class RegionSet:
    """Regions per frame index, plus the scene cuts no track may cross."""

    frames: dict[int, list[Region]] = field(default_factory=dict)
    cuts: frozenset[int] = frozenset()


def load_regions(path: str | Path) -> RegionSet:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("version", 1) != 1:
        raise ValueError(f"{path}: unsupported regions version {data.get('version')}")
    frames = {
        int(index): [Region.from_json(item) for item in items]
        for index, items in data.get("frames", {}).items()
        if items
    }
    return RegionSet(frames, frozenset(int(cut) for cut in data.get("cuts", [])))


def save_regions(regions: RegionSet, path: str | Path) -> None:
    data = {
        "version": 1,
        "frames": {str(index): [r.to_json() for r in items] for index, items in sorted(regions.frames.items())},
        "cuts": sorted(regions.cuts),
    }
    path = Path(path)
    partial = path.with_name(path.name + ".part")
    partial.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    os.replace(partial, path)


def union_box(regions: Sequence[Region]) -> Box:
    return (
        min(r.box[0] for r in regions),
        min(r.box[1] for r in regions),
        max(r.box[2] for r in regions),
        max(r.box[3] for r in regions),
    )


def clip_box(box: Box, width: int, height: int) -> Box | None:
    """``box`` limited to the frame, or None when nothing of it is inside."""
    x0, y0 = max(0.0, box[0]), max(0.0, box[1])
    x1, y1 = min(float(width), box[2]), min(float(height), box[3])
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return x0, y0, x1, y1


def rasterize(
    regions: Sequence[Region], origin: tuple[int, int], shape: tuple[int, int], *, ellipse: bool
) -> np.ndarray:
    """Bool mask of ``regions`` on an (h, w) grid whose top-left pixel is source pixel
    ``origin`` (x, y). A box region fills its inscribed ellipse when ``ellipse`` is set."""
    mask = np.zeros(shape, dtype=np.uint8)
    ox, oy = origin
    for region in regions:
        if region.polygon is not None:
            points = np.round(region.polygon - np.array([ox, oy], dtype=np.float32)).astype(np.int32)
            # One call per polygon: a single fillPoly call would cut holes where polygons overlap.
            cv2.fillPoly(mask, [points], 1)
            continue
        x0, y0, x1, y1 = region.box
        if ellipse:
            # Pixels whose centres fall inside the ellipse inscribed in the box.
            cols = slice(max(math.floor(x0) - ox, 0), min(max(math.ceil(x1) - ox, 0), shape[1]))
            rows = slice(max(math.floor(y0) - oy, 0), min(max(math.ceil(y1) - oy, 0), shape[0]))
            ys, xs = np.mgrid[rows, cols]
            u = (xs + ox + 0.5 - (x0 + x1) / 2) / ((x1 - x0) / 2)
            v = (ys + oy + 0.5 - (y0 + y1) / 2) / ((y1 - y0) / 2)
            mask[rows, cols] |= (u * u + v * v <= 1).astype(np.uint8)
        else:
            top_left = (math.floor(x0) - ox, math.floor(y0) - oy)
            bottom_right = (math.ceil(x1) - 1 - ox, math.ceil(y1) - 1 - oy)
            cv2.rectangle(mask, top_left, bottom_right, 1, thickness=-1)
    return mask.astype(bool)


def random_regions(frame_count: int, width: int, height: int, rng: np.random.Generator) -> RegionSet:
    """One box drifting smoothly across the frame, for pipeline smoke tests (not training)."""
    short = min(width, height)
    size = rng.uniform(0.12, 0.3) * short
    aspect = rng.uniform(0.7, 1.4)
    w, h = size * aspect, size / aspect
    x, y = rng.uniform(0, width - w), rng.uniform(0, height - h)
    vx, vy = rng.normal(0, short / 400, size=2)
    frames: dict[int, list[Region]] = {}
    for index in range(frame_count):
        vx = 0.98 * vx + rng.normal(0, short / 2000)
        vy = 0.98 * vy + rng.normal(0, short / 2000)
        x, y = x + vx, y + vy
        if not 0 <= x <= width - w:
            vx, x = -vx, min(max(x, 0.0), width - w)
        if not 0 <= y <= height - h:
            vy, y = -vy, min(max(y, 0.0), height - h)
        frames[index] = [Region((x, y, x + w, y + h))]
    return RegionSet(frames)
