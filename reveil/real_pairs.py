"""Build training windows from aligned retail and uncensored releases.

Composite targets preserve the retail pixels outside the real mosaic. Source videos
are decoded once in forward order; active windows retain canvas-sized arrays.
Only leak canvases needed by a future window are spooled to temporary disk.
Composite targets match leak blur and previous-frame mixing to the retail and gate
blend seams. Leak targets use colour fitting only, without blur, mixing or a seam gate.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import sys
import tempfile
from bisect import bisect_left, bisect_right
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import closing
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

# Each title has its own process; small dot products must not spin up BLAS pools.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import cv2
import numpy as np
import yaml

from reveil.camera import (
    CameraConfig, CropFrame, RealCameraConfig, crop_to_canvas,
    plan_tight_crops as plan_crops, real_window_canvas,
)
from reveil.frame_map import Alignment, load_alignment
from reveil.mosaic import grow
from reveil.pairs import DEFAULT_CAPTION, _clipped, _seed_for, _work_area, _write_atomic, rebuild_dataset, safe_stem
from reveil.regions import Box, Region, RegionSet, clip_box, load_regions, rasterize, union_box
from reveil.tracks import build_tracks, window_starts
from reveil.video import VideoInfo, decoded_frames, write_video


@dataclass(frozen=True)
class RealPairConfig:
    window: int = 121
    hop: int = 121
    max_windows_per_title: int | None = None
    fps: float = 24.0
    caption: str = DEFAULT_CAPTION
    output_crf: int = 10
    target: str = "composite"
    mask_grow_blocks: float = 1.0
    feather_blocks: float = 0.25
    overlay_margin: float = 0.02
    colour_sample_frames: int = 8
    match_quality: bool = True
    quality_tolerance: float = 0.03
    blur_sigmas: tuple[float, ...] = (0, 0.2, 0.4)
    prev_mixes: tuple[float, ...] = (0, 0.1, 0.2, 0.3, 0.4)
    ring_blocks: float = 1.0
    min_ring_pixels: int = 256
    ring_max_residual: float | None = 6.0
    candidate_factor: float = 1.5
    redetect: bool = True
    redetect_min_diff: float = 12
    redetect_max_std: float = 3
    camera: RealCameraConfig = field(default_factory=RealCameraConfig)

    def __post_init__(self) -> None:
        # Existing callers may still supply the shared synthetic camera config.
        if isinstance(self.camera, CameraConfig) and not isinstance(self.camera, RealCameraConfig):
            object.__setattr__(self, "camera", RealCameraConfig(**asdict(self.camera)))
        if self.target not in {"composite", "leak"}:
            raise ValueError("target must be 'composite' or 'leak'")
        if self.window < 1 or self.hop < 1 or self.fps <= 0 or self.colour_sample_frames < 1:
            raise ValueError("window, hop, fps and colour_sample_frames must be positive")
        if self.max_windows_per_title is not None and self.max_windows_per_title < 0:
            raise ValueError("max_windows_per_title must be non-negative")
        if min(self.mask_grow_blocks, self.feather_blocks, self.overlay_margin) < 0:
            raise ValueError("mask growth, feathering and overlay margin must be non-negative")
        if not self.blur_sigmas or any(not math.isfinite(s) or s < 0 for s in self.blur_sigmas):
            raise ValueError("blur_sigmas must be non-empty, finite and non-negative")
        if not self.prev_mixes or any(not math.isfinite(w) or not 0 <= w <= 1 for w in self.prev_mixes):
            raise ValueError("prev_mixes must be non-empty, finite and within [0, 1]")
        if not math.isfinite(self.quality_tolerance) or self.quality_tolerance < 0:
            raise ValueError("quality_tolerance must be finite and non-negative")
        if not math.isfinite(self.ring_blocks) or self.ring_blocks < 0 or self.min_ring_pixels < 1:
            raise ValueError("ring_blocks must be non-negative and min_ring_pixels positive")
        if self.ring_max_residual is not None and (not math.isfinite(self.ring_max_residual)
                                                   or self.ring_max_residual < 0):
            raise ValueError("ring_max_residual must be None or finite and non-negative")
        if not math.isfinite(self.candidate_factor) or self.candidate_factor <= 0:
            raise ValueError("candidate_factor must be finite and positive")
        if any(not math.isfinite(v) or v < 0 for v in (self.redetect_min_diff, self.redetect_max_std)):
            raise ValueError("redetect thresholds must be finite and non-negative")
        if min(self.camera.canvas, self.camera.large_canvas) < 1 or any(
            not math.isfinite(v) or v < 0 for v in
            (self.camera.border_fraction, self.camera.min_border, self.camera.large_min_side)
        ):
            raise ValueError("camera canvases must be positive and borders and threshold non-negative")


@dataclass
class RealWindowPlan:
    key: str
    track: int
    start: int
    boxes: list[Box]
    regions: list[list[Region]]
    crops: list[CropFrame]
    work: tuple[int, int, int, int]
    leak_indices: tuple[int, ...]
    previous_leak_index: int | None = None
    priority: float = 0.0


def _block(region: Region, long_side: int) -> float:
    value = region.block
    return float(value) if value is not None and float(value) > 0 else 0.015 * long_side


def _masks(
    regions: list[Region], work: tuple[int, int, int, int], long_side: int, config: RealPairConfig,
    *, feather: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """The grown exclusion mask and union of individually feathered region masks.

    A halo lets regions outside a work boundary grow into it, without introducing
    an artificial blend edge at that boundary. Block sizes can differ per region.
    """
    x0, y0, x1, y1 = work
    shape = (y1 - y0, x1 - x0)
    grown = np.zeros(shape, bool)
    alpha = np.zeros(shape, np.float32)
    for region in regions:
        block = _block(region, long_side)
        radius = round(config.mask_grow_blocks * block)
        sigma = config.feather_blocks * block if feather else 0
        halo = radius + math.ceil(4 * sigma) + 1
        mask = rasterize([region], (x0 - halo, y0 - halo),
                         (shape[0] + 2 * halo, shape[1] + 2 * halo), ellipse=False)
        mask = grow(mask, radius)
        inner = (slice(halo, halo + shape[0]), slice(halo, halo + shape[1]))
        grown |= mask[inner]
        if feather:
            soft = mask.astype(np.float32)
            if sigma > 0:
                soft = cv2.GaussianBlur(soft, (0, 0), sigmaX=sigma)
            np.maximum(alpha, soft[inner], out=alpha)
    return grown, alpha.clip(0, 1)


def _overlay_slice(box: Box, work: tuple[int, int, int, int], margin: float) -> tuple[slice, slice] | None:
    x0, y0, x1, y1 = work
    left, top = max(x0, math.floor(box[0] - margin)), max(y0, math.floor(box[1] - margin))
    right, bottom = min(x1, math.ceil(box[2] + margin)), min(y1, math.ceil(box[3] + margin))
    if left >= right or top >= bottom:
        return None
    return slice(top - y0, bottom - y0), slice(left - x0, right - x0)


def _ring_mask(regions: list[Region], work: tuple[int, int, int, int], long_side: int,
               config: RealPairConfig) -> np.ndarray:
    """Union of per-region rings, grown in work pixels with the mask's block sizes."""
    x0, y0, x1, y1 = work
    shape = (y1 - y0, x1 - x0)
    outer = np.zeros(shape, bool)
    for region in regions:
        block = _block(region, long_side)
        radius, ring_radius = round(config.mask_grow_blocks * block), round(config.ring_blocks * block)
        if ring_radius == 0:
            continue
        halo = radius + ring_radius + 1
        mask = rasterize([region], (x0 - halo, y0 - halo),
                         (shape[0] + 2 * halo, shape[1] + 2 * halo), ellipse=False)
        mask = grow(grow(mask, radius), ring_radius)
        outer |= mask[halo:halo + shape[0], halo:halo + shape[1]]
    return outer


def _exclude_border(mask: np.ndarray) -> None:
    mask[:4] = mask[-4:] = True
    mask[:, :4] = mask[:, -4:] = True


def plan_real_windows(
    alignment: Alignment, regions: RegionSet, title: str, config: RealPairConfig, rng: np.random.Generator,
) -> list[RealWindowPlan]:
    """Track windows with complete alignment and no visible overlay mismatch."""
    width, height = alignment.retail_size
    info = VideoInfo(width, height, alignment.retail_fps, alignment.retail_frames)
    clipped = _clipped(regions, info)
    # _clipped intentionally constructs base Regions; restore the optional block
    # metadata from the corresponding originals, keeping its clipping behavior.
    for index, items in clipped.frames.items():
        originals = [r for r in regions.frames[index] if clip_box(r.box, width, height)]
        clipped.frames[index] = [replace(original, box=r.box) for original, r in zip(originals, items)]
    plans = []
    long_side = max(width, height)
    margin = config.overlay_margin * long_side
    analysed = sorted(clipped.frames)
    cuts = sorted(clipped.cuts)

    def all_regions(index: int, own: list[Region]) -> list[Region]:
        if index in clipped.frames:
            return clipped.frames[index]
        pos = bisect_left(analysed, index)
        scene = bisect_right(cuts, index)
        neighbors = [i for i in analysed[max(0, pos - 1):pos + 1] if bisect_right(cuts, i) == scene]
        nearest = min(neighbors, key=lambda i: (abs(i - index), i)) if neighbors else None
        items = clipped.frames[nearest] if nearest is not None else []
        return items + [r for r in own if all(r is not other for other in items)]

    for number, track in enumerate(build_tracks(clipped)):
        length = len(track.boxes)
        if length < config.window:
            continue
        boxes = [union_box(items) for items in track.regions]
        crops = plan_crops(boxes, width, height, info.fps, config.camera)
        offset = int(rng.integers(0, min(config.hop, length - config.window) + 1))
        for first in window_starts(length, config.window, config.hop, offset):
            start, stop = track.start + first, first + config.window
            if not alignment.window_ok(start, config.window):
                continue
            window_crops = real_window_canvas(crops[first:stop], width, height, config.camera)
            # Include all real mosaics visible in this work area, including those
            # belonging to a neighboring track, in both fitting and compositing.
            window_regions = [all_regions(i, track.regions[first + i - start])
                              for i in range(start, start + config.window)]
            rejected = False
            for offset_in_window, crop in enumerate(window_crops):
                active = [overlay for overlay in regions.overlays
                          if overlay.active(start + offset_in_window)
                          and _overlay_slice(overlay.box, crop.box, margin) is not None]
                if not active:
                    continue
                if config.target == "leak":
                    rejected = True
                    break
                grown, _ = _masks(window_regions[offset_in_window], crop.box, long_side, config, feather=False)
                if any(grown[_overlay_slice(overlay.box, crop.box, margin)].any() for overlay in active):
                    rejected = True
                    break
            if not rejected:
                plans.append(RealWindowPlan(
                    key=f"{safe_stem(title)}-t{number:03d}-f{start:06d}", track=number, start=start,
                    boxes=boxes[first:stop], regions=window_regions, crops=window_crops,
                    work=_work_area(window_crops, info),
                    leak_indices=tuple(int(i) for i in alignment.leak_indices(start, start + config.window)),
                    previous_leak_index=(alignment.leak_index(start - 1)
                                         if config.target == "composite" and config.match_quality else None),
                ))
    if config.max_windows_per_title is not None:
        cap = config.max_windows_per_title
        candidates = max(cap + 1, math.ceil(config.candidate_factor * cap))
        if len(plans) > candidates:
            keep = rng.choice(len(plans), candidates, replace=False)
            plans = [plans[i] for i in sorted(keep)]
    for plan in plans:
        plan.priority = float(rng.random())
    return sorted(plans, key=lambda plan: (plan.start, plan.track))


def _fit_colour(leak: np.ndarray, retail: np.ndarray) -> dict:
    """Fit retail ~= a*leak+b independently per RGB channel, trimming twice."""
    return _fit_colour_scored(leak, retail)[0]


def _fit_colour_scored(leak: np.ndarray, retail: np.ndarray) -> tuple[dict, float]:
    """The same fit, also scoring only the samples retained in each channel."""
    gains, offsets, residuals = [], [], []
    for channel in range(3):
        x, y = leak[:, channel].astype(np.float64), retail[:, channel].astype(np.float64)
        a, b = 1.0, 0.0
        for iteration in range(3):
            if len(x) < 2:
                break
            dx = x - x.mean()
            variance = np.dot(dx, dx)
            a = float(np.dot(dx, y - y.mean()) / variance) if variance > 1e-8 else 1.0
            b = float(y.mean() - a * x.mean())
            if iteration < 2:
                residual = np.abs(y - (a * x + b))
                keep = residual <= np.quantile(residual, 0.8)
                x, y = x[keep], y[keep]
        gains.append(a)
        offsets.append(b)
        residuals.append(np.abs(y - (a * x + b)))
    score = float(np.concatenate(residuals).mean())
    return {"a": gains, "b": offsets, "samples": len(leak)}, score


def _blur(image: np.ndarray, sigma: float) -> np.ndarray:
    image = image.astype(np.float32)
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma) if sigma else image


def _ring_rejected(ring: dict, config: RealPairConfig) -> bool:
    return bool(config.target == "composite" and config.ring_max_residual
                and ring["score"] is not None and ring["score"] > config.ring_max_residual)


def _select_quality(scores: dict[tuple[float, float], float], tolerance: float) -> tuple[float, float]:
    """Prefer less mixing, then less blur, within a fraction of the best score."""
    limit = min(scores.values()) * (1 + tolerance)
    return min((key for key, score in scores.items() if score <= limit), key=lambda key: (key[1], key[0]))


def _mask_to_canvas(mask: np.ndarray, crop: CropFrame) -> np.ndarray:
    image = np.repeat(mask.astype(np.float32)[..., None], 3, axis=2)
    return crop_to_canvas(image, crop)[..., 0].copy()


def _redetect(retail: np.ndarray, matched: np.ndarray, block: float, excluded: np.ndarray,
              config: RealPairConfig) -> np.ndarray:
    """Flat retail pixels differing from the fitted leak, cleaned at block scale."""
    r = cv2.cvtColor(retail.astype(np.float32), cv2.COLOR_RGB2GRAY)
    d = cv2.cvtColor(matched.astype(np.float32), cv2.COLOR_RGB2GRAY)
    size = max(3, round(block / 2))
    mean = cv2.boxFilter(r, -1, (size, size))
    variance = cv2.boxFilter(r * r, -1, (size, size)) - mean * mean
    std = np.sqrt(np.maximum(variance, 0))
    difference = np.abs(r - d)
    difference[excluded] = 0  # excluded overlays/borders cannot seed neighboring evidence
    difference = cv2.boxFilter(difference, -1, (size, size))
    evidence = (difference > config.redetect_min_diff) & (std < config.redetect_max_std) & ~excluded
    count, labels, stats, _ = cv2.connectedComponentsWithStats(evidence.astype(np.uint8), connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= max(1, math.ceil(block * block))
    keep[0] = False
    if count == 1 or not keep.any():
        return np.zeros(evidence.shape, bool)
    return grow(keep[labels], max(1, round(block))) & ~excluded


class _WindowBuffer:
    """Active canvas buffers; future leak shots use a lossless canvas spool."""

    def __init__(self, plan: RealWindowPlan, temp_dir: Path, config: RealPairConfig) -> None:
        self.plan, self.temp_dir = plan, temp_dir
        self.active = False
        count = len(plan.crops)
        self.retail: list[np.ndarray | None] = [None] * count
        self.leak: list[np.ndarray | None] = [None] * count
        self.previous: list[np.ndarray | None] = [None] * count
        self.alpha: list[np.ndarray | None] = [None] * count
        self.region_masks: dict[int, np.ndarray] = {}
        self.exclusions: dict[int, np.ndarray] = {}
        self.ring_pixels: dict[int, np.ndarray] = {}
        self.sample_masks: dict[int, np.ndarray] = {}
        self.sample_indices = tuple(np.linspace(0, count - 1, min(config.colour_sample_frames, count), dtype=int))
        self.spooled: dict[tuple[str, int], Path] = {}
        self.retail_count = 0
        self.leak_count = 0

    def activate(self) -> None:
        self.active = True
        for (side, index), path in self.spooled.items():
            getattr(self, side)[index] = np.load(path, allow_pickle=False)
            path.unlink()
        self.spooled.clear()

    def add(self, side: str, index: int, frame: np.ndarray, alignment: Alignment,
            regions: RegionSet, config: RealPairConfig) -> None:
        x0, y0, x1, y1 = self.plan.work
        work = frame[y0:y1, x0:x1]
        if work.shape != (y1 - y0, x1 - x0, 3):
            raise RuntimeError(f"{self.plan.key}: {side} dimensions do not match the alignment")
        crop = self.plan.crops[index].shifted(x0, y0)
        canvas = crop_to_canvas(work, crop).copy()
        if side in {"leak", "previous"}:
            if self.active:
                getattr(self, side)[index] = canvas
            else:
                path = self.temp_dir / f"{self.plan.key}-{side}-{index:04d}.npy"
                np.save(path, canvas, allow_pickle=False)
                self.spooled[side, index] = path
            if side == "leak":
                self.leak_count += 1
            return
        self.retail[index] = canvas
        self.retail_count += 1
        sampled = index in self.sample_indices
        if config.target == "composite" or sampled:
            grown, alpha = _masks(self.plan.regions[index], self.plan.work, max(alignment.retail_size),
                                  config, feather=config.target == "composite")
            if config.target == "composite":
                self.alpha[index] = _mask_to_canvas(alpha, crop).clip(0, 1)
                if config.redetect:
                    self.region_masks[index] = _mask_to_canvas(grown, crop) > 1e-6
            overlays = np.zeros(grown.shape, bool)
            for overlay in regions.overlays:
                if overlay.active(self.plan.start + index):
                    area = _overlay_slice(overlay.box, self.plan.work,
                                          config.overlay_margin * max(alignment.retail_size))
                    if area is not None:
                        grown[area] = True
                        overlays[area] = True
            excluded = _mask_to_canvas(grown, crop) > 1e-6
            if sampled:
                self.sample_masks[index] = excluded
            if config.target == "composite":
                if config.redetect:
                    self.exclusions[index] = _mask_to_canvas(overlays, crop) > 1e-6
                    _exclude_border(self.exclusions[index])
                ring = _ring_mask(self.plan.regions[index], self.plan.work,
                                  max(alignment.retail_size), config)
                ring = _mask_to_canvas(ring & ~grown, crop) > 1e-6
                excluded = excluded.copy()
                _exclude_border(excluded)
                pixels = np.flatnonzero(ring & ~excluded)
                self.ring_pixels[index] = pixels

    def _samples(self, *, border: bool = False) -> tuple[dict[int, np.ndarray], np.ndarray]:
        samples, ys = {}, []
        for index in self.sample_indices:
            excluded = self.sample_masks[index].copy()
            if border:
                _exclude_border(excluded)
            pixels = np.flatnonzero(~excluded)
            # Bound regression memory on large work areas; deterministic spatial
            # samples cover every selected frame, with at most 32K pixels each.
            if len(pixels) > 32768:
                pixels = pixels[np.linspace(0, len(pixels) - 1, 32768, dtype=int)]
            samples[index] = pixels
            ys.append(self.retail[index].reshape(-1, 3)[pixels])
        y = np.concatenate(ys)
        if len(y) < 2:
            raise ValueError(f"{self.plan.key}: no uncensored, overlay-free pixels for colour fitting")
        return samples, y

    def colour_fit(self) -> dict:
        samples, y = self._samples()
        x = np.concatenate([self.leak[index].reshape(-1, 3)[pixels] for index, pixels in samples.items()])
        return _fit_colour(x, y)

    def _previous(self, index: int) -> np.ndarray | None:
        if index and self.plan.crops[index - 1] == self.plan.crops[index]:
            return self.leak[index - 1]
        return self.previous[index]

    def quality_fit(self, config: RealPairConfig) -> tuple[dict, dict]:
        enabled = config.target == "composite" and config.match_quality
        samples, y = self._samples(border=enabled)
        fits, scores = {}, {}
        identity = None
        sigmas = sorted(set(config.blur_sigmas) | {0}) if enabled else [0]
        mixes = sorted(set(config.prev_mixes) | {0}) if enabled else [0]
        for sigma in sigmas:
            current, previous = [], []
            for index, pixels in samples.items():
                image = _blur(self.leak[index], sigma)
                current.append(image.reshape(-1, 3)[pixels])
                prior = self._previous(index) if enabled else None
                image = _blur(prior, sigma) if prior is not None else image
                previous.append(image.reshape(-1, 3)[pixels])
            x, p = np.concatenate(current), np.concatenate(previous)
            for mix in mixes:
                candidate = (1 - mix) * x + mix * p if mix else x
                fit, score = _fit_colour_scored(candidate, y)
                if sigma == 0 and mix == 0:
                    identity = score
                if enabled and (sigma not in config.blur_sigmas or mix not in config.prev_mixes):
                    continue
                key = float(sigma), float(mix)
                fits[key], scores[key] = fit, score
        sigma, mix = _select_quality(scores, config.quality_tolerance)
        return fits[sigma, mix], {"sigma": sigma, "prev_mix": mix,
                                 "residual": scores[sigma, mix], "identity_residual": identity}

    def _matched(self, index: int, quality: dict, gain: np.ndarray, offset: np.ndarray) -> np.ndarray:
        image = _blur(self.leak[index], quality["sigma"])
        previous = self._previous(index)
        mix = quality["prev_mix"]
        if mix and previous is not None:
            image *= 1 - mix
            image += mix * _blur(previous, quality["sigma"])
        image *= gain
        image += offset
        np.clip(image, 0, 255, out=image)
        return image

    def ring_score(self, quality: dict, fit: dict, alignment: Alignment,
                   config: RealPairConfig) -> tuple[dict, dict]:
        residuals = []
        shares = []
        gain, offset = np.asarray(fit["a"], np.float32), np.asarray(fit["b"], np.float32)
        blocks = [_block(r, max(alignment.retail_size)) for items in self.plan.regions for r in items]
        source_block = float(np.median(blocks)) if blocks else 0.015 * max(alignment.retail_size)
        indices = range(len(self.plan.crops)) if config.target == "composite" else ()
        for index in indices:
            image = self._matched(index, quality, gain, offset)
            pixels = self.ring_pixels.get(index, np.empty(0, dtype=int))
            share = 0.0
            if config.redetect:
                crop = self.plan.crops[index]
                block = source_block * crop.canvas / max(crop.size)
                region_mask, excluded = self.region_masks[index], self.exclusions[index]
                detected = _redetect(self.retail[index], image, block, excluded, config)
                added = detected & ~region_mask
                share = float(added.mean())
                if added.any():
                    final_mask = region_mask | detected
                    soft = final_mask.astype(np.float32)
                    sigma = config.feather_blocks * block
                    if sigma > 0:
                        soft = cv2.GaussianBlur(soft, (0, 0), sigmaX=sigma)
                    soft[excluded] = 0
                    # Preserve individual region feather widths outside new areas.
                    self.alpha[index] = np.maximum(self.alpha[index], soft).clip(0, 1)
                    ring = np.zeros(region_mask.shape, bool)
                    ring.flat[pixels] = True
                    ring |= grow(added, round(config.ring_blocks * block))
                    pixels = np.flatnonzero(ring & ~final_mask & ~excluded)
            shares.append(share)
            if len(pixels) >= config.min_ring_pixels:
                retail = self.retail[index].reshape(-1, 3)[pixels]
                matched = image.reshape(-1, 3)[pixels]
                residuals.append(float(np.abs(retail - matched).mean()))
        return {"score": float(np.percentile(residuals, 90)) if residuals else None,
                "median": float(np.median(residuals)) if residuals else None,
                "max": max(residuals) if residuals else None, "frames": len(residuals)}, {
                    "added_share_mean": float(np.mean(shares)) if shares else 0.0,
                    "added_share_max": max(shares, default=0.0),
                }

    def finish(self, title: str, alignment: Alignment, out_dir: Path, config: RealPairConfig) -> dict:
        plan = self.plan
        fit, quality = self.quality_fit(config)
        ring, redetect = self.ring_score(quality, fit, alignment, config)
        gain, offset = np.asarray(fit["a"], np.float32), np.asarray(fit["b"], np.float32)

        def canvases(target: bool):
            for index in range(len(plan.crops)):
                if target:
                    image = self._matched(index, quality, gain, offset)
                    if config.target == "composite":
                        alpha = self.alpha[index]
                        retail = self.retail[index]
                        image = retail * (1 - alpha[..., None]) + image * alpha[..., None]
                    image = np.rint(image).clip(0, 255).astype(np.uint8)
                else:
                    image = self.retail[index]
                yield image

        rejected = _ring_rejected(ring, config)
        for sub, target in (("clean", True), ("mosaic", False)):
            path = out_dir / sub / f"{plan.key}.mp4"
            if rejected:
                path.unlink(missing_ok=True)
            else:
                write_video(path, canvases(target), config.fps, crf=config.output_crf)
        self.retail.clear()
        self.leak.clear()
        self.previous.clear()
        self.alpha.clear()
        self.sample_masks.clear()
        self.ring_pixels.clear()
        self.region_masks.clear()
        self.exclusions.clear()
        return {
            "video": f"clean/{plan.key}.mp4", "reference_video": f"mosaic/{plan.key}.mp4",
            "caption": config.caption, "key": plan.key, "title": title,
            "retail": alignment.retail, "leak": alignment.leak, "track": plan.track,
            "start": plan.start, "leak_start": min(plan.leak_indices), "leak_stop": max(plan.leak_indices) + 1,
            "frames": len(plan.crops), "canvas": plan.crops[0].canvas,
            "work": list(plan.work), "crops": [list(c.box) for c in plan.crops],
            "target": config.target, "colour_fit": fit, "leak_indices": list(plan.leak_indices),
            "quality": quality, "ring": ring, "redetect": redetect,
        }


def _render(title: str, alignment: Alignment, regions: RegionSet, plans: list[RealWindowPlan],
            out_dir: Path, temp_dir: Path, config: RealPairConfig, decode_threads: int = 2) -> list[dict]:
    demands = defaultdict(list)
    for plan in plans:
        for index, leak_index in enumerate(plan.leak_indices):
            demands[leak_index].append((plan, "leak", index))
        if config.target == "composite" and config.match_quality:
            if plan.previous_leak_index is not None:
                demands[plan.previous_leak_index].append((plan, "previous", 0))
            for index in range(1, len(plan.crops)):
                if plan.crops[index - 1] != plan.crops[index]:
                    demands[plan.leak_indices[index - 1]].append((plan, "previous", index))
    buffers: dict[str, _WindowBuffer] = {}

    def buffer(plan: RealWindowPlan) -> _WindowBuffer:
        if plan.key not in buffers:
            buffers[plan.key] = _WindowBuffer(plan, temp_dir, config)
        return buffers[plan.key]

    rows, active = [], []
    upcoming = iter(plans)
    plan = next(upcoming, None)
    leak_position = -1
    last_frame = max(p.start + len(p.crops) for p in plans) - 1
    with closing(decoded_frames(alignment.retail, threads=decode_threads)) as retail_frames, closing(decoded_frames(alignment.leak, threads=decode_threads)) as leak_frames:
        for frame_index, retail in enumerate(retail_frames):
            while plan is not None and plan.start == frame_index:
                render = buffer(plan)
                render.activate()
                active.append(render)
                plan = next(upcoming, None)
            if active:
                retail = retail.to_ndarray(format="rgb24")
                required = max(r.plan.leak_indices[frame_index - r.plan.start] for r in active)
                required = max([required] + [r.plan.previous_leak_index for r in active
                                            if frame_index == r.plan.start
                                            and r.plan.previous_leak_index is not None])
                while leak_position < required:
                    try:
                        leak = next(leak_frames)
                    except StopIteration as error:
                        raise RuntimeError(f"{title}: leak ended before mapped frame {required}") from error
                    leak_position += 1
                    needed = demands.pop(leak_position, [])
                    if needed:
                        leak = leak.to_ndarray(format="rgb24")
                        if (leak.shape[1], leak.shape[0]) != alignment.retail_size:
                            leak = cv2.resize(leak, alignment.retail_size, interpolation=cv2.INTER_AREA)
                        for needed_plan, side, index in needed:
                            buffer(needed_plan).add(side, index, leak, alignment, regions, config)
                for render in active:
                    render.add("retail", frame_index - render.plan.start, retail, alignment, regions, config)
                for render in list(active):
                    if render.retail_count == len(render.plan.crops):
                        if render.leak_count != len(render.plan.crops):
                            raise RuntimeError(f"{render.plan.key}: incomplete leak window")
                        rows.append(render.finish(title, alignment, out_dir, config))
                        active.remove(render)
                        del buffers[render.plan.key]
            if frame_index >= last_frame:
                break
    if len(rows) != len(plans):
        raise RuntimeError(f"{title}: retail ended early ({len(rows)} of {len(plans)} windows)")
    return sorted(rows, key=lambda row: (row["start"], row["track"]))


def _title_stats(rows: list[dict], kept: int, config: RealPairConfig) -> dict:
    rejected = sum(_ring_rejected(row["ring"], config) for row in rows)
    histograms = {}
    for name in ("sigma", "prev_mix"):
        counts = defaultdict(int)
        for row in rows:
            counts[str(row["quality"][name])] += 1
        histograms[f"{name}_histogram"] = dict(sorted(counts.items(), key=lambda item: float(item[0])))
    scores = [row["ring"]["score"] for row in rows if row["ring"]["score"] is not None]
    return {"candidates": len(rows), "rejected_ring": rejected, "accepted": len(rows) - rejected,
            "kept": kept, "dropped_extra": len(rows) - rejected - kept, **histograms,
            "ring_percentiles": {f"p{p}": float(np.percentile(scores, p)) if scores else None
                                 for p in (10, 50, 90)}}


def build_title_pairs(
    title: str, alignment_path: str | Path, regions_path: str | Path, out_dir: str | Path,
    config: RealPairConfig = RealPairConfig(), *, seed: int = 0, overwrite: bool = False, decode_threads: int = 2,
) -> list[dict]:
    """Build one title, resuming from its atomic metadata file when already complete."""
    cv2.setNumThreads(1)
    out_dir = Path(out_dir)
    stem = safe_stem(title)
    meta_path = out_dir / "meta" / f"{stem}.json"
    if meta_path.exists() and not overwrite:
        return json.loads(meta_path.read_text(encoding="utf-8"))["windows"]
    for sub in ("clean", "mosaic", "meta"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    alignment, regions = load_alignment(alignment_path), load_regions(regions_path)
    plans = plan_real_windows(alignment, regions, title, config, np.random.default_rng(_seed_for(seed, stem)))
    rows = []
    if plans:
        with tempfile.TemporaryDirectory(prefix="reveil-real-") as temp_dir:
            rows = _render(title, alignment, regions, plans, out_dir, Path(temp_dir), config, decode_threads)
    accepted = [row for row in rows if not _ring_rejected(row["ring"], config)]
    priorities = {plan.key: plan.priority for plan in plans}
    if config.max_windows_per_title is not None:
        ranked = sorted(accepted, key=lambda row: (priorities[row["key"]], row["key"]))
        accepted = ranked[:config.max_windows_per_title]
        for row in ranked[config.max_windows_per_title:]:
            for name in ("video", "reference_video"):
                (out_dir / row[name]).unlink(missing_ok=True)
    accepted.sort(key=lambda row: (row["start"], row["track"]))
    meta = {"title": title, "retail": alignment.retail, "leak": alignment.leak, "windows": accepted,
            "stats": _title_stats(rows, len(accepted), config)}
    _write_atomic(meta_path, json.dumps(meta))
    return accepted


def load_config(path: Path | None) -> RealPairConfig:
    """Read optional pairs and camera YAML sections, rejecting unknown settings."""
    data = (yaml.safe_load(Path(path).read_text(encoding="utf-8")) if path else None) or {}
    unknown = set(data) - {"pairs", "camera"}
    if unknown:
        raise ValueError(f"{path}: unknown sections {sorted(unknown)}")
    camera = replace(RealCameraConfig(), **(data.get("camera") or {}))
    return replace(RealPairConfig(), **(data.get("pairs") or {}), camera=camera)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reveil-real-pairs", description=__doc__.split("\n\n")[0])
    parser.add_argument("--align-dir", type=Path, required=True)
    parser.add_argument("--regions-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="YAML overrides: pairs / camera sections")
    parser.add_argument("--target", choices=("composite", "leak"))
    parser.add_argument("--max-windows-per-title", type=int)
    parser.add_argument("--no-match-quality", action="store_true", help="disable composite blur and temporal matching")
    parser.add_argument("--ring-max-residual", type=float, help="maximum seam-ring residual in grey levels; 0 disables (default: 6)")
    parser.add_argument("--candidate-factor", type=float, help="candidate windows per capped output window (default: 1.5)")
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--decode-threads", type=int, default=2, help="threads per video decoder (default: 2)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--only", nargs="+", help="title stems to include")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    overrides = {name: getattr(args, name) for name in ("target", "max_windows_per_title", "ring_max_residual", "candidate_factor")
                 if getattr(args, name) is not None}
    if args.no_match_quality:
        overrides["match_quality"] = False
    config = replace(config, **overrides)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.decode_threads < 1:
        parser.error("--decode-threads must be positive")
    jobs, stems = [], {}
    for alignment_path in sorted(args.align_dir.glob("*.json")):
        title = alignment_path.stem
        if args.only and title not in args.only:
            continue
        stem = safe_stem(title)
        if stem in stems:
            parser.error(f"{title!r} and {stems[stem]!r} share the output name {stem!r}")
        stems[stem] = title
        regions_path = args.regions_dir / f"{title}.json"
        if not regions_path.is_file():
            print(f"skip {title}: no {regions_path}", file=sys.stderr)
            continue
        alignment = load_alignment(alignment_path)
        if (args.out / "meta" / f"{stem}.json").exists() and not args.overwrite:
            size = 0
        else:
            size = Path(alignment.retail).stat().st_size + Path(alignment.leak).stat().st_size
        jobs.append((size, title, alignment_path, regions_path))
    if not jobs:
        parser.error("no titles with alignment and region files")
    args.out.mkdir(parents=True, exist_ok=True)
    failed = 0
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {
            pool.submit(build_title_pairs, title, alignment_path, regions_path, args.out, config,
                        seed=args.seed, overwrite=args.overwrite, decode_threads=args.decode_threads): title
            for _, title, alignment_path, regions_path in sorted(jobs, key=lambda job: job[0], reverse=True)
        }
        for done, future in enumerate(as_completed(futures), 1):
            title = futures[future]
            try:
                rows = future.result()
                meta = json.loads((args.out / "meta" / f"{safe_stem(title)}.json").read_text(encoding="utf-8"))
                stats = meta.get("stats")
                summary = (f"; candidates={stats['candidates']}, ring-rejected={stats['rejected_ring']}, "
                           f"accepted={stats['accepted']}, extra={stats['dropped_extra']}, "
                           f"sigma={stats['sigma_histogram']}, prev_mix={stats['prev_mix_histogram']}, "
                           f"ring={stats['ring_percentiles']}" if stats else "")
                print(f"[{done}/{len(jobs)}] {title}: {len(rows)} pairs{summary}", flush=True)
            except Exception as error:  # noqa: BLE001 - finish independent titles
                failed += 1
                print(f"[{done}/{len(jobs)}] {title}: FAILED: {error!r}", file=sys.stderr, flush=True)
    total = rebuild_dataset(args.out, config.caption)
    print(f"{total} pairs in {args.out / 'dataset.jsonl'}" + (f"; {failed} titles failed" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
