"""Build training windows from aligned retail and uncensored releases.

Composite targets preserve the retail pixels outside the real mosaic. Source videos
are decoded once in forward order; active windows retain canvas-sized arrays.
Only leak canvases needed by a future window are spooled to temporary disk.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import closing
from dataclasses import dataclass, field, replace
from pathlib import Path

# Each title has its own process; small dot products must not spin up BLAS pools.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import cv2
import numpy as np
import yaml

from reveil.camera import CameraConfig, CropFrame, crop_to_canvas, plan_crops
from reveil.frame_map import Alignment, load_alignment
from reveil.mosaic import grow
from reveil.pairs import DEFAULT_CAPTION, _clipped, _seed_for, _work_area, _write_atomic, rebuild_dataset, safe_stem
from reveil.regions import Box, Region, RegionSet, clip_box, load_regions, rasterize
from reveil.tracks import build_tracks, window_starts
from reveil.video import VideoInfo, read_frames, write_video


@dataclass(frozen=True)
class RealPairConfig:
    window: int = 121
    hop: int = 121
    max_windows_per_title: int | None = None
    fps: float = 24.0
    caption: str = DEFAULT_CAPTION
    output_crf: int = 10
    target: str = "composite"
    mask_grow_blocks: float = 0.5
    feather_blocks: float = 0.25
    overlay_margin: float = 0.02
    colour_sample_frames: int = 8
    camera: CameraConfig = field(default_factory=CameraConfig)

    def __post_init__(self) -> None:
        if self.target not in {"composite", "leak"}:
            raise ValueError("target must be 'composite' or 'leak'")
        if self.window < 1 or self.hop < 1 or self.fps <= 0 or self.colour_sample_frames < 1:
            raise ValueError("window, hop, fps and colour_sample_frames must be positive")
        if self.max_windows_per_title is not None and self.max_windows_per_title < 0:
            raise ValueError("max_windows_per_title must be non-negative")
        if min(self.mask_grow_blocks, self.feather_blocks, self.overlay_margin) < 0:
            raise ValueError("mask growth, feathering and overlay margin must be non-negative")


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
    for number, track in enumerate(build_tracks(clipped)):
        length = len(track.boxes)
        if length < config.window:
            continue
        crops = plan_crops(track.boxes, width, height, info.fps, config.camera)
        offset = int(rng.integers(0, min(config.hop, length - config.window) + 1))
        for first in window_starts(length, config.window, config.hop, offset):
            start, stop = track.start + first, first + config.window
            if not alignment.window_ok(start, config.window):
                continue
            window_crops = crops[first:stop]
            # Include all real mosaics visible in this work area, including those
            # belonging to a neighboring track, in both fitting and compositing.
            window_regions = [clipped.frames.get(i, track.regions[first + i - start])
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
                    boxes=track.boxes[first:stop], regions=window_regions, crops=window_crops,
                    work=_work_area(window_crops, info),
                    leak_indices=tuple(int(i) for i in alignment.leak_indices(start, start + config.window)),
                ))
    if config.max_windows_per_title is not None and len(plans) > config.max_windows_per_title:
        keep = rng.choice(len(plans), config.max_windows_per_title, replace=False)
        plans = [plans[i] for i in sorted(keep)]
    return sorted(plans, key=lambda plan: (plan.start, plan.track))


def _fit_colour(leak: np.ndarray, retail: np.ndarray) -> dict:
    """Fit retail ~= a*leak+b independently per RGB channel, trimming twice."""
    gains, offsets = [], []
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
    return {"a": gains, "b": offsets, "samples": len(leak)}


def _mask_to_canvas(mask: np.ndarray, crop: CropFrame) -> np.ndarray:
    image = np.repeat(mask.astype(np.float32)[..., None], 3, axis=2)
    return crop_to_canvas(image, crop)[..., 0].copy()


class _WindowBuffer:
    """Active canvas buffers; future leak shots use a lossless canvas spool."""

    def __init__(self, plan: RealWindowPlan, temp_dir: Path, config: RealPairConfig) -> None:
        self.plan, self.temp_dir = plan, temp_dir
        self.active = False
        count = len(plan.crops)
        self.retail: list[np.ndarray | None] = [None] * count
        self.leak: list[np.ndarray | None] = [None] * count
        self.alpha: list[np.ndarray | None] = [None] * count
        self.sample_masks: dict[int, np.ndarray] = {}
        self.sample_indices = tuple(np.linspace(0, count - 1, min(config.colour_sample_frames, count), dtype=int))
        self.spooled: dict[int, Path] = {}
        self.retail_count = 0
        self.leak_count = 0

    def activate(self) -> None:
        self.active = True
        for index, path in self.spooled.items():
            self.leak[index] = np.load(path, allow_pickle=False)
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
        if side == "leak":
            if self.active:
                self.leak[index] = canvas
            else:
                path = self.temp_dir / f"{self.plan.key}-leak-{index:04d}.npy"
                np.save(path, canvas, allow_pickle=False)
                self.spooled[index] = path
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
            if sampled:
                for overlay in regions.overlays:
                    if overlay.active(self.plan.start + index):
                        area = _overlay_slice(overlay.box, self.plan.work,
                                              config.overlay_margin * max(alignment.retail_size))
                        if area is not None:
                            grown[area] = True
                self.sample_masks[index] = _mask_to_canvas(grown, crop) > 1e-6

    def colour_fit(self) -> dict:
        xs, ys = [], []
        for index in self.sample_indices:
            pixels = np.flatnonzero(~self.sample_masks[index])
            # Bound regression memory on large work areas; deterministic spatial
            # samples cover every selected frame, with at most 32K pixels each.
            if len(pixels) > 32768:
                pixels = pixels[np.linspace(0, len(pixels) - 1, 32768, dtype=int)]
            xs.append(self.leak[index].reshape(-1, 3)[pixels])
            ys.append(self.retail[index].reshape(-1, 3)[pixels])
        x, y = np.concatenate(xs), np.concatenate(ys)
        if len(x) < 2:
            raise ValueError(f"{self.plan.key}: no uncensored, overlay-free pixels for colour fitting")
        return _fit_colour(x, y)

    def finish(self, title: str, alignment: Alignment, out_dir: Path, config: RealPairConfig) -> dict:
        plan = self.plan
        fit = self.colour_fit()
        gain, offset = np.asarray(fit["a"], np.float32), np.asarray(fit["b"], np.float32)

        def canvases(target: bool):
            for index in range(len(plan.crops)):
                if target:
                    image = self.leak[index].astype(np.float32)
                    image *= gain
                    image += offset
                    np.clip(image, 0, 255, out=image)
                    if config.target == "composite":
                        alpha = self.alpha[index]
                        retail = self.retail[index]
                        image = retail * (1 - alpha[..., None]) + image * alpha[..., None]
                    image = np.rint(image).clip(0, 255).astype(np.uint8)
                else:
                    image = self.retail[index]
                yield image

        write_video(out_dir / "clean" / f"{plan.key}.mp4", canvases(True), config.fps, crf=config.output_crf)
        write_video(out_dir / "mosaic" / f"{plan.key}.mp4", canvases(False), config.fps, crf=config.output_crf)
        self.retail.clear()
        self.leak.clear()
        self.alpha.clear()
        self.sample_masks.clear()
        return {
            "video": f"clean/{plan.key}.mp4", "reference_video": f"mosaic/{plan.key}.mp4",
            "caption": config.caption, "key": plan.key, "title": title,
            "retail": alignment.retail, "leak": alignment.leak, "track": plan.track,
            "start": plan.start, "leak_start": min(plan.leak_indices), "leak_stop": max(plan.leak_indices) + 1,
            "frames": len(plan.crops), "work": list(plan.work), "crops": [list(c.box) for c in plan.crops],
            "target": config.target, "colour_fit": fit, "leak_indices": list(plan.leak_indices),
        }


def _render(title: str, alignment: Alignment, regions: RegionSet, plans: list[RealWindowPlan],
            out_dir: Path, temp_dir: Path, config: RealPairConfig, decode_threads: int = 2) -> list[dict]:
    demands = defaultdict(list)
    for plan in plans:
        for index, leak_index in enumerate(plan.leak_indices):
            demands[leak_index].append((plan, index))
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
    with closing(read_frames(alignment.retail, threads=decode_threads)) as retail_frames, closing(read_frames(alignment.leak, threads=decode_threads)) as leak_frames:
        for frame_index, retail in enumerate(retail_frames):
            while plan is not None and plan.start == frame_index:
                render = buffer(plan)
                render.activate()
                active.append(render)
                plan = next(upcoming, None)
            if active:
                required = max(r.plan.leak_indices[frame_index - r.plan.start] for r in active)
                while leak_position < required:
                    try:
                        leak = next(leak_frames)
                    except StopIteration as error:
                        raise RuntimeError(f"{title}: leak ended before mapped frame {required}") from error
                    leak_position += 1
                    needed = demands.pop(leak_position, [])
                    if needed:
                        if (leak.shape[1], leak.shape[0]) != alignment.retail_size:
                            leak = cv2.resize(leak, alignment.retail_size, interpolation=cv2.INTER_AREA)
                        for needed_plan, index in needed:
                            buffer(needed_plan).add("leak", index, leak, alignment, regions, config)
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
    meta = {"title": title, "retail": alignment.retail, "leak": alignment.leak, "windows": rows}
    _write_atomic(meta_path, json.dumps(meta))
    return rows


def load_config(path: Path | None) -> RealPairConfig:
    """Read optional pairs and camera YAML sections, rejecting unknown settings."""
    data = (yaml.safe_load(Path(path).read_text(encoding="utf-8")) if path else None) or {}
    unknown = set(data) - {"pairs", "camera"}
    if unknown:
        raise ValueError(f"{path}: unknown sections {sorted(unknown)}")
    camera = replace(CameraConfig(), **(data.get("camera") or {}))
    return replace(RealPairConfig(), **(data.get("pairs") or {}), camera=camera)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reveil-real-pairs", description=__doc__.split("\n\n")[0])
    parser.add_argument("--align-dir", type=Path, required=True)
    parser.add_argument("--regions-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="YAML overrides: pairs / camera sections")
    parser.add_argument("--target", choices=("composite", "leak"))
    parser.add_argument("--max-windows-per-title", type=int)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--decode-threads", type=int, default=2, help="threads per video decoder (default: 2)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--only", nargs="+", help="title stems to include")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    overrides = {name: getattr(args, name) for name in ("target", "max_windows_per_title")
                 if getattr(args, name) is not None}
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
                print(f"[{done}/{len(jobs)}] {title}: {len(future.result())} pairs", flush=True)
            except Exception as error:  # noqa: BLE001 - finish independent titles
                failed += 1
                print(f"[{done}/{len(jobs)}] {title}: FAILED: {error!r}", file=sys.stderr, flush=True)
    total = rebuild_dataset(args.out, config.caption)
    print(f"{total} pairs in {args.out / 'dataset.jsonl'}" + (f"; {failed} titles failed" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
