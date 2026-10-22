"""Find real mosaics and one-sided overlays in aligned retail/leak video pairs.

    python -m reveil.realmask --align-dir ALIGNMENTS/ --out REGIONS/

Both videos are decoded forward, without seeks. Only synthetic data belongs in tests;
this module deliberately has no image output or preview mode.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

# Each title has its own process; small dot products must not spin up BLAS pools.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import av
import cv2
import numpy as np

from reveil.annotate import CutDetector, mask_regions
from reveil.align import _luma
from reveil.frame_map import Alignment, load_alignment
from reveil.regions import Overlay, Region, RegionSet, save_regions
from reveil.tracks import box_iou

_FULL_RANGE = np.rint((np.arange(256).clip(16, 235) - 16) * (255 / 219)).astype(np.uint8)


@dataclass(frozen=True)
class MaskConfig:
    scale: float = 0.5  # detection size over the retail size
    stride: int = 2  # analyse every stride-th retail frame
    block_min_fraction: float = 0.0075  # smallest grid period over the retail long side, with a 4px floor
    block_max: int = 96  # largest grid period, in retail pixels
    residual_min: float = 8.0  # minimum blurred absolute residual, grey levels
    residual_sigma: float = 4.0  # residual threshold over robust noise sigma
    trim: float = 0.2  # worst residual share removed at each robust refit
    fit_samples: int = 16384  # at most this many pixels for gain, offset, and MAD
    close: int = 7  # candidate closing kernel, detection pixels
    open: int = 3  # candidate opening kernel, detection pixels
    fill: int = 31  # component closing kernel before testing, detection pixels
    min_area: float = 0.0005  # smallest component/polygon share of frame area
    peak_min: float = 0.2  # minimum normalized autocorrelation peak
    harmonic_share: float = 0.7  # a half/third-period peak must reach this share of the best
    profile_clip: float = 0.8  # cap profile energy at this nonzero quantile to limit isolated edges
    axis_tolerance: float = 0.15  # relative disagreement allowed between grid axes
    flat_max: float = 3.0  # maximum median retail block-interior standard deviation
    flat_ratio: float = 0.3  # maximum retail/leak block-interior standard deviation ratio
    cell_share: float = 0.5  # fraction of a cell inside a component to include it
    simplify: float = 0.002  # polygon simplification over outline length
    cut_threshold: float = 30.0  # mean thumbnail difference at a scene cut
    overlay_chunk: int = 9000  # retail frames per independent overlay tally
    overlay_frequency: float = 0.4  # analysed-frame share needed to call a cell persistent
    overlay_edge_ratio: float = 3.0  # minimum one-sided mean Sobel energy ratio
    overlay_iou: float = 0.5  # minimum IoU for merging consecutive chunk rectangles

    def __post_init__(self) -> None:
        if not 0 < self.scale <= 1 or self.stride < 1:
            raise ValueError("scale must be in (0, 1] and stride must be positive")
        if self.block_min_fraction <= 0 or self.block_max < 4:
            raise ValueError("block_min_fraction must be positive and block_max at least 4 pixels")
        if not 0 <= self.trim < 1 or self.fit_samples < 3 or self.overlay_chunk < 1:
            raise ValueError("invalid fit or chunk settings")


@dataclass(frozen=True)
class MosaicArea:
    mask: np.ndarray  # bool mask in full retail resolution
    block: float  # grid period in full retail pixels
    phase: tuple[float, float]  # grid corner modulo block, in full retail pixels
    retail_std: float
    leak_std: float


@dataclass
class MosaicRegions(RegionSet):
    stats: dict = field(default_factory=dict)


def _period(profile: np.ndarray, low: float, high: float, config: MaskConfig) -> tuple[float, float] | None:
    """Strongest autocorrelation period, preferring its supported fundamental."""
    values = np.asarray(profile, dtype=np.float64)
    values = values - values.mean()
    energy = float(values @ values)
    first, last = max(2, int(np.ceil(low))), min(int(np.floor(high)), len(values) // 2)
    if energy < 1e-8 or first > last:
        return None
    # Profiles are short (one component side), so direct correlation avoids FFT setup.
    corr = np.correlate(values, values, mode="full")[len(values) - 1 :]
    # Normalize by overlapping energy: distant peaks otherwise lose to their multiples.
    squares = np.r_[0.0, np.cumsum(values * values)]
    lags = np.arange(len(values))
    denom = np.sqrt(squares[len(values) - lags] * (squares[-1] - squares[lags]))
    corr = corr / np.maximum(denom, 1e-8)
    peaks = [lag for lag in range(first, last + 1)
             if corr[lag] >= corr[lag - 1] and corr[lag] >= corr[lag + 1]]
    if not peaks:
        return None
    best = max(peaks, key=lambda lag: corr[lag])
    if corr[best] < config.peak_min:
        return None
    while True:
        fundamental = [lag for lag in peaks if lag < best
                       and corr[lag] >= max(config.peak_min, config.harmonic_share * corr[best])
                       and any(abs(lag - best / divisor) <= max(1, best / divisor * 0.08) for divisor in (2, 3))]
        if not fundamental:
            break
        best = min(fundamental)
    a, b, c = corr[best - 1 : best + 2]
    offset = float(np.clip(0.5 * (a - c) / (a - 2 * b + c), -0.5, 0.5)) if a - 2 * b + c else 0.0
    return float(np.clip(best + offset, low, high)), float(b)


def estimate_block(x_profile: np.ndarray, y_profile: np.ndarray, config: MaskConfig) -> float | None:
    """Grid period in detection pixels; either axis may be absent in a thin area."""
    # Search below the acceptance minimum too: otherwise an 8px codec grid can
    # masquerade as a 16px mosaic through its autocorrelation harmonics.
    low, high = 4 * config.scale, config.block_max * config.scale
    x, y = (_period(p, low, high, config) for p in (x_profile, y_profile))
    if x is None or y is None:
        return (x or y)[0] if x is not None or y is not None else None
    small, large = sorted((x[0], y[0]))
    if large / small - 1 <= config.axis_tolerance:
        return (x[0] * x[1] + y[0] * y[1]) / (x[1] + y[1])
    if abs(large / small - 2) <= 2 * config.axis_tolerance:
        return small
    return None


def _phase(profile: np.ndarray, block: float, origin: int) -> float:
    # Fold edge energy onto a subpixel grid; Sobel straddles the cell boundary.
    bins = max(4, round(block * 4))
    positions = np.mod((np.arange(len(profile)) + origin + 0.5) / block, 1)
    folded = np.bincount(np.floor(positions * bins).astype(int), weights=profile, minlength=bins)
    radius = max(1, round(bins / block * 0.8))
    smooth = sum(np.roll(folded, step) for step in range(-radius, radius + 1))
    peak = int(np.argmax(smooth))
    distance = (np.arange(bins) - peak + bins / 2) % bins - bins / 2
    weights = folded * (np.abs(distance) <= 2 * radius)
    offset = float(np.sum(distance * weights) / max(weights.sum(), 1e-8))
    return float(((peak + offset + 0.5) / bins * block) % block)


def _residual(retail: np.ndarray, leak: np.ndarray, config: MaskConfig) -> tuple[np.ndarray, float]:
    step = max(1, int(np.ceil(np.sqrt(retail.size / config.fit_samples))))
    x = leak[::step, ::step].reshape(-1).astype(np.float64)
    y = retail[::step, ::step].reshape(-1).astype(np.float64)
    keep = np.ones(len(x), dtype=bool)
    gain, offset = 1.0, 0.0
    for iteration in range(3):
        xx, yy = x[keep], y[keep]
        xm, ym = xx.mean(), yy.mean()
        centred = xx - xm
        gain = float(centred @ (yy - ym) / max(float(centred @ centred), 1e-8))
        offset = float(ym - gain * xm)
        residual = y - (gain * x + offset)
        if iteration < 2:
            keep = np.abs(residual) <= np.quantile(np.abs(residual), 1 - config.trim)
    sigma = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
    return retail.astype(np.float32) - (gain * leak.astype(np.float32) + offset), sigma


def _cells(retail: np.ndarray, leak: np.ndarray, mask: np.ndarray, block: float,
           phase: tuple[float, float], config: MaskConfig) -> tuple[np.ndarray, float, float] | None:
    """Whole grid cells supported by the component, and median interior deviations."""
    height, width = mask.shape
    edges = []
    for length, start in zip((width, height), phase):
        edges.append(np.unique(np.clip(np.rint(start + np.arange(-1, int(length / block) + 2) * block),
                                       0, length).astype(int)))
    xs, ys = edges
    x0, y0 = np.meshgrid(xs[:-1], ys[:-1])
    x1, y1 = np.meshgrid(xs[1:], ys[1:])

    def sums(image: np.ndarray, left, top, right, bottom) -> np.ndarray:
        integral = cv2.integral(image.astype(np.float32), sdepth=cv2.CV_64F)
        return integral[bottom, right] - integral[top, right] - integral[bottom, left] + integral[top, left]

    area = (x1 - x0) * (y1 - y0)
    chosen = sums(mask, x0, y0, x1, y1) >= config.cell_share * area
    # Ignore compression ringing and mixed downsample pixels at cell edges.
    inset = max(1, round(block * 0.12))
    valid = chosen & ((x1 - x0) > 2 * inset + 1) & ((y1 - y0) > 2 * inset + 1)
    if not valid.any():
        return None
    left, right, top, bottom = x0[valid] + inset, x1[valid] - inset, y0[valid] + inset, y1[valid] - inset
    counts = (right - left) * (bottom - top)
    stds = []
    for image in (retail, leak):
        mean = sums(image, left, top, right, bottom) / counts
        square = sums(image.astype(np.float32) ** 2, left, top, right, bottom) / counts
        stds.append(float(np.median(np.sqrt(np.maximum(0, square - mean * mean)))))
    filled = np.repeat(np.repeat(chosen, np.diff(ys), axis=0), np.diff(xs), axis=1).astype(np.uint8)
    return filled, stds[0], stds[1]


def _analyse(retail_grey: np.ndarray, leak_grey: np.ndarray, config: MaskConfig
             ) -> tuple[list[MosaicArea], list[tuple[tuple[int, int, int, int], np.ndarray]], np.ndarray, np.ndarray]:
    if retail_grey.ndim != 2 or leak_grey.ndim != 2:
        raise ValueError("frame_mosaic expects two full-range grey frames")
    height, width = retail_grey.shape
    if leak_grey.shape != retail_grey.shape:
        leak_grey = cv2.resize(leak_grey, (width, height), interpolation=cv2.INTER_AREA)
    size = max(1, round(width * config.scale)), max(1, round(height * config.scale))
    retail = cv2.resize(retail_grey, size, interpolation=cv2.INTER_AREA)
    leak = cv2.resize(leak_grey, size, interpolation=cv2.INTER_AREA)
    residual, sigma = _residual(retail, leak, config)
    candidate = (cv2.blur(np.abs(residual), (3, 3)) > max(config.residual_min, config.residual_sigma * sigma)).astype(np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((config.close, config.close), np.uint8))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((config.open, config.open), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    found, other = [], []
    scale = size[0] / width
    block_min = max(4, config.block_min_fraction * max(width, height))
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < config.min_area * candidate.size:
            continue
        # Keep morphology local; explicit zero padding avoids closing into the crop border.
        pad = config.fill // 2 + 1
        x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(size[0], x + w + pad), min(size[1], y + h + pad)
        component = (labels[y0:y1, x0:x1] == label).astype(np.uint8)
        component = cv2.morphologyEx(component, cv2.MORPH_CLOSE, np.ones((config.fill, config.fill), np.uint8))
        r, l = retail[y0:y1, x0:x1], leak[y0:y1, x0:x1]
        interior = cv2.erode(component, np.ones((5, 5), np.uint8))
        gx = np.abs(cv2.Sobel(r, cv2.CV_32F, 1, 0, ksize=3)) * interior
        gy = np.abs(cv2.Sobel(r, cv2.CV_32F, 0, 1, ksize=3)) * interior
        xp, yp = gx.sum(axis=0), gy.sum(axis=1)
        # A scene edge inside the component can dwarf the regularly spaced grid edges.
        # Limit those outliers before correlation; retain raw energy for phase fitting.
        profiles = [np.minimum(p, np.quantile(p[p > 0], config.profile_clip)) if (p > 0).any() else p
                    for p in (xp, yp)]
        block = estimate_block(*profiles, replace(config, scale=scale))
        accepted = False
        if block is not None and block / scale >= block_min:
            phase = _phase(xp, block, x0), _phase(yp, block, y0)
            local_phase = (phase[0] - x0) % block, (phase[1] - y0) % block
            cells = _cells(r, l, component, block, local_phase, config)
            if cells is not None:
                filled, retail_std, leak_std = cells
                if retail_std < config.flat_max and retail_std < config.flat_ratio * leak_std:
                    radius = max(1, round(block / 2))
                    # Dilate on the whole detection canvas, preserving cells near crop edges.
                    canvas = np.zeros_like(candidate)
                    canvas[y0:y1, x0:x1] = filled
                    canvas = cv2.dilate(canvas, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8))
                    mask = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
                    found.append(MosaicArea(mask, block / scale, (phase[0] / scale, phase[1] / scale), retail_std, leak_std))
                    accepted = True
        if not accepted:
            other.append(((x0, y0, x1, y1), component))
    return found, other, retail, leak


def frame_mosaic(retail_grey: np.ndarray, leak_grey: np.ndarray, config: MaskConfig) -> list[MosaicArea]:
    """Detect grid mosaics, returning masks and grid measurements in retail pixels."""
    return _analyse(retail_grey, leak_grey, config)[0]


class _OverlayChunk:
    def __init__(self) -> None:
        self.count = 0
        self.hits = np.zeros((90, 160), dtype=np.uint32)
        self.energy = np.zeros((2, 90, 160), dtype=np.float64)

    def add(self, components, retail: np.ndarray, leak: np.ndarray, config: MaskConfig) -> None:
        self.count += 1
        covered = np.zeros_like(self.hits, dtype=bool)
        for (x0, y0, x1, y1), mask in components:
            energies = []
            for frame in (retail, leak):
                patch = frame[y0:y1, x0:x1]
                gradient = cv2.magnitude(cv2.Sobel(patch, cv2.CV_32F, 1, 0), cv2.Sobel(patch, cv2.CV_32F, 0, 1))
                energies.append(float(cv2.mean(gradient, mask)[0]))
            side = int(energies[1] > energies[0])
            if energies[side] < config.overlay_edge_ratio * max(energies[1 - side], 1e-6):
                continue
            canvas = np.zeros(retail.shape, dtype=np.uint8)
            canvas[y0:y1, x0:x1] = mask
            cells = cv2.resize(canvas.astype(np.float32), (160, 90), interpolation=cv2.INTER_AREA) > 0
            covered |= cells
            self.energy[:, cells] += np.asarray(energies)[:, None]
        self.hits += covered

    def finish(self, start: int, stop: int, width: int, height: int, config: MaskConfig) -> list[Overlay]:
        overlays = []
        if not self.count:
            return overlays
        frequent = self.hits / self.count >= config.overlay_frequency
        count, labels, stats, _ = cv2.connectedComponentsWithStats(frequent.astype(np.uint8), connectivity=8)
        for label in range(1, count):
            x, y, w, h, _ = stats[label]
            cells = labels == label
            side = int(self.energy[1, cells].sum() > self.energy[0, cells].sum())
            frequency = float(self.hits[cells].mean() / self.count)
            box = (max(0, x - 1) * width / 160, max(0, y - 1) * height / 90,
                   min(160, x + w + 1) * width / 160, min(90, y + h + 1) * height / 90)
            overlays.append(Overlay(box, ("retail", "leak")[side], frequency, (start, stop)))
        return overlays


def _merge_overlays(previous: list[Overlay], chunk: list[Overlay], config: MaskConfig) -> None:
    used = set()
    for overlay in chunk:
        matches = [(box_iou(old.box, overlay.box), index) for index, old in enumerate(previous)
                   if index not in used and old.side == overlay.side and old.frames[1] == overlay.frames[0]]
        iou, index = max(matches, default=(0.0, -1))
        if iou > config.overlay_iou:
            old = previous[index]
            a, b = old.frames[1] - old.frames[0], overlay.frames[1] - overlay.frames[0]
            box = (min(old.box[0], overlay.box[0]), min(old.box[1], overlay.box[1]),
                   max(old.box[2], overlay.box[2]), max(old.box[3], overlay.box[3]))
            previous[index] = Overlay(box, old.side, (old.frequency * a + overlay.frequency * b) / (a + b),
                                      (old.frames[0], overlay.frames[1]))
            used.add(index)
        else:
            previous.append(overlay)
            used.add(len(previous) - 1)


def _decoded_frames(path: str, threads: int):
    """Yield decoded frames without converting frames skipped by the analysis stride."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.thread_count = threads
        yield from container.decode(stream)


def mosaic_regions_for_pair(alignment: Alignment, config: MaskConfig, *, progress=None,
                            decode_threads: int = 2) -> RegionSet:
    """Detect a pair sequentially; the returned MosaicRegions also exposes ``stats``.

    ``progress(done, total)`` runs after every retail frame, including skipped frames.
    Frames mapped behind the leak decoder are skipped and counted in the statistics.
    """
    started = time.perf_counter()
    result = MosaicRegions()
    cuts, histogram, shares, overlays = set(), Counter(), [], []
    detector = CutDetector(config.cut_threshold)
    chunk, chunk_start = _OverlayChunk(), 0
    analysed, leak_index, frame_count, skipped_backward = 0, -1, 0, 0
    leak_frame = None
    retail_frames = _decoded_frames(alignment.retail, decode_threads)
    leak_frames = _decoded_frames(alignment.leak, decode_threads)
    try:
        for index, retail_frame in enumerate(retail_frames):
            if index >= alignment.retail_frames:
                raise ValueError("retail contains more frames than its alignment")
            frame_count = index + 1
            retail_y = _luma(retail_frame)
            # Scene cuts only need a thumbnail; stretch the tiny luma image.
            thumb = cv2.resize(retail_y, (64, 36), interpolation=cv2.INTER_AREA)
            if detector.grey(cv2.LUT(thumb, _FULL_RANGE)) and index:
                cuts.add(index)
            if index - chunk_start >= config.overlay_chunk:
                _merge_overlays(overlays, chunk.finish(chunk_start, index, *alignment.retail_size, config), config)
                chunk, chunk_start = _OverlayChunk(), index
            target = alignment.leak_index(index)
            if target is not None and target < leak_index:
                skipped_backward += 1
                target = None
            if index % config.stride == 0 and target is not None:
                while leak_index < target:
                    leak_frame = next(leak_frames, None)
                    if leak_frame is None:
                        raise ValueError(f"leak ended before aligned frame {target}")
                    leak_index += 1
                retail_grey = cv2.LUT(retail_y, _FULL_RANGE)
                leak_grey = cv2.LUT(_luma(leak_frame), _FULL_RANGE)
                areas, other, retail, leak = _analyse(retail_grey, leak_grey, config)
                analysed += 1
                chunk.add(other, retail, leak, config)
                items = []
                union = np.zeros(retail_grey.shape, dtype=bool)
                for area in areas:
                    polygons = mask_regions(area.mask, grow_px=0, min_area_px=config.min_area * union.size,
                                            simplify=config.simplify)
                    items.extend(Region(p.box, p.polygon, block=area.block) for p in polygons)
                    histogram[int(4 * round(area.block / 4))] += 1
                    union |= area.mask
                if items:
                    result.frames[index] = items
                    shares.append(float(union.mean()))
            if progress is not None:
                progress(index + 1, alignment.retail_frames)
    finally:
        retail_frames.close()
        leak_frames.close()
    if frame_count != alignment.retail_frames:
        raise ValueError(f"retail ended at {frame_count}, expected {alignment.retail_frames} frames")
    _merge_overlays(overlays, chunk.finish(chunk_start, frame_count, *alignment.retail_size, config), config)
    result.cuts, result.overlays = frozenset(cuts), tuple(overlays)
    result.stats = {
        "frames_analysed": analysed, "frames_with_mosaic": len(result.frames),
        "skipped_backward_frames": skipped_backward,
        "block_histogram": dict(sorted(histogram.items())),
        "median_mosaic_area_share": float(np.median(shares)) if shares else 0.0,
        "overlay_count": len(overlays), "seconds": round(time.perf_counter() - started, 3),
    }
    return result


def _mask_in_worker(alignment_path: Path, output: Path, config: MaskConfig, decode_threads: int = 2) -> dict:
    cv2.setNumThreads(1)
    regions = mosaic_regions_for_pair(load_alignment(alignment_path), config, decode_threads=decode_threads)
    save_regions(regions, output, extra={"settings": {**asdict(config), "alignment": str(alignment_path)},
                                        "stats": regions.stats})
    return regions.stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reveil-realmask", description=__doc__.split("\n\n")[0])
    parser.add_argument("--align-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--decode-threads", type=int, default=2, help="threads per video decoder (default: 2)")
    parser.add_argument("--stride", type=int, default=MaskConfig.stride)
    parser.add_argument("--scale", type=float, default=MaskConfig.scale)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--only", nargs="+", help="alignment title stems to process")
    args = parser.parse_args(argv)
    try:
        config = MaskConfig(stride=args.stride, scale=args.scale)
        if args.workers < 1:
            raise ValueError("workers must be positive")
        if args.decode_threads < 1:
            raise ValueError("--decode-threads must be positive")
    except ValueError as error:
        parser.error(str(error))
    paths = sorted(args.align_dir.glob("*.json"))
    if args.only:
        paths = [path for path in paths if path.stem in args.only]
    if not paths:
        parser.error("no alignment files found")
    jobs = [path for path in paths if args.overwrite or not (args.out / path.name).is_file()]
    args.out.mkdir(parents=True, exist_ok=True)
    failed = 0

    def report(path: Path, stats: dict | None = None, error: Exception | None = None) -> None:
        nonlocal failed
        if error is None:
            print(f"{path.stem}: {stats}", flush=True)
            if stats["skipped_backward_frames"]:
                print(f"{path.stem}: warning: skipped {stats['skipped_backward_frames']} retail frames mapped "
                      "behind the forward-only leak decoder", file=sys.stderr, flush=True)
        else:
            failed += 1
            print(f"{path.stem}: FAILED: {error!r}", file=sys.stderr, flush=True)

    if args.workers == 1:
        for path in jobs:
            try:
                report(path, _mask_in_worker(path, args.out / path.name, config, args.decode_threads))
            except Exception as error:  # noqa: BLE001 - keep processing other titles
                report(path, error=error)
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(_mask_in_worker, path, args.out / path.name, config, args.decode_threads): path for path in jobs}
            for future in as_completed(futures):
                try:
                    report(futures[future], future.result())
                except Exception as error:  # noqa: BLE001
                    report(futures[future], error=error)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
