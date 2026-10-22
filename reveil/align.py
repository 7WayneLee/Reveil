"""Align a retail release to any ordering of matching footage in its leak.

    python -m reveil.align --censored SOURCES/censored --uncensored SOURCES/uncensored --out ALIGN

Only small grey signatures are retained. Global correlation proposes matches; trimmed
distances verify them, and a temporal path preserves timing through ambiguous stills.
"""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

# Each title has its own process; small dot products must not spin up BLAS pools.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import av
import cv2
import numpy as np

from reveil.frame_map import Alignment, save_alignment
from reveil.realpairs import find_real_pairs
from reveil.video import VideoInfo

SIGNATURE_VERSION = 2  # Full-resolution legal-range luma, averaged by OpenCV.


@dataclass(frozen=True)
class AlignConfig:
    sig_size: tuple[int, int] = (32, 18)  # Width and height of the legal-range luma signature.
    anchor_interval_s: float = 2.0  # Retail time between global-search anchors.
    search_fps: float = 6.0  # Leak frames per second in the global correlation search.
    candidates: int = 5  # Distinct leak neighbourhoods retained for each anchor.
    candidate_separation_s: float = 0.5  # Suppress neighbouring peaks before taking another candidate.
    min_anchors: int = 3  # Consecutive supporting anchors needed to establish an offset.
    offset_tolerance_s: float = 0.1  # Maximum deviation from a segment's median offset.
    max_anchor_dist: float = 0.12  # Reject global proposals above this trimmed z-distance.
    anchor_motion_margin: float = 0.015  # An anchor must distinguish its match from frames two retail periods away.
    max_frame_dist: float = 0.12  # Absolute ceiling for a reported non-flat frame match.
    noise_margin: float = 0.025  # Additional distance allowed above a segment's measured encoding noise.
    flat_std: float = 2.0  # Signatures below this grey-level standard deviation give no evidence.
    z_epsilon: float = 1e-3  # Stabilise normalisation without changing non-flat signatures appreciably.
    trim_fraction: float = 0.8  # Fraction of least-different cells used in frame distance.
    refine_periods: float = 2.0  # Search radius in retail frame periods around predicted leak time.
    still_margin: float = 0.008  # A competing frame must improve by this much to defeat timing locally.
    shift_penalty: float = 0.035  # Cost of a one-frame timing change; sustained evidence can pay it.
    timing_bias: float = 1e-5  # Break ambiguous path ties in favour of the segment's predicted frame.
    continuation_margin: float = 0.01  # Prefer an existing offset when candidate quality is this close.
    anchor_chunk: int = 128  # Bound the global correlation matrix to this many retail anchors.
    frame_chunk: int = 256  # Bound temporary normalised frame and distance arrays.

    def __post_init__(self) -> None:
        if len(self.sig_size) != 2 or min(self.sig_size) < 1:
            raise ValueError("sig_size must contain two positive sides")
        if not 0 < self.trim_fraction <= 1:
            raise ValueError("trim_fraction must be in (0, 1]")
        for name, value in vars(self).items():
            if name not in {"sig_size", "trim_fraction"} and value <= 0:
                raise ValueError(f"{name} must be positive")


def _luma(frame: av.VideoFrame) -> np.ndarray:
    """Borrow 8-bit planar YUV luma, falling back to full-resolution grey conversion."""
    fmt = frame.format
    if (fmt.name.startswith("yuv") and not fmt.name.startswith("yuvj") and fmt.is_planar
            and all(component.bits == 8 for component in fmt.components)):
        return np.frombuffer(frame.planes[0], np.uint8).reshape(frame.height, frame.planes[0].line_size)[:, :frame.width]
    return frame.to_ndarray(format="gray")


def _repair_times(times, fps: float) -> tuple[np.ndarray, int]:
    """Interpolate absent PTS, then separate repeated/backward times by a microsecond."""
    repaired = np.asarray(times, np.float64).copy()
    missing = ~np.isfinite(repaired)
    indices = np.arange(len(repaired))
    finite = indices[~missing]
    if len(finite):
        inside = missing & (indices > finite[0]) & (indices < finite[-1])
        repaired[inside] = np.interp(indices[inside], finite, repaired[finite])
        ends = missing & ~inside
        repaired[ends] = indices[ends] / fps
    else:
        repaired[:] = indices / fps
    count = int(missing.sum())
    for index in range(1, len(repaired)):
        if repaired[index] <= repaired[index - 1]:
            repaired[index] = repaired[index - 1] + 1e-6
            count += int(not missing[index])
    return repaired, count


def frame_signatures(path, *, config: AlignConfig = AlignConfig(), decode_threads: int = 2,
                     stats: dict | None = None) -> tuple[np.ndarray, np.ndarray, VideoInfo]:
    """Decode once, keeping repaired PTS and area-averaged, clipped luma signatures.

    ``stats`` optionally receives the number of repaired presentation times.
    Source frames are discarded after their signatures have been copied.
    """
    times, blocks, block = [], [], np.empty((4096, config.sig_size[1], config.sig_size[0]), np.uint8)
    used, total = 0, 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.thread_count = decode_threads
        fps = float(stream.average_rate or stream.guessed_rate or 24)
        width, height = stream.codec_context.width, stream.codec_context.height
        for frame in container.decode(stream):
            # Swscale AREA aliases at large reductions. Clip super-whites before
            # averaging so legal-range retail and unrestricted leaks agree.
            small = cv2.resize(np.clip(_luma(frame), 16, 235), config.sig_size, interpolation=cv2.INTER_AREA)
            block[used] = small
            timestamp = float(frame.pts * frame.time_base) if frame.pts is not None and frame.time_base is not None else np.nan
            times.append(timestamp)
            used, total = used + 1, total + 1
            if used == len(block):
                blocks.append(block)
                block, used = np.empty_like(block), 0
    if used:
        blocks.append(block[:used].copy())
    signatures = np.concatenate(blocks) if blocks else np.empty((0, config.sig_size[1], config.sig_size[0]), np.uint8)
    times, repaired = _repair_times(times, fps)
    if stats is not None:
        stats["repaired_times"] = repaired
    return times, signatures, VideoInfo(width, height, fps, total)


def _cached_signatures(path: Path, cache_dir: Path | None, config: AlignConfig, decode_threads: int):
    def decode():
        stats = {}
        times, signatures, info = frame_signatures(path, config=config, decode_threads=decode_threads, stats=stats)
        return times, signatures, info, stats["repaired_times"]

    if cache_dir is None:
        return decode()
    path = path.absolute()
    stat = path.stat()
    key = hashlib.sha1(f"{SIGNATURE_VERSION}\0{path}\0{stat.st_size}\0{stat.st_mtime_ns}".encode()).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{key}.npz"
    if target.is_file():
        try:
            with np.load(target, allow_pickle=False) as cached:
                signatures = cached["signatures"]
                if signatures.shape[1:] == config.sig_size[::-1]:
                    width, height, fps, frames = cached["info"].tolist()
                    return cached["times"], signatures, VideoInfo(int(width), int(height), fps, int(frames)), int(cached["repaired_times"])
        except (OSError, ValueError, KeyError):
            pass  # An interrupted or obsolete cache is safe to rebuild.
    result = decode()
    times, signatures, info, repaired = result
    with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".npz", delete=False) as partial:
        temporary = Path(partial.name)
    try:
        np.savez(temporary, times=times, signatures=signatures, repaired_times=repaired,
                 info=np.array([info.width, info.height, info.fps, info.frames]))
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def _normalise(signatures: np.ndarray, config: AlignConfig):
    z = signatures.reshape(len(signatures), -1).astype(np.float32)
    z -= z.mean(axis=1, keepdims=True)
    std = z.std(axis=1)
    z /= std[:, None] + config.z_epsilon
    return z, std >= config.flat_std


def _distance(a: np.ndarray, b: np.ndarray, config: AlignConfig) -> np.ndarray:
    difference = np.abs(a - b)
    count = max(1, int(difference.shape[-1] * config.trim_fraction))
    return np.partition(difference, count - 1, axis=-1)[..., :count].mean(axis=-1)


def _nearest(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    right = np.searchsorted(times, values).clip(0, len(times) - 1)
    left = np.maximum(0, right - 1)
    return np.where(abs(times[left] - values) <= abs(times[right] - values), left, right)


def _sample(times: np.ndarray, interval: float) -> np.ndarray:
    return np.unique(_nearest(times, np.arange(times[0], times[-1] + interval / 2, interval)))


def _anchor_candidates(t_a, sig_a, t_b, sig_b, config):
    anchors = _sample(t_a, config.anchor_interval_s)
    z_a, good_a = _normalise(sig_a[anchors], config)
    anchors, z_a = anchors[good_a], z_a[good_a]
    coarse = _sample(t_b, 1 / config.search_fps)
    z_b, good_b = _normalise(sig_b[coarse], config)
    coarse, z_b = coarse[good_b], z_b[good_b]
    candidates = [[] for _ in anchors]
    if not len(coarse):
        return anchors, candidates
    dims = z_b.shape[1]
    period_b = float(np.median(np.diff(t_b))) if len(t_b) > 1 else 1 / config.search_fps
    period_a = float(np.median(np.diff(t_a))) if len(t_a) > 1 else period_b
    evidence_radius = max(1, int(round(2 * period_a / period_b)))
    radius = max(1, int(np.ceil(1 / config.search_fps / period_b)))
    for begin in range(0, len(anchors), config.anchor_chunk):
        stop = min(len(anchors), begin + config.anchor_chunk)
        correlation = (z_a[begin:stop] @ z_b.T) / dims
        for _ in range(config.candidates):
            best = correlation.argmax(axis=1)
            for row, peak in enumerate(best):
                if not np.isfinite(correlation[row, peak]):
                    continue
                center = coarse[peak]
                indices = np.arange(max(0, center - radius), min(len(t_b), center + radius + 1))
                nearby, good = _normalise(sig_b[indices], config)
                distances = _distance(z_a[begin + row], nearby, config)
                distances[~good] = np.inf
                winner = int(np.argmin(distances))
                if distances[winner] <= config.max_anchor_dist:
                    j = int(indices[winner])
                    neighbours = np.array([j - evidence_radius, j + evidence_radius])
                    neighbours = neighbours[(neighbours >= 0) & (neighbours < len(t_b))]
                    z_neighbours, _ = _normalise(sig_b[neighbours], config) if len(neighbours) else (None, None)
                    # Near-static images can verify an established time prediction,
                    # but cannot establish or change an offset by themselves.
                    distinct = len(neighbours) and _distance(z_a[begin + row], z_neighbours, config).min() >= distances[winner] + config.anchor_motion_margin
                    if distinct:
                        candidates[begin + row].append((j, float(t_b[j] - t_a[anchors[begin + row]]), float(distances[winner])))
                lo, hi = np.searchsorted(t_b[coarse], t_b[center] + np.array([-1, 1]) * config.candidate_separation_s)
                correlation[row, lo:hi + 1] = -np.inf
    return anchors, candidates


def _segments(anchors, candidates, t_a, config):
    """Track every plausible offset; leak chronology is deliberately unconstrained."""
    tracks, active = [], []
    for ai, options in enumerate(candidates):
        active = [track for track in active if track["last"] >= ai - 1]
        for j, offset, distance in sorted(options, key=lambda item: item[2]):
            matches = [track for track in active if track["last"] < ai and abs(track["offset"] - offset) <= config.offset_tolerance_s]
            if matches:
                track = min(matches, key=lambda tr: abs(tr["offset"] - offset))
                track["points"].append((ai, j, offset, distance))
                track["last"] = ai
                track["offset"] = float(np.median([p[2] for p in track["points"][-9:]]))
            else:
                track = {"points": [(ai, j, offset, distance)], "last": ai, "offset": offset}
                active.append(track)
                tracks.append(track)
    valid = []
    for track in tracks:
        points = track["points"]
        if len(points) >= config.min_anchors:
            valid.append({"points": points, "first": points[0][0], "last": points[-1][0],
                          "offset": float(np.median([p[2] for p in points])),
                          "quality": float(np.median([p[3] for p in points]))})
    chosen, previous = [], None
    for ai in range(len(anchors)):
        options = [track for track in valid if track["first"] <= ai <= track["last"]]
        if not options:
            chosen.append(None)
            continue
        best = min(options, key=lambda tr: (tr["quality"], abs(tr["offset"])))
        continuing = [tr for tr in options if previous is not None and abs(tr["offset"] - previous["offset"]) <= config.offset_tolerance_s
                      and tr["quality"] <= best["quality"] + config.continuation_margin]
        if continuing:
            best = min(continuing, key=lambda tr: abs(tr["offset"] - previous["offset"]))
        chosen.append(best)
        previous = best
    segments = []
    for ai, track in enumerate(chosen):
        if track is None:
            continue
        if segments and segments[-1]["track"] is track and segments[-1]["last"] == ai - 1:
            segments[-1]["last"] = ai
        else:
            segments.append({"track": track, "first": ai, "last": ai})
    return [segment for segment in segments if segment["last"] - segment["first"] + 1 >= config.min_anchors]


def _predictions(indices, segment, anchors, t_a, config):
    points = [p for p in segment["track"]["points"] if segment["first"] <= p[0] <= segment["last"]]
    offsets = np.array([p[2] for p in points])
    # A median of neighbouring anchors prevents one ambiguous still from moving time.
    smoothed = np.array([np.median(offsets[max(0, k - 2): k + 3]) for k in range(len(offsets))])
    return t_a[indices] + np.interp(t_a[indices], t_a[anchors[[p[0] for p in points]]], smoothed)


def _frame_costs(start, stop, segment, anchors, t_a, sig_a, t_b, sig_b, config):
    period_a = float(np.median(np.diff(t_a))) if len(t_a) > 1 else 1.0
    period_b = float(np.median(np.diff(t_b))) if len(t_b) > 1 else period_a
    radius = max(1, int(np.ceil(config.refine_periods * period_a / period_b - 1e-9)))
    deltas = np.arange(-radius, radius + 1)
    indices = np.arange(start, stop)
    predicted = _predictions(indices, segment, anchors, t_a, config)
    centers = _nearest(t_b, predicted)
    targets = centers[:, None] + deltas
    valid = (targets >= 0) & (targets < len(t_b))
    targets = targets.clip(0, len(t_b) - 1)
    valid &= abs(t_b[targets] - predicted[:, None]) <= config.refine_periods * period_a + 1e-9
    valid &= (predicted[:, None] >= t_b[0] - period_b / 2) & (predicted[:, None] <= t_b[-1] + period_b / 2)
    costs = np.empty(targets.shape, np.float32)
    flat = np.empty(len(indices), bool)
    for begin in range(0, len(indices), config.frame_chunk):
        end = min(len(indices), begin + config.frame_chunk)
        a, good_a = _normalise(sig_a[start + begin:start + end], config)
        b, good_b = _normalise(sig_b[targets[begin:end].ravel()], config)
        cost = _distance(a[:, None, :], b.reshape(end - begin, len(deltas), -1), config)
        cost[~good_b.reshape(cost.shape)] = np.inf
        cost[~good_a] = np.inf
        cost[~good_a, radius] = 0  # Flat retail frames keep the segment's timing prediction.
        costs[begin:end], flat[begin:end] = cost, ~good_a
    costs[~valid] = np.inf
    return targets, costs, flat, radius


def _exact_map(targets, costs, flat, radius, threshold, expected_step, config):
    """A small Viterbi band: timing wins ties, persistent frame shifts win evidence."""
    count, states = costs.shape
    if not count:
        return np.empty(0, np.int64), np.empty(0, np.float32)
    path = np.full(count, radius, np.int64)
    available = np.flatnonzero(costs.min(axis=1) <= threshold)
    splits = np.flatnonzero(np.diff(available) != 1) + 1
    for block in np.split(available, splits):
        if not len(block):
            continue
        first, end = int(block[0]), int(block[-1]) + 1
        local = costs[first:end]
        emission = np.maximum(local - local.min(axis=1, keepdims=True) - config.still_margin, 0)
        # A weak tie-break keeps the segment's predicted frame throughout a still shot.
        emission += abs(np.arange(states) - radius)[None, :] * config.timing_bias
        back = np.zeros((len(block), states), np.int16)
        score = emission[0].copy()
        for row in range(first + 1, end):
            # Penalise changes in actual timing, independent of the interpolated band
            # centre. A repeated frame costs one shift even if the centre also moves.
            steps = targets[row, None, :] - targets[row - 1, :, None]
            transitions = score[:, None] + config.shift_penalty * abs(steps - expected_step)
            transitions[steps <= 0] = np.inf
            back[row - first] = transitions.argmin(axis=0)
            score = transitions.min(axis=0) + emission[row - first]
        state = int(score.argmin())
        for row in range(end - 1, first - 1, -1):
            path[row] = state
            state = int(back[row - first, state])
    mapping = targets[np.arange(count), path].copy()
    distances = costs[np.arange(count), path].copy()
    mapping[distances > threshold] = -1
    last = -1
    for row, target in enumerate(mapping):
        if target < 0:
            continue
        if target <= last:
            mapping[row] = -1
        else:
            # If a repeated leak image permits the same skip on either side of a
            # frame, keep the earlier copy and place the skip after the duplicate.
            state = path[row]
            if last >= 0 and target - last > int(round(expected_step)) and state > 0:
                earlier = targets[row, state - 1]
                if earlier > last and costs[row, state - 1] <= distances[row] + config.still_margin:
                    mapping[row] = target = earlier
                    distances[row] = costs[row, state - 1]
            last = target
    return mapping, distances


def _runs(mapping):
    good = np.flatnonzero(mapping >= 0)
    if not len(good):
        return []
    splits = np.flatnonzero((np.diff(good) != 1) | (np.diff(mapping[good]) != 1)) + 1
    return [(int(part[0]), int(mapping[part[0]]), len(part)) for part in np.split(good, splits)]


def align_signatures(t_a, sig_a, t_b, sig_b, config: AlignConfig = AlignConfig()) -> tuple[list, list, dict]:
    """Return maximal +1 runs, retail-ordered offset segments, and measured statistics.

    Times must be strictly increasing presentation timestamps, in seconds. Signature
    geometry must agree; source video geometry need not agree.
    """
    started = time.perf_counter()
    t_a, t_b = np.asarray(t_a, np.float64), np.asarray(t_b, np.float64)
    sig_a, sig_b = np.asarray(sig_a), np.asarray(sig_b)
    for times, signatures in ((t_a, sig_a), (t_b, sig_b)):
        if times.ndim != 1 or signatures.ndim != 3 or len(times) != len(signatures):
            raise ValueError("times (N,) and signatures (N, h, w) must have matching lengths")
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("presentation times must be finite and strictly increasing")
    if sig_a.shape[1:] != sig_b.shape[1:]:
        raise ValueError("signature sizes must agree")
    mapping, distances = np.full(len(t_a), -1, np.int64), np.full(len(t_a), np.nan, np.float32)
    reported = []
    search_done = started
    if len(t_a) and len(t_b):
        anchors, candidates = _anchor_candidates(t_a, sig_a, t_b, sig_b, config)
        segments = _segments(anchors, candidates, t_a, config)
        search_done = time.perf_counter()
        # First extend each supported segment to the neighbouring evidence. At each
        # boundary choose a single switch minimising the two sides' distance sum.
        boundaries = [0]
        for left, right in zip(segments, segments[1:]):
            lo, hi = int(anchors[left["last"]]), int(anchors[right["first"]]) + 1
            _, lc, _, _ = _frame_costs(lo, hi, left, anchors, t_a, sig_a, t_b, sig_b, config)
            _, rc, _, _ = _frame_costs(lo, hi, right, anchors, t_a, sig_a, t_b, sig_b, config)
            lcost, rcost = np.minimum(lc.min(axis=1), config.max_frame_dist), np.minimum(rc.min(axis=1), config.max_frame_dist)
            total = np.r_[0, np.cumsum(lcost)] + np.r_[np.cumsum(rcost[::-1])[::-1], 0]
            boundaries.append(lo + int(total.argmin()))
        boundaries.append(len(t_a))
        for number, segment in enumerate(segments):
            start, stop = boundaries[number:number + 2]
            targets, costs, flat, radius = _frame_costs(start, stop, segment, anchors, t_a, sig_a, t_b, sig_b, config)
            points = [p for p in segment["track"]["points"] if segment["first"] <= p[0] <= segment["last"]]
            noise = float(np.median([p[3] for p in points]))
            threshold = min(config.max_frame_dist, noise + config.noise_margin)
            expected_step = float(np.median(np.diff(t_a)) / np.median(np.diff(t_b))) if min(len(t_a), len(t_b)) > 1 else 1.0
            matches, measured = _exact_map(targets, costs, flat, radius, threshold, expected_step, config)
            # Flat images carry no matching evidence outside the supported span.
            supported = (np.arange(start, stop) >= anchors[segment["first"]]) & (np.arange(start, stop) <= anchors[segment["last"]])
            matches[flat & ~supported] = -1
            mapping[start:stop], distances[start:stop] = matches, measured
            reported.append({"retail_start": start, "retail_stop": stop, "offset_s": segment["track"]["offset"],
                             "median_dist": noise, "anchors": len(points)})
    runs = _runs(mapping)
    matched_distances = distances[(mapping >= 0) & np.isfinite(distances)]
    stats = {"matched_share": float(np.mean(mapping >= 0)) if len(mapping) else 0.0,
             "distance_percentiles": {str(p): float(np.percentile(matched_distances, p)) for p in (50, 90, 95, 99)} if len(matched_distances) else {},
             "runs": len(runs), "timings": {"search_s": search_done - started, "refine_s": time.perf_counter() - search_done,
                                               "align_s": time.perf_counter() - started}}
    return runs, reported, stats


def align_pair(retail, leak, *, cache_dir=None, config: AlignConfig = AlignConfig(), decode_threads: int = 2) -> Alignment:
    started = time.perf_counter()
    retail, leak = Path(retail), Path(leak)
    cache_dir = Path(cache_dir) if cache_dir is not None else None
    t_a, sig_a, a, repaired_a = _cached_signatures(retail, cache_dir, config, decode_threads)
    t_b, sig_b, b, repaired_b = _cached_signatures(leak, cache_dir, config, decode_threads)
    decode_s = time.perf_counter() - started
    runs, segments, stats = align_signatures(t_a, sig_a, t_b, sig_b, config)
    stats["repaired_times"] = {"retail": repaired_a, "leak": repaired_b}
    stats["timings"].update(decode_s=decode_s, total_s=time.perf_counter() - started)
    return Alignment(str(retail), str(leak), len(t_a), len(t_b), a.fps, b.fps, (int(a.width), int(a.height)),
                     (int(b.width), int(b.height)), tuple(runs), tuple(segments), stats)


def _align_job(title, retail, leak, output, cache, decode_threads=2):
    cv2.setNumThreads(1)
    alignment = align_pair(retail, leak, cache_dir=cache, decode_threads=decode_threads)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_alignment(alignment, output)
    offsets = ", ".join(f"{segment['offset_s']:+.3f}s" for segment in alignment.segments) or "none"
    return f"{title}: {alignment.stats['matched_share']:.1%} matched; offsets [{offsets}]; {len(alignment.runs)} runs; {alignment.stats['timings']['total_s']:.1f}s"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reveil-align", description=__doc__.split("\n\n")[0])
    parser.add_argument("--censored", type=Path, help="retail title directories")
    parser.add_argument("--uncensored", type=Path, help="leak title directories")
    parser.add_argument("--retail", type=Path, help="one retail video")
    parser.add_argument("--leak", type=Path, help="one leak video")
    parser.add_argument("--out", type=Path, required=True, help="alignment directory, or JSON path in single-pair mode")
    parser.add_argument("--cache", type=Path, help="reuse decoded signature arrays")
    parser.add_argument("--workers", type=int, default=1, help="one title per spawned process")
    parser.add_argument("--decode-threads", type=int, default=2, help="threads per video decoder (default: 2)")
    parser.add_argument("--overwrite", action="store_true", help="replace completed alignment files")
    parser.add_argument("--only", nargs="+", help="align only these title directory names")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.decode_threads < 1:
        parser.error("--decode-threads must be positive")
    single = args.retail is not None or args.leak is not None
    if single:
        if args.retail is None or args.leak is None or args.censored is not None or args.uncensored is not None or args.only:
            parser.error("use --retail and --leak together, without directory-mode arguments")
        jobs = [(args.retail.stem, args.retail, args.leak, args.out, args.cache)]
    else:
        if args.censored is None or args.uncensored is None:
            parser.error("provide --censored and --uncensored, or --retail and --leak")
        pairs, warnings = find_real_pairs(args.censored, args.uncensored)
        for warning in warnings:
            print(f"skip {warning}", file=sys.stderr)
        jobs = [(p.title, p.retail, p.leak, args.out / f"{p.title}.json", args.cache) for p in pairs if args.only is None or p.title in args.only]
    pending = []
    for job in jobs:
        if job[3].exists() and not args.overwrite:
            print(f"{job[0]}: already done; skip", flush=True)
        else:
            pending.append(job)
    failed = 0
    if args.workers == 1:
        for job in pending:
            try:
                print(_align_job(*job, decode_threads=args.decode_threads), flush=True)
            except Exception as error:  # noqa: BLE001 - other titles can still finish
                print(f"{job[0]}: FAILED: {error!r}", file=sys.stderr, flush=True)
                failed += 1
    elif pending:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(_align_job, *job, decode_threads=args.decode_threads): job[0] for job in pending}
            for future in as_completed(futures):
                try:
                    print(future.result(), flush=True)
                except Exception as error:  # noqa: BLE001
                    print(f"{futures[future]}: FAILED: {error!r}", file=sys.stderr, flush=True)
                    failed += 1
    return int(bool(failed))


if __name__ == "__main__":
    sys.exit(main())
