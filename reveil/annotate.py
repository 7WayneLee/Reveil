"""Region files from a detector: where each source video gets its synthetic mosaics.

    python -m reveil.annotate --videos SOURCES/ --out REGIONS/ --model nsfw.onnx

Every frame (or every ``stride``-th) goes through a YOLO detect or segment model
(``reveil.detect``): a segment model's mask becomes outline polygons, a detect model's
boxes become box regions, both grown a little. Boxes of excluded classes (say the anus,
which releases leave uncensored) are cut out of them. Regions are linked into tracks; tracks
shorter than ``min_track_seconds`` are dropped and gaps up to ``max_gap_seconds`` are
filled by moving the nearer frame's regions along the interpolated box. Scene cuts found
from frame differences go into the file too, so no track or training window spans one.
Videos that already have a region file are skipped, so an interrupted run resumes, as long
as the settings recorded next to the region files are unchanged; otherwise every video is
annotated again.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from reveil.detect import DetectorConfig, FrameDetections, YoloOnnx
from reveil.regions import Box, Region, RegionSet, clip_box, save_regions, union_box
from reveil.tracks import build_tracks
from reveil.video import probe, read_frames

PREVIEW_WIDTH = 480
SETTINGS_FILE = "annotate_settings.json"
SETTINGS_VERSION = 2  # raised when the annotation itself changes, so old region files are redone


@dataclass(frozen=True)
class AnnotateConfig:
    batch: int = 16  # frames per detector call
    stride: int = 1  # run the detector on every stride-th frame; the gaps are filled
    grow: float = 0.01  # region growth, fraction of the frame's short side
    min_area: float = 0.0005  # smallest region kept, fraction of the frame area
    simplify: float = 0.005  # polygon simplification, fraction of the outline's length
    max_gap_seconds: float = 0.5  # missed detections bridged within a track
    min_track_seconds: float = 1.0  # shorter tracks are dropped
    cut_threshold: float = 30.0  # mean grey difference (0-255) of 64x36 thumbnails at a cut
    preview_frames: int = 12


def mask_regions(mask: np.ndarray, *, grow_px: int, min_area_px: float, simplify: float) -> list[Region]:
    """One polygon region per separate area of a bool mask: areas under ``min_area_px`` are
    dropped, the rest grown by ``grow_px``."""
    pixels = mask.astype(np.uint8)
    contours, _ = cv2.findContours(pixels, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area_px]
    if contours and grow_px > 0:
        pixels = np.zeros_like(pixels)
        cv2.drawContours(pixels, contours, -1, 1, thickness=-1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow_px + 1, 2 * grow_px + 1))
        contours, _ = cv2.findContours(cv2.dilate(pixels, kernel), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for contour in contours:
        outline = cv2.approxPolyDP(contour, simplify * cv2.arcLength(contour, True), True) if simplify > 0 else contour
        if len(outline) < 3:
            outline = contour
        if len(outline) >= 3:
            regions.append(Region.from_json({"polygon": outline[:, 0, :].tolist()}))
    return regions


def _grown_boxes(boxes: np.ndarray, grow_px: int, frame_w: int, frame_h: int) -> list[Box]:
    grown = (clip_box((x0 - grow_px, y0 - grow_px, x1 + grow_px, y1 + grow_px), frame_w, frame_h)
             for x0, y0, x1, y1 in boxes.tolist())
    return [box for box in grown if box is not None]


def _overlap(a: Box, b: Box) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _paint(mask: np.ndarray, box: Box, value: bool) -> None:
    x0, y0, x1, y1 = box
    mask[int(y0) : int(np.ceil(y1)), int(x0) : int(np.ceil(x1))] = value


def detection_regions(detections: FrameDetections, frame_w: int, frame_h: int, config: AnnotateConfig) -> list[Region]:
    """Regions of the kept detections, grown a little; the boxes of excluded classes (grown
    as much) are cut out of them."""
    grow_px = round(config.grow * min(frame_w, frame_h))
    min_area_px = config.min_area * frame_w * frame_h
    excluded = _grown_boxes(detections.excluded, grow_px, frame_w, frame_h)
    if detections.mask is not None:
        mask = detections.mask
        if excluded:
            mask = mask.copy()
            for box in excluded:
                _paint(mask, box, False)
        return mask_regions(mask, grow_px=grow_px, min_area_px=min_area_px, simplify=config.simplify)
    boxes = [box for box in _grown_boxes(detections.boxes, grow_px, frame_w, frame_h)
             if (box[2] - box[0]) * (box[3] - box[1]) >= min_area_px]
    if not any(_overlap(box, cut) for box in boxes for cut in excluded):
        return [Region(box) for box in boxes]
    mask = np.zeros((frame_h, frame_w), dtype=bool)
    for box in boxes:
        _paint(mask, box, True)
    for box in excluded:
        _paint(mask, box, False)
    return mask_regions(mask, grow_px=0, min_area_px=min_area_px, simplify=config.simplify)


class CutDetector:
    """Flags a frame whose small grey thumbnail differs from the previous frame's by more
    than ``threshold`` (mean absolute difference, 0-255)."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self._last: np.ndarray | None = None

    def __call__(self, frame: np.ndarray) -> bool:
        grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        thumb = cv2.resize(grey, (64, 36), interpolation=cv2.INTER_AREA).astype(np.int16)
        cut = self._last is not None and float(np.abs(thumb - self._last).mean()) > self.threshold
        self._last = thumb
        return cut


def _work_size(width: int, height: int, side: int | None) -> tuple[int, int]:
    """The frame size detection runs at: the video scaled down to fit a ``side`` square."""
    if not side or max(width, height) <= side:
        return width, height
    scale = side / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def _move(region: Region, source: Box, target: Box) -> Region:
    """``region`` carried from ``source`` box to ``target`` box (shifted and scaled)."""
    sx = (target[2] - target[0]) / max(source[2] - source[0], 1e-6)
    sy = (target[3] - target[1]) / max(source[3] - source[1], 1e-6)
    if region.polygon is not None:
        points = np.stack(
            [target[0] + (region.polygon[:, 0] - source[0]) * sx, target[1] + (region.polygon[:, 1] - source[1]) * sy],
            axis=1,
        )
        return Region.from_json({"polygon": points.tolist()})
    x0, y0, x1, y1 = region.box
    return Region(
        (target[0] + (x0 - source[0]) * sx, target[1] + (y0 - source[1]) * sy,
         target[0] + (x1 - source[0]) * sx, target[1] + (y1 - source[1]) * sy)
    )


def smooth_tracks(regions: RegionSet, fps: float, config: AnnotateConfig) -> RegionSet:
    """Per-frame detections as tracks: short tracks dropped, gaps filled."""
    max_gap = max(config.stride - 1, round(config.max_gap_seconds * fps))
    min_frames = round(config.min_track_seconds * fps)
    frames: dict[int, list[Region]] = {}
    for track in build_tracks(regions, max_gap=max_gap):
        if len(track.boxes) < min_frames:
            continue
        for offset, (box, group, filled) in enumerate(zip(track.boxes, track.regions, track.filled)):
            if filled:
                source = union_box(group)
                group = [_move(region, source, box) for region in group]
            frames.setdefault(track.start + offset, []).extend(group)
    return RegionSet(frames, regions.cuts)


def _draw(frame: np.ndarray, regions: list[Region], scale: float) -> np.ndarray:
    image = frame.copy()
    for region in regions:
        if region.polygon is not None:
            points = np.round(region.polygon * scale).astype(np.int32)
            cv2.polylines(image, [points], True, (0, 255, 0), 2)
        else:
            x0, y0, x1, y1 = (round(v * scale) for v in region.box)
            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
    return image


def write_preview(samples: dict[int, np.ndarray], regions: RegionSet, scale: float, path: Path, count: int) -> None:
    """A grid of sampled frames (RGB, already scaled by ``scale``) with their regions outlined."""
    marked = [i for i in sorted(samples) if i in regions.frames] or sorted(samples)
    if not marked:
        return
    chosen = [marked[round(k * (len(marked) - 1) / max(count - 1, 1))] for k in range(min(count, len(marked)))]
    tiles = []
    for index in dict.fromkeys(chosen):
        tile = _draw(samples[index], regions.frames.get(index, []), scale)
        cv2.putText(tile, str(index), (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        tiles.append(tile)
    columns = min(4, len(tiles))
    blank = np.zeros_like(tiles[0])
    tiles += [blank] * (-len(tiles) % columns)
    rows = [np.concatenate(tiles[i : i + columns], axis=1) for i in range(0, len(tiles), columns)]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])


def annotate_video(
    video: str | Path,
    detector,
    out_path: str | Path,
    config: AnnotateConfig = AnnotateConfig(),
    *,
    preview_path: str | Path | None = None,
) -> dict:
    """Write the region file of ``video``; ``detector(frames)`` returns ``FrameDetections``
    per frame. Returns a summary of what was found.

    When the detector has a ``size`` (its square input side), frames are decoded straight
    to that size, which is all the detail the model sees anyway, and the regions are mapped
    back to video pixels."""
    info = probe(video)
    work_w, work_h = _work_size(info.width, info.height, getattr(detector, "size", None))
    work_box = (-0.5, -0.5, work_w - 0.5, work_h - 0.5)  # pixel centres line up
    video_box = (-0.5, -0.5, info.width - 0.5, info.height - 0.5)
    detected: dict[int, list[Region]] = {}
    cuts: set[int] = set()
    is_cut = CutDetector(config.cut_threshold)
    batch: list[np.ndarray] = []
    batch_frames: list[int] = []
    scale = min(1.0, PREVIEW_WIDTH / info.width)
    preview_size = (max(1, round(info.width * scale)), max(1, round(info.height * scale)))
    sample_every = max(1, info.frames // max(1, config.preview_frames * 4))
    samples: dict[int, np.ndarray] = {}

    def flush() -> None:
        for index, found in zip(batch_frames, detector(batch)):
            regions = detection_regions(found, work_w, work_h, config)
            if regions and work_box != video_box:
                regions = [_move(region, work_box, video_box) for region in regions]
            if regions:
                detected[index] = regions
        batch.clear()
        batch_frames.clear()

    frame_count = 0
    size = None if work_box == video_box else (work_w, work_h)
    for index, frame in enumerate(read_frames(video, size)):
        frame_count = index + 1
        if is_cut(frame) and index:
            cuts.add(index)
        if preview_path is not None and index % sample_every == 0:
            samples[index] = cv2.resize(frame, preview_size, interpolation=cv2.INTER_AREA)
        if index % config.stride == 0:
            batch.append(frame)
            batch_frames.append(index)
            if len(batch) == config.batch:
                flush()
    if batch:
        flush()

    regions = smooth_tracks(RegionSet(detected, frozenset(cuts)), info.fps, config)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_regions(regions, out_path)
    if preview_path is not None:
        write_preview(samples, regions, scale, Path(preview_path), config.preview_frames)
    return {
        "size": f"{info.width}x{info.height}",
        "frames": frame_count,
        "detected_frames": len(detected),
        "region_frames": len(regions.frames),
        "region_seconds": round(len(regions.frames) / info.fps, 1),
        "cuts": len(cuts),
    }


_detector = None


def _start_worker(model: str, detector_config: DetectorConfig) -> None:
    global _detector
    _detector = YoloOnnx(model, detector_config)


def _annotate_in_worker(video: Path, out_path: Path, config: AnnotateConfig, preview_path: Path | None) -> dict:
    return annotate_video(video, _detector, out_path, config, preview_path=preview_path)


def main(argv: list[str] | None = None) -> int:
    from reveil.make_pairs import find_videos

    parser = argparse.ArgumentParser(prog="reveil-annotate", description=__doc__.split("\n\n")[0])
    parser.add_argument("--videos", nargs="+", type=Path, required=True, help="video files or directories")
    parser.add_argument("--out", type=Path, required=True, help="directory for the <video name>.json region files")
    parser.add_argument("--model", type=Path, required=True, help="YOLO detect/segment model exported to ONNX")
    parser.add_argument("--classes", nargs="*", default=[], help="class names or indices to keep (default: all)")
    parser.add_argument("--exclude-classes", nargs="*", default=[],
                        help="classes whose boxes are cut out of the kept regions (e.g. ANUS_EXPOSED)")
    parser.add_argument("--conf", type=float, default=DetectorConfig.conf)
    parser.add_argument("--iou", type=float, default=DetectorConfig.iou)
    parser.add_argument("--imgsz", type=int, help="model input size (default: from the model)")
    parser.add_argument("--batch", type=int, default=AnnotateConfig.batch)
    parser.add_argument("--stride", type=int, default=AnnotateConfig.stride)
    parser.add_argument("--grow", type=float, default=AnnotateConfig.grow)
    parser.add_argument("--min-track-seconds", type=float, default=AnnotateConfig.min_track_seconds)
    parser.add_argument("--max-gap-seconds", type=float, default=AnnotateConfig.max_gap_seconds)
    parser.add_argument("--preview-dir", type=Path, help="write <video name>.jpg previews here")
    parser.add_argument("--workers", type=int, default=1, help="videos annotated at once, each with its own model")
    parser.add_argument("--overwrite", action="store_true", help="annotate videos that already have a region file")
    args = parser.parse_args(argv)

    detector_config = DetectorConfig(
        conf=args.conf, iou=args.iou, imgsz=args.imgsz, classes=tuple(args.classes), exclude=tuple(args.exclude_classes)
    )
    config = AnnotateConfig(
        batch=args.batch, stride=args.stride, grow=args.grow,
        min_track_seconds=args.min_track_seconds, max_gap_seconds=args.max_gap_seconds,
    )
    videos = find_videos(args.videos)
    if not videos:
        sys.exit(f"reveil-annotate: no videos in {', '.join(map(str, args.videos))}")
    seen: dict[str, Path] = {}
    for video in videos:
        if video.stem in seen:
            sys.exit(f"reveil-annotate: {video} and {seen[video.stem]} share the name {video.stem!r}; rename one")
        seen[video.stem] = video

    detector = YoloOnnx(args.model, detector_config)  # also checks the model and the class lists
    print(f"model {args.model.name}: {detector.task}, input {detector.size}, classes {detector.names}, "
          f"keeping {detector.kept_classes}, cutting out {detector.excluded_classes}, on {detector.providers[0]}",
          flush=True)

    # Region files are reused only if they were written with the same settings.
    settings = {
        "version": SETTINGS_VERSION, "model": args.model.name, "classes": detector.kept_classes,
        "exclude": detector.excluded_classes, "conf": args.conf, "iou": args.iou, "imgsz": detector.size,
        "stride": config.stride, "grow": config.grow,
        "min_track_seconds": config.min_track_seconds, "max_gap_seconds": config.max_gap_seconds,
    }
    settings_path = args.out / SETTINGS_FILE
    try:
        changed = json.loads(settings_path.read_text()) != settings
    except (OSError, ValueError):
        changed = True
    done = [v for v in videos if (args.out / f"{v.stem}.json").is_file()]
    if changed and done:
        print(f"the settings differ from those of the {len(done)} existing region files: annotating them again",
              flush=True)
    jobs = [v for v in videos if args.overwrite or changed or v not in done]
    print(f"{len(videos)} videos, {len(videos) - len(jobs)} already annotated", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings, indent=1))

    def report(done: int, video: Path, result: dict | None, error: Exception | None) -> None:
        if error is None:
            print(f"[{done}/{len(jobs)}] {video.name}: {result}", flush=True)
        else:
            print(f"[{done}/{len(jobs)}] {video.name}: FAILED: {error!r}", file=sys.stderr, flush=True)

    def preview(video: Path) -> Path | None:
        return args.preview_dir / f"{video.stem}.jpg" if args.preview_dir else None

    failed = 0
    if args.workers <= 1:
        for done, video in enumerate(jobs, start=1):
            try:
                result = annotate_video(video, detector, args.out / f"{video.stem}.json", config, preview_path=preview(video))
                report(done, video, result, None)
            except Exception as error:  # noqa: BLE001 - report and keep going with the other videos
                failed += 1
                report(done, video, None, error)
    else:
        del detector
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=context, initializer=_start_worker,
            initargs=(str(args.model), detector_config),
        ) as pool:
            futures = {
                pool.submit(_annotate_in_worker, video, args.out / f"{video.stem}.json", config, preview(video)): video
                for video in jobs
            }
            for done, future in enumerate(as_completed(futures), start=1):
                try:
                    report(done, futures[future], future.result(), None)
                except Exception as error:  # noqa: BLE001
                    failed += 1
                    report(done, futures[future], None, error)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
