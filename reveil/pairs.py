"""Training pairs for the restoration IC-LoRA: the same window, clean and censored.

Every region track of a source video is cut into ``window``-frame windows, each with its
own mosaic style and compression. The mosaic is drawn at source resolution over the
window's work area (the 16-pixel-aligned union of its camera views); the censored work
area is encoded with x264 or x265 at a random CRF, as a released video is, and decoded
again. The clean and the censored work area are then framed onto the canvas by the same
camera views (``reveil.camera``), so the pair differs only by the censoring.

Output, ready for ltx-trainer's ``process_dataset.py``::

    out/clean/<key>.mp4     target: canvas x canvas, ``window`` frames at ``PairConfig.fps``
    out/mosaic/<key>.mp4    reference
    out/meta/<stem>.json    every window of one source with its parameters
    out/dataset.jsonl       {"video", "reference_video", "caption"} rows, rebuilt from meta/
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from reveil.camera import CameraConfig, CropFrame, crop_to_canvas, plan_crops
from reveil.mosaic import MosaicConfig, MosaicStyle, censor, sample_style
from reveil.regions import Box, Region, RegionSet, clip_box, load_regions, random_regions, rasterize
from reveil.tracks import build_tracks, window_starts
from reveil.video import VideoInfo, VideoWriter, probe, read_frames, write_video

logger = logging.getLogger(__name__)

DEFAULT_CAPTION = "A sharp, detailed, high quality video."
WORK_ALIGN = 16  # work areas start on the codec's 16-pixel macroblock grid


@dataclass(frozen=True)
class PairConfig:
    window: int = 121  # frames per pair; 1 + 8k for the LTX VAE
    hop: int = 121  # frames between window starts within a track; = window means no overlap
    max_windows_per_video: int | None = None
    fps: float = 24.0  # frame rate written into the pairs; every source frame is kept
    caption: str = DEFAULT_CAPTION
    crf: tuple[int, int] = (18, 32)  # compression of the censored video, drawn per window
    x265_probability: float = 0.3
    preset: str = "veryfast"
    output_crf: int = 10  # near-lossless storage of the finished pairs
    camera: CameraConfig = field(default_factory=CameraConfig)
    mosaic: MosaicConfig = field(default_factory=MosaicConfig)


@dataclass
class WindowPlan:
    key: str
    track: int
    start: int  # source frame index of the window's first frame
    boxes: list[Box]  # union region box per frame, source pixels
    regions: list[list[Region]]
    crops: list[CropFrame]
    work: tuple[int, int, int, int]  # source area the window reads, x0, y0, x1, y1
    style: MosaicStyle
    codec: str
    crf: int


def safe_stem(path: str | Path) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(path).stem)


def _seed_for(seed: int, stem: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{stem}".encode()).digest()[:8], "little")


def _clipped(regions: RegionSet, info: VideoInfo) -> RegionSet:
    """Regions limited to the frame and to the frames the video has."""
    frames = {}
    for index, items in regions.frames.items():
        if not 0 <= index < info.frames:
            continue
        kept = [Region(box, r.polygon) for r in items if (box := clip_box(r.box, info.width, info.height))]
        if kept:
            frames[index] = kept
    return RegionSet(frames, regions.cuts)


def _work_area(crops: list[CropFrame], info: VideoInfo) -> tuple[int, int, int, int]:
    x0 = min(c.box[0] for c in crops) // WORK_ALIGN * WORK_ALIGN
    y0 = min(c.box[1] for c in crops) // WORK_ALIGN * WORK_ALIGN
    x1 = min(info.width, -(-max(c.box[2] for c in crops) // WORK_ALIGN) * WORK_ALIGN)
    y1 = min(info.height, -(-max(c.box[3] for c in crops) // WORK_ALIGN) * WORK_ALIGN)
    return x0, y0, x1, y1


def plan_windows(
    regions: RegionSet, info: VideoInfo, stem: str, config: PairConfig, rng: np.random.Generator
) -> list[WindowPlan]:
    """The windows of one source video, ordered by start frame."""
    plans = []
    for number, track in enumerate(build_tracks(_clipped(regions, info))):
        length = len(track.boxes)
        if length < config.window:
            continue
        crops = plan_crops(track.boxes, info.width, info.height, info.fps, config.camera)
        offset = int(rng.integers(0, min(config.hop, length - config.window) + 1))
        for start in window_starts(length, config.window, config.hop, offset):
            stop = start + config.window
            plans.append(
                WindowPlan(
                    key=f"{stem}-t{number:03d}-f{track.start + start:06d}",
                    track=number,
                    start=track.start + start,
                    boxes=track.boxes[start:stop],
                    regions=track.regions[start:stop],
                    crops=crops[start:stop],
                    work=_work_area(crops[start:stop], info),
                    style=sample_style(rng, info.width, info.height, config.mosaic),
                    codec="libx265" if rng.random() < config.x265_probability else "libx264",
                    crf=int(rng.integers(config.crf[0], config.crf[1] + 1)),
                )
            )
    if config.max_windows_per_video is not None and len(plans) > config.max_windows_per_video:
        keep = rng.choice(len(plans), size=config.max_windows_per_video, replace=False)
        plans = [plans[i] for i in sorted(keep)]
    return sorted(plans, key=lambda plan: plan.start)


def _grid_origin(plan: WindowPlan, index: int) -> tuple[int, int]:
    """A block corner of frame ``index`` in work-area coordinates."""
    x0, y0 = plan.work[:2]
    if plan.style.moving_grid:
        box = plan.boxes[index]
        return math.floor(box[0]) - x0, math.floor(box[1]) - y0
    return plan.style.grid_origin[0] - x0, plan.style.grid_origin[1] - y0


class _WindowRender:
    """One window while the source decodes: clean canvases are kept, censored work-area
    frames go straight into the compression encoder."""

    def __init__(self, plan: WindowPlan, temp_dir: Path, config: PairConfig) -> None:
        self.plan = plan
        x0, y0, x1, y1 = plan.work
        self._compressed = temp_dir / f"{plan.key}.mp4"
        self._encoder = VideoWriter(
            self._compressed, x1 - x0, y1 - y0, config.fps, codec=plan.codec, crf=plan.crf, preset=config.preset
        )
        self.clean: list[np.ndarray] = []

    @property
    def done(self) -> bool:
        return len(self.clean) == len(self.plan.crops)

    def add(self, frame: np.ndarray) -> None:
        plan, index = self.plan, len(self.clean)
        x0, y0, x1, y1 = plan.work
        work = frame[y0:y1, x0:x1]
        self.clean.append(crop_to_canvas(work, plan.crops[index].shifted(x0, y0)))
        mask = rasterize(plan.regions[index], (x0, y0), work.shape[:2], ellipse=plan.style.ellipse)
        self._encoder.write(censor(work, mask, plan.style, _grid_origin(plan, index)))

    def finish(self, out_dir: Path, config: PairConfig, source: Path) -> dict:
        self._encoder.close()
        plan = self.plan
        x0, y0, x1, y1 = plan.work
        mosaic = [
            crop_to_canvas(frame[: y1 - y0, : x1 - x0], crop.shifted(x0, y0))
            for frame, crop in zip(read_frames(self._compressed), plan.crops)
        ]
        self._compressed.unlink()
        if len(mosaic) != len(self.clean):
            raise RuntimeError(f"{plan.key}: {len(mosaic)} of {len(self.clean)} censored frames came back")
        write_video(out_dir / "clean" / f"{plan.key}.mp4", self.clean, config.fps, crf=config.output_crf)
        write_video(out_dir / "mosaic" / f"{plan.key}.mp4", mosaic, config.fps, crf=config.output_crf)
        self.clean = []
        return {
            "video": f"clean/{plan.key}.mp4",
            "reference_video": f"mosaic/{plan.key}.mp4",
            "caption": config.caption,
            "key": plan.key,
            "source": str(source),
            "track": plan.track,
            "start": plan.start,
            "frames": len(plan.crops),
            "work": list(plan.work),
            "crops": [list(crop.box) for crop in plan.crops],
            "style": asdict(plan.style),
            "codec": plan.codec,
            "crf": plan.crf,
        }

    def abort(self) -> None:
        try:
            self._encoder.close()
        except Exception:  # noqa: BLE001 - best effort while unwinding
            pass
        self._compressed.unlink(missing_ok=True)


def _render(source: Path, plans: list[WindowPlan], out_dir: Path, temp_dir: Path, config: PairConfig) -> list[dict]:
    """Decode ``source`` once, rendering every window as its frames arrive."""
    rows: list[dict] = []
    active: list[_WindowRender] = []
    upcoming = iter(plans)
    plan = next(upcoming, None)
    last_frame = max(p.start + len(p.crops) for p in plans) - 1
    try:
        for index, frame in enumerate(read_frames(source)):
            while plan is not None and plan.start == index:
                active.append(_WindowRender(plan, temp_dir, config))
                plan = next(upcoming, None)
            for render in active:
                render.add(frame)
            for render in [r for r in active if r.done]:
                rows.append(render.finish(out_dir, config, source))
                active.remove(render)
            if index >= last_frame:
                break
    finally:
        for render in active:
            render.abort()
    missing = len(active) + (plan is not None) + sum(1 for _ in upcoming)
    if missing:
        logger.warning("%s: ended early, %d planned windows dropped", source, missing)
    return rows


def _write_atomic(path: Path, text: str) -> None:
    partial = path.with_name(path.name + ".part")
    partial.write_text(text, encoding="utf-8")
    os.replace(partial, path)


def build_video_pairs(
    source: str | Path,
    regions: str | Path | RegionSet | None,
    out_dir: str | Path,
    config: PairConfig = PairConfig(),
    *,
    seed: int = 0,
    overwrite: bool = False,
) -> list[dict]:
    """Every training pair of one source video, as meta rows. ``regions`` is a regions file,
    a ``RegionSet``, or None for random drifting boxes (pipeline smoke tests only). A source
    whose meta file exists is skipped unless ``overwrite``."""
    source, out_dir = Path(source), Path(out_dir)
    stem = safe_stem(source)
    meta_path = out_dir / "meta" / f"{stem}.json"
    if meta_path.exists() and not overwrite:
        return json.loads(meta_path.read_text(encoding="utf-8"))["windows"]
    for sub in ("clean", "mosaic", "meta"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    info = probe(source)
    rng = np.random.default_rng(_seed_for(seed, stem))
    if regions is None:
        region_set = random_regions(info.frames, info.width, info.height, rng)
    elif isinstance(regions, RegionSet):
        region_set = regions
    else:
        region_set = load_regions(regions)
    plans = plan_windows(region_set, info, stem, config, rng)
    rows: list[dict] = []
    if plans:
        with tempfile.TemporaryDirectory(prefix="reveil-") as temp_dir:
            rows = _render(source, plans, out_dir, Path(temp_dir), config)
    meta = {"source": str(source), "width": info.width, "height": info.height, "fps": info.fps, "windows": rows}
    text = json.dumps(meta)
    _write_atomic(meta_path, text)
    return json.loads(text)["windows"]  # as a rerun reads them back


def rebuild_dataset(out_dir: str | Path, caption: str) -> int:
    """Write ``dataset.jsonl`` from every source's meta file (pairs whose files exist), all
    with ``caption``; returns the row count."""
    out_dir = Path(out_dir)
    lines = []
    for meta_path in sorted((out_dir / "meta").glob("*.json")):
        for row in json.loads(meta_path.read_text(encoding="utf-8"))["windows"]:
            if (out_dir / row["video"]).is_file() and (out_dir / row["reference_video"]).is_file():
                entry = {"video": row["video"], "reference_video": row["reference_video"], "caption": caption}
                lines.append(json.dumps(entry) + "\n")
    _write_atomic(out_dir / "dataset.jsonl", "".join(lines))
    return len(lines)
