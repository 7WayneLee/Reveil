"""Build (clean, mosaic) training pairs from source videos and their region files.

    python -m reveil.make_pairs --videos SOURCES/ --regions REGIONS/ --out DATA/

Each video ``name.ext`` needs ``REGIONS/name.json`` (see ``reveil.regions``); videos without
one are skipped. ``--random-regions`` uses drifting random boxes instead, to smoke-test the
pipeline only. Finished sources are skipped on a rerun, so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import fields, replace
from pathlib import Path

import yaml

from reveil.camera import CameraConfig
from reveil.mosaic import MosaicConfig
from reveil.pairs import PairConfig, build_video_pairs, rebuild_dataset, safe_stem

VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts", ".wmv", ".flv"}


def _with_tuples(dataclass_type, values: dict) -> dict:
    """YAML lists as tuples for the tuple-typed range fields."""
    tuple_fields = {f.name for f in fields(dataclass_type) if str(f.type).startswith("tuple")}
    return {k: tuple(v) if k in tuple_fields and isinstance(v, list) else v for k, v in values.items()}


def load_config(path: Path | None) -> PairConfig:
    """``PairConfig`` with the overrides of a YAML file holding optional ``pairs``,
    ``camera`` and ``mosaic`` sections (field names as in the dataclasses)."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path else None
    data = data or {}
    unknown = set(data) - {"pairs", "camera", "mosaic"}
    if unknown:
        raise ValueError(f"{path}: unknown sections {sorted(unknown)}")
    camera = replace(CameraConfig(), **_with_tuples(CameraConfig, data.get("camera") or {}))
    mosaic = replace(MosaicConfig(), **_with_tuples(MosaicConfig, data.get("mosaic") or {}))
    return replace(PairConfig(), **_with_tuples(PairConfig, data.get("pairs") or {}), camera=camera, mosaic=mosaic)


def find_videos(paths: list[Path]) -> list[Path]:
    videos = []
    for path in paths:
        if path.is_dir():
            videos.extend(p for p in sorted(path.rglob("*")) if p.suffix.lower() in VIDEO_SUFFIXES)
        elif path.is_file():
            videos.append(path)
        else:
            raise FileNotFoundError(path)
    return videos


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reveil-make-pairs", description=__doc__.split("\n\n")[0])
    parser.add_argument("--videos", nargs="+", type=Path, required=True, help="video files or directories")
    parser.add_argument("--regions", type=Path, help="directory of <video name>.json region files")
    parser.add_argument("--random-regions", action="store_true", help="random drifting boxes (smoke tests only)")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--config", type=Path, help="YAML overrides: pairs / camera / mosaic sections")
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-windows-per-video", type=int)
    parser.add_argument("--caption", help="the one caption every pair is trained with")
    parser.add_argument("--overwrite", action="store_true", help="rebuild sources that are already done")
    args = parser.parse_args(argv)
    if args.regions is None and not args.random_regions:
        parser.error("--regions is required (or --random-regions for a smoke test)")

    config = load_config(args.config)
    if args.max_windows_per_video is not None:
        config = replace(config, max_windows_per_video=args.max_windows_per_video)
    if args.caption is not None:
        config = replace(config, caption=args.caption)

    videos = find_videos(args.videos)
    if not videos:
        sys.exit(f"reveil-make-pairs: no videos ({', '.join(sorted(VIDEO_SUFFIXES))}) in {', '.join(map(str, args.videos))}")
    jobs: list[tuple[Path, Path | None]] = []
    stems: dict[str, Path] = {}
    for video in videos:
        stem = safe_stem(video)
        if stem in stems:
            sys.exit(f"reveil-make-pairs: {video} and {stems[stem]} share the name {stem!r}; rename one")
        stems[stem] = video
        if args.random_regions:
            jobs.append((video, None))
            continue
        regions = args.regions / f"{video.stem}.json"
        if regions.is_file():
            jobs.append((video, regions))
        else:
            print(f"skip {video}: no {regions}", file=sys.stderr)
    if not jobs:
        sys.exit(
            f"reveil-make-pairs: none of the {len(videos)} videos has a region file in {args.regions} "
            f"(e.g. {videos[0].stem}.json); add region files, or use --random-regions for a smoke test"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    failed = 0
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        futures = {
            pool.submit(build_video_pairs, video, regions, args.out, config, seed=args.seed, overwrite=args.overwrite): video
            for video, regions in jobs
        }
        for done, future in enumerate(as_completed(futures), start=1):
            video = futures[future]
            try:
                print(f"[{done}/{len(jobs)}] {video.name}: {len(future.result())} pairs", flush=True)
            except Exception as error:  # noqa: BLE001 - report and keep going with the other videos
                failed += 1
                print(f"[{done}/{len(jobs)}] {video.name}: FAILED: {error!r}", file=sys.stderr, flush=True)
    total = rebuild_dataset(args.out, config.caption)
    print(f"{total} pairs in {args.out / 'dataset.jsonl'}" + (f"; {failed} videos failed" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
