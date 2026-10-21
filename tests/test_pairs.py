import json
from dataclasses import replace

import numpy as np
import pytest

from reveil.camera import CameraConfig
from reveil.make_pairs import find_videos, largest_first, load_config, main
from reveil.mosaic import MosaicConfig
from reveil.pairs import PairConfig, build_video_pairs, rebuild_dataset
from reveil.regions import Region, RegionSet
from reveil.video import probe, read_frames

SMALL = PairConfig(camera=CameraConfig(canvas=128), output_crf=4)
# Lossless x264 and blocks that stay visible on a 128 px canvas, so a pair differs only
# where it was censored.
VISIBLE = PairConfig(
    camera=CameraConfig(canvas=128),
    mosaic=MosaicConfig(block_fraction=(0.04, 0.05), blur_probability=0.0),
    crf=(0, 0),
    x265_probability=0.0,
    output_crf=0,
)


def moving_box(frames=160):
    return RegionSet({f: [Region((60 + f / 2, 70, 140 + f / 2, 150))] for f in range(frames)})


def test_pairs_differ_only_where_censored(make_video, tmp_path):
    video = make_video()
    out = tmp_path / "data"
    rows = build_video_pairs(video, moving_box(), out, VISIBLE, seed=1)
    assert len(rows) == 1
    row = rows[0]
    clean = np.stack(list(read_frames(out / row["video"])))
    mosaic = np.stack(list(read_frames(out / row["reference_video"])))
    assert clean.shape == mosaic.shape == (121, 128, 128, 3)
    assert probe(out / row["video"]).fps == 24.0

    # Map the region of the middle frame onto the canvas and compare inside and outside it.
    frame = 60
    x0, y0, x1, y1 = row["crops"][frame]
    scale = 128 / max(x1 - x0, y1 - y0)
    source = (60 + (row["start"] + frame) / 2, 70, 140 + (row["start"] + frame) / 2, 150)
    cx0, cy0 = int((source[0] - x0) * scale) + 2, int((source[1] - y0) * scale) + 2
    cx1, cy1 = int((source[2] - x0) * scale) - 2, int((source[3] - y0) * scale) - 2
    diff = np.abs(clean[frame].astype(int) - mosaic[frame].astype(int)).mean(axis=2)
    inside = diff[cy0:cy1, cx0:cx1].mean()
    outside = np.ones_like(diff, bool)
    pad = max(4, int(row["style"]["dilate"] * scale) + int(row["style"]["block"] * scale) + 4)
    outside[max(cy0 - pad, 0) : cy1 + pad, max(cx0 - pad, 0) : cx1 + pad] = False
    assert inside > 3 * diff[outside].mean()


def test_finished_sources_are_skipped_and_the_dataset_is_rebuilt(make_video, tmp_path):
    video = make_video()
    out = tmp_path / "data"
    first = build_video_pairs(video, moving_box(), out, SMALL, seed=1)
    stamp = (out / first[0]["video"]).stat().st_mtime_ns
    again = build_video_pairs(video, moving_box(), out, SMALL, seed=1)
    assert again == first and (out / first[0]["video"]).stat().st_mtime_ns == stamp
    assert rebuild_dataset(out, "a caption") == 1
    (line,) = (out / "dataset.jsonl").read_text().splitlines()
    assert json.loads(line) == {"video": first[0]["video"], "reference_video": first[0]["reference_video"], "caption": "a caption"}


def test_short_tracks_make_no_pairs(make_video, tmp_path):
    video = make_video(frames=100)
    rows = build_video_pairs(video, moving_box(100), tmp_path / "data", SMALL)
    assert rows == []
    assert json.loads((tmp_path / "data" / "meta" / "clip.json").read_text())["windows"] == []


def test_cli_smoke_run_with_random_regions(make_video, tmp_path):
    video = make_video(frames=130)
    config = tmp_path / "pairs.yaml"
    config.write_text("pairs:\n  output_crf: 4\n  crf: [20, 22]\ncamera:\n  canvas: 64\n")
    loaded = load_config(config)
    assert loaded.crf == (20, 22) and loaded.camera.canvas == 64
    out = tmp_path / "data"
    code = main(["--videos", str(video.parent), "--random-regions", "--out", str(out), "--workers", "1",
                 "--config", str(config), "--caption", "x"])
    assert code == 0
    rows = [json.loads(line) for line in (out / "dataset.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["caption"] == "x"
    assert (out / rows[0]["reference_video"]).is_file()


def test_cli_explains_missing_videos_and_region_files(make_video, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit, match="no videos"):
        main(["--videos", str(empty), "--random-regions", "--out", str(tmp_path / "out")])
    video = make_video(frames=10)
    (tmp_path / "regions").mkdir()
    with pytest.raises(SystemExit, match=r"none of the 1 videos has a region file .*clip\.json"):
        main(["--videos", str(video.parent), "--regions", str(tmp_path / "regions"), "--out", str(tmp_path / "out")])


def test_find_videos_follows_symlinked_folders_once(tmp_path):
    shared = tmp_path / "shared" / "FC2"  # a Drive shortcut's target
    (shared / "unknow 1" / "FC2-PPV-1").mkdir(parents=True)
    (shared / "unknow 1" / "FC2-PPV-1" / "a-1.MP4").touch()
    (shared / "b.mkv").touch()
    (shared / "notes.txt").touch()
    (shared / "loop").symlink_to(shared, target_is_directory=True)
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "FC2").symlink_to(shared, target_is_directory=True)
    found = find_videos([sources])
    assert [p.relative_to(sources).as_posix() for p in found] == ["FC2/b.mkv", "FC2/unknow 1/FC2-PPV-1/a-1.MP4"]
    assert find_videos([sources / "FC2"]) == [sources / "FC2" / "b.mkv", sources / "FC2/unknow 1/FC2-PPV-1/a-1.MP4"]


def test_config_rejects_unknown_fields(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text("mosaic:\n  no_such_field: 1\n")
    try:
        load_config(config)
    except TypeError:
        pass
    else:
        raise AssertionError("unknown field accepted")
    assert replace(PairConfig(), hop=104).hop == 104


def test_the_biggest_videos_go_first(tmp_path):
    sizes = {"a.mp4": 10, "b.mp4": 30, "c.mp4": 20}
    for name, size in sizes.items():
        (tmp_path / name).write_bytes(bytes(size))
    assert [p.name for p in largest_first(sorted(tmp_path.iterdir()))] == ["b.mp4", "c.mp4", "a.mp4"]
