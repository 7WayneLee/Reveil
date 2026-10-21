import json

import numpy as np
import pytest

from conftest import write_yolo_onnx
from reveil.annotate import AnnotateConfig, CutDetector, annotate_video, detection_regions, main, mask_regions, smooth_tracks
from reveil.detect import FrameDetections
from reveil.pairs import PairConfig, build_video_pairs
from reveil.camera import CameraConfig
from reveil.regions import Region, RegionSet, load_regions
from reveil.video import write_video


def square_frames(count, width=320, height=240, *, cut_at=None):
    """A bright square drifting over dark noise; after ``cut_at`` the background turns light."""
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 40, size=(height, width, 3), dtype=np.uint8)
    for t in range(count):
        frame = noise.copy()
        if cut_at is not None and t >= cut_at:
            frame += 150
        x = 60 + t // 2
        frame[80:140, x : x + 60] = 255
        yield frame


class BrightDetector:
    """Finds the bright square; misses the frames in ``blind``."""

    def __init__(self, blind=()):
        self.blind = set(blind)
        self.calls = 0

    def __call__(self, frames):
        results = []
        for frame in frames:
            index = self.calls
            self.calls += 1
            mask = (frame.min(axis=2) > 240) & (index not in self.blind)
            results.append(FrameDetections(np.zeros((0, 4)), np.zeros(0), np.zeros(0, int), mask))
        return results


def test_mask_regions_split_areas_drop_specks_and_grow():
    mask = np.zeros((100, 100), bool)
    mask[10:30, 10:30] = True
    mask[60:80, 50:90] = True
    mask[95, 95] = True
    regions = mask_regions(mask, grow_px=0, min_area_px=20, simplify=0.0)
    assert len(regions) == 2
    grown = mask_regions(mask, grow_px=3, min_area_px=20, simplify=0.0)
    widths = sorted(r.box[2] - r.box[0] for r in grown)
    assert widths[0] >= 19 + 6


def test_detect_model_boxes_are_grown_and_clipped():
    found = FrameDetections(np.array([[0.0, 10, 30, 40]]), np.array([0.9]), np.array([0]), None)
    (region,) = detection_regions(found, 200, 100, AnnotateConfig(grow=0.05))
    assert region.box == (0, 5, 35, 45)


def test_cut_detector_flags_hard_cuts_only():
    frames = list(square_frames(20, cut_at=10))
    detect = CutDetector(30.0)
    assert [i for i, f in enumerate(frames) if detect(f)] == [10]


def test_tracks_fill_gaps_by_moving_regions_and_drop_short_ones():
    square = lambda x: Region.from_json({"polygon": [[x, 0], [x + 10, 0], [x + 10, 10], [x, 10]]})  # noqa: E731
    frames = {f: [square(f)] for f in range(40) if not 10 <= f < 14}
    frames.update({f: [Region((500, 500, 520, 520))] for f in range(100, 105)})
    smoothed = smooth_tracks(RegionSet(frames), 30.0, AnnotateConfig(max_gap_seconds=0.2, min_track_seconds=0.5))
    assert sorted(smoothed.frames) == list(range(40))
    (moved,) = smoothed.frames[12]
    assert moved.box == pytest.approx((12, 0, 22, 10))


def test_annotate_video_writes_regions_cuts_and_a_preview(tmp_path):
    video = tmp_path / "clip.mp4"
    write_video(video, square_frames(300, cut_at=200), 30.0, crf=16)
    out = tmp_path / "regions" / "clip.json"
    summary = annotate_video(
        video, BrightDetector(blind=range(50, 56)), out, AnnotateConfig(batch=7), preview_path=tmp_path / "p.jpg"
    )
    regions = load_regions(out)
    assert regions.cuts == {200}
    assert sorted(regions.frames) == list(range(300))  # the blind frames were filled
    assert summary["detected_frames"] == 294 and summary["cuts"] == 1
    x0, y0, x1, y1 = regions.frames[53][0].box
    assert abs(x0 - (60 + 53 // 2)) <= 4 and abs(y0 - 80) <= 4
    assert (tmp_path / "p.jpg").stat().st_size > 1000

    # The region file feeds the pair builder: the 200 frames before the cut make one
    # window, the 100 after it are too short for one.
    rows = build_video_pairs(video, out, tmp_path / "data", PairConfig(camera=CameraConfig(canvas=64)), seed=0)
    assert len(rows) == 1 and rows[0]["start"] + 121 <= 200


def test_strided_detection_still_covers_every_frame(tmp_path):
    video = tmp_path / "clip.mp4"
    write_video(video, square_frames(60), 30.0, crf=16)
    out = tmp_path / "clip.json"
    summary = annotate_video(video, BrightDetector(), out, AnnotateConfig(stride=3, min_track_seconds=0.5))
    assert summary["detected_frames"] == 20
    assert sorted(load_regions(out).frames) == list(range(58))  # up to the last detected frame


def test_cli_with_an_onnx_model_skips_finished_videos(tmp_path):
    videos = tmp_path / "videos"
    videos.mkdir()
    write_video(videos / "a.mp4", square_frames(40, width=128, height=96), 30.0, crf=16)
    model = write_yolo_onnx(tmp_path / "seg.onnx")
    out = tmp_path / "regions"
    args = ["--videos", str(videos), "--out", str(out), "--model", str(model), "--classes", "region",
            "--min-track-seconds", "0.5", "--preview-dir", str(tmp_path / "previews")]
    assert main(args) == 0
    data = json.loads((out / "a.json").read_text())
    assert len(data["frames"]) == 40
    assert (tmp_path / "previews" / "a.jpg").is_file()
    stamp = (out / "a.json").stat().st_mtime_ns
    assert main(args) == 0 and (out / "a.json").stat().st_mtime_ns == stamp
