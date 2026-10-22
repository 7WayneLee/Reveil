import json

import cv2
import numpy as np
import pytest

from conftest import write_yolo_onnx
from reveil.annotate import (
    AnnotateConfig, CutDetector, annotate_video, detection_regions, main, mark_flat, mask_regions, smooth_tracks,
)
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
        self.shapes = set()

    def __call__(self, frames):
        results = []
        for frame in frames:
            index = self.calls
            self.calls += 1
            self.shapes.add(frame.shape)
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


def test_excluded_boxes_are_cut_out_of_the_kept_regions():
    found = FrameDetections(np.array([[20.0, 20, 60, 60]]), np.array([0.9]), np.array([0]), None,
                            excluded=np.array([[40.0, 40, 80, 80]]))
    (region,) = detection_regions(found, 100, 100, AnnotateConfig(grow=0.0))
    assert region.box == pytest.approx((20, 20, 59, 59), abs=1)
    from reveil.regions import rasterize

    mask = rasterize([region], (0, 0), (100, 100), ellipse=False)
    assert mask[25, 25] and mask[55, 55] and not mask[65, 65]  # the detected box stays whole; only the rest is cut
    grown = detection_regions(found, 100, 100, AnnotateConfig(grow=0.1))  # 10 px of growth
    mask = rasterize(grown, (0, 0), (100, 100), ellipse=False)
    assert mask[15, 15] and mask[59, 59] and not mask[62, 50] and mask[62, 30]  # no growth into the excluded box
    apart = FrameDetections(found.boxes, found.scores, found.classes, None, excluded=np.array([[70.0, 70, 90, 90]]))
    assert [r.box for r in detection_regions(apart, 100, 100, AnnotateConfig(grow=0.0))] == [(20, 20, 60, 60)]


def test_regions_carry_their_detection_scores():
    found = FrameDetections(np.array([[0.0, 10, 30, 40], [60.0, 10, 90, 40]]), np.array([0.9, 0.3]), np.array([0, 0]),
                            None, excluded=np.array([[20.0, 30, 40, 50]]))
    first, second = sorted(detection_regions(found, 200, 100, AnnotateConfig(grow=0.0)), key=lambda r: r.box)
    assert first.polygon is not None and first.score == pytest.approx(0.9)  # cut by the excluded box
    assert second.score == pytest.approx(0.3)
    assert Region.from_json(second.to_json()).score == pytest.approx(0.3)


def test_detections_mostly_inside_a_doubt_box_are_doubtful():
    face = np.array([[0.0, 0, 100, 100]])
    found = FrameDetections(np.array([[20.0, 50, 80, 110], [60.0, 60, 140, 140]]), np.array([0.9, 0.8]),
                            np.array([0, 0]), None, doubt=face)  # 5/6 and 1/4 inside the face
    inside, apart = sorted(detection_regions(found, 200, 200, AnnotateConfig(grow=0.0)), key=lambda r: r.box)
    assert inside.doubt == "inside" and apart.doubt is None
    assert Region.from_json(inside.to_json()).doubt == "inside"
    assert all(r.doubt is None for r in detection_regions(found, 200, 200, AnnotateConfig(grow=0.0, doubt_inside=0.9)))
    # cut by an excluded box, a region is doubtful only when all its detections are
    cut = FrameDetections(found.boxes[:1], found.scores[:1], found.classes[:1], None,
                          excluded=np.array([[70.0, 100, 90, 120]]), doubt=face)
    (region,) = detection_regions(cut, 200, 200, AnnotateConfig(grow=0.0))
    assert region.polygon is not None and region.doubt == "inside"


def test_regions_over_an_even_patch_are_doubtful():
    frame = np.full((100, 200, 3), 12, np.uint8)
    frame[:, 100:] = np.random.default_rng(0).integers(0, 255, size=(100, 100, 3), dtype=np.uint8)
    dark, busy = mark_flat([Region((10, 10, 60, 60)), Region((120, 10, 170, 60))], frame, 8.0)
    assert dark.doubt == "flat" and busy.doubt is None
    assert mark_flat([Region((10, 10, 60, 60))], frame, 0.0)[0].doubt is None  # turned off


def test_mostly_doubtful_tracks_are_dropped():
    def frames(doubtful):
        return {f: [Region((10, 10, 40, 40), score=0.9, doubt="inside" if f in doubtful else None)] for f in range(60)}

    kept, dropped = smooth_tracks(RegionSet(frames(range(35))), 30.0, AnnotateConfig(min_track_score=0.0))
    assert not kept.frames and len(dropped) == 60
    kept, dropped = smooth_tracks(RegionSet(frames(range(20))), 30.0, AnnotateConfig())
    assert len(kept.frames) == 60 and not dropped  # a real one, now and then inside a face
    assert kept.frames[5][0].doubt == "inside"


def test_tracks_the_detector_was_unsure_of_are_dropped():
    frames = {f: [Region((10 + f, 10, 40 + f, 40), score=0.8), Region((200, 10, 230, 40), score=0.3)]
              for f in range(60)}
    frames[30] = frames[30][:1] + [Region((200, 10, 230, 40), score=0.9)]  # one confident frame is not enough
    kept, dropped = smooth_tracks(RegionSet(frames), 30.0, AnnotateConfig())
    assert all(len(v) == 1 and v[0].box[0] < 100 for v in kept.frames.values()) and len(kept.frames) == 60
    assert sorted(dropped) == list(range(60)) and all(v[0].box[0] == 200 for v in dropped.values())
    # a doubtful region overlapping a sure one is still dropped
    frames = {f: [Region((100, 100, 200, 200), score=0.9), Region((180, 80, 400, 300), score=0.3)] for f in range(60)}
    kept, dropped = smooth_tracks(RegionSet(frames), 30.0, AnnotateConfig())
    assert all([r.box for r in v] == [(100, 100, 200, 200)] for v in kept.frames.values()) and len(dropped) == 60


def test_cut_detector_flags_hard_cuts_only():
    frames = list(square_frames(20, cut_at=10))
    detect = CutDetector(30.0)
    assert [i for i, f in enumerate(frames) if detect(f)] == [10]


def test_cut_detector_rgb_and_grey_agree():
    frames = list(square_frames(20, cut_at=10))
    rgb, grey, thumbnail = CutDetector(30), CutDetector(30), CutDetector(30)
    full, small = [], []
    for frame in frames:
        y = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        full.append(grey.grey(y))
        small.append(thumbnail.grey(cv2.resize(y, (64, 36), interpolation=cv2.INTER_AREA)))
    assert [rgb(frame) for frame in frames] == full == small
    assert np.flatnonzero(full).tolist() == [10]


def test_tracks_fill_gaps_by_moving_regions_and_drop_short_ones():
    square = lambda x: Region.from_json({"polygon": [[x, 0], [x + 10, 0], [x + 10, 10], [x, 10]]})  # noqa: E731
    frames = {f: [square(f)] for f in range(40) if not 10 <= f < 14}
    frames.update({f: [Region((500, 500, 520, 520))] for f in range(100, 105)})
    smoothed, dropped = smooth_tracks(RegionSet(frames), 30.0, AnnotateConfig(max_gap_seconds=0.2, min_track_seconds=0.5))
    assert not dropped
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


class BoxDetector:
    """Finds the same box on every frame."""

    def __init__(self, box):
        self.box = np.array([box], dtype=float)

    def __call__(self, frames):
        return [FrameDetections(self.box, np.array([0.9]), np.array([0]), None) for _ in frames]


def test_a_box_on_dark_frames_is_dropped(tmp_path):
    video = tmp_path / "dark.mp4"
    write_video(video, (np.full((240, 320, 3), 14, np.uint8) for _ in range(60)), 30.0, crf=16)
    out = tmp_path / "dark.json"
    summary = annotate_video(video, BoxDetector((100, 80, 200, 160)), out, AnnotateConfig(), preview_path=tmp_path / "p.jpg")
    assert summary["region_frames"] == 0 and summary["dropped_seconds"] == 2.0
    assert not load_regions(out).frames
    assert json.loads(out.read_text())["dropped"]["0"][0]["doubt"] == "flat"
    summary = annotate_video(tmp_path / "dark.mp4", BoxDetector((100, 80, 200, 160)), out, AnnotateConfig(min_contrast=0))
    assert summary["region_frames"] == 60


def test_detection_runs_at_the_model_size_and_maps_back_to_video_pixels(tmp_path):
    video = tmp_path / "clip.mp4"
    write_video(video, square_frames(40), 30.0, crf=16)  # 320x240
    detector = BrightDetector()
    detector.size = 160  # a model with a 160 px input
    annotate_video(video, detector, tmp_path / "clip.json", AnnotateConfig(min_track_seconds=0.5),
                   preview_path=tmp_path / "p.jpg")
    assert detector.shapes == {(120, 160, 3)}
    x0, y0, x1, y1 = load_regions(tmp_path / "clip.json").frames[20][0].box
    assert abs(x0 - 70) <= 3 and abs(x1 - 129) <= 3 and abs(y0 - 80) <= 3 and abs(y1 - 139) <= 3  # the square, grown
    assert (tmp_path / "p.jpg").stat().st_size > 1000


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
            "--min-track-seconds", "0.5", "--preview-dir", str(tmp_path / "previews"),
            "--min-contrast", "0"]  # the stand-in model's box ignores the picture
    assert main(args) == 0
    data = json.loads((out / "a.json").read_text())
    assert len(data["frames"]) == 40
    assert (tmp_path / "previews" / "a.jpg").is_file()
    stamp = (out / "a.json").stat().st_mtime_ns
    assert main(args) == 0 and (out / "a.json").stat().st_mtime_ns == stamp
    assert main([*args, "--conf", "0.5"]) == 0 and (out / "a.json").stat().st_mtime_ns != stamp  # new settings
