import json
import time

import cv2
import numpy as np
import pytest

from conftest import textured_frames
from reveil.frame_map import Alignment, save_alignment
from reveil.mosaic import pixelate
from reveil.realmask import (
    MaskConfig, _merge_overlays, estimate_block, frame_mosaic, main, mosaic_regions_for_pair,
)
from reveil.regions import Overlay, load_regions, rasterize, save_regions
from reveil.tracks import build_tracks
from reveil.video import write_video


@pytest.mark.parametrize("period", [12, 20, 37, 64])
@pytest.mark.parametrize("missing", [False, True])
def test_block_period_with_noise_and_missing_axis(period, missing):
    rng = np.random.default_rng(period)
    x, y = np.zeros(800), np.zeros(600)
    x[7::period], y[11::period] = 20, 15
    x += rng.normal(0, 0.5, x.shape)
    if not missing:
        y += rng.normal(0, 0.5, y.shape)
    else:
        y[:] = 0
    assert estimate_block(x, y, MaskConfig(scale=1)) == pytest.approx(period, rel=0.03)


def test_period_accepts_double_axis_and_rejects_unrelated_axes():
    def pulses(period):
        p = np.zeros(800)
        p[7::period] = 20
        return p
    assert estimate_block(pulses(20), pulses(40), MaskConfig(scale=1)) == pytest.approx(20, rel=0.03)
    assert estimate_block(pulses(20), pulses(29), MaskConfig(scale=1)) is None
    assert estimate_block(np.zeros(100), np.zeros(100), MaskConfig()) is None


def _truth(index, width=640, height=360):
    block, origin = (12, (5, 7)) if index < 80 else (20, (9, 3))
    x, y, w, h = 130 + index // 4, 110 + int(8 * np.sin(index / 20)), 128, 84
    mask = np.zeros((height, width), dtype=bool)
    if index >= 20:
        x0, y0 = origin[0] + (x - origin[0]) // block * block, origin[1] + (y - origin[1]) // block * block
        x1 = origin[0] + int(np.ceil((x + w - origin[0]) / block)) * block
        y1 = origin[1] + int(np.ceil((y + h - origin[1]) / block)) * block
        mask[y0:y1, x0:x1] = True
    return mask, block, origin


@pytest.fixture(scope="module")
def real_pair(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic-realmask")
    rng = np.random.default_rng(8)
    texture = cv2.resize(rng.normal(0, 28, (180, 320, 3)).astype(np.float32), (640, 360),
                         interpolation=cv2.INTER_LINEAR)
    frames = []
    for image in textured_frames(160, 640, 360, seed=3):
        image = np.clip(image.astype(np.float32) + texture, 0, 255).astype(np.uint8)
        image[:55, 580:] = 110
        frames.append(image)
    retail = []
    for index, image in enumerate(frames):
        truth, block, origin = _truth(index)
        image = pixelate(image, truth, block, origin)
        # A small retail-only logo with several high-contrast strokes.
        cv2.rectangle(image, (600, 10), (631, 33), (20, 20, 20), -1)
        cv2.line(image, (603, 13), (627, 29), (240, 240, 240), 5)
        cv2.line(image, (603, 29), (627, 13), (240, 240, 240), 5)
        grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)[..., None]
        retail.append(np.clip(grey + 0.98 * (image.astype(float) - grey), 16, 235).astype(np.uint8))
    leak_path, retail_path = root / "leak.mp4", root / "retail.mp4"
    write_video(leak_path, [frames[0]] * 5 + frames, 30, crf=16)
    write_video(retail_path, retail, 30, crf=16)
    alignment = Alignment(str(retail_path), str(leak_path), 160, 165, 30, 30,
                          (640, 360), (640, 360), ((0, 5, 160),))
    started = time.perf_counter()
    result = mosaic_regions_for_pair(alignment, MaskConfig())
    return root, alignment, result, time.perf_counter() - started


def test_encoded_moving_mosaics_logo_and_tracks(real_pair):
    root, _, result, _ = real_pair
    good = []
    for index in range(0, 160, 2):
        items = result.frames.get(index, [])
        if index < 20:
            assert not items
            continue
        truth, block, _ = _truth(index)
        predicted = rasterize(items, (0, 0), truth.shape, ellipse=False)
        good.append((predicted & truth).sum() / truth.sum() >= 0.85 and predicted.sum() <= 2 * truth.sum())
        assert not predicted[10:34, 600:632].any()
        assert all(item.block == pytest.approx(block, rel=0.15) for item in items)
    assert np.mean(good) >= 0.9, (np.mean(good), result.stats)
    assert all(index % 2 == 0 for index in result.frames)
    assert len(result.overlays) == 1
    overlay = result.overlays[0]
    assert overlay.side == "retail" and overlay.frames == (0, 160)
    assert overlay.box[0] <= 600 and overlay.box[1] <= 10
    assert overlay.box[2] >= 632 and overlay.box[3] >= 34
    path = root / "regions.json"
    save_regions(result, path, extra={"settings": {"scale": 0.5}, "stats": result.stats})
    loaded = load_regions(path)
    assert loaded.overlays == result.overlays
    assert loaded.frames.keys() == result.frames.keys()
    assert all(region.block is not None for items in loaded.frames.values() for region in items)
    tracks = build_tracks(loaded)
    assert 1 <= len(tracks) <= 4
    assert max(track.stop - track.start for track in tracks) >= 120


def test_cli_metadata_skip_filter_and_spawn(real_pair, tmp_path, capsys):
    _, alignment, _, _ = real_pair
    aligns, out = tmp_path / "align", tmp_path / "regions"
    aligns.mkdir()
    save_alignment(alignment, aligns / "title.json")
    # Filter a corrupt unrelated file before loading, and exercise the spawn worker.
    (aligns / "unrelated.json").write_text("not json")
    args = ["--align-dir", str(aligns), "--out", str(out), "--only", "title", "--workers", "2"]
    assert main(args) == 0
    data = json.loads((out / "title.json").read_text())
    assert data["settings"]["alignment"] == str(aligns / "title.json")
    assert data["settings"]["stride"] == 2
    assert data["stats"]["frames_analysed"] == 80
    assert data["stats"]["overlay_count"] == 1
    assert all("block" in item for items in data["frames"].values() for item in items)
    before = (out / "title.json").stat().st_mtime_ns
    assert main(args) == 0
    assert (out / "title.json").stat().st_mtime_ns == before
    assert main(args + ["--overwrite"]) == 0
    assert (out / "title.json").stat().st_mtime_ns != before
    assert "title:" in capsys.readouterr().out


def test_overlay_chunks_merge_only_when_consecutive_and_same_side():
    config = MaskConfig()
    overlays = []
    _merge_overlays(overlays, [Overlay((10, 10, 40, 30), "retail", 0.8, (0, 100))], config)
    _merge_overlays(overlays, [Overlay((11, 10, 41, 30), "retail", 0.6, (100, 200))], config)
    assert len(overlays) == 1 and overlays[0].frames == (0, 200)
    assert overlays[0].frequency == pytest.approx(0.7)
    _merge_overlays(overlays, [Overlay((10, 10, 40, 30), "retail", 0.8, (300, 400)),
                              Overlay((10, 10, 40, 30), "leak", 0.9, (200, 300))], config)
    assert len(overlays) == 3


def test_forward_decode_cuts_unmatched_frames_and_unequal_sizes(monkeypatch):
    import reveil.realmask as module

    decoded = {"retail": [], "leak": []}
    retail = [np.full((90, 160, 3), 50 if i < 3 else 180, np.uint8) for i in range(8)]
    leak = [cv2.resize(frame, (320, 180)) for frame in retail]

    def read(path):
        frames = retail if path == "retail" else leak
        for index, frame in enumerate(frames):
            decoded[path].append(index)
            yield frame

    monkeypatch.setattr(module, "read_frames", read)
    alignment = Alignment("retail", "leak", 8, 8, 30, 30, (160, 90), (320, 180), ((0, 0, 2), (4, 4, 4)))
    progress = []
    result = mosaic_regions_for_pair(alignment, MaskConfig(), progress=lambda done, total: progress.append((done, total)))
    assert result.cuts == {3}  # odd skipped frame still goes through CutDetector
    assert result.stats["frames_analysed"] == 3
    assert decoded == {"retail": list(range(8)), "leak": list(range(7))}
    assert progress == [(i, 8) for i in range(1, 9)]
    assert not result.frames


def test_logos_appear_only_in_their_chunks_on_either_side(monkeypatch):
    import reveil.realmask as module

    def read(path):
        for index in range(120):
            frame = np.broadcast_to((40 + np.arange(160, dtype=np.uint8))[None, :, None], (90, 160, 3)).copy()
            if (path == "retail" and 40 <= index < 80) or (path == "leak" and index >= 80):
                cv2.rectangle(frame, (136, 6), (155, 21), (20, 20, 20), -1)
                cv2.line(frame, (139, 8), (151, 18), (235, 235, 235), 3)
                cv2.line(frame, (139, 18), (151, 8), (235, 235, 235), 3)
            yield frame

    monkeypatch.setattr(module, "read_frames", read)
    alignment = Alignment("retail", "leak", 120, 120, 30, 30, (160, 90), (160, 90), ((0, 0, 120),))
    result = mosaic_regions_for_pair(alignment, MaskConfig(scale=1, overlay_chunk=40))
    assert not result.frames
    assert [(overlay.side, overlay.frames) for overlay in result.overlays] == [
        ("retail", (40, 80)), ("leak", (80, 120)),
    ]
    assert all(overlay.frequency == 1 for overlay in result.overlays)


def test_multiple_mosaics_and_sharpness_is_not_mosaic():
    rng = np.random.default_rng(3)
    leak = rng.integers(30, 225, (360, 640, 3), dtype=np.uint8)
    first, second = np.zeros((360, 640), bool), np.zeros((360, 640), bool)
    first[50:145, 60:190], second[200:305, 390:530] = True, True
    retail = pixelate(pixelate(leak, first, 12, (5, 7)), second, 20, (9, 3))
    grey = lambda image: cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    areas = frame_mosaic(grey(retail), grey(leak), MaskConfig())
    assert len(areas) == 2
    assert sorted(a.block for a in areas) == pytest.approx([12, 20], rel=0.15)
    assert not frame_mosaic(grey(cv2.GaussianBlur(leak, (3, 3), 0.7)), grey(leak), MaskConfig())


@pytest.mark.parametrize("scale", [0.5, 1.0])
def test_larger_leak_is_resized_and_measurements_use_retail_pixels(scale):
    rng = np.random.default_rng(21)
    leak = rng.integers(30, 225, (360, 640, 3), dtype=np.uint8)
    base = cv2.resize(leak, (320, 180), interpolation=cv2.INTER_AREA)
    mask = np.zeros((180, 320), bool)
    mask[50:125, 80:200] = True
    retail = pixelate(base, mask, 12, (5, 7))
    areas = frame_mosaic(cv2.cvtColor(retail, cv2.COLOR_RGB2GRAY), cv2.cvtColor(leak, cv2.COLOR_RGB2GRAY),
                         MaskConfig(scale=scale))
    assert len(areas) == 1
    area = areas[0]
    assert area.mask.shape == mask.shape and area.mask.dtype == bool
    assert area.block == pytest.approx(12, rel=0.15)
    assert area.phase == pytest.approx((5, 7), abs=1.1)
    assert area.retail_std < 3 and area.retail_std < 0.3 * area.leak_std


@pytest.mark.parametrize("kwargs", [{"stride": 0}, {"scale": 0}, {"scale": 2}, {"block_max": 7}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        MaskConfig(**kwargs)


def test_backward_alignment_is_skipped_and_analysis_resumes(monkeypatch, tmp_path, capsys):
    import reveil.realmask as module

    decoded = {"retail": [], "leak": []}

    def read(path):
        for index in range(16):
            decoded[path].append(index)
            yield np.full((90, 160, 3), 100, np.uint8)

    monkeypatch.setattr(module, "read_frames", read)
    alignment = Alignment("retail", "leak", 16, 16, 30, 30, (160, 90), (160, 90),
                          ((0, 4, 4), (4, 0, 4), (8, 8, 8)))
    result = mosaic_regions_for_pair(alignment, MaskConfig())
    assert result.stats["skipped_backward_frames"] == 4
    assert result.stats["frames_analysed"] == 6
    assert result.stats["frames_with_mosaic"] == 0
    assert decoded == {"retail": list(range(16)), "leak": list(range(15))}
    aligns = tmp_path / "align"
    aligns.mkdir()
    save_alignment(alignment, aligns / "backward.json")
    assert main(["--align-dir", str(aligns), "--out", str(tmp_path / "regions")]) == 0
    warning = capsys.readouterr().err
    assert warning.count("warning:") == 1 and "skipped 4 retail frames" in warning
