from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from reveil.align import AlignConfig, align_pair, align_signatures, frame_signatures, main
from reveil.frame_map import load_alignment
from reveil.mosaic import MosaicStyle, censor
from reveil.realpairs import find_real_pairs
from reveil.video import write_video


FPS = 12
# Time-compress the synthetic pieces: 40 frames here represent a multi-second shot.
CONFIG = replace(AlignConfig(), anchor_interval_s=0.5)


def scenes(count, seed=0, *, still=False):
    """Smooth random moving blobs, with new shots every 40--80 frames."""
    rng = np.random.default_rng(seed)
    frames, age, length, base = [], 0, 0, None
    for index in range(count):
        if age == length:
            base = cv2.resize(rng.uniform(15, 240, (9, 16, 3)).astype(np.float32), (128, 72), interpolation=cv2.INTER_CUBIC)
            base = cv2.GaussianBlur(base, (0, 0), 1.4)
            age, length = 0, int(rng.integers(40, 81))
        shift = age * 0.65
        if still and 120 <= index < 180:
            # One near-static shot; encoding noise can beat the true adjacent frame.
            if index == 120:
                frozen = frames[-1].astype(np.float32)
            image = np.clip(frozen + (index - 120) * 0.006, 0, 255)
        else:
            matrix = np.float32([[1, 0, shift], [0, 1, 0.31 * age]])
            image = cv2.warpAffine(base, matrix, (128, 72), borderMode=cv2.BORDER_REFLECT)
        frames.append(np.rint(image).clip(0, 255).astype(np.uint8))
        age += 1
    return frames


def write_pair(tmp_path, retail, leak, *, fps_a=FPS, fps_b=FPS, config=CONFIG):
    a, b = tmp_path / "retail.mp4", tmp_path / "leak.mp4"
    write_video(a, retail, fps_a, crf=18, preset="ultrafast")
    write_video(b, leak, fps_b, crf=18, preset="ultrafast")
    return align_pair(a, b, config=config)


def assert_exact(alignment, expected, *, coverage=0.98, copies=False):
    actual = alignment.leak_indices(0, alignment.retail_frames)
    expected = np.asarray(expected)
    matched = actual >= 0
    if copies:
        assert np.all((actual[matched] == expected[matched]) | (actual[matched] == expected[matched] + 1))
    else:
        wrong = np.flatnonzero(matched & (actual != expected))
        assert not len(wrong), [(int(i), int(actual[i]), int(expected[i])) for i in wrong[:20]]
    eligible = expected >= 0
    assert matched[eligible].mean() >= coverage
    assert np.all(actual[~eligible] == -1)
    for segment in alignment.segments:
        indices = actual[segment["retail_start"]:segment["retail_stop"]]
        assert np.all(np.diff(indices[indices >= 0]) > 0)


def test_extra_head_and_tail(tmp_path):
    source = scenes(400)
    leak = scenes(37, 8) + source + scenes(20, 9)
    alignment = write_pair(tmp_path, source, leak)
    assert_exact(alignment, np.arange(400) + 37, coverage=1)
    assert alignment.runs == ((0, 37, 400),)
    assert alignment.stats["matched_share"] == 1
    assert alignment.stats["timings"]["align_s"] > 0


def test_inserted_leak_footage(tmp_path):
    source = scenes(420)
    leak = source[:201] + scenes(45, 6) + source[201:]
    expected = np.arange(420) + (np.arange(420) >= 201) * 45
    alignment = write_pair(tmp_path, source, leak)
    assert_exact(alignment, expected, coverage=1)
    assert alignment.runs == ((0, 0, 201), (201, 246, 219))


def test_repeats_and_different_fps(tmp_path):
    source, leak, expected = scenes(480), [], []
    for index, frame in enumerate(source):
        expected.append(len(leak))
        leak.append(frame)
        if (index + 1) % 97 == 0:
            leak.append(frame)
    # A different stream label, compensated by repeats, as in 29.97 vs 30 retail.
    alignment = write_pair(tmp_path, source, leak, fps_b=FPS * len(leak) / len(source))
    assert_exact(alignment, expected, coverage=1)
    assert len(alignment.runs) == 5


@pytest.mark.parametrize("unrelated", [False, True])
def test_retail_only_stretch(tmp_path, unrelated):
    source = scenes(420)
    if unrelated:
        retail = source[:210] + scenes(30, 25) + source[210:]
        expected = np.r_[np.arange(210), np.full(30, -1), np.arange(210, 420)]
        leak = source
    else:
        retail, leak = source, source[:210] + source[240:]
        expected = np.r_[np.arange(210), np.full(30, -1), np.arange(210, 390)]
    alignment = write_pair(tmp_path, retail, leak)
    assert_exact(alignment, expected)


def test_mosaic_and_different_resolution(tmp_path):
    source, retail = scenes(360), []
    style = MosaicStyle("pixelate", 6, (0, 0), True, False, 0, False, 0)
    for index, frame in enumerate(source):
        mask = np.zeros(frame.shape[:2], bool)
        x = 14 + index % 60
        mask[24:40, x:x + 24] = True  # About five percent once grid blocks are filled.
        retail.append(censor(frame, mask, style, (x, 24)))
    leak = [cv2.resize(frame, (256, 144), interpolation=cv2.INTER_NEAREST) for frame in source]
    alignment = write_pair(tmp_path, retail, leak)
    assert_exact(alignment, np.arange(360), coverage=1)
    assert alignment.retail_size == (128, 72)
    assert alignment.leak_size == (256, 144)


def test_content_in_second_half(tmp_path):
    source = scenes(360)
    alignment = write_pair(tmp_path, source, scenes(400, 15) + source)
    assert_exact(alignment, np.arange(360) + 400, coverage=1)


def test_reordered_edited_subset_and_reused_piece(tmp_path):
    leak = scenes(1500, 13)
    pieces = [(90, 170), (250, 310), (420, 520), (1250, 1330), (600, 670), (770, 880), (250, 310), (1050, 1100)]
    expected = np.concatenate([np.arange(start, stop) for start, stop in pieces])
    retail = [leak[index] for index in expected]
    alignment = write_pair(tmp_path, retail, leak)
    assert_exact(alignment, expected)
    offsets = [segment["offset_s"] for segment in alignment.segments]
    assert len(offsets) >= 8
    assert np.any(np.diff(offsets) < -10)


def test_double_frame_rate(tmp_path):
    source = scenes(360)
    leak = [frame for frame in source for _ in range(2)]
    alignment = write_pair(tmp_path, source, leak, fps_b=2 * FPS)
    assert_exact(alignment, np.arange(360) * 2, coverage=1, copies=True)


def test_retail_tail_has_no_counterpart(tmp_path):
    source = scenes(600)
    alignment = write_pair(tmp_path, source, source[:330])
    assert_exact(alignment, np.r_[np.arange(330), np.full(270, -1)])


def test_still_timing_and_persistent_frame_shift(tmp_path):
    source = scenes(420, still=True)
    leak = source[:240] + [source[239]] + source[240:]
    expected = np.arange(420) + (np.arange(420) >= 240)
    alignment = write_pair(tmp_path, source, leak)
    assert_exact(alignment, expected, coverage=1)
    assert alignment.runs == ((0, 0, 240), (240, 241, 180))


@pytest.mark.parametrize("seed", [2, 5])
def test_ambiguous_still_anchors_cannot_create_segments(tmp_path, seed):
    source = scenes(420, seed, still=True)
    leak = scenes(37, seed + 20) + source[:240] + [source[239]] + source[240:] + scenes(20, seed + 40)
    expected = np.arange(420) + 37 + (np.arange(420) >= 240)
    alignment = write_pair(tmp_path, source, leak)
    assert_exact(alignment, expected, coverage=1)


def test_flat_frames_are_not_evidence():
    flat = np.full((120, 18, 32), 16, np.uint8)
    runs, segments, stats = align_signatures(np.arange(120) / FPS, flat, np.arange(120) / FPS, flat, CONFIG)
    assert runs == segments == []
    assert stats["matched_share"] == 0


def test_flat_shot_inside_supported_segment(tmp_path):
    source = scenes(360)
    source[110:140] = [np.full((72, 128, 3), 16, np.uint8)] * 30
    alignment = write_pair(tmp_path, source, scenes(37, 8) + source)
    assert_exact(alignment, np.arange(360) + 37, coverage=1)


def test_dropped_frame_and_late_leak_start(tmp_path):
    source = scenes(420)
    leak = source[40:230] + source[231:]
    expected = np.r_[np.full(40, -1), np.arange(190), -1, np.arange(190, 379)]
    alignment = write_pair(tmp_path, source, leak)
    assert_exact(alignment, expected)


def test_pts_cache_and_cli(tmp_path, monkeypatch):
    source = scenes(180)
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    write_video(a, source, FPS, crf=18)
    write_video(b, scenes(15, 4) + source, FPS, crf=18)
    cache = tmp_path / "cache"
    first = align_pair(a, b, cache_dir=cache)
    assert len(list(cache.glob("*.npz"))) == 2
    times, signatures, info = frame_signatures(a)
    assert times.dtype == np.float64
    assert signatures.dtype == np.uint8
    assert signatures.shape == (180, 18, 32)
    np.testing.assert_allclose(np.diff(times), 1 / FPS)
    assert (info.width, info.height, info.frames) == (128, 72, 180)
    import reveil.align as module

    monkeypatch.setattr(module, "frame_signatures", lambda *a, **k: pytest.fail("cache was not reused"))
    second = align_pair(a, b, cache_dir=cache)
    assert first.runs == second.runs
    output = tmp_path / "out" / "title.json"
    argv = ["--retail", str(a), "--leak", str(b), "--out", str(output), "--cache", str(cache)]
    assert main(argv) == 0
    assert load_alignment(output).runs == first.runs
    before = output.stat().st_mtime_ns
    assert main(argv) == 0
    assert output.stat().st_mtime_ns == before


def test_find_real_pairs(tmp_path):
    a, b = tmp_path / "censored", tmp_path / "uncensored"
    a.mkdir()
    b.mkdir()

    def upload(root, title, name, data):
        directory = root / title
        directory.mkdir(exist_ok=True)
        (directory / name).write_bytes(data)

    for title in ("good", "ambiguous", "same", "missing-video", "missing-side"):
        upload(a, title, "retail.MP4", b"retail")
    upload(b, "good", "226655.xyz leak.mkv", b"leaked")
    upload(b, "ambiguous", "a.mp4", b"a")
    upload(b, "ambiguous", "b.mov", b"b")
    upload(b, "same", "different-name.mp4", b"retail")
    (b / "missing-video").mkdir()
    target_a, target_b = tmp_path / "shortcut-a", tmp_path / "shortcut-b"
    target_a.mkdir()
    target_b.mkdir()
    (target_a / "one.mp4").write_bytes(b"a")
    (target_b / "two.mp4").write_bytes(b"b")
    (a / "shortcut").symlink_to(target_a, target_is_directory=True)
    (b / "shortcut").symlink_to(target_b, target_is_directory=True)
    pairs, warnings = find_real_pairs(a, b)
    assert [p.title for p in pairs] == ["good", "shortcut"]
    assert len(warnings) == 4
    assert any("ambiguous" in warning and "2 videos" in warning for warning in warnings)
    assert any("same upload" in warning for warning in warnings)
    assert any("missing leak video" in warning for warning in warnings)
    assert any("missing leak directory" in warning for warning in warnings)
    assert pairs[0].retail.name != pairs[0].leak.name


def test_directory_cli_spawn_only_and_overwrite(tmp_path):
    a, b = tmp_path / "censored" / "title", tmp_path / "uncensored" / "title"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    source = scenes(180)
    write_video(a / "retail.mp4", source, FPS, crf=18)
    write_video(b / "226655.xyz leak.mp4", scenes(12, 9) + source, FPS, crf=18)
    output = tmp_path / "align"
    command = [sys.executable, "-m", "reveil.align", "--censored", str(a.parent), "--uncensored", str(b.parent),
               "--out", str(output), "--workers", "2", "--only", "title"]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "title:" in result.stdout and "matched" in result.stdout
    target = output / "title.json"
    assert_exact(load_alignment(target), np.arange(180) + 12, coverage=1)
    before = target.stat().st_mtime_ns
    subprocess.run(command, check=True, capture_output=True)
    assert target.stat().st_mtime_ns == before
    subprocess.run(command + ["--overwrite"], check=True, capture_output=True)
    assert target.stat().st_mtime_ns > before
    result = subprocess.run([sys.executable, "-m", "reveil.align", "--help"], check=True, capture_output=True, text=True)
    assert "--retail" in result.stdout and "--censored" in result.stdout


def test_invalid_timestamps_and_config():
    with pytest.raises(ValueError, match="positive"):
        AlignConfig(search_fps=0)
    with pytest.raises(ValueError, match="strictly increasing"):
        align_signatures([0, 0], np.zeros((2, 18, 32)), [0, 1], np.zeros((2, 18, 32)))
