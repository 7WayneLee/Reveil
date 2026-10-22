"""Real-pair training behavior, using generated video only."""

import json
from dataclasses import replace

import cv2
import numpy as np
import pytest

from conftest import textured_frames
from reveil.camera import CameraConfig, CropFrame, crop_to_canvas
from reveil.frame_map import Alignment, save_alignment
from reveil.mosaic import MosaicStyle, censor
from reveil.pairs import _seed_for, rebuild_dataset, safe_stem
from reveil.real_pairs import RealPairConfig, _fit_colour, _masks, load_config, main, plan_real_windows, build_title_pairs
from reveil.regions import Overlay, Region, RegionSet, rasterize, save_regions
from reveil.video import probe, read_frames, write_video


@pytest.fixture(scope="module")
def real_pair(tmp_path_factory):
    root = tmp_path_factory.mktemp("real-pair-synthetic")
    width, height, count = 320, 180, 290
    retail, leak = root / "retail.mp4", root / "leak.mp4"
    frames = {}
    for index in range(20, 281):
        x = 30 + round(0.65 * (index - 20))
        points = np.array([(x, 64), (x + 63, 64), (x + 63, 123), (x, 123)], np.float32)
        frames[index] = [Region((x, 64, x + 63, 123), points, block=12)]
    logo = Overlay((6, 5, 38, 24), "retail", 1.0)
    regions = RegionSet(frames, overlays=(logo,))
    style = MosaicStyle("pixelate", 12, (0, 0), False, False, 0, False, 0)

    def source_frames():
        for frame in textured_frames(count + 10, width, height):
            # Keep fine luminance texture but milder chroma, so a second 4:2:0
            # encode does not dominate the restoration error being measured.
            grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)[..., None]
            yield np.rint(0.2 * frame + 0.8 * grey).astype(np.uint8)

    def retail_frames():
        for t, frame in enumerate(source_frames()):
            if t < 10:
                continue
            index = t - 10
            if index in frames:
                mask = rasterize(frames[index], (0, 0), (height, width), ellipse=False)
                frame = censor(frame, mask, style, (0, 0))
            frame[5:24, 6:38] = (240, 225, 235)
            frame[10:19, 12:32] = (20, 35, 25)
            yield frame

    # Scale the leak so retail ~= leak / 0.98; the fitted transform must restore
    # retail's colours. A separate regression test verifies the reverse direction.
    leak_frames = (np.rint((f.astype(np.float32) - 127.5) * 0.98 + 127.5).clip(0, 255).astype(np.uint8)
                   for f in source_frames())
    write_video(leak, leak_frames, 30, crf=16)
    write_video(retail, retail_frames(), 30, crf=16)
    alignment = Alignment(str(retail), str(leak), count, count + 10, 30, 30,
                          (width, height), (width, height), ((0, 10, count),))
    align_dir, regions_dir = root / "align", root / "regions"
    align_dir.mkdir()
    regions_dir.mkdir()
    alignment_path, regions_path = align_dir / "TITLE.json", regions_dir / "TITLE.json"
    save_alignment(alignment, alignment_path)
    save_regions(regions, regions_path)
    return {
        "root": root, "alignment": alignment, "regions": regions,
        "alignment_path": alignment_path, "regions_path": regions_path,
        "retail": list(read_frames(retail)), "leak": list(read_frames(leak)),
        "config": RealPairConfig(window=41, hop=41, camera=CameraConfig(canvas=128)),
    }


def _plans(data, config=None, seed=0):
    return plan_real_windows(data["alignment"], data["regions"], "TITLE", config or data["config"],
                             np.random.default_rng(_seed_for(seed, safe_stem("TITLE"))))


def _build(data, out, config=None, **kwargs):
    return build_title_pairs("TITLE", data["alignment_path"], data["regions_path"], out,
                             config or data["config"], **kwargs)


def test_composite_layout_reference_colours_and_resume(real_pair):
    data = real_pair
    out = data["root"] / "composite"
    rows = _build(data, out)
    plans = {plan.key: plan for plan in _plans(data)}
    assert len(rows) == len(plans) >= 4
    assert rebuild_dataset(out, data["config"].caption) == len(rows)
    dataset = [json.loads(line) for line in (out / "dataset.jsonl").read_text().splitlines()]
    assert len(dataset) == len(rows)
    assert {r["video"] for r in dataset} == {r["video"] for r in rows}
    assert json.loads((out / "meta/TITLE.json").read_text())["windows"] == rows
    inside_errors, mosaic_errors, outside_errors = [], [], []
    for row in rows:
        plan = plans[row["key"]]
        targets = list(read_frames(out / row["video"]))
        references = list(read_frames(out / row["reference_video"]))
        assert len(targets) == len(references) == 41
        assert probe(out / row["video"]).fps == 24
        assert row["leak_start"] == row["start"] + 10
        assert row["leak_stop"] == row["leak_start"] + 41
        assert np.allclose(row["colour_fit"]["a"], 1 / 0.98, atol=0.01)
        gain, offset = np.array(row["colour_fit"]["a"]), np.array(row["colour_fit"]["b"])
        for index, (target, reference, crop) in enumerate(zip(targets, references, plan.crops)):
            assert target.shape == reference.shape == (128, 128, 3)
            expected_reference = crop_to_canvas(data["retail"][plan.start + index], crop)
            assert np.abs(reference.astype(float) - expected_reference).mean() < 2
            grown, _ = _masks(plan.regions[index], (0, 0, 320, 180), 320, data["config"], feather=False)
            canvas_grown = crop_to_canvas(np.repeat(grown[..., None].astype(np.uint8), 3, axis=2) * 255, crop)[..., 0]
            outside = canvas_grown == 0
            outside_errors.append(np.abs(target.astype(float) - reference)[outside].mean())
            true = rasterize(plan.regions[index], (0, 0), (180, 320), ellipse=False)
            canvas_true = crop_to_canvas(np.repeat(true[..., None].astype(np.uint8), 3, axis=2) * 255, crop)[..., 0]
            inside = cv2.erode(canvas_true, np.ones((9, 9), np.uint8)) == 255
            leak_canvas = crop_to_canvas(data["leak"][plan.leak_indices[index]], crop)
            matched = np.clip(leak_canvas.astype(float) * gain + offset, 0, 255)
            inside_errors.append(np.abs(target.astype(float) - matched)[inside].mean())
            mosaic_errors.append(np.abs(reference.astype(float) - matched)[inside].mean())
    assert np.mean(outside_errors) < 2
    assert np.mean(inside_errors) < 2
    assert np.mean(mosaic_errors) > 3 * np.mean(inside_errors)
    stamps = {p: p.stat().st_mtime_ns for p in out.rglob("*") if p.is_file()}
    assert _build(data, out, seed=99) == rows
    assert all(p.stat().st_mtime_ns == stamp for p, stamp in stamps.items())


def test_leak_targets_skip_logo_crops(real_pair):
    data = real_pair
    config = replace(data["config"], target="leak")
    plans = _plans(data, config)
    assert 0 < len(plans) < len(_plans(data))
    out = data["root"] / "leak-target"
    rows = _build(data, out, config)
    assert len(rows) == len(plans)
    for row, plan in zip(rows, plans):
        for index, crop in enumerate(plan.crops):
            for overlay in data["regions"].overlays:
                if overlay.active(plan.start + index):
                    margin = config.overlay_margin * 320
                    assert crop.box[0] >= overlay.box[2] + margin or crop.box[1] >= overlay.box[3] + margin
        gain, offset = np.array(row["colour_fit"]["a"]), np.array(row["colour_fit"]["b"])
        for index, frame in enumerate(read_frames(out / row["video"])):
            expected = crop_to_canvas(data["leak"][plan.leak_indices[index]], plan.crops[index])
            expected = np.rint(expected.astype(float) * gain + offset).clip(0, 255)
            assert np.abs(frame.astype(float) - expected).mean() < 2


def test_planning_gaps_cuts_overlays_and_seed(real_pair):
    data = real_pair
    alignment, regions, config = data["alignment"], data["regions"], data["config"]
    gapped = replace(alignment, runs=((0, 10, 100), (110, 120, 180)))
    plans = plan_real_windows(gapped, regions, "TITLE", config, np.random.default_rng(0))
    assert plans
    assert all(not (p.start < 110 and p.start + 41 > 100) for p in plans)
    cut_regions = replace(regions, cuts=frozenset({145}))
    plans = plan_real_windows(alignment, cut_regions, "TITLE", config, np.random.default_rng(0))
    assert all(not p.start < 145 < p.start + 41 for p in plans)
    # An active watermark on the mosaic invalidates composite windows too.
    obstructed = replace(regions, overlays=(Overlay((0, 0, 320, 180), "leak", 1, (100, 150)),))
    plans = plan_real_windows(alignment, obstructed, "TITLE", config, np.random.default_rng(0))
    assert plans and all(p.start + 41 <= 100 or p.start >= 150 for p in plans)
    capped = replace(config, max_windows_per_title=2)
    assert [p.key for p in _plans(data, capped, 4)] == [p.key for p in _plans(data, capped, 4)]
    assert len(_plans(data, capped)) == 2


def test_colour_fit_direction_and_robustness():
    rng = np.random.default_rng(3)
    leak = rng.uniform(20, 220, (10000, 3))
    retail = leak * 0.98 + 2.55
    retail[:1000] = rng.uniform(0, 255, (1000, 3))
    fit = _fit_colour(leak, retail)
    assert np.allclose(fit["a"], 0.98, atol=1e-5)
    assert np.allclose(fit["b"], 2.55, atol=1e-3)


def test_region_block_changes_mask_and_missing_block_falls_back():
    config = RealPairConfig()
    box = (40, 40, 60, 60)
    small, _ = _masks([Region(box, block=4)], (0, 0, 100, 100), 100, config)
    large, _ = _masks([Region(box, block=20)], (0, 0, 100, 100), 100, config)
    fallback, _ = _masks([Region(box)], (0, 0, 100, 100), 100, config)
    explicit, _ = _masks([Region(box, block=1.5)], (0, 0, 100, 100), 100, config)
    assert large.sum() > small.sum()
    assert np.array_equal(fallback, explicit)


def test_cli_spawn_config_and_only_filter(real_pair, tmp_path, capsys):
    data = real_pair
    config_path = tmp_path / "config.yaml"
    config_path.write_text("pairs:\n  window: 41\n  hop: 41\n  output_crf: 10\ncamera:\n  canvas: 128\n")
    assert load_config(config_path).camera.canvas == 128
    out = tmp_path / "data"
    assert main(["--align-dir", str(data["alignment_path"].parent), "--regions-dir", str(data["regions_path"].parent),
                 "--out", str(out), "--config", str(config_path), "--workers", "1", "--only", "TITLE",
                 "--max-windows-per-title", "1", "--target", "composite"]) == 0
    assert len((out / "dataset.jsonl").read_text().splitlines()) == 1
    assert "[1/1] TITLE: 1 pairs" in capsys.readouterr().out
    config_path.write_text("mosaic: {}\n")
    with pytest.raises(ValueError, match="unknown sections"):
        load_config(config_path)
    config_path.write_text("pairs:\n  invalid: 1\n")
    with pytest.raises(TypeError):
        load_config(config_path)


def test_resize_backward_mapping_and_single_forward_decode(real_pair, tmp_path, monkeypatch):
    # Exercise scale and reordered leak shots without another encode/decode cycle.
    # All frame arrays come from the generated fixture, never real footage.
    import reveil.real_pairs as module

    data = real_pair
    alignment = replace(data["alignment"], runs=((0, 160, 130), (130, 10, 160)), leak_size=(640, 360))
    alignment_path = tmp_path / "reordered.json"
    save_alignment(alignment, alignment_path)
    regions_path = tmp_path / "regions.json"
    save_regions(replace(data["regions"], overlays=()), regions_path)
    calls = []

    def fake_decode(path):
        calls.append(path)
        if path == alignment.retail:
            yield from data["retail"]
        else:
            for frame in data["leak"]:
                yield cv2.resize(frame, (640, 360), interpolation=cv2.INTER_NEAREST)

    monkeypatch.setattr(module, "read_frames", fake_decode)
    config = replace(data["config"], target="leak", max_windows_per_title=2, colour_sample_frames=2)
    out = tmp_path / "out"
    rows = build_title_pairs("TITLE", alignment_path, regions_path, out, config)
    assert len(rows) == 2
    assert any(np.any(np.diff(row["leak_indices"]) < 0) for row in rows)
    assert calls == [alignment.retail, alignment.leak]
    for row in rows:
        gain, offset = np.array(row["colour_fit"]["a"]), np.array(row["colour_fit"]["b"])
        for index, target in enumerate(read_frames(out / row["video"])):
            crop = CropFrame(tuple(row["crops"][index]), 128)
            leak = crop_to_canvas(data["leak"][row["leak_indices"][index]], crop)
            expected = np.rint(leak * gain + offset).clip(0, 255)
            assert np.abs(target.astype(float) - expected).mean() < 2
