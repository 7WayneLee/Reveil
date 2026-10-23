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
from reveil.video import decoded_frames, probe, read_frames, write_video


class _RGBFrame:
    """Expose an already converted synthetic frame through the decoder interface."""

    def __init__(self, image):
        self.image = image

    def to_ndarray(self, *, format):
        assert format == "rgb24"
        return self.image


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
        "config": RealPairConfig(window=41, hop=41, match_quality=False, camera=CameraConfig(canvas=128)),
    }


def _plans(data, config=None, seed=0):
    return plan_real_windows(data["alignment"], data["regions"], "TITLE", config or data["config"],
                             np.random.default_rng(_seed_for(seed, safe_stem("TITLE"))))


def _build(data, out, config=None, **kwargs):
    return build_title_pairs("TITLE", data["alignment_path"], data["regions_path"], out,
                             config or data["config"], **kwargs)


@pytest.mark.parametrize("target,match_quality", [("composite", False), ("composite", True), ("leak", True)])
def test_selective_conversion_is_byte_identical_to_eager_read_frames(real_pair, tmp_path, monkeypatch,
                                                                    target, match_quality):
    import reveil.real_pairs as module

    config = replace(real_pair["config"], target=target, match_quality=match_quality,
                     max_windows_per_title=2, ring_max_residual=0)

    def eager_frames(path, *, threads=None):
        # The legacy route converts every frame in read_frames, even when unused.
        # The wrapper returns that RGB array without converting it a second time.
        for image in read_frames(path, threads=threads):
            yield _RGBFrame(image)

    with monkeypatch.context() as eager:
        eager.setattr(module, "decoded_frames", eager_frames)
        expected_rows = _build(real_pair, tmp_path / "eager", config)
    actual_rows = _build(real_pair, tmp_path / "selective", config)
    assert len(actual_rows) == len(expected_rows) == 2
    assert actual_rows == expected_rows
    for sub in ("clean", "mosaic", "meta"):
        expected = {path.name: path.read_bytes() for path in (tmp_path / "eager" / sub).iterdir()}
        actual = {path.name: path.read_bytes() for path in (tmp_path / "selective" / sub).iterdir()}
        assert actual == expected


@pytest.mark.parametrize("match_quality", [False, True])
def test_render_converts_exactly_the_demanded_frames(real_pair, tmp_path, monkeypatch, match_quality):
    import reveil.real_pairs as module

    data = real_pair
    config = replace(data["config"], window=17, hop=8, match_quality=match_quality,
                     max_windows_per_title=2, candidate_factor=2, ring_max_residual=0)
    plans = _plans(data, config)
    retail_used = {index for plan in plans for index in range(plan.start, plan.start + len(plan.crops))}
    leak_used = {index for plan in plans for index in plan.leak_indices}
    if match_quality:
        leak_used.update(plan.previous_leak_index for plan in plans if plan.previous_leak_index is not None)
    expected = {data["alignment"].retail: sorted(retail_used), data["alignment"].leak: sorted(leak_used)}
    decoded = {path: [] for path in expected}
    converted = {path: [] for path in expected}
    calls = []

    class CountedFrame:
        def __init__(self, frame, path, index):
            self.frame, self.path, self.index = frame, path, index

        def to_ndarray(self, *, format):
            assert format == "rgb24"
            converted[self.path].append(self.index)
            return self.frame.to_ndarray(format=format)

    def counted_frames(path, *, threads=None):
        calls.append((path, threads))
        for index, frame in enumerate(decoded_frames(path, threads=threads)):
            decoded[path].append(index)
            yield CountedFrame(frame, path, index)

    monkeypatch.setattr(module, "decoded_frames", counted_frames)
    assert len(_build(data, tmp_path / "out", config, decode_threads=3)) == 2
    assert converted == expected
    assert calls == [(data["alignment"].retail, 3), (data["alignment"].leak, 3)]
    for path, indices in expected.items():
        assert decoded[path] == list(range(max(indices) + 1))
        assert len(converted[path]) < len(decoded[path])


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
    assert len(_plans(data, capped)) == 3


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
                 "--max-windows-per-title", "1", "--target", "composite", "--no-match-quality",
                 "--ring-max-residual", "0", "--candidate-factor", "2"]) == 0
    assert len((out / "dataset.jsonl").read_text().splitlines()) == 1
    assert "[1/1] TITLE: 1 pairs" in capsys.readouterr().out
    config_path.write_text("mosaic: {}\n")
    with pytest.raises(ValueError, match="unknown sections"):
        load_config(config_path)
    config_path.write_text("pairs:\n  invalid: 1\n")
    with pytest.raises(TypeError):
        load_config(config_path)


def test_cli_resumes_prepared_pairs_with_missing_source_videos(tmp_path, capsys):
    title = "TITLE (prepared)"
    align_dir, regions_dir, out = tmp_path / "align", tmp_path / "regions", tmp_path / "data"
    align_dir.mkdir()
    regions_dir.mkdir()
    alignment = Alignment(str(tmp_path / "missing-retail.mp4"), str(tmp_path / "missing-leak.mp4"),
                          1, 1, 24, 24, (64, 64), (64, 64), ((0, 0, 1),))
    save_alignment(alignment, align_dir / f"{title}.json")
    save_regions(RegionSet({}), regions_dir / f"{title}.json")
    for sub in ("clean", "mosaic", "meta"):
        (out / sub).mkdir(parents=True)
    rows = [{"video": f"clean/{key}.mp4", "reference_video": f"mosaic/{key}.mp4"}
            for key in ("kept", "missing-clean", "missing-mosaic")]
    for clip in ("clean/kept.mp4", "mosaic/kept.mp4", "mosaic/missing-clean.mp4", "clean/missing-mosaic.mp4"):
        (out / clip).write_bytes(b"prepared clip")
    meta_path = out / "meta" / f"{safe_stem(title)}.json"
    meta_path.write_text(json.dumps({"title": title, "windows": rows}))
    (out / "dataset.jsonl").write_text("stale dataset\n")
    args = ["--align-dir", str(align_dir), "--regions-dir", str(regions_dir),
            "--out", str(out), "--workers", "1"]

    assert main(args) == 0
    assert f"[1/1] {title}: 3 pairs" in capsys.readouterr().out
    assert [json.loads(line) for line in (out / "dataset.jsonl").read_text().splitlines()] == [
        {**rows[0], "caption": RealPairConfig().caption}]
    with pytest.raises(FileNotFoundError):
        main([*args, "--overwrite"])
    meta_path.unlink()
    with pytest.raises(FileNotFoundError):
        main(args)


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

    def fake_decode(path, *, threads=None):
        calls.append((path, threads))
        if path == alignment.retail:
            yield from map(_RGBFrame, data["retail"])
        else:
            for frame in data["leak"]:
                yield _RGBFrame(cv2.resize(frame, (640, 360), interpolation=cv2.INTER_NEAREST))

    monkeypatch.setattr(module, "decoded_frames", fake_decode)
    config = replace(data["config"], target="leak", max_windows_per_title=2, colour_sample_frames=2)
    out = tmp_path / "out"
    rows = build_title_pairs("TITLE", alignment_path, regions_path, out, config, decode_threads=3)
    assert len(rows) == 2
    assert any(np.any(np.diff(row["leak_indices"]) < 0) for row in rows)
    assert calls == [(alignment.retail, 3), (alignment.leak, 3)]
    for row in rows:
        gain, offset = np.array(row["colour_fit"]["a"]), np.array(row["colour_fit"]["b"])
        for index, target in enumerate(read_frames(out / row["video"])):
            crop = CropFrame(tuple(row["crops"][index]), 128)
            leak = crop_to_canvas(data["leak"][row["leak_indices"][index]], crop)
            expected = np.rint(leak * gain + offset).clip(0, 255)
            assert np.abs(target.astype(float) - expected).mean() < 2


def _quality_pair(tmp_path, monkeypatch, *, count=10, indices=None, start=1,
                  sigma=1.0, prev_mix=0.2, colour=True, unmatched_previous=False):
    """Lossless in-memory moving texture; fake encoding retains exact output bytes."""
    import reveil.real_pairs as module

    cv2.setNumThreads(1)
    indices = list(range(count)) if indices is None else list(indices)
    size = 96
    rng = np.random.default_rng(28)
    texture = rng.integers(5, 46, (size, size, 3), dtype=np.uint8) * 5
    leak = [np.roll(texture, 3 * index, axis=1) for index in range(max(indices) + 1)]
    gain = np.array([0.96, 1.02, 0.98], np.float32) if colour else np.ones(3, np.float32)
    offset = np.array([3, -2, 2], np.float32) if colour else np.zeros(3, np.float32)
    truth, retail = [], []
    region = Region((32, 32, 64, 64), block=8)
    frames = {i: [region] for i in range(start, count)}
    for i, index in enumerate(indices):
        previous = leak[indices[i - 1]] if i and not (unmatched_previous and i == start) else leak[index]
        image = (1 - prev_mix) * leak[index].astype(np.float32) + prev_mix * previous.astype(np.float32)
        if sigma:
            image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
        image = np.rint(image * gain + offset).clip(0, 255).astype(np.uint8)
        truth.append(image.copy())
        if i in frames:
            image = image.copy()
            patch = image[32:64, 32:64]
            small = cv2.resize(patch, (4, 4), interpolation=cv2.INTER_AREA)
            image[32:64, 32:64] = cv2.resize(small, (32, 32), interpolation=cv2.INTER_NEAREST)
        retail.append(image)
    runs = []
    for i, index in enumerate(indices):
        if unmatched_previous and i < start:
            continue
        if runs and runs[-1][0] + runs[-1][2] == i and runs[-1][1] + runs[-1][2] == index:
            r, l, n = runs[-1]
            runs[-1] = (r, l, n + 1)
        else:
            runs.append((i, index, 1))
    alignment = Alignment(str(tmp_path / "retail.mp4"), str(tmp_path / "leak.mp4"),
                          count, len(leak), 24, 48 if indices[1] - indices[0] == 2 else 24,
                          (size, size), (size, size), tuple(runs))
    regions = RegionSet(frames)
    alignment_path, regions_path = tmp_path / "alignment.json", tmp_path / "regions.json"
    save_alignment(alignment, alignment_path)
    save_regions(regions, regions_path)
    calls, written = [], {}

    def decode(path, *, threads=None):
        calls.append(path)
        yield from map(_RGBFrame, retail if path == alignment.retail else leak)

    def encode(path, images, fps, *, crf):
        array = np.stack(list(images))
        written[path] = array
        path.write_bytes(array.tobytes())

    monkeypatch.setattr(module, "decoded_frames", decode)
    monkeypatch.setattr(module, "write_video", encode)
    config = RealPairConfig(window=count - start, hop=count - start, colour_sample_frames=5,
                            camera=CameraConfig(canvas=size, context=size))
    return {"root": tmp_path, "alignment": alignment, "regions": regions,
            "alignment_path": alignment_path, "regions_path": regions_path,
            "leak": leak, "retail": retail, "truth": truth, "config": config,
            "calls": calls, "written": written}


def test_quality_recovers_blur_mix_and_records_meta(tmp_path, monkeypatch):
    data = _quality_pair(tmp_path, monkeypatch)
    out = tmp_path / "out"
    rows = _build(data, out)
    assert len(rows) == 1
    row = rows[0]
    assert row["quality"]["sigma"] == 1.0
    assert row["quality"]["prev_mix"] == 0.2
    assert row["quality"]["residual"] < row["quality"]["identity_residual"] / 10
    assert row["ring"]["score"] < 0.4
    assert row["ring"]["median"] <= row["ring"]["max"] < 0.4
    assert row["ring"]["frames"] == data["config"].window
    plan, = _plans(data)
    expected = []
    for i, crop in enumerate(plan.crops):
        leak = crop_to_canvas(data["leak"][plan.leak_indices[i]], crop).astype(np.float32)
        previous_index = plan.previous_leak_index if i == 0 else plan.leak_indices[i - 1]
        if previous_index is not None:
            previous = crop_to_canvas(data["leak"][previous_index], crop).astype(np.float32)
            mix = row["quality"]["prev_mix"]
            leak = (1 - mix) * leak + mix * previous
        sigma = row["quality"]["sigma"]
        if sigma:
            leak = cv2.GaussianBlur(leak, (0, 0), sigmaX=sigma)
        leak *= np.asarray(row["colour_fit"]["a"], np.float32)
        leak += np.asarray(row["colour_fit"]["b"], np.float32)
        leak = leak.clip(0, 255)
        _, alpha = _masks(plan.regions[i], plan.work, 96, data["config"])
        alpha = crop_to_canvas(np.repeat(alpha[..., None], 3, axis=2), crop)[..., 0, None]
        retail = crop_to_canvas(data["retail"][plan.start + i], crop)
        expected.append(np.rint(retail * (1 - alpha) + leak * alpha).clip(0, 255).astype(np.uint8))
    actual = data["written"][out / row["video"]]
    assert np.abs(actual.astype(float) - np.stack(expected)).mean() < 0.01
    meta = json.loads((out / "meta/TITLE.json").read_text())
    stats = meta["stats"]
    assert {k: stats[k] for k in ("candidates", "rejected_ring", "accepted", "kept", "dropped_extra")} == {
        "candidates": 1, "rejected_ring": 0, "accepted": 1, "kept": 1, "dropped_extra": 0,
    }
    assert stats["sigma_histogram"] == {"1.0": 1}
    assert stats["prev_mix_histogram"] == {"0.2": 1}
    assert all(p == row["ring"]["score"] for p in stats["ring_percentiles"].values())
    assert rebuild_dataset(out, "caption") == 1
    assert json.loads((out / "dataset.jsonl").read_text()) == {
        "video": row["video"], "reference_video": row["reference_video"], "caption": "caption",
    }
    assert data["calls"] == [data["alignment"].retail, data["alignment"].leak]


def test_no_quality_matching_is_byte_identical_to_legacy(tmp_path, monkeypatch):
    data = _quality_pair(tmp_path, monkeypatch)
    config = replace(data["config"], match_quality=False, ring_max_residual=0)
    plan, = _plans(data, config)
    # Reproduce the original sampling and channel fits independently, including
    # its border pixels, rather than comparing two calls to the new fitter.
    xs, ys, alphas = [], [], []
    for i, crop in enumerate(plan.crops):
        grown, alpha = _masks(plan.regions[i], plan.work, 96, config)
        alphas.append(crop_to_canvas(np.repeat(alpha[..., None], 3, axis=2), crop)[..., 0])
        if i not in np.linspace(0, config.window - 1, config.colour_sample_frames, dtype=int):
            continue
        excluded = crop_to_canvas(np.repeat(grown[..., None].astype(np.float32), 3, axis=2), crop)[..., 0] > 1e-6
        pixels = np.flatnonzero(~excluded)
        xs.append(crop_to_canvas(data["leak"][plan.leak_indices[i]], crop).reshape(-1, 3)[pixels])
        ys.append(crop_to_canvas(data["retail"][plan.start + i], crop).reshape(-1, 3)[pixels])
    x, y = np.concatenate(xs), np.concatenate(ys)
    gains, offsets = [], []
    for channel in range(3):
        a, b = 1.0, 0.0
        xc, yc = x[:, channel].astype(float), y[:, channel].astype(float)
        for iteration in range(3):
            dx = xc - xc.mean()
            variance = np.dot(dx, dx)
            a = float(np.dot(dx, yc - yc.mean()) / variance) if variance > 1e-8 else 1.0
            b = float(yc.mean() - a * xc.mean())
            if iteration < 2:
                residual = np.abs(yc - (a * xc + b))
                keep = residual <= np.quantile(residual, 0.8)
                xc, yc = xc[keep], yc[keep]
        gains.append(a)
        offsets.append(b)
    expected = []
    for i, crop in enumerate(plan.crops):
        image = crop_to_canvas(data["leak"][plan.leak_indices[i]], crop).astype(np.float32)
        image *= np.asarray(gains, np.float32)
        image += np.asarray(offsets, np.float32)
        np.clip(image, 0, 255, out=image)
        alpha = alphas[i][..., None]
        retail = crop_to_canvas(data["retail"][plan.start + i], crop)
        image = retail * (1 - alpha) + image * alpha
        expected.append(np.rint(image).clip(0, 255).astype(np.uint8))
    row, = _build(data, tmp_path / "out", config)
    assert row["colour_fit"] == {"a": gains, "b": offsets, "samples": len(x)}
    assert (tmp_path / "out" / row["video"]).read_bytes() == np.stack(expected).tobytes()
    assert row["quality"]["sigma"] == row["quality"]["prev_mix"] == 0
    assert row["quality"]["residual"] == row["quality"]["identity_residual"]


@pytest.mark.parametrize("shift", [0, 8])
def test_ring_gate_rejects_spatial_mismatch_before_encoding(tmp_path, monkeypatch, shift):
    data = _quality_pair(tmp_path, monkeypatch, sigma=0, prev_mix=0)
    if shift:
        for frame in data["leak"]:
            frame[16:80, 16:80] = np.roll(frame[16:80, 16:80], shift, axis=1)
    out = tmp_path / "out"
    rows = _build(data, out)
    meta = json.loads((out / "meta/TITLE.json").read_text())
    assert len(rows) == (0 if shift else 1)
    assert meta["stats"]["rejected_ring"] == bool(shift)
    assert (meta["stats"]["ring_percentiles"]["p90"] > 6) == bool(shift)
    if shift:
        assert not data["written"]
        assert not list(out.glob("*/*.mp4"))


def test_cap_replaces_rejected_priority_and_deletes_extras(tmp_path, monkeypatch):
    data = _quality_pair(tmp_path, monkeypatch, count=37, sigma=0, prev_mix=0)
    config = replace(data["config"], window=5, hop=5, max_windows_per_title=1,
                     candidate_factor=4, match_quality=False)
    plans = _plans(data, config, seed=4)
    assert len(plans) == 4
    first = min(plans, key=lambda p: p.priority)
    for index in first.leak_indices:
        frame = data["leak"][index]
        frame[16:80, 16:80] = np.roll(frame[16:80, 16:80], 8, axis=1)
    expected = min((p for p in plans if p is not first), key=lambda p: p.priority)
    for name in ("out-a", "out-b"):
        out = tmp_path / name
        row, = _build(data, out, config, seed=4)
        assert row["key"] == expected.key
        assert sorted(p.name for p in out.glob("clean/*.mp4")) == [f"{expected.key}.mp4"]
        assert sorted(p.name for p in out.glob("mosaic/*.mp4")) == [f"{expected.key}.mp4"]
        stats = json.loads((out / "meta/TITLE.json").read_text())["stats"]
        assert stats["candidates"] == 4
        assert stats["rejected_ring"] == 1
        assert stats["accepted"] == 3
        assert stats["kept"] == 1
        assert stats["dropped_extra"] == 2


@pytest.mark.parametrize("mapping", ["unmatched", "jump", "double", "moving-crop", "previous-ahead"])
def test_previous_frame_follows_retail_mapping_and_current_crop(tmp_path, monkeypatch, mapping):
    import reveil.real_pairs as module

    indices = {"jump": [0, 1, 2, 3, 14, 15, 16, 17, 18, 19],
               "double": list(range(0, 20, 2)),
               "previous-ahead": [19, 0, 1, 2, 3, 4, 5, 6, 7, 8]}.get(mapping)
    data = _quality_pair(tmp_path, monkeypatch, indices=indices, sigma=0, colour=False,
                         unmatched_previous=mapping == "unmatched")
    if mapping == "moving-crop":
        monkeypatch.setattr(module, "plan_crops", lambda boxes, *args: [
            CropFrame((16 + i % 3, 16, 80 + i % 3, 80), 64) for i in range(len(boxes))])
    config = replace(data["config"], blur_sigmas=(0,))
    plan, = _plans(data, config)
    if mapping == "unmatched":
        assert plan.previous_leak_index is None
    else:
        assert plan.previous_leak_index == data["alignment"].leak_index(plan.start - 1)
    out = tmp_path / "out"
    row, = _build(data, out, config)
    assert row["quality"]["prev_mix"] == 0.2
    assert row["quality"]["residual"] < 1e-8
    assert row["ring"]["score"] < 1e-4
    actual = data["written"][out / row["video"]]
    # Deep inside the mosaic, alpha is one. The intended unmosaicked image is
    # exactly integral here, including frame zero with no previous match.
    for i, crop in enumerate(plan.crops):
        _, alpha = _masks(plan.regions[i], plan.work, 96, config)
        local_crop = crop.shifted(*plan.work[:2])
        inside = crop_to_canvas(np.repeat(alpha[..., None], 3, axis=2), local_crop)[..., 0] > 0.99999
        expected = crop_to_canvas(data["truth"][plan.start + i], crop)
        assert np.array_equal(actual[i][inside], expected[inside])
    assert data["calls"] == [data["alignment"].retail, data["alignment"].leak]


@pytest.mark.parametrize("threshold", [0, None])
def test_ring_gate_can_be_disabled(tmp_path, monkeypatch, threshold):
    data = _quality_pair(tmp_path, monkeypatch, sigma=0, prev_mix=0)
    for frame in data["leak"]:
        frame[16:80, 16:80] = np.roll(frame[16:80, 16:80], 8, axis=1)
    config = replace(data["config"], match_quality=False, ring_max_residual=threshold)
    row, = _build(data, tmp_path / "out", config)
    assert row["ring"]["score"] > 6


def test_no_usable_ring_is_kept_and_leak_targets_stay_colour_only(tmp_path, monkeypatch):
    data = _quality_pair(tmp_path, monkeypatch)
    config = replace(data["config"], min_ring_pixels=100000)
    row, = _build(data, tmp_path / "no-ring", config)
    assert row["ring"] == {"score": None, "median": None, "max": None, "frames": 0}
    config = replace(data["config"], target="leak")
    row, = _build(data, tmp_path / "leak", config)
    assert row["quality"]["sigma"] == row["quality"]["prev_mix"] == 0
    assert row["ring"]["score"] is None
    expected = []
    for leak in data["leak"][1:]:
        image = leak.astype(np.float32) * np.asarray(row["colour_fit"]["a"], np.float32)
        image += np.asarray(row["colour_fit"]["b"], np.float32)
        expected.append(np.rint(image).clip(0, 255).astype(np.uint8))
    assert (tmp_path / "leak" / row["video"]).read_bytes() == np.stack(expected).tobytes()


def test_quality_config_yaml_and_cli_flags(tmp_path, capsys):
    path = tmp_path / "config.yaml"
    path.write_text("pairs:\n  match_quality: false\n  quality_tolerance: 0.05\n"
                    "  blur_sigmas: [0, 1]\n  prev_mixes: [0, 0.2]\n"
                    "  ring_blocks: 2\n  min_ring_pixels: 128\n  ring_max_residual: null\n  candidate_factor: 2\n")
    config = load_config(path)
    assert not config.match_quality
    assert config.quality_tolerance == 0.05
    assert config.blur_sigmas == [0, 1]
    assert config.prev_mixes == [0, 0.2]
    assert config.ring_blocks == 2 and config.min_ring_pixels == 128
    assert config.ring_max_residual is None and config.candidate_factor == 2
    with pytest.raises(SystemExit) as result:
        main(["--help"])
    assert result.value.code == 0
    help_text = capsys.readouterr().out
    assert all(flag in help_text for flag in ("--no-match-quality", "--ring-max-residual", "--candidate-factor"))


def test_quality_and_ring_exclude_border_and_active_overlay(tmp_path, monkeypatch):
    data = _quality_pair(tmp_path, monkeypatch)
    overlay = Overlay((21, 34, 25, 62), "retail", 1.0, (3, 6))
    regions = replace(data["regions"], overlays=(overlay,))
    save_regions(regions, data["regions_path"])
    for index, frame in enumerate(data["retail"]):
        frame[:4] = frame[-4:] = 0
        frame[:, :4] = frame[:, -4:] = 0
        if overlay.active(index):
            frame[34:62, 21:25] = (255, 0, 255)
    row, = _build(data, tmp_path / "out")
    assert row["quality"]["sigma"] == 1
    assert row["quality"]["prev_mix"] == 0.2
    assert row["ring"]["score"] < 0.4
    assert row["ring"]["frames"] == data["config"].window


def test_ring_uses_each_regions_block_and_excludes_all_grown_masks(tmp_path):
    import reveil.real_pairs as module

    items = [Region((0, 32, 24, 64), block=8), Region((44, 32, 60, 48), block=12)]
    crop = CropFrame((0, 0, 96, 96), 96)
    config = RealPairConfig(window=1, min_ring_pixels=1)
    plan = module.RealWindowPlan("ring", 0, 0, [items[0].box], [items], [crop], crop.box, (0,))
    alignment = Alignment("retail", "leak", 1, 1, 24, 24, (96, 96), (96, 96), ((0, 0, 1),))
    buffer = module._WindowBuffer(plan, tmp_path, config)
    buffer.activate()
    buffer.add("retail", 0, np.zeros((96, 96, 3), np.uint8), alignment, RegionSet({0: items}), config)
    ring = np.zeros((96, 96), bool)
    ring.flat[buffer.ring_pixels[0]] = True
    grown, _ = _masks(items, crop.box, 96, config, feather=False)
    assert not (ring & grown).any()
    assert not ring[:4].any() and not ring[-4:].any()
    assert not ring[:, :4].any() and not ring[:, -4:].any()
    assert ring[40, 77] and not ring[40, 79]  # second region: 6 growth + 12 ring
    assert ring[75, 12] and not ring[77, 12]  # first region: 4 growth + 8 ring


def test_forward_spool_keeps_previous_canvases_for_future_moving_crops(tmp_path, monkeypatch):
    import reveil.real_pairs as module

    indices = list(range(20, 29)) + list(range(10))
    data = _quality_pair(tmp_path, monkeypatch, count=19, indices=indices, sigma=0, colour=False)
    monkeypatch.setattr(module, "plan_crops", lambda boxes, *args: [
        CropFrame((16 + i % 3, 16, 80 + i % 3, 80), 64) for i in range(len(boxes))])
    config = replace(data["config"], window=5, hop=5, blur_sigmas=(0,))
    rows = _build(data, tmp_path / "out", config)
    assert len(rows) >= 2
    assert all(row["quality"]["prev_mix"] == 0.2 for row in rows)
    assert all(row["quality"]["residual"] < 1e-8 and row["ring"]["score"] < 1e-4 for row in rows)
    assert data["calls"] == [data["alignment"].retail, data["alignment"].leak]


def test_quality_ties_prefer_smallest_mix_then_sigma(tmp_path, monkeypatch):
    data = _quality_pair(tmp_path, monkeypatch, sigma=0, prev_mix=0, colour=False)
    for frame in data["retail"] + data["leak"]:
        frame[:] = 100
    # Deliberately unordered grids also verify the tie rule is independent of YAML order.
    config = replace(data["config"], blur_sigmas=(1, 0.4, 0), prev_mixes=(0.4, 0.2, 0))
    row, = _build(data, tmp_path / "out", config)
    assert row["quality"] == {"sigma": 0.0, "prev_mix": 0.0, "residual": 0.0, "identity_residual": 0.0}


@pytest.mark.parametrize(("scores", "tolerance", "expected"), [
    ({(0, 0.2): 1.0, (1, 0): 1.0}, 0, (1, 0)),
    ({(1, 0.1): 1.0, (0.4, 0.1): 1.0}, 0, (0.4, 0.1)),
    ({(0.7, 0.2): 1.0, (1, 0.1): 1.02, (0, 0.3): 1.01}, 0.03, (1, 0.1)),
    ({(0.7, 0.2): 1.0, (0, 0): 1.031}, 0.03, (0.7, 0.2)),
    ({(1, 0.2): 1.0, (0.7, 0.2): 1.03}, 0.03, (0.7, 0.2)),
    ({(0.7, 0.2): 1.0, (0, 0): 1.001}, 0, (0.7, 0.2)),
    ({(0.7, 0.2): 0, (0, 0): 0}, 0.03, (0, 0)),
])
def test_quality_selection_tolerance(scores, tolerance, expected):
    from reveil.real_pairs import _select_quality

    assert _select_quality(scores, tolerance) == expected


def test_quality_tolerance_rejects_invalid_values():
    for tolerance in (-0.01, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="quality_tolerance"):
            RealPairConfig(quality_tolerance=tolerance)


def test_legacy_encoded_fixture_prefers_identity_with_default_tolerance(real_pair):
    data = real_pair
    config = replace(data["config"], match_quality=True)
    rows = _build(data, data["root"] / "quality-control", config)
    assert len(rows) == len(_plans(data, config)) >= 4
    for row in rows:
        assert row["quality"]["sigma"] == row["quality"]["prev_mix"] == 0
        assert row["quality"]["residual"] == row["quality"]["identity_residual"]
