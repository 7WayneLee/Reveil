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


def test_plain_leak_preserves_both_source_canvases_and_skips_matching(real_pair, tmp_path, monkeypatch):
    import reveil.real_pairs as module

    data = real_pair
    config = replace(data["config"], target="leak", colour_fit=False, match_quality=True,
                     max_windows_per_title=2)
    plans = {p.key: p for p in _plans(data, config)}
    assert plans and all(p.previous_leak_index is None for p in plans.values())
    encode = module.write_video

    def checked_encode(path, images, fps, **options):
        plan = plans[path.stem]
        target = path.parent.name == "clean"

        def checked_canvases():
            for index, image in enumerate(images):
                source = (data["leak"][plan.leak_indices[index]] if target
                          else data["retail"][plan.start + index])
                np.testing.assert_array_equal(image, crop_to_canvas(source, plan.crops[index]))
                yield image

        return encode(path, checked_canvases(), fps, **options)

    def unexpected(*args, **kwargs):
        pytest.fail("plain leak windows must skip masks, colour/quality fitting and composite work")

    for name in ("_masks", "_ring_mask", "_fit_colour", "_fit_colour_scored", "_blur", "_redetect"):
        monkeypatch.setattr(module, name, unexpected)
    for name in ("_samples", "quality_fit", "ring_score", "_matched"):
        monkeypatch.setattr(module._WindowBuffer, name, unexpected)
    monkeypatch.setattr(module, "write_video", checked_encode)
    out = tmp_path / "plain"
    rows = _build(data, out, config)
    assert len(rows) == 2
    assert json.loads((out / "meta/TITLE.json").read_text())["windows"] == rows
    for row in rows:
        plan = plans[row["key"]]
        assert row["colour_fit"] is None
        assert row["quality"] == {"sigma": 0, "prev_mix": 0, "residual": None, "identity_residual": None}
        assert row["ring"] == {"score": None, "median": None, "max": None, "frames": 0}
        assert row["redetect"] == {"added_share_mean": 0, "added_share_max": 0}
        targets = list(read_frames(out / row["video"]))
        references = list(read_frames(out / row["reference_video"]))
        assert len(targets) == len(references) == config.window
        assert row["crops"] == [list(c.box) for c in plan.crops]
        for index, (target, reference, crop) in enumerate(zip(targets, references, plan.crops)):
            expected_target = crop_to_canvas(data["leak"][plan.leak_indices[index]], crop)
            expected_reference = crop_to_canvas(data["retail"][plan.start + index], crop)
            assert np.abs(target.astype(float) - expected_target).mean() < 2
            assert np.abs(reference.astype(float) - expected_reference).mean() < 2


@pytest.mark.parametrize("side", ["retail", "leak"])
def test_plain_leak_still_drops_views_touching_active_overlays(real_pair, side):
    data = real_pair
    config = replace(data["config"], target="leak", colour_fit=False)
    plans = _plans(data, config)
    assert [p.key for p in plans] == [p.key for p in _plans(data, replace(config, colour_fit=True))]
    overlay = Overlay((0, 0, 320, 180), side, 1, (100, 150))
    regions = replace(data["regions"], overlays=(overlay,))
    plans = plan_real_windows(data["alignment"], regions, "TITLE", config, np.random.default_rng(0))
    assert plans and all(p.start + config.window <= 100 or p.start >= 150 for p in plans)


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


@pytest.mark.parametrize("setting", ["cli", "yaml"])
def test_cli_and_yaml_disable_colour_fit(real_pair, tmp_path, setting):
    data = real_pair
    config_path = tmp_path / "plain.yaml"
    config_path.write_text("pairs:\n  window: 3\n  hop: 121\n  target: leak\n"
                           f"  colour_fit: {'true' if setting == 'cli' else 'false'}\n"
                           "camera:\n  canvas: 128\n")
    assert load_config(config_path).colour_fit == (setting == "cli")
    out = tmp_path / setting
    args = ["--align-dir", str(data["alignment_path"].parent),
            "--regions-dir", str(data["regions_path"].parent), "--out", str(out),
            "--config", str(config_path), "--workers", "1", "--max-windows-per-title", "1"]
    if setting == "cli":
        args.append("--no-colour-fit")
    assert main(args) == 0
    row, = json.loads((out / "meta/TITLE.json").read_text())["windows"]
    assert row["target"] == "leak" and row["colour_fit"] is None


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


@pytest.mark.parametrize("colour_fit", [False, True])
def test_resize_backward_mapping_and_single_forward_decode(real_pair, tmp_path, monkeypatch, colour_fit):
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
    rendered = {}
    encode = module.write_video

    def capture(path, images, fps, **options):
        frames = list(images)
        rendered[path] = frames
        return encode(path, frames, fps, **options)

    def fake_decode(path, *, threads=None):
        calls.append((path, threads))
        if path == alignment.retail:
            yield from map(_RGBFrame, data["retail"])
        else:
            for frame in data["leak"]:
                yield _RGBFrame(cv2.resize(frame, (640, 360), interpolation=cv2.INTER_NEAREST))

    monkeypatch.setattr(module, "decoded_frames", fake_decode)
    monkeypatch.setattr(module, "write_video", capture)
    config = replace(data["config"], target="leak", max_windows_per_title=2,
                     colour_sample_frames=2, colour_fit=colour_fit)
    out = tmp_path / "out"
    rows = build_title_pairs("TITLE", alignment_path, regions_path, out, config, decode_threads=3)
    assert len(rows) == 2
    assert any(np.any(np.diff(row["leak_indices"]) < 0) for row in rows)
    assert calls == [(alignment.retail, 3), (alignment.leak, 3)]
    for row in rows:
        if colour_fit:
            gain, offset = np.array(row["colour_fit"]["a"]), np.array(row["colour_fit"]["b"])
        else:
            assert row["colour_fit"] is None
            gain, offset = 1, 0
        # Tight views retain fine chroma that is lost to 4:2:0 even at CRF zero;
        # verify the mapping before coding while still exercising MP4 output.
        assert probe(out / row["video"]).frames == config.window
        for index, target in enumerate(rendered[out / row["video"]]):
            crop = CropFrame(tuple(row["crops"][index]), 128)
            leak = crop_to_canvas(data["leak"][row["leak_indices"][index]], crop)
            expected = np.rint(leak * gain + offset).clip(0, 255)
            assert np.abs(target.astype(float) - expected).mean() < 0.01


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
                            blur_sigmas=(0, 0.2, 0.4, 1.0),  # exercise configurable sigma-1 matching
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


@pytest.mark.parametrize("match_quality", [False, True])
@pytest.mark.parametrize("colour", [False, True])
def test_composite_without_colour_fit_uses_identity_colours(tmp_path, monkeypatch, match_quality, colour):
    import reveil.real_pairs as module

    data = _quality_pair(tmp_path, monkeypatch, colour=colour)
    config = replace(data["config"], colour_fit=False, match_quality=match_quality,
                     redetect=False, ring_max_residual=0)

    def unexpected(*args, **kwargs):
        pytest.fail("disabled colour fitting must not run a regression")

    monkeypatch.setattr(module, "_fit_colour", unexpected)
    monkeypatch.setattr(module, "_fit_colour_scored", unexpected)
    if not match_quality:
        monkeypatch.setattr(module._WindowBuffer, "_samples", unexpected)
    out = tmp_path / "identity-colours"
    row, = _build(data, out, config)
    assert row["colour_fit"] is None
    assert row["quality"]["sigma"] == (1 if match_quality else 0)
    assert row["quality"]["prev_mix"] == (0.2 if match_quality else 0)
    if match_quality:
        assert row["quality"]["residual"] < row["quality"]["identity_residual"] / 10
    else:
        assert row["quality"]["residual"] is row["quality"]["identity_residual"] is None
    assert row["ring"]["frames"] == config.window
    plan, = _plans(data, config)
    expected = []
    for index, crop in enumerate(plan.crops):
        image = crop_to_canvas(data["leak"][plan.leak_indices[index]], crop).astype(np.float32)
        if match_quality:
            image = cv2.GaussianBlur(image, (0, 0), sigmaX=1)
            previous_index = plan.previous_leak_index if index == 0 else plan.leak_indices[index - 1]
            if previous_index is not None:
                previous = crop_to_canvas(data["leak"][previous_index], crop).astype(np.float32)
                image = 0.8 * image + 0.2 * cv2.GaussianBlur(previous, (0, 0), sigmaX=1)
        _, alpha = _masks(plan.regions[index], plan.work, 96, config)
        alpha = crop_to_canvas(np.repeat(alpha[..., None], 3, axis=2), crop)[..., 0, None]
        retail = crop_to_canvas(data["retail"][plan.start + index], crop)
        expected.append(np.rint(retail * (1 - alpha) + image * alpha).clip(0, 255).astype(np.uint8))
    assert np.abs(data["written"][out / row["video"]].astype(float) - np.stack(expected)).mean() < 0.01
    np.testing.assert_array_equal(data["written"][out / row["reference_video"]], data["retail"][plan.start:])


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
    # Keep the mocked 64px views moving rather than growing them to a 96px
    # full-frame view under the new no-upscaling rule.
    camera = replace(data["config"].camera, canvas=64) if mapping == "moving-crop" else data["config"].camera
    config = replace(data["config"], blur_sigmas=(0,), camera=camera)
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
    assert all(flag in help_text for flag in ("--no-colour-fit", "--no-match-quality",
                                             "--ring-max-residual", "--candidate-factor"))


def test_quality_and_ring_exclude_border_and_active_overlay(tmp_path, monkeypatch):
    data = _quality_pair(tmp_path, monkeypatch)
    overlay = Overlay((17, 34, 21, 62), "retail", 1.0, (3, 6))
    regions = replace(data["regions"], overlays=(overlay,))
    save_regions(regions, data["regions_path"])
    for index, frame in enumerate(data["retail"]):
        frame[:4] = frame[-4:] = 0
        frame[:, :4] = frame[:, -4:] = 0
        if overlay.active(index):
            frame[34:62, 17:21] = (255, 0, 255)
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
    assert ring[40, 83] and not ring[40, 85]  # second region: 12 growth + 12 ring
    assert ring[79, 12] and not ring[81, 12]  # first region: 8 growth + 8 ring


def test_forward_spool_keeps_previous_canvases_for_future_moving_crops(tmp_path, monkeypatch):
    import reveil.real_pairs as module

    indices = list(range(20, 29)) + list(range(10))
    data = _quality_pair(tmp_path, monkeypatch, count=19, indices=indices, sigma=0, colour=False)
    monkeypatch.setattr(module, "plan_crops", lambda boxes, *args: [
        CropFrame((16 + i % 3, 16, 80 + i % 3, 80), 64) for i in range(len(boxes))])
    config = replace(data["config"], window=5, hop=5, blur_sigmas=(0,),
                     camera=replace(data["config"].camera, canvas=64))
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


def test_real_pair_defaults_and_yaml_for_tight_views_and_redetection(tmp_path):
    config = RealPairConfig()
    assert config.colour_fit
    assert config.mask_grow_blocks == 1 and config.feather_blocks == 0.25
    assert config.blur_sigmas == (0, 0.2, 0.4)
    assert config.prev_mixes == (0, 0.1, 0.2, 0.3, 0.4) and config.quality_tolerance == 0.03
    assert config.redetect and config.redetect_min_diff == 12 and config.redetect_max_std == 3
    assert config.camera.canvas == 512 and config.camera.large_canvas == 768
    assert config.camera.large_min_side == 640
    assert config.camera.border_fraction == 0.06 and config.camera.min_border == 20
    path = tmp_path / "tight.yaml"
    path.write_text("pairs:\n  redetect: false\n  redetect_min_diff: 15\n  redetect_max_std: 2\n"
                    "camera:\n  canvas: 256\n  large_canvas: 512\n  large_min_side: 400\n"
                    "  min_border: 12\n  border_fraction: 0.04\n")
    custom = load_config(path)
    assert not custom.redetect and custom.redetect_min_diff == 15 and custom.redetect_max_std == 2
    assert custom.camera.canvas == 256 and custom.camera.large_canvas == 512
    assert custom.camera.large_min_side == 400 and custom.camera.min_border == 12
    assert custom.camera.border_fraction == 0.04


def test_odd_frame_masks_include_other_tracks_and_do_not_cross_cuts():
    first = Region((150, 150, 210, 210), block=8)
    other = Region((260, 150, 320, 210), block=8)
    future = Region((400, 150, 460, 210), block=8)
    regions = RegionSet({0: [first, other], 2: [first, other], 4: [first, future]}, frozenset({4}))
    alignment = Alignment("retail", "leak", 5, 5, 24, 24, (640, 640), (640, 640), ((0, 0, 5),))
    config = RealPairConfig(window=3, hop=3)
    plans = plan_real_windows(alignment, regions, "odd", config, np.random.default_rng(0))
    assert len(plans) == 2
    for plan in plans:
        mask, _ = _masks(plan.regions[1], (0, 0, 640, 640), 640, config, feather=False)
        assert mask[180, 180] and mask[180, 290]
        assert not mask[180, 430]
    # At frame 3 the next analysed frame is a new scene: use frame 2 even
    # though frame 4 is equally near, and never borrow that scene's future area.
    from reveil.tracks import Track
    import reveil.real_pairs as module
    # The last gap is not normally emitted by build_tracks, so make that boundary
    # explicit without reading or decoding any footage.
    from unittest.mock import patch
    with patch.object(module, "build_tracks", return_value=[Track(3, [first.box], [[first]])]):
        plan, = plan_real_windows(alignment, regions, "cut", replace(config, window=1), np.random.default_rng(0))
    mask, _ = _masks(plan.regions[0], (0, 0, 640, 640), 640, config, feather=False)
    assert mask[180, 290] and not mask[180, 430]


def test_gap_view_contains_bordered_regions_even_when_track_box_is_interpolated():
    first = Region((100, 100, 600, 600), block=16)
    later = Region((500, 100, 1000, 600), block=16)
    regions = RegionSet({0: [first], 2: [later]})
    alignment = Alignment("retail", "leak", 3, 3, 24, 24, (1920, 1080), (1920, 1080), ((0, 0, 3),))
    plan, = plan_real_windows(alignment, regions, "gap", RealPairConfig(window=3), np.random.default_rng(0))
    assert plan.boxes[1] == first.box
    x0, y0, x1, y1 = plan.crops[1].box
    assert x0 <= 70 and y0 <= 70 and x1 >= 630 and y1 >= 630


def _missed_pair(tmp_path, monkeypatch, *, missed=True, overlay_side=None):
    """Two synthetic pixelated patches, one absent from the regions file."""
    import reveil.real_pairs as module

    size, count = 192, 3
    y, x = np.indices((size, size))
    grey = np.where((x + y) % 2, 220, 30).astype(np.uint8)
    leak = np.repeat(grey[..., None], 3, axis=2)
    retail = leak.copy()
    region = Region((48, 72, 80, 104), block=8)
    retail[72:104, 48:80] = 125
    if missed:
        retail[64:112, 120:160] = 125
    if overlay_side:
        retail[64:112, 120:160] = 125
        leak[64:112, 120:160] = 240 if overlay_side == "leak" else 30
    overlays = (Overlay((120, 64, 160, 112), overlay_side, 1, (0, 1)),) if overlay_side else ()
    regions = RegionSet({i: [region] for i in range(count)}, overlays=overlays)
    alignment = Alignment(str(tmp_path / "retail.mp4"), str(tmp_path / "leak.mp4"), count, count,
                          24, 24, (size, size), (size, size), ((0, 0, count),))
    align_path, regions_path = tmp_path / "align.json", tmp_path / "regions.json"
    save_alignment(alignment, align_path)
    save_regions(regions, regions_path)
    written = {}

    def decode(path, *, threads=None):
        yield from (_RGBFrame(retail if path == alignment.retail else leak) for _ in range(count))

    def encode(path, images, fps, *, crf):
        written[path] = np.stack(list(images))
        path.write_bytes(written[path].tobytes())

    monkeypatch.setattr(module, "decoded_frames", decode)
    monkeypatch.setattr(module, "write_video", encode)
    config = RealPairConfig(window=count, hop=count, match_quality=False,
                            camera=CameraConfig(canvas=size), overlay_margin=0)
    return {"alignment": alignment, "alignment_path": align_path, "regions_path": regions_path,
            "regions": regions, "config": config, "retail": retail, "leak": leak, "written": written}


@pytest.mark.parametrize("colour_fit", [False, True])
def test_redetection_replaces_missed_pixelation_and_keeps_region_only_fit(tmp_path, monkeypatch, colour_fit):
    data = _missed_pair(tmp_path, monkeypatch)
    config = replace(data["config"], colour_fit=colour_fit)
    out = tmp_path / "out"
    row, = _build(data, out, config)
    assert (row["colour_fit"] is not None) == colour_fit
    assert row["canvas"] == 192
    assert 0 < row["redetect"]["added_share_mean"] <= row["redetect"]["added_share_max"]
    assert row["ring"]["score"] < 1e-4
    actual = data["written"][out / row["video"]]
    error = np.abs(actual[:, 64:112, 120:160].astype(float) - data["leak"][64:112, 120:160])
    assert error.max() <= 1 and error.mean() < 0.01  # feathered outer corner only
    off, = _build(data, tmp_path / "off", replace(config, redetect=False))
    assert off["redetect"] == {"added_share_mean": 0, "added_share_max": 0}
    assert off["colour_fit"] == row["colour_fit"] and off["quality"] == row["quality"]
    unchanged = data["written"][tmp_path / "off" / off["video"]]
    assert np.abs(unchanged[:, 70:106, 126:154].astype(float) - actual[:, 70:106, 126:154]).mean() > 50


def test_redetection_adds_nothing_when_all_pixelation_is_in_regions(tmp_path, monkeypatch):
    data = _missed_pair(tmp_path, monkeypatch, missed=False)
    row, = _build(data, tmp_path / "out")
    assert row["redetect"] == {"added_share_mean": 0, "added_share_max": 0}


@pytest.mark.parametrize("side", ["retail", "leak"])
def test_redetection_excludes_active_overlays_of_either_side(tmp_path, monkeypatch, side):
    data = _missed_pair(tmp_path, monkeypatch, overlay_side=side)
    row, = _build(data, tmp_path / "out")
    actual = data["written"][tmp_path / "out" / row["video"]]
    # The overlay is active only on frame zero: later frames deliberately prove
    # that exclusion follows its active range rather than excluding it forever.
    np.testing.assert_array_equal(actual[0, 64:112, 120:160], data["retail"][64:112, 120:160])
    assert row["redetect"]["added_share_max"] > row["redetect"]["added_share_mean"] > 0


def test_redetection_ignores_small_components_and_canvas_border():
    from reveil.real_pairs import _redetect, _exclude_border

    retail = np.full((96, 96, 3), 125, np.uint8)
    leak = retail.copy()
    leak[1:3, :] = 240
    leak[46:48, 46:48] = 240
    excluded = np.zeros((96, 96), bool)
    _exclude_border(excluded)
    assert not _redetect(retail, leak, 8, excluded, RealPairConfig()).any()


def test_planned_custom_canvas_is_shared_and_recorded(tmp_path, monkeypatch):
    data = _missed_pair(tmp_path, monkeypatch, missed=False)
    camera = replace(data["config"].camera, canvas=128, large_canvas=160, large_min_side=127)
    config = replace(data["config"], camera=camera)
    row, = _build(data, tmp_path / "out", config)
    assert row["canvas"] == 160
    assert data["written"][tmp_path / "out" / row["video"]].shape == (3, 160, 160, 3)


def test_seam_gate_measures_ring_around_redetected_area(tmp_path, monkeypatch):
    import reveil.real_pairs as module

    data = _missed_pair(tmp_path, monkeypatch)
    original = module._redetect

    def marked_patch(*args):
        detected = original(*args)
        # Add a mismatch immediately outside the new seam, far from the known
        # region's ring. It must enter the score for the final union mask.
        retail, matched = args[:2]
        matched[44:64, 120:160] = retail[44:64, 120:160] * 0.0
        return detected

    monkeypatch.setattr(module, "_redetect", marked_patch)
    row, = _build(data, tmp_path / "out", replace(data["config"], ring_max_residual=0))
    assert row["ring"]["score"] > 1
    assert row["redetect"]["added_share_mean"] > 0


def test_redetection_without_regions_uses_fallback_block(tmp_path, monkeypatch):
    import reveil.real_pairs as module

    data = _missed_pair(tmp_path, monkeypatch)
    config = data["config"]
    crop = CropFrame((0, 0, 192, 192), 192)
    plan = module.RealWindowPlan("fallback", 0, 0, [crop.box], [[]], [crop], crop.box, (0,))
    buffer = module._WindowBuffer(plan, tmp_path, config)
    buffer.activate()
    for side in ("retail", "leak"):
        buffer.add(side, 0, data[side], data["alignment"], RegionSet(), config)
    out = tmp_path / "fallback"
    for name in ("clean", "mosaic"):
        (out / name).mkdir(parents=True)
    row = buffer.finish("TITLE", data["alignment"], out, config)
    assert row["redetect"]["added_share_mean"] > 0
    restored = data["written"][out / row["video"]][0]
    error = np.abs(restored[70:106, 126:154].astype(float) - data["leak"][70:106, 126:154])
    assert error.max() == 0
