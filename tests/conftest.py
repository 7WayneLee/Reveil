from pathlib import Path

import numpy as np
import pytest

from reveil.video import write_video


def textured_frames(count: int, width: int, height: int, seed: int = 0):
    """Moving gradients over fixed fine noise: enough texture that censoring shows."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 60, size=(height, width, 3), dtype=np.int16)
    ys, xs = np.mgrid[0:height, 0:width]
    for t in range(count):
        base = np.stack(
            [
                (xs * 255 // width + 2 * t) % 256,
                (ys * 255 // height + 3 * t) % 256,
                ((xs + ys) * 255 // (width + height) + t) % 256,
            ],
            axis=-1,
        )
        yield np.clip(base * 0.75 + noise, 0, 255).astype(np.uint8)


def write_yolo_onnx(path: Path, *, task: str = "segment", size: int = 64) -> Path:
    """A stand-in for an exported ultralytics model: whatever the input, it predicts a
    'region' box (24, 24, 40, 40) with a near duplicate, a low-score box, and an 'other'
    box (46, 46, 54, 54), in input pixels; segment models also get prototype masks."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    classes, masks, anchors = 2, 32 if task == "segment" else 0, 4
    pred = np.zeros((1, 4 + classes + masks, anchors), np.float32)
    for a, (cx, cy, w, h, cls, score) in enumerate(
        [(32, 32, 16, 16, 0, 0.9), (33, 32, 16, 16, 0, 0.8), (10, 10, 6, 6, 0, 0.1), (50, 50, 8, 8, 1, 0.95)]
    ):
        pred[0, :4, a] = cx, cy, w, h
        pred[0, 4 + cls, a] = score
        if masks:
            pred[0, 4 + classes + cls, a] = 1.0  # class c uses prototype c
    side = size // 4
    protos = np.full((1, 32, side, side), -10.0, np.float32)
    protos[0, 0, 5:12, 5:12] = 10.0  # input pixels 20..48
    protos[0, 1, 11:15, 11:15] = 10.0  # input pixels 44..60
    init = [
        numpy_helper.from_array(np.array([0], np.int64), "zero"),
        numpy_helper.from_array(np.array([1], np.int64), "one"),
        numpy_helper.from_array(pred, "pred"),
        numpy_helper.from_array(np.array(pred.shape[1:], np.int64), "rest0"),
    ]
    nodes = [
        helper.make_node("Shape", ["images"], ["shape"]),
        helper.make_node("Slice", ["shape", "zero", "one"], ["batch"]),
        helper.make_node("Concat", ["batch", "rest0"], ["shape0"], axis=0),
        helper.make_node("Expand", ["pred", "shape0"], ["output0"]),
    ]
    outputs = [helper.make_tensor_value_info("output0", TensorProto.FLOAT, ["batch", *pred.shape[1:]])]
    if task == "segment":
        init += [
            numpy_helper.from_array(protos, "protos"),
            numpy_helper.from_array(np.array(protos.shape[1:], np.int64), "rest1"),
        ]
        nodes += [
            helper.make_node("Concat", ["batch", "rest1"], ["shape1"], axis=0),
            helper.make_node("Expand", ["protos", "shape1"], ["output1"]),
        ]
        outputs.append(helper.make_tensor_value_info("output1", TensorProto.FLOAT, ["batch", *protos.shape[1:]]))
    images = helper.make_tensor_value_info("images", TensorProto.FLOAT, ["batch", 3, size, size])
    model = helper.make_model(
        helper.make_graph(nodes, "yolo", [images], outputs, initializer=init),
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 8
    helper.set_model_props(model, {"names": "{0: 'region', 1: 'other'}", "task": task, "imgsz": f"[{size}, {size}]"})
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


@pytest.fixture
def make_video(tmp_path: Path):
    def make(name: str = "clip.mp4", frames: int = 160, width: int = 320, height: int = 240, fps: float = 30.0) -> Path:
        path = tmp_path / "videos" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        write_video(path, textured_frames(frames, width, height), fps, crf=16)
        return path

    return make
