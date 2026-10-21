import numpy as np
import pytest

from conftest import write_yolo_onnx
from reveil.detect import DetectorConfig, decode, instance_union, letterbox, nms, to_frame_boxes

pytest.importorskip("onnxruntime")


def test_letterbox_centres_the_scaled_frame_on_grey():
    frame = np.full((96, 128, 3), 200, np.uint8)
    canvas, box = letterbox(frame, 64)
    assert canvas.shape == (64, 64, 3)
    assert (box.scale, box.left, box.top, box.width, box.height) == (0.5, 0, 8, 64, 48)
    assert (canvas[:8] == 114).all() and (canvas[8:56] == 200).all() and (canvas[56:] == 114).all()


def test_nms_drops_overlaps_and_keeps_separate_boxes():
    boxes = np.array([[0, 0, 10, 10], [1, 0, 11, 10], [20, 20, 30, 30]], float)
    keep = nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5)
    assert keep.tolist() == [0, 2]


def test_decode_filters_scores_and_classes():
    pred = np.zeros((4 + 2, 3), np.float32)
    pred[:, 0] = [10, 10, 4, 4, 0.9, 0.0]
    pred[:, 1] = [30, 30, 4, 4, 0.2, 0.0]  # below the threshold
    pred[:, 2] = [50, 50, 4, 4, 0.0, 0.8]  # class 1
    boxes, scores, classes, coefficients = decode(pred, 2, np.array([0]), 0.3, 0.5, 10)
    np.testing.assert_allclose(boxes, [[8, 8, 12, 12]])
    assert classes.tolist() == [0] and coefficients.shape == (1, 0)
    _, _, both, _ = decode(pred, 2, np.array([0, 1]), 0.3, 0.5, 10)
    assert sorted(both.tolist()) == [0, 1]


def test_decode_drops_anchors_whose_best_class_is_unwanted():
    pred = np.zeros((4 + 2, 2), np.float32)
    pred[:, 0] = [10, 10, 4, 4, 0.4, 0.7]  # more likely class 1, though class 0 is above the threshold
    pred[:, 1] = [11, 10, 4, 4, 0.6, 0.0]  # overlaps the first
    boxes, _, classes, _ = decode(pred, 2, np.array([0]), 0.3, 0.5, 10)
    np.testing.assert_allclose(boxes, [[9, 8, 13, 12]])
    _, _, both, _ = decode(pred, 2, np.array([0, 1]), 0.3, 0.5, 10)
    assert sorted(both.tolist()) == [0, 1], "NMS runs per class"


def test_instance_masks_are_cropped_to_their_boxes():
    protos = np.full((1, 16, 16), 10.0, np.float32)  # the prototype covers everything
    union = instance_union(np.array([[1.0]], np.float32), protos, np.array([[16, 16, 32, 32]], float), 64)
    assert union.shape == (64, 64)
    assert union[24, 24] > 0.9 and union[8, 8] < 0.1 and union[48, 48] < 0.1


def test_boxes_map_back_to_frame_pixels():
    _, box = letterbox(np.zeros((96, 128, 3), np.uint8), 64)
    np.testing.assert_allclose(to_frame_boxes(np.array([[24.0, 24, 40, 40]]), box, 128, 96), [[48, 32, 80, 64]])


def test_segment_model_end_to_end(tmp_path):
    from reveil.detect import YoloOnnx

    detector = YoloOnnx(write_yolo_onnx(tmp_path / "seg.onnx"), DetectorConfig(classes=("region",)))
    assert detector.task == "segment" and detector.size == 64 and detector.kept_classes == ["region"]
    frames = [np.zeros((96, 128, 3), np.uint8)] * 3
    results = detector(frames)
    assert len(results) == 3
    found = results[0]
    np.testing.assert_allclose(found.boxes, [[48, 32, 80, 64]], atol=0.01)
    assert found.mask.shape == (96, 128)
    ys, xs = np.nonzero(found.mask)
    assert 46 <= xs.min() <= 50 and 78 <= xs.max() <= 82 and 30 <= ys.min() <= 34 and 62 <= ys.max() <= 66

    everything = YoloOnnx(write_yolo_onnx(tmp_path / "seg2.onnx"))(frames[:1])[0]
    assert sorted(everything.classes.tolist()) == [0, 1]
    assert everything.mask[84, 100] and not everything.mask[5, 5]


def test_end_to_end_exports_are_refused(tmp_path):
    import onnx
    from onnx import helper

    from reveil.detect import YoloOnnx

    path = write_yolo_onnx(tmp_path / "seg.onnx")
    model = onnx.load(str(path))
    helper.set_model_props(model, {"names": "{0: 'region'}", "task": "segment", "imgsz": "[64, 64]"})  # 1 class
    onnx.save(model, str(path))
    with pytest.raises(ValueError, match="nms=False"):
        YoloOnnx(path)


def test_excluded_classes_come_back_as_separate_boxes(tmp_path):
    from reveil.detect import YoloOnnx

    path = write_yolo_onnx(tmp_path / "det.onnx", task="detect", names=("genitalia", "anus", "face"))
    detector = YoloOnnx(path, DetectorConfig(classes=("genitalia",), exclude=("anus",)))
    assert detector.kept_classes == ["genitalia"] and detector.excluded_classes == ["anus"]
    found = detector([np.zeros((96, 128, 3), np.uint8)])[0]
    np.testing.assert_allclose(found.boxes, [[48, 32, 80, 64]], atol=0.01)
    np.testing.assert_allclose(found.excluded, [[92, 76, 108, 92]], atol=0.01)
    with pytest.raises(ValueError, match="both kept and excluded"):
        YoloOnnx(path, DetectorConfig(classes=("genitalia", "anus"), exclude=("anus",)))


def test_doubt_classes_come_back_as_separate_boxes(tmp_path):
    from reveil.detect import YoloOnnx

    path = write_yolo_onnx(tmp_path / "det.onnx", task="detect", names=("genitalia", "face", "anus"))
    detector = YoloOnnx(path, DetectorConfig(classes=("genitalia",), exclude=("anus",), doubt=("face",)))
    assert detector.doubt_classes == ["face"]
    found = detector([np.zeros((96, 128, 3), np.uint8)])[0]
    np.testing.assert_allclose(found.boxes, [[48, 32, 80, 64]], atol=0.01)
    np.testing.assert_allclose(found.doubt, [[92, 76, 108, 92]], atol=0.01)
    assert found.excluded.shape == (0, 4)
    with pytest.raises(ValueError, match="neither kept nor excluded"):
        YoloOnnx(path, DetectorConfig(classes=("genitalia",), exclude=("face",), doubt=("face",)))


def test_detect_model_and_unknown_classes(tmp_path):
    from reveil.detect import YoloOnnx

    path = write_yolo_onnx(tmp_path / "det.onnx", task="detect")
    found = YoloOnnx(path, DetectorConfig(classes=("0",)))([np.zeros((96, 128, 3), np.uint8)])[0]
    assert found.mask is None and len(found.boxes) == 1
    with pytest.raises(ValueError, match="unknown class"):
        YoloOnnx(path, DetectorConfig(classes=("penguin",)))
