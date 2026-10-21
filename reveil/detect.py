"""Ultralytics YOLO detect / segment models exported to ONNX, run with onnxruntime.

Only the exported ONNX file is used, not the ultralytics package: export it once with
``yolo export model=<weights>.pt format=onnx dynamic=True``. Frames are RGB uint8
(H, W, 3); each is letterboxed into the square model input the way ultralytics does
(centred, grey 114 padding) and the results are mapped back to frame pixels. A segment
model's instance masks are cropped to their boxes and merged into one mask per frame.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

PAD_VALUE = 114
MASK_THRESHOLD = 0.5
DEFAULT_SIZE = 640


@dataclass(frozen=True)
class DetectorConfig:
    conf: float = 0.3  # minimum class score
    iou: float = 0.5  # NMS overlap
    imgsz: int | None = None  # model input side; None reads it from the model (else 640)
    classes: tuple[str, ...] = ()  # class names or indices to keep; empty keeps all
    exclude: tuple[str, ...] = ()  # classes whose boxes are cut out of the kept regions
    max_detections: int = 50


@dataclass
class FrameDetections:
    boxes: np.ndarray  # (n, 4) x0, y0, x1, y1 in frame pixels
    scores: np.ndarray  # (n,)
    classes: np.ndarray  # (n,) class indices
    mask: np.ndarray | None  # (H, W) bool union of the instance masks; None for detect models
    excluded: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))  # (m, 4) boxes of excluded classes


@dataclass(frozen=True)
class Letterbox:
    scale: float
    left: int
    top: int
    width: int  # size of the scaled frame inside the input
    height: int


def letterbox(frame: np.ndarray, size: int) -> tuple[np.ndarray, Letterbox]:
    h, w = frame.shape[:2]
    scale = min(size / h, size / w)
    width, height = max(1, round(w * scale)), max(1, round(h * scale))
    left, top = (size - width) // 2, (size - height) // 2
    canvas = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    if (w, h) != (width, height):
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    canvas[top : top + height, left : left + width] = frame
    return canvas, Letterbox(scale, left, top, width, height)


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    """Indices of the boxes kept by greedy non-maximum suppression, best first."""
    order = np.argsort(-scores, kind="stable")
    areas = (boxes[:, 2] - boxes[:, 0]).clip(0) * (boxes[:, 3] - boxes[:, 1]).clip(0)
    keep = []
    while order.size:
        best, rest = order[0], order[1:]
        keep.append(best)
        w = (np.minimum(boxes[best, 2], boxes[rest, 2]) - np.maximum(boxes[best, 0], boxes[rest, 0])).clip(0)
        h = (np.minimum(boxes[best, 3], boxes[rest, 3]) - np.maximum(boxes[best, 1], boxes[rest, 1])).clip(0)
        inter = w * h
        union = areas[best] + areas[rest] - inter
        order = rest[inter <= iou * np.maximum(union, 1e-9)]
    return np.asarray(keep, dtype=np.int64)


def decode(
    pred: np.ndarray, num_classes: int, keep: np.ndarray, conf: float, iou: float, max_detections: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One image's raw output (4 + classes + mask coefficients, anchors) -> boxes (x0, y0,
    x1, y1 in input pixels), scores, class indices and mask coefficients of the detections
    whose best class is one of ``keep``, after per-class NMS. As in ultralytics, each anchor
    takes its best class first: one that is most likely an unwanted class is dropped even
    when a wanted class also scores above ``conf``."""
    pred = pred.T
    class_scores = pred[:, 4 : 4 + num_classes]
    classes = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(len(pred)), classes]
    chosen = (scores >= conf) & np.isin(classes, keep)
    pred, classes, scores = pred[chosen], classes[chosen], scores[chosen]
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    apart = classes[:, None] * (boxes.max(initial=0) + 1)  # shifts each class away, so NMS runs per class
    order = nms(boxes + apart, scores, iou)[:max_detections]
    return boxes[order], scores[order], classes[order], pred[order, 4 + num_classes :]


def instance_union(coefficients: np.ndarray, protos: np.ndarray, boxes: np.ndarray, size: int) -> np.ndarray:
    """Merged instance masks (sigmoid of coefficients x prototypes, each cropped to its box)
    as float probabilities at the input resolution."""
    count, height, width = protos.shape
    union = np.zeros((height, width), dtype=np.float32)
    if len(coefficients):
        masks = 1 / (1 + np.exp(-(coefficients @ protos.reshape(count, -1))))
        masks = masks.reshape(-1, height, width)
        ys = (np.arange(height) + 0.5)[:, None] * size / height
        xs = (np.arange(width) + 0.5)[None, :] * size / width
        for mask, (x0, y0, x1, y1) in zip(masks, boxes):
            inside = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
            np.maximum(union, mask * inside, out=union)
    return cv2.resize(union, (size, size), interpolation=cv2.INTER_LINEAR)


def to_frame_mask(union: np.ndarray, box: Letterbox, frame_w: int, frame_h: int) -> np.ndarray:
    content = union[box.top : box.top + box.height, box.left : box.left + box.width]
    if content.shape != (frame_h, frame_w):
        content = cv2.resize(content, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
    return content > MASK_THRESHOLD


def to_frame_boxes(boxes: np.ndarray, box: Letterbox, frame_w: int, frame_h: int) -> np.ndarray:
    out = boxes.copy()
    out[:, [0, 2]] = ((out[:, [0, 2]] - box.left) / box.scale).clip(0, frame_w)
    out[:, [1, 3]] = ((out[:, [1, 3]] - box.top) / box.scale).clip(0, frame_h)
    return out


def _literal(text: str | None):
    try:
        return ast.literal_eval(text) if text else None
    except (ValueError, SyntaxError):
        return None


class YoloOnnx:
    """Runs an exported YOLO model on batches of frames; CUDA when onnxruntime has it."""

    def __init__(self, path: str | Path, config: DetectorConfig = DetectorConfig(), providers: list[str] | None = None):
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            try:  # CUDA / cuDNN from the nvidia pip packages (onnxruntime-gpu on Colab)
                ort.preload_dlls()
            except Exception:  # noqa: BLE001 - CPU still works
                pass
        if providers is None:
            available = ort.get_available_providers()
            providers = ["CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available:  # GPU memory grows only as needed: several workers share it
                providers.insert(0, ("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}))
        self.session = ort.InferenceSession(str(path), providers=providers)
        self.config = config
        metadata = self.session.get_modelmeta().custom_metadata_map
        model_input = self.session.get_inputs()[0]
        self._input_name = model_input.name
        self._input_dtype = np.float16 if "float16" in model_input.type else np.float32
        self._fixed_batch = model_input.shape[0] if isinstance(model_input.shape[0], int) else None
        outputs = self.session.get_outputs()
        self.task = metadata.get("task") or ("segment" if len(outputs) > 1 else "detect")
        fixed_size = model_input.shape[2] if isinstance(model_input.shape[2], int) else None
        size_meta = _literal(metadata.get("imgsz"))
        self.size = config.imgsz or fixed_size or (size_meta[0] if size_meta else DEFAULT_SIZE)
        channels = outputs[0].shape[1]
        names = _literal(metadata.get("names"))
        if not isinstance(names, dict) and isinstance(channels, int):
            names = {i: str(i) for i in range(channels - 4 - (32 if self.task == "segment" else 0))}
        self.names: dict[int, str] = {int(k): str(v) for k, v in (names or {}).items()}
        expected = 4 + len(self.names) + (32 if self.task == "segment" else 0)
        if isinstance(channels, int) and channels != expected:
            raise ValueError(
                f"{path}: output {outputs[0].shape} is not a raw YOLO {self.task} output with {expected} channels "
                "(4 box + classes + mask coefficients); export it again with nms=False"
            )
        self._keep = self._class_indices(config.classes)
        self._exclude = self._class_indices(config.exclude) if config.exclude else np.zeros(0, dtype=np.int64)
        if np.isin(self._exclude, self._keep).any():
            raise ValueError(f"classes {self.excluded_classes} are both kept and excluded")

    def _class_indices(self, classes: tuple[str, ...]) -> np.ndarray:
        if not classes:
            return np.asarray(sorted(self.names), dtype=np.int64)
        by_name = {name: index for index, name in self.names.items()}
        indices = []
        for item in classes:
            if item in by_name:
                indices.append(by_name[item])
            elif item.isdigit() and int(item) in self.names:
                indices.append(int(item))
            else:
                raise ValueError(f"unknown class {item!r}; the model has {self.names}")
        return np.asarray(indices, dtype=np.int64)

    @property
    def providers(self) -> list[str]:
        return self.session.get_providers()

    @property
    def kept_classes(self) -> list[str]:
        return [self.names[int(index)] for index in self._keep]

    @property
    def excluded_classes(self) -> list[str]:
        return [self.names[int(index)] for index in self._exclude]

    def __call__(self, frames: list[np.ndarray]) -> list[FrameDetections]:
        step = self._fixed_batch or len(frames)
        results = []
        for start in range(0, len(frames), step):
            results.extend(self._run(frames[start : start + step]))
        return results

    def _run(self, frames: list[np.ndarray]) -> list[FrameDetections]:
        canvases, boxes_in = zip(*(letterbox(frame, self.size) for frame in frames))
        # NHWC uint8 -> NCHW scaled to 0..1 in one pass; padded up to a fixed batch size
        count = len(frames)
        batch = np.empty((self._fixed_batch or count, 3, self.size, self.size), dtype=self._input_dtype)
        np.multiply(np.stack(canvases).transpose(0, 3, 1, 2), np.float32(1 / 255), out=batch[:count])
        batch[count:] = 0
        outputs = self.session.run(None, {self._input_name: batch})
        wanted = np.concatenate([self._keep, self._exclude])
        results = []
        for i, (frame, box) in enumerate(zip(frames, boxes_in)):
            h, w = frame.shape[:2]
            found, scores, classes, coefficients = decode(
                outputs[0][i].astype(np.float32, copy=False), len(self.names), wanted,
                self.config.conf, self.config.iou, self.config.max_detections,
            )
            kept = np.isin(classes, self._keep)
            excluded = to_frame_boxes(found[~kept], box, w, h)
            found, scores, classes, coefficients = found[kept], scores[kept], classes[kept], coefficients[kept]
            mask = None
            if self.task == "segment":
                union = instance_union(coefficients, outputs[1][i].astype(np.float32, copy=False), found, self.size)
                mask = to_frame_mask(union, box, w, h)
            results.append(FrameDetections(to_frame_boxes(found, box, w, h), scores, classes, mask, excluded))
        return results
