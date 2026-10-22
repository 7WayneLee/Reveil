"""Video decode and encode through PyAV (FFmpeg); frames are (H, W, 3) RGB uint8 arrays."""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int  # from the container, or counted from the packets when it has none


def probe(path: str | Path) -> VideoInfo:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate or 24
        frames = int(stream.frames or 0)
        if not frames:
            frames = sum(1 for packet in container.demux(stream) if packet.size)
        return VideoInfo(stream.codec_context.width, stream.codec_context.height, float(rate), frames)


def read_frames(path: str | Path, size: tuple[int, int] | None = None,
                threads: int | None = None) -> Iterator[np.ndarray]:
    """Decoded RGB frames; ``size`` scales during conversion, ``threads`` caps the decoder."""
    scaled = {} if size is None else {"width": size[0], "height": size[1], "interpolation": "AREA"}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if threads is not None:
            stream.codec_context.thread_count = threads
        for frame in container.decode(stream):
            yield frame.to_ndarray(format="rgb24", **scaled)


class VideoWriter:
    """Encodes frames of one size into ``path``. Odd sides are padded by one edge pixel,
    since 4:2:0 needs even sides; ``read_frames`` then returns the padded size."""

    def __init__(
        self,
        path: str | Path,
        width: int,
        height: int,
        fps: float,
        *,
        codec: str = "libx264",
        crf: int = 10,
        preset: str = "fast",
    ) -> None:
        self.width, self.height = width, height
        self._pad = (height % 2, width % 2)
        self._container = av.open(str(path), mode="w")
        stream = self._container.add_stream(codec, rate=Fraction(fps).limit_denominator(1001))
        stream.width, stream.height = width + self._pad[1], height + self._pad[0]
        stream.pix_fmt = "yuv420p"
        options = {"crf": str(crf), "preset": preset}
        if codec == "libx265":
            options["x265-params"] = "log-level=error"
        stream.options = options
        self._stream = stream

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"frame {frame.shape} does not match the {self.width}x{self.height} stream")
        if any(self._pad):
            frame = np.pad(frame, ((0, self._pad[0]), (0, self._pad[1]), (0, 0)), mode="edge")
        packet_frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")
        for packet in self._stream.encode(packet_frame):
            self._container.mux(packet)

    def close(self) -> None:
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def write_video(path: str | Path, frames: Iterable[np.ndarray], fps: float, **options) -> int:
    """Write ``frames`` to ``path`` atomically (a ``.part`` file renamed when complete);
    returns the frame count."""
    path = Path(path)
    partial = path.with_name(path.stem + ".part" + path.suffix)
    count = 0
    writer = None
    try:
        for frame in frames:
            if writer is None:
                writer = VideoWriter(partial, frame.shape[1], frame.shape[0], fps, **options)
            writer.write(frame)
            count += 1
    finally:
        if writer is not None:
            writer.close()
    if not count:
        raise ValueError(f"no frames to write to {path}")
    os.replace(partial, path)
    return count
