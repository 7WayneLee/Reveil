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


@pytest.fixture
def make_video(tmp_path: Path):
    def make(name: str = "clip.mp4", frames: int = 160, width: int = 320, height: int = 240, fps: float = 30.0) -> Path:
        path = tmp_path / "videos" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        write_video(path, textured_frames(frames, width, height), fps, crf=16)
        return path

    return make
