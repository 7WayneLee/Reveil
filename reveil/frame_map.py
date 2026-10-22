"""Decode-order frame matches between a retail release and its uncensored leak."""

from __future__ import annotations

import json
import os
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Alignment:
    """Matched runs follow retail order, allowing the leak's shots to be reordered."""

    retail: str
    leak: str
    retail_frames: int
    leak_frames: int
    retail_fps: float
    leak_fps: float
    retail_size: tuple[int, int]  # width, height
    leak_size: tuple[int, int]
    runs: tuple[tuple[int, int, int], ...]  # retail start, leak start, length
    segments: tuple[dict, ...] = ()
    stats: dict = field(default_factory=dict)
    _starts: tuple[int, ...] = field(init=False, repr=False, compare=False)
    _run_array: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.retail_frames < 0 or self.leak_frames < 0:
            raise ValueError("frame counts must be non-negative")
        previous_start, previous_stop = -1, 0
        for retail_start, leak_start, length in self.runs:
            if length < 1:
                raise ValueError("run length must be at least 1")
            if retail_start < 0 or retail_start + length > self.retail_frames:
                raise ValueError("run retail indices are outside [0, retail_frames)")
            if leak_start < 0 or leak_start + length > self.leak_frames:
                raise ValueError("run leak indices are outside [0, leak_frames)")
            if retail_start < previous_start:
                raise ValueError("runs must be sorted by retail_start")
            if retail_start < previous_stop:
                raise ValueError("runs overlap on the retail side")
            previous_start, previous_stop = retail_start, retail_start + length
        object.__setattr__(self, "_starts", tuple(run[0] for run in self.runs))
        run_array = np.asarray(self.runs, dtype=np.int64).reshape(-1, 3)
        run_array.setflags(write=False)
        object.__setattr__(self, "_run_array", run_array)

    def leak_index(self, retail_index: int) -> int | None:
        """The leak frame matching one retail frame, or None in an unmatched gap."""
        run_index = bisect_right(self._starts, retail_index) - 1
        if run_index < 0:
            return None
        retail_start, leak_start, length = self.runs[run_index]
        offset = retail_index - retail_start
        return leak_start + offset if offset < length else None

    def leak_indices(self, start: int, stop: int) -> np.ndarray:
        """Leak indices for a retail range, with -1 for each unmatched frame."""
        indices = np.arange(start, stop, dtype=np.int64)
        result = np.full(indices.shape, -1, dtype=np.int64)
        if not self.runs or not indices.size:
            return result
        run_indices = np.searchsorted(self._run_array[:, 0], indices, side="right") - 1
        runs = self._run_array[np.maximum(run_indices, 0)]
        matched = (run_indices >= 0) & (indices < runs[:, 0] + runs[:, 2])
        result[matched] = runs[matched, 1] + indices[matched] - runs[matched, 0]
        return result

    def window_ok(self, start: int, length: int) -> bool:
        """Whether every retail frame has a target, even across reordered leak shots."""
        if start < 0 or length < 0 or start + length > self.retail_frames:
            return False
        return bool(np.all(self.leak_indices(start, start + length) >= 0))

    def matched_frames(self) -> int:
        """The number of retail frames with a matching leak frame."""
        return sum(length for _, _, length in self.runs)

    def leak_jumps(self, start: int, stop: int) -> int:
        """Count discontinuities between adjacent matched frames; +1 and +2 are smooth."""
        first = max(1, bisect_right(self._starts, start))
        last = bisect_right(self._starts, stop - 1)
        jumps = 0
        for index in range(first, last):
            retail_start, leak_start, _ = self.runs[index]
            previous_retail, previous_leak, length = self.runs[index - 1]
            if previous_retail + length == retail_start:
                step = leak_start - (previous_leak + length - 1)
                jumps += step not in (1, 2)
        return jumps


def save_alignment(alignment: Alignment, path: str | Path) -> None:
    """Replace an alignment atomically so readers never see a partial frame map."""
    data = {
        "version": 1,
        "retail": alignment.retail,
        "leak": alignment.leak,
        "retail_frames": alignment.retail_frames,
        "leak_frames": alignment.leak_frames,
        "retail_fps": alignment.retail_fps,
        "leak_fps": alignment.leak_fps,
        "retail_size": alignment.retail_size,
        "leak_size": alignment.leak_size,
        "runs": alignment.runs,
        "segments": alignment.segments,
        "stats": alignment.stats,
    }
    path = Path(path)
    partial = path.with_name(path.name + ".part")
    partial.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    os.replace(partial, path)


def load_alignment(path: str | Path) -> Alignment:
    """Read a versioned frame map, checking its run bounds and retail ordering."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("version") != 1:
        raise ValueError(f"{path}: unsupported alignment version {data.get('version')}")
    return Alignment(
        retail=data["retail"],
        leak=data["leak"],
        retail_frames=data["retail_frames"],
        leak_frames=data["leak_frames"],
        retail_fps=data["retail_fps"],
        leak_fps=data["leak_fps"],
        retail_size=tuple(data["retail_size"]),
        leak_size=tuple(data["leak_size"]),
        runs=tuple(tuple(run) for run in data["runs"]),
        segments=tuple(data.get("segments", [])),
        stats=data.get("stats", {}),
    )
