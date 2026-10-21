"""Region tracks: one censored area followed across frames, never across a scene cut.

Overlapping regions of a frame form one group; groups are linked frame to frame by box
IoU (greedy, best match first). A track survives up to ``max_gap`` frames without a
match; those frames get an interpolated box and the regions of the nearer known frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from reveil.regions import Box, Region, RegionSet, union_box


@dataclass
class Track:
    start: int  # first frame index
    boxes: list[Box]  # union box of the track's regions on each frame from ``start``
    regions: list[list[Region]]  # the regions on each frame

    @property
    def stop(self) -> int:
        return self.start + len(self.boxes)


@dataclass(eq=False)
class _Building:
    last: int
    frames: dict[int, tuple[Box, list[Region]]] = field(default_factory=dict)

    def add(self, frame: int, box: Box, group: list[Region]) -> None:
        self.frames[frame] = (box, group)
        self.last = frame

    def finish(self) -> Track:
        known = sorted(self.frames)
        boxes, regions = [], []
        for left, right in zip(known, known[1:] + [known[-1] + 1]):
            box, group = self.frames[left]
            boxes.append(box)
            regions.append(group)
            if right - left > 1:
                next_box, next_group = self.frames[right]
                for frame in range(left + 1, right):
                    t = (frame - left) / (right - left)
                    boxes.append(tuple(a + (b - a) * t for a, b in zip(box, next_box)))
                    regions.append(group if t <= 0.5 else next_group)
        return Track(known[0], boxes, regions)


def box_iou(a: Box, b: Box) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def _touch(a: Box, b: Box) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def group_regions(regions: list[Region]) -> list[list[Region]]:
    """Regions whose boxes overlap, directly or through others, as one group."""
    parent = list(range(len(regions)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            if _touch(regions[i].box, regions[j].box):
                parent[root(i)] = root(j)
    groups: dict[int, list[Region]] = {}
    for i, region in enumerate(regions):
        groups.setdefault(root(i), []).append(region)
    return list(groups.values())


def build_tracks(regions: RegionSet, *, max_gap: int = 3, link_iou: float = 0.1) -> list[Track]:
    """Every region track of ``regions``, ordered by start frame."""
    if not regions.frames:
        return []
    done: list[_Building] = []
    live: list[_Building] = []
    for frame in range(max(regions.frames) + 1):
        if frame in regions.cuts:
            done.extend(live)
            live = []
        done.extend(t for t in live if frame - t.last - 1 > max_gap)
        live = [t for t in live if frame - t.last - 1 <= max_gap]
        groups = group_regions(regions.frames.get(frame, []))
        if not groups:
            continue
        boxes = [union_box(group) for group in groups]
        candidates = sorted(
            (
                (box_iou(track.frames[track.last][0], box), ti, gi)
                for ti, track in enumerate(live)
                for gi, box in enumerate(boxes)
            ),
            reverse=True,
        )
        used_tracks: set[int] = set()
        used_groups: set[int] = set()
        for iou, ti, gi in candidates:
            if iou < link_iou:
                break
            if ti in used_tracks or gi in used_groups:
                continue
            live[ti].add(frame, boxes[gi], groups[gi])
            used_tracks.add(ti)
            used_groups.add(gi)
        for gi, (box, group) in enumerate(zip(boxes, groups)):
            if gi not in used_groups:
                track = _Building(frame)
                track.add(frame, box, group)
                live.append(track)
    done.extend(live)
    return sorted((t.finish() for t in done), key=lambda track: track.start)


def window_starts(length: int, window: int, hop: int, offset: int = 0) -> list[int]:
    """Starts of the whole ``window``-frame windows in a ``length``-frame track."""
    return list(range(offset, length - window + 1, hop))
