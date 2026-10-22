"""Discover retail/leak pairs by title directory, including Drive shortcuts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from reveil.make_pairs import VIDEO_SUFFIXES, find_videos


@dataclass(frozen=True)
class RealPair:
    title: str
    retail: Path
    leak: Path


def _upload_digest(path: Path) -> bytes:
    """Hash both ends, including their overlap for uploads smaller than 8 MiB."""
    block = 4 * 1024 * 1024
    digest = hashlib.sha256()
    with path.open("rb") as source:
        digest.update(source.read(block))
        source.seek(max(0, path.stat().st_size - block))
        digest.update(source.read(block))
    return digest.digest()


def find_real_pairs(censored_dir: Path, uncensored_dir: Path) -> tuple[list[RealPair], list[str]]:
    """Exactly one video on each side of each direct retail title directory."""
    censored_dir, uncensored_dir = Path(censored_dir), Path(uncensored_dir)
    pairs, warnings = [], []
    if not censored_dir.is_dir():
        return [], [f"missing censored directory: {censored_dir}"]
    for title_dir in sorted(censored_dir.iterdir()):
        if not title_dir.is_dir():
            continue
        videos, problems = [], []
        for label, directory in (("retail", title_dir), ("leak", uncensored_dir / title_dir.name)):
            if not directory.is_dir():
                problems.append(f"missing {label} directory {directory}")
                videos.append([])
                continue
            found = [p for p in find_videos([directory]) if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES]
            videos.append(found)
            if not found:
                problems.append(f"missing {label} video in {directory}")
            elif len(found) != 1:
                problems.append(f"ambiguous {label}: {len(found)} videos in {directory}")
        if problems:
            warnings.append(f"{title_dir.name}: " + "; ".join(problems))
            continue
        retail, leak = videos[0][0], videos[1][0]
        try:
            identical = retail.stat().st_size == leak.stat().st_size and _upload_digest(retail) == _upload_digest(leak)
        except OSError as error:
            warnings.append(f"{title_dir.name}: cannot read upload: {error}")
            continue
        if identical:
            warnings.append(f"{title_dir.name}: retail and leak are the same upload (size and SHA-256 of both ends)")
            continue
        pairs.append(RealPair(title_dir.name, retail, leak))
    return pairs, warnings
