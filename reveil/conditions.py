"""Share one caption embedding across a preprocessed ltx-trainer dataset.

Every training pair uses the same caption, so the per-sample ``conditions/*.pt`` files that
``process_dataset.py`` writes are copies of one embedding (MBs each). ``share`` keeps one of
them as ``shared_condition.pt`` beside ``conditions/`` and turns every per-sample file into
a symlink to it. ``restore`` recreates those links for every target latent, for example after
``latents/``, ``reference_latents/`` and ``shared_condition.pt`` are copied back from Google
Drive, which keeps no symlinks.

    python -m reveil.conditions share  DATA/.precomputed --dataset DATA/dataset.jsonl
    python -m reveil.conditions restore DATA/.precomputed
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

SHARED_NAME = "shared_condition.pt"


def _captions(dataset: Path) -> set[str]:
    if dataset.suffix == ".jsonl":
        rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        rows = json.loads(dataset.read_text(encoding="utf-8"))
    return {row["caption"] for row in rows}


def _link(path: Path, target: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.exists():
        path.unlink()
    path.symlink_to(os.path.relpath(target, path.parent))


def share(root: str | Path, dataset: str | Path) -> int:
    """Replace the per-sample condition files under ``root`` with links to one shared file;
    refuses unless every row of ``dataset`` has the same caption. Returns the files linked."""
    root, dataset = Path(root), Path(dataset)
    captions = _captions(dataset)
    if len(captions) != 1:
        raise ValueError(f"{dataset} has {len(captions)} different captions; sharing needs exactly one")
    shared = root / SHARED_NAME
    files = sorted(p for p in (root / "conditions").rglob("*.pt") if not p.is_symlink())
    if not files:
        return 0
    if not shared.exists():
        shutil.copy2(files[0], shared)
    for path in files:
        _link(path, shared)
    return len(files)


def restore(root: str | Path) -> int:
    """Link ``conditions/<sample>.pt`` to the shared file for every ``latents/<sample>.pt``;
    returns the links made."""
    root = Path(root)
    shared = root / SHARED_NAME
    if not shared.is_file():
        raise FileNotFoundError(f"{shared} is missing; run 'share' on the full dataset first")
    latents = root / "latents"
    count = 0
    for latent in latents.rglob("*.pt"):
        _link(root / "conditions" / latent.relative_to(latents), shared)
        count += 1
    return count


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="reveil-conditions", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    share_parser = commands.add_parser("share", help="link every per-sample caption embedding to one file")
    share_parser.add_argument("root", type=Path, help="the .precomputed directory")
    share_parser.add_argument("--dataset", type=Path, help="metadata file (default: ../dataset.jsonl)")
    restore_parser = commands.add_parser("restore", help="recreate the links for every target latent")
    restore_parser.add_argument("root", type=Path, help="the .precomputed directory")
    args = parser.parse_args(argv)
    if args.command == "share":
        dataset = args.dataset or args.root.resolve().parent / "dataset.jsonl"
        print(f"linked {share(args.root, dataset)} condition files to {args.root / SHARED_NAME}")
    else:
        print(f"restored {restore(args.root)} condition links")


if __name__ == "__main__":
    main()
