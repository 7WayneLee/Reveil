import json

import pytest

from reveil.conditions import SHARED_NAME, main, restore, share


def make_precomputed(tmp_path, captions):
    data = tmp_path / "data"
    root = data / ".precomputed"
    for name in ("a", "b"):
        for sub in ("latents", "reference_latents", "conditions"):
            path = root / sub / "clean" / f"{name}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{sub}-{name}".encode())
    rows = [{"video": f"clean/{n}.mp4", "reference_video": f"mosaic/{n}.mp4", "caption": c} for n, c in zip("ab", captions)]
    (data / "dataset.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return root


def test_share_links_every_condition_to_one_file_and_restore_rebuilds(tmp_path):
    root = make_precomputed(tmp_path, ["same", "same"])
    main(["share", str(root)])
    shared = root / SHARED_NAME
    assert shared.read_bytes() == b"conditions-a"
    for name in ("a", "b"):
        link = root / "conditions" / "clean" / f"{name}.pt"
        assert link.is_symlink() and link.resolve() == shared.resolve()
    for link in (root / "conditions").rglob("*.pt"):
        link.unlink()
    assert restore(root) == 2
    assert (root / "conditions" / "clean" / "b.pt").read_bytes() == b"conditions-a"


def test_share_refuses_mixed_captions(tmp_path):
    root = make_precomputed(tmp_path, ["one", "two"])
    with pytest.raises(ValueError):
        share(root, root.parent / "dataset.jsonl")
