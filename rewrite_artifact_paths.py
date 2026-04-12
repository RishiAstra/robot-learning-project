from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch


OLD_ROOT = "/home/kirke/Documents/Projects/CS391R/CS395 L43DV"
NEW_ROOT = "/home/kirke/Documents/Projects/CS391R"


def rewrite_obj(obj: Any, old_root: str, new_root: str) -> Any:
    if isinstance(obj, str):
        return obj.replace(old_root, new_root)
    if isinstance(obj, dict):
        return {k: rewrite_obj(v, old_root, new_root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rewrite_obj(v, old_root, new_root) for v in obj]
    if isinstance(obj, tuple):
        return tuple(rewrite_obj(v, old_root, new_root) for v in obj)
    return obj


def rewrite_json(path: Path, old_root: str, new_root: str) -> None:
    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    data = json.loads(path.read_text())
    rewritten = rewrite_obj(data, old_root, new_root)
    path.write_text(json.dumps(rewritten, indent=4) + "\n")


def rewrite_checkpoint(path: Path, old_root: str, new_root: str) -> None:
    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    ckpt = torch.load(path, map_location="cpu")
    rewritten = rewrite_obj(ckpt, old_root, new_root)
    torch.save(rewritten, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", default=OLD_ROOT)
    parser.add_argument("--new-root", default=NEW_ROOT)
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()

    for raw_path in args.paths:
        path = Path(raw_path)
        if path.suffix == ".json":
            rewrite_json(path, args.old_root, args.new_root)
        elif path.suffix == ".pth":
            rewrite_checkpoint(path, args.old_root, args.new_root)
        else:
            raise ValueError(f"Unsupported path type: {path}")
        print(f"rewrote {path}")


if __name__ == "__main__":
    main()
