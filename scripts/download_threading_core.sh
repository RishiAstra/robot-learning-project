#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

PYTHON_BIN="${PYTHON_BIN:-$HOME/.conda/envs/mimicgen/bin/python}"
DOWNLOAD_DIR="${1:-$WORKSPACE_ROOT/mimicgen/datasets/core}"

NUMBA_DISABLE_JIT=1 PYTHONPATH="$WORKSPACE_ROOT/mimicgen:$WORKSPACE_ROOT/robomimic:$WORKSPACE_ROOT/robosuite-task-zoo${PYTHONPATH:+:$PYTHONPATH}" \
"$PYTHON_BIN" - "$DOWNLOAD_DIR" <<'PY'
import shutil
import sys
from pathlib import Path

import h5py
from huggingface_hub import hf_hub_download

repo_id = "amandlek/mimicgen_datasets"
download_dir = Path(sys.argv[1]).expanduser().resolve()
download_dir.mkdir(parents=True, exist_ok=True)

for task in ("threading_d0", "threading_d1", "threading_d2"):
    target = download_dir / f"{task}.hdf5"
    force_download = False
    if target.exists():
        try:
            with h5py.File(target, "r"):
                print(f"exists {target}")
                continue
        except Exception as exc:
            print(f"re-downloading corrupt file {target}: {exc}")
            target.unlink()
            force_download = True

    path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=f"core/{task}.hdf5",
        force_download=force_download,
    )
    shutil.copy2(path, target)
    with h5py.File(target, "r"):
        pass
    print(target)
PY
