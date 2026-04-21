#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

VARIANT="${1:-D2}"
EPOCHS="${2:-2000}"
OUT_DIR="${3:-$PROJECT_ROOT/tmp_runs/threading_bc_${VARIANT}_e${EPOCHS}}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/.conda/envs/mimicgen/bin/python}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUT_DIR"

case "$VARIANT" in
    D0|D1|D2) ;;
    *)
        echo "unsupported threading variant: $VARIANT" >&2
        exit 1
        ;;
esac

DATASET="$WORKSPACE_ROOT/mimicgen/datasets/core/threading_${VARIANT,,}.hdf5"
CONFIG_SRC="$WORKSPACE_ROOT/mimicgen/datasets/core_train_configs/bc_rnn_low_dim_ds_threading_${VARIANT}_seed_101.json"
CONFIG_TMP="$OUT_DIR/bc_rnn_low_dim_ds_threading_${VARIANT}_seed_101.json"

if [[ ! -f "$DATASET" ]]; then
    echo "missing dataset: $DATASET" >&2
    exit 1
fi

if [[ ! -f "$CONFIG_SRC" ]]; then
    echo "missing config: $CONFIG_SRC" >&2
    exit 1
fi

NUMBA_DISABLE_JIT=1 PYTHONPATH="$WORKSPACE_ROOT/mimicgen:$WORKSPACE_ROOT/robomimic:$WORKSPACE_ROOT/robosuite-task-zoo${PYTHONPATH:+:$PYTHONPATH}" \
"$PYTHON_BIN" - "$CONFIG_SRC" "$CONFIG_TMP" "$DATASET" "$OUT_DIR/results" "$EPOCHS" "$VARIANT" "$RUN_TAG" <<'PY'
import json
import sys
from pathlib import Path

config_src = Path(sys.argv[1]).expanduser().resolve()
config_tmp = Path(sys.argv[2]).expanduser().resolve()
dataset = Path(sys.argv[3]).expanduser().resolve()
output_dir = Path(sys.argv[4]).expanduser().resolve()
epochs = int(sys.argv[5])
variant = sys.argv[6]
run_tag = sys.argv[7]

config = json.loads(config_src.read_text())
config["train"]["data"] = [{"path": str(dataset)}]
config["train"]["output_dir"] = str(output_dir)
config["train"]["num_epochs"] = epochs
config["train"]["num_data_workers"] = 0
config["experiment"]["name"] = f"bc_rnn_low_dim_ds_threading_{variant}_seed_101_core_{run_tag}"

config_tmp.write_text(json.dumps(config, indent=4) + "\n")
print(config_tmp)
PY

NUMBA_DISABLE_JIT=1 PYTHONPATH="$WORKSPACE_ROOT/mimicgen:$WORKSPACE_ROOT/robomimic:$WORKSPACE_ROOT/robosuite-task-zoo${PYTHONPATH:+:$PYTHONPATH}" \
"$PYTHON_BIN" "$WORKSPACE_ROOT/robomimic/robomimic/scripts/train.py" --config "$CONFIG_TMP"
