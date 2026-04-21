#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

PYTHON_BIN="${PYTHON_BIN:-$HOME/.conda/envs/mimicgen/bin/python}"
CKPT="${1:-$WORKSPACE_ROOT/mimicgen/datasets/core_training_results_quick/bc_rnn_low_dim_ds_square_D2_seed_101_quick_r3/20260410144913/models/model_epoch_20_demo_success_0.0.pth}"
TOTAL_STEPS="${2:-2000}"
CHECKPOINT_INTERVAL="${3:-1000}"
OUT_DIR="${4:-$PROJECT_ROOT/tmp_runs/dapg}"
ACTOR_BC_WEIGHT="${ACTOR_BC_WEIGHT:-0.5}"
ACTOR_BC_DECAY="${ACTOR_BC_DECAY:-0.99995}"

mkdir -p "$OUT_DIR"

"$PYTHON_BIN" "$PROJECT_ROOT/train_rl.py" \
    --method sac_dapg \
    --ckpt-path "$CKPT" \
    --total-steps "$TOTAL_STEPS" \
    --checkpoint-interval "$CHECKPOINT_INTERVAL" \
    --actor-bc-weight "$ACTOR_BC_WEIGHT" \
    --actor-bc-decay "$ACTOR_BC_DECAY" \
    --output-dir "$OUT_DIR" \
    "${@:5}"
