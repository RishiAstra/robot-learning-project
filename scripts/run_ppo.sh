#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

PYTHON_BIN="${PYTHON_BIN:-$HOME/.conda/envs/mimicgen/bin/python}"
CKPT="${1:-$WORKSPACE_ROOT/mimicgen/datasets/core_training_results_quick/bc_rnn_low_dim_ds_square_D2_seed_101_quick_r3/20260410144913/models/model_epoch_20_demo_success_0.0.pth}"
TOTAL_STEPS="${2:-2000}"
EVAL_INTERVAL="${3:-1000}"
EVAL_EPISODES="${4:-5}"
OUT_DIR="${5:-$PROJECT_ROOT/tmp_runs/ppo}"

mkdir -p "$OUT_DIR"

"$PYTHON_BIN" "$PROJECT_ROOT/train_rl.py" \
    --method ppo \
    --ckpt-path "$CKPT" \
    --total-steps "$TOTAL_STEPS" \
    --eval-interval "$EVAL_INTERVAL" \
    --eval-episodes "$EVAL_EPISODES" \
    --output-dir "$OUT_DIR" \
    "${@:6}"
