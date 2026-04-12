#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

CKPT="${1:-$WORKSPACE_ROOT/mimicgen/datasets/core_training_results_quick/bc_rnn_low_dim_ds_square_D2_seed_101_quick_r3/20260410144913/models/model_epoch_20_demo_success_0.0.pth}"
TOTAL_STEPS="${2:-2000}"
EVAL_INTERVAL="${3:-1000}"
EVAL_EPISODES="${4:-5}"
OUT_DIR="${5:-$PROJECT_ROOT/tmp_runs/dapg}"

mkdir -p "$OUT_DIR"

python "$PROJECT_ROOT/train_rl.py" \
    --method sac_dapg \
    --ckpt-path "$CKPT" \
    --total-steps "$TOTAL_STEPS" \
    --eval-interval "$EVAL_INTERVAL" \
    --eval-episodes "$EVAL_EPISODES" \
    --output-dir "$OUT_DIR"

