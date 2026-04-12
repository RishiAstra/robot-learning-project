#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

CKPT="${1:-$WORKSPACE_ROOT/mimicgen/datasets/core_training_results_quick/bc_rnn_low_dim_ds_square_D2_seed_101_quick_r3/20260410144913/models/model_epoch_20_demo_success_0.0.pth}"
N_ROLLOUTS="${2:-20}"
HORIZON="${3:-400}"
ENV_NAME="${4:-}"
OUT_DIR="${5:-$PROJECT_ROOT/tmp_runs/bc_eval}"

mkdir -p "$OUT_DIR"

ARGS=(
    --agent "$CKPT"
    --n-rollouts "$N_ROLLOUTS"
    --horizon "$HORIZON"
    --output-json "$OUT_DIR/bc_eval.json"
)

if [[ -n "$ENV_NAME" ]]; then
    ARGS+=(--env "$ENV_NAME")
fi

python "$PROJECT_ROOT/evaluate_checkpoints.py" "${ARGS[@]}"

