#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

ENV_NAME="${1:-Lift}"
TOTAL_STEPS="${2:-5000}"
EVAL_INTERVAL="${3:-1000}"
EVAL_EPISODES="${4:-5}"
OUT_DIR="${5:-$PROJECT_ROOT/tmp_runs/robosuite_smoke}"

mkdir -p "$OUT_DIR"

python "$PROJECT_ROOT/validate_sac_robosuite.py" \
    --env "$ENV_NAME" \
    --total-steps "$TOTAL_STEPS" \
    --eval-interval "$EVAL_INTERVAL" \
    --eval-episodes "$EVAL_EPISODES" \
    --output-json "$OUT_DIR/${ENV_NAME}_smoke.json"
