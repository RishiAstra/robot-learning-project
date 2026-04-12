#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"

export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export PYTHONPATH="${PROJECT_ROOT}:${WORKSPACE_ROOT}/robomimic:${WORKSPACE_ROOT}/mimicgen:${WORKSPACE_ROOT}/robosuite-task-zoo${PYTHONPATH:+:${PYTHONPATH}}"

