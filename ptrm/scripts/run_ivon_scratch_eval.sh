#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper for the generic IVON evaluator.
# Common overrides:
#   RUN_NAME=my_run METHOD=ivon_compare_selection K=20 ./run_ivon_scratch_eval.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/ivon_scratch}"
RUN_NAME="${RUN_NAME:-ivon_scratch_baseline_seed0}"

export PROJECT_ROOT
export IVON_CHECKPOINT="${IVON_CHECKPOINT:-${OUTPUT_ROOT}/${RUN_NAME}/checkpoints/checkpoint.pt}"
exec "${SCRIPT_DIR}/run_ivon_eval.sh" "$@"
