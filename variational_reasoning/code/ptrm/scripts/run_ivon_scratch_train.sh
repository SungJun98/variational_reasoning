#!/usr/bin/env bash
set -euo pipefail

# Common overrides:
#   RUN_NAME=my_run CUDA_VISIBLE_DEVICES=1 ./run_ivon_scratch_train.sh
# Additional train_scratch.py options can be appended directly to this command.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"

PYTHON="${PYTHON:-python3}"
TRM_REPO="${TRM_REPO:-${PROJECT_ROOT}/third_party/TinyRecursiveModels}"
DATASET="${DATASET:-${PROJECT_ROOT}/data/sudoku-extreme-1k-aug-1000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/ivon_scratch}"
RUN_NAME="${RUN_NAME:-ivon_scratch_baseline_seed0}"
SCHEDULE_JSON="${SCHEDULE_JSON:-${PROJECT_ROOT}/variational_reasoning/code/ptrm/configs/ivon_scratch_schedule.json}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export DISABLE_COMPILE="${DISABLE_COMPILE:-1}"

ARGS=(
  -m variational_reasoning.code.ptrm.train_scratch
  --trm-repo "${TRM_REPO}"
  --dataset "${DATASET}"
  --output-root "${OUTPUT_ROOT}"
  --run-name "${RUN_NAME}"
  --schedule-json "${SCHEDULE_JSON}"
  --device "${DEVICE:-cuda:0}"
  --seed "${SEED:-0}"
  --train-steps "${TRAIN_STEPS:-50000}"
  --global-batch-size "${GLOBAL_BATCH_SIZE:-128}"
  --disable-compile
  --no-ivon-rescale-lr
)

[[ -z "${STOP_STEP:-}" ]] || ARGS+=(--stop-step "${STOP_STEP}")
[[ -z "${RESUME_CHECKPOINT:-}" ]] || ARGS+=(--resume-checkpoint "${RESUME_CHECKPOINT}")

if [[ "${WANDB_ENABLED:-0}" == "1" ]]; then
  ARGS+=(--wandb --wandb-project "${WANDB_PROJECT:-ptrm-ivon-from-scratch}" --wandb-mode "${WANDB_MODE:-online}")
  [[ -z "${WANDB_ENTITY:-}" ]] || ARGS+=(--wandb-entity "${WANDB_ENTITY}")
  [[ -z "${WANDB_GROUP:-}" ]] || ARGS+=(--wandb-group "${WANDB_GROUP}")
fi

cd "${PROJECT_ROOT}"
exec "${PYTHON}" "${ARGS[@]}" "$@"
