#!/usr/bin/env bash
set -euo pipefail

# BASE_CHECKPOINT is required. Additional train_ft.py options can be appended
# directly to this command and override the defaults below.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

PYTHON="${PYTHON:-python3}"
OPTIMIZER="${OPTIMIZER:-ivon}"
TRM_REPO="${TRM_REPO:-${PROJECT_ROOT}/third_party/TinyRecursiveModels}"
DATASET="${DATASET:-${PROJECT_ROOT}/data/sudoku-extreme-1k-aug-1000}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/${OPTIMIZER}_ft}"
RUN_NAME="${RUN_NAME:-${OPTIMIZER}_ft_reference_seed0}"
SCHEDULE_JSON="${SCHEDULE_JSON:-${PROJECT_ROOT}/ptrm/configs/${OPTIMIZER}_ft_schedule.json}"

if [[ -z "${BASE_CHECKPOINT}" ]]; then
  echo "BASE_CHECKPOINT must point to the pretrained PTRM/TRM checkpoint to fine-tune." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export DISABLE_COMPILE="${DISABLE_COMPILE:-1}"

ARGS=(
  -m ptrm.train_ft
  --optimizer "${OPTIMIZER}"
  --trm-repo "${TRM_REPO}"
  --dataset "${DATASET}"
  --base-checkpoint "${BASE_CHECKPOINT}"
  --base-state-key "${BASE_STATE_KEY:-model_state_dict}"
  --output-root "${OUTPUT_ROOT}"
  --run-name "${RUN_NAME}"
  --schedule-json "${SCHEDULE_JSON}"
  --device "${DEVICE:-cuda:0}"
  --seed "${SEED:-0}"
  --train-steps "${TRAIN_STEPS:-200}"
  --global-batch-size "${GLOBAL_BATCH_SIZE:-128}"
  --train-group-count-for-epochs "${TRAIN_GROUP_COUNT_FOR_EPOCHS:-1000}"
  --status-interval "${STATUS_INTERVAL:-50}"
  --history-interval "${HISTORY_INTERVAL:-10}"
  --checkpoint-interval "${CHECKPOINT_INTERVAL:-100}"
  --disable-compile
)

[[ -z "${STOP_STEP:-}" ]] || ARGS+=(--stop-step "${STOP_STEP}")
[[ -z "${RESUME_CHECKPOINT:-}" ]] || ARGS+=(--resume-checkpoint "${RESUME_CHECKPOINT}")

if [[ "${WANDB_ENABLED:-0}" == "1" ]]; then
  ARGS+=(--wandb --wandb-project "${WANDB_PROJECT:-ptrm-${OPTIMIZER}-fine-tuning}" --wandb-mode "${WANDB_MODE:-online}")
  [[ -z "${WANDB_ENTITY:-}" ]] || ARGS+=(--wandb-entity "${WANDB_ENTITY}")
  [[ -z "${WANDB_GROUP:-}" ]] || ARGS+=(--wandb-group "${WANDB_GROUP}")
fi

cd "${PROJECT_ROOT}"
exec "${PYTHON}" "${ARGS[@]}" "$@"
