#!/usr/bin/env bash
set -euo pipefail

# Common overrides:
#   RUN_NAME=my_run METHOD=ivon_compare_selection K=20 ./run_ivon_scratch_eval.sh
# Additional eval.py options can be appended directly to this command.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

PYTHON="${PYTHON:-python3}"
TRM_REPO="${TRM_REPO:-${PROJECT_ROOT}/third_party/TinyRecursiveModels}"
DATASET="${DATASET:-${PROJECT_ROOT}/data/sudoku-extreme-1k-aug-1000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/ivon_scratch}"
RUN_NAME="${RUN_NAME:-ivon_scratch_baseline_seed0}"
CHECKPOINT="${IVON_CHECKPOINT:-${OUTPUT_ROOT}/${RUN_NAME}/checkpoints/checkpoint.pt}"

METHOD="${METHOD:-ivon_parameter_sampling}"
K="${K:-10}"
DEPTH="${DEPTH:-16}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"

RESULT_DIR="${EVAL_OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}/eval_${METHOD}_k${K}_d${DEPTH}}"
if (( NUM_SHARDS == 1 )); then
  OUTPUT_JSON="${OUTPUT_JSON:-${RESULT_DIR}/eval.json}"
  PROGRESS_JSON="${PROGRESS_JSON:-${RESULT_DIR}/progress.json}"
else
  OUTPUT_JSON="${OUTPUT_JSON:-${RESULT_DIR}/shard_${SHARD_INDEX}.json}"
  PROGRESS_JSON="${PROGRESS_JSON:-${RESULT_DIR}/progress_${SHARD_INDEX}.json}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export DISABLE_COMPILE="${DISABLE_COMPILE:-1}"

mkdir -p "${RESULT_DIR}"
cd "${PROJECT_ROOT}"
exec "${PYTHON}" -m ptrm.eval \
  --trm-repo "${TRM_REPO}" \
  --dataset "${DATASET}" \
  --ivon-checkpoint "${CHECKPOINT}" \
  --model-state-key "${MODEL_STATE_KEY:-model_state_dict}" \
  --method "${METHOD}" \
  --output-json "${OUTPUT_JSON}" \
  --progress-json "${PROGRESS_JSON}" \
  --device "${DEVICE:-cuda:0}" \
  --seed "${SEED:-0}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-128}" \
  --k "${K}" \
  --depth "${DEPTH}" \
  --ivon-posterior-scale "${IVON_POSTERIOR_SCALE:-1.0}" \
  --num-shards "${NUM_SHARDS}" \
  --shard-index "${SHARD_INDEX}" \
  --progress \
  "$@"
