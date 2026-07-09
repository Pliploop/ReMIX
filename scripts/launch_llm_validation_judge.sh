#!/usr/bin/env bash
# sbatch launcher for the text-only LLM validation judge.
#
# Mirrors launch_{andrena,sae}_instruction.sh: it submits the worker
# scripts/run_llm_validation_judge.sh with the right partition/account/GPU flags
# and forwards config through --export.
#
# Examples:
#   # judge the frozen human-validation slice with Qwen on andrena (2x A100):
#   PROFILE=andrena \
#   FROZEN_SIDECAR_JSON=/gpfs/scratch/acw749/datasets/music4all_instruct/music4all_v1/validation/assignment_axis_focused_5_v1.sidecar.json \
#   MODEL_ID=Qwen/Qwen2.5-7B-Instruct \
#   scripts/launch_llm_validation_judge.sh
#
#   # judge a whole run with Gemma on sae (1 GPU):
#   PROFILE=sae RUN_ROOT=/gpfs/scratch/acw749/datasets/music4all_instruct/music4all_v1 \
#   MODEL_ID=google/gemma-2-9b-it scripts/launch_llm_validation_judge.sh
#
#   # quick smoke test (5 items):
#   PROFILE=sae FROZEN_SIDECAR_JSON=.../assignment_axis_focused_5_v1.sidecar.json \
#   MODEL_ID=Qwen/Qwen2.5-7B-Instruct MAX_CHAINS=2 LIMIT=5 \
#   scripts/launch_llm_validation_judge.sh

set -euo pipefail

PROFILE="${PROFILE:-andrena}"
RUN_ROOT="${RUN_ROOT:-/gpfs/scratch/acw749/datasets/music4all_instruct/music4all_v1}"
RUN_ROOT="${RUN_ROOT%/}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_TAG="$(basename "${MODEL_ID}" | tr '/:' '__')"
JOB_NAME="${JOB_NAME:-llm_valid_${MODEL_TAG}}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/validation/logs}"

mkdir -p "${LOG_DIR}"

case "${PROFILE}" in
  andrena)
    SBATCH_RES=(-p andrena -A pilot_andrena --gres=gpu:nvidia_a100-pcie-40gb:2)
    : "${TENSOR_PARALLEL_SIZE:=2}"
    ;;
  sae)
    SBATCH_RES=(-p sae -A pilot_sae_gpu --gres=gpu:1 --constraint="hopper|ampere")
    : "${TENSOR_PARALLEL_SIZE:=1}"
    ;;
  *)
    echo "Unknown PROFILE=${PROFILE}; expected andrena or sae." >&2
    exit 2
    ;;
esac

# Forward everything the worker reads. Unset optional vars become empty and the
# worker falls back to its own defaults.
EXPORTS="ALL"
for var in RUN_ROOT MODEL_ID BACKEND INSTRUCTION_FOLDER INSTRUCTION_FIELD \
           TENSOR_PARALLEL_SIZE GPU_MEMORY_UTILIZATION MAX_MODEL_LEN BATCH_SIZE \
           MAX_NEW_TOKENS TEMPERATURE MAX_CHAINS LIMIT ASSIGNMENT_JSONL FROZEN_SIDECAR_JSON; do
  if [[ -n "${!var:-}" ]]; then
    EXPORTS="${EXPORTS},${var}=${!var}"
  fi
done

sbatch \
  -J "${JOB_NAME}" \
  "${SBATCH_RES[@]}" \
  -n 1 \
  --cpus-per-gpu=12 \
  --mem-per-cpu=7500M \
  -t 12:00:00 \
  -o "${LOG_DIR}/slurm-%x_%j.out" \
  -e "${LOG_DIR}/slurm-%x_%j.err" \
  --export="${EXPORTS}" \
  scripts/run_llm_validation_judge.sh
