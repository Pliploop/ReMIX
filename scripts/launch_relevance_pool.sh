#!/usr/bin/env bash
# sbatch launcher for the ReMIX-B relevance pool. Mirrors launch_andrena_instruction.sh.
# gemma-4-31B judge on 2x A100 (tp=2), so default profile is andrena.
#
# Examples:
#   # smoke (50 steps):
#   RUN_ROOT=/gpfs/scratch/acw749/datasets/mtg_jamendo_instruct/v1 MAX_STEPS=50 \
#     scripts/launch_relevance_pool.sh
#   # full:
#   RUN_ROOT=/gpfs/scratch/acw749/datasets/music4all_instruct/music4all_v1 \
#     scripts/launch_relevance_pool.sh

set -euo pipefail

PROFILE="${PROFILE:-andrena}"
RUN_ROOT="${RUN_ROOT:-/gpfs/scratch/acw749/datasets/music4all_instruct/music4all_v1}"
RUN_ROOT="${RUN_ROOT%/}"
RUN_NAME="${RUN_NAME:-${RUN_ROOT##*/}}"
JOB_NAME="${JOB_NAME:-relpool_${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/instructions_axis_focused_5/relevance_pool/logs}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
mkdir -p "${LOG_DIR}"

case "${PROFILE}" in
  andrena) SBATCH_RES=(-p andrena -A pilot_andrena --gres=gpu:nvidia_a100-pcie-40gb:2); : "${TENSOR_PARALLEL_SIZE:=2}" ;;
  sae)     SBATCH_RES=(-p sae -A pilot_sae_gpu --gres=gpu:1 --constraint="hopper|ampere"); : "${TENSOR_PARALLEL_SIZE:=1}" ;;
  *) echo "Unknown PROFILE=${PROFILE}; expected andrena or sae." >&2; exit 2 ;;
esac

EXPORTS="ALL"
for var in PROFILE RUN_ROOT RUN_NAME FOLDER JUDGE_MODEL_ID BACKEND TENSOR_PARALLEL_SIZE \
           MAX_MODEL_LEN GPU_MEMORY_UTILIZATION MAX_STEPS; do
  if [[ -n "${!var:-}" ]]; then EXPORTS="${EXPORTS},${var}=${!var}"; fi
done

sbatch \
  -J "${JOB_NAME}" \
  "${SBATCH_RES[@]}" \
  -n 1 --cpus-per-gpu=12 --mem-per-cpu=7500M \
  -t "${TIME_LIMIT}" \
  -o "${LOG_DIR}/slurm-%x_%j.out" \
  -e "${LOG_DIR}/slurm-%x_%j.err" \
  --export="${EXPORTS}" \
  scripts/run_relevance_pool.sh
