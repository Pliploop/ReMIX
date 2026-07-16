#!/usr/bin/env bash
# Run a text-only LLM over the same rubric/inputs the human raters see, writing
# llm_ratings.jsonl next to the run's human_ratings.jsonl.
#
# Usage:
#   RUN_ROOT=/path/to/run MODEL_ID=google/gemma-2-9b-it scripts/run_llm_validation_judge.sh
#   RUN_ROOT=/path/to/run MODEL_ID=Qwen/Qwen2.5-7B-Instruct scripts/run_llm_validation_judge.sh
#   # small smoke test:
#   RUN_ROOT=/path/to/run MODEL_ID=Qwen/Qwen2.5-7B-Instruct MAX_CHAINS=2 LIMIT=5 scripts/run_llm_validation_judge.sh

set -euo pipefail

export PATH="/data/home/acw749/conda-envs/instruct_embed/bin:${PATH}"

PROFILE="${PROFILE:-sae}"
RUN_ROOT="${RUN_ROOT:-/gpfs/scratch/acw749/datasets/music4all_instruct/music4all_v1}"
RUN_ROOT="${RUN_ROOT%/}"
INSTRUCTION_FOLDER="${INSTRUCTION_FOLDER:-instructions_axis_focused_5}"
INSTRUCTION_FIELD="${INSTRUCTION_FIELD:-history_unaware_instruction}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
BACKEND="${BACKEND:-auto}"
OUTPUT_NAME="${OUTPUT_NAME:-llm_ratings.jsonl}"

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
QUANTIZATION="${QUANTIZATION:-}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-768}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_CHAINS="${MAX_CHAINS:-0}"
LIMIT="${LIMIT:-0}"

# Mirror scripts/run_instruction_raw.sh exactly: the gcc-12 module provides the
# newer libstdc++ (CXXABI_1.3.15) that vLLM's import chain needs.
module load cuda/12.6.2-gcc-12.2.0
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export CUDA_PATH="${CUDA_HOME}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
# The cuda module is gcc-12 (CXXABI <= 1.3.13) and so is the system libstdc++,
# but the env's libicui18n.so.78 (pulled in by vLLM -> diskcache -> sqlite3)
# needs CXXABI_1.3.15, which only the env's own bundled libstdc++ provides.
# Put the env lib FIRST so that one wins.
export LD_LIBRARY_PATH="/data/home/acw749/conda-envs/instruct_embed/lib:${LD_LIBRARY_PATH:-}"
if [[ "${PROFILE}" == "sae" ]]; then
  export VLLM_USE_DEEP_GEMM=0
  export VLLM_MOE_USE_DEEP_GEMM=0
fi

cd /data/home/acw749/Jamendo-Instruct

CMD=(
  /data/home/acw749/conda-envs/instruct_embed/bin/python scripts/llm_validation_judge.py
  --run-root "${RUN_ROOT}"
  --instruction-folder "${INSTRUCTION_FOLDER}"
  --instruction-field "${INSTRUCTION_FIELD}"
  --model-id "${MODEL_ID}"
  --backend "${BACKEND}"
  --output-name "${OUTPUT_NAME}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
  --batch-size "${BATCH_SIZE}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
  --max-chains "${MAX_CHAINS}"
  --limit "${LIMIT}"
)

if [[ -n "${KV_CACHE_DTYPE}" ]]; then
  CMD+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
fi
if [[ -n "${QUANTIZATION}" ]]; then
  CMD+=(--quantization "${QUANTIZATION}")
fi

# Optional: restrict to the frozen human-validation slice so the LLM judges
# exactly the items assigned to raters.
if [[ -n "${ASSIGNMENT_JSONL:-}" ]]; then
  CMD+=(--assignment-jsonl "${ASSIGNMENT_JSONL}")
fi
if [[ -n "${FROZEN_SIDECAR_JSON:-}" ]]; then
  CMD+=(--frozen-sidecar-json "${FROZEN_SIDECAR_JSON}")
fi

exec env PYTHONPATH=src "${CMD[@]}" "$@"
