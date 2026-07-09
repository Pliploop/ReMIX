#!/usr/bin/env bash

set -euo pipefail

export PATH="/data/home/acw749/conda-envs/instruct_embed/bin:${PATH}"

PROFILE="${PROFILE:-andrena}"
RUN_ROOT="${RUN_ROOT:-/gpfs/scratch/acw749/datasets/music4all_instruct/music4all_v1}"
RUN_ROOT="${RUN_ROOT%/}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RUN_ROOT%/*}}"
RUN_NAME="${RUN_NAME:-${RUN_ROOT##*/}}"
OUTPUT_NAME="${OUTPUT_NAME:-instructions_axis_focused_5}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/${OUTPUT_NAME}}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-27B-FP8}"
MODEL_PARAMS_B="${MODEL_PARAMS_B:-27}"

case "${PROFILE}" in
  andrena)
    TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
    ENFORCE_EAGER="${ENFORCE_EAGER:-false}"
    GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-16}"
    KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
    ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-true}"
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-24576}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
    ;;
  sae)
    TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
    ENFORCE_EAGER="${ENFORCE_EAGER:-false}"
    GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-32}"
    KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
    ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-true}"
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-24576}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
    ;;
  *)
    echo "Unknown PROFILE=${PROFILE}; expected andrena or sae." >&2
    exit 2
    ;;
esac

mkdir -p "${OUTPUT_DIR}/step_json" "${OUTPUT_DIR}/claims"

module load cuda/12.6.2-gcc-12.2.0
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export CUDA_PATH="${CUDA_HOME}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
# The cuda module's gcc-12 (and the system) libstdc++ lack CXXABI_1.3.15, which
# vLLM's import chain (diskcache -> sqlite3 -> libicui18n.so.78) requires; only
# the env's bundled libstdc++ has it, so put it first.
export LD_LIBRARY_PATH="/data/home/acw749/conda-envs/instruct_embed/lib:${LD_LIBRARY_PATH:-}"
if [[ "${PROFILE}" == "sae" ]]; then
  export VLLM_USE_DEEP_GEMM=0
  export VLLM_MOE_USE_DEEP_GEMM=0
fi

cd /data/home/acw749/Jamendo-Instruct

CMD=(
  /data/home/acw749/conda-envs/instruct_embed/bin/python -m jamendo_instruct.run
  stage=instructions
  "runtime.output_root=${OUTPUT_ROOT}"
  "runtime.run_name=${RUN_NAME}"
  "stage.io.output_dir=${OUTPUT_DIR}"
  "stage.io.output_step_json_dir=${OUTPUT_DIR}/step_json"
  "stage.models.model_id=${MODEL_ID}"
  "stage.models.params_b=${MODEL_PARAMS_B}"
  stage.runtime.backend=vllm
  stage.runtime.llm_model_family=qwen3_6
  stage.runtime.vllm_dtype=auto
  stage.runtime.vllm_quantization=fp8
  "stage.runtime.vllm_tensor_parallel_size=${TENSOR_PARALLEL_SIZE}"
  "stage.runtime.vllm_gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
  stage.runtime.vllm_max_model_len=12288
  "stage.runtime.vllm_enforce_eager=${ENFORCE_EAGER}"
  stage.runtime.vllm_gdn_prefill_backend=triton
  "stage.runtime.vllm_max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"
  stage.lyrics.max_chars_per_view=300
  stage.lyrics.max_chars_for_diff=500
  "stage.runtime.generation_batch_size=${GENERATION_BATCH_SIZE}"
  stage.generation.temperature=0.45
  stage.generation.top_p=0.9
  stage.generation.max_new_tokens=768
  stage.behavior.strict_json_retry_attempts=4
  "stage.behavior.claim_dir=${OUTPUT_DIR}/claims"
  stage.behavior.write_step_json=true
  stage.behavior.overwrite_existing=false
  stage.axis_guidance.enabled=true
  "stage.axis_guidance.state_path=${OUTPUT_DIR}/axis_guidance_state.json"
)

if [[ -n "${KV_CACHE_DTYPE}" ]]; then
  CMD+=("stage.runtime.vllm_kv_cache_dtype=${KV_CACHE_DTYPE}")
fi
if [[ -n "${ENABLE_PREFIX_CACHING}" ]]; then
  CMD+=("stage.runtime.vllm_enable_prefix_caching=${ENABLE_PREFIX_CACHING}")
fi
if [[ -n "${MAX_NUM_SEQS}" ]]; then
  CMD+=("stage.runtime.vllm_max_num_seqs=${MAX_NUM_SEQS}")
fi

exec env PYTHONPATH=src "${CMD[@]}"
