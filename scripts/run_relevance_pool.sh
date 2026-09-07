#!/usr/bin/env bash
# Build the ReMIX-B relevance pool (graded candidate qrels) for a run's test/val
# split. Reads the validated-instruction gate; writes chain_step_relevance_pools.jsonl.
#
# Point the io at the canonical instructions_axis_focused_5 folder (the stage
# defaults still assume the old `instructions/` + `validation/` layout).
#
# Usage:
#   RUN_ROOT=/path/to/run PROFILE=andrena scripts/run_relevance_pool.sh
#   MAX_STEPS=50 RUN_ROOT=... scripts/run_relevance_pool.sh   # smoke

set -euo pipefail

export PATH="/data/home/acw749/conda-envs/instruct_embed/bin:${PATH}"

# HF cache on scratch + offline (compute nodes have no internet).
export HF_HOME="${HF_HOME:-/gpfs/scratch/acw749/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/gpfs/scratch/acw749/hf_cache/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/gpfs/scratch/acw749/hf_cache/transformers}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

PROFILE="${PROFILE:-andrena}"
RUN_ROOT="${RUN_ROOT:-/gpfs/scratch/acw749/datasets/music4all_instruct/music4all_v1}"
RUN_ROOT="${RUN_ROOT%/}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RUN_ROOT%/*}}"
RUN_NAME="${RUN_NAME:-${RUN_ROOT##*/}}"
FOLDER="${FOLDER:-instructions_axis_focused_5}"

JUDGE_MODEL_ID="${JUDGE_MODEL_ID:-google/gemma-4-31B-it}"
BACKEND="${BACKEND:-vllm}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
# gemma-31B bf16 tp=2 on 2x40GB has little KV headroom; cap concurrent seqs so the
# vLLM sampler warmup does not OOM (default 256 is too many).
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
# The EmbeddingGemma text encoder follows stage.runtime.device. The vLLM judge
# manages its own GPUs regardless, and gemma-31B leaves no GPU room for a second
# model, so run the (tiny) text encoder on CPU to avoid an OOM collision.
DEVICE="${DEVICE:-cpu}"
MAX_STEPS="${MAX_STEPS:-0}"   # 0 = all; set small for a smoke run
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SPLITS="${SPLITS:-test}"                 # benchmark = test split; set "" for all
RUN_SOLVABILITY="${RUN_SOLVABILITY:-false}"   # naive baseline; a separate step does baselines
# Each shard writes its own output so parallel jobs never clobber one file.
OUTPUT_POOLS_JSONL="${OUTPUT_POOLS_JSONL:-chain_step_relevance_pools.shard${SHARD_INDEX}.jsonl}"

module load cuda/12.6.2-gcc-12.2.0
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export CUDA_PATH="${CUDA_HOME}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/data/home/acw749/conda-envs/instruct_embed/lib:${LD_LIBRARY_PATH:-}"
if [[ "${PROFILE}" == "sae" ]]; then
  export VLLM_USE_DEEP_GEMM=0
  export VLLM_MOE_USE_DEEP_GEMM=0
fi

cd /data/home/acw749/Jamendo-Instruct

FOLDER_DIR="${RUN_ROOT}/${FOLDER}"
CMD=(
  /data/home/acw749/conda-envs/instruct_embed/bin/python -m jamendo_instruct.run
  stage=relevance_pool
  "runtime.output_root=${OUTPUT_ROOT}"
  "runtime.run_name=${RUN_NAME}"
  # Redirect the three folder-specific io paths to the canonical instructions folder.
  "stage.io.input_prepared_jsonl=${FOLDER_DIR}/chain_step_instruction_inputs.jsonl"
  "stage.io.input_validation_jsonl=${FOLDER_DIR}/validation/validated_instructions.jsonl"
  "stage.io.output_dir=${FOLDER_DIR}/relevance_pool"
  "stage.io.output_pools_jsonl=${OUTPUT_POOLS_JSONL}"
  "stage.behavior.num_shards=${NUM_SHARDS}"
  "stage.behavior.shard_index=${SHARD_INDEX}"
  "stage.behavior.run_solvability_audit=${RUN_SOLVABILITY}"
  "stage.behavior.use_text_encoder_audit=${RUN_SOLVABILITY}"
  "stage.models.judge_model_id=${JUDGE_MODEL_ID}"
  "stage.runtime.backend=${BACKEND}"
  "stage.runtime.device=${DEVICE}"
  "stage.runtime.vllm_tensor_parallel_size=${TENSOR_PARALLEL_SIZE}"
  "stage.runtime.vllm_max_model_len=${MAX_MODEL_LEN}"
  "stage.runtime.vllm_max_num_seqs=${MAX_NUM_SEQS}"
  "stage.runtime.vllm_gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
)
if [[ "${MAX_STEPS}" != "0" ]]; then
  CMD+=("stage.behavior.max_steps=${MAX_STEPS}")
fi
# Hydra list override: [] for all splits, [test] / [test,validation] otherwise.
if [[ -n "${SPLITS}" ]]; then
  CMD+=("stage.behavior.pool_splits=[${SPLITS}]")
else
  CMD+=("stage.behavior.pool_splits=[]")
fi

exec env PYTHONPATH=src "${CMD[@]}" "$@"
