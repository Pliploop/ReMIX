#!/usr/bin/env bash

set -euo pipefail

RUN_ROOT="${RUN_ROOT:-/gpfs/scratch/acw749/datasets/music4all_instruct/music4all_v1}"
OUTPUT_NAME="${OUTPUT_NAME:-instructions_axis_focused_5}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/${OUTPUT_NAME}}"
JOB_NAME="${JOB_NAME:-instr_qwen36_axis_27b_sae1}"

mkdir -p "${OUTPUT_DIR}/logs"

sbatch \
  -J "${JOB_NAME}" \
  -p sae \
  -A pilot_sae_gpu \
  -n 1 \
  --cpus-per-gpu=12 \
  --gres=gpu:1 \
  --constraint="hopper|ampere" \
  --mem-per-cpu=7500M \
  -t 24:00:00 \
  -o "${OUTPUT_DIR}/logs/slurm-%x_%j.out" \
  -e "${OUTPUT_DIR}/logs/slurm-%x_%j.err" \
  --export=ALL,PROFILE=sae,RUN_ROOT="${RUN_ROOT}",OUTPUT_NAME="${OUTPUT_NAME}",OUTPUT_DIR="${OUTPUT_DIR}" \
  scripts/run_instruction_raw.sh
