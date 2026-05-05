#!/usr/bin/env bash
#SBATCH -J music4all_caption
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -t 48:00:00
#SBATCH -o /gpfs/scratch/acw749/datasets/music4all_instruct/logs/slurm-%x_%j.out
#SBATCH -e /gpfs/scratch/acw749/datasets/music4all_instruct/logs/slurm-%x_%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/data/home/acw749/Jamendo-Instruct}
PYTHON=${PYTHON:-/data/home/acw749/conda-envs/instruct_embed/bin/python}
AUDIO_ROOT=${AUDIO_ROOT:-/data/EECS-Pauwels-C4DM/music4all/audios}
OUTPUT_JSONL=${OUTPUT_JSONL:-/gpfs/scratch/acw749/datasets/music4all_instruct/metadata/final_caption30sec.jsonl}
BATCH_SIZE=${BATCH_SIZE:-1}
TORCH_DTYPE=${TORCH_DTYPE:-auto}
LOG_DIR=${LOG_DIR:-/gpfs/scratch/acw749/datasets/music4all_instruct/logs}

mkdir -p "${LOG_DIR}" "$(dirname "${OUTPUT_JSONL}")"

cd "${REPO_ROOT}"

echo "[music4all-caption-slurm] job_id=${SLURM_JOB_ID:-unknown}"
echo "[music4all-caption-slurm] host=$(hostname)"
echo "[music4all-caption-slurm] start=$(date --iso-8601=seconds)"
echo "[music4all-caption-slurm] repo=${REPO_ROOT}"
echo "[music4all-caption-slurm] output=${OUTPUT_JSONL}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

"${PYTHON}" scripts/music4all_caption_audio.py \
  --audio-root "${AUDIO_ROOT}" \
  --output-jsonl "${OUTPUT_JSONL}" \
  --batch-size "${BATCH_SIZE}" \
  --torch-dtype "${TORCH_DTYPE}"

echo "[music4all-caption-slurm] end=$(date --iso-8601=seconds)"
