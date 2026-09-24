#!/usr/bin/env bash
# sbatch launcher for ReMIX-C training (src/remix_c). Arguments are hydra overrides.
# One SLURM task per GPU (Lightning DDP under SLURM expects ntasks-per-node == trainer.devices).
#
#   scripts/launch_remix_c.sh model=small objective=contrastive
#   GPUS=1 TIME_LIMIT=00:30:00 scripts/launch_remix_c.sh model=small trainer.devices=1 trainer.max_steps=50
set -euo pipefail
cd "$(dirname "$0")/.."
ENV=/data/home/acw749/conda-envs/instruct_embed
GPUS="${GPUS:-2}"
mkdir -p logs
sbatch -J "${JOB_NAME:-remixc}" -p andrena -A pilot_andrena --gres="gpu:nvidia_a100-pcie-40gb:${GPUS}" \
  --nodes=1 --ntasks-per-node="${GPUS}" --cpus-per-task=10 --mem="${MEM:-120G}" -t "${TIME_LIMIT:-48:00:00}" \
  -o logs/remixc_%j.out -e logs/remixc_%j.err \
  --wrap "export PATH=${ENV}/bin:\$PATH PYTHONPATH=src HF_HOME=/gpfs/scratch/acw749/hf_cache HF_HUB_OFFLINE=1 && \
          srun python -m remix_c.train trainer.devices=${GPUS} $*"
