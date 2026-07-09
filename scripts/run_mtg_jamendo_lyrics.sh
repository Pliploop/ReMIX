#!/usr/bin/env bash
set -euo pipefail

cd /data/home/acw749/Jamendo-Instruct

RUN_NAME="${RUN_NAME:-v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/gpfs/scratch/acw749/datasets/mtg_jamendo_instruct}"

/data/home/acw749/conda-envs/instruct_embed/bin/python -m jamendo_instruct.run \
  dataset=mtgjamendo \
  runtime.output_root="${OUTPUT_ROOT}" \
  runtime.run_name="${RUN_NAME}" \
  stage=lyrics \
  stage.filters.only_vocal_tracks=false \
  "$@"
