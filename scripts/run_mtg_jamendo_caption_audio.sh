#!/usr/bin/env bash
set -euo pipefail

cd /data/home/acw749/Jamendo-Instruct

export PYTHONPATH=src:scripts

METADATA_DIR="${METADATA_DIR:-/gpfs/scratch/acw749/datasets/mtg_jamendo_instruct/metadata}"

/data/home/acw749/conda-envs/instruct_embed/bin/python scripts/mtg_jamendo_caption_audio.py \
  --metadata-jsonl "${METADATA_DIR}/mtg_jamendo_tracks.jsonl" \
  --output-jsonl "${METADATA_DIR}/final_caption30sec.jsonl" \
  "$@"
