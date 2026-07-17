#!/usr/bin/env bash
# UMAP spherical projection for the /explore page.
#
# PCA keeps only ~29% of the variance in the first three components, so a PCA
# sphere discards most of the structure the chains were sampled from. UMAP with
# output_metric='haversine' embeds straight onto the sphere and keeps the
# neighbourhoods intact. It is CPU-heavy, hence a batch job rather than the
# login node.
#
# Usage:
#   bash scripts/launch_explorer_umap.sh
#   ONLY=mtg_jamendo bash scripts/launch_explorer_umap.sh

set -euo pipefail

PYTHON="${PYTHON:-/data/home/acw749/conda-envs/instruct_embed/bin/python}"
REPO="${REPO:-/data/home/acw749/Jamendo-Instruct}"
ONLY="${ONLY:-}"
# computeshort (1h cap, usually near-empty) backfills far sooner than `compute`
# when a big long-running array is hogging the main partition. A single-dataset
# UMAP finishes in ~10 min, so the cap is not a constraint.
PARTITION="${PARTITION:-computeshort}"
TIME_LIMIT="${TIME_LIMIT:-00:45:00}"
CPUS="${CPUS:-16}"

mkdir -p "${REPO}/logs"

# --no-filter-chains: ship every resolvable chain (the page has the rating
# toggle + slider). FILTER=1 flips back to validated-only if you ever want the
# small curated export instead.
ARGS="--method umap"
if [[ "${FILTER:-0}" != "1" ]]; then
  ARGS="${ARGS} --no-filter-chains"
fi
if [[ -n "${ONLY}" ]]; then
  ARGS="${ARGS} --only ${ONLY}"
fi

sbatch \
  -J remix_umap \
  -p "${PARTITION}" \
  -n 1 \
  --cpus-per-task="${CPUS}" \
  --mem=48G \
  -t "${TIME_LIMIT}" \
  -o "${REPO}/logs/umap_%j.out" \
  -e "${REPO}/logs/umap_%j.err" \
  --wrap "cd ${REPO} && OMP_NUM_THREADS=${CPUS} NUMBA_NUM_THREADS=${CPUS} ${PYTHON} -u scripts/export_explorer_data.py ${ARGS}"
