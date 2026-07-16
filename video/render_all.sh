#!/usr/bin/env bash
# Render every scene and concatenate them into one film.
#
#   bash render_all.sh          # -ql draft
#   bash render_all.sh -qh      # 1080p (submit this; do not run it on a login node)

set -euo pipefail

QUALITY="${1:--ql}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIM="${MANIM:-/data/home/acw749/conda-envs/manim/bin/manim}"
FFMPEG="${FFMPEG:-/data/home/acw749/conda-envs/manim/bin/ffmpeg}"

cd "$HERE"
export PYTHONPATH="$HERE"

# Scene order is the film's order.
SCENES=(
  "scenes/s00_cold_open.py:ColdOpen"
  "scenes/s01_enrich.py:Enrich"
  "scenes/s02_neighbourhood.py:Neighbourhood"
  "scenes/s03_chains.py:Chains"
  "scenes/s04_instructions.py:Instructions"
  "scenes/s05_validation.py:Validation"
  "scenes/s06_assemble.py:Assemble"
)

for entry in "${SCENES[@]}"; do
  file="${entry%%:*}"
  klass="${entry##*:}"
  echo "==> ${klass}"
  "$MANIM" "$QUALITY" --format=mp4 "$file" "$klass"
done

# Concatenate in order. Re-encode rather than stream-copy: scenes can differ in
# encoder settings, and a copy-concat then produces a broken file.
list="$(mktemp)"
for entry in "${SCENES[@]}"; do
  file="${entry%%:*}"
  klass="${entry##*:}"
  stem="$(basename "$file" .py)"
  found="$(find media/videos/"$stem" -name "${klass}.mp4" | head -1)"
  if [[ -z "$found" ]]; then
    echo "!! missing render for ${klass}" >&2
    exit 1
  fi
  echo "file '$(realpath "$found")'" >> "$list"
done

mkdir -p out
"$FFMPEG" -y -f concat -safe 0 -i "$list" -c:v libx264 -pix_fmt yuv420p -crf 18 out/remix.mp4
rm -f "$list"
echo "==> out/remix.mp4"
