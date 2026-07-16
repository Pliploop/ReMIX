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

# The render directory for this quality. Required: media/ accumulates a
# directory per quality, so picking a scene's mp4 with `find | head -1` silently
# grabs whichever resolution the filesystem lists first -- mixing stale renders
# of different sizes into the concat and dropping seconds off the film.
case "$QUALITY" in
  -ql) RES_DIR="480p15" ;;
  -qm) RES_DIR="720p30" ;;
  -qh) RES_DIR="1080p60" ;;
  -qk) RES_DIR="2160p60" ;;
  *) echo "!! unknown quality ${QUALITY}" >&2; exit 1 ;;
esac

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
  found="media/videos/${stem}/${RES_DIR}/${klass}.mp4"
  if [[ ! -f "$found" ]]; then
    echo "!! missing ${RES_DIR} render for ${klass}" >&2
    exit 1
  fi
  echo "file '$(realpath "$found")'" >> "$list"
done

mkdir -p out
"$FFMPEG" -y -f concat -safe 0 -i "$list" -c:v libx264 -pix_fmt yuv420p -crf 18 out/remix.mp4
rm -f "$list"
echo "==> out/remix.mp4"
