#!/usr/bin/env bash
# Render the film at 1080p60 on a compute node.
#
# -qh is manim's high quality preset: 1920x1080 at 60fps. manim.cfg deliberately
# does not set frame_rate -- setting it there overrides the preset and silently
# caps the "60fps" render at 30.
#
# Rendering is CPU-bound and takes far longer than the draft, so it is a batch
# job, not something to run on a login node.
#
# Usage:
#   bash launch_render_hq.sh
#   CPUS=32 TIME_LIMIT=04:00:00 bash launch_render_hq.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPUS="${CPUS:-16}"
MEM="${MEM:-32G}"
TIME_LIMIT="${TIME_LIMIT:-03:00:00}"

mkdir -p "${HERE}/../logs"

sbatch \
  -J remix_video_hq \
  -p compute \
  -n 1 \
  --cpus-per-task="${CPUS}" \
  --mem="${MEM}" \
  -t "${TIME_LIMIT}" \
  -o "${HERE}/../logs/video_hq_%j.out" \
  -e "${HERE}/../logs/video_hq_%j.err" \
  --wrap "cd ${HERE} && OMP_NUM_THREADS=${CPUS} bash render_all.sh -qh"
