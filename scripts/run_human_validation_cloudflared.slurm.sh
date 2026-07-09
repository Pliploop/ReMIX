#!/usr/bin/env bash
#SBATCH --job-name=ji_cloudflared_validation
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00

set -euo pipefail

cd /data/home/acw749/Jamendo-Instruct

export PYTHONPATH=src
export PATH="/gpfs/scratch/acw749/tools/cloudflared:${PATH}"
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

DATASET_NAME="${DATASET_NAME:-music4all_v1}"
INSTRUCTION_NAME="${INSTRUCTION_NAME:-instructions_axis_focused_5}"
DATASET_BASE="${DATASET_BASE:-/gpfs/scratch/acw749/datasets/music4all_instruct}"
RUN_ROOT="${RUN_ROOT:-${DATASET_BASE}/${DATASET_NAME}}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/validation/logs}"
PORT="${PORT:-7861}"
MAX_CHAINS="${MAX_CHAINS:-0}"
CHAIN_OFFSET="${CHAIN_OFFSET:-0}"
INSTRUCTION_SLUG="${INSTRUCTION_SLUG:-${INSTRUCTION_NAME#instructions_}}"
INSTRUCTIONS_JSONL="${INSTRUCTIONS_JSONL:-${RUN_ROOT}/${INSTRUCTION_NAME}/chain_step_instructions.jsonl}"
ASSIGNMENT_JSONL="${ASSIGNMENT_JSONL:-${RUN_ROOT}/validation/assignment_${INSTRUCTION_SLUG}_v1.jsonl}"
FROZEN_SIDECAR_JSON="${FROZEN_SIDECAR_JSON:-${ASSIGNMENT_JSONL%.jsonl}.sidecar.json}"
ADMIN_PASSWORD="${ADMIN_PASSWORD:-jamendo-admin}"

mkdir -p "${LOG_DIR}"

JOB_ID="${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)_$$}"
URL_FILE="${LOG_DIR}/human_validation_cloudflared_${JOB_ID}.url"
APP_LOG="${LOG_DIR}/human_validation_streamlit_${JOB_ID}.log"
CLOUDFLARED_LOG="${LOG_DIR}/human_validation_cloudflared_${JOB_ID}.log"
rm -f "${URL_FILE}"

cleanup() {
  if [[ -n "${CLOUDFLARED_PID:-}" ]]; then
    kill "${CLOUDFLARED_PID}" 2>/dev/null || true
  fi
  if [[ -n "${APP_PID:-}" ]]; then
    kill "${APP_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

APP_CMD=(
  /data/home/acw749/conda-envs/instruct_embed/bin/python -m jamendo_instruct.demo.human_validation_app
  --run-root "${RUN_ROOT}"
  --chain-offset "${CHAIN_OFFSET}"
  --max-chains "${MAX_CHAINS}"
  --host 127.0.0.1
  --port "${PORT}"
  --admin-password "${ADMIN_PASSWORD}"
)
if [[ -n "${INSTRUCTIONS_JSONL}" ]]; then
  APP_CMD+=(--instructions-jsonl "${INSTRUCTIONS_JSONL}")
fi
if [[ -n "${ASSIGNMENT_JSONL}" && -f "${ASSIGNMENT_JSONL}" ]]; then
  APP_CMD+=(--assignment-jsonl "${ASSIGNMENT_JSONL}")
fi
if [[ -n "${FROZEN_SIDECAR_JSON}" && -f "${FROZEN_SIDECAR_JSON}" ]]; then
  APP_CMD+=(--frozen-sidecar-json "${FROZEN_SIDECAR_JSON}")
fi

"${APP_CMD[@]}" >"${APP_LOG}" 2>&1 &
APP_PID=$!

for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${PORT}/" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "${APP_PID}" 2>/dev/null; then
    echo "Validation Streamlit exited before binding port ${PORT}; see ${APP_LOG}" >&2
    exit 1
  fi
  sleep 2
done

cloudflared tunnel --url "http://127.0.0.1:${PORT}" --no-autoupdate >"${CLOUDFLARED_LOG}" 2>&1 &
CLOUDFLARED_PID=$!

for _ in $(seq 1 120); do
  if ! kill -0 "${CLOUDFLARED_PID}" 2>/dev/null; then
    echo "cloudflared exited before publishing a tunnel; see ${CLOUDFLARED_LOG}" >&2
    exit 1
  fi
  URL=$(/data/home/acw749/conda-envs/instruct_embed/bin/python - "${CLOUDFLARED_LOG}" <<'PY' || true
import re
import sys

log_path = sys.argv[1]
try:
    text = open(log_path, "r", encoding="utf-8", errors="replace").read()
except OSError:
    raise SystemExit(1)

matches = re.findall(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", text)
if matches:
    print(matches[-1])
PY
)
  if [[ -n "${URL}" ]]; then
    echo "${URL}" | tee "${URL_FILE}"
    break
  fi
  sleep 2
done

if [[ ! -s "${URL_FILE}" ]]; then
  echo "cloudflared did not publish a public URL; see ${CLOUDFLARED_LOG}" >&2
  exit 1
fi

while true; do
  if ! kill -0 "${APP_PID}" 2>/dev/null; then
    echo "Validation Streamlit exited after tunnel startup; see ${APP_LOG}" >&2
    exit 1
  fi
  if ! kill -0 "${CLOUDFLARED_PID}" 2>/dev/null; then
    echo "cloudflared exited after tunnel startup; see ${CLOUDFLARED_LOG}" >&2
    wait "${CLOUDFLARED_PID}" || true
    exit 1
  fi
  sleep 30
done
