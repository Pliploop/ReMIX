#!/usr/bin/env bash
#SBATCH --job-name=ji_cloudflared_demo
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
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/demo/logs}"
PORT="${PORT:-7860}"
MAX_CHAINS="${MAX_CHAINS:-0}"
INSTRUCTIONS_JSONL="${INSTRUCTIONS_JSONL:-${RUN_ROOT}/${INSTRUCTION_NAME}/chain_step_instructions.jsonl}"

mkdir -p "${LOG_DIR}"

JOB_ID="${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)_$$}"
URL_FILE="${LOG_DIR}/cloudflared_${JOB_ID}.url"
APP_LOG="${LOG_DIR}/streamlit_inner_${JOB_ID}.log"
CLOUDFLARED_LOG="${LOG_DIR}/cloudflared_${JOB_ID}.log"
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
  /data/home/acw749/conda-envs/instruct_embed/bin/python -m jamendo_instruct.demo.chains_demo
  --run-root "${RUN_ROOT}"
  --max-chains "${MAX_CHAINS}"
  --host 127.0.0.1
  --port "${PORT}"
)
if [[ -n "${INSTRUCTIONS_JSONL}" ]]; then
  APP_CMD+=(--instructions-jsonl "${INSTRUCTIONS_JSONL}")
fi
"${APP_CMD[@]}" >"${APP_LOG}" 2>&1 &
APP_PID=$!

for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${PORT}/" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "${APP_PID}" 2>/dev/null; then
    echo "Streamlit exited before binding port ${PORT}; see ${APP_LOG}" >&2
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

echo "Streamlit log: ${APP_LOG}"
echo "cloudflared log: ${CLOUDFLARED_LOG}"

while true; do
  if ! kill -0 "${APP_PID}" 2>/dev/null; then
    wait "${APP_PID}"
    app_status=$?
    echo "Streamlit exited with status ${app_status}; see ${APP_LOG}" >&2
    exit "${app_status}"
  fi
  if ! kill -0 "${CLOUDFLARED_PID}" 2>/dev/null; then
    wait "${CLOUDFLARED_PID}"
    tunnel_status=$?
    echo "cloudflared exited with status ${tunnel_status}; see ${CLOUDFLARED_LOG}" >&2
    exit "${tunnel_status}"
  fi
  sleep 5
done
