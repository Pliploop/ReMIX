#!/usr/bin/env bash
#SBATCH --job-name=ji_ngrok_demo
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00

set -euo pipefail

cd /data/home/acw749/Jamendo-Instruct

export PYTHONPATH=src
export PATH="/gpfs/scratch/acw749/tools/ngrok:${PATH}"

RUN_ROOT="${RUN_ROOT:-/gpfs/scratch/acw749/datasets/music4all_instruct/music4all_v1}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/demo/logs}"
PORT="${PORT:-7860}"
MAX_CHAINS="${MAX_CHAINS:-0}"
INSTRUCTIONS_JSONL="${INSTRUCTIONS_JSONL:-}"

mkdir -p "${LOG_DIR}"

URL_FILE="${LOG_DIR}/ngrok_${SLURM_JOB_ID}.url"
APP_LOG="${LOG_DIR}/streamlit_inner_${SLURM_JOB_ID}.log"
NGROK_LOG="${LOG_DIR}/ngrok_${SLURM_JOB_ID}.log"
rm -f "${URL_FILE}"

cleanup() {
  if [[ -n "${NGROK_PID:-}" ]]; then
    kill "${NGROK_PID}" 2>/dev/null || true
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

NGROK_CMD=(
  ngrok http "http://127.0.0.1:${PORT}"
  --log stdout
  --log-format logfmt
)
if [[ -n "${NGROK_AUTHTOKEN:-}" ]]; then
  NGROK_CMD+=(--authtoken "${NGROK_AUTHTOKEN}")
fi
"${NGROK_CMD[@]}" >"${NGROK_LOG}" 2>&1 &
NGROK_PID=$!

for _ in $(seq 1 120); do
  if ! kill -0 "${NGROK_PID}" 2>/dev/null; then
    echo "ngrok exited before publishing a tunnel; see ${NGROK_LOG}" >&2
    exit 1
  fi
  URL=$(/data/home/acw749/conda-envs/instruct_embed/bin/python - "${NGROK_LOG}" <<'PY' || true
import re
import sys

log_path = sys.argv[1]
try:
    text = open(log_path, "r", encoding="utf-8", errors="replace").read()
except OSError:
    raise SystemExit(1)

matches = re.findall(r"https://[a-zA-Z0-9.-]+\.ngrok(?:-free)?\.(?:app|dev)", text)
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
  echo "ngrok did not publish a public URL; see ${NGROK_LOG}" >&2
  exit 1
fi

wait "${NGROK_PID}"
