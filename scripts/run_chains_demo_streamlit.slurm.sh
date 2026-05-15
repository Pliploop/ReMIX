#!/usr/bin/env bash
#SBATCH --job-name=ji_streamlit_demo
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00

set -euo pipefail

cd /data/home/acw749/Jamendo-Instruct

export PYTHONPATH=src

RUN_ROOT="${RUN_ROOT:-/gpfs/scratch/acw749/datasets/music4all_instruct/music4all_v1}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/demo/logs}"
PORT="${PORT:-7860}"
MAX_CHAINS="${MAX_CHAINS:-0}"

mkdir -p "${LOG_DIR}"

APP_LOG="${LOG_DIR}/streamlit_direct_${SLURM_JOB_ID}.log"
URL_FILE="${LOG_DIR}/streamlit_direct_${SLURM_JOB_ID}.url"
rm -f "${URL_FILE}"

/data/home/acw749/conda-envs/instruct_embed/bin/python -m jamendo_instruct.demo.chains_demo \
  --run-root "${RUN_ROOT}" \
  --max-chains "${MAX_CHAINS}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  >"${APP_LOG}" 2>&1 &
APP_PID=$!

cleanup() {
  if [[ -n "${APP_PID:-}" ]]; then
    kill "${APP_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${PORT}/" >/dev/null 2>&1; then
    echo "http://${HOSTNAME}:${PORT}" | tee "${URL_FILE}"
    break
  fi
  if ! kill -0 "${APP_PID}" 2>/dev/null; then
    echo "Streamlit exited before binding port ${PORT}; see ${APP_LOG}" >&2
    exit 1
  fi
  sleep 2
done

if [[ ! -s "${URL_FILE}" ]]; then
  echo "Streamlit did not bind port ${PORT}; see ${APP_LOG}" >&2
  exit 1
fi

wait "${APP_PID}"
