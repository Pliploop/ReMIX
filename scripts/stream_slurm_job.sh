#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/stream_slurm_job.sh JOB_ID [LINES]

Pretty-stream stdout/stderr for a Slurm job.

Examples:
  scripts/stream_slurm_job.sh 9957339
  scripts/stream_slurm_job.sh 9957339 200
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage >&2
  exit 2
fi

job_id="$1"
lines="${2:-80}"

if ! [[ "$lines" =~ ^[0-9]+$ ]]; then
  echo "LINES must be a non-negative integer: $lines" >&2
  exit 2
fi

field_from_scontrol() {
  local field="$1"
  scontrol show job "$job_id" -o 2>/dev/null \
    | tr ' ' '\n' \
    | awk -F= -v key="$field" '$1 == key {print substr($0, length(key) + 2); exit}'
}

stdout_path="$(field_from_scontrol StdOut || true)"
stderr_path="$(field_from_scontrol StdErr || true)"
job_name="$(field_from_scontrol JobName || true)"
job_state="$(field_from_scontrol JobState || true)"
work_dir="$(field_from_scontrol WorkDir || true)"

if [[ -z "$stdout_path" || -z "$stderr_path" ]]; then
  sacct_row="$(
    sacct -j "$job_id" -X -n -P \
      --format=JobName%80,State%30,Elapsed,WorkDir%200,StdOut%300,StdErr%300 \
      2>/dev/null \
      | head -n 1 || true
  )"
  if [[ -n "$sacct_row" ]]; then
    IFS='|' read -r sacct_name sacct_state _elapsed sacct_work_dir sacct_stdout sacct_stderr <<<"$sacct_row"
    job_name="${job_name:-$sacct_name}"
    job_state="${job_state:-$sacct_state}"
    work_dir="${work_dir:-$sacct_work_dir}"
    stdout_path="${stdout_path:-$sacct_stdout}"
    stderr_path="${stderr_path:-$sacct_stderr}"
  fi
fi

if [[ -z "$stdout_path" && -z "$stderr_path" ]]; then
  echo "Could not resolve stdout/stderr for job $job_id via scontrol or sacct." >&2
  exit 1
fi

resolve_relative_path() {
  local path="$1"
  if [[ -z "$path" || "$path" == "(null)" ]]; then
    return 0
  fi
  if [[ "$path" == /* || -z "$work_dir" || "$work_dir" == "(null)" ]]; then
    printf '%s\n' "$path"
  else
    printf '%s/%s\n' "$work_dir" "$path"
  fi
}

stdout_path="$(resolve_relative_path "$stdout_path")"
stderr_path="$(resolve_relative_path "$stderr_path")"

printf '\033[1;36mSlurm job\033[0m %s' "$job_id"
if [[ -n "${job_name:-}" ]]; then
  printf '  \033[2m%s\033[0m' "$job_name"
fi
if [[ -n "${job_state:-}" ]]; then
  printf '  \033[1m%s\033[0m' "$job_state"
fi
printf '\n'

printf 'stdout: \033[2m%s\033[0m\n' "${stdout_path:-<none>}"
printf 'stderr: \033[2m%s\033[0m\n' "${stderr_path:-<none>}"
printf '\033[2mShowing last %s lines, then following. Press Ctrl-C to stop.\033[0m\n\n' "$lines"

tail_args=()
if [[ -n "$stdout_path" && "$stdout_path" != "(null)" ]]; then
  tail_args+=("$stdout_path")
fi
if [[ -n "$stderr_path" && "$stderr_path" != "(null)" && "$stderr_path" != "$stdout_path" ]]; then
  tail_args+=("$stderr_path")
fi

if [[ ${#tail_args[@]} -eq 0 ]]; then
  echo "No usable log paths found for job $job_id." >&2
  exit 1
fi

tail -n "$lines" -F "${tail_args[@]}" 2>&1 \
  | awk -v out="$stdout_path" -v err="$stderr_path" '
      BEGIN {
        src = "log"
        cyan = "\033[36m"
        red = "\033[31m"
        dim = "\033[2m"
        reset = "\033[0m"
      }
      /^==> .* <==$/ {
        path = $0
        sub(/^==> /, "", path)
        sub(/ <==$/, "", path)
        if (path == out) {
          src = "stdout"
          color = cyan
        } else if (path == err) {
          src = "stderr"
          color = red
        } else {
          src = "log"
          color = dim
        }
        print dim "==> " path " <==" reset
        next
      }
      {
        if (src == "stderr") {
          color = red
        } else if (src == "stdout") {
          color = cyan
        } else {
          color = dim
        }
        print color "[" src "]" reset " " $0
      }
    '
