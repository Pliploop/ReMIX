#!/usr/bin/env bash
# Generic worker to run any jamendo_instruct pipeline stage under SLURM with the
# same environment the instruction/validation-judge jobs use. All arguments are
# forwarded verbatim to `python -m jamendo_instruct.run`, e.g.:
#
#   PROFILE=sae scripts/run_pipeline_stage.sh stage=validation \
#     runtime.output_root=/path runtime.run_name=music4all_v1 ...
#
# Env knobs: PROFILE (sae|andrena) toggles the sae deep-gemm guard.

set -euo pipefail

export PATH="/data/home/acw749/conda-envs/instruct_embed/bin:${PATH}"
PROFILE="${PROFILE:-sae}"

module load cuda/12.6.2-gcc-12.2.0
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export CUDA_PATH="${CUDA_HOME}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
# The env's bundled libstdc++ has CXXABI_1.3.15 (needed by vLLM's import chain);
# the cuda module's gcc-12 and the system libstdc++ do not. Put it first.
export LD_LIBRARY_PATH="/data/home/acw749/conda-envs/instruct_embed/lib:${LD_LIBRARY_PATH:-}"
if [[ "${PROFILE}" == "sae" ]]; then
  export VLLM_USE_DEEP_GEMM=0
  export VLLM_MOE_USE_DEEP_GEMM=0
fi

cd /data/home/acw749/Jamendo-Instruct

exec env PYTHONPATH=src /data/home/acw749/conda-envs/instruct_embed/bin/python -m jamendo_instruct.run "$@"
