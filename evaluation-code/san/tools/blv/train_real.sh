#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source "${SCRIPT_DIR}/conda_utils.sh"
activate_conda_env

export BLV_PROJECT_ROOT="${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
MMSEG_ROOT="${MMSEG_ROOT:-$(python -c 'import mmseg, os; print(os.path.dirname(os.path.dirname(mmseg.__file__)))')}"
GPUS="${GPUS:-2}"

CONFIG="${1:-${PROJECT_ROOT}/configs/blv/mask2former_finetune_real.py}"
WORKDIR="${2:-${PROJECT_ROOT}/work_dirs/blv_track_B}"
EXTRA_ARGS=("${@:3}")

bash "${MMSEG_ROOT}/tools/dist_train.sh" "${CONFIG}" "${GPUS}" --work-dir "${WORKDIR}" "${EXTRA_ARGS[@]}"
