#!/bin/bash
#SBATCH --job-name=blv_eval_split
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#
# Run Stage-1 split-domain evaluation (real_only + opensrc_only) on a checkpoint.
#
# Usage:
#   sbatch tools/blv/slurm_eval_stage1_split.sh MODEL CHECKPOINT [WORK_DIR]
#
# MODEL: mask2former | segformer | san
# CHECKPOINT: path to a .pth file from a Stage-1 run
# WORK_DIR: optional output dir; defaults to work_dirs/eval_split/<model>_<ts>/
#
# Example:
#   sbatch tools/blv/slurm_eval_stage1_split.sh \
#       mask2former \
#       work_dirs/full/mask2former_real_30-12-27-04-2026/best_mIoU_iter_24000.pth
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${BLV_PROJECT_ROOT:-}" ]; then
    PROJECT_ROOT="${BLV_PROJECT_ROOT}"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"

module purge || true
module load conda/latest || true
module load cuda/11.8 || true

if [ -f "${PROJECT_ROOT}/tools/blv/conda_utils.sh" ]; then
    source "${PROJECT_ROOT}/tools/blv/conda_utils.sh"
elif [ -f "${SCRIPT_DIR}/conda_utils.sh" ]; then
    source "${SCRIPT_DIR}/conda_utils.sh"
else
    echo "ERROR: conda_utils.sh not found."
    exit 1
fi
activate_conda_env

export BLV_PROJECT_ROOT="${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"

MODEL="${1:?Usage: slurm_eval_stage1_split.sh MODEL CHECKPOINT [WORK_DIR]}"
CHECKPOINT="${2:?Usage: slurm_eval_stage1_split.sh MODEL CHECKPOINT [WORK_DIR]}"

CONFIG="${PROJECT_ROOT}/configs/blv/${MODEL}_finetune_real.py"
if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: Config not found: ${CONFIG}"
    exit 1
fi

if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT}"
    exit 1
fi
CHECKPOINT="$(realpath "${CHECKPOINT}")"

DT="$(date '+%M-%H-%d-%m-%Y')"
WORK_DIR="${3:-${PROJECT_ROOT}/work_dirs/eval_split/${MODEL}_${DT}}"
mkdir -p "${WORK_DIR}"
mkdir -p "${PROJECT_ROOT}/logs"

echo "Model      : ${MODEL}"
echo "Config     : ${CONFIG}"
echo "Checkpoint : ${CHECKPOINT}"
echo "Work dir   : ${WORK_DIR}"
echo "Job ID     : ${SLURM_JOB_ID:-local}"

cd "${PROJECT_ROOT}"
python tools/blv/eval_stage1_split.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --work-dir "${WORK_DIR}"
