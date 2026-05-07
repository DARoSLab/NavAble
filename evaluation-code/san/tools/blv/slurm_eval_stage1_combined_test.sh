#!/bin/bash
#SBATCH --job-name=blv_eval_combined
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#
# Run combined-domain test eval (real_v2/test + opensrc/test as ConcatDataset)
# on a Stage-1 checkpoint. Reuses the *_finetune_real.py config (which already
# defines test_dataloader = ConcatDataset(real_v2, opensrc) over the test split).
#
# Usage:
#   sbatch tools/blv/slurm_eval_stage1_combined_test.sh MODEL CHECKPOINT [WORK_DIR]
#
# MODEL: mask2former | segformer | san
# CHECKPOINT: path to a .pth file from a Stage-1 run
# WORK_DIR: optional output dir; defaults to work_dirs/eval_combined_test/<model>_<ts>/
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

source "${PROJECT_ROOT}/tools/blv/conda_utils.sh"
activate_conda_env

export BLV_PROJECT_ROOT="${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export WANDB_MODE=disabled

_pick_free_master_port() {
    python - <<'PY'
import os, random, socket
def is_free(p):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("127.0.0.1", p))
    except OSError:
        return False
    finally:
        s.close()
    return True
cands = []
jid = os.environ.get('SLURM_JOB_ID', '')
if jid.isdigit():
    base = 20000 + (int(jid) % 30000)
    for off in range(1024):
        cands.append(20000 + ((base - 20000 + off) % 45536))
for _ in range(2048):
    cands.append(random.randint(20000, 65535))
for p in cands:
    if is_free(p):
        print(p); raise SystemExit(0)
raise SystemExit('ERROR: no free port')
PY
}
FREE_PORT="$(_pick_free_master_port)"
export PORT="${FREE_PORT}"
export MASTER_PORT="${FREE_PORT}"
export MASTER_ADDR=127.0.0.1

MODEL="${1:?Usage: slurm_eval_stage1_combined_test.sh MODEL CHECKPOINT [WORK_DIR]}"
CKPT="${2:?Usage: slurm_eval_stage1_combined_test.sh MODEL CHECKPOINT [WORK_DIR]}"

CONFIG="${PROJECT_ROOT}/configs/blv/${MODEL}_finetune_real.py"
if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: Config not found: ${CONFIG}"
    exit 1
fi
if [ ! -f "${CKPT}" ]; then
    echo "ERROR: Checkpoint not found: ${CKPT}"
    exit 1
fi
CKPT="$(realpath "${CKPT}")"

DT="$(date '+%M-%H-%d-%m-%Y')"
SPLIT="${SPLIT:-test}"
if [ "${SPLIT}" != "val" ] && [ "${SPLIT}" != "test" ]; then
    echo "ERROR: SPLIT must be 'val' or 'test' (got '${SPLIT}')"
    exit 1
fi
WORK_DIR="${3:-${PROJECT_ROOT}/work_dirs/eval_combined_${SPLIT}/${MODEL}_${DT}}"
mkdir -p "${WORK_DIR}"
mkdir -p "${PROJECT_ROOT}/logs"

MMSEG_ROOT="${MMSEG_ROOT:-$(python -c 'import mmseg, os; print(os.path.dirname(os.path.dirname(mmseg.__file__)))')}"

echo "Model      : ${MODEL}"
echo "Config     : ${CONFIG}"
echo "Checkpoint : ${CKPT}"
echo "Split      : ${SPLIT}"
echo "Work dir   : ${WORK_DIR}"
echo "Job ID     : ${SLURM_JOB_ID:-local}"

bash "${MMSEG_ROOT}/tools/dist_test.sh" "${CONFIG}" "${CKPT}" 1 \
    --work-dir "${WORK_DIR}" \
    --cfg-options \
        test_evaluator.output_metrics_path="${WORK_DIR}/metrics.json" \
        test_dataloader.dataset.datasets.0.data_prefix.img_path="img_dir/${SPLIT}" \
        test_dataloader.dataset.datasets.0.data_prefix.seg_map_path="ann_dir/${SPLIT}" \
        test_dataloader.dataset.datasets.1.data_prefix.img_path="img_dir/${SPLIT}" \
        test_dataloader.dataset.datasets.1.data_prefix.seg_map_path="ann_dir/${SPLIT}"
