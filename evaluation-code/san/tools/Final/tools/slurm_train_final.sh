#!/bin/bash
#SBATCH --job-name=blv_final
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=44:00:00
#SBATCH --exclude=gpu026
#SBATCH --requeue
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#
# Training submission for the Final_results_and_checkpoints paper-canonical pipeline.
# All outputs go under Final_results_and_checkpoints/{work_dirs,checkpoints}.
#
# Usage:
#   sbatch Final_results_and_checkpoints/tools/slurm_train_final.sh MODEL CONFIG
#
# MODEL: segformer | mask2former | san
# CONFIG: A | B | C | D
#   A = real-only training
#   B = real + opensrc training
#   C = real(×6) + synth_0.1 mixed training
#   D = real(×12) + synth_0.2 mixed training
#
# Example:
#   sbatch Final_results_and_checkpoints/tools/slurm_train_final.sh segformer C
#
# All val/test runs use real_final/ ONLY. Best ckpt selected by real-domain mIoU
# under the patched (honest) BLVMetric.
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

# GPU sanity check — fail fast on broken nodes so SLURM requeues instead of running for hours.
echo "==== GPU sanity check on $(hostname) ===="
nvidia-smi || { echo "NODE_BAD: nvidia-smi failed on $(hostname)"; scontrol requeue "${SLURM_JOB_ID}"; exit 75; }
python - <<'PY' || { echo "NODE_BAD: torch.cuda check failed on $(hostname)"; scontrol requeue "${SLURM_JOB_ID}"; exit 75; }
import sys, torch
if not torch.cuda.is_available():
    print("torch.cuda.is_available() == False"); sys.exit(1)
try:
    x = torch.zeros(8, device='cuda')
    y = (x + 1).sum().item()
    print(f"GPU OK: {torch.cuda.get_device_name(0)} test_sum={y}")
except Exception as e:
    print(f"GPU test failed: {e}"); sys.exit(1)
PY
echo "==== GPU sanity OK, proceeding ===="

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

MODEL="${1:?Usage: slurm_train_final.sh MODEL CONFIG}"
CONFIG="${2:?Usage: slurm_train_final.sh MODEL CONFIG}"

case "${CONFIG}" in
    A) CONFIG_NAME="finalA_real" ;;
    B) CONFIG_NAME="finalB_real_opensrc" ;;
    C) CONFIG_NAME="finalC_real_synth01" ;;
    D) CONFIG_NAME="finalD_real_synth02" ;;
    *) echo "ERROR: CONFIG must be A, B, C, or D (got '${CONFIG}')"; exit 1 ;;
esac

CONFIG_PATH="${PROJECT_ROOT}/Final_results_and_checkpoints/configs/${MODEL}_${CONFIG_NAME}.py"
if [ ! -f "${CONFIG_PATH}" ]; then
    echo "ERROR: Config not found: ${CONFIG_PATH}"
    exit 1
fi

DT="$(date '+%M-%H-%d-%m-%Y')"
RUN_NAME="${MODEL}_${CONFIG_NAME}_${DT}"
WORKDIR="${PROJECT_ROOT}/Final_results_and_checkpoints/work_dirs/${RUN_NAME}"
mkdir -p "${WORKDIR}"
mkdir -p "${PROJECT_ROOT}/logs"

MMSEG_ROOT="${MMSEG_ROOT:-$(python -c 'import mmseg, os; print(os.path.dirname(os.path.dirname(mmseg.__file__)))')}"

echo "==============================================="
echo " Final pipeline training: ${RUN_NAME}"
echo "==============================================="
echo "Model    : ${MODEL}"
echo "Config   : ${CONFIG_NAME} (${CONFIG_PATH})"
echo "Work dir : ${WORKDIR}"
echo "Job ID   : ${SLURM_JOB_ID:-local}"
echo "Time     : ${SLURM_TIMELIMIT:-44h}"
echo ""

# Run training
bash "${MMSEG_ROOT}/tools/dist_train.sh" "${CONFIG_PATH}" 1 \
    --work-dir "${WORKDIR}" \
    --cfg-options \
        "default_hooks.visualization.draw=False"

# Stage best ckpt to Final_results_and_checkpoints/checkpoints/
FINETUNED_DIR="${PROJECT_ROOT}/Final_results_and_checkpoints/checkpoints/${RUN_NAME}"
BEST_CKPT="$(ls "${WORKDIR}"/best_mIoU*.pth 2>/dev/null | tail -1 || true)"
if [ -z "${BEST_CKPT}" ] && [ -f "${WORKDIR}/last_checkpoint" ]; then
    LAST_REF="$(cat "${WORKDIR}/last_checkpoint")"
    [ -f "${LAST_REF}" ] && BEST_CKPT="${LAST_REF}"
fi
if [ -z "${BEST_CKPT}" ]; then
    BEST_CKPT="$(ls "${WORKDIR}"/iter_*.pth 2>/dev/null | sort -V | tail -1 || true)"
fi
if [ -n "${BEST_CKPT}" ]; then
    mkdir -p "${FINETUNED_DIR}"
    cp "${BEST_CKPT}" "${FINETUNED_DIR}/"
    echo ">>> Staged ckpt: ${FINETUNED_DIR}/$(basename "${BEST_CKPT}")"
else
    echo "WARNING: No checkpoint found in ${WORKDIR}"
fi

echo "==============================================="
echo " DONE: ${RUN_NAME}"
echo "==============================================="
