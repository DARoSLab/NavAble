#!/bin/bash
#SBATCH --job-name=blv_trial
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=logs/slurm_trial_%x_%j.out
#SBATCH --error=logs/slurm_trial_%x_%j.err
#
# Fast queue trial launcher for Unity smoke tests with W&B enabled by default.
#
# Purpose:
#   - Get onto available GPUs quickly (request any GPU type, 1 GPU total)
#   - Run short smoke jobs to verify config/import/checkpoint/W&B plumbing
#   - Keep CLI mostly compatible with tools/blv/slurm_train.sh
#
# Usage:
#   sbatch tools/blv/trial_slurm_train.sh MODEL TRACK [--smoke-iters N] [--no-wandb] [--resume CKPT]
#
# Examples:
#   sbatch tools/blv/trial_slurm_train.sh san synthetic
#   sbatch tools/blv/trial_slurm_train.sh segformer real --smoke-iters 20
#   sbatch tools/blv/trial_slurm_train.sh mask2former real_synthetic --no-wandb
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Under SLURM the script is copied to /var/spool/... so BASH_SOURCE is not the
# repo path. Prefer BLV_PROJECT_ROOT or SLURM_SUBMIT_DIR to locate the project.
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

# shellcheck source=conda_utils.sh
if [ -f "${PROJECT_ROOT}/tools/blv/conda_utils.sh" ]; then
    source "${PROJECT_ROOT}/tools/blv/conda_utils.sh"
elif [ -f "${SCRIPT_DIR}/conda_utils.sh" ]; then
    source "${SCRIPT_DIR}/conda_utils.sh"
else
    echo "ERROR: conda_utils.sh not found."
    echo "  looked in:"
    echo "    ${PROJECT_ROOT}/tools/blv/conda_utils.sh"
    echo "    ${SCRIPT_DIR}/conda_utils.sh"
    exit 1
fi
activate_conda_env

export BLV_PROJECT_ROOT="${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
# Same strategy as slurm_train.sh: verify a free rendezvous port per job.
# By default, ignore inherited MASTER_PORT to prevent cross-job collisions.
# To force a specific inherited port, set BLV_RESPECT_MASTER_PORT=1.
_pick_free_master_port() {
    python - <<'PY'
import os
import random
import socket

def is_free(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", port))
    except OSError:
        return False
    finally:
        sock.close()
    return True

candidates = []
job_id = os.environ.get('SLURM_JOB_ID', '')
if job_id.isdigit():
    base = 20000 + (int(job_id) % 30000)
    for offset in range(0, 1024):
        candidates.append(20000 + ((base - 20000 + offset) % 45536))

for _ in range(2048):
    candidates.append(random.randint(20000, 65535))

for port in candidates:
    if is_free(port):
        print(port)
        raise SystemExit(0)

raise SystemExit('ERROR: Unable to find a free MASTER_PORT')
PY
}

if [ "${BLV_RESPECT_MASTER_PORT:-0}" = "1" ] && [ -n "${PORT:-}" ]; then
    export MASTER_PORT="${PORT}"
    echo ">>> PORT pinned by environment: ${PORT}"
elif [ "${BLV_RESPECT_MASTER_PORT:-0}" = "1" ] && [ -n "${MASTER_PORT:-}" ]; then
    export PORT="${MASTER_PORT}"
    echo ">>> MASTER_PORT pinned by environment: ${MASTER_PORT}"
else
    FREE_PORT="$(_pick_free_master_port)"
    export PORT="${FREE_PORT}"
    export MASTER_PORT="${FREE_PORT}"
fi
if [ -z "${MASTER_ADDR:-}" ] && [ "${SLURM_NNODES:-1}" -le 1 ]; then
    export MASTER_ADDR=127.0.0.1
fi
echo ">>> Distributed rendezvous: MASTER_ADDR=${MASTER_ADDR:-unset} PORT=${PORT}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
MMSEG_ROOT="${MMSEG_ROOT:-$(python -c 'import mmseg, os; print(os.path.dirname(os.path.dirname(mmseg.__file__)))')}"
GPUS=1

MODEL="${1:?Usage: trial_slurm_train.sh MODEL TRACK [--smoke-iters N] [--no-wandb] [--resume CKPT]}"
TRACK="${2:?Usage: trial_slurm_train.sh MODEL TRACK [--smoke-iters N] [--no-wandb] [--resume CKPT]}"

# Trial jobs are always smoke jobs.
SMOKE=1
SMOKE_ITERS=20
NOWANDB=0
RESUME_CKPT=""
EXTRA_ARGS=()
ARGS=("${@:3}")
IDX=0
while [ ${IDX} -lt ${#ARGS[@]} ]; do
    arg="${ARGS[${IDX}]}"
    if [ "${arg}" = "--smoke-iters" ]; then
        IDX=$((IDX + 1))
        SMOKE_ITERS="${ARGS[${IDX}]}"
    elif [ "${arg}" = "--no-wandb" ]; then
        NOWANDB=1
    elif [ "${arg}" = "--resume" ]; then
        IDX=$((IDX + 1))
        RESUME_CKPT="${ARGS[${IDX}]}"
    else
        EXTRA_ARGS+=("${arg}")
    fi
    IDX=$((IDX + 1))
done

DT="$(date '+%M-%H-%d-%m-%Y')"
RUN_TYPE="trial_smoke"

CONFIG="${PROJECT_ROOT}/configs/blv/${MODEL}_finetune_${TRACK}.py"
if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: Config not found: ${CONFIG}"
    exit 1
fi

# Hardware preflight for trial jobs:
# Mask2Former (Swin-L) needs both modern CUDA arch support for mmcv's
# ms_deform_attn op and enough VRAM to avoid immediate OOM even in smoke mode.
GPU_INFO="$(python - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    print("none,0,0")
else:
    p = torch.cuda.get_device_properties(0)
    total_gb = p.total_memory / (1024**3)
    cc = p.major + p.minor / 10.0
    print(f"{p.name},{total_gb:.2f},{cc:.1f}")
PY
)"
GPU_NAME="$(echo "${GPU_INFO}" | cut -d, -f1)"
GPU_MEM_GB="$(echo "${GPU_INFO}" | cut -d, -f2)"
GPU_CC="$(echo "${GPU_INFO}" | cut -d, -f3)"
echo "GPU      : ${GPU_NAME} (${GPU_MEM_GB} GiB, CC ${GPU_CC})"

if [ "${MODEL}" = "mask2former" ]; then
    MASK2F_OK="$(python - <<'PY'
import torch
ok = True
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    ok = False
else:
    p = torch.cuda.get_device_properties(0)
    mem_gb = p.total_memory / (1024**3)
    cc = p.major + p.minor / 10.0
    # cc>=7.0 avoids many old-arch mmcv CUDA op issues; >=20GB avoids common OOM.
    ok = (cc >= 7.0) and (mem_gb >= 20.0)
print("1" if ok else "0")
PY
)"
    if [ "${MASK2F_OK}" != "1" ]; then
        echo "ERROR: This GPU is not suitable for Mask2Former trial smoke."
        echo "  Required (recommended minimum): compute capability >= 7.0 and VRAM >= 20 GiB."
        echo "  Current: ${GPU_NAME} (${GPU_MEM_GB} GiB, CC ${GPU_CC})"
        echo "  Why: low-VRAM GPUs OOM, and older GPUs often fail with"
        echo "       'ms_deformable_im2col_cuda: no kernel image is available'."
        echo "  Use san/segformer on this node, or request a newer/larger GPU (e.g., l40s/a40/a100)."
        exit 1
    fi
fi

# Ensure required pretrained checkpoint exists (download if missing).
MODEL_CKPT_KEY=""
MODEL_CKPT_FILE=""
case "${MODEL}" in
    mask2former)
        MODEL_CKPT_KEY="mask2former"
        MODEL_CKPT_FILE="mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640.pth"
        ;;
    segformer)
        MODEL_CKPT_KEY="segformer"
        MODEL_CKPT_FILE="segformer_mit-b5_640x640_160k_ade20k_20210801_121243-41d2845b.pth"
        ;;
    san)
        MODEL_CKPT_KEY="san_b16"
        MODEL_CKPT_FILE="san-vit-b16_20230906-fd0a7684.pth"
        ;;
esac

if [ -n "${MODEL_CKPT_FILE}" ]; then
    PRETRAINED_DIR="${PROJECT_ROOT}/checkpoints/pretrained"
    REQUIRED_CKPT="${PRETRAINED_DIR}/${MODEL_CKPT_FILE}"
    if [ ! -f "${REQUIRED_CKPT}" ]; then
        echo ">>> Missing pretrained checkpoint: ${REQUIRED_CKPT}"
        echo ">>> Attempting automatic download (${MODEL_CKPT_KEY}) ..."
        python "${PROJECT_ROOT}/tools/blv/download_checkpoints.py" \
            --model "${MODEL_CKPT_KEY}" \
            --dest-dir "${PRETRAINED_DIR}"
    fi
    if [ ! -f "${REQUIRED_CKPT}" ]; then
        echo "ERROR: Required checkpoint still missing after download attempt: ${REQUIRED_CKPT}"
        exit 1
    fi
fi

# W&B auth check (same non-interactive safety as slurm_train.sh).
if [ "${NOWANDB}" -eq 0 ]; then
    if [ -n "${WANDB_API_KEY:-}" ]; then
        export WANDB_API_KEY
        echo ">>> W&B auth: using WANDB_API_KEY env var"
    elif grep -q "api.wandb.ai" "${HOME}/.netrc" 2>/dev/null; then
        echo ">>> W&B auth: using credentials from ~/.netrc (wandb login)"
    else
        echo "ERROR: No W&B credentials found."
        echo "  Fix (choose one):"
        echo "    a) Run 'wandb login' once interactively on this cluster."
        echo "    b) Set WANDB_API_KEY=<your-key> in your environment."
        echo "    c) Pass --no-wandb to disable W&B for this run."
        exit 1
    fi
fi

SMOKE_INTERVAL=$(( SMOKE_ITERS / 2 ))
[ "${SMOKE_INTERVAL}" -lt 1 ] && SMOKE_INTERVAL=1
EXTRA_ARGS+=(
    "--cfg-options"
    "train_cfg.max_iters=${SMOKE_ITERS}"
    "train_cfg.val_begin=$((SMOKE_ITERS + 1))"
    "train_cfg.val_interval=${SMOKE_INTERVAL}"
    "default_hooks.checkpoint.interval=${SMOKE_INTERVAL}"
    "train_dataloader.batch_size=1"
)
echo ">>> TRIAL SMOKE MODE (${SMOKE_ITERS} iters, batch=1, val skipped)"

if [ "${NOWANDB}" -eq 1 ]; then
    export WANDB_MODE=disabled
    EXTRA_ARGS+=(
        "--cfg-options"
        "vis_backends=[dict(type='LocalVisBackend')]"
        "default_hooks.visualization.draw=False"
    )
    echo ">>> W&B DISABLED for this run"
fi

if [ -n "${RESUME_CKPT}" ]; then
    if [ ! -f "${RESUME_CKPT}" ]; then
        echo "ERROR: Resume checkpoint not found: ${RESUME_CKPT}"
        exit 1
    fi
    RESUME_CKPT="$(realpath "${RESUME_CKPT}")"
    WORKDIR="$(dirname "${RESUME_CKPT}")"
    echo "${RESUME_CKPT}" > "${WORKDIR}/last_checkpoint"
    EXTRA_ARGS+=("--resume")
    echo ">>> RESUME from ${RESUME_CKPT}"
    echo ">>> Work dir (reused): ${WORKDIR}"
else
    WORKDIR="${PROJECT_ROOT}/work_dirs/${RUN_TYPE}/${MODEL}_${TRACK}_${DT}"
    mkdir -p "${WORKDIR}"
fi
mkdir -p "${PROJECT_ROOT}/logs"

echo "Model    : ${MODEL}"
echo "Track    : ${TRACK}"
echo "Run type : ${RUN_TYPE}"
echo "Config   : ${CONFIG}"
echo "Work dir : ${WORKDIR}"
echo "GPUs     : ${GPUS}"
echo "Job ID   : ${SLURM_JOB_ID:-local}"

bash "${MMSEG_ROOT}/tools/dist_train.sh" "${CONFIG}" "${GPUS}" \
    --work-dir "${WORKDIR}" "${EXTRA_ARGS[@]}"
