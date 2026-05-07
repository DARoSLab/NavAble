#!/bin/bash
# Unified local training entry point for all mmseg-based BLV models.
#
# Usage:
#   bash tools/blv/train.sh MODEL TRACK [--smoke] [--smoke-iters N] [--no-wandb] [--resume CKPT] [EXTRA_ARGS...]
#
#   MODEL           : mask2former | segformer | san
#   TRACK           : synthetic | real | real_synthetic
#   --smoke         : quick sanity check (default 500 iters, batch=1, no val)
#   --smoke-iters N : override smoke iteration count (implies --smoke)
#   --no-wandb      : disable Weights & Biases logging/visualization for this run
#   --resume CKPT   : true resume from CKPT — restores optimizer/scheduler/iter state.
#                     The checkpoint's parent directory is reused as WORKDIR so all
#                     existing logs and checkpoints stay in the same place.
#                     CKPT is written to last_checkpoint before mmseg is launched.
#
# Examples:
#   bash tools/blv/train.sh mask2former synthetic
#   bash tools/blv/train.sh segformer real
#   bash tools/blv/train.sh san real_synthetic --smoke --no-wandb
#   CUDA_VISIBLE_DEVICES=0 GPUS=1 bash tools/blv/train.sh mask2former real --smoke-iters 50
#   bash tools/blv/train.sh san synthetic --resume work_dirs/full/san_synthetic_.../iter_56500.pth
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=conda_utils.sh
source "${SCRIPT_DIR}/conda_utils.sh"
activate_conda_env

export BLV_PROJECT_ROOT="${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
# Do NOT default CUDA_VISIBLE_DEVICES here.  SLURM sets it to the allocated
# GPU IDs automatically; interactive sessions should see all available GPUs.
# Forcing "0,1" on a 1-GPU node causes RuntimeError: invalid device ordinal.
# Override explicitly if needed: CUDA_VISIBLE_DEVICES=0 GPUS=1 bash train.sh …
# Distributed rendezvous: pick a free port by default to avoid EADDRINUSE when
# MASTER_PORT is inherited from shell/session scripts.
# Set BLV_RESPECT_MASTER_PORT=1 to force usage of an inherited MASTER_PORT.
_pick_free_master_port() {
    python - <<'PY'
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

for _ in range(4096):
    port = random.randint(20000, 65535)
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
if [ -z "${MASTER_ADDR:-}" ]; then
    export MASTER_ADDR=127.0.0.1
fi
echo ">>> Distributed rendezvous: MASTER_ADDR=${MASTER_ADDR} PORT=${PORT}"

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
MMSEG_ROOT="${MMSEG_ROOT:-$(python -c 'import mmseg, os; print(os.path.dirname(os.path.dirname(mmseg.__file__)))')}"
# Auto-detect GPU count from what CUDA actually sees on this node/allocation.
GPUS="${GPUS:-$(python -c 'import torch; print(max(1, torch.cuda.device_count()))')}"

MODEL="${1:?Usage: train.sh MODEL TRACK [--smoke] [--smoke-iters N] [--no-wandb] [--resume CKPT] [EXTRA_ARGS...]}"
TRACK="${2:?Usage: train.sh MODEL TRACK [--smoke] [--smoke-iters N] [--no-wandb] [--resume CKPT] [EXTRA_ARGS...]}"

SMOKE=0
SMOKE_ITERS=500
NOWANDB=0
RESUME_CKPT=""
EXTRA_ARGS=()
ARGS=("${@:3}")
IDX=0
while [ ${IDX} -lt ${#ARGS[@]} ]; do
    arg="${ARGS[${IDX}]}"
    if [ "${arg}" = "--smoke" ]; then
        SMOKE=1
    elif [ "${arg}" = "--smoke-iters" ]; then
        SMOKE=1
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

CONFIG="${PROJECT_ROOT}/configs/blv/${MODEL}_finetune_${TRACK}.py"
if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: Config not found: ${CONFIG}"
    exit 1
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
        echo "Hint: either run tools/blv/download_checkpoints.py manually or copy from another machine."
        exit 1
    fi
fi

if [ "${SMOKE}" -eq 1 ]; then
    RUN_TYPE="smoke"
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
    echo ">>> SMOKE TEST MODE (${SMOKE_ITERS} iters, batch=1, val skipped)"
else
    RUN_TYPE="full"
fi

# --no-wandb: SHORT-TERM FALLBACK ONLY.
# draw=False is set here to keep training alive when W&B is unavailable or the
# env is broken.  The permanent fix for W&B crashes is in environment.yml
# (numpy 2.2.x + pandas 2.2.x from conda-forge, same ABI build).  Do NOT
# hard-code draw=False in any config file; re-enable W&B after env repair.
if [ "${NOWANDB}" -eq 1 ]; then
    export WANDB_MODE=disabled
    EXTRA_ARGS+=(
        "--cfg-options"
        "vis_backends=[dict(type='LocalVisBackend')]"
        "default_hooks.visualization.draw=False"
    )
    echo ">>> W&B DISABLED for this run (short-term fallback — see environment.yml for permanent fix)"
fi

# --resume: reuse the checkpoint's parent dir so all existing logs/ckpts stay
# together, write last_checkpoint so mmengine finds the right starting point,
# then pass --resume to dist_train.sh so optimizer/scheduler/iter are restored.
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

echo "Model    : ${MODEL}"
echo "Track    : ${TRACK}"
echo "Run type : ${RUN_TYPE}"
echo "Config   : ${CONFIG}"
echo "Work dir : ${WORKDIR}"
echo "GPUs     : ${GPUS}"

bash "${MMSEG_ROOT}/tools/dist_train.sh" "${CONFIG}" "${GPUS}" \
    --work-dir "${WORKDIR}" "${EXTRA_ARGS[@]}"

# After training, export checkpoint to checkpoints/finetuned/
if [ "${SMOKE}" -eq 0 ]; then
    FINETUNED_DIR="${PROJECT_ROOT}/checkpoints/finetuned/${MODEL}_${TRACK}_${DT}"

    BEST_CKPT="$(ls "${WORKDIR}"/best_mIoU*.pth 2>/dev/null | tail -1 || true)"

    if [ -z "${BEST_CKPT}" ] && [ -f "${WORKDIR}/last_checkpoint" ]; then
        LAST_REF="$(cat "${WORKDIR}/last_checkpoint")"
        if [ -f "${LAST_REF}" ]; then
            BEST_CKPT="${LAST_REF}"
        fi
    fi

    if [ -z "${BEST_CKPT}" ]; then
        BEST_CKPT="$(ls "${WORKDIR}"/iter_*.pth 2>/dev/null | sort -V | tail -1 || true)"
    fi

    if [ -n "${BEST_CKPT}" ]; then
        mkdir -p "${FINETUNED_DIR}"
        cp "${BEST_CKPT}" "${FINETUNED_DIR}/"
        echo ">>> Saved checkpoint to ${FINETUNED_DIR}/$(basename "${BEST_CKPT}")"
    else
        echo "WARNING: No checkpoint found in ${WORKDIR}"
    fi
fi
