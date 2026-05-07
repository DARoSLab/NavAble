#!/bin/bash
# Zero-shot (Track A) evaluation on the real val split.
#
# Usage:
#   bash tools/blv/run_eval.sh [MODEL] [CHECKPOINT] [--smoke] [--no-wandb]
#
#   MODEL      : mask2former (default) | segformer | san
#   CHECKPOINT : path to pretrained checkpoint (defaults to checkpoints/pretrained/{model}.pth)
#   --smoke    : quick sanity check (first 20 images)
#   --no-wandb : disable Weights & Biases logging/visualization for this eval
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=conda_utils.sh
source "${SCRIPT_DIR}/conda_utils.sh"
activate_conda_env

export BLV_PROJECT_ROOT="${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
MMSEG_ROOT="${MMSEG_ROOT:-$(python -c 'import mmseg, os; print(os.path.dirname(os.path.dirname(mmseg.__file__)))')}"

MODEL="${1:-mask2former}"
SMOKE=0
NOWANDB=0
EXTRA_ARGS=()

shift || true
CHECKPOINT_ARG=""
for arg in "$@"; do
    if [ "${arg}" = "--smoke" ]; then
        SMOKE=1
    elif [ "${arg}" = "--no-wandb" ]; then
        NOWANDB=1
    elif [ -z "${CHECKPOINT_ARG}" ] && [ -f "${arg}" ]; then
        CHECKPOINT_ARG="${arg}"
    else
        EXTRA_ARGS+=("${arg}")
    fi
done

if [ -z "${CHECKPOINT_ARG}" ]; then
    case "${MODEL}" in
        mask2former)
            CHECKPOINT_ARG="${PROJECT_ROOT}/checkpoints/pretrained/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640.pth"
            ;;
        segformer)
            CHECKPOINT_ARG="${PROJECT_ROOT}/checkpoints/pretrained/segformer_mit-b5_640x640_160k_ade20k_20210801_121243-41d2845b.pth"
            ;;
        san)
            CHECKPOINT_ARG="${PROJECT_ROOT}/checkpoints/pretrained/san-vit-b16_20230906-fd0a7684.pth"
            ;;
        *)
            echo "ERROR: Unknown model '${MODEL}'. Choose: mask2former | segformer | san"
            exit 1
            ;;
    esac
fi

if [ ! -f "${CHECKPOINT_ARG}" ]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT_ARG}"
    echo "  Run 'python tools/blv/download_checkpoints.py' first."
    exit 1
fi

CONFIG="${PROJECT_ROOT}/configs/blv/${MODEL}_eval_blv.py"

DT="$(date '+%M-%H-%d-%m-%Y')"

if [ "${SMOKE}" -eq 1 ]; then
    RUN_TYPE="smoke"
    EXTRA_ARGS+=("--cfg-options" "test_dataloader.dataset.indices=list(range(20))")
    echo ">>> SMOKE TEST MODE (20 images)"
else
    RUN_TYPE="full"
fi

if [ "${NOWANDB}" -eq 1 ]; then
    export WANDB_MODE=disabled
    EXTRA_ARGS+=(
        "--cfg-options"
        "vis_backends=[dict(type='LocalVisBackend')]"
        "default_hooks.visualization.draw=False"
    )
    echo ">>> W&B DISABLED for this eval"
fi

WORKDIR="${PROJECT_ROOT}/work_dirs/${RUN_TYPE}/track_A_${MODEL}_${DT}"
mkdir -p "${WORKDIR}"

echo "Model     : ${MODEL}"
echo "Checkpoint: ${CHECKPOINT_ARG}"
echo "Work dir  : ${WORKDIR}"

python "${MMSEG_ROOT}/tools/test.py" "${CONFIG}" "${CHECKPOINT_ARG}" \
    --work-dir "${WORKDIR}" "${EXTRA_ARGS[@]}"
