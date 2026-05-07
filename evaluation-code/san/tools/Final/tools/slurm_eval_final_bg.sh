#!/bin/bash
#SBATCH --job-name=blv_final_eval_bg
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --exclude=gpu026
#SBATCH --requeue
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#
# Real_final/test BG-INCLUSIVE evaluation, METRICS ONLY (no visualizations).
# - Reuses the same Final_results_and_checkpoints config + checkpoint
# - Test pipeline injects RemapSegLabel(255 -> 11) so bg pixels are kept as
#   class 11 (instead of ignored)
# - Test dataloader metainfo extended to 12 classes (fg + 'background')
# - Test evaluator overrides:
#       num_classes = 12
#       excluded_class_indices = [10]              (turnstile, kept excluded)
#       segformer_extra_channel_fg_only = False    (no bg-channel stripping)
#       segformer_conf_threshold = 0.0             (no thresholding)
#       synthesize_bg_channel = True               (for M2F/SAN: append
#                                                   bg = 1 - max(fg) so the
#                                                   12-class argmax sees bg)
# - W&B run name: test-bg-<arch>-final-<letter>-<data>
#
# Usage:
#   sbatch Final_results_and_checkpoints/tools/slurm_eval_final_bg.sh MODEL CONFIG CHECKPOINT
#
# Example:
#   sbatch Final_results_and_checkpoints/tools/slurm_eval_final_bg.sh \
#     mask2former C \
#     Final_results_and_checkpoints/work_dirs/mask2former_finalC_real_synth01_<DT>/best_mIoU_iter_<N>.pth
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

# GPU sanity check — fail fast on broken nodes so SLURM requeues.
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

MODEL="${1:?Usage: slurm_eval_final_bg.sh MODEL CONFIG CHECKPOINT}"
CONFIG="${2:?Usage: slurm_eval_final_bg.sh MODEL CONFIG CHECKPOINT}"
CKPT="${3:?Usage: slurm_eval_final_bg.sh MODEL CONFIG CHECKPOINT}"

case "${CONFIG}" in
    A) CONFIG_NAME="finalA_real";          DATA_TAG="real" ;;
    B) CONFIG_NAME="finalB_real_opensrc";  DATA_TAG="real-opensrc" ;;
    C) CONFIG_NAME="finalC_real_synth01";  DATA_TAG="real-synth01" ;;
    D) CONFIG_NAME="finalD_real_synth02";  DATA_TAG="real-synth02" ;;
    *) echo "ERROR: CONFIG must be A, B, C, or D"; exit 1 ;;
esac

CFG_LOWER="$(echo "${CONFIG}" | tr 'A-Z' 'a-z')"
WANDB_RUN_NAME="test-bg-${MODEL}-final-${CFG_LOWER}-${DATA_TAG}"

CONFIG_PATH="${PROJECT_ROOT}/Final_results_and_checkpoints/configs/${MODEL}_${CONFIG_NAME}.py"
[ -f "${CONFIG_PATH}" ] || { echo "ERROR: Config not found: ${CONFIG_PATH}"; exit 1; }
[ -f "${CKPT}" ] || { echo "ERROR: Checkpoint not found: ${CKPT}"; exit 1; }
CKPT="$(realpath "${CKPT}")"

DT="$(date '+%M-%H-%d-%m-%Y')"
WORK_DIR="${PROJECT_ROOT}/Final_results_and_checkpoints/eval_results_bg/${MODEL}_${CONFIG_NAME}_${DT}"
mkdir -p "${WORK_DIR}"
mkdir -p "${PROJECT_ROOT}/logs"

# Generate per-job test config:
# - Replace WandbVisBackend with WandbMetricsOnlyBackend
# - Inject RemapSegLabel(255 -> 11) before PackSegInputs in test_pipeline
# - Override test_dataloader.dataset.metainfo with 12-class list
# - Override test_evaluator for bg-inclusive metric (num_classes=12,
#   synthesize_bg_channel=True, no thresholding, no fg-only stripping)
# - draw=False on visualization hook
TEST_CFG="${WORK_DIR}/test_config.py"
python - <<PY
import os
from mmengine import Config

cfg = Config.fromfile("${CONFIG_PATH}")

# Custom imports: include the metrics-only W&B backend
imports = list(cfg.custom_imports.get("imports", []))
extra = "blv_pipeline.mmseg_plugins.visualization"
if extra not in imports:
    imports.append(extra)
cfg.custom_imports = dict(imports=imports, allow_failed_imports=False)

# Override vis_backends: swap WandbVisBackend -> WandbMetricsOnlyBackend
new_backends = []
for b in cfg.visualizer.vis_backends:
    btype = b.get("type")
    if btype == "WandbVisBackend":
        new_backends.append(dict(
            type="WandbMetricsOnlyBackend",
            init_kwargs=dict(
                project="blv-seg-final",
                name="${WANDB_RUN_NAME}",
                tags=["test", "config${CONFIG}", "${MODEL}", "bg-inclusive"],
            ),
        ))
    else:
        new_backends.append(dict(b))
cfg.visualizer.vis_backends = new_backends

cfg.default_hooks.visualization = dict(type="SegVisualizationHook", draw=False)

# ---- Test pipeline: inject RemapSegLabel(255 -> 11) before PackSegInputs ----
# After dataset.reduce_zero_label=True + LoadAnnotations: bg pixels are 255.
# We remap them to class 11 so the metric sees bg as a real class.
old_pipeline = list(cfg.test_dataloader.dataset.pipeline)
new_pipeline = []
inserted = False
for step in old_pipeline:
    step_dict = dict(step)
    if step_dict.get("type") == "PackSegInputs" and not inserted:
        new_pipeline.append(dict(type="RemapSegLabel", src_label=255, dst_label=11))
        inserted = True
    new_pipeline.append(step_dict)
if not inserted:
    raise RuntimeError("PackSegInputs not found in test_pipeline; cannot inject RemapSegLabel")
cfg.test_dataloader.dataset.pipeline = new_pipeline

# ---- Test dataloader: extend dataset metainfo to 12 classes ----
fg_classes = (
    'elevator', 'elevator_button', 'door_button', 'crosswalk',
    'pedestrian_signal', 'aps_button', 'bus_stop', 'bus_stop_sign',
    'handrail', 'escalator', 'turnstile',
)
fg_palette = [
    (214, 39, 40), (255, 127, 14), (148, 103, 189), (44, 160, 44),
    (23, 190, 207), (227, 119, 194), (188, 189, 34), (127, 127, 127),
    (140, 86, 75), (31, 119, 180), (174, 199, 232),
]
all_classes_12 = fg_classes + ('background',)
all_palette_12 = fg_palette + [(0, 0, 0)]
cfg.test_dataloader.dataset.metainfo = dict(
    classes=all_classes_12,
    palette=all_palette_12,
)

# ---- Test evaluator: bg-inclusive overrides ----
cfg.test_evaluator = dict(
    type="BLVMetric",
    iou_metrics=["mIoU", "mAP50-95", "Prec", "Rec"],
    zero_shot_remap=False,
    num_classes=12,
    ignore_index=255,
    segformer_conf_threshold=0.0,
    mask_fg_conf_threshold=0.0,
    segformer_extra_channel_fg_only=False,
    synthesize_bg_channel=True,
    output_metrics_path=None,
    excluded_class_indices=[10],
)

cfg.dump("${TEST_CFG}")
print(f"Wrote {os.path.basename('${TEST_CFG}')}")
PY

MMSEG_ROOT="${MMSEG_ROOT:-$(python -c 'import mmseg, os; print(os.path.dirname(os.path.dirname(mmseg.__file__)))')}"

echo "Model       : ${MODEL}"
echo "Config      : ${CONFIG_NAME}  (BG-INCLUSIVE eval, num_classes=12, turnstile excluded)"
echo "Checkpoint  : ${CKPT}"
echo "Work dir    : ${WORK_DIR}"
echo "W&B run     : ${WANDB_RUN_NAME}"
echo "Vis         : DISABLED (draw=False)"
echo "Job ID      : ${SLURM_JOB_ID:-local}"

bash "${MMSEG_ROOT}/tools/dist_test.sh" "${TEST_CFG}" "${CKPT}" 1 \
    --work-dir "${WORK_DIR}" \
    --cfg-options \
        test_evaluator.output_metrics_path="${WORK_DIR}/metrics.json"

echo "DONE. Metrics: ${WORK_DIR}/metrics.json"
