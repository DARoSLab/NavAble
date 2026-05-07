#!/bin/bash
#SBATCH --job-name=blv_san_zs_viz
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
# SAN zero-shot eval on real_final/test under the honest BLVMetric.
# - No fine-tuning: COCO-Stuff pretrained CLIP backbone with BLV text prompts
# - 5% sample (~74 imgs) of vis saved to visualizations/san-zeroshot-real-final/
# - Metrics dumped to metrics.json in same dir
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

# GPU sanity check — fail fast on broken nodes.
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

RUN_NAME="san-zeroshot-real-final"
CONFIG_PATH="${PROJECT_ROOT}/Final_results_and_checkpoints/configs/san_zeroshot_real_final.py"
PRETRAINED_CKPT="${PROJECT_ROOT}/checkpoints/pretrained/san-vit-b16_20230906-fd0a7684.pth"
[ -f "${CONFIG_PATH}" ]      || { echo "ERROR: Config not found: ${CONFIG_PATH}"; exit 1; }
[ -f "${PRETRAINED_CKPT}" ]  || { echo "ERROR: Checkpoint not found: ${PRETRAINED_CKPT}"; exit 1; }
PRETRAINED_CKPT="$(realpath "${PRETRAINED_CKPT}")"

VIZ_DIR="${PROJECT_ROOT}/Final_results_and_checkpoints/visualizations/${RUN_NAME}"
mkdir -p "${VIZ_DIR}"
mkdir -p "${PROJECT_ROOT}/logs"

# Generate temp config:
# - LocalVisBackend only (no W&B)
# - IntervalSegVisualizationHook with interval=20 → ~5% of 1,482 ≈ 74 PNGs
TEST_CFG="${VIZ_DIR}/test_config.py"
python - <<PY
from mmengine import Config
cfg = Config.fromfile("${CONFIG_PATH}")

imports = list(cfg.custom_imports.get("imports", []))
extra = "blv_pipeline.mmseg_plugins.visualization"
if extra not in imports:
    imports.append(extra)
cfg.custom_imports = dict(imports=imports, allow_failed_imports=False)

cfg.visualizer.vis_backends = [dict(type="LocalVisBackend")]

cfg.default_hooks.visualization = dict(
    type="IntervalSegVisualizationHook",
    draw=True,
    interval=20,
)

cfg.dump("${TEST_CFG}")
print(f"Wrote ${TEST_CFG}")
PY

MMSEG_ROOT="${MMSEG_ROOT:-$(python -c 'import mmseg, os; print(os.path.dirname(os.path.dirname(mmseg.__file__)))')}"

echo "Run name   : ${RUN_NAME}"
echo "Config     : ${CONFIG_PATH}"
echo "Checkpoint : ${PRETRAINED_CKPT} (COCO-Stuff pretrained, no BLV fine-tuning)"
echo "Viz dir    : ${VIZ_DIR}/vis_data/vis_image/"
echo "Job ID     : ${SLURM_JOB_ID:-local}"

bash "${MMSEG_ROOT}/tools/dist_test.sh" "${TEST_CFG}" "${PRETRAINED_CKPT}" 1 \
    --work-dir "${VIZ_DIR}" \
    --show-dir "${VIZ_DIR}" \
    --cfg-options \
        test_evaluator.output_metrics_path="${VIZ_DIR}/metrics.json"

echo "DONE. Metrics: ${VIZ_DIR}/metrics.json"
echo "DONE. Vis:     ${VIZ_DIR}/vis_data/vis_image/"
