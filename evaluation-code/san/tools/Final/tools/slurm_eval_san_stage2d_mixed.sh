#!/bin/bash
#SBATCH --job-name=blv_san_2d_mix
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
# One-off: evaluate the OLD Stage-2D mixed SAN checkpoint
#   (real_v2 + opensrc + synth_v2 — full 180K synth)
# on data/real_final/test under the honest BLVMetric.
#
# Architecture matches Final-SAN configs (BLVSideAdapterCLIPHead, num_classes=11,
# same vocabulary), so we reuse san_finalA_real.py as the config — its
# test_dataloader already points at data/real_final/test, and we just override
# the test checkpoint at the dist_test.sh CLI.
#
# NOTE: best_mIoU_iter_20000.pth was selected by the pre-2026-05-02 buggy
# BLVMetric. Model weights are unaffected; only the selection criterion was
# inflated. The honest test number we get here is unbiased.
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

WANDB_RUN_NAME="test-san-stage2d-mixed-on-real-final"
CONFIG_PATH="${PROJECT_ROOT}/Final_results_and_checkpoints/configs/san_finalA_real.py"
CKPT="${PROJECT_ROOT}/checkpoints/finetuned/san_mixed_46-08-29-04-2026/best_mIoU_iter_20000.pth"

[ -f "${CONFIG_PATH}" ] || { echo "ERROR: Config not found: ${CONFIG_PATH}"; exit 1; }
[ -f "${CKPT}" ] || { echo "ERROR: Checkpoint not found: ${CKPT}"; exit 1; }
CKPT="$(realpath "${CKPT}")"

DT="$(date '+%M-%H-%d-%m-%Y')"
WORK_DIR="${PROJECT_ROOT}/Final_results_and_checkpoints/eval_results/san_stage2d_mixed_on_real_final_${DT}"
mkdir -p "${WORK_DIR}"
mkdir -p "${PROJECT_ROOT}/logs"

# Generate per-job test config: WandbMetricsOnlyBackend, viz off.
TEST_CFG="${WORK_DIR}/test_config.py"
python - <<PY
import os
from mmengine import Config

cfg = Config.fromfile("${CONFIG_PATH}")

imports = list(cfg.custom_imports.get("imports", []))
extra = "blv_pipeline.mmseg_plugins.visualization"
if extra not in imports:
    imports.append(extra)
cfg.custom_imports = dict(imports=imports, allow_failed_imports=False)

new_backends = []
for b in cfg.visualizer.vis_backends:
    btype = b.get("type")
    if btype == "WandbVisBackend":
        new_backends.append(dict(
            type="WandbMetricsOnlyBackend",
            init_kwargs=dict(
                project="blv-seg-final",
                name="${WANDB_RUN_NAME}",
                tags=["test", "stage2d-mixed", "san", "cross-eval"],
            ),
        ))
    else:
        new_backends.append(dict(b))
cfg.visualizer.vis_backends = new_backends

cfg.default_hooks.visualization = dict(
    type="SegVisualizationHook",
    draw=False,
)

cfg.dump("${TEST_CFG}")
print(f"Wrote {os.path.basename('${TEST_CFG}')}")
PY

MMSEG_ROOT="${MMSEG_ROOT:-$(python -c 'import mmseg, os; print(os.path.dirname(os.path.dirname(mmseg.__file__)))')}"

echo "Run name    : ${WANDB_RUN_NAME}"
echo "Config      : ${CONFIG_PATH}  (architecture from Final-SAN, test_dataloader → real_final/test)"
echo "Checkpoint  : ${CKPT}  (Stage-2D mixed: real_v2 + opensrc + synth_v2)"
echo "Work dir    : ${WORK_DIR}"
echo "Vis         : DISABLED"
echo "Job ID      : ${SLURM_JOB_ID:-local}"

bash "${MMSEG_ROOT}/tools/dist_test.sh" "${TEST_CFG}" "${CKPT}" 1 \
    --work-dir "${WORK_DIR}" \
    --cfg-options \
        test_evaluator.output_metrics_path="${WORK_DIR}/metrics.json"

echo "DONE. Metrics: ${WORK_DIR}/metrics.json"
