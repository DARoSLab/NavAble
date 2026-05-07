"""Run test.py twice on a Stage-1 checkpoint, once per source domain.

The Stage-1 *_finetune_real.py configs use ConcatDataset(real_v2, opensrc) for
the test_dataloader. This helper plucks the two sub-datasets out programmatically
and runs one test pass each, so the BLVMetric numbers are reported separately
for the real and opensrc domains.

Usage:
    python tools/blv/eval_stage1_split.py \
        --config configs/blv/mask2former_finetune_real.py \
        --checkpoint work_dirs/.../best_mIoU_iter_XXXXX.pth \
        --work-dir work_dirs/eval_split/<run_name>

Outputs two subdirs under --work-dir:
    real_only/    — metrics on real_v2/test  (1,482 images)
    opensrc_only/ — metrics on opensrc/test  (4,507 images)
"""

from __future__ import annotations

import argparse
import copy
import os

from mmengine.config import Config
from mmengine.runner import Runner


def run_one_domain(cfg_path: str, checkpoint: str, domain_idx: int, out_dir: str) -> None:
    cfg = Config.fromfile(cfg_path)
    cfg.load_from = checkpoint
    cfg.work_dir = out_dir

    test_dl = cfg.test_dataloader
    if test_dl.dataset.get('type') != 'ConcatDataset':
        raise RuntimeError(
            f"Expected test_dataloader.dataset.type == 'ConcatDataset', "
            f"got {test_dl.dataset.get('type')}. "
            f"This script is only meant for Stage-1 *_finetune_real.py configs."
        )
    sub_datasets = test_dl.dataset.datasets
    if domain_idx >= len(sub_datasets):
        raise IndexError(f"domain_idx={domain_idx} but ConcatDataset has {len(sub_datasets)} sub-datasets")

    cfg.test_dataloader.dataset = copy.deepcopy(sub_datasets[domain_idx])

    # Disable W&B for split eval — local logs only.
    cfg.vis_backends = [dict(type='LocalVisBackend')]
    cfg.visualizer.vis_backends = cfg.vis_backends

    os.makedirs(out_dir, exist_ok=True)
    runner = Runner.from_cfg(cfg)
    runner.test()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, help='Path to a *_finetune_real.py config')
    p.add_argument('--checkpoint', required=True, help='Path to .pth checkpoint to evaluate')
    p.add_argument('--work-dir', required=True, help='Parent dir for the two output subdirs')
    args = p.parse_args()

    for domain_idx, domain_name in ((0, 'real_only'), (1, 'opensrc_only')):
        sub_work = os.path.join(args.work_dir, domain_name)
        print(f'\n========================  {domain_name}  ========================')
        print(f'  config:     {args.config}')
        print(f'  checkpoint: {args.checkpoint}')
        print(f'  work_dir:   {sub_work}')
        run_one_domain(args.config, args.checkpoint, domain_idx, sub_work)


if __name__ == '__main__':
    main()
