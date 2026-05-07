#!/usr/bin/env python
"""Run Track A/B/C evaluation and assemble the final benchmark table.

Supports multiple mmseg-based models.  Pass ``--model`` to select which
model's configs and checkpoints to evaluate.  The resulting CSV is
*appended* to so that successive runs with different ``--model`` values
accumulate into one Table-III-style table.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from blv_pipeline.constants import MMSEG_MODELS
from blv_pipeline.runtime import build_subprocess_env, resolve_mmseg_root

CONFIG_DIR = PROJECT_ROOT / 'configs' / 'blv'

DISPLAY_NAMES = {
    'mask2former': 'Mask2Former',
    'segformer': 'SegFormer',
    'san': 'SAN-ViT-B16',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--model',
        default='mask2former',
        choices=list(MMSEG_MODELS),
        help='Model to benchmark.',
    )
    parser.add_argument('--track-a-ckpt', default=None, help='Checkpoint for Track A zero-shot (optional).')
    parser.add_argument('--track-a-config', default=None, help='Config for Track A (auto-resolved if omitted).')
    parser.add_argument('--track-b-workdir', default=None, help='Training workdir for Track B.')
    parser.add_argument('--track-b-config', default=None, help='Config for Track B (auto-resolved if omitted).')
    parser.add_argument('--track-c-workdir', default=None, help='Training workdir for Track C.')
    parser.add_argument('--track-c-config', default=None, help='Config for Track C (auto-resolved if omitted).')
    parser.add_argument('--mmseg-root', default=None, help='Override mmseg repo root.')
    parser.add_argument(
        '--output-csv',
        default=str(PROJECT_ROOT / 'work_dirs' / 'benchmark_results.csv'),
        help='Destination CSV path.',
    )
    parser.add_argument('--wandb-project', default='blv-seg', help='W&B project name.')
    parser.add_argument('--wandb-run-name', default=None, help='W&B run name (auto-generated if omitted).')
    parser.add_argument('--no-wandb', action='store_true', help='Skip W&B logging.')
    return parser.parse_args()


def _resolve_config(explicit: str | None, model: str, suffix: str) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    return CONFIG_DIR / f'{model}_{suffix}.py'


def find_best_checkpoint(workdir: Path) -> Path:
    best = sorted(workdir.glob('best_mIoU*.pth'))
    if best:
        return max(best, key=lambda path: path.stat().st_mtime)
    checkpoints = sorted(workdir.glob('*.pth'))
    if checkpoints:
        return max(checkpoints, key=lambda path: path.stat().st_mtime)
    raise FileNotFoundError(f'No checkpoints found under {workdir}')


def run_test(
    mmseg_root: Path,
    config_path: Path,
    checkpoint_path: Path,
    workdir: Path,
    metrics_path: Path,
) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(mmseg_root / 'tools' / 'test.py'),
        str(config_path),
        str(checkpoint_path),
        '--work-dir',
        str(workdir),
        '--cfg-options',
        f'test_evaluator.output_metrics_path={metrics_path}',
    ]
    print('Running:', ' '.join(command))
    subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        env=build_subprocess_env(),
        check=True,
    )
    with metrics_path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def metrics_row(track_name: str, payload: dict) -> dict:
    summary = payload['summary']
    return {
        f'{track_name} Prec.': summary['Prec'],
        f'{track_name} Rec.': summary['Rec'],
        f'{track_name} mAP50-95': summary['mAP50-95'],
        f'{track_name} mIoU': summary['mIoU'],
    }


def main() -> None:
    args = parse_args()
    model = args.model
    mmseg_root = resolve_mmseg_root(args.mmseg_root)
    display_name = DISPLAY_NAMES.get(model, model)

    row: dict = {}

    # --- Track A (optional) ---
    track_a_metrics = None
    if args.track_a_ckpt:
        track_a_config = _resolve_config(args.track_a_config, model, 'eval_blv')
        track_a_metrics = run_test(
            mmseg_root=mmseg_root,
            config_path=track_a_config,
            checkpoint_path=Path(args.track_a_ckpt).expanduser().resolve(),
            workdir=PROJECT_ROOT / 'work_dirs' / f'blv_track_A_{model}',
            metrics_path=PROJECT_ROOT / 'work_dirs' / f'blv_track_A_{model}' / 'metrics.json',
        )
        row.update(metrics_row('Track A', track_a_metrics))

    # --- Track B ---
    track_b_metrics = None
    if args.track_b_workdir:
        track_b_config = _resolve_config(args.track_b_config, model, 'finetune_real')
        track_b_ckpt = find_best_checkpoint(Path(args.track_b_workdir).expanduser().resolve())
        track_b_metrics = run_test(
            mmseg_root=mmseg_root,
            config_path=track_b_config,
            checkpoint_path=track_b_ckpt,
            workdir=PROJECT_ROOT / 'work_dirs' / f'blv_track_B_{model}_eval',
            metrics_path=PROJECT_ROOT / 'work_dirs' / f'blv_track_B_{model}_eval' / 'metrics.json',
        )
        row.update(metrics_row('Track B', track_b_metrics))

    # --- Track C ---
    track_c_metrics = None
    if args.track_c_workdir:
        track_c_config = _resolve_config(args.track_c_config, model, 'finetune_real_synthetic')
        track_c_ckpt = find_best_checkpoint(Path(args.track_c_workdir).expanduser().resolve())
        track_c_metrics = run_test(
            mmseg_root=mmseg_root,
            config_path=track_c_config,
            checkpoint_path=track_c_ckpt,
            workdir=PROJECT_ROOT / 'work_dirs' / f'blv_track_C_{model}_eval',
            metrics_path=PROJECT_ROOT / 'work_dirs' / f'blv_track_C_{model}_eval' / 'metrics.json',
        )
        row.update(metrics_row('Track C', track_c_metrics))

    # --- Deltas ---
    if track_b_metrics and track_a_metrics:
        row['Delta mIoU (B-A)'] = round(
            track_b_metrics['summary']['mIoU'] - track_a_metrics['summary']['mIoU'], 2,
        )
    if track_c_metrics and track_b_metrics:
        row['Delta mIoU (C-B)'] = round(
            track_c_metrics['summary']['mIoU'] - track_b_metrics['summary']['mIoU'], 2,
        )
    if track_c_metrics and track_a_metrics:
        row['Delta mIoU (C-A)'] = round(
            track_c_metrics['summary']['mIoU'] - track_a_metrics['summary']['mIoU'], 2,
        )

    if not row:
        print('No tracks evaluated. Pass at least one of --track-a-ckpt, --track-b-workdir, --track-c-workdir.')
        return

    new_df = pd.DataFrame([row], index=[display_name])
    output_csv = Path(args.output_csv).expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if output_csv.exists():
        existing = pd.read_csv(output_csv, index_col=0)
        existing.loc[display_name] = new_df.loc[display_name]
        combined = existing
    else:
        combined = new_df

    combined.to_csv(output_csv)
    print(combined.to_string())
    print(f'Saved benchmark CSV to {output_csv}')

    if not args.no_wandb:
        try:
            import wandb
        except ImportError:
            print('wandb is not installed; skipping benchmark logging.')
        else:
            run_name = args.wandb_run_name or f'benchmark-{model}'
            run = wandb.init(project=args.wandb_project, name=run_name)
            wandb.log({
                'benchmark_table': wandb.Table(
                    dataframe=combined.reset_index().rename(columns={'index': 'Method'}),
                ),
            })
            run.finish()


if __name__ == '__main__':
    main()
