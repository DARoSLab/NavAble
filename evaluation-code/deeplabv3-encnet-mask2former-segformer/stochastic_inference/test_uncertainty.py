# Copyright (c) OpenMMLab. All rights reserved.
"""MMSegmentation test script extended with two uncertainty modes:

  Mode A — Per-image pixel uncertainty  (--num-passes > 1)
      Runs N stochastic forward passes per image, averages predictions,
      and saves per-pixel variance / entropy / confidence maps.

  Mode B — Multi-run evaluation robustness  (--num-eval-runs > 1)
      Runs the full test-set evaluation N times, each time with a freshly
      sampled fixed weight-mask on the classifier.  Reports mean ± std for
      every evaluation metric.  Suitable for measuring metric stability and
      generating error-bar numbers for paper tables.

Both modes preserve the standard MMSegmentation evaluation pipeline.
"""
import argparse
import contextlib
import os
import os.path as osp
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner
from mmengine.runner.loops import _parse_losses


# ---------------------------------------------------------------------------
# Weight perturbation helpers
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _perturb_weights(layer: nn.Module, mask_ratio: float):
    """Zero out `mask_ratio` fraction of a layer's weights for one call.

    Supports nn.Conv2d and nn.Linear.  Weights (and bias) are fully
    restored on exit, even if an exception is raised.
    """
    w_orig = layer.weight.data.clone()
    b_orig = layer.bias.data.clone() if layer.bias is not None else None

    keep = torch.rand_like(w_orig) > mask_ratio
    layer.weight.data.mul_(keep.float())

    try:
        yield
    finally:
        layer.weight.data.copy_(w_orig)
        if b_orig is not None:
            layer.bias.data.copy_(b_orig)


def _apply_fixed_mask(layer: nn.Module, mask_ratio: float):
    """Sample and apply one fixed mask to *layer* weights.  Returns the backup."""
    w_orig = layer.weight.data.clone()
    b_orig = layer.bias.data.clone() if layer.bias is not None else None
    keep = torch.rand_like(w_orig) > mask_ratio
    layer.weight.data.mul_(keep.float())
    return w_orig, b_orig


def _restore_weights(layer: nn.Module, w_orig, b_orig):
    """Restore weights saved by _apply_fixed_mask."""
    layer.weight.data.copy_(w_orig)
    if b_orig is not None:
        layer.bias.data.copy_(b_orig)


# ---------------------------------------------------------------------------
# Automatic classifier-layer discovery
# ---------------------------------------------------------------------------

_KNOWN_CLASSIFIER_ATTRS = ['conv_seg', 'cls_embed', 'linear_pred', 'classifier']


def find_classifier_layer(model: nn.Module):
    """Return ``(layer, dotted_attr_path)`` for the final segmentation head.

    Priority:
    1. Well-known attribute names on decode_head (conv_seg, cls_embed, …)
    2. Last Conv2d / Linear in decode_head whose output dim matches num_classes
    """
    inner = model
    while hasattr(inner, 'module'):
        inner = inner.module

    if not hasattr(inner, 'decode_head'):
        raise RuntimeError(
            'Model has no decode_head attribute.  '
            'Only EncoderDecoder-family segmentors are supported.')

    head = inner.decode_head

    for attr in _KNOWN_CLASSIFIER_ATTRS:
        if hasattr(head, attr):
            layer = getattr(head, attr)
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                return layer, f'decode_head.{attr}'

    num_classes = getattr(head, 'num_classes', None) or getattr(
        head, 'out_channels', None)

    candidates = []
    for name, module in head.named_modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        out_dim = (module.out_channels
                   if isinstance(module, nn.Conv2d) else module.out_features)
        if num_classes is None or out_dim in (num_classes, num_classes + 1):
            candidates.append((name, module))

    if candidates:
        name, layer = candidates[-1]
        return layer, f'decode_head.{name}'

    raise RuntimeError(
        'Could not identify the final segmentation classifier layer.  '
        'Please inspect the model architecture manually.')


# ===========================================================================
# MODE A — Per-image pixel uncertainty
# ===========================================================================

class UncertaintyCollector:
    """Accumulates per-image uncertainty stats and optionally saves PNG maps."""

    def __init__(self, output_dir: str | None, save_per_pass: bool):
        self.output_dir = output_dir
        self.save_per_pass = save_per_pass
        self._img_idx = 0
        self._sum = dict(variance=0., entropy=0., confidence=0., mutual_info=0.)
        self._n_images = 0
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    @torch.no_grad()
    def collect(
        self,
        stacked_logits: torch.Tensor,  # (T, B, C, H, W)
        probs: torch.Tensor,           # (T, B, C, H, W)  softmax
        mean_probs: torch.Tensor,      # (B, C, H, W)
        batch_img_metas: list,
    ):
        T, B, C, H, W = probs.shape

        variance   = probs.var(dim=0).mean(dim=1)                               # (B,H,W)
        entropy    = -(mean_probs * torch.log(mean_probs.clamp(min=1e-8))).sum(dim=1)
        confidence = mean_probs.max(dim=1).values
        mean_ent   = -(probs * torch.log(probs.clamp(min=1e-8))).sum(dim=2).mean(dim=0)
        mutual_inf = entropy - mean_ent

        for key, val in zip(
            ['variance', 'entropy', 'confidence', 'mutual_info'],
            [variance, entropy, confidence, mutual_inf]
        ):
            self._sum[key] += val.mean().item() * B
        self._n_images += B

        for i in range(B):
            meta     = batch_img_metas[i] if i < len(batch_img_metas) else {}
            img_path = meta.get('img_path', f'img_{self._img_idx:06d}')
            stem     = Path(img_path).stem
            stats = {
                'variance':    variance[i].cpu().numpy(),
                'entropy':     entropy[i].cpu().numpy(),
                'confidence':  confidence[i].cpu().numpy(),
                'mutual_info': mutual_inf[i].cpu().numpy(),
            }
            if self.save_per_pass:
                stats['per_pass_preds'] = (
                    stacked_logits[:, i].argmax(dim=1).cpu().numpy())
            if self.output_dir:
                self._save_maps(stem, stats)
            self._img_idx += 1

    def _save_maps(self, stem: str, stats: dict):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print_log('matplotlib not found — skipping uncertainty map saving.',
                      logger='current')
            return
        cmap = dict(variance='hot', entropy='plasma',
                    confidence='viridis', mutual_info='magma')
        for key, cm in cmap.items():
            arr = stats.get(key)
            if arr is None:
                continue
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(arr, cmap=cm, interpolation='nearest')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f'{key}  —  {stem}', fontsize=9)
            ax.axis('off')
            fig.savefig(osp.join(self.output_dir, f'{stem}_{key}.png'),
                        bbox_inches='tight', dpi=100)
            plt.close(fig)
        if 'per_pass_preds' in stats:
            np.save(osp.join(self.output_dir, f'{stem}_per_pass_preds.npy'),
                    stats['per_pass_preds'])

    def print_summary(self):
        if self._n_images == 0:
            return
        n = self._n_images
        lines = [f'\n[Pixel-uncertainty summary — {n} images]']
        for key, total in self._sum.items():
            lines.append(f'  {key:<16}: {total / n:.6f}')
        print_log('\n'.join(lines), logger='current')


def patch_model_for_uncertainty(
    model: nn.Module,
    classifier_layer: nn.Module,
    num_passes: int,
    mask_ratio: float,
    collector: UncertaintyCollector,
):
    """Replace model.encode_decode with a stochastic N-pass version.

    Averaged log-probabilities are returned so that argmax (and therefore all
    standard IoU / accuracy metrics) remain correct.
    """
    target = model
    while hasattr(target, 'module'):
        target = target.module

    if not hasattr(target, 'encode_decode'):
        raise RuntimeError(
            'encode_decode not found on the segmentor.  '
            'Only EncoderDecoder-based models are supported.')

    orig_encode_decode = target.encode_decode

    def stochastic_encode_decode(inputs, batch_img_metas):
        all_logits = []
        for _ in range(num_passes):
            with _perturb_weights(classifier_layer, mask_ratio):
                logits = orig_encode_decode(inputs, batch_img_metas)
            all_logits.append(logits.detach())

        stacked   = torch.stack(all_logits, dim=0)    # (T, B, C, H, W)
        probs     = F.softmax(stacked, dim=2)
        mean_prob = probs.mean(dim=0)                  # (B, C, H, W)

        collector.collect(stacked, probs, mean_prob, batch_img_metas)
        return torch.log(mean_prob.clamp(min=1e-8))

    target.encode_decode = stochastic_encode_decode


# ===========================================================================
# MODE B — Multi-run evaluation robustness
# ===========================================================================

def patch_test_loop_for_multi_run(
    runner: Runner,
    classifier_layer: nn.Module,
    num_eval_runs: int,
    mask_ratio: float,
) -> list:
    """Patch runner.test_loop.run() to execute N independent stochastic runs.

    Each run:
      1. Applies a fresh random weight mask to *classifier_layer*.
      2. Iterates the full test dataloader once.
      3. Computes evaluation metrics via the runner's evaluator.
      4. Restores the original weights.

    Returns a list reference that will be populated with per-run metric dicts
    when runner.test() is eventually called.
    """
    loop         = runner.test_loop
    all_metrics: list[dict] = []

    def _run_one_epoch():
        """Reproduce the inner body of TestLoop.run() for one epoch."""
        runner.call_hook('before_test_epoch')
        runner.model.eval()
        loop.test_loss.clear()
        for idx, data_batch in enumerate(loop.dataloader):
            loop.run_iter(idx, data_batch)
        metrics = loop.evaluator.evaluate(len(loop.dataloader.dataset))
        if loop.test_loss:
            metrics.update(_parse_losses(loop.test_loss, 'test'))
        runner.call_hook('after_test_epoch', metrics=metrics)
        return metrics

    def multi_run():
        runner.call_hook('before_test')

        w_clean = classifier_layer.weight.data.clone()
        b_clean = (classifier_layer.bias.data.clone()
                   if classifier_layer.bias is not None else None)

        for run_idx in range(num_eval_runs):
            print_log(
                f'\n{"=" * 60}\n'
                f'  Stochastic Eval  Run {run_idx + 1} / {num_eval_runs}'
                f'  (mask_ratio={mask_ratio})\n'
                f'{"=" * 60}',
                logger='current')

            w_backup, b_backup = _apply_fixed_mask(
                classifier_layer, mask_ratio)

            metrics = _run_one_epoch()
            all_metrics.append(metrics)

            _restore_weights(classifier_layer, w_backup, b_backup)

        # Sanity-check: weights must be identical to pre-run state.
        assert torch.equal(classifier_layer.weight.data, w_clean), \
            'classifier weights were not correctly restored after multi-run loop'

        runner.call_hook('after_test')
        return all_metrics[-1]  # runner.test() expects a return value

    loop.run = multi_run
    return all_metrics


# ---------------------------------------------------------------------------
# Robustness statistics reporting
# ---------------------------------------------------------------------------

def _gather_metric_keys(all_metrics: list[dict]) -> tuple[list, list]:
    """Split metric keys into summary (no dot) and per-class (has dot)."""
    all_keys = dict.fromkeys(k for m in all_metrics for k in m)
    summary  = [k for k in all_keys if '.' not in k]
    perclass = [k for k in all_keys if '.' in k]
    return summary, perclass


def _compute_stats(all_metrics: list[dict], keys: list[str]):
    """Return {key: (values_array, mean, std)} for the given keys."""
    stats = {}
    for k in keys:
        vals = np.array([m[k] for m in all_metrics if k in m], dtype=float)
        stats[k] = (vals, float(vals.mean()), float(vals.std()))
    return stats


def report_robustness_statistics(
    all_metrics: list[dict],
    output_dir: str | None,
    mask_ratio: float,
):
    """Print a formatted table of per-run metrics and mean ± std summary."""
    if not all_metrics:
        print_log('[Robustness] No metrics collected.', logger='current')
        return

    n = len(all_metrics)
    summary_keys, perclass_keys = _gather_metric_keys(all_metrics)
    summary_stats  = _compute_stats(all_metrics, summary_keys)
    perclass_stats = _compute_stats(all_metrics, perclass_keys)

    lines = []
    sep   = '─' * 72

    lines.append(f'\n{sep}')
    lines.append(f'  EVALUATION ROBUSTNESS REPORT')
    lines.append(f'  {n} independent runs  |  mask_ratio = {mask_ratio}')
    lines.append(sep)

    # ---- Summary metrics table --------------------------------------------
    if summary_stats:
        # Header
        hdr = f'  {"Metric":<20}' + ''.join(f'  {"Run " + str(i + 1):>8}'
                                             for i in range(n))
        hdr += f'  {"Mean":>10}  {"Std":>8}  {"CV%":>6}'
        lines.append(hdr)
        lines.append('  ' + '─' * (len(hdr) - 2))

        for key in sorted(summary_stats):
            vals, mean, std = summary_stats[key]
            cv = (std / mean * 100) if mean != 0 else float('nan')
            row = f'  {key:<20}' + ''.join(f'  {v:>8.4f}' for v in vals)
            row += f'  {mean:>10.4f}  {std:>8.4f}  {cv:>5.2f}%'
            lines.append(row)

    # ---- Paper-table format -----------------------------------------------
    lines.append(f'\n  Paper-table format  (mean ± std)  [×100 for %%]')
    lines.append('  ' + '─' * 50)
    for key in sorted(summary_stats):
        vals, mean, std = summary_stats[key]
        lines.append(f'  {key:<20}  {mean * 100:6.2f} ± {std * 100:.2f}')

    # ---- Per-class IoU condensed ------------------------------------------
    if perclass_stats:
        lines.append(f'\n  Per-class breakdown  (mean ± std)  [×100 for %%]')
        lines.append('  ' + '─' * 50)
        for key in sorted(perclass_stats):
            vals, mean, std = perclass_stats[key]
            lines.append(f'  {key:<30}  {mean * 100:6.2f} ± {std * 100:.2f}')

    lines.append(sep)
    full_report = '\n'.join(lines)
    print_log(full_report, logger='current')

    # ---- Optional file output ---------------------------------------------
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Human-readable summary
        txt_path = osp.join(output_dir, 'robustness_summary.txt')
        with open(txt_path, 'w') as f:
            f.write(full_report + '\n')
        print_log(f'[Robustness] Report saved → {txt_path}', logger='current')

        # CSV: one row per run + summary rows
        csv_path = osp.join(output_dir, 'robustness_per_run.csv')
        all_keys  = summary_keys + perclass_keys
        with open(csv_path, 'w') as f:
            f.write('run,' + ','.join(all_keys) + '\n')
            for i, m in enumerate(all_metrics):
                row = str(i + 1)
                for k in all_keys:
                    row += f',{m.get(k, float("nan")):.6f}'
                f.write(row + '\n')
            # mean and std rows
            for label, fn in [('mean', np.mean), ('std', np.std)]:
                row = label
                for k in all_keys:
                    vals = [m[k] for m in all_metrics if k in m]
                    row += f',{fn(vals):.6f}' if vals else ',nan'
                f.write(row + '\n')
        print_log(f'[Robustness] Per-run CSV saved → {csv_path}',
                  logger='current')


# ===========================================================================
# Argument parsing
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMSeg test with stochastic uncertainty / robustness estimation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    # ---- Standard test.py arguments ----------------------------------------
    parser.add_argument('config',     help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='directory to dump evaluation metric results as json')
    parser.add_argument(
        '--out', type=str,
        help='directory to save output predictions for offline evaluation')
    parser.add_argument('--show', action='store_true',
                        help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory to save painted images '
             '(auto-nested under work_dir/timestamp/)')
    parser.add_argument('--wait-time', type=float, default=2,
                        help='display interval in seconds (default: 2)')
    parser.add_argument(
        '--cfg-options', nargs='+', action=DictAction,
        help='override config key=value pairs')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'], default='none',
        help='job launcher')
    parser.add_argument('--tta', action='store_true',
                        help='test time augmentation')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)

    # ---- Shared uncertainty arguments --------------------------------------
    parser.add_argument(
        '--mask-ratio', type=float, default=0.05, metavar='R',
        help='fraction of classifier weights zeroed per perturbation '
             '(default: 0.05)')

    # ---- Mode A: per-image pixel uncertainty --------------------------------
    parser.add_argument(
        '--num-passes', type=int, default=1, metavar='N',
        help='stochastic forward passes per image for pixel-level uncertainty '
             '(default: 1 = disabled).  Set >1 to enable Mode A.')
    parser.add_argument(
        '--uncertainty-dir', type=str, default=None, metavar='DIR',
        help='save per-pixel uncertainty heatmaps (PNG) here '
             '(Mode A only; requires --num-passes > 1)')
    parser.add_argument(
        '--save-per-pass', action='store_true',
        help='also save per-pass argmax prediction maps as .npy '
             '(Mode A; requires --uncertainty-dir)')

    # ---- Mode B: multi-run evaluation robustness ----------------------------
    parser.add_argument(
        '--num-eval-runs', type=int, default=1, metavar='N',
        help='number of independent stochastic evaluation runs for metric '
             'robustness analysis (default: 1 = disabled).  Set >1 to enable '
             'Mode B.  Each run uses a freshly sampled fixed weight mask and '
             'evaluates the complete test set once.')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


# ---------------------------------------------------------------------------
# Visualization hook helper  (identical to tools/test.py)
# ---------------------------------------------------------------------------

def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            cfg.visualizer['save_dir'] = args.show_dir
    else:
        raise RuntimeError(
            'VisualizationHook must be in default_hooks.  '
            "See: visualization=dict(type='VisualizationHook')")
    return cfg


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = parse_args()

    # ------------------------------------------------------------------ #
    # Config loading — identical to tools/test.py                         #
    # ------------------------------------------------------------------ #
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join(
            './work_dirs', osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.tta_model.module = cfg.model
        cfg.model = cfg.tta_model

    if args.out is not None:
        cfg.test_evaluator['output_dir'] = args.out
        cfg.test_evaluator['keep_results'] = True

    # ------------------------------------------------------------------ #
    # Build runner                                                         #
    # ------------------------------------------------------------------ #
    runner = Runner.from_cfg(cfg)

    # ------------------------------------------------------------------ #
    # Discover classifier layer (shared by both modes)                    #
    # ------------------------------------------------------------------ #
    classifier_layer, layer_path = find_classifier_layer(runner.model)
    print_log(
        f'[Uncertainty] classifier layer : {layer_path}  '
        f'({type(classifier_layer).__name__},  '
        f'weight {list(classifier_layer.weight.shape)})',
        logger='current')
    print_log(
        f'[Uncertainty] mask_ratio = {args.mask_ratio}',
        logger='current')

    # ------------------------------------------------------------------ #
    # Mode dispatch                                                        #
    # ------------------------------------------------------------------ #

    if args.num_eval_runs > 1:
        # ── MODE B: Multi-run evaluation robustness ───────────────────── #
        if args.num_passes > 1:
            print_log(
                '[Robustness] --num-passes is ignored in multi-run mode '
                '(--num-eval-runs > 1).  Each run uses a single forward pass '
                'with a fixed weight mask.',
                logger='current')

        print_log(
            f'[Robustness] num_eval_runs = {args.num_eval_runs}  '
            f'(Mode B: full-dataset evaluation repeated per run)',
            logger='current')

        all_run_metrics = patch_test_loop_for_multi_run(
            runner=runner,
            classifier_layer=classifier_layer,
            num_eval_runs=args.num_eval_runs,
            mask_ratio=args.mask_ratio,
        )

        runner.test()

        report_robustness_statistics(
            all_metrics=all_run_metrics,
            output_dir=args.uncertainty_dir,
            mask_ratio=args.mask_ratio,
        )

    else:
        # ── MODE A: Per-image pixel uncertainty ───────────────────────── #
        num_passes = args.num_passes

        if num_passes < 2:
            print_log(
                '[Uncertainty] --num-passes=1: running standard evaluation '
                'with a single fixed weight perturbation (no averaging).',
                logger='current')
            # Still apply one perturbation so the script behaves predictably,
            # but skip the UncertaintyCollector overhead.
            w_backup, b_backup = _apply_fixed_mask(
                classifier_layer, args.mask_ratio)
            runner.test()
            _restore_weights(classifier_layer, w_backup, b_backup)
        else:
            print_log(
                f'[Uncertainty] num_passes = {num_passes}  '
                f'(Mode A: per-image stochastic inference)',
                logger='current')

            if args.save_per_pass and args.uncertainty_dir is None:
                print_log(
                    '[Uncertainty] --save-per-pass requires --uncertainty-dir; '
                    'per-pass maps will not be saved.',
                    logger='current')

            collector = UncertaintyCollector(
                output_dir=args.uncertainty_dir,
                save_per_pass=(args.save_per_pass
                               and args.uncertainty_dir is not None),
            )
            patch_model_for_uncertainty(
                model=runner.model,
                classifier_layer=classifier_layer,
                num_passes=num_passes,
                mask_ratio=args.mask_ratio,
                collector=collector,
            )
            runner.test()
            collector.print_summary()


if __name__ == '__main__':
    main()
