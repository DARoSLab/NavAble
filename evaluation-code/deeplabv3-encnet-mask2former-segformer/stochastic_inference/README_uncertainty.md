# Stochastic Classifier Perturbation for Uncertainty & Robustness Evaluation

> An extension of the standard MMSegmentation `tools/test.py` pipeline that adds
> **stochastic weight masking** to the final segmentation classifier during
> inference.  Two complementary evaluation modes are provided:
> **pixel-wise uncertainty estimation** and
> **multi-run metric robustness analysis**.

---

## Table of Contents

1. [Overview](#overview)
2. [Motivation](#motivation)
3. [Method](#method)
   - [Perturbation Mechanism](#perturbation-mechanism)
   - [Why Only the Final Classifier Layer?](#why-only-the-final-classifier-layer)
   - [Relation to MC Dropout](#relation-to-mc-dropout)
4. [Uncertainty Metrics](#uncertainty-metrics)
5. [Robustness Statistics](#robustness-statistics)
6. [Mode A — Pixel-Wise Uncertainty](#mode-a--pixel-wise-uncertainty)
7. [Mode B — Multi-Run Robustness Evaluation](#mode-b--multi-run-robustness-evaluation)
8. [Supported Architectures](#supported-architectures)
9. [Implementation Details](#implementation-details)
10. [Code Structure](#code-structure)
11. [Installation & Requirements](#installation--requirements)
12. [CLI Reference](#cli-reference)
13. [Usage Examples](#usage-examples)
14. [Output Files](#output-files)
15. [Reproducibility Notes](#reproducibility-notes)
16. [Limitations](#limitations)
17. [Future Work](#future-work)

---

## Overview

`tools/test_uncertainty.py` extends the standard MMSegmentation test pipeline
with **stochastic classifier perturbation**: at inference time, a random binary
mask is applied to the weights of the final segmentation head, zeroing out
approximately `r` (default 5%) of them.  The rest of the evaluation pipeline
(data loading, pre-processing, metrics, logging) remains completely unchanged.

The script exposes two independent evaluation modes:

| Mode | Flag | What it measures |
|---|---|---|
| **A — Pixel Uncertainty** | `--num-passes N` | Per-pixel predictive variance, entropy, and epistemic uncertainty across N stochastic passes per image |
| **B — Metric Robustness** | `--num-eval-runs N` | Stability of evaluation metrics (mIoU, mAcc, …) across N full-dataset runs under fixed-per-run perturbations |

Both modes preserve all standard MMSegmentation evaluation metrics and hooks.

---

## Motivation

Trained segmentation models are rarely evaluated beyond a single deterministic
forward pass.  This leaves two important questions unanswered:

1. **Where is the model uncertain?**
   Pixel-level confidence maps reveal whether the model is unsure about specific
   regions (boundaries, rare classes, out-of-distribution textures) — information
   that a single argmax prediction discards entirely.

2. **How stable are the reported metrics?**
   A model evaluated once with mIoU = 43.2% might achieve 42.8% or 43.6% under
   slightly different conditions.  Reporting a single number without error bars is
   misleading, especially for paper comparisons where differences of < 1 pp are
   routinely cited as improvements.

This script addresses both questions with **minimal overhead** and **zero
architectural changes** to the model.

---

## Method

### Perturbation Mechanism

Let **W** ∈ ℝ^(C_out × C_in × 1 × 1) be the weight tensor of the final
classification layer (a 1×1 convolution or a linear layer).  For each
stochastic forward pass *t*, a random binary mask **M**_t is sampled:

```
M_t[i] ~ Bernoulli(1 − r)    for each weight i
W_t     = W ⊙ M_t            (element-wise product)
```

where *r* is the **mask ratio** (default `r = 0.05`).

The model is then evaluated with **W**_t in place of **W**.  The original
weights are restored exactly after each pass using a `try / finally` context
manager, guaranteeing correctness even in the presence of exceptions or
keyboard interrupts.

The perturbation is **multiplicative and sparse**: on average, 95% of weights
are untouched, so predictions remain close to the unperturbed model while still
exhibiting measurable stochastic variation.

### Why Only the Final Classifier Layer?

Targeting the classifier head exclusively is a deliberate design choice with
three practical advantages:

**1. Computational cost is negligible.**

The classifier layer for common backbones is tiny compared to the rest of
the network:

| Model | Backbone params | Classifier params | Ratio |
|---|---|---|---|
| DeepLabV3+ (ResNet-50) | ~25.6 M | ~10 K (512 → 21) | 0.04% |
| SegFormer-B5 | ~82 M | ~12 K (768 → 19) | 0.01% |
| EncNet (ResNet-50) | ~35 M | ~9 K (512 → 19) | 0.03% |
| Mask2Former (Swin-L) | ~197 M | ~5 K (256 → 19+1) | 0.003% |

Masking and restoring these weights requires only a few element-wise
operations per pass — negligible relative to the backbone forward pass.

**2. The classifier is the sole semantic decision boundary.**

Regardless of architecture, every segmentation model converges to a point
where dense backbone features are projected into per-class logits by a single
linear map.  All semantic ambiguity that is not resolved by the backbone must
pass through this layer.  Perturbing it therefore produces semantically
meaningful variation.

**3. Works on any pretrained model without modification.**

No dropout layers need to be inserted, no retraining is required.  The script
discovers the classifier layer automatically at runtime.

### Relation to MC Dropout

This approach is **related to but distinct from** Monte-Carlo Dropout (Gal &
Ghahramani, 2016):

| Property | MC Dropout | This work |
|---|---|---|
| Perturbation target | Intermediate feature activations | Final weight matrix |
| Requires model modification | Yes (dropout layers must exist) | No |
| Works on arbitrary pretrained models | Only if trained with dropout | Yes |
| Perturbs the full computational graph | Yes | No (backbone unchanged) |
| Interpretable perturbation | Activation noise | Weight ablation |
| Computational overhead | Full N forward passes | 1 full pass + N cheap projections* |

\* The backbone (`extract_feat`) runs once; only the classification head is
re-evaluated N times when combined with feature caching (not yet implemented —
see [Future Work](#future-work)).  In the current implementation, the full
forward pass is repeated N times.

---

## Uncertainty Metrics

For **Mode A** (pixel uncertainty), the following maps are computed per image
given T stochastic predictions {**p**₁, …, **p**_T} where **p**_t ∈ Δ^C is
the softmax probability vector at each pixel:

### Mean Prediction

$$\bar{p} = \frac{1}{T} \sum_{t=1}^{T} p_t$$

The averaged probability is used for the final segmentation prediction.
Its argmax equals the argmax of the mean softmax, preserving metric correctness.

### Prediction Variance

$$\text{Var}(x) = \frac{1}{C} \sum_{c=1}^{C} \text{Var}_t \left[ p_t^{(c)}(x) \right]$$

Mean per-class variance across passes.  High values indicate pixels where
the classifier is sensitive to small weight perturbations.

### Predictive Entropy

$$H[\bar{p}](x) = -\sum_{c=1}^{C} \bar{p}^{(c)}(x) \log \bar{p}^{(c)}(x)$$

Entropy of the mean prediction.  Captures **total uncertainty** (both
aleatoric and epistemic).  Maximised at 1/C uniform and zero at hard
predictions.

### Confidence Map

$$\text{Conf}(x) = \max_c \; \bar{p}^{(c)}(x)$$

The maximum probability assigned to any class.  High confidence implies the
model strongly favours one class even under perturbation.

### Mutual Information (Epistemic Uncertainty)

$$\text{MI}(x) = H[\bar{p}](x) - \frac{1}{T} \sum_{t=1}^{T} H[p_t](x)$$

$$= \underbrace{H\!\left[\mathbb{E}[p]\right]}_{\text{total uncertainty}} \;-\; \underbrace{\mathbb{E}\!\left[H[p]\right]}_{\text{aleatoric uncertainty}}$$

The mutual information between predictions and parameters measures **epistemic
(model) uncertainty** — how much the model's belief changes across different
weight configurations.  Regions with high MI but low aleatoric uncertainty
represent places where the model could improve with more data.

---

## Robustness Statistics

For **Mode B** (multi-run evaluation), N independent runs produce metric
vectors {**s**₁, …, **s**_N} where each **s**_n is a dict of evaluation scores
(mIoU, mAcc, aAcc, per-class IoUs, …):

| Statistic | Formula | Interpretation |
|---|---|---|
| Mean | μ = (1/N) Σ s_n | Expected performance under small classifier perturbations |
| Std | σ = std({s_n}) | Sensitivity of the metric to classifier weight ablation |
| CV% | σ / μ × 100 | Coefficient of variation; model-size-independent stability measure |
| Error bars | μ ± σ | Ready-to-use values for paper tables |

A **low standard deviation** indicates that the metric is robust to small
changes in the classifier — a desirable property that pure deterministic
evaluation cannot quantify.

---

## Mode A — Pixel-Wise Uncertainty

Activated when `--num-passes N` with N > 1.

**Per image**, the script runs N stochastic forward passes.  Each pass applies
a freshly sampled random mask to the classifier weights.  The N output
probability maps are averaged and returned to the standard evaluation pipeline
as if they were a single prediction.  Pixel-level uncertainty statistics are
accumulated and saved.

```
For each image:
  for t = 1 … T:
    M_t ~ Bernoulli(1 − r)   over classifier weights
    W_t = W ⊙ M_t
    p_t = softmax(f(x ; backbone, W_t))
    W   ← restored
  p̄ = mean(p₁, …, p_T)
  ŷ = argmax(p̄)             ← used for mIoU / mAcc
  save: variance, entropy, confidence, mutual_info maps
```

The return value fed to the evaluator is `log(p̄)`, which is equivalent to `p̄`
under argmax — standard IoU and accuracy metrics are unaffected.

---

## Mode B — Multi-Run Robustness Evaluation

Activated when `--num-eval-runs N` with N > 1.

**Per run**, a single fixed random mask is sampled once and held constant for
the entire test-set evaluation.  The full MMSegmentation evaluation loop runs
to completion, producing one complete set of metrics.  This is repeated N
times with independent masks.

```
for n = 1 … N:
  M_n ~ Bernoulli(1 − r)    over classifier weights (fixed for this run)
  W_n = W ⊙ M_n
  s_n = Eval(f(· ; backbone, W_n), D_test)
  W   ← restored
report: {s_n}, mean(s_n), std(s_n)
```

This mode answers: *"How different would the reported numbers be if we had
used a slightly different model?"*  It quantifies **metric variance** that is
invisible to single-run evaluation.

> **Note:** In Mode B the classifier weight mask is **fixed for the entire
> dataset** within each run.  This is different from Mode A where the mask is
> **resampled per image**.

---

## Supported Architectures

The script automatically discovers the final segmentation classifier layer by
scanning the model's `decode_head` for known attribute names, falling back to
a scan of all `Conv2d` / `Linear` modules if necessary.

| Model | Classifier Layer | Type | Auto-detected |
|---|---|---|---|
| EncNet | `decode_head.conv_seg` | `nn.Conv2d` | ✓ |
| DeepLabV3+ | `decode_head.conv_seg` | `nn.Conv2d` | ✓ |
| SegFormer | `decode_head.conv_seg` | `nn.Conv2d` | ✓ |
| FCN | `decode_head.conv_seg` | `nn.Conv2d` | ✓ |
| PSPNet | `decode_head.conv_seg` | `nn.Conv2d` | ✓ |
| UPerNet | `decode_head.conv_seg` | `nn.Conv2d` | ✓ |
| Mask2Former | `decode_head.cls_embed` | `nn.Linear` | ✓ |
| MaskFormer | `decode_head.cls_embed` | `nn.Linear` | ✓ |
| Custom heads | last Conv2d/Linear matching `num_classes` | either | ✓ (fallback) |

The discovery priority is:

1. `conv_seg` → `cls_embed` → `linear_pred` → `classifier` (by attribute name)
2. Last `nn.Conv2d` or `nn.Linear` with output dimension equal to `num_classes`
   or `num_classes + 1` (for heads that include a background/void class)

A log message at startup confirms which layer was selected, including its
weight shape, so misdetections are immediately visible.

---

## Implementation Details

### Weight Perturbation Context Manager

```python
@contextlib.contextmanager
def _perturb_weights(layer: nn.Module, mask_ratio: float):
    w_orig = layer.weight.data.clone()
    b_orig = layer.bias.data.clone() if layer.bias is not None else None
    keep   = torch.rand_like(w_orig) > mask_ratio
    layer.weight.data.mul_(keep.float())
    try:
        yield
    finally:
        layer.weight.data.copy_(w_orig)
        if b_orig is not None:
            layer.bias.data.copy_(b_orig)
```

The `try / finally` block guarantees that weights are restored even if the
forward pass raises an exception.  In-place operations (`mul_`, `copy_`) avoid
unnecessary memory allocation.

### Mode A — `encode_decode` Patching

The segmentor's `encode_decode` method is replaced at the instance level
(without touching the class) after the runner is built.  This intercepts
logit computation for both `whole_inference` and `slide_inference` without
any change to data loading, post-processing, or the evaluator.

### Mode B — `test_loop.run` Patching

The runner's `test_loop.run` method is replaced with a closure that
replicates the exact body of `mmengine.runner.loops.TestLoop.run` N times,
clearing `test_loss` and calling all expected hooks
(`before_test_epoch`, `after_test_epoch`) on each iteration.  The outer
lifecycle hooks (`before_test`, `after_test`, `before_run`, `after_run`) fire
exactly once, preserving checkpoint loading and logger behaviour.

---

## Code Structure

```
tools/test_uncertainty.py
│
├── _perturb_weights()               Context manager: apply + restore weight mask
├── _apply_fixed_mask()              Apply a fixed mask and return weight backups
├── _restore_weights()               Restore from backups saved by _apply_fixed_mask
│
├── find_classifier_layer()          Auto-detect final segmentation head layer
│
├── ── Mode A ──────────────────────────────────────────────────────────────
├── UncertaintyCollector             Accumulates & saves per-pixel maps
│   ├── .collect()                   Compute variance/entropy/confidence/MI
│   ├── ._save_maps()                Save PNG heatmaps via matplotlib
│   └── .print_summary()             Log mean uncertainty stats at end of run
├── patch_model_for_uncertainty()    Monkey-patch encode_decode for N-pass inference
│
├── ── Mode B ──────────────────────────────────────────────────────────────
├── patch_test_loop_for_multi_run()  Patch test_loop.run for N independent eval runs
├── report_robustness_statistics()   Print table + paper format + save CSV/TXT
│
├── parse_args()                     CLI argument parser (superset of test.py)
├── trigger_visualization_hook()     Identical to tools/test.py helper
└── main()                           Mode dispatch and runner orchestration
```

---

## Installation & Requirements

This script requires a working MMSegmentation installation.  No additional
dependencies beyond the standard MMSeg stack are needed for core functionality.

```
# Core requirements (already present in any MMSeg environment)
torch >= 1.9
mmengine >= 0.10
mmcv >= 2.0
mmsegmentation

# Optional (for saving uncertainty heatmaps — Mode A only)
matplotlib
```

---

## CLI Reference

All standard `tools/test.py` arguments are supported unchanged.  Additional
arguments are grouped below.

### Standard Arguments (inherited from `tools/test.py`)

| Argument | Type | Description |
|---|---|---|
| `config` | positional | Path to test config file |
| `checkpoint` | positional | Path to checkpoint `.pth` file |
| `--work-dir DIR` | str | Directory for metric JSON output |
| `--out DIR` | str | Directory for offline prediction output |
| `--show` | flag | Display predictions interactively |
| `--show-dir DIR` | str | Save visualized predictions |
| `--wait-time T` | float | Display interval in seconds (default: 2) |
| `--cfg-options` | key=val… | Override config values |
| `--launcher` | choice | `none` / `pytorch` / `slurm` / `mpi` |
| `--tta` | flag | Enable test-time augmentation |

### Shared Uncertainty Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--mask-ratio R` | float | `0.05` | Fraction of classifier weights zeroed per perturbation |

### Mode A — Pixel-Wise Uncertainty

| Argument | Type | Default | Description |
|---|---|---|---|
| `--num-passes N` | int | `1` | Stochastic forward passes per image. Set `>1` to enable Mode A |
| `--uncertainty-dir DIR` | str | `None` | Save PNG heatmaps here |
| `--save-per-pass` | flag | off | Also save per-pass argmax prediction maps as `.npy` (requires `--uncertainty-dir`) |

### Mode B — Multi-Run Robustness Evaluation

| Argument | Type | Default | Description |
|---|---|---|---|
| `--num-eval-runs N` | int | `1` | Independent evaluation runs. Set `>1` to enable Mode B |

> When `--num-eval-runs > 1`, any `--num-passes` value is silently ignored and
> a single forward pass is used per image within each run.

---

## Usage Examples

### Standard Evaluation (no uncertainty)

Identical to `tools/test.py` — use this as a baseline to verify correctness:

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=0 \
python tools/test_uncertainty.py \
    configs/[project_name]/real/encnet.py \
    work_dirs/[project_name]/real/encnet_ft_real/best_fg_mIoU_iter_68000.pth
```

---

### Mode A — Pixel-Wise Uncertainty Estimation

Run 10 stochastic passes per image and save all four uncertainty maps:

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=0 \
python tools/test_uncertainty.py \
    configs/[project_name]/real/encnet.py \
    work_dirs/[project_name]/real/encnet_ft_real/best_fg_mIoU_iter_68000.pth \
    --num-passes 10 \
    --mask-ratio 0.05 \
    --uncertainty-dir results/encnet_uncertainty
```

Also save per-pass prediction arrays (for post-hoc analysis):

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=0 \
python tools/test_uncertainty.py \
    configs/[project_name]/real/segformer.py \
    work_dirs/[project_name]/real/segformer_ft/best_mIoU_iter_40000.pth \
    --num-passes 20 \
    --mask-ratio 0.03 \
    --uncertainty-dir results/segformer_uncertainty \
    --save-per-pass
```

---

### Mode B — Multi-Run Robustness Evaluation

Run 5 independent evaluations and report mean ± std for all metrics:

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=0 \
python tools/test_uncertainty.py \
    configs/[project_name]/real/encnet.py \
    work_dirs/[project_name]/real/encnet_ft_real/best_fg_mIoU_iter_68000.pth \
    --num-eval-runs 5 \
    --mask-ratio 0.05 \
    --uncertainty-dir results/encnet_robustness
```

Comparing two models for a paper table:

```bash
for MODEL in encnet deeplab segformer; do
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=0 \
    python tools/test_uncertainty.py \
        configs/[project_name]/real/${MODEL}.py \
        work_dirs/[project_name]/real/${MODEL}_ft/best_mIoU.pth \
        --num-eval-runs 5 \
        --mask-ratio 0.05 \
        --uncertainty-dir results/${MODEL}_robustness
done
```

---

### Slide Inference (large images)

Works transparently — `--cfg-options` overrides test mode as usual:

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=0 \
python tools/test_uncertainty.py \
    configs/[project_name]/real/encnet.py \
    work_dirs/[project_name]/real/encnet_ft_real/best_fg_mIoU_iter_68000.pth \
    --num-eval-runs 5 \
    --cfg-options model.test_cfg.mode=slide \
                  model.test_cfg.stride="(256,256)" \
                  model.test_cfg.crop_size="(512,512)"
```

---

## Output Files

### Mode A Output (`--uncertainty-dir`)

For each test image with stem `<name>`, four PNG heatmaps are saved:

| File | Content | Colormap |
|---|---|---|
| `<name>_variance.png` | Mean per-class prediction variance | `hot` |
| `<name>_entropy.png` | Predictive entropy H[p̄] | `plasma` |
| `<name>_confidence.png` | Max mean probability | `viridis` |
| `<name>_mutual_info.png` | Epistemic uncertainty (MI) | `magma` |

If `--save-per-pass` is set:

| File | Content |
|---|---|
| `<name>_per_pass_preds.npy` | `(T, H, W)` int16 argmax prediction per pass |

A summary over all images is logged at the end of the run:

```
[Pixel-uncertainty summary — 500 images]
  variance        : 0.002341
  entropy         : 0.843217
  confidence      : 0.784523
  mutual_info     : 0.013412
```

---

### Mode B Output (`--uncertainty-dir`)

| File | Content |
|---|---|
| `robustness_summary.txt` | Full formatted report (same as terminal output) |
| `robustness_per_run.csv` | Per-run rows + `mean` and `std` rows at the end |

Example CSV:

```csv
run,mIoU,mAcc,aAcc,IoU.background,IoU.road
1,0.432100,0.551200,0.901300,0.812000,0.603000
2,0.441500,0.558700,0.903100,0.814500,0.617200
3,0.437800,0.554300,0.902000,0.810200,0.608900
4,0.429600,0.549800,0.899800,0.809100,0.597400
5,0.435200,0.553100,0.901700,0.811800,0.605100
mean,0.435240,0.553420,0.901580,0.811520,0.606320
std,0.004284,0.003167,0.001155,0.001924,0.007109
```

Terminal report format:

```
────────────────────────────────────────────────────────────────────────
  EVALUATION ROBUSTNESS REPORT
  5 independent runs  |  mask_ratio = 0.05
────────────────────────────────────────────────────────────────────────
  Metric                 Run 1     Run 2     Run 3     Run 4     Run 5        Mean       Std     CV%
  ──────────────────────────────────────────────────────────────────────────────────────────────────
  aAcc                  0.9013    0.9031    0.9020    0.8998    0.9017      0.9016    0.0011   0.13%
  mAcc                  0.5512    0.5587    0.5543    0.5498    0.5531      0.5534    0.0032   0.58%
  mIoU                  0.4321    0.4415    0.4378    0.4296    0.4352      0.4352    0.0043   0.99%

  Paper-table format  (mean ± std)  [×100 for %]
  ──────────────────────────────────────────────────
  aAcc                  90.16 ± 0.11
  mAcc                  55.34 ± 0.32
  mIoU                  43.52 ± 0.43
────────────────────────────────────────────────────────────────────────
```

---

## Reproducibility Notes

- The masks are sampled with `torch.rand_like`, which uses the current
  PyTorch global random state.  To make results reproducible across runs,
  set a seed before calling the script:

  ```bash
  python -c "import torch; torch.manual_seed(42)" && \
  python tools/test_uncertainty.py ...
  ```

  Alternatively, pass `--cfg-options randomness.seed=42` if the config
  supports it.

- For **Mode B**, each run uses an **independently sampled** mask.  The
  reported statistics characterise the distribution of metric values under
  random 5% ablations, not a specific fixed ablation.  Results will vary
  between executions unless the seed is fixed.

- For **Mode A**, masks are re-sampled for every image **and** every pass.
  The pixel-level statistics are aggregates and are stable across runs for
  large enough test sets; per-image maps will vary unless the seed is fixed.

- Setting `PYTHONHASHSEED=0` and using `torch.backends.cudnn.deterministic=True`
  (via `--cfg-options`) ensures full determinism for a fixed mask sequence.

---

## Limitations

1. **Full backbone re-execution per pass (Mode A).**
   The current implementation calls `encode_decode` N times, which includes
   the backbone.  For large models (Swin-L, ViT-L), this can be slow.
   A feature-caching optimisation (run backbone once, re-project N times) is
   planned but not yet implemented.

2. **EncoderDecoder family only.**
   The script patches `encode_decode`, which is defined in
   `mmseg.models.segmentors.EncoderDecoder`.  Models that override the full
   prediction pipeline (e.g., custom segmentors not inheriting from
   `EncoderDecoder`) may not be automatically compatible.

3. **Single-GPU only (tested).**
   The monkey-patching approach works with `torch.nn.DataParallel` (single
   node, multi-GPU) since the `.module` unwrapping is handled.  Proper
   `DistributedDataParallel` support across multiple nodes has not been tested.

4. **Perturbation targets weights, not learned representations.**
   This method probes sensitivity to the final decision boundary.  It does not
   measure uncertainty arising from ambiguous backbone features.  Regions
   where the backbone produces noisy or low-quality features may not be
   reflected in the uncertainty maps.

5. **Mask ratio is fixed and global.**
   All output channels are masked at the same rate.  Class-specific masking
   (e.g., a higher rate for rare classes) is not supported.

6. **Mode B error bars reflect perturbation sensitivity, not data variance.**
   The standard deviation across runs characterises how sensitive the metric
   is to small classifier weight ablations.  It is not equivalent to
   cross-validation variance or bootstrap confidence intervals.

---

## Future Work

- [ ] **Feature caching (Mode A):** Run backbone once per image; apply N
  random projections directly on cached features.  Would reduce Mode A
  overhead to ~1 × backbone + N × head passes.
- [ ] **Class-conditioned masking:** Apply higher mask rates to channels
  corresponding to underrepresented or difficult classes.
- [ ] **Structured masks:** Block-wise or channel-wise masks to probe
  specific output neuron groups.
- [ ] **Bootstrap confidence intervals (Mode B):** In addition to
  perturbation-based error bars, add support for paired bootstrap over the
  test set for metric significance testing.
- [ ] **Uncertainty calibration metrics:** ECE (Expected Calibration Error)
  and reliability diagrams computed from the stochastic passes.
- [ ] **Multi-GPU Mode B support:** Properly synchronise per-run mask
  sampling across DDP workers.
- [ ] **Integration with TTA:** Combine stochastic weight masking with
  test-time augmentation for combined uncertainty estimates.

---

## Citation

If you use this script in your research, please cite the base MMSegmentation
framework:

```bibtex
@misc{mmseg2020,
    title={{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark},
    author={MMSegmentation Contributors},
    howpublished={\url{https://github.com/open-mmlab/mmsegmentation}},
    year={2020}
}
```

---

## License

This file is part of a custom extension to MMSegmentation and follows the
same Apache 2.0 license as the base repository.
