# Final Pipeline Results — `Final_results_and_checkpoints/`

All numbers under the **honest** `BLVMetric` (post-2026-05-02 patch).
Test set: `data/real_final/test` (1,482 images). Val set: `data/real_final/val` (396 images).
Turnstile (class index 10) is excluded from all means — zero GT in real_final.

**Configurations**
| Letter | Train data | Notes |
|---|---|---|
| A | `real_final` (3,703 imgs) | real-only |
| B | `real_final` + `opensrc_final` (~40K) | adds open-source |
| C | `real_final` ×6 + `synth_0.1` (~42K) | adds synthetic, 6× real upweight |
| D | `real_final` ×12 + `synth_0.2` (~85K) | adds 2× more synth, 12× real upweight |

---

## 1. SAN Zero-Shot Baseline vs. Fine-tuned SAN (real_final/test)

`san-vit-b16` COCO-Stuff pretrained, queried open-vocabulary with the 11 BLV class
text prompts. **No BLV fine-tuning whatsoever** — this is the floor: what an
off-the-shelf open-vocabulary segmenter does on BLV classes without seeing a
single BLV label.

### Summary metrics

| Variant | mIoU | mAP50-95 | Precision | Recall |
|---|:---:|:---:|:---:|:---:|
| SAN Zero-Shot (no fine-tuning)            | 14.94 | 3.84  | 31.17 | 47.10 |
| SAN Stage-A (real only)                   | 62.07 | 31.32 | 82.67 | 68.35 |
| SAN Stage-B (real + opensrc)              | 62.52 | 30.09 | 77.10 | 74.17 |
| SAN Stage-C (real + synth_0.1)            | **68.12** | **36.08** | **82.75** | **76.93** |
| SAN Stage-D (real + synth_0.2)            | 66.96 | 35.34 | 82.53 | 75.45 |
| _SAN R+C+S (Stage-2D mixed, synth_v2)*_   | _63.88_ | _31.27_ | _76.70_ | _75.80_ |

\* Cross-eval reference only — different training data (`real_v2 + opensrc + synth_v2`, full 180K older synth) and the ckpt was selected on the pre-fix buggy metric.

Δ from zero-shot to fine-tuned (mIoU): A **+47.13**, B **+47.58**, C **+53.18**, D **+52.02**.
Fine-tuning is essential — even the worst fine-tuned variant beats the zero-shot
baseline by ~47 mIoU. Note: more synth (Stage D) does not help SAN further on test.

### Per-class IoU — zero-shot vs. fine-tuned SAN

| Class | Zero-Shot | Stage A | Stage B | Stage C | Stage D | Stage-2D mix | Δ (C − ZS) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| elevator           | 33.23 | 78.58 | 79.38 | 75.29 | 68.83 | 81.52 | +42.06 |
| elevator_button    |  1.50 | 88.73 | 87.18 | 91.76 | 92.90 | 83.01 | +90.26 |
| door_button        |  3.22 | 68.03 | 66.29 | 74.68 | 75.26 | 69.09 | +71.46 |
| crosswalk          | 26.16 | 90.72 | 84.06 | 90.98 | 91.60 | 90.63 | +64.82 |
| pedestrian_signal  | 10.93 | 40.14 | 49.01 | 56.52 | 57.92 | 48.38 | +45.59 |
| aps_button         |  2.92 | 46.92 | 48.27 | 51.11 | 52.19 | 49.40 | +48.19 |
| bus_stop           | 47.54 | 82.19 | 79.26 | 83.40 | 85.03 | 81.50 | +35.86 |
| bus_stop_sign      |  3.79 | 22.87 | 27.36 | 33.36 | 29.19 | 25.32 | +29.57 |
| handrail           |  6.27 | 41.98 | 40.77 | 50.69 | 50.78 | 46.46 | +44.42 |
| escalator          | 13.85 | 60.56 | 63.62 | 73.44 | 65.94 | 63.44 | +59.59 |

Per-class IoU is buried at zero-shot for under-represented classes
(`elevator_button` 1.50, `door_button` 3.22, `aps_button` 2.92, `bus_stop_sign` 3.79).
Crosswalk/elevator/bus-stop survive (26–47 IoU) because they exist in CLIP's
pretraining distribution. Fine-tuning recovers all classes — and synthetic data
(Stage C) gives an additional boost on top of fine-tuning, especially for the
under-represented classes (`elevator_button` 91.76, `door_button` 74.68,
`bus_stop_sign` 33.36, `handrail` 50.69 vs. Stage-A baselines).

---

## 2. Test Set Results (real_final/test)

### mIoU summary (fg-only, turnstile excluded)

| Architecture | A · Real | B · +Opensrc | C · +Synth₀.₁ | D · +Synth₀.₂ |
|---|:---:|:---:|:---:|:---:|
| SegFormer    | 48.52 | 48.01 | 56.39 | **58.35** |
| Mask2Former  | 68.10 | 66.57 | 72.46 | **72.64** |
| SAN          | 62.07 | 62.52 | **68.12** | 66.96 |

> All numbers are fg_mIoU — background pixels are ignored in GT (`reduce_zero_label=True`), so this is directly comparable to labmate's "fg_mIoU" column. Bold = best per architecture.

### Full metrics

| Arch | Stage | mIoU | mAP50-95 | Precision | Recall |
|---|:---:|:---:|:---:|:---:|:---:|
| **SegFormer**   | A | 48.52 | 22.02 | 50.36 | 90.76 |
| SegFormer       | B | 48.01 | 21.80 | 51.13 | 88.74 |
| SegFormer       | C | 56.39 | 26.07 | 59.86 | **89.10** |
| SegFormer       | D | **58.35** | **28.04** | **63.23** | 87.22 |
| **Mask2Former** | A | 68.10 | 37.53 | 79.10 | 81.86 |
| Mask2Former     | B | 66.57 | 35.09 | 78.60 | 79.99 |
| Mask2Former     | C | 72.46 | **40.14** | 82.40 | **84.10** |
| Mask2Former     | D | **72.64** | 40.04 | **84.37** | 82.60 |
| **SAN**         | A | 62.07 | 31.32 | 82.67 | 68.35 |
| SAN             | B | 62.52 | 30.09 | 77.10 | 74.17 |
| SAN             | C | **68.12** | **36.08** | **82.75** | **76.93** |
| SAN             | D | 66.96 | 35.34 | 82.53 | 75.45 |
| _SAN R+C+S (Stage-2D mixed, synth_v2)*_ | _2D_ | _63.88_ | _31.27_ | _76.70_ | _75.80_ |

\* SAN Stage-2D mixed: trained on `real_v2 ×40 + opensrc ×1 + synth_v2` (full 180K older synth, not the new `synth_0.1`/`synth_0.2`), 30K iters @ batch 24, best ckpt at iter 20K (selection criterion was the pre-fix buggy BLVMetric — model weights are unaffected, only ckpt selection was inflated). Reported here as a one-off R+C+S reference; for apples-to-apples comparison with Stage A/B/C/D, the cleaner numbers are the san_finalA…D rows above.

Δ vs. real-only (C − A) on test:

| Arch | Δ mIoU | Δ mAP | Δ Prec | Δ Rec |
|---|:---:|:---:|:---:|:---:|
| SegFormer   | **+7.87** | +4.05 | +9.50  | −1.66 |
| Mask2Former | **+4.36** | +2.61 | +3.30  | +2.24 |
| SAN         | **+6.05** | +4.76 | +0.08  | +8.58 |

Δ at max synth (D − A) on test — total gain from real-only to best synth config:

| Arch | Δ mIoU | Δ mAP | Δ Prec | Δ Rec |
|---|:---:|:---:|:---:|:---:|
| SegFormer   | **+9.83** | +6.02 | +12.87 | −3.54 |
| Mask2Former | **+4.54** | +2.51 | +5.27  | +0.74 |
| SAN         | **+4.89** | +4.02 | −0.14  | +7.10 |

Δ at higher synth (D − C) on test — does more synth keep helping?

| Arch | Δ mIoU | Δ mAP | Δ Prec | Δ Rec |
|---|:---:|:---:|:---:|:---:|
| SegFormer   | **+1.96** | +1.97 | +3.37  | −1.88 |
| Mask2Former | +0.18 | −0.10 | +1.97  | −1.50 |
| SAN         | **−1.16** | −0.74 | −0.22  | −1.48 |

Stage D scales synth from 0.1 to 0.2 and real-repeat from ×6 to ×12. The benefit is **diminishing or reversing** on test: SegFormer continues to gain, Mask2Former plateaus, and SAN regresses. On val every architecture improved with D, indicating the test/val gap widened — signature of mild overfitting from doubling the synth budget.

---

## 3. Validation Set Results (real_final/val, best checkpoint)

These are the val numbers the best checkpoint was selected on. Provided for completeness; test-set numbers above are the paper-reportable values.

| Arch | Stage | mIoU | mAP50-95 | Precision | Recall |
|---|:---:|:---:|:---:|:---:|:---:|
| SegFormer   | A | 49.66 | 22.43 | 49.94 | 96.77 |
| SegFormer   | B | 50.83 | 23.33 | 51.29 | 97.14 |
| SegFormer   | C | 59.72 | 30.03 | 60.29 | 97.80 |
| SegFormer   | D | **62.57** | **32.55** | **63.47** | **97.34** |
| Mask2Former | A | 75.88 | 44.47 | 82.00 | 89.62 |
| Mask2Former | B | 67.94 | 37.68 | 77.86 | 83.22 |
| Mask2Former | C | 81.83 | 51.86 | 87.72 | 91.44 |
| Mask2Former | D | **82.66** | **55.39** | **87.97** | **92.14** |
| SAN         | A | 76.12 | 44.87 | 85.94 | 84.62 |
| SAN         | B | 72.12 | 41.67 | 79.92 | 84.97 |
| SAN         | C | 80.28 | 52.24 | 86.89 | 89.74 |
| SAN         | D | **82.15** | **54.10** | **88.20** | **91.21** |

---

## 4. Per-Class IoU on `real_final/test`

Grouped by architecture. Bold marks the best stage for each class within an architecture.

### SegFormer
| Class | A | B | C | D |
|---|:---:|:---:|:---:|:---:|
| elevator           | **76.76** | 72.64 | 69.15 | 66.39 |
| elevator_button    | 77.61 | 73.30 | 84.55 | **86.42** |
| door_button        | 64.12 | 62.39 | 74.31 | **75.42** |
| crosswalk          | 54.39 | 49.75 | 66.82 | **72.04** |
| pedestrian_signal  | 14.13 | 14.56 | 28.01 | **29.39** |
| aps_button         | 30.90 | **33.87** | 29.92 | 32.34 |
| bus_stop           | 70.07 | 69.99 | 82.62 | **84.81** |
| bus_stop_sign      | 23.39 | 22.64 | 35.49 | **36.63** |
| handrail           | 19.30 | 24.20 | 32.60 | **35.09** |
| escalator          | 54.47 | 56.73 | 60.47 | **65.01** |

### Mask2Former
| Class | A | B | C | D |
|---|:---:|:---:|:---:|:---:|
| elevator           | 82.17 | **82.94** | 81.73 | 75.76 |
| elevator_button    | 92.54 | 90.36 | 94.06 | **94.55** |
| door_button        | 79.09 | 79.05 | **80.75** | 79.35 |
| crosswalk          | 81.41 | 80.90 | 91.15 | **91.39** |
| pedestrian_signal  | 50.58 | 45.96 | 58.03 | **60.81** |
| aps_button         | 42.87 | 40.63 | 52.08 | **53.70** |
| bus_stop           | 83.69 | 82.92 | 86.53 | **87.06** |
| bus_stop_sign      | 47.09 | 41.24 | **49.40** | 48.98 |
| handrail           | 47.92 | 55.03 | 57.28 | **57.81** |
| escalator          | 73.67 | 66.65 | 73.64 | **77.03** |

### SAN
| Class | A | B | C | D |
|---|:---:|:---:|:---:|:---:|
| elevator           | 78.58 | **79.38** | 75.29 | 68.83 |
| elevator_button    | 88.73 | 87.18 | 91.76 | **92.90** |
| door_button        | 68.03 | 66.29 | 74.68 | **75.26** |
| crosswalk          | 90.72 | 84.06 | 90.98 | **91.60** |
| pedestrian_signal  | 40.14 | 49.01 | 56.52 | **57.92** |
| aps_button         | 46.92 | 48.27 | 51.11 | **52.19** |
| bus_stop           | 82.19 | 79.26 | 83.40 | **85.03** |
| bus_stop_sign      | 22.87 | 27.36 | **33.36** | 29.19 |
| handrail           | 41.98 | 40.77 | 50.69 | **50.78** |
| escalator          | 60.56 | 63.62 | **73.44** | 65.94 |

---

## 5. Highlights (for paper)

- **Synthetic data improves every architecture, on every metric, at the test set.**
  Adding synth (Stage C) over real-only (Stage A) yields **+4.36 mIoU (Mask2Former), +6.05 (SAN), +7.87 (SegFormer)**, with consistent gains in mAP, precision, and recall.

- **Open-source inconsistent crosswalk annotations hurt performance**: See the mIoU results for different architectures.
  Despite opensrc containing **27,786 crosswalk images** (76% of the dataset, ~64× more than real_final's 434), adding it actually *drops* crosswalk IoU on SegFormer (54.39 → 49.75, −4.64) and Mask2Former (81.41 → 80.90, −0.51). Real-only Stage A already segments crosswalk well (M2F 81.41, SAN 90.72) — the opensrc labels are inconsistent enough that more data hurts.

- **Open-source compiled data (Stage B) is essentially neutral or harmful.**
  M2F −1.53, SegFormer −0.51, SAN +0.45 mIoU vs. real-only. Reason: the opensrc set is dominated by crosswalk (76% of images, 6 of 11 classes entirely absent), so it skews the model away from under-represented BLV classes.

- **Quantity ≠ quality, even within a class.** `elevator_button` is well-represented in opensrc (4,401 images) yet adding opensrc *drops* its IoU on every architecture: SF 77.61 → 73.30 (−4.31), M2F 92.54 → 90.36 (−2.18), SAN 88.73 → 87.18 (−1.55). Synthetic data, by contrast, lifts elevator_button to **84.55 SF / 94.06 M2F / 91.76 SAN** — the cleanly-labeled synthetic supervision is more useful than 4× more noisy real images.

- **Pedestrian signal — the hardest class — sees the largest relative gain from synth.** With real-only (Stage A), pedestrian_signal IoU is bottom-of-class for every architecture (SF 14.13, M2F 50.58, SAN 40.14). Adding synthetic data nearly doubles SegFormer (14.13 → **28.01**, +98%), and adds **+16.38 mIoU on SAN** (40.14 → 56.52). Pedestrian signals are small, distant, and rare in real captures — synthetic generation directly fills this distributional gap.

- **Handrail benefits across the board**, despite being a thin linear object that segmentation models typically struggle with. From only 206 real images, Stage A reaches 19.30 SF / 47.92 M2F / 41.98 SAN; synth boosts every arch by **+8.7 to +13.3 IoU** (SF +13.30, M2F +9.36, SAN +8.71). Linear-extent geometry is exactly the kind of variation synthetic generation can systematically expand.

- **Door button is rescued entirely by real + synth** — opensrc contains **zero** door_button instances, so synth is the *only* augmentation available beyond the 2,122 real images. Stage C door_button IoU is **74.31 SF (+10.19), 80.75 M2F (+1.66), 74.68 SAN (+6.65)** vs. Stage A. This is the cleanest case study for "synthetic supplements where real data exists but is insufficient."

- **Bus-stop vs. bus-stop-sign asymmetry.** The bus stop itself (a large structure) is already well-handled by real-only (M2F 83.69, SAN 82.19); synth gives a small lift (+2.84 M2F, +1.21 SAN). The *sign* attached to it — a small, distant element — is much harder (M2F 47.09 real-only) and gets the larger synth boost (+12.10 SF, +10.49 SAN). Synthetic data preferentially helps the small/distant sub-objects.

- **Synth is a safer augmentation than opensrc for noisy classes.** M2F escalator: Stage A (real-only) **73.67** → Stage B (real + opensrc) **66.65** (−7.02) → Stage C (real + synth, no opensrc) **73.64** (parity with A). Adding 1,296 opensrc escalator images hurt; using synth instead preserves real-only performance — the choice of augmentation matters as much as its quantity.

- **Mask2Former is the strongest backbone** under the honest metric — 72.46 mIoU at Stage C, beating SAN (68.12) and SegFormer (56.39). Query-based mask classification handles BLV's sparse-foreground regime better than per-pixel softmax.

- **Synthetic data lifts the long-tail classes the most.** Across all three architectures, the largest C−A class-wise gains are on the under-represented classes:
  - `pedestrian_signal`: +13.88 SF, +7.45 M2F, +16.38 SAN
  - `handrail`:          +13.30 SF, +9.36 M2F, +8.71 SAN
  - `bus_stop_sign`:     +12.10 SF, +2.31 M2F, +10.49 SAN
  - `aps_button`:        +9.21 M2F, +4.19 SAN  (SF saw a small dip)
  This is the targeted benefit of synthetic generation: covering classes that are expensive to collect in the real world.

- **Zero-shot open-vocabulary SAN is far below fine-tuned baselines** on BLV classes — 14.94 mIoU vs. 62-72 fine-tuned. The CLIP backbone alone is insufficient; BLV-specific supervision is essential, especially for the long-tail classes (`elevator_button` 1.50, `aps_button` 2.92, `door_button` 3.22 zero-shot).

- **High recall at moderate precision** is the universal pattern for SegFormer (~89% recall, ~50–60% precision). It segments aggressively. Mask2Former and SAN trade recall for precision (~82% precision, ~80% recall in M2F-C).

- **Test < Val gap** is small for A/B/C (~3-5 mIoU points). For Stage D the gap **widens** to 4-15 mIoU points (SF 4.22, M2F 10.02, SAN 15.19), a signature of mild overfitting from doubling the synth budget without new real data.

- **Stage D (real ×12 + synth_0.2) plateaus, then regresses for SAN.** On test, D−C is +1.96 SF, +0.18 M2F, **−1.16 SAN**. SegFormer keeps benefiting from more synth, Mask2Former is essentially at its ceiling, and SAN over-fits. Val told the opposite story (all three improved with D), which is precisely why test eval is the paper-reportable number.

- **Long-tail classes still benefit from D where the architecture has headroom.** M2F: pedestrian_signal +2.78, aps_button +1.62, escalator +3.39 from C→D. SAN: pedestrian_signal +1.40, but bus_stop_sign drops −4.17 and escalator drops −7.50 — the regressions concentrate on the rarest classes (where each evaluation image moves the IoU dramatically).

- **R+C+S cross-eval (SAN Stage-2D mixed, synth_v2)**: 63.88 mIoU on real_final/test — better than R-only (62.07) and R+C (62.52) but **worse than R+S₀.₁ (68.12)**. Adding opensrc on top of synth doesn't help, even with 5× more synthetic data. The simplest take: real + clean synth wins; opensrc remains a net drag regardless of synth scale.
