# BLV Segmentation — Code Release

Semantic segmentation pipeline for Blind/Low-Vision (BLV) accessibility landmarks.
Three architectures (Mask2Former, SAN, SegFormer) are evaluated across four data
configurations (A–D) to measure the contribution of synthetic Isaac Sim data to
real-world BLV segmentation.

---

## Repository Structure

```
blv_pipeline/                    Core library (custom MMSeg plugins)
  constants.py                   Class names, palettes, checkpoint URLs
  class_mapping_ade20k.py        ADE20K → BLV remapping (zero-shot SAN)
  data_utils.py                  COCO → semantic segmentation utilities
  mmseg_plugins/
    datasets/blv_dataset.py      BLVDataset, BLVDatasetV2Fg (reduce_zero_label)
    datasets/remap_seg_label.py  RemapSegLabel transform (255 → bg class id)
    decode_heads/
      blv_mask2former_head.py    M2F head with per-query fg filtering at inference
      blv_san_head.py            SAN head with per-query fg filtering
    evaluation/blv_metric.py     BLVMetric: mIoU, mAP50-95, Precision, Recall
    visualization/               SegVisualizationHook, WandbMetricsOnlyBackend

configs/
  blv/                           Training/eval configs for all stages
  Final/                         Paper-canonical configs (A/B/C/D × 3 archs)

tools/
  blv/
    slurm_train.sh               SLURM training launcher (MODEL TRACK)
    download_checkpoints.py      Download ADE20K / COCO-Stuff pretrained weights
    conda_utils.sh               activate_conda_env helper
    convert_real_coco_to_semseg.py  Convert raw COCO real data → mmseg format
    preprocess_synthetic.py      Convert Isaac Sim renders → mmseg format
    split_synthetic_dataset.py   Train/val/test split for synthetic data
    eval_stage1_split.py         Split-domain evaluation utility
  Final/
    slurm_train_final.sh         Train any Final config
    slurm_eval_final.sh          Evaluate with visualizations (~10% sample)
    slurm_eval_final_no_vis.sh   Evaluate metrics only (faster)
    slurm_eval_final_bg.sh       BG-inclusive evaluation (12-class)
  parse_eval_results.py          Parse metrics.json files into a summary table

data/
  README.md                      Data format specification and setup

checkpoints/
  README.md                      Pretrained checkpoint download instructions

results/
  results_final.md               Full results tables (test + val + per-class)
```

---

## Environment Setup

Requires CUDA 12.1. Tested on NVIDIA L40S and RTX 4090.

```bash
conda env create -f environment.yml
conda activate openmmlab
export BLV_PROJECT_ROOT="$(pwd)"
export PYTHONPATH="${BLV_PROJECT_ROOT}:${PYTHONPATH}"
```

Key dependencies (pinned in `environment.yml`):

- Python 3.10, PyTorch 2.4.1+cu121
- mmengine 0.10.7, mmcv 2.1.0, mmdet 3.3.0, mmsegmentation 1.2.2
- wandb 0.24.2, prettytable, pycocotools

---

## Data Setup

See [data/README.md](data/README.md) for the full data format specification.

Four data sources are used across the four experimental configurations:

| Config | Training data                         |
| ------ | ------------------------------------- |
| A      | `real_final` only (3,703 images)      |
| B      | `real_final` + `opensrc_final` (~40K) |
| C      | `real_final` + `synth`                 |
| D      | `real_final` + `synth` (larger)        |

After placing datasets under `data/`, set:

```bash
export BLV_PROJECT_ROOT="$(pwd)"
```

All configs resolve dataset paths via `$BLV_PROJECT_ROOT/data/`.

---

## Pretrained Checkpoints

Download the three required ADE20K / COCO-Stuff pretrained weights:

```bash
python tools/blv/download_checkpoints.py
```

This fetches into `checkpoints/pretrained/`:

- `mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640.pth`
- `segformer_mit-b5_640x640_160k_ade20k_20210801_121243-41d2845b.pth`
- `san-vit-b16_20230906-fd0a7684.pth`

See [checkpoints/README.md](checkpoints/README.md) for direct URLs and SHA256 hashes.

---

## Training

### Final paper configs (A / B / C / D)

```bash
# sbatch tools/Final/slurm_train_final.sh <MODEL> <STAGE>
# MODEL: segformer | mask2former | san
# STAGE: A | B | C | D

sbatch tools/Final/slurm_train_final.sh segformer A
sbatch tools/Final/slurm_train_final.sh mask2former C
sbatch tools/Final/slurm_train_final.sh san D
```

Config files: `configs/Final/<model>_final<letter>_<data>.py`

### General configs (earlier ablation stages)

```bash
# sbatch tools/blv/slurm_train.sh <model> <track>
# track: real | real_synth_v2 | synth_v2 | mixed | ...

sbatch tools/blv/slurm_train.sh mask2former real_synth_v2
```

---

## Evaluation

### Test set evaluation (metrics only, no visualizations)

```bash
sbatch tools/Final/slurm_eval_final_no_vis.sh <MODEL> <STAGE> <CKPT_PATH>
# Output: work_dirs/eval_results/<MODEL>_final<STAGE>_<DT>/metrics.json
```

### Test set evaluation with visualizations (~10% image sample)

```bash
sbatch tools/Final/slurm_eval_final.sh <MODEL> <STAGE> <CKPT_PATH>
```

### BG-inclusive evaluation (12-class, bg as synthesized class 11)

```bash
sbatch tools/Final/slurm_eval_final_bg.sh <MODEL> <STAGE> <CKPT_PATH>
```

### Parse results from multiple runs

```bash
python tools/parse_eval_results.py --results-dir work_dirs/eval_results/
```

---

## Key Implementation Details

### BLVMetric (`blv_pipeline/mmseg_plugins/evaluation/blv_metric.py`)

Custom metric computing mIoU, mAP50-95, Precision, and Recall.

- `excluded_class_indices=[10]` — NaNs turnstile (class 10) before computing
  mean. Turnstile has zero GT coverage in `real_final`; including it would
  deflate the mean by ~8 pts.
- All configs use `BLVDatasetV2Fg` (`reduce_zero_label=True`), so background
  pixels map to `ignore_index=255` and are never entered into the confusion
  matrix. All reported mIoU numbers are **foreground-only** (fg_mIoU).
- `synthesize_bg_channel=True` enables a 12-class evaluation where background
  is synthesized as `1 − max(fg_softmax)` for M2F/SAN (which have no learned
  bg channel).

### Per-query fg filtering (`blv_pipeline/mmseg_plugins/decode_heads/`)

Mask2Former and SAN use Q=100 queries. At inference, ~98 queries are matched
to "no object" during training — they crowd out the 2-3 fg-specialist queries
in the einsum aggregation. Fix: zero-out queries whose per-query fg confidence
`max_c softmax(cls)[q,0:C]` is below `query_fg_threshold` (default 0.1) before
the einsum. This is applied in `BLVMask2FormerHead.predict()` and
`BLVSideAdapterCLIPHead.predict_by_feat()`.

### Dataset classes

`BLV_V2_CLASSES_FG` (11 classes, indices 0–10):
`elevator`, `elevator_button`, `door_button`, `crosswalk`, `pedestrian_signal`,
`aps_button`, `bus_stop`, `bus_stop_sign`, `handrail`, `escalator`, `turnstile`

---

## Results

See [results/results_final.md](results/results_final.md) for complete test and
val tables across all 12 model × data configurations.

Summary (fg_mIoU on `real_final/test`, 1,482 images, turnstile excluded):

| Architecture | A · Real | B · +Opensrc | C · +Synth | D · +Synth (larger) |
| ------------ |:--------:|:------------:|:-------------:|:-------------:|
| SegFormer    | 48.52    | 48.01        | 56.39         | **58.35**     |
| Mask2Former  | 68.10    | 66.57        | 72.46         | **72.64**     |
| SAN          | 62.07    | 62.52        | **68.12**     | 66.96         |

Synthetic data (Stage C) improves every architecture over real-only (Stage A):
+7.87 mIoU (SegFormer), +4.36 (Mask2Former), +6.05 (SAN).

---

## Cluster / HPC Notes

SLURM scripts target `gpu-preempt` partition with L40S GPUs. To run locally
outside SLURM, activate the conda environment and use `dist_test.sh` directly:

```bash
source tools/blv/conda_utils.sh && activate_conda_env
MMSEG_ROOT=$(python -c 'import mmseg, os; print(os.path.dirname(os.path.dirname(mmseg.__file__)))')
bash "${MMSEG_ROOT}/tools/dist_test.sh" <config> <checkpoint> 1 \
    --work-dir work_dirs/my_eval/ \
    --cfg-options test_evaluator.output_metrics_path=work_dirs/my_eval/metrics.json
```
