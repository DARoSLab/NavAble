# NAVable: Semantic Segmentation with MMSegmentation

## Overview

This repository provides training and evaluation pipelines for semantic segmentation on the **NAVable** dataset using [MMSegmentation](https://github.com/open-mmlab/mmsegmentation).

**Contents:**

| Component                            | Description                                                      |
| ------------------------------------ | ---------------------------------------------------------------- |
| `mmseg/datasets/my_mmseg_dataset.py` | Custom dataset class for 12-class NAVable segmentation           |
| `mmseg/evaluation/fg_iou_metric.py`  | Custom metric: mIoU + foreground-only mIoU (`fg_mIoU`)           |
| `configs/navable/`                   | Training configs for 4 models × 6 dataset settings               |
| `data_processing/`                   | Scripts to convert raw data (real + synthetic) into MMSeg format |

---

## 1. Installation

### 1.1 Clone MMSegmentation

```bash
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
```

### 1.2 Create Environment

```bash
conda create -n mmseg python=3.10 -y
conda activate mmseg

pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
pip install -e .

# Additional dependencies
pip install -r /path/to/this/repo/requirements.txt
```

### 1.3 Tested Environment

| Package        | Version                            |
| -------------- | ---------------------------------- |
| Python         | 3.10                               |
| PyTorch        | 2.x                                |
| CUDA           | 12.x                               |
| MMEngine       | ≥ 0.10.0                           |
| MMCV           | ≥ 2.0.0                            |
| MMSegmentation | ≥ 1.0.0                            |
| MMDetection    | ≥ 3.0.0 (required for Mask2Former) |

---

## 2. File Placement

Copy the provided files into your MMSegmentation installation:

```
mmsegmentation/
├── mmseg/
│   ├── datasets/
│   │   └── my_mmseg_dataset.py          ← copy from mmseg/datasets/
│   └── evaluation/
│       └── fg_iou_metric.py             ← copy from mmseg/evaluation/
│
└── configs/
    └── navable/
        ├── real/                         ← copy from configs/navable/
        ├── real_only/
        ├── realsyn/
        ├── realsyn_nocur/
        ├── realsyn_nocur_partial/
        └── realsyn_partial/
```

> **Note:** The directory structure of this repository mirrors the MMSegmentation layout, so you can simply copy the `mmseg/` and `configs/` directories into your MMSegmentation clone.

---

## 3. Dataset Format

Prepare your data in the following MMSeg-compatible structure:

```
DATA_ROOT/
├── img_dir/
│   ├── train/
│   ├── val/
│   └── test/
├── ann_dir/
│   ├── train/
│   ├── val/
│   └── test/
└── meta.json
```

See `data_processing/README.md` for scripts to convert raw data into this format.

### Class Taxonomy (12 classes)

| ID  | Class Name        |
| --- | ----------------- |
| 0   | background        |
| 1   | elevator          |
| 2   | elevator_button   |
| 3   | door_button       |
| 4   | crosswalk         |
| 5   | pedestrian_signal |
| 6   | aps_button        |
| 7   | bus_stop          |
| 8   | bus_stop_sign     |
| 9   | handrail          |
| 10  | escalator         |
| 11  | turnstile         |

---

## 4. Configuration

Each config combines a **dataset setting** with a **model architecture**.

### Dataset Settings

| Setting                 | Description                                |
| ----------------------- | ------------------------------------------ |
| `real_only`             | Real dataset only                          |
| `real`                  | Real + curated synthetic                   |
| `realsyn`               | Real + full synthetic                      |
| `realsyn_nocur`         | Real + synthetic (without curation)        |
| `realsyn_partial`       | Real + curated synthetic (subset)          |
| `realsyn_nocur_partial` | Real + synthetic without curation (subset) |

### Model Architectures

| Model       | Config Base                   |
| ----------- | ----------------------------- |
| DeepLabV3+  | ResNet-101-D8, crop 512×512   |
| EncNet      | ResNet-101-D8, crop 512×512   |
| Mask2Former | Swin-L (IN-22K), crop 640×640 |
| SegFormer   | MiT-B5, crop 640×640          |

### ⚠️ IMPORTANT: Update Data Paths Before Training

Edit the `data_roots` and `data_root` fields in your config:

```python
data_roots = [
    '/absolute/path/to/real_mmseg_data',
    '/absolute/path/to/synthetic_mmseg_data',
]
```

Validation and test dataloaders also require updating `data_root`:

```python
data_root='PATH_TO_REAL_TEST_DATA',  # ← change this
```

---

## 5. Training

```bash
# Single GPU
python tools/train.py configs/navable/realsyn/encnet.py

# Multi-GPU (e.g., 4 GPUs)
bash tools/dist_train.sh configs/navable/realsyn/encnet.py 4
```

---

## 6. Evaluation

```bash
# Evaluate with best checkpoint
python tools/test.py \
    configs/navable/realsyn/encnet.py \
    work_dirs/navable_realsyn_encnet/best_fg_mIoU_*.pth

# Save predictions
python tools/test.py \
    configs/navable/realsyn/encnet.py \
    checkpoint.pth \
    --out results.pkl
```

---

## 7. Metrics

We use a custom `FgIoUMetric` that reports:

| Metric       | Description                                                  |
| ------------ | ------------------------------------------------------------ |
| `mIoU`       | Mean IoU over all 12 classes                                 |
| `fg_mIoU`    | Mean IoU over foreground classes only (excluding background) |
| `mFscore`    | Mean F-score over all classes                                |
| `fg_mFscore` | Mean F-score over foreground classes only                    |

Best model selection is based on `fg_mIoU`:

```python
save_best = 'fg_mIoU'
```

---

## 8. Notes

### EncNet Behavior

EncNet may collapse rare classes when mixing real and synthetic datasets:

- Some classes (e.g., `bus_stop`) may have 0 predictions
- The model may default to predicting background

**Recommendations:**

- Check per-class prediction distribution
- Monitor class imbalance across datasets
- Inspect GT vs. prediction overlap

### Troubleshooting

| Issue                            | Possible Causes                                            |
| -------------------------------- | ---------------------------------------------------------- |
| Class IoU = 0                    | No predictions for that class; class imbalance; domain gap |
| Model predicts mostly background | Adjust `class_weight`; verify dataset distribution         |

---

## 9. Acknowledgement

Built on top of [MMSegmentation](https://github.com/open-mmlab/mmsegmentation).

---

## License

This project is released under the [MIT License](LICENSE).
