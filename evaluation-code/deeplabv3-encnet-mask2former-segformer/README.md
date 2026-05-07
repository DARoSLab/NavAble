# NAVable: Semantic Segmentation with MMSegmentation

## Overview

This project provides training and evaluation pipelines for semantic segmentation using MMSegmentation.

It includes:
- Custom dataset (`MyMMSegDataset`)
- Custom metric (`FgIoUMetric`)
- Multiple experiment settings (real / synthetic / curated)
- Multiple model architectures (EncNet, DeepLabV3+, Mask2Former, SegFormer)

---

## 1. Installation

### 1.1 Clone MMSegmentation

```bash
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
```

### 1.2 Environment

```bash
conda create -n mmseg python=3.10 -y
conda activate mmseg

pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
pip install -e .
```

---

## 2. Add Project Files

Place files into MMSegmentation:

```
mmsegmentation/
├── mmseg/
│   ├── datasets/
│   │   └── my_mmseg_dataset.py
│   └── evaluation/
│       └── fg_iou_metric.py
│
├── configs/
│   └── my_models/
│       └── [project_name]/
│           ├── real/
│           ├── real_only/
│           ├── realsyn/
│           ├── realsyn_nocur/
│           └── realsyn_nocur_partial/
```

Important:
- `my_mmseg_dataset.py` → `mmseg/datasets/`
- `fg_iou_metric.py` → `mmseg/evaluation/`

---

## 3. Dataset Format

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

### Classes

```
background
elevator
elevator_button
door_button
crosswalk
pedestrian_signal
aps_button
bus_stop
bus_stop_sign
handrail
escalator
turnstile
```

---

## 4. Config Structure

Each config = (dataset setting) + (model)

### Dataset settings

| Folder | Description |
|--------|------------|
| real_only | real dataset only |
| real | real + curated |
| realsyn | real + synthetic |
| realsyn_nocur | synthetic without curation |
| *_partial | subset experiments |

### Models

- deeplabv3plus
- encnet
- mask2former
- segformer

---

## 5. IMPORTANT: Modify Data Paths

Before training, edit config:

```python
data_roots = [
    'PATH_TO_REAL_DATA',
    'PATH_TO_CURATED_OR_SYN_DATA',
]
```

❗ This must be changed. Hardcoded paths will not work.

---

## 6. Training

Example:

```bash
python tools/train.py \
configs/my_models/[project_name]/realsyn/encnet.py
```

---

## 7. Evaluation

```bash
python tools/test.py \
configs/my_models/[project_name]/realsyn/encnet.py \
work_dirs/.../best.pth
```

Save predictions:

```bash
python tools/test.py \
configs/... \
checkpoint.pth \
--out results.pkl
```

---

## 8. Metrics

We use custom metric:

- mIoU (all classes)
- fg_mIoU (excluding background)

Best model selection:

```python
save_best = 'fg_mIoU'
```

---

## 9. Notes on EncNet (Important)

EncNet may collapse rare classes when mixing datasets.

Observed behavior:
- Some classes (e.g., `bus_stop`) may have 0 predictions
- Model predicts background instead

Recommendation:
- Check per-class prediction distribution
- Inspect GT vs prediction overlap
- Monitor class imbalance

---

## 10. Troubleshooting

### Class IoU = 0

Possible reasons:
- No predictions for that class
- Severe class imbalance
- Domain gap (real vs synthetic)
- Small object size

### Model predicts mostly background

- Check `class_weight`
- Verify dataset distribution
- Visualize predictions

---

## 11. Reproducibility Notes

Environment:

- Python 3.10
- PyTorch 2.x
- CUDA 12.x
- MMEngine 0.10.x
- MMCV >= 2.0

---

## 12. Acknowledgement

Built on top of MMSegmentation:  
https://github.com/open-mmlab/mmsegmentation