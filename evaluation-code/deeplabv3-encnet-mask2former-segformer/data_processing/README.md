# Data Processing Pipeline

Convert raw Real (COCO-format) and Synthetic datasets into MMSegmentation-compatible format.

---

## Directory Structure

```
data_processing/
├── real/
│   ├── adjust_json.py                  # Normalize COCO annotation JSON files
│   └── coco_to_mmseg.py               # Convert COCO annotations → MMSeg masks
└── synthetic/
    ├── classify_to_processed.py        # Flatten raw synthetic data by object type
    └── make_mmseg_from_flat_synthetic.py  # Convert synthetic data → MMSeg format
```

---

## Real Data Processing

**Input:** COCO-style annotated dataset

```
raw_dataset/
├── train/<class_name>/{images/, annotations.json}
├── val/<class_name>/{images/, annotations.json}
└── test/<class_name>/{images/, annotations.json}
```

### Step 1: Normalize Annotations

```bash
python data_processing/real/adjust_json.py --root raw_dataset
```

This normalizes all COCO JSON files into a consistent format, producing `*_adjusted.json` files.

### Step 2: Convert to MMSeg Format

```bash
python data_processing/real/coco_to_mmseg.py \
    --root raw_dataset \
    --out-root outputs/real_mmseg \
    --clean
```

**Output:**
```
outputs/real_mmseg/
├── img_dir/{train,val,test}/
├── ann_dir/{train,val,test}/
├── splits/{train.txt,val.txt,test.txt}
└── meta.json
```

---

## Synthetic Data Processing

**Input:** Raw synthetic data (e.g., from NVIDIA Omniverse / Isaac Sim)

### Step 1: Flatten Directory Structure

```bash
python data_processing/synthetic/classify_to_processed.py \
    --src-root synthetic_raw \
    --out-root synthetic_processed
```

Reorganizes files into:
```
synthetic_processed/<object_name>/
├── rgb/
├── semantic_segmentation/
└── semantic_segmentation_labels/
```

### Step 2: Convert to MMSeg Format

```bash
python data_processing/synthetic/make_mmseg_from_flat_synthetic.py \
    --dataset-root synthetic_processed \
    --out-root outputs/synthetic_mmseg \
    --clean
```

**Output:**
```
outputs/synthetic_mmseg/
├── img_dir/{train,val,test}/
├── ann_dir/{train,val,test}/
├── splits/{train.txt,val.txt,test.txt}
└── meta.json
```

---

## Merging Real + Synthetic

To create a combined dataset for training:

```bash
mkdir -p outputs/merged_mmseg/img_dir/train
mkdir -p outputs/merged_mmseg/ann_dir/train

cp outputs/real_mmseg/img_dir/train/* outputs/merged_mmseg/img_dir/train/
cp outputs/synthetic_mmseg/img_dir/train/* outputs/merged_mmseg/img_dir/train/

cp outputs/real_mmseg/ann_dir/train/* outputs/merged_mmseg/ann_dir/train/
cp outputs/synthetic_mmseg/ann_dir/train/* outputs/merged_mmseg/ann_dir/train/
```

> **Note:** Also merge the split `.txt` files when combining datasets.

---

## Class Taxonomy

| ID | Class |
|----|-------|
| 0 | background |
| 1 | elevator |
| 2 | elevator_button |
| 3 | door_button |
| 4 | crosswalk |
| 5 | pedestrian_signal |
| 6 | aps_button |
| 7 | bus_stop |
| 8 | bus_stop_sign |
| 9 | handrail |
| 10 | escalator |
| 11 | turnstile |

---

## Notes

- **Real data** uses polygon annotations, rasterized into pixel-level masks
- **Synthetic data** uses RGBA segmentation images, mapped to class IDs via label JSON files
- Filename collisions are handled via folder-name prefixing
- Missing images or empty masks typically indicate annotation mismatches

---

## Summary

| Source | Pipeline |
|--------|----------|
| Real | COCO JSON → polygon rasterization → MMSeg masks |
| Synthetic | RGBA segmentation → class mapping → MMSeg masks |

Both outputs are fully compatible with MMSegmentation training.
