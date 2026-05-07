# Data Processing Pipeline (Real + Synthetic → MMSeg)

This repository provides a unified pipeline to convert both Real (COCO-format) and Synthetic datasets into MMSegmentation format.

---

## Project Structure

project/
├── real/
│   ├── adjust_json.py
│   └── coco_to_mmseg.py
├── synthetic/
│   ├── classify_to_processed.py
│   └── make_mmseg_from_flat_synthetic.py
├── raw_dataset/
│   ├── train/
│   ├── val/
│   └── test/
├── synthetic_raw/
└── outputs/
    ├── real_mmseg/
    └── synthetic_mmseg/

---

## Full Pipeline (Run All)

# REAL
python real/adjust_json.py --root raw_dataset
python real/coco_to_mmseg.py --root raw_dataset --out-root outputs/real_mmseg --clean

# SYNTHETIC
python synthetic/classify_to_processed.py --src-root synthetic_raw --out-root synthetic_processed
python synthetic/make_mmseg_from_flat_synthetic.py --dataset-root synthetic_processed --out-root outputs/synthetic_mmseg --clean

---

## Real Data Processing

Input: COCO-style dataset

Structure:
raw_dataset/
  train/<class_name>/{images, annotations_adjusted.json}
  val/...
  test/...

Step 1: Normalize JSON
→ adjust_json.py converts all annotations into consistent COCO format (*_adjusted.json)

Step 2: Convert to MMSeg
→ coco_to_mmseg.py generates:
  img_dir/{train,val,test}
  ann_dir/{train,val,test}
  splits/{train.txt,val.txt,test.txt}
  meta.json

---

## Synthetic Data Processing

Input: raw synthetic data

Step 1: Flatten structure
→ classify_to_processed.py reorganizes files into:
  synthetic_processed/<object>/
    ├── rgb/
    ├── semantic_segmentation/
    └── semantic_segmentation_labels/

Step 2: Convert to MMSeg
→ make_mmseg_from_flat_synthetic.py generates:
  img_dir/{train,val,test}
  ann_dir/{train,val,test}
  splits/
  meta.json

---

## Optional: Merge Real + Synthetic

mkdir -p outputs/merged_mmseg/img_dir/train
mkdir -p outputs/merged_mmseg/ann_dir/train

cp outputs/real_mmseg/img_dir/train/* outputs/merged_mmseg/img_dir/train/
cp outputs/synthetic_mmseg/img_dir/train/* outputs/merged_mmseg/img_dir/train/

cp outputs/real_mmseg/ann_dir/train/* outputs/merged_mmseg/ann_dir/train/
cp outputs/synthetic_mmseg/ann_dir/train/* outputs/merged_mmseg/ann_dir/train/

Note: You must also merge split txt files manually.

---

## Class Taxonomy

0  background
1  elevator
2  elevator_button
3  door_button
4  crosswalk
5  pedestrian_signal
6  aps_button
7  bus_stop
8  bus_stop_sign
9  handrail
10 escalator
11 turnstile

---

## Notes

- Real uses polygon annotations → rasterized to masks
- Synthetic uses RGBA segmentation → mapped to class IDs
- Filename collisions are handled via prefixing or deduplication
- Missing images or empty masks usually indicate annotation mismatch

---

## Summary

Real: COCO → polygon → mask → MMSeg  
Synthetic: RGBA → class mapping → mask → MMSeg

Both outputs are fully compatible with MMSegmentation training.
