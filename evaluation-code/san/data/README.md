# Data Setup

All datasets must be placed under `$BLV_PROJECT_ROOT/data/` in mmseg-compatible
format before training or evaluation.

## Expected Layout

```
data/
  real_final/              Real-world BLV captures (collected and annotated)
    img_dir/
      train/  (3,703 PNG images)
      val/    (396 PNG images)
      test/   (1,482 PNG images)
    ann_dir/
      train/  (3,703 PNG label maps)
      val/    (396 PNG label maps)
      test/   (1,482 PNG label maps)

  opensrc_final/           Compiled open-source images (crosswalk-heavy)
    img_dir/train/         (~40K images)
    ann_dir/train/

  synth/                   Synthetic data (Isaac Sim)
    img_dir/train/
    ann_dir/train/
```

## Label Format

Segmentation maps are single-channel PNG files with the following class IDs:

| ID | Class |
|----|-------|
| 0  | background (mapped to ignore_index=255 at load time via `reduce_zero_label=True`) |
| 1  | elevator |
| 2  | elevator_button |
| 3  | door_button |
| 4  | crosswalk |
| 5  | pedestrian_signal |
| 6  | aps_button |
| 7  | bus_stop |
| 8  | bus_stop_sign |
| 9  | handrail |
| 10 | escalator |
| 11 | turnstile |
| 255 | ignore_index (unlabeled regions) |

After `reduce_zero_label=True` (applied by `BLVDatasetV2Fg`), IDs shift down
by 1: foreground classes become 0–10, background becomes 255 (ignored).

## Preprocessing Scripts

**Real data (COCO-format → mmseg):**
```bash
export BLV_RAW_REAL_DATA=/path/to/your/coco_format_real_dataset
python tools/blv/convert_real_coco_to_semseg.py \
    --src "$BLV_RAW_REAL_DATA" \
    --dst data/real_final
```

**Synthetic data (Isaac Sim renders → mmseg):**
```bash
python tools/blv/preprocess_synthetic.py \
    --src /path/to/isaac_sim_output \
    --dst data/synth_raw

python tools/blv/split_synthetic_dataset.py \
    --src data/synth_raw \
    --dst data/synth
```

## Dataset Statistics

| Split | real_final | opensrc_final | synth (full) |
|-------|:---:|:---:|:---:|
| Train | 3,703 | ~36,049 | 180,886 |
| Val   | 396   | — | 17,617 |
| Test  | 1,482 | — | 34,711 |

The synthetic subsets used in training are random samples drawn from the
full synthetic training split.
