# GuideTWSI Dataset

Detailed documentation for the GuideTWSI dataset, the largest and most diverse Tactile Walking Surface Indicator (TWSI) dataset for blind and low-vision navigation research.

## Overview

GuideTWSI comprises three sub-datasets totaling ~39.5K images:

| Dataset | Scale | Type | Source | Geography | Modalities |
|---------|-------|------|--------|-----------|------------|
| **RBar-22K** | ~22K | Real/bars | SideGuide, Tenji10K, TP, 69 Roboflow repos | Mostly Asia | RGB, Seg |
| **SDome-15K** | 15K+ | Synthetic/domes | UE4 + AirSim pipeline | United States | RGB+D, BBx, Seg |
| **RDome-2K** | 2.4K+ | Real/domes | Unitree Go2 robot collection | United States | RGB, Seg |

## Sub-Dataset Details

### RBar-22K: Curated Real-World TWSI Data

A comprehensive compilation of real-world tactile paving images from multiple public sources:

- **SideGuide** [1]: Large-scale sidewalk dataset (~8.2K images). We extracted only images containing the tactile paving class.
- **Tenji10K** [2]: 10K first-person directional bar images from Japan.
- **TP** [3]: ~1.4K samples captured under various appearances and lighting.
- **69 Roboflow community repositories**: Curated from publicly available datasets (see Appendix below).

**Curation process:**
1. Gathered data from all sources listed above
2. Removed duplicates across datasets
3. Standardized annotation formats
4. Performed quality control — discarded 785 samples without proper segmentation masks or with only bounding boxes / missing labels
5. Converted all resources into unified formats:
   - Run-Length Encoding (RLE) for SAM2.1
   - Polygon-only annotations (class ID + polygon coordinates without bounding boxes) for YOLOv11-seg
6. Manually overlaid masks on RGB images to verify annotation quality

**Result:** 19,925 high-quality, mask-annotated real-world TWSI images (primarily directional bars).

### SDome-15K: Synthetic Truncated Dome Dataset

Over 15,010 photorealistic images of truncated domes generated using our UE4 + AirSim pipeline.

**Specifications:**
- 10 diverse UE4 environments (City Park, Downtown, Suburban, etc.)
- Varying weather/lighting (sunny, overcast, rainy, snowy, daytime, sunset, night)
- 8 custom truncated dome paving types (ADA-compliant dimensions)
- 271 tactile paving spots across all environments
- Robot-relevant camera viewpoints (circular orbit + top-down sweep)

**Modalities per sample:**
1. RGB image
2. Pixel-wise semantic segmentation mask
3. Instance segmentation mask
4. Depth map
5. 2D bounding boxes (COCO-style)
6. Camera intrinsic parameters

### RDome-2K: Robot-Collected Truncated Dome Data

2,466 real-world truncated dome images collected from a robot's perspective.

**Collection setup:**
- Unitree Go2 quadruped robot
- Intel RealSense D435 camera (70° downward tilt)
- Diverse environments: campus, residential, suburban, rural
- Different times of day and lighting conditions

**Annotation:**
- Manually annotated using Roboflow's auto-segmentation tool
- Human-verified for quality

## Annotation Formats

### YOLO Polygon Format
```
<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
```
Normalized polygon coordinates. Class ID `0` = tactile_paving.

### COCO JSON Format
```json
{
  "images": [{"id": 1, "file_name": "image.jpg", "width": 1024, "height": 1024}],
  "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [x, y, w, h], "segmentation": {...}}],
  "categories": [{"id": 1, "name": "tactile_paving"}]
}
```

### SAM2 RLE Format
```json
{
  "image": {"image_id": 1, "file_name": "image.jpg", "height": 1024, "width": 1024},
  "annotations": [{"id": 1, "bbox": [x, y, w, h], "segmentation": {"counts": "...", "size": [h, w]}}]
}
```

## Dataset Statistics

| Split | RBar-22K | SDome-15K | RDome-2K | Total |
|-------|----------|-----------|----------|-------|
| Train | 17,554 (88%) | 13,209 (88%) | — | 30,763 |
| Val | 1,185 (6%) | 901 (6%) | — | 2,086 |
| Test | 1,186 (6%) | 900 (6%) | 2,466 (100%) | 4,552 |

- **Train/Val split ratio:** 88% / 6% / 6% (for RBar-22K and SDome-15K)
- **RDome-2K** is reserved entirely for testing

## Download

```bash
# Install HuggingFace CLI
pip install huggingface_hub

# Download the full dataset
huggingface-cli download ANONYMIZED/GuideTWSI --local-dir ./data

# Download specific subsets
huggingface-cli download ANONYMIZED/GuideTWSI --include "RBar-22K/*" --local-dir ./data
huggingface-cli download ANONYMIZED/GuideTWSI --include "SDome-15K/*" --local-dir ./data
huggingface-cli download ANONYMIZED/GuideTWSI --include "RDome-2K/*" --local-dir ./data
```

## References

1. K. Park et al., "SideGuide: A large-scale sidewalk dataset for guiding impaired people," IROS 2020.
2. T. Takano et al., "Tactile paving detection and tracking using Tenji10K dataset," IEEJ Trans., 2024.
3. X. Zhang et al., "Grfb-unet: A new multi-scale attention network with group receptive field block for tactile paving segmentation," Expert Systems with Applications, 2024.

---

## Appendix A: Roboflow Data Sources

The following 69 Roboflow community repositories were used as sources for RBar-22K.
After curation (deduplication, quality control, format standardization), 785 samples were excluded.

| # | Project | Items | Type | URL |
|---|---------|-------|------|-----|
| 1 | block (braille-block/block-wawkg) | 161 | detect | [link](https://universe.roboflow.com/braille-block/block-wawkg) |
| 2 | Station Detect (internship-a0crg) | 100 | seg | [link](https://universe.roboflow.com/internship-a0crg/station-detect-usdl0) |
| 3 | Yellow block (yellowload) | 113 | seg | [link](https://universe.roboflow.com/yellowload/yellow-block-fjfc6) |
| 4 | braille_block_1class (demo-1g4jj) | 200 | seg | [link](https://universe.roboflow.com/demo-1g4jj/braille_block_1class/dataset/2) |
| 5 | yolo (school-f9udu) | 46 | detect | [link](https://universe.roboflow.com/school-f9udu/yolo-s47cy/dataset/1/download) |
| 6 | b_bv2 (brailblockdetection) | 854 | detect | [link](https://universe.roboflow.com/brailblockdetection/b_bv2) |
| 7 | -b1zsb (project-8nx9x) | 26 | detect | [link](https://universe.roboflow.com/project-8nx9x/-b1zsb) |
| 8 | braille (leein) | 466 | detect | [link](https://universe.roboflow.com/leein/braille-jru5w) |
| 9 | 1class_unit_v2 (demo-1g4jj) | 200 | seg | [link](https://universe.roboflow.com/demo-1g4jj/1class_unit_v2) |
| 10 | 1class_unit (demo-1g4jj) | 204 | seg | [link](https://universe.roboflow.com/demo-1g4jj/1class_unit) |
| 11 | b_bv3 (brailblockdetection) | 1,142 | seg | [link](https://universe.roboflow.com/brailblockdetection/b_bv3) |
| 12 | Detecting Braille blocks (hakujou) | 35 | seg | [link](https://universe.roboflow.com/hakujou/detecting-braille-blocks) |
| 13 | My First Project (pandas-32) | 254 | detect | [link](https://universe.roboflow.com/pandas-32/my-first-project-9gzc2) |
| 14 | find braileblock (apple-lehpf) | 297 | detect | [link](https://universe.roboflow.com/apple-lehpf/find-braileblock) |
| 15 | Picture (koreatech-mkeqv) | 5,367 | detect | [link](https://universe.roboflow.com/koreatech-mkeqv/picture-2zlsn) |
| 16 | 점자블록 (brailblockdetection/-vw3qv) | 635 | detect | [link](https://universe.roboflow.com/brailblockdetection/-vw3qv) |
| 17 | 점자블록 구별하기 (project-luraa) | 289 | detect | [link](https://universe.roboflow.com/project-luraa/-kvzhs) |
| 18 | braille_block_full (demo-1g4jj) | 200 | seg | [link](https://universe.roboflow.com/demo-1g4jj/braille_block_full) |
| 19 | y (yolov11-malq7) | 315 | detect | [link](https://universe.roboflow.com/yolov11-malq7/y-w7tqb) |
| 20 | test (demo-1g4jj) | 50 | seg | [link](https://universe.roboflow.com/demo-1g4jj/test-bxq9j) |
| 21 | Sidewalk (ivus-zgxmv) | 1,568 (284 braille) | detect | [link](https://universe.roboflow.com/ivus-zgxmv/sidewalk-icdle) |
| 22 | kickboard (eva-hyfvg) | 497 | detect | [link](https://universe.roboflow.com/eva-hyfvg/kickboard-1ao8q) |
| 23 | platform_rail 2 (lin-justin) | 1,344 | detect | [link](https://universe.roboflow.com/lin-justin/platform_rail-2) |
| 24 | braille block detection (school-f9udu) | 57 | detect | [link](https://universe.roboflow.com/school-f9udu/braille-block-detection) |
| 25 | 1class_full (demo-1g4jj) | 200 | seg | [link](https://universe.roboflow.com/demo-1g4jj/1class_full) |
| 26 | tactile (sefa-edndk) | 523 | seg | [link](https://universe.roboflow.com/sefa-edndk/tactile) |
| 27 | tactile pavement (susam) | 171 | detect | [link](https://universe.roboflow.com/susam/tactile-pavement) |
| 28 | Paving Tactile Detection (raihan-aria) | 2,140 | detect | [link](https://universe.roboflow.com/raihan-aria/paving-tactile-detection) |
| 29 | tactile-paving-detection (thesis-jjmi5) | 206 | detect | [link](https://universe.roboflow.com/thesis-jjmi5/tactile-paving-detection) |
| 30 | tactile paving segmentation (crosswalk-signal-detection) | 35 | seg | [link](https://universe.roboflow.com/crosswalk-signal-detection/tactile-paving-segmentation-eqfcf) |
| 31 | Tactile Guide Path (tony-lo) | 100 | detect | [link](https://universe.roboflow.com/tony-lo/tactile-guide-path) |
| 32 | tactile-paving-segmentation (thesis-jjmi5) | 219 | seg | [link](https://universe.roboflow.com/thesis-jjmi5/tactile-paving-segmentation) |
| 33 | 점자블록1 (klugboard) | 25 | detect | [link](https://universe.roboflow.com/klugboard/-1-w8phx) |
| 34 | guide-bricks-dataset (ntut-ksxbq) | 1,374 | detect | [link](https://universe.roboflow.com/ntut-ksxbq/guide-bricks-dataset-20250625) |
| 35 | Suelo podotactiles (jose-tcf8t) | 100 | detect | [link](https://universe.roboflow.com/jose-tcf8t/suelo-podotactiles) |
| 36 | secmeli-2 (secmeli2) | 171 | detect | [link](https://universe.roboflow.com/secmeli2/secmeli-2) |
| 37 | project (jihoon-xw2ya) | 1,298 | seg | [link](https://universe.roboflow.com/jihoon-xw2ya/project-3xdqo) |
| 38 | blindpath (blindpath) | 55 | detect | [link](https://universe.roboflow.com/blindpath/blindpath) |
| 39 | blindstick (amsrp9) | 88 | detect | [link](https://universe.roboflow.com/amsrp9/blindstick-qiuse) |
| 40 | bpv2 (bpd) | 75 | detect | [link](https://universe.roboflow.com/bpd/bpv2) |
| 41 | Detecting Braille blocks (hakujou) [dup] | 35 | seg | [link](https://universe.roboflow.com/hakujou/detecting-braille-blocks) |
| 42 | imageSegmantashion (helloworld-su7hv) | 75 | seg | [link](https://universe.roboflow.com/helloworld-su7hv/imagesegmantashion) |
| 43 | Blind road (cv-i8trm) | 120 | seg | [link](https://universe.roboflow.com/cv-i8trm/blind-road) |
| 44 | Yellow Guiding Block (tomatt) | 159 | detect | [link](https://universe.roboflow.com/tomatt/yellow-guiding-block) |
| 45 | yellow (jihoon-xw2ya) | 323 | seg | [link](https://universe.roboflow.com/jihoon-xw2ya/yellow-8jvpq) |
| 46 | find apple (apple-lehpf) | 150 | detect | [link](https://universe.roboflow.com/apple-lehpf/find-apple-l1tmd) |
| 47 | furniture model (mhh-hj28t) | 98 | detect | [link](https://universe.roboflow.com/mhh-hj28t/furniture-model) |
| 48 | manhole (jia-he) | 51 | detect | [link](https://universe.roboflow.com/jia-he/manhole-lpao3) |
| 49 | braille block detect (taejang-school) | 150 | detect | [link](https://universe.roboflow.com/taejang-school/-braille-block-detect) |
| 50 | guidage rue (test2-acmaw) | 142 | detect | [link](https://universe.roboflow.com/test2-acmaw/guidage-rue) |
| 51 | TDI_test_3 (boveaurieltravailtds) | 109 | detect | [link](https://universe.roboflow.com/boveaurieltravailtds/tdi_test_3) |
| 52 | 71 (minwoo-kang) | 292 | detect | [link](https://universe.roboflow.com/minwoo-kang/71) |
| 53 | Braille_Blocks_Model (project1-rehwf) | 127 | detect | [link](https://universe.roboflow.com/project1-rehwf/braille_blocks_model) |
| 54 | tenjima50 (kuroko-4tq9e) | 100 | detect | [link](https://universe.roboflow.com/kuroko-4tq9e/tenjima50) |
| 55 | tenjima400 (kuroko-4tq9e) | 350 | detect | [link](https://universe.roboflow.com/kuroko-4tq9e/tenjima400) |
| 56 | Bande de vigilance (bande-de-vigilance) | 107 | detect | [link](https://universe.roboflow.com/bande-de-vigilance/bande-de-vigilance) |
| 57 | split (cv-i8trm) | 120 | seg | [link](https://universe.roboflow.com/cv-i8trm/split-noz5q) |
| 58 | braille block (hongikkyu) | 365 | classify | [link](https://universe.roboflow.com/hongikkyu/braille-block-vnuex) |
| 59 | braille block (klugboard) | 370 | detect | [link](https://universe.roboflow.com/klugboard/braille-block-meut1) |
| 60 | Detection Braille-block (jin-lhvrf) | 3,009 | detect | [link](https://universe.roboflow.com/jin-lhvrf/detection-braille-block-xhazj) |
| 61 | braille block poly (school-f9udu) | 38 | detect | [link](https://universe.roboflow.com/school-f9udu/braille-block-poly) |
| 62 | Braille Block Detect (yourking) | 787 | detect | [link](https://universe.roboflow.com/yourking/braille-block-detect) |

**Aggregate totals (pre-curation):**
- Detection: 19,925 items
- Segmentation: 5,192 items
- Non-usable segmentation: 785 items
- Classification: 365 items
- Image-only (no annotations): 459 items

## Appendix B: BLV Objects Dataset (#TODO: Cleanup and aggregate)

| # | Project | Items | Type | URL | Classes |
|---|---------|-------|------|-----|---------|
| 1 | Elevator Buttons - Original Version (unlimited-robotics) | 2002 | detect | [link](https://universe.roboflow.com/unlimited-robotics/elevator-buttons-original-version) | Elevator Button, Door button, Push button |
| 2 | Elevator Buttons (elevatordataset) | 2019 | detect | [link](https://universe.roboflow.com/elevatordataset/elevator-buttons-bdt8h) | Elevator Button, Door button, Push button |
| 3 | open elevator (elevator-0iq4p) | 205 | detect | [link](https://universe.roboflow.com/elevator-0iq4p/open-elevator) | Elevator, Door |
| 4 | Elevator_Buttons (SeniorDesign) | 34 | detect | [link](https://universe.roboflow.com/seniordesign-6mvd2/elevator_buttons) | Elevator Button, Door button, Push button |
| 5 | Elevator buttons (elevator-buttons) | 55 | detect | [link](https://universe.roboflow.com/elevator-buttons/elevator-buttons-unhwb) | Elevator Button, Door button, Push button |
| 6 | pedestrian lights (FYP Crossing Roads) | 626 | detect | [link](https://universe.roboflow.com/fyp-crossing-roads/pedestrian-lights-0bbig) | Pedestrian signal, Traffic-signal |
| 7 | Traffic Signs Dataset (Linköping Univ.) | 20000+ | detect | [link](https://www.cvl.isy.liu.se/research/datasets/traffic-signs-dataset/) | Cross-walk |
| 8 | DeepDoors2 (DeepDoors2) | 6000 | seg | [link](https://datasetninja.com/deep-doors2) | Door |
| 9 | yolov8-crosswalk-detection (tcc-xn3st) | 1049 | detect | [link](https://universe.roboflow.com/tcc-xn3st/yolov8-crosswalk-detection) | Cross-walk |
| 10 | LISA Traffic Light (LISA-Traffic-Light) | 43016 | detect | [link](https://datasetninja.com/lisa-traffic-light) | Traffic-signal |
| 11 | pedestrian Traffic Light (ono-gedd7) | 926 | detect | [link](https://universe.roboflow.com/ono-gedd7/pedestrian-traffic-light-puf4a) | Pedestrian signal, Traffic-signal |
| 12 | Pedestrian Traffic Lights detection (Mendeley) | 809 | cls | [link](https://data.mendeley.com/datasets/9tm59d3nsn/1) | Pedestrian signal, Traffic-signal |
| 13 | Bus Stop Detection (Testing-meesi) | 470 | detect | [link](https://universe.roboflow.com/testing-meesi/bus-stop-detection-cjrim) | Bus-stop, Bus-stop sign stand, Cross-walk, Traffic-signal, Pedestrian signal |
| 14 | Bus Stop Detection (Univ. of Hong Kong) | 544 | detect | [link](https://universe.roboflow.com/the-university-of-hong-kong-zsdbj/bus-stop-detection) | Bus-stop, Bus-stop sign stand |
| 15 | Argoverse-HD (Argoverse-HD) | 66953 | detect | [link](https://datasetninja.com/argoverse-hd) | Traffic-signal |
| 16 | elevator-button (elevatornumber) | 79 | detect | [link](https://universe.roboflow.com/elevatornumber/elevator-button-jvh8p) | Elevator Button, Door button, Push button |
| 17 | Elevator Button Recognition (Sun Moon University) | 1600 | detect | [link](https://universe.roboflow.com/sun-moon-university/elevator-button-recognition) | Elevator Button, Door button, Push button |
| 18 | FPVCrosswalk2025 (Mendeley) | 3300 | seg | [link](https://data.mendeley.com/datasets/mcr2jwk5bp/1) | Cross-walk |
| 19 | town-samedata | 787 | detect | [link](https://universe.roboflow.com/test2-hopj9/town-samedata) | Elevator, Door, Cross-walk, Pedestrian signal, Bus-stop, Traffic-signal |
| 20 | door-button | 3809 | detect | [link](https://universe.roboflow.com/prithvi-tm/door-button) | Door button |
| 21 | elevator-buttons | 363 | detect | [link](https://universe.roboflow.com/school-xaqzz/elevator-buttons-5muuc) | Elevator Button |
| 22 | elevator button detection | 921 | detect | [link](https://universe.roboflow.com/school-3rzal/elevator-button-detection-miikd) | Elevator Button |
| 23 | DA2 | 2400 | detect | [link](https://universe.roboflow.com/odpredatrian/da2-fl9qe) | Cross-walk, Elevator |
| 24 | blind | 1880 | detect | [link](https://universe.roboflow.com/yolontut/blind) | Cross-walk |
| 25 | Quicc Testt | 37 | detect | [link](https://universe.roboflow.com/cardiff-university/quicc-testt) | Door button |
| 26 | Road segmentation | 7500 | seg | [link](https://universe.roboflow.com/object-detection-yy9t1/road-segmentation-olxqu) | Cross-walk, Traffic-signal |
| 27 | Cityscapes-external-v1 | 2650 | seg | [link](https://universe.roboflow.com/polesdataset/cityscapes-external-v1) | Traffic-signal |
| 28 | N_O_1 | 9900 | detect | [link](https://universe.roboflow.com/dice-6h29i/n_o_1) | Cross-walk, Traffic-signal |
| 29 | object detection | 9990 | detect | [link](https://universe.roboflow.com/sign-boards/object-detection-5zcsg) | Cross-walk, Traffic-signal |
| 30 | pedestrian-signal-lights | 13049 | detect | [link](https://universe.roboflow.com/pedestrian-traffic-signal/pedestrian-signal-lights-d2upo/dataset/2) | Pedestrian signal |
| 31 | ped-push-button | 781 | detect | [link](https://universe.roboflow.com/garbage-y0auo/ped-push-button/dataset/1) | Push button |
| 32 | bus stop case | 744 | detect | [link](https://universe.roboflow.com/summer-2025/bus-stop-case) | Bus-stop |
| 33 | BUTTON DATASET (OCR-RCNN) | 3718 | detect | [link](https://github.com/zhudelong/elevator_button_recognition) | Elevator Button |
| 34 | blindpeople | 2440 | detect | [link](https://universe.roboflow.com/yolov4-zrnkl/blindpeople) | Cross-walk, Bus-stop sign stand, Traffic-signal |
| 35 | door-detection | 297 | detect | [link](https://universe.roboflow.com/door-object-detection/door-detection-26wc3) | Door |
| 36 | Open & Closed door Detection | 240 | cls | [link](https://universe.roboflow.com/open-closed-doors-detection/open-closed-door-detection) | Door |
| 37 | door-detection | 1993 | detect | [link](https://universe.roboflow.com/nathaly-espinoza/door-detection-zqt59) | Door |
| 38 | door_project | 129 | seg | [link](https://universe.roboflow.com/sbu/door_project) | Door |
| 39 | Pedestrian-crossing | 1490 | seg | [link](https://universe.roboflow.com/avrorav/pedestrian-crossing-2) | Cross-walk |
| 40 | BLV-Road-Nav-Accessibility (Shohan29531) | 21 videos | detect | [link](https://github.com/Shohan29531/BLV-Road-Nav-Accessibility) | 90 accessibility objects |
