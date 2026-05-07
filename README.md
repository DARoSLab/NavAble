# NavAble: A Synthetic Data Pipeline and Benchmark for BLV Navigation

[![License: CC BY 4.0](https://img.shields.io/badge/Dataset-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![License: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Dataset on HF](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/NavAble/NeurIPS_2026_BLV)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org/)

> **NeurIPS 2026 Datasets and Benchmarks Track Submission**

NavAble is an end-to-end framework for generating, collecting, and evaluating synthetic data for semantic segmentation of **blind and low-vision (BLV)** accessibility infrastructure. The project spans three stages: (1) automated generation of simulation-ready 3D assets from web-crawled images, (2) multi-modal synthetic data collection in NVIDIA Isaac Sim, and (3) benchmark evaluation of five semantic segmentation architectures across real, curated, and synthetic data configurations.

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Components Overview](#components-overview)
  - [1. Synthetic Data Generation Pipeline](#1-synthetic-data-generation-pipeline)
  - [2. Synthetic Data Collector (Isaac Sim)](#2-synthetic-data-collector-isaac-sim)
  - [3. Evaluation — SAN](#3-evaluation--san-mask2former-segformer)
  - [4. Evaluation — DeepLabV3+, EncNet, Mask2Former, SegFormer](#4-evaluation--deeplabv3-encnet-mask2former-segformer)
- [Class Taxonomy](#class-taxonomy)
- [Dataset Access](#dataset-access)
- [Quick Start](#quick-start)
- [Key Results](#key-results)
- [Hardware Requirements](#hardware-requirements)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Repository Structure

```
NavAble/
├── NavAble Synthetic Data Generation/    # 3D asset generation pipeline (image → GLB/PLY → USDZ)
│   ├── src/isaacnav/                     #   Core library: crawling, validation, masking, reconstruction, conversion
│   ├── configs/                          #   Pipeline & taxonomy YAML configs
│   ├── scripts/                          #   CLI entry points (run_pipeline.py, run_batch.py)
│   ├── envs/                             #   Conda environment & one-command installer
│   └── extern/                           #   Git submodules (SAM-3D-objects, Grounded-SAM-2)
│
├── navable_synth_data_collector/         # Isaac Sim extension for synthetic data collection
│   ├── backend/                          #   Session, capture, trajectory, gamepad, stage control
│   ├── ui/sections/                      #   GUI panels (8 collapsible sections)
│   ├── cli/                              #   Headless CLI (navable-collect)
│   ├── config/                           #   Extension & project YAML config
│   └── tests/                            #   Unit tests (no Isaac Sim required)
│
└── evaluation-code/                      # Semantic segmentation evaluation
    ├── san/                              #   SAN + Mask2Former + SegFormer (custom BLV pipeline)
    │   ├── blv_pipeline/                 #     Custom MMSeg plugins (datasets, metrics, decode heads)
    │   ├── configs/Final/                #     Paper-canonical configs (A/B/C/D × 3 architectures)
    │   └── tools/                        #     SLURM launchers, data conversion, checkpoint download
    │
    └── deeplabv3-encnet-mask2former-     #   DeepLabV3+, EncNet, Mask2Former, SegFormer (MMSeg)
        segformer/
        ├── mmseg/                        #     Custom dataset class + FgIoU metric
        ├── configs/navable/              #     4 models × 6 dataset settings
        └── data_processing/              #     COCO → MMSeg & Synthetic → MMSeg converters
```

---

## Components Overview

### 1. Synthetic Data Generation Pipeline

**Directory:** `NavAble Synthetic Data Generation/`

A configurable, strategy-pattern-based pipeline that converts web-crawled images of urban accessibility infrastructure into simulation-ready 3D assets (`.glb`, `.ply`, `.usdz`).

#### Pipeline Stages

```
┌──────────┐   ┌──────────┐   ┌──────────────┐   ┌──────────┐   ┌──────────┐
│  Image   │──▶│   VLM    │──▶│  Grounded    │──▶│  SAM 3D  │──▶│  USD/Z   │
│ Crawling │   │Validation│   │  SAM 2       │   │ Objects  │   │ Convert  │
└──────────┘   └──────────┘   └──────────────┘   └──────────┘   └──────────┘
  Wikimedia     Gemini          Class-aware        3D Mesh        GLB/PLY
  Bing/DDG      Filtering       Segmentation       Generation     → USDZ
```

| Stage | Description | Key Models |
|-------|-------------|------------|
| **Crawl** | Multi-source image acquisition (Wikimedia, Bing, DuckDuckGo) | — |
| **Validate** | VLM-driven class verification and reconstruction suitability check | Gemini |
| **Segment** | Class-aware open-vocabulary segmentation with per-class prompt expansion | Grounding DINO + SAM 2.1 |
| **Reconstruct** | Single-image 3D mesh and Gaussian-splat generation | SAM 3D Objects (Meta) |
| **Convert** | Mesh → USD/USDZ packaging for Isaac Sim compatibility | trimesh, Open3D, pxr |

#### Key Design Decisions

- **Per-class prompt expansion**: composite structural objects (e.g., escalators, APS buttons) are segmented using multiple sub-component DINO prompts whose masks are unioned into a single coherent object mask.
- **VLM-driven class disambiguation**: paragraph-form accept/reject prompts enforce precise class definitions, filtering out concept drift from open-vocabulary web search.
- **Subprocess isolation**: USD conversion runs in a separate process to avoid `spconv` × `pxr` CUDA conflicts.

#### Setup & Usage

```bash
cd "NavAble Synthetic Data Generation"
git submodule update --init --recursive
bash envs/install.sh
conda activate isaacnav

# Single image
python scripts/run_pipeline.py --config configs/pipeline.yaml \
    --image data/input/elevator/example.jpg --classes "Elevator"

# Full crawl-to-asset pipeline
python scripts/run_pipeline.py --config configs/pipeline.yaml --crawl
```

📄 See [`NavAble Synthetic Data Generation/README.md`](NavAble%20Synthetic%20Data%20Generation/README.md) for full documentation.

---

### 2. Synthetic Data Collector (Isaac Sim)

**Directory:** `navable_synth_data_collector/`

An NVIDIA Isaac Sim 5.1 / Omniverse Kit extension for collecting multi-modal synthetic training data (RGB, semantic segmentation, 2D bounding boxes) using the generated 3D assets.

#### Features

| Feature | Description |
|---------|-------------|
| **Gamepad FPS camera** | Fly through USD environments with an XInput controller |
| **Trajectory record/playback** | Record camera paths to JSON; replay deterministically |
| **Replicator capture** | Multi-modal output via `omni.replicator.core` |
| **Asset browser** | Cycle through USD assets with automatic semantic labeling |
| **Location management** | Named spawn points per environment with saved transforms |
| **Collect-all automation** | One-click matrix capture: environment × location × trajectory × asset |
| **Headless CLI** | `navable-collect` command for batch production runs |

#### Setup

1. Copy the extension into Isaac Sim's extension search path
2. Enable **"NavAble Synthetic Data Extension"** in the Extension Manager
3. Configure paths in `config/config.yaml`

```bash
# Headless batch capture
pip install -e .[test]
navable-collect collect-all --config config/config.yaml
```

📄 See [`navable_synth_data_collector/README.md`](navable_synth_data_collector/README.md) for full documentation.

---

### 3. Evaluation — SAN, Mask2Former, SegFormer

**Directory:** `evaluation-code/san/`

Custom BLV segmentation pipeline built on MMSegmentation, evaluating three architectures (SAN, Mask2Former, SegFormer) across four data configurations.

#### Data Configurations

| Config | Training Data | Description |
|--------|---------------|-------------|
| **A** | `real_final` (3,703 images) | Real data only |
| **B** | `real_final` + `opensrc_final` (~40K) | Real + open-source curated |
| **C** | `real_final` ×6 + `synth_0.1` (~42K) | Real (oversampled) + synthetic |
| **D** | `real_final` ×12 + `synth_0.2` (~85K) | Real (oversampled) + more synthetic |

#### Custom Components

- **`BLVMetric`** — mIoU, mAP₅₀₋₉₅, Precision, Recall; excludes turnstile (zero GT coverage in real data)
- **`BLVDatasetV2Fg`** — foreground-only evaluation (`reduce_zero_label=True`)
- **Per-query fg filtering** — zeros out low-confidence queries in Mask2Former/SAN to prevent background-dominant query suppression

#### Setup & Usage

```bash
cd evaluation-code/san
conda env create -f environment.yml
conda activate openmmlab
export BLV_PROJECT_ROOT="$(pwd)"

# Download pretrained checkpoints
python tools/blv/download_checkpoints.py

# Train
sbatch tools/Final/slurm_train_final.sh segformer A

# Evaluate
sbatch tools/Final/slurm_eval_final_no_vis.sh mask2former C <CKPT_PATH>
```

📄 See [`evaluation-code/san/README.md`](evaluation-code/san/README.md) for full documentation.

---

### 4. Evaluation — DeepLabV3+, EncNet, Mask2Former, SegFormer

**Directory:** `evaluation-code/deeplabv3-encnet-mask2former-segformer/`

Standard MMSegmentation-based evaluation of four architectures across six dataset settings, designed as drop-in plugins for any MMSeg installation.

#### Models & Dataset Settings

| Model | Backbone | Crop Size |
|-------|----------|-----------|
| DeepLabV3+ | ResNet-101-D8 | 512×512 |
| EncNet | ResNet-101-D8 | 512×512 |
| Mask2Former | Swin-L (IN-22K) | 640×640 |
| SegFormer | MiT-B5 | 640×640 |

| Setting | Description |
|---------|-------------|
| `real_only` | Real dataset only |
| `real` | Real + curated synthetic |
| `realsyn` | Real + full synthetic |
| `realsyn_nocur` | Real + synthetic (no curation) |
| `realsyn_partial` | Real + curated synthetic (subset) |
| `realsyn_nocur_partial` | Real + uncurated synthetic (subset) |

#### Custom Components

- **`MyMMSegDataset`** — 12-class NAVable dataset registration
- **`FgIoUMetric`** — mIoU + foreground-only mIoU (`fg_mIoU`), best-model selection on `fg_mIoU`
- **Data processing scripts** — COCO → MMSeg and Synthetic → MMSeg format converters

#### Setup & Usage

```bash
# Clone MMSegmentation and install
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation && pip install -e .

# Copy NavAble plugins into MMSeg
cp -r <this-repo>/mmseg/ mmsegmentation/mmseg/
cp -r <this-repo>/configs/navable/ mmsegmentation/configs/

# Train
python tools/train.py configs/navable/realsyn/encnet.py

# Evaluate
python tools/test.py configs/navable/realsyn/encnet.py <checkpoint.pth>
```

📄 See [`evaluation-code/deeplabv3-encnet-mask2former-segformer/README.md`](evaluation-code/deeplabv3-encnet-mask2former-segformer/README.md) for full documentation.

---

## Class Taxonomy

NavAble defines **11 foreground classes** of BLV-relevant urban accessibility infrastructure (+ background):

| ID | Class | Description |
|----|-------|-------------|
| 0 | `background` | Non-accessibility scene content |
| 1 | `elevator` | Elevator entrances (doors and frame) |
| 2 | `elevator_button` | Elevator call buttons and control panels |
| 3 | `door_button` | ADA-compliant automatic door opener buttons |
| 4 | `crosswalk` | Pedestrian crosswalks |
| 5 | `pedestrian_signal` | Pedestrian crossing signals |
| 6 | `aps_button` | Accessible pedestrian signal buttons |
| 7 | `bus_stop` | Bus stop shelters and structures |
| 8 | `bus_stop_sign` | Bus stop signage |
| 9 | `handrail` | Structural handrails in public spaces |
| 10 | `escalator` | Mechanical escalators |
| 11 | `turnstile` | Full-height turnstiles and revolving doors |

---

## Dataset Access

The NavAble dataset is hosted on Hugging Face: [**NavAble/NeurIPS_2026_BLV**](https://huggingface.co/datasets/NavAble/NeurIPS_2026_BLV)

```python
from huggingface_hub import snapshot_download

# Download all 3D assets (~23 GB)
snapshot_download(
    repo_id="NavAble/NeurIPS_2026_BLV", repo_type="dataset",
    allow_patterns=["synthetic_objects/**"],
    local_dir="./data/assets"
)
```

| Component | Details |
|-----------|---------|
| **3D Assets** | 500 assets across 9 classes (GLB, PLY, USDZ) |
| **Real Data** | ~3,700 images with COCO-format polygon annotations |
| **Synthetic Data** | Multi-modal renders (RGB + semantic segmentation + bounding boxes) |

---

## Quick Start

### Option A: Generate 3D Assets

```bash
cd "NavAble Synthetic Data Generation"
bash envs/install.sh && conda activate isaacnav
python scripts/run_pipeline.py --config configs/pipeline.yaml --crawl
```

### Option B: Collect Synthetic Training Data

```bash
# Install the Isaac Sim extension, then:
navable-collect collect-all --config config/config.yaml
```

### Option C: Train & Evaluate Segmentation Models

```bash
# SAN / Mask2Former / SegFormer
cd evaluation-code/san
conda env create -f environment.yml && conda activate openmmlab
sbatch tools/Final/slurm_train_final.sh mask2former C

# DeepLabV3+ / EncNet
cd evaluation-code/deeplabv3-encnet-mask2former-segformer
python tools/train.py configs/navable/realsyn/deeplabv3plus.py
```

---

## Key Results

Foreground mIoU (fg_mIoU) on `real_final/test` (1,482 images, turnstile excluded):

| Architecture | A · Real Only | B · +Open-Source | C · +Synth₀.₁ | D · +Synth₀.₂ |
|:-------------|:-------------:|:----------------:|:--------------:|:--------------:|
| SegFormer    | 48.52         | 48.01            | 56.39          | **58.35**      |
| Mask2Former  | 68.10         | 66.57            | 72.46          | **72.64**      |
| SAN          | 62.07         | 62.52            | **68.12**      | 66.96          |

**Key finding:** Synthetic data (Config C) improves every architecture over real-only baselines: **+7.87** mIoU (SegFormer), **+4.36** (Mask2Former), **+6.05** (SAN). Context-aware architectures (Mask2Former, SAN, SegFormer) benefit substantially from synthetic augmentation, while traditional CNN-based models (DeepLabV3+, EncNet) show diminishing or negative returns — highlighting that bridging the sim-to-real gap requires architectures capable of learning domain-invariant contextual representations.

---

## Hardware Requirements

| Component | Requirements |
|-----------|-------------|
| **SDG Pipeline** | Linux, NVIDIA RTX GPU (driver ≥ 570), CUDA 12.8, Python 3.11 |
| **Data Collector** | Isaac Sim 5.1.0, NVIDIA RTX GPU, XInput gamepad (optional) |
| **Evaluation** | CUDA 12.1, NVIDIA L40S / RTX 4090 (tested), Python 3.10 |

---

## License

| Component | License |
|-----------|---------|
| **Dataset** (3D assets, annotations) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| **Code** (pipelines, evaluation) | [MIT](https://opensource.org/licenses/MIT) |

---

## Acknowledgments

This project builds on the following open-source tools:

- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) — Semantic segmentation framework
- [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) — Single-image 3D reconstruction
- [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) — Open-vocabulary segmentation
- [Segment Anything Model 2](https://github.com/facebookresearch/sam2) — Foundation segmentation model
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) — Open-vocabulary object detection
- [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim) — Simulation platform
- [NVIDIA Omniverse Replicator](https://developer.nvidia.com/omniverse/replicator) — Synthetic data generation
