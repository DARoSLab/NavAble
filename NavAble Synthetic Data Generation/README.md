# NavAble: Synthetic 3D Asset Generation for BLV Navigation

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Dataset on HF](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/NavAble/NeurIPS_2026_BLV)

This repository implements the **3D asset generation pipeline** described in Section 3.2 of the NavAble paper. It converts web-crawled images of urban accessibility infrastructure into simulation-ready 3D assets (.glb, .ply, .usdz) for training and evaluating navigation systems that assist **blind and low-vision (BLV)** individuals.

The generated assets are released on Hugging Face: [NavAble/NeurIPS_2026_BLV](https://huggingface.co/datasets/NavAble/NeurIPS_2026_BLV)

## Asset Generation Pipeline

```
┌──────────┐   ┌──────────┐   ┌──────────────┐   ┌──────────┐   ┌──────────┐
│  Image   │──▶│  VLM     │──▶│  Grounded    │──▶│  SAM-3D  │──▶│  USD/Z   │
│ Crawling │   │ Validation│  │  SAM 2       │   │ Objects  │   │ Convert  │
└──────────┘   └──────────┘   └──────────────┘   └──────────┘   └──────────┘
  Wikimedia     Gemini          Class-aware        3D Mesh        GLB/PLY
  Bing          Filtering       Segmentation       Generation     → USDZ
```

Each stage is pluggable via the strategy pattern -- swap implementations by editing `configs/pipeline.yaml`.

## Dataset Statistics

The released 3D asset library contains **500** assets across **9** BLV-relevant object classes, with each asset providing a `.glb` mesh, a Gaussian-splat `.ply`, and a `.usdz` bundle.

| Class | Description |
|-------|-------------|
| aps_button | Accessible pedestrian signal buttons |
| bus_stop | Bus stop shelters and structures |
| door_button | ADA-compliant automatic door opener buttons |
| elevator | Elevator entrances (doors and frame) |
| elevator_button | Elevator call buttons and control panels |
| escalator | Mechanical escalators |
| handrail | Structural handrails in public spaces |
| pedestrian_signal | Pedestrian crossing signals |
| turnstile | Full-height turnstiles and revolving doors |

| Metric | Value |
|--------|-------|
| Total 3D Assets | 500 |
| Object Classes | 9 |
| Output Formats | GLB, PLY, USDZ |
| Total Size | ~23 GB |

Browse the assets: [synthetic_objects/](https://huggingface.co/datasets/NavAble/NeurIPS_2026_BLV/tree/main/synthetic_objects)

## Directory Structure

```
NavAble/
├── README.md
├── LICENSE
├── pyproject.toml
├── src/isaacnav/
│   ├── pipeline.py
│   ├── crawling/
│   ├── masking/
│   ├── reconstruction/
│   ├── conversion/
│   ├── scoring/
│   └── data/
├── scripts/
│   ├── run_pipeline.py
│   └── run_batch.py
├── configs/
│   ├── pipeline.yaml
│   ├── object_categories.yaml
│   └── object_categories_high_lev.yaml
├── extern/
│   ├── sam-3d-objects/
│   └── Grounded-SAM-2/
├── envs/
│   ├── environment.yaml
│   └── install.sh
├── data/
└── output/
```

## Installation

### Prerequisites

- Linux with NVIDIA GPU (driver >= 570)
- [Conda](https://docs.conda.io/) or [Mamba](https://mamba.readthedocs.io/)
- Git with submodule support
- [HuggingFace CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) (for SAM-3D checkpoints)

### Setup

```bash
git clone https://github.com/NavAble/NavAble-Synthetic-Data-Generation.git
cd NavAble-Synthetic-Data-Generation

# Initialize submodules
git submodule update --init --recursive

# One-command setup (creates conda env, installs deps, downloads checkpoints)
bash envs/install.sh

# Activate environment
conda activate isaacnav
```

The install script handles:

1. Creating a conda environment with Python 3.11 and CUDA 12.8
2. Installing SAM-3D-objects with PyTorch3D compiled for your GPU
3. Installing Grounded SAM 2 (SAM 2.1 + Grounding DINO)
4. Installing the NavAble package
5. Downloading model checkpoints (SAM ViT-H, SAM 2.1, SAM-3D)
6. Verifying the installation

### Environment Details

The environment targets CUDA 12.8 and PyTorch 2.7.1. Key dependencies are defined in `envs/environment.yaml` and include:

- **3D processing**: trimesh, Open3D, PyTorch3D, Kaolin, gsplat
- **Segmentation**: SAM, SAM 2.1, Grounding DINO
- **Reconstruction**: SAM-3D-objects (Meta)
- **Validation**: Gemini API (via `GEMINI_API_KEY` env var)

## Usage

### Generate Assets from a Single Image

```bash
python scripts/run_pipeline.py --config configs/pipeline.yaml \
    --image data/input/elevator/example.jpg --classes "Elevator"
```

### Process a Directory of Images

```bash
python scripts/run_pipeline.py --config configs/pipeline.yaml \
    --image-dir data/input/
```

### Crawl and Process End-to-End

```bash
python scripts/run_pipeline.py --config configs/pipeline.yaml --crawl
```

### Batch Processing

```bash
python scripts/run_batch.py create-manifest \
    --image-dir data/input/ --classes "Elevator" "Escalator" \
    --manifest data/jobs/batch.json

python scripts/run_batch.py run --manifest data/jobs/batch.json
```

### Download Released Assets

```python
from huggingface_hub import snapshot_download

# All assets for a single class
snapshot_download(
    repo_id="NavAble/NeurIPS_2026_BLV", repo_type="dataset",
    allow_patterns=["synthetic_objects/elevator/**"],
    local_dir="./data/assets"
)

# All 3D assets
snapshot_download(
    repo_id="NavAble/NeurIPS_2026_BLV", repo_type="dataset",
    allow_patterns=["synthetic_objects/**"],
    local_dir="./data/assets"
)
```

### Load in Isaac Sim

```python
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage

world = World()
add_reference_to_stage(
    usd_path="data/assets/synthetic_objects/elevator/asset_001/mesh.usdz",
    prim_path="/World/Elevator"
)
world.reset()
```

## Configuration

Edit `configs/pipeline.yaml` to control the pipeline. Key options:

```yaml
target_classes:
  - "Elevator"
  - "Escalator"
  # ...

masking:
  strategy: "grounded_sam2"   # or "sam_heuristic"

reconstruction:
  strategy: "sam3d"
```

## Data Sources & Licensing

All source images are collected from platforms with permissive licenses:

| Source | License | Attribution Required |
|--------|---------|---------------------|
| Wikimedia Commons | CC BY / CC BY-SA / CC0 | Yes (except CC0) |
| Flickr | CC BY / CC0 / Public Domain | Yes (except CC0/PD) |
| Pexels | Pexels License | No |

## Acknowledgments

- [SAM-3D-objects](https://github.com/facebookresearch/sam-3d-objects) for 3D reconstruction
- [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) for class-aware segmentation
- [Segment Anything Model 2](https://github.com/facebookresearch/sam2) for image segmentation
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) for open-vocabulary object detection
- [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim) for simulation environment

## License

This dataset is released under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

See [LICENSE](LICENSE) for full terms.
