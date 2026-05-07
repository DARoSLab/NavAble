#!/bin/bash
# NavAble Pipeline Installation Script
# ======================================
# One-command setup for the complete pipeline.
# Targets: CUDA 12.8, PyTorch 2.7.1, RTX 5090 (Blackwell SM_120).
#
# Usage:
#   bash envs/install.sh
#
# Prerequisites:
#   - conda or mamba installed
#   - NVIDIA GPU with driver >= 570 (RTX 5090 / Blackwell)
#   - Git with submodule support
#   - HuggingFace CLI (for SAM3D checkpoints)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo "NavAble Pipeline Installation"
echo "============================================"
echo "Repo root: $REPO_ROOT"
echo "Target:    CUDA 12.8 / PyTorch 2.7.1 / RTX 5090"
echo ""

# Detect conda/mamba
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
else
    echo "ERROR: conda or mamba not found. Please install Miniconda or Mambaforge."
    exit 1
fi
echo "Using: $CONDA_CMD"

# Verify GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
    echo "Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo 'unknown')"
fi

# ============================================
# Step 1: Initialize git submodules
# ============================================
echo ""
echo "[1/7] Initializing git submodules..."
cd "$REPO_ROOT"
git submodule update --init --recursive

# ============================================
# Step 2: Create conda environment
# ============================================
echo ""
echo "[2/7] Creating conda environment (Python 3.11 + CUDA 12.8 toolkit)..."
$CONDA_CMD env create -f envs/environment.yaml 2>/dev/null \
    || $CONDA_CMD env update -f envs/environment.yaml --prune

# Activate
eval "$(conda shell.bash hook)"
conda activate isaacnav

# Set CUDA_HOME to the conda env's CUDA toolkit
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

echo "CUDA_HOME=$CUDA_HOME"
echo "nvcc version: $(nvcc --version 2>/dev/null | grep release || echo 'not found')"

# Pip index URLs for NVIDIA packages
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu128"

# ============================================
# Step 3: Install SAM-3D-objects
# ============================================
echo ""
echo "[3/7] Installing SAM-3D-objects..."
if [ -d "extern/sam-3d-objects" ]; then
    cd extern/sam-3d-objects

    # IMPORTANT: SAM3D's requirements.txt pins CUDA 12.1 packages (torchaudio==2.5.1+cu121,
    # spconv-cu121, cuda-python==12.1.0, etc.) which conflict with our CUDA 12.8 env.
    # We use --no-deps to register the sam3d_objects package without pulling those pins,
    # since all compatible versions are already installed via environment.yaml.
    echo "  Installing sam3d_objects package (--no-deps, deps from environment.yaml)..."
    pip install --no-deps -e .

    # Dev extras (not CUDA-dependent)
    echo "  Installing dev extras..."
    pip install pytest findpydeps pipdeptree lovely_tensors

    # SAM3D-specific packages not in environment.yaml but needed at runtime
    echo "  Installing additional SAM3D runtime dependencies..."
    pip install --no-deps easydict rootutils randomname gdown ftfy h5py av decord

    # PyTorch3D -- must be compiled from source for Blackwell SM_120
    echo "  Building PyTorch3D from source (this may take 10-15 minutes)..."
    export FORCE_CUDA=1
    export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"
    pip install --no-build-isolation --no-deps \
        "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47"

    # flash_attn (already in environment.yaml, but verify)
    python -c "import flash_attn" 2>/dev/null || pip install flash_attn==2.8.3

    # Kaolin + gsplat (inference extras)
    echo "  Installing Kaolin + gsplat..."
    export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.7.1_cu128.html"
    pip install kaolin==0.17.0
    pip install --no-build-isolation \
        "gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7"

    # seaborn already in environment.yaml; gradio for demo UI
    pip install gradio==5.49.0

    # Patch hydra
    if [ -f "patching/hydra" ]; then
        echo "  Applying hydra patch..."
        python patching/hydra
    fi

    cd "$REPO_ROOT"
else
    echo "  WARNING: extern/sam-3d-objects not found. Run: git submodule update --init"
fi

# ============================================
# Step 4: Install Grounded SAM 2
# ============================================
echo ""
echo "[4/7] Installing Grounded SAM 2 (SAM 2.1 + Grounding DINO)..."
if [ -d "extern/Grounded-SAM-2" ]; then
    cd extern/Grounded-SAM-2

    # Install SAM 2 (compiles CUDA connected_components op)
    pip install -e .

    # Install Grounding DINO (compiles CUDA deformable attention op)
    # --no-build-isolation: its setup.py tries to pip install torch in a clean env,
    # but torch is already installed and must be reused for CUDA 12.8 compat
    if [ -d "grounding_dino" ]; then
        cd grounding_dino
        pip install --no-build-isolation -e .
        cd ..
    fi

    cd "$REPO_ROOT"
else
    echo "  WARNING: extern/Grounded-SAM-2 not found. Run: git submodule update --init"
fi

# ============================================
# Step 5: Install NavAble package
# ============================================
echo ""
echo "[5/7] Installing NavAble package..."
cd "$REPO_ROOT"
pip install --no-build-isolation -e .

# ============================================
# Step 6: Download model checkpoints
# ============================================
echo ""
echo "[6/7] Downloading model checkpoints..."

# SAM ViT-H checkpoint (for heuristic baseline)
SAM_CHECKPOINT="extern/sam-3d-objects/checkpoints/sam_vit_h_4b8939.pth"
if [ -d "extern/sam-3d-objects" ] && [ ! -f "$SAM_CHECKPOINT" ]; then
    echo "  Downloading SAM ViT-H checkpoint (~2.5 GB)..."
    mkdir -p "$(dirname $SAM_CHECKPOINT)"
    wget -q --show-progress -O "$SAM_CHECKPOINT" \
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
else
    echo "  SAM ViT-H checkpoint: OK"
fi

# SAM3D checkpoints (requires HuggingFace authentication)
if [ -d "extern/sam-3d-objects" ] && [ ! -d "extern/sam-3d-objects/checkpoints/hf" ]; then
    echo "  Downloading SAM-3D-objects checkpoints from HuggingFace..."
    echo "  (Requires: huggingface-cli login && access request at https://huggingface.co/facebook/sam-3d-objects)"
    pip install -q 'huggingface-hub[cli]<1.0' 2>/dev/null
    TAG=hf
    hf download \
        --repo-type model \
        --local-dir extern/sam-3d-objects/checkpoints/${TAG}-download \
        --max-workers 1 \
        facebook/sam-3d-objects 2>/dev/null \
        && mv extern/sam-3d-objects/checkpoints/${TAG}-download/checkpoints extern/sam-3d-objects/checkpoints/${TAG} \
        && rm -rf extern/sam-3d-objects/checkpoints/${TAG}-download \
        || echo "  Note: SAM3D checkpoints require HuggingFace access request"
else
    echo "  SAM3D checkpoints: OK"
fi

# SAM 2.1 checkpoint
SAM2_CHECKPOINT="extern/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
if [ -d "extern/Grounded-SAM-2" ] && [ ! -f "$SAM2_CHECKPOINT" ]; then
    echo "  Downloading SAM 2.1 Hiera Large checkpoint..."
    mkdir -p "$(dirname $SAM2_CHECKPOINT)"
    wget -q --show-progress -O "$SAM2_CHECKPOINT" \
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
else
    echo "  SAM 2.1 checkpoint: OK"
fi

# ============================================
# Step 7: Verify installation
# ============================================
echo ""
echo "[7/7] Verifying installation..."

python -c "
import torch
print(f'  PyTorch:    {torch.__version__}')
print(f'  CUDA:       {torch.version.cuda}')
print(f'  cuDNN:      {torch.backends.cudnn.version()}')
print(f'  GPU:        {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NOT AVAILABLE\"}')
print(f'  SM arch:    {torch.cuda.get_device_capability(0) if torch.cuda.is_available() else \"N/A\"}')
" 2>/dev/null || echo "  WARNING: PyTorch verification failed"

python -c "import isaacnav; print(f'  NavAble:    {isaacnav.__version__}')" 2>/dev/null || echo "  WARNING: navable import failed"
python -c "import trimesh; print(f'  Trimesh:    {trimesh.__version__}')" 2>/dev/null || true
python -c "import open3d; print(f'  Open3D:     {open3d.__version__}')" 2>/dev/null || true
python -c "import transformers; print(f'  Transformers: {transformers.__version__}')" 2>/dev/null || true

echo ""
echo "============================================"
echo "Installation complete!"
echo ""
echo "Activate environment:"
echo "  conda activate isaacnav"
echo ""
echo "Quick start:"
echo "  python scripts/run_pipeline.py --config configs/pipeline.yaml \\"
echo "      --image data/raw_images/original.jpg --classes Elevator"
echo ""
echo "Batch processing:"
echo "  python scripts/run_batch.py create-manifest \\"
echo "      --image-dir data/raw_images/ --classes Elevator --manifest jobs.json"
echo "  python scripts/run_batch.py run --manifest jobs.json"
echo "============================================"
