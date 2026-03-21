#!/bin/bash
# Don't use set -e — we want to handle errors package by package

SCRATCH="/scratch/u6ej/as1748.u6ej"
CONDA_PREFIX="$SCRATCH/miniforge3"
ENV_NAME="arrwm"
REPO_DIR="$SCRATCH/ARRWM"
LOG="$SCRATCH/setup_conda_env.log"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== Step 1: Download Miniforge3 for aarch64 ==="
cd "$SCRATCH"
if [ ! -f Miniforge3-Linux-aarch64.sh ]; then
    wget -q --show-progress \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh \
        -O Miniforge3-Linux-aarch64.sh
    log "Downloaded Miniforge installer."
else
    log "Installer already present, skipping download."
fi

log "=== Step 2: Install Miniforge3 to $CONDA_PREFIX ==="
if [ ! -d "$CONDA_PREFIX" ]; then
    bash Miniforge3-Linux-aarch64.sh -b -p "$CONDA_PREFIX"
    log "Miniforge installed."
else
    log "Miniforge already installed at $CONDA_PREFIX, skipping."
fi

# Activate conda
source "$CONDA_PREFIX/bin/activate"
log "Conda version: $(conda --version)"

log "=== Step 3: Create conda environment '$ENV_NAME' with Python 3.10 ==="
if conda env list | grep -q "^$ENV_NAME "; then
    log "Environment '$ENV_NAME' already exists, skipping creation."
else
    conda create -y -n "$ENV_NAME" python=3.10
    log "Environment created."
fi

conda activate "$ENV_NAME"
log "Active env: $CONDA_DEFAULT_ENV"

log "=== Step 4: Install PyTorch with CUDA 12.4 support ==="
python -c "import torch" 2>/dev/null && log "PyTorch already installed, skipping." || {
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124
    log "PyTorch installed."
}
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

log "=== Step 5: Install CLIP from GitHub ==="
pip install git+https://github.com/openai/CLIP.git || log "WARN: CLIP install failed"

log "=== Step 6: Install pip requirements one by one ==="

# Packages that need nvidia-pyindex first
NVIDIA_PKGS=(nvidia-tensorrt)

# Packages to skip (unavailable on aarch64 or problematic)
SKIP_PKGS=(pycuda onnxruntime-gpu)

install_pkg() {
    local pkg="$1"
    # Skip commented lines and empty lines
    [[ "$pkg" =~ ^#.*$ ]] && return
    [[ -z "$pkg" ]] && return
    # Skip git+ (handled separately)
    [[ "$pkg" =~ ^git\+ ]] && return
    # Skip nvidia-pyindex (install it first)
    [[ "$pkg" == "nvidia-pyindex" ]] && return

    log "  Installing: $pkg"
    pip install "$pkg" --quiet 2>&1 | tail -3
    if [ $? -ne 0 ]; then
        log "  WARN: Failed to install '$pkg' — skipping"
    else
        log "  OK: $pkg"
    fi
}

# Install nvidia-pyindex first so nvidia-tensorrt can find the right index
log "  Installing nvidia-pyindex..."
pip install nvidia-pyindex --quiet && log "  OK: nvidia-pyindex" || log "  WARN: nvidia-pyindex failed"

# Now install nvidia-tensorrt via their index
log "  Installing nvidia-tensorrt via NVIDIA index..."
pip install nvidia-tensorrt --quiet 2>&1 | tail -3 \
    && log "  OK: nvidia-tensorrt" \
    || log "  WARN: nvidia-tensorrt not available on this platform (aarch64 pip wheels may be absent)"

# Install all other requirements
while IFS= read -r line; do
    # Trim whitespace
    pkg="${line//[$'\t\r\n']}"
    pkg="${pkg## }"
    pkg="${pkg%% }"

    # Skip blank, comments, git lines, and nvidia packages (handled above)
    [[ -z "$pkg" || "$pkg" =~ ^# || "$pkg" =~ ^git\+ ]] && continue
    [[ "$pkg" == "nvidia-pyindex" || "$pkg" == "nvidia-tensorrt" ]] && continue

    install_pkg "$pkg"
done < "$REPO_DIR/requirements.txt"

log "=== Setup Complete ==="
echo ""
echo "To activate the environment, run:"
echo "  source $CONDA_PREFIX/bin/activate && conda activate $ENV_NAME"
echo ""
python --version
pip list | grep -E "torch|diffusers|transformers|accelerate|peft|open.clip|clip"
