#!/bin/bash
# Run this on a GPU node:
#   srun --gpus=1 --time=01:30:00 --pty /bin/bash --login
# Then:
#   bash /scratch/u6ej/as1748.u6ej/ARRWM/install_flash_attn.sh
#
# Environment: torch 2.10.0+cu128, Python 3.10, GH200 (sm_90a)
# Note: no cuda/12.7 or 12.8 module exists on this system; cuda/12.6 nvcc is used
# for compilation. The minor version mismatch vs torch's cu128 is harmless — torch
# bundles its own CUDA 12.8 runtime libs inside the wheel.

CONDA_PREFIX="/scratch/u6ej/as1748.u6ej/miniforge3"
ENV_NAME="arrwm"

echo "[$(date '+%H:%M:%S')] Loading modules (GCC 14 + CUDA 12.6 for nvcc)..."
module load PrgEnv-gnu   # provides gcc-native/14.2
module load cuda/12.6    # closest available nvcc; torch bundles its own cu128 runtime
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
echo "[$(date '+%H:%M:%S')] CUDA_HOME=$CUDA_HOME"
echo "[$(date '+%H:%M:%S')] nvcc:  $(nvcc --version | grep release)"
echo "[$(date '+%H:%M:%S')] g++:   $(g++ --version | head -1)"

# Explicitly point CXX/CC to GCC 14 so conda's compiler_compat shim doesn't
# redirect to the system GCC 7.5 (which PyTorch 2.10 headers reject)
export CC=$(which gcc)
export CXX=$(which g++)
echo "[$(date '+%H:%M:%S')] CC=$CC  CXX=$CXX"

# nvcc 12.6 rejects GCC > 13 unless told to allow it.
# Wrap nvcc to always pass -allow-unsupported-compiler.
NVCC_REAL=$(which nvcc)
NVCC_WRAP="/tmp/nvcc_wrap"
mkdir -p "$NVCC_WRAP"
cat > "$NVCC_WRAP/nvcc" << NVCC_EOF
#!/bin/bash
exec "$NVCC_REAL" -allow-unsupported-compiler "\$@"
NVCC_EOF
chmod +x "$NVCC_WRAP/nvcc"
export PATH="$NVCC_WRAP:$PATH"
echo "[$(date '+%H:%M:%S')] nvcc wrapper: $(which nvcc)"

echo "[$(date '+%H:%M:%S')] Activating conda environment..."
source "$CONDA_PREFIX/bin/activate"
conda activate "$ENV_NAME"

echo "[$(date '+%H:%M:%S')] Installing ninja for fast parallel compilation..."
pip install ninja --quiet
echo "[$(date '+%H:%M:%S')] ninja: $(ninja --version 2>/dev/null || echo 'not found')"

echo "[$(date '+%H:%M:%S')] Python:  $(which python)"
echo "[$(date '+%H:%M:%S')] PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "[$(date '+%H:%M:%S')] CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "[$(date '+%H:%M:%S')] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

echo ""
echo "[$(date '+%H:%M:%S')] Installing flash-attn (this takes 20-40 minutes to compile)..."

# --no-build-isolation: use the installed torch 2.10.0+cu128 (not a fresh download)
# MAX_JOBS=4: parallelise nvcc compilation across 4 cores
MAX_JOBS=4 pip install flash-attn --no-build-isolation

echo ""
echo "[$(date '+%H:%M:%S')] Done!"
python -c "import flash_attn; print(f'flash-attn version: {flash_attn.__version__}')"
