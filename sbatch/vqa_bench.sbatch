#!/bin/bash
#SBATCH --job-name=vqa-bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH TMPDIR=/tmp
mkdir -p third_party
if [ ! -d third_party/DOVER ]; then
  git clone --depth 1 https://github.com/VQAssessment/DOVER.git third_party/DOVER
fi
pip install -q --no-deps -e third_party/DOVER || true
# decord has no aarch64 wheel -> imageio-backed shim (DOVER's script dir is on sys.path)
cat > third_party/DOVER/decord.py <<'PYEOF'
"""Minimal decord shim for aarch64: VideoReader via imageio."""
import numpy as np, imageio

class _Batch:
    def __init__(self, arr): self._a = arr
    def asnumpy(self): return self._a

class VideoReader:
    def __init__(self, path, *a, **k):
        r = imageio.get_reader(path)
        self._frames = [np.asarray(f) for f in r]
        r.close()
    def __len__(self): return len(self._frames)
    def __getitem__(self, i): return _Batch(self._frames[i])
    def get_batch(self, ids): return _Batch(np.stack([self._frames[int(i)] for i in ids]))
    def get_avg_fps(self): return 16.0

class bridge:
    @staticmethod
    def set_bridge(*a, **k): pass
cpu = lambda *a, **k: None
gpu = lambda *a, **k: None
PYEOF
if [ ! -f third_party/DOVER/pretrained_weights/DOVER.pth ]; then
  mkdir -p third_party/DOVER/pretrained_weights
  wget -q https://github.com/VQAssessment/DOVER/releases/download/v0.1.0/DOVER.pth -O third_party/DOVER/pretrained_weights/DOVER.pth
fi
python utils/vqa_bench.py
echo "VQA-BENCH done $(date)"
