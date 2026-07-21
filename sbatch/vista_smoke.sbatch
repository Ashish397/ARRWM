#!/bin/bash
#SBATCH --job-name=vista-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
set -e
ARR=/scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache TMPDIR=/tmp

# 1) vista.safetensors (~10GB) + laion open_clip ViT-H (auto-cached on first use)
cd $ARR/third_party/Vista
mkdir -p ckpts
if [ ! -f ckpts/vista.safetensors ]; then
  hf download OpenDriveLab/Vista vista.safetensors --local-dir ckpts/
fi
ls -la ckpts/

# 2) seed clips (no-op if already made)
cd $ARR
PYTHONPATH=$ARR SW_WINDOWS="8:1" python utils/decode_seed_windows.py

# 3) smoke: 2 windows x {F,R}, 4 rounds = 91 frames @10fps, 50 steps, timed
cd $ARR/third_party/Vista
export VS_WINDOWS="8:1" VS_DIRS="F:R" VS_ROUNDS=4 VS_STEPS=50 VS_SPEED=2.0 VS_ANGLE=260
export VS_OUT=$ARR/logs/eval_final/baseline_smoke/vista
PYTHONPATH= python vista_runner.py
echo "VISTA-SMOKE done $(date)"
