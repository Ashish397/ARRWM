#!/bin/bash
#SBATCH --job-name=mg2-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
set -e
ARR=/scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache TMPDIR=/tmp

# 1) checkpoints: universal distilled DiT + VAE + CLIP (~12GB)
cd $ARR/third_party/Matrix-Game/Matrix-Game-2
if [ ! -f Matrix-Game-2.0/base_distilled_model/base_distill.safetensors ]; then
  hf download Skywork/Matrix-Game-2.0 --local-dir Matrix-Game-2.0 \
    --include "base_distilled_model/*" "Wan2.1_VAE.pth" "models_clip*" "xlm-roberta-large/*" "*.json"
fi
ls -la Matrix-Game-2.0/ Matrix-Game-2.0/base_distilled_model/ 2>/dev/null

# 2) seed clips (no-op if wcam-smoke already made them)
cd $ARR
PYTHONPATH=$ARR SW_WINDOWS="8:1" python utils/decode_seed_windows.py

# 3) smoke: 2 windows x {F,R}, 150 latents = 597 pixel frames, timed
cd $ARR/third_party/Matrix-Game/Matrix-Game-2
export MG_WINDOWS="8:1" MG_DIRS="F:R" MG_NUMLAT=150
export MG_OUT=$ARR/logs/eval_final/baseline_smoke/matrixgame
PYTHONPATH= python matrixgame_runner.py
echo "MG2-SMOKE done $(date)"
