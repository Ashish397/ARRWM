#!/bin/bash
#SBATCH --job-name=mg2base-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
set -e
ARR=/scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache TMPDIR=/tmp

cd $ARR
PYTHONPATH=$ARR SW_WINDOWS="8:1" python utils/decode_seed_windows.py

# FULL (non-distilled) foundation model, dense 48-step warped schedule
cd $ARR/third_party/Matrix-Game/Matrix-Game-2
ls -la Matrix-Game-2.0/base_model/
export MG_WINDOWS="8:1" MG_DIRS="F:R" MG_NUMLAT=150 MG_KDIM=6
export MG_CONFIG=configs/inference_yaml/inference_universal_base48.yaml
export MG_CKPT=Matrix-Game-2.0/base_model/diffusion_pytorch_model.safetensors
export MG_OUT=$ARR/logs/eval_final/baseline_smoke/matrixgame_base
PYTHONPATH= python matrixgame_runner.py
echo "MG2BASE-SMOKE done $(date)"
