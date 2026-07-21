#!/bin/bash
#SBATCH --job-name=wcam-smoke
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

# 1) WorldCam finetuned DiT (~3GB)
cd $ARR/third_party/WorldCam
if [ ! -f weights/finetuned_dit.safetensors ]; then
  hf download worldcam/worldcam --local-dir weights/ --include "*.safetensors" --repo-type model
fi
ls -la weights/

# 2) real 65-frame seed clips for windows 8 and 1 (GPU decode)
cd $ARR
PYTHONPATH=$ARR SW_WINDOWS="8:1" python utils/decode_seed_windows.py

# 3) WorldCam smoke: 2 windows x {F,R}, 50 AR steps (200 gen frames), timed
cd $ARR/third_party/WorldCam
export WC_WINDOWS="8:1" WC_DIRS="F:R" WC_STEPT=0.02 WC_STEPR=1.5 WC_ARSTEPS=50
export WC_OUT=$ARR/logs/eval_final/baseline_smoke/worldcam
PYTHONPATH= python worldcam_runner.py
echo "WCAM-SMOKE done $(date)"
