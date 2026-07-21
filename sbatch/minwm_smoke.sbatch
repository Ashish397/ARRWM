#!/bin/bash
#SBATCH --job-name=minwm-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM/third_party/minWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache TMPDIR=/tmp
# DMD checkpoint
if [ ! -f ckpts/Wan21/Action2V/dmd/model.pt ]; then
  hf download MIN-Lab/minWM --local-dir ./ckpts --include "Wan21/Action2V/dmd/*" 2>&1 | tail -2 \
    || huggingface-cli download MIN-Lab/minWM --local-dir ./ckpts --include "Wan21/Action2V/dmd/*" 2>&1 | tail -2
fi
ls -la ckpts/Wan21/Action2V/dmd/ | head

python minwm_runner.py
echo "MINWM-SMOKE done $(date)"
