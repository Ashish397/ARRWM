#!/bin/bash
#SBATCH --job-name=turnrev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export TMPDIR=/tmp TORCHINDUCTOR_CACHE_DIR=/tmp/ti_$USER TRITON_CACHE_DIR=/tmp/triton_$USER
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH
export MASTER_ADDR=127.0.0.1 MASTER_PORT=$((20000+RANDOM%10000)) WORLD_SIZE=1 RANK=0 LOCAL_RANK=0
echo "TURNREV start $(date) | CONFIG=$TR_CONFIG CKPT=$TR_CKPT CHUNKS=$TR_CHUNKS PATTERN=$TR_PATTERN"
python utils/turn_reverse_eval.py
echo "TURNREV done $(date)"
