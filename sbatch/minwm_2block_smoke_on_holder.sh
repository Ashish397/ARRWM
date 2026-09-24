#!/bin/bash
set -euo pipefail
: "${ALLOC:?set ALLOC to the minWM holder job id}"
ROOT=/scratch/u6ex/as1748.u6ex/ARRWM
OUT="$ROOT/logs/eval_final/minwm_2block_smoke"
mkdir -p "$OUT" "$ROOT/logs"
srun --jobid="$ALLOC" --overlap --nodes=1 --ntasks=1 --gpus-per-node=1 \
  --cpus-per-task=12 --gpu-bind=none bash -lc '
set -euo pipefail
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
cd /scratch/u6ex/as1748.u6ex/ARRWM/third_party/minWM
export AF_ROOT=/scratch/u6ex/as1748.u6ex/ARRWM
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache
export TMPDIR=/tmp PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM
export CUDA_VISIBLE_DEVICES=0 MW_CPU_T5=1 MW_SEED_LAT=8 MW_NUMLAT=12
export MW_WINDOWS=u31 MW_DIRS=F MW_TAG=_seed8_smoke MW_SAVE_LAT=1
export MW_SEED_FMT="$AF_ROOT/analysis/eval_final/seed65_e1/seed65_{wi}.mp4"
export MW_OUT=/scratch/u6ex/as1748.u6ex/ARRWM/logs/eval_final/minwm_2block_smoke
python minwm_runner.py
'
