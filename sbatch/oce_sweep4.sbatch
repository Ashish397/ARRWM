#!/bin/bash
#SBATCH --job-name=oce-sw4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --gpus-per-node=4
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH
export TMPDIR=/tmp
# full step list for this shard, split interleaved across the 4 GPUs
ALL=$(python -c "print(','.join(str(s) for s in range($OCE_MIN,$OCE_MAX+1,$OCE_STRIDE)))")
IFS=',' read -ra ARR <<< "$ALL"
echo "shard steps: $ALL  (splitting across 4 GPUs)"
for g in 0 1 2 3; do
  sub=""; i=$g
  while [ $i -lt ${#ARR[@]} ]; do sub="${sub},${ARR[$i]}"; i=$((i+4)); done
  sub=${sub#,}
  [ -z "$sub" ] && continue
  echo "GPU $g -> steps $sub"
  CUDA_VISIBLE_DEVICES=$g WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
    TORCHINDUCTOR_CACHE_DIR=/tmp/ti_${USER}_g$g TRITON_CACHE_DIR=/tmp/tr_${USER}_g$g \
    MASTER_ADDR=127.0.0.1 MASTER_PORT=$((21000+g)) \
    OCE_STEPS="$sub" OCE_OUT="${OCE_OUT}/gpu${g}" \
    python utils/offline_control_eval.py > logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_gpu${g}.log 2>&1 &
done
wait
echo "SWEEP4 all-GPUs done $(date)"
