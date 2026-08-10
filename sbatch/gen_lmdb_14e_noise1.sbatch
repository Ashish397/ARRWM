#!/bin/bash
#SBATCH --job-name=lmdb14e-ns1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=05:30:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# ONE pilot variant (env V), 4-GPU sharded; short walltime for backfill.
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH TMPDIR=/tmp
declare -A NCTX=( [gt]=1280 [flip2]=640 [dir4]=320 [dir8]=160 )
N=${NCTX[$V]}; PER=$(( N / 4 ))
OUTD=/projects/u6ex/fbots/frodobots_lmdb/v14e_pilot_${V}
mkdir -p $OUTD
echo "=== variant $V: $N contexts ==="
PIDS=()
for G in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$G GL_VARIANT=$V GL_OUT=$OUTD \
    GL_NUM=$PER GL_SHARD=$G GL_NSHARDS=4 \
    GL_NOISE_VARIANT=1 python gen_lmdb_14e.py > logs/lmdb14e_pilot_${V}_g${G}.log 2>&1 &
  PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait $p; done
echo "=== variant $V done: $(ls $OUTD | grep -c '\.pt$') files ==="
