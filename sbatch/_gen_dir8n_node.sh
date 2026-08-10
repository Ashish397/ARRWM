#!/bin/bash
# Per-node worker for the dir8n generation: 4 GPU shards per node.
# SLURM_PROCID = node index (0..7); shard = node*4 + gpu, 32 shards total.
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache
export PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH TMPDIR=/tmp
NODE=${SLURM_PROCID:-0}
OUTD=/projects/u6ex/fbots/frodobots_lmdb/v14e_pilot_dir8n
mkdir -p $OUTD
PIDS=()
for G in 0 1 2 3; do
  S=$((NODE * 4 + G))
  CUDA_VISIBLE_DEVICES=$G GL_VARIANT=dir8n GL_OUT=$OUTD GL_NOISE_VARIANT=${GL_NV:-0} \
    GL_NUM=40 GL_SHARD=$S GL_NSHARDS=32 \
    python gen_lmdb_14e.py > logs/lmdb14e_dir8n_s${S}.log 2>&1 &
  PIDS+=($!)
done
RC=0
for p in "${PIDS[@]}"; do wait $p || RC=1; done
echo "node $NODE done rc=$RC files_now=$(ls $OUTD | grep -c '\.pt$')"
exit $RC
