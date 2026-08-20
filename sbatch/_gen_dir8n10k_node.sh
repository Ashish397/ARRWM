#!/bin/bash
# Per-node worker for the dir8n 10K-POOL generation: 4 GPU shards per node.
# SLURM_PROCID = node index (0..7); shard = node*4 + gpu, 32 shards total.
# Pool: paper_assets/v14e_train_windows_10k.json (10,003 windows; the old
# 6,000-window balanced pool is a strict per-class prefix, so the 1,237
# already-generated contexts are skipped by the per-file existence check).
# Idempotent: atomic per-chain writes, existing files skipped — safe to run
# the same job several times back-to-back until the pool is exhausted.
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache
export PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH TMPDIR=/tmp
NODE=${SLURM_PROCID:-0}
OUTD=/projects/u6ex/fbots/frodobots_lmdb/v14e_pilot_dir8n_10k
mkdir -p $OUTD
# PARALLEL-JOB SHARDING: several sbatch instances run CONCURRENTLY, each
# owning a disjoint shard range of one global partition (SHARD_BASE +
# NSHARDS_TOTAL via --export) — disjoint shards = disjoint contexts, so
# concurrent jobs never touch the same output file.
BASE=${SHARD_BASE:-0}
NST=${NSHARDS_TOTAL:-32}
NUM=${GL_NUM_PER:-313}
PIDS=()
for G in 0 1 2 3; do
  S=$((BASE + NODE * 4 + G))
  CUDA_VISIBLE_DEVICES=$G GL_VARIANT=dir8n GL_OUT=$OUTD \
    GL_POOL=/scratch/u6ex/as1748.u6ex/ARRWM/paper_assets/v14e_train_windows_10k.json \
    GL_NUM=$NUM GL_SHARD=$S GL_NSHARDS=$NST \
    python gen_lmdb_14e.py > logs/lmdb14e_dir8n10k_s${S}.log 2>&1 &
  PIDS+=($!)
done
RC=0
for p in "${PIDS[@]}"; do wait $p || RC=1; done
echo "node $NODE done rc=$RC files_now=$(ls $OUTD | grep -c '\.pt$')"
exit $RC
