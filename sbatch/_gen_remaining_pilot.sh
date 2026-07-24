set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:/scratch/u6ex/as1748.u6ex/ARRWM/action-forcing TMPDIR=/tmp
# reduced matched pilot: 160 windows/variant. contexts = 160/n_actions.
declare -A NCTX=( [flip2]=80 [dir4]=40 [dir8]=20 )
for V in flip2 dir4 dir8; do
  N=${NCTX[$V]}; PER=$(( (N + 1) / 2 ))
  OUTD=/projects/u6ex/fbots/frodobots_lmdb/v14e_pilot_${V}
  mkdir -p $OUTD
  echo "=== $V: $N contexts (2 shards x $PER) ==="
  PIDS=()
  for G in 0 1; do
    CUDA_VISIBLE_DEVICES=$G GL_VARIANT=$V GL_OUT=$OUTD GL_NUM=$PER GL_SHARD=$G GL_NSHARDS=2 \
      python gen_lmdb_14e.py > logs/lmdb14e_red_${V}_g${G}.log 2>&1 &
    PIDS+=($!)
  done
  for p in "${PIDS[@]}"; do wait $p; done
  echo "=== $V done: $(ls $OUTD | grep -c '\.pt$') files ==="
done
echo "GEN-REMAINING done"
