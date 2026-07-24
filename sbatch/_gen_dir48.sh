cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:/scratch/u6ex/as1748.u6ex/ARRWM/action-forcing TMPDIR=/tmp
export CUDA_VISIBLE_DEVICES=0
declare -A NCTX=( [dir4]=20 [dir8]=10 )
for V in dir4 dir8; do
  N=${NCTX[$V]}
  OUTD=/projects/u6ex/fbots/frodobots_lmdb/v14e_pilot_${V}
  mkdir -p $OUTD
  echo "=== $V: $N contexts single-GPU ==="
  GL_VARIANT=$V GL_OUT=$OUTD GL_NUM=$N GL_SHARD=0 GL_NSHARDS=1 \
    python gen_lmdb_14e.py > logs/lmdb14e_red_${V}_g0.log 2>&1 || echo "$V exited nonzero"
  echo "=== $V: $(ls $OUTD | grep -c '\.pt$') files ==="
done
echo "GEN-DIR48 done"
