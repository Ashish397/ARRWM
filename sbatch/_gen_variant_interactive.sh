set -e -o pipefail
V=${GEN_V:?set GEN_V}
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:/scratch/u6ex/as1748.u6ex/ARRWM/action-forcing TMPDIR=/tmp
declare -A NCTX=( [gt]=1280 [flip2]=640 [dir4]=320 [dir8]=160 )
N=${NCTX[$V]}; NSH=${NGPU:-3}; PER=$(( (N + NSH - 1) / NSH ))
OUTD=/projects/u6ex/fbots/frodobots_lmdb/v14e_pilot_${V}
mkdir -p $OUTD
PIDS=()
for G in $(seq 0 $(( NSH - 1 ))); do
  CUDA_VISIBLE_DEVICES=$G GL_VARIANT=$V GL_OUT=$OUTD GL_NUM=$PER GL_SHARD=$G GL_NSHARDS=$NSH \
    python gen_lmdb_14e.py > logs/lmdb14e_int_${V}_g${G}.log 2>&1 &
  PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait $p; done
echo "GEN-INT $V done: $(ls $OUTD | grep -c '\.pt$') files"
