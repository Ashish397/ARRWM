set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:/scratch/u6ex/as1748.u6ex/ARRWM/action-forcing TMPDIR=/tmp
export CUDA_VISIBLE_DEVICES=${PHYS_GPU:?}
OUTD=/projects/u6ex/fbots/frodobots_lmdb/v14e_pilot_${GEN_V}
mkdir -p $OUTD
GL_VARIANT=$GEN_V GL_OUT=$OUTD GL_NUM=${GL_NUM:?} GL_SHARD=${GL_SHARD:?} GL_NSHARDS=${GL_NSHARDS:?} \
  python gen_lmdb_14e.py > logs/lmdb14e_oneshard_${GEN_V}_s${GL_SHARD}.log 2>&1
echo "ONESHARD $GEN_V shard $GL_SHARD done: $(ls $OUTD | grep -c '\.pt$') files"
