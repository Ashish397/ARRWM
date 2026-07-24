set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:/scratch/u6ex/as1748.u6ex/ARRWM/action-forcing TMPDIR=/tmp
SMOKE_ROOT=/scratch/u6ex/as1748.u6ex/ARRWM/.lmdb14e_smoke
rm -rf $SMOKE_ROOT
for V in gt flip2 dir4 dir8; do
  echo "=== smoke variant $V ==="
  GL_VARIANT=$V GL_OUT=$SMOKE_ROOT/$V GL_NUM=2 GL_SHARD=0 GL_NSHARDS=1 python gen_lmdb_14e.py
done
echo "=== load-side check ==="
SMOKE_ROOT=$SMOKE_ROOT python utils/smoke_lmdb14e_check.py
echo "SMOKE-INTERACTIVE done"
