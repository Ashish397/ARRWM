set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache
export PYTHONPATH=$PWD:$PWD/action-forcing TMPDIR=/tmp
export AF_SNAPSHOT_STEPS="0,15,18,19,-1" AF_EVAL_STEPS=20
export CUDA_VISIBLE_DEVICES=0
STEPS=${PILOT_STEPS:-250}
for V in gt flip2 dir4 dir8; do
  VDIR=$V; [ "$V" = "gt" ] && VDIR=gt_cap
  LAMCF=1.0; [ "$V" = "gt" ] && LAMCF=0.0
  echo "=== train pilot $V (lambda_cf=$LAMCF, $STEPS steps) ==="
  python action-forcing/train.py \
    --config configs/action_ode_distill_F.yaml \
    chunked_lmdb=true auto_resume=false \
    clean_root=/projects/u6ex/fbots/frodobots_lmdb/v14e_pilot_${VDIR} \
    cf_root=/projects/u6ex/fbots/frodobots_lmdb/v14e_pilot_${VDIR} \
    clean_only=false require_cf=false lambda_cf=${LAMCF} \
    random_steps="[0,15,18,19]" \
    generator_ckpt=/scratch/u6ex/as1748.u6ex/ARRWM/logs/v14e_pca8_raw/causal_lora_step0005000.pt \
    max_steps=${STEPS} checkpoint_interval=${STEPS} \
    logdir=/scratch/u6ex/as1748.u6ex/ARRWM/logs/ode14e_pilot/run_${VDIR} \
    config_name=pilot_${V} > logs/ode14e_pilot_${V}.log 2>&1
  echo "=== $V done: $(ls logs/ode14e_pilot/pilot_${V}/*.pt 2>/dev/null | tail -1) ==="
done
echo "TRAIN-PILOTS-SEQ done"
