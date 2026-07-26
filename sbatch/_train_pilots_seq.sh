set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache
export PYTHONPATH=$PWD:$PWD/action-forcing TMPDIR=/tmp
export AF_SNAPSHOT_STEPS="0,15,18,19,-1" AF_EVAL_STEPS=20
export CUDA_VISIBLE_DEVICES=0
STEPS=${PILOT_STEPS:-250}
for V in gt flip2 dir4 dir8 mixed; do
  VDIR=$V; [ "$V" = "gt" ] && VDIR=gt_cap
  LAMCF=1.0; [ "$V" = "gt" ] && LAMCF=0.0
  if ls /scratch/u6ex/as1748.u6ex/ARRWM/logs/ode14e_pilot/run2_${VDIR}/action_ode_step0000250.pt >/dev/null 2>&1; then
    echo "PILOT-SKIP $V (checkpoint exists)"; continue
  fi
  echo "=== train pilot $V (lambda_cf=$LAMCF, $STEPS steps) ==="
  python action-forcing/train.py \
    --config configs/action_ode_distill_F.yaml \
    chunked_lmdb=true ode_chunked_supervision=true \
    cd_teacher_loss_enabled=false cd_student_loss_enabled=false \
    clean_root=/projects/u6ex/fbots/frodobots_lmdb/v14e_pilot_${VDIR} \
    cf_root=/projects/u6ex/fbots/frodobots_lmdb/v14e_pilot_${VDIR} \
    clean_only=false require_cf=false lambda_cf=${LAMCF} \
    random_steps="[0,15,18,19]" eval_inference_steps=20 \
    generator_ckpt=/scratch/u6ex/as1748.u6ex/ARRWM/logs/v14e_pca8_raw/causal_lora_step0005000.pt \
    total_steps=${STEPS} save_interval=50 eval_interval=${STEPS} \
    logdir=/scratch/u6ex/as1748.u6ex/ARRWM/logs/ode14e_pilot/run2_${VDIR} \
    config_name=pilot_${V} > logs/ode14e_pilot_${V}.log 2>&1
  echo "=== $V done: $(ls logs/ode14e_pilot/pilot_${V}/*.pt 2>/dev/null | tail -1) ==="
  # PROBE IMMEDIATELY (user directive): record flow + videos + scorecard now
  CKPT=$(ls logs/ode14e_pilot/run2_${VDIR}/action_ode_step0000250.pt 2>/dev/null | tail -1)
  if [ -n "$CKPT" ] && [ ! -f analysis/eval_final/flow_viz/flow_pilot2_${V}/r08_FL_s1/steps.npz ]; then
    echo "=== probe $V ==="
    FR_RUN=pilot2_${V} FR_CONFIG=configs/ar_eval_dmd_student.yaml FR_CKPT=$CKPT \
      FR_RUNGS="1000,625,357.142857,208.333333" FR_VIDEO=1 FR_CHUNKS=6 FR_NSEEDS=2 \
      python utils/flow_record_ode_student.py || echo "PROBE-FAIL $V"
    FDD_NSEEDS=2 FDD_PAIRS="pilot2_${V}=14e8" FDD_FIGNAME=flow_pilot2_${V}_scorecard \
      python utils/flow_diverge_dmd3.py || echo "SCORE-FAIL $V"
  fi
done
echo "TRAIN-PILOTS-SEQ done"
