#!/bin/bash
# Shared driver for ODE-structure arms C/D/E on the flip2 pilot data.
# Usage (inside sbatch): ARM=<mts|nr|nrmts> bash train_flip2_arms_cde.sh
#   mts   (arm C): multi-step teacher correction (teachersup steps=5 @ t=625)
#   nr    (arm D): next-rung state regression targets
#   nrmts (arm E): both combined
# Baselines: arm A = run2_flip2 (v1), arm B = run3_flip2_ts (1-step teachersup).
set -x
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=$PWD:$PWD/action-forcing TMPDIR=/tmp
export AF_SNAPSHOT_STEPS="0,15,18,19,-1" AF_EVAL_STEPS=20

case "$ARM" in
  mts)   EXTRA="ode_teachersup_enabled=true ode_teachersup_weight=0.5 ode_teachersup_steps=5 ode_teachersup_rungs=[625.0]";;
  nr)    EXTRA="ode_nextrung_targets=true";;
  nrmts) EXTRA="ode_nextrung_targets=true ode_teachersup_enabled=true ode_teachersup_weight=0.5 ode_teachersup_steps=5 ode_teachersup_rungs=[625.0]";;
  v2)    EXTRA="ode_nextrung_targets=true ode_teachersup_enabled=true ode_teachersup_weight=0.15 ode_teachersup_steps=5 ode_teachersup_rungs=[625.0]";;
  *) echo "bad ARM=$ARM"; exit 1;;
esac

LOGDIR=logs/ode14e_pilot/run3_flip2_${ARM}
if [ ! -f $LOGDIR/action_ode_step0000250.pt ]; then
  python action-forcing/train.py \
    --config configs/action_ode_distill_F.yaml \
    chunked_lmdb=true ode_chunked_supervision=true \
    cd_teacher_loss_enabled=false cd_student_loss_enabled=false \
    $EXTRA \
    clean_root=/projects/u6ex/fbots/frodobots_lmdb/v14e_pilot_flip2 \
    cf_root=/projects/u6ex/fbots/frodobots_lmdb/v14e_pilot_flip2 \
    clean_only=false require_cf=false lambda_cf=1.0 \
    random_steps="[0,15,18,19]" eval_inference_steps=20 \
    generator_ckpt=/scratch/u6ex/as1748.u6ex/ARRWM/logs/v14e_pca8_raw/causal_lora_step0005000.pt \
    total_steps=250 save_interval=250 eval_interval=250 ckpt_skip_optimizer=true ckpt_local_stage=true \
    logdir=$LOGDIR config_name=pilot_flip2_${ARM} > logs/ode14e_pilot_flip2_${ARM}.log 2>&1
fi

CKPT=$LOGDIR/action_ode_step0000250.pt
if [ -f "$CKPT" ]; then
  FR_RUN=pilot3_flip2${ARM} FR_CONFIG=configs/ar_eval_dmd_student.yaml FR_CKPT=$CKPT \
    FR_RUNGS="1000,625,357.142857,208.333333" FR_VIDEO=1 FR_CHUNKS=6 FR_NSEEDS=2 \
    python utils/flow_record_ode_student.py || echo "PROBE-FAIL $ARM"
  FDD_NSEEDS=2 FDD_PAIRS="pilot3_flip2${ARM}=14e8" FDD_FIGNAME=flow_pilot3_flip2${ARM}_scorecard \
    python utils/flow_diverge_dmd3.py || echo "SCORE-FAIL $ARM"
  MC_RUNS=pilot3_flip2${ARM} python utils/motion_check.py || true
  SD_RUNS=pilot2_flip2:pilot3_flip2${ARM} python utils/stat_drift.py || true
else
  echo "TRAIN-FAIL $ARM (no checkpoint)"; tail -30 logs/ode14e_pilot_flip2_${ARM}.log
fi
echo "ARM-${ARM} done $(date)"
