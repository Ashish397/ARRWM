#!/bin/bash
#SBATCH --job-name=flip2-ts
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# A/B arm B: flip2 pilot + ON-POLICY TEACHER SUPERVISION (user-directed
# 2026-07-27). Arm A = run2_flip2 (v1, already trained). Same data, steps,
# init; only ode_teachersup_* added. Probe immediately after (user standing
# directive), same pinned-rung fixed-RoPE protocol as pilots-v2.
set -x
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=$PWD:$PWD/action-forcing TMPDIR=/tmp
export AF_SNAPSHOT_STEPS="0,15,18,19,-1" AF_EVAL_STEPS=20

LOGDIR=logs/ode14e_pilot/run3_flip2_ts
if [ ! -f $LOGDIR/action_ode_step0000250.pt ]; then
  python action-forcing/train.py \
    --config configs/action_ode_distill_F.yaml \
    chunked_lmdb=true ode_chunked_supervision=true \
    cd_teacher_loss_enabled=false cd_student_loss_enabled=false \
    ode_teachersup_enabled=true ode_teachersup_weight=0.5 \
    clean_root=/projects/u6ex/fbots/frodobots_lmdb/v14e_pilot_flip2 \
    cf_root=/projects/u6ex/fbots/frodobots_lmdb/v14e_pilot_flip2 \
    clean_only=false require_cf=false lambda_cf=1.0 \
    random_steps="[0,15,18,19]" eval_inference_steps=20 \
    generator_ckpt=/scratch/u6ex/as1748.u6ex/ARRWM/logs/v14e_pca8_raw/causal_lora_step0005000.pt \
    total_steps=250 save_interval=50 eval_interval=250 \
    logdir=$LOGDIR config_name=pilot_flip2_ts > logs/ode14e_pilot_flip2_ts.log 2>&1
fi

CKPT=$LOGDIR/action_ode_step0000250.pt
if [ -f "$CKPT" ]; then
  FR_RUN=pilot3_flip2ts FR_CONFIG=configs/ar_eval_dmd_student.yaml FR_CKPT=$CKPT \
    FR_RUNGS="1000,625,357.142857,208.333333" FR_VIDEO=1 FR_CHUNKS=6 FR_NSEEDS=2 \
    python utils/flow_record_ode_student.py || echo "PROBE-FAIL flip2ts"
  FDD_NSEEDS=2 FDD_PAIRS="pilot3_flip2ts=14e8" FDD_FIGNAME=flow_pilot3_flip2ts_scorecard \
    python utils/flow_diverge_dmd3.py || echo "SCORE-FAIL flip2ts"
  MC_RUNS=pilot3_flip2ts python utils/motion_check.py || true
  SD_RUNS=pilot2_flip2:pilot3_flip2ts python utils/stat_drift.py || true
else
  echo "TRAIN-FAIL flip2ts (no checkpoint)"; tail -30 logs/ode14e_pilot_flip2_ts.log
fi
echo "FLIP2-TS done $(date)"
