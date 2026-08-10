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
  nrcg)  EXTRA="ode_nextrung_targets=true generator_action_z_guidance_weight=0.75";;
  nrmom) EXTRA="ode_nextrung_targets=true ode_moment_loss_weight=0.5";;
  nrmom2) EXTRA="ode_nextrung_targets=true ode_moment_loss_weight=0.5 ode_moment_ref=committed";;
  nrfull) EXTRA="ode_nextrung_targets=true ode_moment_loss_weight=0.5 ode_moment_ref=committed ode_sep_loss_weight=0.5";;
  nrmom2h) EXTRA="ode_nextrung_targets=true ode_moment_loss_weight=2.0 ode_moment_ref=committed";;
  fan4)  EXTRA="ode_nextrung_targets=true ode_moment_loss_weight=0.5 ode_moment_ref=committed ode_sep_loss_weight=0.5"; export ODE_RANDOM_CF=1; DATAV=dir4;;
  v1mom)  EXTRA="ode_moment_loss_weight=0.5 ode_moment_ref=committed";;
  mn)     EXTRA="ode_nextrung_targets=true ode_moment_loss_weight=0.5 ode_moment_ref=committed";;  # nrmom2 recipe on MULTI-NOISE flip2 (ns1 chains present)
  nrad)   EXTRA="ode_nextrung_targets=true ode_moment_loss_weight=0.5 ode_moment_ref=committed ode_actdelta_loss_weight=0.5";;  # + action-effect vector matching
  rep)    EXTRA="ode_noiserep_loss_weight=0.5";;                       # standard MSE + gaussian-signature repulsor
  repkl)  EXTRA="ode_loss_type=kl ode_noiserep_loss_weight=0.5";;      # KL base + gaussian-signature repulsor
  acfg)   EXTRA="ode_nextrung_targets=true ode_moment_loss_weight=0.5 ode_moment_ref=committed ode_action_cfg_dropout=0.15";;  # action-CFG training (nrmom2 base)
  kl)     EXTRA="ode_loss_type=kl";;                                   # pure KL on standard flip2 pair training (isolate KL from the 8-dir batch)
  klcg)   EXTRA="ode_loss_type=kl generator_action_z_guidance_weight=0.75";;  # KL + action-critic guidance (critic re-injects the action gradient KL lacks)
  alldir8)   EXTRA="alldir_batches=true"; export ODE_ALLDIR=1 ODE_ALLDIR_PAD8=1; DATAV=dir8;;
  alldir8kl) EXTRA="alldir_batches=true ode_loss_type=kl"; export ODE_ALLDIR=1 ODE_ALLDIR_PAD8=1; DATAV=dir8;;
  alldir8n)   EXTRA="alldir_batches=true"; export ODE_ALLDIR=1; DATAV=dir8n;;                    # big dir8+noop LMDB; cN = clean side, groups of 8
  alldir8nkl) EXTRA="alldir_batches=true ode_loss_type=kl"; export ODE_ALLDIR=1; DATAV=dir8n;;
  c10mse)    EXTRA="alldir_batches=true ode_curriculum=true ode_curriculum_epochs=10"; export ODE_ALLDIR=1; DATAV=dir8n;;
  c10kl)     EXTRA="alldir_batches=true ode_curriculum=true ode_curriculum_epochs=10 ode_loss_type=kl"; export ODE_ALLDIR=1; DATAV=dir8n;;
  c10mserep) EXTRA="alldir_batches=true ode_curriculum=true ode_curriculum_epochs=10 ode_gexcl_weight=0.5 ode_gexcl_band=0.05"; export ODE_ALLDIR=1; DATAV=dir8n;;
  c10klrep)  EXTRA="alldir_batches=true ode_curriculum=true ode_curriculum_epochs=10 ode_loss_type=kl ode_gexcl_weight=0.5 ode_gexcl_band=0.05"; export ODE_ALLDIR=1; DATAV=dir8n;;
  c10mserep2) EXTRA="alldir_batches=true ode_curriculum=true ode_curriculum_epochs=10 ode_gexcl_weight=0.5 ode_gexcl_band=0.05"; export ODE_ALLDIR=1; DATAV=dir8n;;   # v2: guaranteed BACKWARD slot + absolute bad-threshold
  a8ngx)   EXTRA="alldir_batches=true ode_gexcl_weight=0.5 ode_gexcl_band=0.05"; export ODE_ALLDIR=1; DATAV=dir8n;;                        # AXIS2: bounded gaussian exclusion (commit clock)
  a8ned)   EXTRA="alldir_batches=true ode_edist_weight=0.5"; export ODE_ALLDIR=1; DATAV=dir8n;;                                          # AXIS3: energy-distance distribution matching over the fan
  a8nedgx) EXTRA="alldir_batches=true ode_edist_weight=0.5 ode_gexcl_weight=0.5 ode_gexcl_band=0.05"; export ODE_ALLDIR=1; DATAV=dir8n;;    # AXIS2+3
  a8nedheun) EXTRA="alldir_batches=true ode_edist_weight=0.5 ode_nextrung_targets=true ode_nextrung_solver=heun"; export ODE_ALLDIR=1; DATAV=dir8n;;  # AXIS1+3
  nrsolveul)  EXTRA="alldir_batches=true ode_nextrung_targets=true ode_nextrung_solver=euler";    export ODE_ALLDIR=1; DATAV=dir8n;;  # solver control (1st-order chord)
  nrsolvmid)  EXTRA="alldir_batches=true ode_nextrung_targets=true ode_nextrung_solver=midpoint"; export ODE_ALLDIR=1; DATAV=dir8n;;  # 2nd-order central-secant targets
  nrsolvheun) EXTRA="alldir_batches=true ode_nextrung_targets=true ode_nextrung_solver=heun";     export ODE_ALLDIR=1; DATAV=dir8n;;  # 2nd-order trapezoid targets
  kl8n)    EXTRA="ode_loss_type=kl"; DATAV=dir8n;;                                               # normal pair training, KL loss, dir8n data
  klcg8n)  EXTRA="ode_loss_type=kl generator_action_z_guidance_weight=0.75"; DATAV=dir8n;;       # normal KL + action-critic guidance
  msecg8n) EXTRA="generator_action_z_guidance_weight=0.75"; DATAV=dir8n;;                        # normal MSE + action-critic guidance
  emd1)   EXTRA="ode_emdrep_loss_weight=0.5"; DATAV=dir8n;;                                      # standard MSE + 1/d^2 EMD repulsor (d = W2 to N(0,1))
  emd2)   EXTRA="ode_emdrep_delta_weight=0.5"; DATAV=dir8n;;                                     # standard MSE + 1/(d_cur - d_prevchunk) chunk-contraction repulsor
  emdc)   EXTRA="ode_emdrep_loss_weight=0.5 ode_emdrep_delta_weight=0.5 ode_emdrep_commit_only=true"; DATAV=dir8n;;  # commit-clock EMD: terms only at the final rung (solution 2)
  emdz)   EXTRA="ode_emdhead_weight=0.5 ode_emdhead_delta_weight=0.5"; DATAV=dir8n; EMDZ=1;;     # zero-init learned transport head (AdaLN-zero analog), flow map protected
  emdzdb) EXTRA="ode_emdhead_weight=1.0 ode_emdhead_objective=deadband ode_emdhead_band_hi=0.05 ode_emdhead_id_weight=0.1"; DATAV=dir8n; EMDZ=1;;  # zero-init head, VICReg-style SATISFIABLE deadband (no-decay target, beats teacher) + identity-minimality
  v2)    EXTRA="ode_nextrung_targets=true ode_teachersup_enabled=true ode_teachersup_weight=0.15 ode_teachersup_steps=5 ode_teachersup_rungs=[625.0]";;
  *) echo "bad ARM=$ARM"; exit 1;;
esac

TS=${TOTAL_STEPS:-250}
CKPT_NAME=action_ode_step$(printf %07d $TS).pt
LOGDIR=logs/ode14e_pilot/run3_flip2_${ARM}
if [ ! -f "$LOGDIR/.train_done" ]; then
  ${LAUNCHER:-python} action-forcing/train.py \
    --config configs/action_ode_distill_F.yaml \
    chunked_lmdb=true ode_chunked_supervision=true \
    cd_teacher_loss_enabled=false cd_student_loss_enabled=false \
    $EXTRA \
    clean_root=/projects/u6ex/fbots/frodobots_lmdb/v14e_pilot_${DATAV:-flip2} \
    cf_root=/projects/u6ex/fbots/frodobots_lmdb/v14e_pilot_${DATAV:-flip2} \
    clean_only=false require_cf=false lambda_cf=1.0 \
    random_steps="[0,15,18,19]" eval_inference_steps=20 \
    generator_ckpt=/scratch/u6ex/as1748.u6ex/ARRWM/logs/v14e_pca8_raw/causal_lora_step0005000.pt \
    total_steps=$TS save_interval=${SAVE_EVERY:-$TS} eval_interval=$TS ckpt_skip_optimizer=true ckpt_local_stage=true \
    logdir=$LOGDIR config_name=pilot_flip2_${ARM} > logs/ode14e_pilot_flip2_${ARM}.log 2>&1 \
    && touch "$LOGDIR/.train_done"
fi

CKPT=$LOGDIR/$CKPT_NAME
[ -f "$CKPT" ] || CKPT=$(ls -t $LOGDIR/action_ode_step*.pt 2>/dev/null | head -1)
if [ -n "$EMDZ" ]; then
  # emdz probes must serve WITH the learned transport head applied.
  export ODE_EMDHEAD=1 ODE_EMDHEAD_CKPT=$PWD/$CKPT
fi
if [ -f "$CKPT" ]; then
  FR_RUN=pilot3_flip2${ARM} FR_CONFIG=configs/ar_eval_dmd_student.yaml FR_CKPT=$CKPT \
    FR_RUNGS="1000,625,357.142857,208.333333" FR_VIDEO=1 FR_CHUNKS=6 FR_NSEEDS=2 \
    python utils/flow_record_ode_student.py || echo "PROBE-FAIL $ARM"
  FDD_NSEEDS=2 FDD_PAIRS="pilot3_flip2${ARM}=14e8" FDD_FIGNAME=flow_pilot3_flip2${ARM}_scorecard \
    python utils/flow_diverge_dmd3.py || echo "SCORE-FAIL $ARM"
  MC_RUNS=pilot3_flip2${ARM} python utils/motion_check.py || true
  SD_RUNS=pilot2_flip2:pilot3_flip2${ARM} python utils/stat_drift.py || true
  # TARGET METRIC: match vs the SEED-MATCHED dense teacher (>= 0.95 goal).
  # Falls back to the differently-seeded teacher videos (ceiling ~0.85)
  # if the matched reference has not been recorded yet.
  if [ -d analysis/eval_final/flow_viz/.motion_check/teacher20_matched ]; then
    TM_RUNS=pilot3_flip2${ARM}:pilot4_alldir8n2 TM_REF_RUN=teacher20_matched \
      TM_OUT=analysis/eval_final/flow_viz/teacher_match_${ARM}.csv \
      python utils/teacher_match.py || true
  else
    TM_RUNS=pilot3_flip2${ARM}:pilot4_alldir8n2 \
      TM_OUT=analysis/eval_final/flow_viz/teacher_match_${ARM}.csv \
      python utils/teacher_match.py || true
  fi
else
  echo "TRAIN-FAIL $ARM (no checkpoint)"; tail -30 logs/ode14e_pilot_flip2_${ARM}.log
fi
echo "ARM-${ARM} done $(date)"
