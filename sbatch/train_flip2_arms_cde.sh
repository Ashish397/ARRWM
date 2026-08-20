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
  rollaw)     EXTRA="ode_rollout=true ode_rollout_commit=teacher ode_actw_enabled=true ode_actw_alpha=1.0"; export ODE_ROLLOUT=1;;             # rollout + MSE, loss upweighted by CoTracker-measured ACTION ERROR
  rollklaw)   EXTRA="ode_rollout=true ode_rollout_commit=teacher ode_loss_type=kl_local ode_actw_enabled=true ode_actw_alpha=1.0"; export ODE_ROLLOUT=1;;  # same, kl_local base
  rollawsmoke) EXTRA="ode_rollout=true ode_rollout_commit=teacher ode_actw_enabled=true ode_actw_alpha=1.0"; export ODE_ROLLOUT=1;;            # SMOKE for the action-weight mechanic
  rollaws)    EXTRA="ode_rollout=true ode_rollout_commit=teacher ode_actw_enabled=true ode_curriculum_mode=actsplit"; export ODE_ROLLOUT=1;;    # MSE + action weight + 3-bottom/2-top sampling
  rollall9)   EXTRA="ode_rollout=true ode_rollout_commit=teacher ode_actw_enabled=true ode_curriculum_mode=all9"; export ODE_ROLLOUT=1;;        # ALTERNATIVE: train on all 9 directions
  rollklawv)  EXTRA="ode_rollout=true ode_rollout_commit=teacher ode_loss_type=kl_local ode_actw_enabled=true ode_varw_enabled=true ode_curriculum_mode=actsplit"; export ODE_ROLLOUT=1;;  # KL variant: ACTION + VARIANCE weighted
  rollklawv9) EXTRA="ode_rollout=true ode_rollout_commit=teacher ode_loss_type=kl_local ode_actw_enabled=true ode_varw_enabled=true ode_curriculum_mode=all9"; export ODE_ROLLOUT=1;;      # KL variant, all 9
  rollklawvr)  EXTRA="ode_rollout=true ode_rollout_commit=teacher ode_loss_type=kl_local ode_actw_enabled=true ode_varw_enabled=true ode_gexcl_weight=0.5 ode_gexcl_band=0.05 ode_curriculum_mode=actsplit"; export ODE_ROLLOUT=1;;  # KL + action + variance + CONTRACTION BARRIER, 3-bottom/2-top
  rollklawvr9) EXTRA="ode_rollout=true ode_rollout_commit=teacher ode_loss_type=kl_local ode_actw_enabled=true ode_varw_enabled=true ode_gexcl_weight=0.5 ode_gexcl_band=0.05 ode_curriculum_mode=all9"; export ODE_ROLLOUT=1;;      # same, all 9 directions
  # --- smoke twins (same recipe, own LOGDIR) -------------------------------
  rollklsmoke)    EXTRA="ode_rollout=true ode_rollout_commit=teacher ode_loss_type=kl_local"; export ODE_ROLLOUT=1;;
  rollawssmoke)   EXTRA="ode_rollout=true ode_rollout_commit=teacher ode_actw_enabled=true ode_curriculum_mode=actsplit"; export ODE_ROLLOUT=1;;
  rollall9smoke)  EXTRA="ode_rollout=true ode_rollout_commit=teacher ode_actw_enabled=true ode_curriculum_mode=all9"; export ODE_ROLLOUT=1;;
  rollklawvsmoke) EXTRA="ode_rollout=true ode_rollout_commit=teacher ode_loss_type=kl_local ode_actw_enabled=true ode_varw_enabled=true ode_curriculum_mode=actsplit"; export ODE_ROLLOUT=1;;
  rollklawvrsmoke) EXTRA="ode_rollout=true ode_rollout_commit=teacher ode_loss_type=kl_local ode_actw_enabled=true ode_varw_enabled=true ode_gexcl_weight=0.5 ode_gexcl_band=0.05 ode_curriculum_mode=actsplit"; export ODE_ROLLOUT=1;;
  # --- VARIATIONAL RECTIFIED FLOW MATCHING (arXiv:2502.09616, adapted) ------
  # v(x_t,t) -> v(x_t,t,z); p(z|x0,xt,t,a) conditional prior, q(z|x0,x1,xt,t,a)
  # posterior, loss = base + beta*KL(q||p). Two variants: MSE base and KL base.
  rollvz)    EXTRA="ode_rollout=true ode_rollout_commit=teacher ode_vrfm=true ode_vrfm_beta=1e-3"; export ODE_ROLLOUT=1 ODE_VRFM=1;;                        # VRFM on MSE
  rollklvz)  EXTRA="ode_rollout=true ode_rollout_commit=teacher ode_loss_type=kl_local ode_vrfm=true ode_vrfm_beta=1e-3"; export ODE_ROLLOUT=1 ODE_VRFM=1;; # VRFM on KL
  rollvzsmoke)   EXTRA="ode_rollout=true ode_rollout_commit=teacher ode_vrfm=true ode_vrfm_beta=1e-3"; export ODE_ROLLOUT=1 ODE_VRFM=1;;
  rollklvzsmoke) EXTRA="ode_rollout=true ode_rollout_commit=teacher ode_loss_type=kl_local ode_vrfm=true ode_vrfm_beta=1e-3"; export ODE_ROLLOUT=1 ODE_VRFM=1;;
  # ---- BASE VALIDATION: rollout + plain MSE, nothing else. Two variants
  # differing ONLY in direction scheduling, so the curriculum is the single
  # isolated factor. Both keep the 10-epoch stop so the budgets match.
  #   mse9 : every direction every epoch (8 compass as cf + no-op as the clean
  #          branch = all 9 trained on every sample)
  #   msec : the hard-direction ODE curriculum (prunes to 2-4 dirs after ep 0)
  rollmse9)  EXTRA="ode_rollout=true ode_rollout_commit=teacher ode_curriculum=true ode_curriculum_epochs=10 ode_curriculum_mode=all9"; export ODE_ROLLOUT=1;;
  rollmsec)  EXTRA="ode_rollout=true ode_rollout_commit=teacher ode_curriculum=true ode_curriculum_epochs=10 ode_curriculum_mode=hard"; export ODE_ROLLOUT=1;;
  rollklc)   EXTRA="ode_rollout=true ode_rollout_commit=teacher ode_loss_type=kl_local ode_curriculum=true ode_curriculum_epochs=10 ode_curriculum_mode=hard"; export ODE_ROLLOUT=1;;   # SAMPLED (hard-curriculum) kl_local — canonical name; recipe == rollkl default
  rollkl9)   EXTRA="ode_rollout=true ode_rollout_commit=teacher ode_loss_type=kl_local ode_curriculum=true ode_curriculum_epochs=10 ode_curriculum_mode=all9"; export ODE_ROLLOUT=1;;   # all-9 kl_local control (STAGED, not submitted)
  # Gaussian repulsor on the MSE rollout base (N(0,1), inverse-square, no
  # deadband). Queued only after the base is validated.
  rollmse9g) EXTRA="ode_rollout=true ode_rollout_commit=teacher ode_curriculum=true ode_curriculum_epochs=10 ode_curriculum_mode=all9 ode_grep_weight=0.5"; export ODE_ROLLOUT=1;;
  rollsmoke) EXTRA="ode_rollout=true ode_rollout_commit=teacher"; export ODE_ROLLOUT=1;;   # smoke of the rollout stage
  # Gaussian repulsor on the rollout MSE base (roll + 1/d^2 from N(0,1),
  # _grep_chunk). MU-ONLY at w=0.1 (user sign-off 2026-08-14 after the d2
  # measurement): every sigma_c sits BELOW 1 on this data, so the sigma half
  # of d2 grows as the rollout contracts (5.52 -> 7.47 over chunks 0-5) and
  # 1/d2 would REWARD contraction; the mu half is right-signed (mean collapse
  # shrinks Sum mu_c^2 3.25 -> 1.56, toward the gaussian mean, and the term
  # resists it). w=0.1 puts the dose at regulariser scale (~0.04 vs base
  # ~0.03), not the 2-3.5x dominance w=0.5 would give. Same hard curriculum
  # as roll so the repulsor is the single variable vs the roll arm.
  rollrep)      EXTRA="ode_rollout=true ode_rollout_commit=teacher ode_grep_weight=0.1 ode_grep_components=mu"; export ODE_ROLLOUT=1;;
  rollrepsmoke) EXTRA="ode_rollout=true ode_rollout_commit=teacher ode_grep_weight=0.1 ode_grep_components=mu"; export ODE_ROLLOUT=1;;
  # STAGED, NOT YET SUBMITTED (user 2026-08-14: wait for rollkl + rollrep
  # results first). kl_local base + mu-repulsor: kl_local's per-cell KL pulls
  # dispersion up toward the teacher (variance channel) while the mu-only
  # 1/d^2 term fights the gaussian-residue mean drift (exposure-bias channel).
  rollklrep)      EXTRA="ode_rollout=true ode_rollout_commit=teacher ode_loss_type=kl_local ode_grep_weight=0.1 ode_grep_components=mu"; export ODE_ROLLOUT=1;;
  rollklrepsmoke) EXTRA="ode_rollout=true ode_rollout_commit=teacher ode_loss_type=kl_local ode_grep_weight=0.1 ode_grep_components=mu"; export ODE_ROLLOUT=1;;
  # v2 DIRECTIONAL repulsor on the kl_local base (2026-08-15): projection onto
  # the teacher chunk-mean direction — purple-cheat immune (CPU-verified:
  # orthogonal offset relief == 0, descent cos 1.0 with teacher means).
  rollklrep2)      EXTRA="ode_rollout=true ode_rollout_commit=teacher ode_loss_type=kl_local ode_grepdir_weight=0.1"; export ODE_ROLLOUT=1;;
  rollklrep2smoke) EXTRA="ode_rollout=true ode_rollout_commit=teacher ode_loss_type=kl_local ode_grepdir_weight=0.1"; export ODE_ROLLOUT=1;;
  # KLTS (user design 2026-08-15): learned per-(chunk,rung) temperature on the
  # denoising chord, kl_local base, zero-init (tau=1). Trains in the emd-head
  # high-lr group; exported in the ckpt and applied at serve (ODE_KLTS_CKPT).
  rollklts)      EXTRA="ode_rollout=true ode_rollout_commit=teacher ode_loss_type=kl_local ode_klts=true"; export ODE_ROLLOUT=1;;
  rollkltssmoke) EXTRA="ode_rollout=true ode_rollout_commit=teacher ode_loss_type=kl_local ode_klts=true"; export ODE_ROLLOUT=1;;
  # STAGED, NOT SUBMITTED — ONLINE ATTRACTOR TRACKER (af_model/
  # attractor_tracker.py): per-direction AR(1) attractor estimation in stat
  # space from the training rollouts + inverse-square repulsion from a
  # target-network-frozen copy; pointness/distinctness/consistency gates give
  # auto-shutoff when no point attractor is findable (manifold hypothesis:
  # degenerate solutions are point collapses, the true solution is a
  # manifold, so gate-dark = free convergence).
  rollar)      EXTRA="ode_rollout=true ode_rollout_commit=teacher ode_attractor_enabled=true ode_attractor_weight=0.1"; export ODE_ROLLOUT=1;;
  rollarsmoke) EXTRA="ode_rollout=true ode_rollout_commit=teacher ode_attractor_enabled=true ode_attractor_weight=0.1 ode_attractor_warmup=5 ode_attractor_freeze_k=5"; export ODE_ROLLOUT=1;;  # smoke: tiny warmup/freeze so the smoke actually exercises the actuation path
  rolledsmoke) EXTRA="ode_rollout=true ode_rollout_commit=teacher ode_edist_weight=0.005"; export ODE_ROLLOUT=1;;  # edist SMOKE
  rolled)   EXTRA="ode_rollout=true ode_rollout_commit=teacher ode_edist_weight=0.005"; export ODE_ROLLOUT=1;;                   # rollout + MSE + within-context energy distance (0.005: at 0.5 it dominated the base 116-683x)
  rollkled) EXTRA="ode_rollout=true ode_rollout_commit=teacher ode_loss_type=kl_local ode_edist_weight=0.005"; export ODE_ROLLOUT=1;; # rollout + KL + within-context energy distance
  rollkl)   EXTRA="ode_rollout=true ode_rollout_commit=teacher ode_loss_type=kl_local"; export ODE_ROLLOUT=1;;  # rollout + KL: per-channel Gaussian KL on every committed chunk
  roll)     EXTRA="ode_rollout=true ode_rollout_commit=teacher"; export ODE_ROLLOUT=1;;                 # KV-cache AR rollout: student trained the way 14e is INFERENCED
  rollsf)   EXTRA="ode_rollout=true ode_rollout_commit=schedule ode_rollout_commit_p=0.5"; export ODE_ROLLOUT=1;;  # + scheduled self-forcing (student commits its own chunk 50% of the time)
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
  # ---- 10K-SCENE CAMPAIGN (2026-08-16): 2x2 objective x schedule on the
  # v14e_pilot_dir8n_10k pool (10,003 windows; backward capped at 1,603 by
  # class availability). At 10k scenes one epoch is ~2,500 batches, so a
  # 500-step run never leaves the full-diversity epoch-0 phase — the
  # sampled-vs-all9 axis here is pure batch composition (grouped fan vs
  # shuffled), no pruning. Ckpt every 100, rotation OFF (keep 99: the
  # default keep-3 hard-unlinked the round-2 step-200 peaks).
  roll10k)      EXTRA="ode_rollout=true ode_rollout_commit=teacher keep_last_ckpts=99"; export ODE_ROLLOUT=1; DATAV=dir8n_10k;;
  rollkl10k)    EXTRA="ode_rollout=true ode_rollout_commit=teacher ode_loss_type=kl_local keep_last_ckpts=99"; export ODE_ROLLOUT=1; DATAV=dir8n_10k;;
  rollmse910k)  EXTRA="ode_rollout=true ode_rollout_commit=teacher ode_curriculum=true ode_curriculum_epochs=10 ode_curriculum_mode=all9 keep_last_ckpts=99"; export ODE_ROLLOUT=1; DATAV=dir8n_10k;;
  rollkl910k)   EXTRA="ode_rollout=true ode_rollout_commit=teacher ode_loss_type=kl_local ode_curriculum=true ode_curriculum_epochs=10 ode_curriculum_mode=all9 keep_last_ckpts=99"; export ODE_ROLLOUT=1; DATAV=dir8n_10k;;
  roll10ksmoke) EXTRA="ode_rollout=true ode_rollout_commit=teacher keep_last_ckpts=99"; export ODE_ROLLOUT=1; DATAV=dir8n_10k;;
  # PROPER ENERGY SCORE arm (user 2026-08-16): m=2 noise branches per chunk,
  # strictly proper scoring rule (CRPS generalization) vs the single teacher
  # realization — distributional without a critic; spread term sign-correct
  # by construction. all9 schedule: never worse than sampled in rounds 1-2
  # and the single-delta control vs rollmse910k. ~2x step time (double
  # ladder), hence the longer wall at submit.
  rolles10k)      EXTRA="ode_rollout=true ode_rollout_commit=teacher ode_es=true ode_curriculum=true ode_curriculum_epochs=10 ode_curriculum_mode=all9 keep_last_ckpts=99"; export ODE_ROLLOUT=1; DATAV=dir8n_10k;;
  rolles10ksmoke) EXTRA="ode_rollout=true ode_rollout_commit=teacher ode_es=true ode_curriculum=true ode_curriculum_epochs=10 ode_curriculum_mode=all9 keep_last_ckpts=99"; export ODE_ROLLOUT=1; DATAV=dir8n_10k;;
  *) echo "bad ARM=$ARM"; exit 1;;
esac

# DEFAULTS (user directive): 8 directions + no-op data, intelligent
# (hard-direction) sampling. Flip/counterfactual style is retired.
: "${DATAV:=dir8n}"
case "$EXTRA" in
  *alldir_batches*) : ;;                       # arm set it explicitly
  *) EXTRA="alldir_batches=true $EXTRA"; export ODE_ALLDIR=1 ;;
esac
case "$EXTRA" in
  # NOT *ode_curriculum* -- that glob also matches `ode_curriculum_mode=`, so
  # the actsplit/all9 arms silently ran with the curriculum (and its 10-epoch
  # stop) DISABLED, differing from the controls on two axes at once.
  *ode_curriculum=*) : ;;
  *) EXTRA="ode_curriculum=true ode_curriculum_epochs=${CURRIC_EPOCHS:-10} $EXTRA" ;;
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
    logdir=$LOGDIR config_name=pilot_flip2_${ARM} \
    wandb_name=pilot_flip2_${ARM} > logs/ode14e_pilot_flip2_${ARM}.log 2>&1 \
    && touch "$LOGDIR/.train_done"
  # LAUNCH-FLAKE RETRY (2026-08-15): three arm launches died on transient
  # startup networking (cotracker hub RemoteDisconnected 6015319, wandb
  # sentry 6020346, rendezvous socket timeout 6021280) — all BEFORE the
  # first training step. Retry once iff the log shows a network signature
  # and training never actually started ("Trainer ready" absent); a real
  # code failure repeats identically and still TRAIN-FAILs.
  if [ ! -f "$LOGDIR/.train_done" ] \
     && grep -qaE "RendezvousConnectionError|DistNetworkError|RemoteDisconnected|Failed to recv|sentry" "logs/ode14e_pilot_flip2_${ARM}.log" \
     && ! grep -qa "Trainer ready" "logs/ode14e_pilot_flip2_${ARM}.log"; then
    echo "LAUNCH-FLAKE $ARM: startup network failure, retrying once in 90s"
    sleep 90
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
      logdir=$LOGDIR config_name=pilot_flip2_${ARM} \
      wandb_name=pilot_flip2_${ARM} >> logs/ode14e_pilot_flip2_${ARM}.log 2>&1 \
      && touch "$LOGDIR/.train_done"
  fi
  if [ ! -f "$LOGDIR/.train_done" ]; then
    echo "TRAIN-FAIL $ARM — training did not complete; last 40 lines:"
    tail -40 logs/ode14e_pilot_flip2_${ARM}.log
    exit 1        # so smokes can gate the full runs via --dependency=afterok
  fi
fi

CKPT=$LOGDIR/$CKPT_NAME
[ -f "$CKPT" ] || CKPT=$(ls -t $LOGDIR/action_ode_step*.pt 2>/dev/null | head -1)
if [[ "$EXTRA" == *ode_klts=true* ]]; then
  # KLTS probes must serve WITH the learned temperature applied.
  export ODE_KLTS_CKPT=$PWD/$CKPT
fi
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
  # Per-ARM output paths: both tools wrote FIXED filenames, so with 12 arms
  # finishing independently the last one to run silently overwrote the rest.
  FV_OUT=analysis/eval_final/flow_viz
  mkdir -p "$FV_OUT/mc_${ARM}"
  MC_OUT=$FV_OUT/mc_${ARM} MC_RUNS=pilot3_flip2${ARM} python utils/motion_check.py || true
  SD_OUT=$FV_OUT/stat_drift_${ARM}.csv SD_RUNS=pilot2_flip2:pilot3_flip2${ARM} \
    python utils/stat_drift.py || true
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
