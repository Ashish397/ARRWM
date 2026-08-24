#!/bin/bash
# #####################################################################
# ##  NEVER EDIT THIS SCRIPT WHILE A RUN IS EXECUTING IT.            ##
# #####################################################################
# bash does not slurp a script into memory. It reads it incrementally
# and remembers a BYTE OFFSET into the file. Editing the file IN PLACE
# (`vi`, `sed -i` on some builds, an editor that truncates+rewrites,
# any tool that keeps the same inode) makes the running shell resume at
# a stale offset in changed bytes -- it will execute a fragment of a
# line, skip a command, or silently run the wrong branch. It does not
# error; it corrupts the tail of the run.
#
# HIT FOR REAL on 2026-08-22: this file and `run_ganfix_poolrich.sh`
# were edited at 10:27 while the 09:47 poolrich run was executing both,
# putting the chained eval at risk.
#
# BEFORE editing, check what is live:
#     squeue -u $USER
#     ls logs/.holder_cmd_*.sh logs/.holder_running_*.sh
#     tail -1 logs/hold-*_<jobid>.out      # "HOLDER command exit=" = idle
#
# THE REMEDY -- write a temp file in the SAME directory and rename over
# the original. rename(2) is atomic and gives the new content a NEW
# inode, so a running bash keeps reading the old inode to completion:
#     cp script.sh script.sh.tmp
#     <edit script.sh.tmp>
#     mv -f script.sh.tmp script.sh          # atomic, same filesystem
# In Python: write the temp, then `os.replace(tmp, path)`
# (plus `shutil.copymode(path, tmp)` to keep the +x bit).
#
# If a run IS live and you cannot wait: CLONE the script to a new name
# and edit the clone. Never touch the one being executed.
# #####################################################################
set -euo pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

: "${HOLDER:?set HOLDER to a running holder job ID}"
: "${PORTOFF:?set PORTOFF to a unique integer}"
: "${DARM:?set DARM to the run tag}"

MODE=${MODE:-none}
MAXSTEPS=${MAXSTEPS:-200}
GANW=${GANW:-0.01}
RUNSTAMP=${RUNSTAMP:-$(date +%H%M%S)}
EVAL_CHUNKS=${EVAL_CHUNKS:-100}
export RUNSTAMP MAXSTEPS
export CKPT_EVERY=100000 STAT_ANCHOR=1.0
export ODE_CKPT=logs/ode14e_pilot/run3_flip2_rollkl10k/action_ode_step0000400.pt
export AR_HEAD=0.0 TF_HEAD=1.0

RUN_NAME="dmd10k_${DARM}_h${HOLDER}_${RUNSTAMP}"
RUN_DIR="logs/dmd10k_${DARM}/${RUN_NAME}"
EVAL_CKPT=${EVAL_CKPT:-"$RUN_DIR/eval_step$(printf '%04d' "$MAXSTEPS").pt"}

# ---- rollout-horizon knobs (defaults reproduce the nogan200/wave01 arms) ----
# "Train the failure horizon": the mangle onset measured by
# analysis/rollout_realism.py is 8-12 s, while training only ever rolls
# MAXLEN latent frames (60 = 15 s). Raising these makes the generator LIVE
# through deeper self-context. max_length=90 is already validated OOM-free
# (rolldepth arm, ROLLING_CAMPAIGN.md wave-3 verdicts).
MAXLEN=${MAXLEN:-60}
RDMIN=${RDMIN:-2}
RDMAX=${RDMAX:-6}
MAXROLLS=${MAXROLLS:-6}

# ---- disc timestep (item 8: deconfound wavelet x timestep) ----------------
# The wavelet branch historically forced disc t=0. APT reports t=0 diffusion
# features collapse discrimination, so "wavelet behaviour" may really be a
# timestep effect. DISC_T=t0|sampled picks the cell; every other knob is held.
DISC_T=${DISC_T:-t0}
case "$DISC_T" in
  t0)      DISC_T_EXTRA="ladd_disc_sample_t=false" ;;
  sampled) DISC_T_EXTRA="ladd_disc_sample_t=true ladd_disc_t_min=20 ladd_disc_t_max=980 ladd_disc_timestep_shift=5.0" ;;
  *) echo "Unknown DISC_T=$DISC_T (expected t0 or sampled)" >&2; exit 2 ;;
esac

# Start from the KL-ODE student, never from another DMD checkpoint. These
# are full CARN lineages whose global step begins at zero.
COMMON_EXTRA="auto_resume=false strict_resume_load=true resume_load_fake_score=false fake_score_init_from_teacher=true boundary_vae_roundtrip=true max_rolls_per_ride=${MAXROLLS} rolling_random_depth_enabled=true rolling_random_depth_min=${RDMIN} rolling_random_depth_max=${RDMAX} dmd_only_first_chunk_per_ride=false dmd_only_last_chunk_per_ride=false dmd_42f_rolling_sup_new=true streaming_chunk_size=18 num_chunks_roll_forward=3 streaming_min_new_frame=9 streaming_max_length=${MAXLEN} dmd_rolling_ctx_last_rung=true rollout_viz_source=finish dmd_42f_clean_match_enabled=true dmd_42f_clean_match_max_drift_frames=12 dmd_42f_clean_match_min_improve=0.15 dmd_42f_clean_drift_enabled=true dmd_42f_clean_drift_chunks=3 dmd_42f_clean_drift_couple_rope=true dmd_42f_clean_match_drift_compose=true dmd_42f_fix_clean_counterpart=true dmd_supervise_roll_mode=random dmd_sample_at_rungs=false dmd_score_t_min=20 dmd_score_t_max=980 timestep_shift=5.0 dmd_loss_start_step=0 dmd_loss_warmup_steps=20 dmd_normalization_enabled=true dmd_normalization_denom_floor=0.05 dmd_ar_normalization_source=ar lr=1e-5 fake_lr=2e-6 streaming_fake_updates_per_gen=4 fake_score_ema_weight=0.95 ema_weight=0.99 ema_start_step=200 carn_seam_affine_lambda=0.5 flash_dmd_enabled=true flash_dmd_gan_t=60 save_full_checkpoint=false eval_checkpoint_path=${EVAL_CKPT}"

case "$MODE" in
  none)
    GAN_EXTRA="gan_enabled=false gan_loss_weight=0.0"
    ;;
  wavelet)
    GAN_EXTRA="gan_enabled=true gan_loss_weight=${GANW} gan_lr=5e-6 gan_updates_per_step=1 gan_disc_start_step=20 gan_critic_warmup_steps=40 gan_warmup_steps=25 gan_warmup_shape=linear ladd_proj_dim=256 ladd_scalar_output=true ladd_freeze_projector_mixing=true ladd_use_csm=true ladd_use_lateral_proj=false ladd_use_prompt_cond=false ladd_cmap_dim=0 ladd_stat_head_enabled=false ladd_wavelet_hf_enabled=true ladd_wavelet_hf_augment=true ladd_wavelet_hf_drop_ll=true ladd_wavelet_hf_ll_weight=0.15 ladd_wavelet_hf_adapter_init_gain=0.1 ladd_gt_transition_enabled=true ladd_gt_transition_match=true ladd_gt_transition_match_k=4 ladd_gt_transition_match_pool=8 ladd_gt_transition_match_max_real=12 ladd_gt_transition_mean_equalize=true ladd_gt_transition_xeq_per_channel=false ladd_gt_transition_std_equalize=false ladd_gt_transition_xeq_preserve_delta=true ladd_gt_transition_gen_detach_former=true ladd_real_pool_cross_ride=4096 ladd_real_pool_push_per_ride=8 ${DISC_T_EXTRA} ladd_real_match_fake_t=false ladd_r1_gamma=1.0 ladd_r1_every_n_steps=2 ladd_r1_num_samples=6 ladd_r1_normalize_tokens=false ladd_diff_aug_policy=flip ladd_gt_transition_action_blind=true"
    ;;
  projected)
    GAN_EXTRA="gan_enabled=true gan_loss_weight=${GANW} gan_lr=5e-6 gan_updates_per_step=1 gan_disc_start_step=20 gan_critic_warmup_steps=40 gan_warmup_steps=25 gan_warmup_shape=linear ladd_proj_dim=256 ladd_scalar_output=true ladd_freeze_projector_mixing=true ladd_use_csm=true ladd_use_lateral_proj=false ladd_use_prompt_cond=false ladd_cmap_dim=0 ladd_stat_head_enabled=false ladd_wavelet_hf_enabled=false ladd_wavelet_hf_augment=false ladd_gt_transition_enabled=true ladd_gt_transition_match=true ladd_gt_transition_match_k=4 ladd_gt_transition_match_pool=8 ladd_gt_transition_match_max_real=12 ladd_gt_transition_mean_equalize=true ladd_gt_transition_xeq_per_channel=false ladd_gt_transition_std_equalize=false ladd_gt_transition_xeq_preserve_delta=true ladd_gt_transition_gen_detach_former=true ladd_real_pool_cross_ride=4096 ladd_real_pool_push_per_ride=8 ladd_disc_sample_t=true ladd_disc_t_min=20 ladd_disc_t_max=980 ladd_disc_timestep_shift=5.0 ladd_real_match_fake_t=false ladd_r1_gamma=1.0 ladd_r1_every_n_steps=2 ladd_r1_num_samples=6 ladd_r1_normalize_tokens=false ladd_diff_aug_policy=flip ladd_gt_transition_action_blind=true"
    ;;
  raw)
    # Item 8 cell: the wavelet arm with the WAVELET OFF and nothing else
    # changed, so `raw vs wavelet` and `t0 vs sampled` (DISC_T) form a clean
    # 2x2 against fullcarn_bidir_kl_wave01.
    GAN_EXTRA="gan_enabled=true gan_loss_weight=${GANW} gan_lr=5e-6 gan_updates_per_step=1 gan_disc_start_step=20 gan_critic_warmup_steps=40 gan_warmup_steps=25 gan_warmup_shape=linear ladd_proj_dim=256 ladd_scalar_output=true ladd_freeze_projector_mixing=true ladd_use_csm=true ladd_use_lateral_proj=false ladd_use_prompt_cond=false ladd_cmap_dim=0 ladd_stat_head_enabled=false ladd_wavelet_hf_enabled=false ladd_wavelet_hf_augment=false ladd_gt_transition_enabled=true ladd_gt_transition_match=true ladd_gt_transition_match_k=4 ladd_gt_transition_match_pool=8 ladd_gt_transition_match_max_real=12 ladd_gt_transition_mean_equalize=true ladd_gt_transition_xeq_per_channel=false ladd_gt_transition_std_equalize=false ladd_gt_transition_xeq_preserve_delta=true ladd_gt_transition_gen_detach_former=true ladd_real_pool_cross_ride=4096 ladd_real_pool_push_per_ride=8 ${DISC_T_EXTRA} ladd_real_match_fake_t=false ladd_r1_gamma=1.0 ladd_r1_every_n_steps=2 ladd_r1_num_samples=6 ladd_r1_normalize_tokens=false ladd_diff_aug_policy=flip ladd_gt_transition_action_blind=true"
    ;;
  strict)
    # STRICT ACTION-CONDITIONED transition critic on the IDENTICAL DMD recipe
    # as the `none` control and the `wavelet` arm (COMMON_EXTRA above). This is
    # the matched arm the Immediate Decision Rule asks for: same route, seed,
    # checkpoint step, inference CARN -- the ONLY delta vs `none` is the GAN.
    # Critic settings = the "full assembly" (cross-ride bank 4096, sampled disc
    # t shared real/fake, scalar logit, frozen projector mixing, R1 gamma 1,
    # 5 disc updates, lr_D 1e-5, strict action conditioning, no wavelet, no
    # mean equalization).
    GAN_EXTRA="gan_enabled=true gan_loss_weight=${GANW} gan_lr=1e-5 gan_updates_per_step=5 gan_disc_start_step=20 gan_critic_warmup_steps=20 gan_warmup_steps=25 gan_warmup_shape=linear ladd_proj_dim=256 ladd_scalar_output=true ladd_freeze_projector_mixing=true ladd_use_csm=true ladd_use_lateral_proj=false ladd_use_prompt_cond=false ladd_cmap_dim=0 ladd_stat_head_enabled=false ladd_wavelet_hf_enabled=false ladd_wavelet_hf_augment=false ladd_gt_transition_enabled=true ladd_gt_transition_match=true ladd_gt_transition_match_k=4 ladd_gt_transition_match_pool=8 ladd_gt_transition_match_max_real=12 ladd_gt_transition_mean_equalize=false ladd_gt_transition_cross_equalize=false ladd_gt_transition_std_equalize=false ladd_gt_transition_gen_detach_former=true ladd_real_pool_cross_ride=4096 ladd_real_pool_push_per_ride=8 ladd_disc_sample_t=true ladd_disc_t_min=20 ladd_disc_t_max=980 ladd_disc_timestep_shift=5.0 ladd_real_match_fake_t=false ladd_r1_gamma=1.0 ladd_r1_every_n_steps=2 ladd_r1_num_samples=6 ladd_r1_normalize_tokens=false ladd_diff_aug_policy=flip ladd_gt_transition_action_blind=false"
    ;;
  *)
    echo "Unknown MODE=$MODE (expected none, wavelet, raw, projected, or strict)" >&2
    exit 2
    ;;
esac

export DEXTRA="$COMMON_EXTRA $GAN_EXTRA real_teacher_causal_mask=false fake_score_causal_mask=false dmd_grad_target_norm=1.0"
echo "Starting full CARN lineage: arm=$DARM mode=$MODE steps=$MAXSTEPS init=$ODE_CKPT"
bash sbatch/_roll_holder.sh > "logs/exp_${DARM}.err" 2>&1

source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache
export HF_HUB_CACHE=$HF_HOME
export HUGGINGFACE_HUB_CACHE=$HF_HOME
export TRANSFORMERS_CACHE=$HF_HOME
export TMPDIR=/tmp
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

test -f "$EVAL_CKPT"
OUT="eval/${DARM}_step${MAXSTEPS}_carn05_60s"
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 LOCAL_RANK=0 python utils/eval_causal_AR.py \
  --student_ckpt "$EVAL_CKPT" \
  --config configs/ar_eval_dmd_student.yaml \
  --manifest logs/v14_balanced_weunz/.ride_manifest.pt \
  --rank_zarr 20240216101235.zarr \
  --rank_offset 100 \
  --rank_mode dataset \
  --rank_tag madrid60 \
  --encoded_root /projects/u6ex/fbots/frodobots_encoded_weunz \
  --caption_root /projects/u6ex/fbots/frodobots_captions/train \
  --motion_root /projects/u6ex/fbots/frodobots_motion \
  --ss_vae_checkpoint action_query/checkpoints/ss_vae_8free.pt \
  --seed 42 \
  --denoising_steps 4 \
  --mode ar \
  --ar_initial_chunks 3 \
  --ar_gen_chunks "$EVAL_CHUNKS" \
  --cache_chunks 7 \
  --no-ar_cache \
  --infinity_rope \
  --carn_seam_affine_lambda 0.5 \
  --label "${DARM}_step${MAXSTEPS}_carn05" \
  --output_dir "$OUT" \
  > "logs/eval_${DARM}_step${MAXSTEPS}_60s.err" 2>&1

echo "Completed full CARN lineage and 60-second eval: $DARM"
