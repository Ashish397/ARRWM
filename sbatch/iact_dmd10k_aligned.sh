#!/bin/bash
# INTERACTIVE DMD-10k run on an existing holder, with the FULL aligned
# override block of sbatch/train_dmd10k_stat.sbatch.
#
# WHY THIS EXISTS: sbatch/iact_dmd_smoke.sh is NOT an arm proxy. It is
# missing every one of the 2026-08-17 serve-contract alignment overrides
#   seed_prefill_mode=real, kv_cache_seed_headroom=true, infinity_rope=true,
#   streaming_chunk_size=12, dmd_context_clean_frames=9 (it has 18),
#   local_attn_size 21 (it has 24), dmd_42f_num_chunks=4 /
#   dmd_42f_gt_after_chunks=1 (it has 3/2), seed=1234,
#   sample_7chunk_enabled=false, keep_last_n_checkpoints
# i.e. exactly the settings whose absence produced the six degraded runs.
# The block below is a byte-for-byte copy of the sbatch's --override list
# (only log_dir / run_name / max_steps / checkpoint_interval differ).
#
#   ALLOC=<jobid> TAG=mse ODE_CKPT=<path> STEPS=35 CKPT_EVERY=5 \
#     bash sbatch/iact_dmd10k_aligned.sh [extra=overrides]
set -x
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR HF_HUB_CACHE=$CACHE_DIR HUGGINGFACE_HUB_CACHE=$CACHE_DIR TRANSFORMERS_CACHE=$CACHE_DIR
export ARRWM_ACTION_ENCODER=pca_raw
export ARRWM_MANIFEST_PICKLE=${DMD_MANIFEST:-/scratch/u6ex/as1748.u6ex/ARRWM/analysis/.dmd_weunz_manifest_min69.pt}
export TMPDIR=/tmp
unset LOCALDIR APPTAINER_CACHEDIR
export NCCL_CROSS_NIC=1 NCCL_SOCKET_IFNAME=hsn NCCL_DEBUG=WARN NCCL_IB_TIMEOUT=50
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
export TORCH_SHOW_CPP_STACKTRACES=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1

ALLOC=${ALLOC:?}; TAG=${TAG:?}; STEPS=${STEPS:-35}; CKPT_EVERY=${CKPT_EVERY:-5}
SAMP=${SAMP:-5}
ODE_CKPT=${ODE_CKPT:?}
NODES=${NODES:-$(squeue -j $ALLOC -h -o "%D" | tr -d ' ')}
# NODELIST pins the run to specific nodes of the holder — needed when an
# earlier step is still holding a GPU on one of them. Set NODES to match.
NODELIST=${NODELIST:-}
SRUN_W=""
if [ -n "$NODELIST" ]; then
  SRUN_W="-w $NODELIST"
  MASTER_ADDR=$(scontrol show hostnames "$NODELIST" | head -n1)
else
  MASTER_ADDR=$(scontrol show hostnames "$(squeue -j $ALLOC -h -o '%N')" | head -n1)
fi
[ -n "$MASTER_ADDR" ] || { echo "alloc $ALLOC has no nodes"; exit 1; }
TAGSUM=$(printf %s "$TAG" | cksum | cut -d' ' -f1)
MASTER_PORT=$((20000 + (ALLOC + TAGSUM) % 20000))
export MASTER_ADDR MASTER_PORT

LOG=logs/iact_dmd10k_${TAG}.log
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/dmd10k_iact_${TAG}
rm -rf "$LOGDIR"; mkdir -p "$LOGDIR"

srun --overlap --jobid=$ALLOC $SRUN_W --nodes=$NODES --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=64 \
  torchrun --nnodes=$NODES --nproc_per_node=4 \
    --rdzv_id=iactdmd${ALLOC}_${TAG} --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  trainer/causal_action_forcing_train.py \
  --config configs/action_forcing_phase3_dmd.yaml \
  --override \
    log_dir=$LOGDIR \
    wandb_dir=wandb \
    wandb_project=longlive-phase-3 \
    max_steps=$STEPS \
    checkpoint_interval=$CKPT_EVERY \
    keep_last_n_checkpoints=99 \
    sample_interval=$SAMP \
    vis_save_local=true \
    encoded_root=/projects/u6ex/fbots/frodobots_encoded_weunz \
    holdout_zarr_list=configs/holdout_rides.txt \
    holdout_eval_root=/projects/u6ex/fbots/frodobots_encoded_weu_holdout \
    holdout_eval_mode=cycle \
    sample_7chunk_enabled=false \
    max_ride_frames=900 \
    max_ride_frames_random=false \
    seed=1234 \
    action_teacher_mode="off" \
    action_critic_aux_enabled=false \
    state_probe_aux_enabled=false \
    generator_action_z_guidance_weight=0.0 \
    lora_action_critic_z_guidance_weight=0.0 \
    action_dims=[0,1] \
    action_critic_dims=[0,1] \
    denoising_step_list=[1000,625,357.142857,208.333333] \
    max_rolls_per_ride=1 \
    dfake_gen_update_ratio=5 \
    model_kwargs.local_attn_size=21 \
    num_workers=0 \
    sample_fps=16 \
    boundary_vae_roundtrip=false \
    local_attn_size_schedule=[[0,21]] \
    infinity_rope=true \
    seed_prefill_mode=real \
    kv_cache_seed_headroom=true \
    real_teacher_causal_mask=false \
    real_guidance_scale=0.0 \
    dmd_context_clean_frames=9 \
    streaming_chunk_size=12 \
    dmd_loss_weight=1.0 \
    dmd_loss_start_step=0 \
    dmd_loss_warmup_steps=20 \
    dmd_normalization_enabled=true \
    dmd_mae_gate_enabled=true \
    dmd_mae_gate_r_full=2.0 \
    dmd_mae_gate_ema=0.0 \
    dmd_mae_gate_min_weight=0.0 \
    dmd_mae_gate_exponent=0.75 \
    stat_anchor_loss_weight=0.0 \
    stat_anchor_mode=gt_window \
    stat_anchor_match_k=2 \
    stat_anchor_M2_short_weight=0.1 \
    stat_anchor_M2_long_weight=0.0 \
    stat_anchor_TV_short_weight=0.1 \
    stat_anchor_TV_long_weight=0.0 \
    stat_anchor_M1_short_weight=3e-9 \
    stat_anchor_M1_long_weight=0 \
    stat_anchor_SOS_short_weight=0.0 \
    stat_anchor_SOS_long_weight=0.0 \
    stat_anchor_STD_short_weight=0.0 \
    stat_anchor_STD_long_weight=0.0 \
    stat_anchor_rel_tol_short=0.0 \
    stat_anchor_rel_tol_long=0.0 \
    dmd_asymmetric_scoring_enabled=false \
    dmd_42f_enabled=true \
    dmd_42f_2chunk=false \
    dmd_42f_seed_last=false \
    dmd_42f_gt_after_chunks=${D42_AFTER:-1} \
    dmd_42f_rand_sup_slot=false \
    dmd_42f_allsup=false \
    dmd_42f_gt_anchor=true \
    dmd_42f_num_chunks=${D42_NS:-4} \
    dmd_42f_fix_clean_counterpart=false \
    dmd_42f_clean_drift_enabled=false \
    dmd_42f_clean_match_enabled=false \
    dmd_only_first_chunk_per_ride=true \
    flash_dmd_enabled=false \
    fake_score_ema_weight=0.95 \
    motion_start_threshold=5.0 \
    fake_score_lora_enabled=false \
    gan_enabled=false \
    gan_backbone=ladd_teacher_feat \
    gan_loss_weight=0.0 \
    real_teacher_train_online=false \
    dmd_frozen_teacher_pass_enabled=false \
    real_score_ema_weight=0.0 \
    aux_teacher_loss_weight=0.0 \
    aux_real_clean_x_source=gt \
    aux_noisy_from_raw_gt=true \
    aux_clean_x_random_window=false \
    use_8bit_adam=true \
    gen_gradient_checkpointing=true \
    forward_noiser_enabled=false \
    carn_recurse=false \
    forward_noiser_apply_gt_former=false \
    ladd_adjacent_chunks_enabled=false \
    real_score_gradient_checkpointing=true \
    fake_score_gradient_checkpointing=true \
    strict_ode_load=true \
    ode_generator_checkpoint=$ODE_CKPT \
    v14_teacher_checkpoint=/scratch/u6ex/as1748.u6ex/ARRWM/logs/v14e_pca8_raw/causal_lora_step0005000.pt \
    student_phase_lora_enabled=false \
    "$@" \
    run_name=dmd10k_iact_${TAG} > $LOG 2>&1
RC=$?
echo "=== EXIT $RC log=$LOG ==="
exit $RC
