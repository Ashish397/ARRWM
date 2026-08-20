#!/bin/bash
# INTERACTIVE DMD-10k smoke — srun --overlap into an existing allocation.
#   ALLOC=<jobid> TAG=d1 STEPS=3 bash sbatch/iact_dmd_smoke.sh [extra=overrides]
# Mirrors sbatch/train_dmd10k_stat.sbatch exactly (same override block), tiny
# step count, fresh logfile per attempt. Per-TAG rdzv port (holder lesson).
set -x
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR HF_HUB_CACHE=$CACHE_DIR HUGGINGFACE_HUB_CACHE=$CACHE_DIR TRANSFORMERS_CACHE=$CACHE_DIR
export ARRWM_ACTION_ENCODER=pca_raw
# Scan-free start: pre-built weunz manifest (version 4, from the 21.9GB
# .rec_scratch manifest). Override DMD_MANIFEST for the 10-ride smoke set.
export ARRWM_MANIFEST_PICKLE=${DMD_MANIFEST:-/scratch/u6ex/as1748.u6ex/ARRWM/analysis/.dmd_weunz_manifest_min69.pt}
CKPT_INT=${CKPT_INT:-100000}
export NCCL_CROSS_NIC=1 NCCL_SOCKET_IFNAME=hsn NCCL_DEBUG=WARN NCCL_IB_TIMEOUT=50
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
export TORCH_SHOW_CPP_STACKTRACES=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1

ALLOC=${ALLOC:?}; TAG=${TAG:-d1}; STEPS=${STEPS:-3}
ODE_CKPT=${ODE_CKPT:-/scratch/u6ex/as1748.u6ex/ARRWM/logs/ode14e_pilot/run3_flip2_roll10k/action_ode_step0000200.pt}
NODES=$(squeue -j $ALLOC -h -o "%D" | tr -d ' ')
MASTER_ADDR=$(scontrol show hostnames "$(squeue -j $ALLOC -h -o '%N')" | head -n1)
[ -n "$MASTER_ADDR" ] || { echo "alloc $ALLOC has no nodes"; exit 1; }
TAGSUM=$(printf %s "$TAG" | cksum | cut -d' ' -f1)
MASTER_PORT=$((20000 + (ALLOC + TAGSUM) % 20000))
export MASTER_ADDR MASTER_PORT

LOG=logs/iact_dmd_${TAG}.log
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/dmd10k_iact_${TAG}
rm -rf "$LOGDIR"; mkdir -p "$LOGDIR"

srun --overlap --jobid=$ALLOC --nodes=$NODES --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=64 \
  torchrun --nnodes=$NODES --nproc_per_node=4 \
    --rdzv_id=iactdmd${ALLOC}_${TAG} --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  trainer/causal_action_forcing_train.py \
  --config configs/action_forcing_phase3_dmd.yaml \
  --override \
    log_dir=$LOGDIR \
    wandb_dir=wandb \
    wandb_project=longlive-phase-3 \
    max_steps=$STEPS \
    checkpoint_interval=$CKPT_INT \
    encoded_root=/projects/u6ex/fbots/frodobots_encoded_weunz \
    holdout_zarr_list=configs/holdout_rides.txt \
    holdout_eval_root=/projects/u6ex/fbots/frodobots_encoded_weu_holdout \
    holdout_eval_mode=cycle \
    max_ride_frames=900 \
    max_ride_frames_random=true \
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
    model_kwargs.local_attn_size=24 \
    num_workers=0 \
    sample_fps=16 \
    boundary_vae_roundtrip=false \
    local_attn_size_schedule=[[0,24]] \
    real_teacher_causal_mask=false \
    real_guidance_scale=0.0 \
    dmd_context_clean_frames=18 \
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
    dmd_asymmetric_scoring_enabled=false \
    dmd_42f_enabled=true \
    dmd_42f_2chunk=false \
    dmd_42f_seed_last=false \
    dmd_42f_gt_after_chunks=2 \
    dmd_42f_rand_sup_slot=false \
    dmd_42f_allsup=false \
    dmd_42f_gt_anchor=true \
    dmd_42f_num_chunks=3 \
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
echo "=== EXIT $RC ==="
exit $RC
