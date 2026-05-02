#!/bin/bash
# Smoke harness for the new knobs (real_teacher_input_source=mix +
# aux_student_gt_mse_loss_weight=0.1 + yinyang removed). Designed to
# be launched inside an EXISTING interactive srun bash allocation via
# ``srun --jobid=$JID --overlap ... bash smoke_run_attached.sh``.
# Mirrors smoke_action_forcing_phase1_gan_online_teacher.sbatch's T2
# block so the same per-step DDP collective pattern is exercised.

set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1_gan.yaml
LOGROOT=/scratch/u6ex/as1748.u6ex/ARRWM/logs/smoke_phase1_gan_aux_inputmix_j${SLURM_JOB_ID:-attached}
T2_LOG="$LOGROOT/T2"
mkdir -p "$T2_LOG"

SMOKE_ENCODED_ROOT=/scratch/u6ex/as1748.u6ex/ARRWM/logs/smoke_combined_rides
SMOKE_WANDB_PROJECT=longlive-phase1-smoke

source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm

CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR
export HF_HUB_CACHE=$CACHE_DIR
export HUGGINGFACE_HUB_CACHE=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR

export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn
export NCCL_DEBUG=WARN
export NCCL_IB_TIMEOUT=50
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MASTER_ADDR=$(hostname)
MASTER_PORT=$((29500 + ${SLURM_JOB_ID:-0} % 16000 + 7))

echo "CONFIG=$CONFIG"
echo "LOGROOT=$LOGROOT"
echo "MASTER=$MASTER_ADDR:$MASTER_PORT"
echo "Starting smoke on $(date)"

timeout --kill-after=30 1800 \
torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --rdzv_id=${SLURM_JOB_ID:-attached}_smk \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  trainer/causal_action_forcing_train.py \
  --config $CONFIG \
  --override \
    encoded_root=$SMOKE_ENCODED_ROOT \
    auto_resume=false \
    motion_ride_min_mean=0.0 \
    motion_skip_dead_rides=false \
    motion_start_threshold=0.0 \
    min_ride_frames=21 \
    boundary_vae_roundtrip=true \
    action_teacher_mode=off \
    action_critic_aux_enabled=false \
    state_probe_aux_enabled=false \
    max_rolls_per_ride=300 \
    collapse_mae_threshold=0.2 \
    mae_extension_threshold=0.2 \
    num_workers=0 \
    real_teacher_causal_mask=false \
    real_guidance_scale=1.0 \
    real_teacher_lr=5e-5 \
    real_teacher_warmup_steps=200 \
    real_teacher_input_source=mix \
    real_teacher_input_mix_gt_p=0.5 \
    dmd_frozen_teacher_pass_enabled=false \
    aux_teacher_loss_weight=1.0 \
    flash_dmd_split_timestep=500 \
    fake_score_ema_weight=0.95 \
    dfake_gen_update_ratio=1 \
    gan_loss_weight=1.0 \
    gan_disc_base_channels=128 \
    gan_warmup_steps=1000 \
    log_dir=$T2_LOG \
    wandb_dir=wandb \
    wandb_project=$SMOKE_WANDB_PROJECT \
    run_name=smoke_aux_inputmix_t2_ddp_j${SLURM_JOB_ID:-attached} \
    max_steps=20 \
    log_interval=1 \
    checkpoint_interval=999 \
    sample_interval=0 \
    vis_save_local=false \
  > "$T2_LOG/T2.log" 2>&1
T2_EXIT=$?

echo ""
echo "----- T2 diagnostics -----"
echo "  exit code:                  $T2_EXIT"
echo "  log file:                   $T2_LOG/T2.log"
echo "  log lines:                  $(wc -l < $T2_LOG/T2.log)"
N_STEPS=$(grep -cE "step=[0-9]+/20 gen_loss" "$T2_LOG/T2.log" || true)
HAS_LORA_TRAIN=$(grep -c "real_teacher_train_online=True.*LoRA params trainable" "$T2_LOG/T2.log" || true)
HAS_RT_OPT=$(grep -c "real_teacher optimizer built" "$T2_LOG/T2.log" || true)
HAS_GAN_DDP=$(grep -c "R3GAN discriminator built.*DDP=True" "$T2_LOG/T2.log" || true)
HAS_AUX_GT=$(grep -c "aux_student_gt_mse_loss" "$T2_LOG/T2.log" || true)
HAS_INPUT_MIX=$(grep -c "real_teacher_input_was_gt" "$T2_LOG/T2.log" || true)
echo "  step=N/20 gen_loss lines:    $N_STEPS"
echo "  LoRA-trainable telemetry:   $HAS_LORA_TRAIN"
echo "  real_teacher optimizer:     $HAS_RT_OPT"
echo "  R3GAN DDP-wrap:             $HAS_GAN_DDP"
echo "  aux_student_gt_mse logs:    $HAS_AUX_GT  (new knob: aux MSE loss)"
echo "  real_teacher_input_was_gt:  $HAS_INPUT_MIX  (new knob: input source mix)"
echo ""

if [ "$T2_EXIT" -eq 124 ] || [ "$T2_EXIT" -eq 137 ]; then
  echo "SMOKE FAILED: timeout (30 min)."
  tail -20 "$T2_LOG/T2.log"
  exit 42
fi
if [ "$T2_EXIT" -ne 0 ]; then
  echo "SMOKE FAILED: trainer exited $T2_EXIT — last 20 log lines:"
  tail -20 "$T2_LOG/T2.log"
  exit 42
fi
if [ "$N_STEPS" -lt 1 ]; then
  echo "SMOKE FAILED: trainer ran clean but logged 0 training steps."
  tail -20 "$T2_LOG/T2.log"
  exit 43
fi
echo "SMOKE PASSED: $N_STEPS/20 training steps logged."
[ "$HAS_AUX_GT" -gt 0 ]    && echo "  ✓ aux_student_gt_mse_loss telemetry observed"
[ "$HAS_INPUT_MIX" -gt 0 ] && echo "  ✓ real_teacher_input_was_gt telemetry observed"
echo "  Logs: $LOGROOT"
