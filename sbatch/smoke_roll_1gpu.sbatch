#!/bin/bash
#SBATCH --job-name=roll1gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# SINGLE-GPU rollout smoke -- exists purely to VERIFY THE CheckpointError FIX FAST.
#
# The 2-node rule for rollout smokes is a DATA-SAMPLING constraint, not a
# property of the rollout: _CurriculumBatches bin-packs to exactly `world`, so
# with world < the max group size (8) it yields an empty batch list and spins.
# Turning the curriculum + alldir batching OFF removes that constraint.
#
# The CheckpointError being chased is LOCAL autograd: gradient checkpointing
# recomputes a block against a kv_cache whose requires_grad state changed after
# _apply_cache_updates wrote into it. That reproduces on ONE GPU with no DDP.
# 1 GPU + 100G backfills onto a partially-used node in minutes, where 2 EMPTY
# nodes queued for 9.5h and never landed (jobs 6001818 / 6004082).
#
# This does NOT replace the 2-node smoke: it cannot exercise DDP no_sync or the
# curriculum sampler. It answers only "does the backward survive checkpointing".
set -x
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=$PWD:$PWD/action-forcing TMPDIR=/tmp
export AF_SNAPSHOT_STEPS="0,15,18,19,-1" AF_EVAL_STEPS=20
export ODE_ROLLOUT=1
export WANDB_MODE=disabled
export MASTER_ADDR=127.0.0.1 MASTER_PORT=$((29500 + SLURM_JOB_ID % 16000))

LOGDIR=logs/ode14e_pilot/roll1gpu_$SLURM_JOB_ID
mkdir -p "$LOGDIR"

torchrun --nnodes=1 --nproc_per_node=1 \
    --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
    --rdzv_endpoint=127.0.0.1:$MASTER_PORT \
  action-forcing/train.py \
    --config configs/action_ode_distill_F.yaml \
    chunked_lmdb=true ode_chunked_supervision=true \
    cd_teacher_loss_enabled=false cd_student_loss_enabled=false \
    ode_rollout=true ode_rollout_commit=teacher \
    alldir_batches=false ode_curriculum=false \
    clean_root=/projects/u6ex/fbots/frodobots_lmdb/v14e_pilot_dir8n \
    cf_root=/projects/u6ex/fbots/frodobots_lmdb/v14e_pilot_dir8n \
    clean_only=false require_cf=false lambda_cf=1.0 \
    random_steps="[0,15,18,19]" eval_inference_steps=20 \
    generator_ckpt=/scratch/u6ex/as1748.u6ex/ARRWM/logs/v14e_pca8_raw/causal_lora_step0005000.pt \
    total_steps=2 save_interval=1000 eval_interval=1000 \
    ckpt_skip_optimizer=true ckpt_local_stage=true \
    logdir=$LOGDIR config_name=roll1gpu wandb_name=roll1gpu \
    > logs/roll1gpu_${SLURM_JOB_ID}.log 2>&1
RC=$?
echo "=== TRAIN EXIT $RC ==="
grep -aE "CheckpointError|Traceback|RuntimeError|AssertionError|step |loss" logs/roll1gpu_${SLURM_JOB_ID}.log | tail -25
exit $RC
