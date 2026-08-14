#!/bin/bash
#SBATCH --job-name=roll-ckptdbg
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# DIAGNOSTIC smoke: reproduce the rollout CheckpointError WITH torch's
# checkpoint debug mode on, so the diverging OP is printed instead of a bare
# shape list.
#
# Two root-cause theories have now been falsified by experiment:
#   1. "later chunks pollute the cache, so backward per rung"   -> job 5998622
#   2. "_apply_cache_updates flips the cache to requires_grad"  -> job 6004035
# Both produced a byte-identical failure (position 5, saved [1536,1536] vs
# recomputed [4683,1536]). The shape list alone cannot distinguish causes,
# so stop inferring from it: AF_CKPT_DEBUG=1 makes torch log the forward and
# recompute op stacks and name the operation that differs.
#
# Short walltime on purpose -- this is expected to FAIL fast; it is run for
# the diagnostic output, not for a training result.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 16000))
export NCCL_CROSS_NIC=1 NCCL_SOCKET_IFNAME=hsn OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AF_CKPT_DEBUG=1
export LAUNCHER="srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=4 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"
export TOTAL_STEPS=2
export SAVE_EVERY=1000
DRIVER_COPY=$SLURM_SUBMIT_DIR/logs/.driver_${SLURM_JOB_ID}.sh
cp /scratch/u6ex/as1748.u6ex/ARRWM/sbatch/train_flip2_arms_cde.sh "$DRIVER_COPY"
ARM=rollsmoke bash "$DRIVER_COPY"
