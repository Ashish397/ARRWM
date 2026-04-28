#!/bin/bash
# One-shot diag launcher: run the action-forcing smoke for a few iters
# with the new pred_real / pred_fake / clean_x diag videos enabled.
# Designed to be invoked via:
#   srun --jobid=<existing> --overlap --ntasks=1 --gres=gpu:1 \
#        bash sbatch/_diag_smoke_launcher.sh <phase_low|phase_high> <log_dir>
set -e -o pipefail

PHASE=${1:?phase}
LOG_DIR=${2:?log_dir}
# Optional 3rd arg: extra ``--override key=value`` pairs to append. Use
# this from callers to A/B knobs (e.g. ``real_guidance_scale=0.0``)
# without editing the smoke yaml.
EXTRA_OVERRIDES=${3:-}

cd /scratch/u6ex/as1748.u6ex/ARRWM

# Capture our args before sourcing conda (which inherits positional args
# from the caller and would error "activate does not accept more than
# one argument" otherwise).
set --
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

if [[ "$PHASE" == "phase_low" ]]; then
    ENCODED_ROOT=/projects/u6ex/fbots/frodobots_encoded_smoke_low
    RUN_NAME=smoke_diag_low
elif [[ "$PHASE" == "phase_high" ]]; then
    ENCODED_ROOT=/projects/u6ex/fbots/frodobots_encoded_smoke_high
    RUN_NAME=smoke_diag_high
else
    echo "unknown phase $PHASE" >&2
    exit 2
fi

PORT=$((30000 + RANDOM % 20000))

# max_steps=4 produces sample videos at step 1, 2, 4 (sample_at_steps:[1] +
# sample_interval:2). Each iter is ~30s warm. checkpoint_interval=99999 in
# config so no in-loop checkpoint; the post-loop save WILL fire — kill the
# job step after the step-4 videos land if you want to skip the 17 GB write.
torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --master_addr=127.0.0.1 \
  --master_port=$PORT \
  trainer/causal_action_forcing_train.py \
  --config configs/smoke_extender.yaml \
  --override \
    encoded_root=$ENCODED_ROOT \
    log_dir=$LOG_DIR \
    wandb_dir=./wandb \
    disable_wandb=true \
    run_name=$RUN_NAME \
    max_steps=4 \
    $EXTRA_OVERRIDES
