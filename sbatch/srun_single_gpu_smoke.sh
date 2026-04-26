#!/bin/bash
# =====================================================================
# Single-GPU smoke runner for an existing srun allocation.
#
# Usage:
#   ./sbatch/srun_single_gpu_smoke.sh <variant_label> <config_path> [extra_overrides...]
#
# Example:
#   ./sbatch/srun_single_gpu_smoke.sh aux_gan_g1 \
#       configs/action_forcing_phase1_aux_gan.yaml \
#       run_name=srun_aux_gan_g1
#
# Designed to be invoked via:
#   srun --jobid=<your_alloc> --overlap bash sbatch/srun_single_gpu_smoke.sh ...
# so the trainer runs inside the user's existing 1-GPU allocation.
# Mirrors the env / torchrun setup used by the production smoke
# sbatch scripts (smoke_action_forcing_phase1*.sbatch) so we get a
# faithful single-GPU reproduction.
# =====================================================================
set -e -o pipefail

VARIANT_LABEL="${1:?need variant label}"
CONFIG="${2:?need config path}"
shift 2
EXTRA_OVERRIDES=("$@")
# Clear positional parameters before sourcing conda. The activate script
# inspects $@ and chokes if it sees more than one arg.
set --

cd /scratch/u6ex/as1748.u6ex/ARRWM

LOGROOT="/scratch/u6ex/as1748.u6ex/ARRWM/logs/srun_smoke_${VARIANT_LABEL}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGROOT"

source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm

CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR
export HF_HUB_CACHE=$CACHE_DIR
export HUGGINGFACE_HUB_CACHE=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
mkdir -p "$CACHE_DIR"

export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn
export NCCL_DEBUG=WARN
export NCCL_IB_TIMEOUT=50
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled
# Synchronous CUDA so any device-side assertion gives us the *real* call
# site instead of failing asynchronously inside the next op.
export CUDA_LAUNCH_BLOCKING=1

# Pick a fresh-ish rdzv port so back-to-back invocations don't collide.
RDZV_PORT=$((29500 + (RANDOM % 10000)))

echo "===================================================================="
echo "  SRUN SINGLE-GPU SMOKE: ${VARIANT_LABEL}"
echo "  CONFIG=${CONFIG}"
echo "  LOGROOT=${LOGROOT}"
echo "  CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING}"
echo "  Started: $(date)"
echo "===================================================================="

torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --rdzv_id="srun_${VARIANT_LABEL}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="127.0.0.1:${RDZV_PORT}" \
  trainer/causal_action_forcing_train.py \
  --config "$CONFIG" \
  --override \
    log_dir="$LOGROOT" \
    wandb_dir=wandb \
    run_name="srun_smoke_${VARIANT_LABEL}" \
    max_steps=1 \
    log_interval=1 \
    checkpoint_interval=999 \
    "${EXTRA_OVERRIDES[@]}" \
  2>&1 | tee "$LOGROOT/run.log"

EXIT_CODE=${PIPESTATUS[0]}

echo "===================================================================="
if [[ $EXIT_CODE -eq 0 ]]; then
  echo "  SMOKE ${VARIANT_LABEL}: PASSED"
else
  echo "  SMOKE ${VARIANT_LABEL}: FAILED (exit=$EXIT_CODE)"
fi
echo "  Finished: $(date)"
echo "  Log: $LOGROOT/run.log"
echo "===================================================================="
exit $EXIT_CODE
