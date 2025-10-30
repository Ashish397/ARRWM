#!/bin/bash
#SBATCH --job-name=latent-actions-stream-3dcnn-eval-ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=23:59:59
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e -o pipefail

cd /home/u5as/as1748.u5as/frodobots/ARRWM

# Activate arrwm conda environment
if [ -f "$HOME/miniforge/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$HOME/miniforge/etc/profile.d/conda.sh"
else
  echo 'conda.sh not found at $HOME/miniforge/etc/profile.d/conda.sh' >&2
  exit 1
fi
conda activate arrwm

# Ensure Hugging Face downloads use scratch space (larger quota than $HOME).
CACHE_DIR='/scratch/u5as/as1748.u5as/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR
export HF_HUB_CACHE=$CACHE_DIR
export HUGGINGFACE_HUB_CACHE=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
mkdir -p "$CACHE_DIR"

# Training roots (override via environment variables when submitting if needed).
ENCODED_ROOT="${ENCODED_ROOT:-/home/u5as/as1748.u5as/frodobots/projects_link/frodobots_encoded}"
ACTIONS_ROOT="${ACTIONS_ROOT:-/home/u5as/as1748.u5as/frodobots/projects_link/frodobots_actions}"
SPLIT="${SPLIT:-train}"
POOL_SIZE="${POOL_SIZE:-32}"
HIDDEN_DIMS="${HIDDEN_DIMS:-256 128}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
CHUNK_SIZE="${CHUNK_SIZE:-8}"
CHUNK_EPOCHS="${CHUNK_EPOCHS:-1}"
MAX_CHUNKS="${MAX_CHUNKS:-0}"
PATIENCE="${PATIENCE:-3}"
TARGET_MSE="${TARGET_MSE:-0.0}"
NUM_WORKERS="${NUM_WORKERS:-32}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
EVAL_CHUNK_SIZE="${EVAL_CHUNK_SIZE:-0}"
EVAL_RIDE_IDS="${EVAL_RIDE_IDS:-}"
WANDB_PROJECT="${WANDB_PROJECT:-frodobots_latent}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_GROUP="${WANDB_GROUP:-}"

# Directory for storing checkpoints and logs specific to this job.
CHECKPOINT_BASE="${CHECKPOINT_BASE:-/projects/u5as/frodobots_lam/latent_actions/checkpoints}"
CHECKPOINT_DIR="$CHECKPOINT_BASE/$SLURM_JOB_ID"
mkdir -p logs
mkdir -p "$CHECKPOINT_DIR"
export PYTHONUNBUFFERED=1

echo "Starting latent action training job on $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Encoded root: $ENCODED_ROOT"
echo "Actions root: $ACTIONS_ROOT"
echo "Target split: $SPLIT"
echo "Evaluation split: ${EVAL_SPLIT:-none}"
echo "Checkpoint directory: $CHECKPOINT_DIR"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

for ACTION_KIND in input output; do
  echo "----"
  echo "Training latent action model for ${ACTION_KIND} actions"
  echo "Checkpoint directory: $CHECKPOINT_DIR"
  EVAL_ARGS=()
  if [ -n "$EVAL_SPLIT" ]; then
    EVAL_ARGS+=(--eval-split "$EVAL_SPLIT")
    if [ "$EVAL_CHUNK_SIZE" -gt 0 ] 2>/dev/null; then
      EVAL_ARGS+=(--eval-chunk-size "$EVAL_CHUNK_SIZE")
    fi
    if [ -n "$EVAL_RIDE_IDS" ]; then
      # shellcheck disable=SC2206
      eval_ids=($EVAL_RIDE_IDS)
      EVAL_ARGS+=(--eval-ride-ids "${eval_ids[@]}")
    fi
  fi

  python latent_actions/train_latent_action_model.py \
    --encoded-root "$ENCODED_ROOT" \
    --actions-root "$ACTIONS_ROOT" \
    --split "$SPLIT" \
    --action-kind "$ACTION_KIND" \
    --pool-size "$POOL_SIZE" \
    --hidden-dims $HIDDEN_DIMS \
    --chunk-size "$CHUNK_SIZE" \
    --chunk-epochs "$CHUNK_EPOCHS" \
    --patience "$PATIENCE" \
    --max-chunks "$MAX_CHUNKS" \
    --num-workers "$NUM_WORKERS" \
    --lr "$LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --target-mse "$TARGET_MSE" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-entity "$WANDB_ENTITY" \
    --wandb-run-name "${WANDB_RUN_NAME:-latent_${ACTION_KIND}_$SLURM_JOB_ID}" \
    --wandb-group "$WANDB_GROUP" \
    --wandb-mode "$WANDB_MODE" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --device auto \
    "${EVAL_ARGS[@]}"
done

echo "Training complete on $(date)"
