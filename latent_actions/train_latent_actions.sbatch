#!/bin/bash
#SBATCH --job-name=latent-actions
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

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
POOL_SIZE="${POOL_SIZE:-4}"
ROUNDING="${ROUNDING:-5}"
HIDDEN_DIMS="${HIDDEN_DIMS:-256 128}"
MAX_EPOCHS="${MAX_EPOCHS:-2000}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
MIN_ACCURACY="${MIN_ACCURACY:-1.0}"
CONFIG_PATH="${CONFIG_PATH:-latent_actions/latent_training.yaml}"

# Directory for storing checkpoints and logs specific to this job.
CHECKPOINT_BASE="${CHECKPOINT_BASE:-latent_actions/checkpoints}"
CHECKPOINT_DIR="$CHECKPOINT_BASE/$SLURM_JOB_ID"
mkdir -p logs
mkdir -p "$CHECKPOINT_DIR"

echo "Starting latent action training job on $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Encoded root: $ENCODED_ROOT"
echo "Actions root: $ACTIONS_ROOT"
echo "Target split: $SPLIT"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Ensemble config: $CONFIG_PATH"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Detected GPUs: $(python - <<'PY'\nimport torch\nprint(torch.cuda.device_count())\nPY)"

for ACTION_KIND in input output; do
  TIMESTAMP=$(date +%Y%m%d-%H%M%S)
  CHECKPOINT_PATH="$CHECKPOINT_DIR/${ACTION_KIND}_actions_${TIMESTAMP}.pt"
  echo "----"
  echo "Training latent action model for ${ACTION_KIND} actions"
  echo "Checkpoint: $CHECKPOINT_PATH"
  python latent_actions/train_latent_action_model.py \
    --encoded-root "$ENCODED_ROOT" \
    --actions-root "$ACTIONS_ROOT" \
    --split "$SPLIT" \
    --action-kind "$ACTION_KIND" \
    --config "$CONFIG_PATH" \
    --pool-size "$POOL_SIZE" \
    --rounding "$ROUNDING" \
    --hidden-dims $HIDDEN_DIMS \
    --max-epochs "$MAX_EPOCHS" \
    --lr "$LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --min-accuracy "$MIN_ACCURACY" \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --device auto
done

echo "Training complete on $(date)"
