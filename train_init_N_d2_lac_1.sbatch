#!/bin/bash
#SBATCH -J train_init_real_N_d2_lac_1_with_action
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH -t 05:00:00
#SBATCH --requeue
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

set -e -o pipefail

# Change to the ARRWM directory where train.py is located
cd /home/u5dk/as1748.u5dk/frodobots/ARRWM

# Project path and config
CONFIG=configs/longlive_train_init_real_N_d2_lac_1.yaml
LOGDIR=/scratch/u5dk/as1748.u5dk/frodobots/dmd2/logs_N_d2_lac_1_with_action
WANDB_SAVE_DIR=wandb
EXPERIMENT_NAME="dmd2-init-N-d2-lac-1-with-action-$(date +%Y%m%d-%H%M%S)"

# Activate arrwm conda environment
source /scratch/u5dk/as1748.u5dk/miniforge3/bin/activate
conda activate arrwm

# Ensure Hugging Face downloads use scratch space (larger quota than $HOME).
CACHE_DIR='/scratch/u5dk/as1748.u5dk/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR
export HF_HUB_CACHE=$CACHE_DIR
export HUGGINGFACE_HUB_CACHE=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
mkdir -p "$CACHE_DIR"

echo "CONFIG=$CONFIG"
echo "Starting training job on $(date)"
echo "Job ID: $SLURM_JOB_ID"

torchrun \
  --nproc_per_node=4 \
  train.py \
  --config_path $CONFIG \
  --logdir $LOGDIR \
  --wandb-save-dir $WANDB_SAVE_DIR \
  --experiment-name "$EXPERIMENT_NAME"

echo "Training completed on $(date)"
