#!/bin/bash
#SBATCH --job-name=wan-bidir-visualise
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=/home/u5as/as1748.u5as/frodobots/logs/%x_%j.out
#SBATCH --error=/home/u5as/as1748.u5as/frodobots/logs/%x_%j.err

set -e -o pipefail

LOG_DIR=/home/u5as/as1748.u5as/frodobots/logs
mkdir -p "$LOG_DIR"

cd /home/u5as/as1748.u5as/frodobots

CONFIG=ARRWM/configs/bidirectional_inference.yaml
SCRIPT=ARRWM/utils/visualise_retrain.py
INPUT_DIR=/projects/u5as/frodobots_captions/output_rides_3/test
CHECKPOINT_STEPS=(5000 4600 4200 3800 3400 3000 2600 2200 1800 1400 1000 600)

CACHE_DIR='/scratch/u5as/as1748.u5as/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR
export HF_HUB_CACHE=$CACHE_DIR
export HUGGINGFACE_HUB_CACHE=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
mkdir -p "$CACHE_DIR"

mkdir -p logs

echo "CONFIG=$CONFIG"
echo "INPUT_DIR=$INPUT_DIR"
echo "Checkpoint steps: ${CHECKPOINT_STEPS[*]}"
echo "Starting visualisation job on $(date)"
echo "Job ID: $SLURM_JOB_ID"

echo "Total checkpoints: ${#CHECKPOINT_STEPS[@]}"

CHECKPOINT_LIST="${CHECKPOINT_STEPS[*]}"

srun --ntasks=${#CHECKPOINT_STEPS[@]} --gres=gpu:1 --cpus-per-task=4 --nodes=3 --distribution=block:block --exclusive \
  bash -lc "set -e; source \"$HOME/miniforge/etc/profile.d/conda.sh\"; conda activate arrwm; STEPS=($CHECKPOINT_LIST); STEP=\${STEPS[\$SLURM_PROCID]}; echo '[Rank' \$SLURM_PROCID '] Running checkpoint' \$STEP; python $SCRIPT --config-path $CONFIG --checkpoint-step \$STEP --input-dir $INPUT_DIR --skip-existing"

echo "Completed all visualisations at $(date)"
