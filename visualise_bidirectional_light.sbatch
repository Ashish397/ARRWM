#!/bin/bash
#SBATCH --job-name=wan-bidir-visualise
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=01:30:00
#SBATCH --output=/home/u5as/as1748.u5as/frodobots/logs/%x_%j.out
#SBATCH --error=/home/u5as/as1748.u5as/frodobots/logs/%x_%j.err

set -e -o pipefail

LOG_DIR=/home/u5as/as1748.u5as/frodobots/logs
mkdir -p "$LOG_DIR"

cd /home/u5as/as1748.u5as/frodobots

LOG_ROOT=/scratch/u5as/as1748.u5as/frodobots/retrain_128/logs
CONFIG=ARRWM/configs/bidirectional_inference.yaml
SCRIPT=ARRWM/utils/visualise_retrain.py
INPUT_DIR=/projects/u5as/frodobots_captions/output_rides_3/test
CHECKPOINT_STEPS=(6900 5700 5100 3900)
# CHECKPOINT_STEPS=(2700 1500 900 300)
OUTPUT_DIR=/scratch/u5as/as1748.u5as/frodobots/retrain_visualisations/128G

CACHE_DIR='/scratch/u5as/as1748.u5as/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR
export HF_HUB_CACHE=$CACHE_DIR
export HUGGINGFACE_HUB_CACHE=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
mkdir -p "$CACHE_DIR"

mkdir -p logs

echo "CONFIG=$CONFIG"
echo "INPUT_DIR=$INPUT_DIR"
echo "LOG_ROOT=$LOG_ROOT"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "Checkpoint steps: ${CHECKPOINT_STEPS[*]}"
echo "Starting visualisation job on $(date)"
echo "Job ID: $SLURM_JOB_ID"

echo "Total checkpoints: $((${#CHECKPOINT_STEPS[@]}))"

# Validate exactly 4 steps (one per GPU/task)
if [ "${#CHECKPOINT_STEPS[@]}" -ne 4 ]; then
  echo "ERROR: Provide exactly 4 checkpoint steps in CHECKPOINT_STEPS (currently ${#CHECKPOINT_STEPS[@]})."
  exit 1
fi

# Create combined arrays for srun
ALL_STEPS=("${CHECKPOINT_STEPS[@]}")

srun --ntasks=4 --gres=gpu:1 --cpus-per-task=4 --nodes=1 --distribution=block:block --exclusive \
  bash -lc "set -e; source \"$HOME/miniforge/etc/profile.d/conda.sh\"; conda activate arrwm; STEPS=(${ALL_STEPS[*]}); STEP=\${STEPS[\$SLURM_PROCID]}; echo '[Rank' \$SLURM_PROCID '] Running checkpoint' \$STEP 'with log root' $LOG_ROOT; python $SCRIPT --config-path $CONFIG --checkpoint-step \$STEP --input-dir $INPUT_DIR --output-dir $OUTPUT_DIR --skip-existing --logs-root $LOG_ROOT"

echo "Completed all visualisations at $(date)"
