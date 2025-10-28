#!/bin/bash
#SBATCH --job-name=wan-bidir-visualise
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=/home/u5as/as1748.u5as/frodobots/logs/%x_%j.out
#SBATCH --error=/home/u5as/as1748.u5as/frodobots/logs/%x_%j.err

set -e -o pipefail

LOG_DIR=/home/u5as/as1748.u5as/frodobots/logs
mkdir -p "$LOG_DIR"

cd /home/u5as/as1748.u5as/frodobots

CHECKPOINT_STEPS=(7200 6000 5100 4200 3000 2000 900 300)
LOG_ROOT_NAMES=(retrain_no_gan_256 retrain_no_gan_128 retrain_no_gan_64 retrain_128 retrain_64 retrain_32)

CONFIG=ARRWM/configs/bidirectional_inference.yaml
SCRIPT=ARRWM/utils/visualise_retrain.py
INPUT_DIR=/projects/u5as/frodobots_captions/output_rides_3/test

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
echo "Log root names: ${LOG_ROOT_NAMES[*]}"
echo "Starting visualisation job on $(date)"
echo "Job ID: $SLURM_JOB_ID"

NUM_CHECKPOINTS=${#CHECKPOINT_STEPS[@]}
NUM_LOG_ROOTS=${#LOG_ROOT_NAMES[@]}
TOTAL_TASKS=$((NUM_CHECKPOINTS * NUM_LOG_ROOTS))

echo "Total tasks: $TOTAL_TASKS (${NUM_CHECKPOINTS} checkpoints × ${NUM_LOG_ROOTS} log roots)"

CHECKPOINT_LIST="${CHECKPOINT_STEPS[*]}"
LOG_ROOT_LIST="${LOG_ROOT_NAMES[*]}"

srun --ntasks=$TOTAL_TASKS --gres=gpu:1 --cpus-per-task=4 --nodes=12 --distribution=block:block --exclusive \
  bash -lc "set -e; source \"$HOME/miniforge/etc/profile.d/conda.sh\"; conda activate arrwm; STEPS=($CHECKPOINT_LIST); LOG_ROOTS=($LOG_ROOT_LIST); LOG_ROOT_IDX=\$((\$SLURM_PROCID / ${NUM_CHECKPOINTS})); STEP_IDX=\$((\$SLURM_PROCID % ${NUM_CHECKPOINTS})); LOG_ROOT_NAME=\${LOG_ROOTS[\$LOG_ROOT_IDX]}; STEP=\${STEPS[\$STEP_IDX]}; LOG_ROOT=/scratch/u5as/as1748.u5as/frodobots/\$LOG_ROOT_NAME/logs; OUTPUT_DIR=/scratch/u5as/as1748.u5as/frodobots/retrain_visualisations/\$LOG_ROOT_NAME; echo '[Rank' \$SLURM_PROCID '] Log root:' \$LOG_ROOT_NAME 'Checkpoint:' \$STEP; python $SCRIPT --config-path $CONFIG --checkpoint-step \$STEP --input-dir $INPUT_DIR --skip-existing --logs-root \$LOG_ROOT --output-dir \$OUTPUT_DIR"

echo "Completed all visualisations at $(date)"
