#!/bin/bash
#SBATCH --job-name=motion-habitat
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/u6ej/as1748.u6ej/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ej/as1748.u6ej/ARRWM/logs/%x_%j.err

set -e -o pipefail

cd /scratch/u6ej/as1748.u6ej/ARRWM

source /scratch/u6ej/as1748.u6ej/miniforge3/bin/activate
conda activate arrwm

CACHE_DIR='/scratch/u6ej/as1748.u6ej/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR
export HF_HUB_CACHE=$CACHE_DIR
export HUGGINGFACE_HUB_CACHE=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
mkdir -p "$CACHE_DIR"
mkdir -p logs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

CONFIG=configs/causal_lora_diffusion_teacher.yaml
CHECKPOINT=logs/causal_lora_teacher/causal_lora_step0003150.pt
MANIFEST=logs/causal_lora_teacher/.ride_manifest.pt
OUT_DIR=testing/motion_habitat_out
NUM_VIDEOS=16
NUM_STEPS=50
VIDS_PER_GPU=$((NUM_VIDEOS / 4))

echo "Motion Habitat — raw teacher z across denoising steps"
echo "Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Checkpoint: $CHECKPOINT"
echo "Videos: $NUM_VIDEOS (${VIDS_PER_GPU}/GPU) | Steps: $NUM_STEPS"
echo "Output: $OUT_DIR"

for GPU_ID in 0 1 2 3; do
    OFFSET=$((GPU_ID * VIDS_PER_GPU))

    python testing/motion_habitat.py \
        --config $CONFIG \
        --checkpoint $CHECKPOINT \
        --manifest_cache $MANIFEST \
        --num_videos $VIDS_PER_GPU \
        --num_steps $NUM_STEPS \
        --device "cuda:${GPU_ID}" \
        --output_dir "$OUT_DIR" \
        --video_offset $OFFSET &

    echo "  GPU $GPU_ID: videos $OFFSET..$(( OFFSET + VIDS_PER_GPU - 1 ))"
done

echo "Waiting for all GPUs ..."
wait

echo "Done: $(date)"
echo "Charts: $(find $OUT_DIR -name '*.png' | wc -l) PNG files in $OUT_DIR"
