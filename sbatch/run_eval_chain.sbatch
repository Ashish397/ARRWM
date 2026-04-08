#!/bin/bash
#SBATCH --job-name=eval-chain-4gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=/scratch/u6ej/as1748.u6ej/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ej/as1748.u6ej/ARRWM/logs/%x_%j.err

# =====================================================================
# Independent video evaluation: 4 models × 4 segments (fixed z2 per seg + z7 block pattern)
#
# GPU 0 → v12 (probe fixes)           step 3250
# GPU 1 → v8  (aux critic)          step 2900
# GPU 2 → v5  (reward_critic_v5b)   step 1400
# GPU 3 → v10 (z_critic_v10_state_tokens) step 1650
#
# Per-segment noisy (z2,z7): segment_noisy_frame_actions.json. Each segment uses a different zarr ride
# (segment_clean_rides.json). clean_x = 21 latents @ offset 100 per ride; MP4 prepends 3 latents from that ride.
# =====================================================================

set -e -o pipefail

cd /scratch/u6ej/as1748.u6ej/ARRWM

source /scratch/u6ej/as1748.u6ej/miniforge3/bin/activate
conda activate arrwm

CACHE_DIR='/scratch/u6ej/as1748.u6ej/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR
export HF_HUB_CACHE=$CACHE_DIR
export HUGGINGFACE_HUB_CACHE=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUTPUT_DIR="eval/eval_chain_out_$(date +%Y%m%d_%H%M%S)"
MANIFEST="logs/z_critic_v10_state_tokens/.ride_manifest.pt"

echo "Output: $OUTPUT_DIR"
echo "Manifest: $MANIFEST"
echo "Starting eval on $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
nvidia-smi

torchrun \
  --nproc_per_node=4 \
  --master_port=$((29500 + SLURM_JOB_ID % 16000)) \
  utils/eval_chain.py \
  --output_dir "$OUTPUT_DIR" \
  --manifest "$MANIFEST" \
  --test_ride_idx 0 \
  --seed 42

echo "Evaluation completed on $(date)"
echo "Results in: $OUTPUT_DIR"
ls -lhR "$OUTPUT_DIR"
