#!/bin/bash
#SBATCH --job-name=eval-causal-4gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=/scratch/u6ej/as1748.u6ej/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ej/as1748.u6ej/ARRWM/logs/%x_%j.err

# v12 only; 6 causal videos per GPU; each GPU uses a different manifest zarr.
# Wall time ~6× single eval_chain segment per GPU (budget 2h).

set -e -o pipefail
cd /scratch/u6ej/as1748.u6ej/ARRWM

source /scratch/u6ej/as1748.u6ej/miniforge3/bin/activate
conda activate arrwm

export HF_HOME=/scratch/u6ej/as1748.u6ej/frodobots/hf_cache
export HF_HUB_CACHE=$HF_HOME
export HUGGINGFACE_HUB_CACHE=$HF_HOME
export TRANSFORMERS_CACHE=$HF_HOME
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT="eval/eval_causal_chain_out_$(date +%Y%m%d_%H%M%S)"
MANIFEST="logs/z_critic_v10_state_tokens/.ride_manifest.pt"

echo "Output: $OUT"
nvidia-smi

torchrun \
  --nproc_per_node=4 \
  --master_port=$((29600 + SLURM_JOB_ID % 16000)) \
  utils/eval_causal_chain.py \
  --output_dir "$OUT" \
  --manifest "$MANIFEST" \
  --seed 42

echo "Done: $OUT"
ls -lhR "$OUT"
