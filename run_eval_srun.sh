#!/bin/bash
# Single-GPU eval_chain launcher, invoked via srun.
# Usage:  run_eval_srun.sh  RIDE_IDX  OUTPUT_DIR
set -euo pipefail

RIDE="${1:?need ride idx}"
OUT="${2:?need output dir}"

cd /scratch/u6ex/as1748.u6ex/ARRWM
# Use conda.sh init (does not consume $1/$2 — unlike sourcing bin/activate,
# which would swallow the script args and pass them to `activate`).
source /scratch/u6ex/as1748.u6ex/miniforge3/etc/profile.d/conda.sh
conda activate arrwm

export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache
export HF_HUB_CACHE=$HF_HOME
export HUGGINGFACE_HUB_CACHE=$HF_HOME
export TRANSFORMERS_CACHE=$HF_HOME
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$OUT"

echo "=== $(date) Node: $(hostname) RIDE=$RIDE OUT=$OUT ==="
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader

WORLD_SIZE=1 LOCAL_RANK=0 python utils/eval_chain.py \
  --config configs/causal_lora_diffusion_teacher_v13.yaml \
  --output_dir "$OUT" \
  --manifest logs/v13_balanced_weunz/.ride_manifest.pt \
  --test_ride_idx "$RIDE" \
  --assignment_index 4 \
  --num_segments 1 \
  --seed 42

echo "=== $(date) Completed ride=$RIDE ==="
ls -lh "$OUT"/v13_balanced_weunz/ 2>/dev/null || true
