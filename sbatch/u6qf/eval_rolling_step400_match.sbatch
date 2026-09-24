#!/bin/bash
# Reusable step-400 rolling reference: m42 x F/L/B, 30 s, common E1 seed.
# Pass COMPARE_CKPT and COMPARE_TAG via sbatch --export. The same protocol can
# be repeated for v3b after its own rolling step-400 checkpoint is written.
#SBATCH --job-name=roll400-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/u6qf/as1748.u6qf/ARRWM_straggler/_logs/ours30s/roll400_eval_%j.out
#SBATCH --error=/scratch/u6qf/as1748.u6qf/ARRWM_straggler/_logs/ours30s/roll400_eval_%j.err

set -euo pipefail
: "${COMPARE_CKPT:?pass the exact rolling step-400 full checkpoint}"
: "${COMPARE_TAG:?pass a unique output tag}"

R=/scratch/u6qf/as1748.u6qf/ARRWM_straggler
A=$R/ARRWM
OUT=$R/_logs/ours30s/out/$COMPARE_TAG
export PATH=$R/miniforge3/envs/arrwm/bin:$PATH
export PYTHONPATH=$A:$A/action-forcing
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=$R/hf_cache
export TORCH_HOME=$R/torch_home
cd "$A"

if [ "${COMPARE_WAIT_FOR_CKPT_S:-0}" -gt 0 ]; then
  deadline=$((SECONDS + COMPARE_WAIT_FOR_CKPT_S))
  while [ ! -s "$COMPARE_CKPT" ]; do
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo "Timed out waiting for $COMPARE_CKPT" >&2
      exit 1
    fi
    sleep 30
  done
  # A full checkpoint is written directly to its final filename. Wait for its
  # size to settle before attempting to load it while training is still live.
  size_before=$(stat -c %s "$COMPARE_CKPT")
  sleep 30
  size_after=$(stat -c %s "$COMPARE_CKPT")
  if [ "$size_before" -ne "$size_after" ]; then
    echo "Checkpoint is still being written: $COMPARE_CKPT" >&2
    exit 1
  fi
fi
test -s "$COMPARE_CKPT"
python interactive/engine.py --save_lean_ckpt --ckpt "$COMPARE_CKPT"
test -s "$COMPARE_CKPT.lean.pt"
mkdir -p "$OUT"
python interactive/external_models/ours_fleet_30s.py \
  --ckpt "$COMPARE_CKPT.lean.pt" --tag "$COMPARE_TAG" --out "$OUT" \
  --bundle experiments/e1/seed_bundle_e1 \
  --wan_model_path "$R/frodobots/Wan2.1-T2V-1.3B/" \
  --windows m42 --dirs F,L,B --denoising_steps 4 --seed_prefill_chunks 3

count=$(find "$OUT" -maxdepth 1 -type f -name "${COMPARE_TAG}_*.mp4.json" | wc -l)
echo "$COMPARE_TAG: $count/3 clips complete"
test "$count" -eq 3
