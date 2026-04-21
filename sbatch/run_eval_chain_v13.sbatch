#!/bin/bash
#SBATCH --job-name=eval-chain-v13-4gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# v13 independent video evaluation
# Launches 4 parallel single-GPU eval_chain processes, each running v13
# (MODEL_ASSIGNMENTS index 4) on a DIFFERENT ride for dataset variety.
#
# GPU 0 -> test_ride_idx 0    (near manifest start)
# GPU 1 -> test_ride_idx 210  (~25% through)
# GPU 2 -> test_ride_idx 425  (~50%)
# GPU 3 -> test_ride_idx 640  (~75%)
#
# Manifest: logs/v13_balanced_weunz/.ride_manifest.pt (851 WEU rides: Madrid/Rome/Stockholm)
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm

CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR
export HF_HUB_CACHE=$CACHE_DIR
export HUGGINGFACE_HUB_CACHE=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TS=$(date +%Y%m%d_%H%M%S)
BASE_OUTPUT="eval/eval_chain_v13_${TS}"
MANIFEST="logs/v13_balanced_weunz/.ride_manifest.pt"

mkdir -p "$BASE_OUTPUT"

echo "Base output: $BASE_OUTPUT"
echo "Manifest: $MANIFEST"
echo "Starting v13 eval on $(date)"
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)"
nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader

# Pick the latest v13 checkpoint and patch MODEL_ASSIGNMENTS[4].ckpt to match
LATEST_CKPT=$(ls -t logs/v13_balanced_weunz/causal_lora_step*.pt | head -1)
echo "Latest v13 checkpoint: $LATEST_CKPT"
python3 - <<PYEOF
import re
path = 'utils/eval_chain.py'
with open(path) as f:
    s = f.read()
pat = r'("ckpt":\s*")logs/v13_balanced_weunz/causal_lora_step[0-9]+\.pt(",\s*\n\s*"label":\s*"v13_balanced_weunz")'
new_ckpt = '${LATEST_CKPT}'
s2, n = re.subn(pat, r'\g<1>' + new_ckpt + r'\g<2>', s)
if n != 1:
    raise SystemExit(f'Expected 1 patch, got {n}')
with open(path, 'w') as f:
    f.write(s2)
print(f'Patched eval_chain.py v13 ckpt -> {new_ckpt}')
PYEOF

RIDE_IDXS=(0 210 425 640)

PIDS=()
for GPU in 0 1 2 3; do
    RIDE=${RIDE_IDXS[$GPU]}
    OUT="$BASE_OUTPUT/gpu${GPU}_ride${RIDE}"
    mkdir -p "$OUT"
    echo "Launching GPU $GPU  ride=$RIDE  out=$OUT"
    (
        CUDA_VISIBLE_DEVICES=$GPU \
        WORLD_SIZE=1 \
        LOCAL_RANK=0 \
        python utils/eval_chain.py \
            --config configs/causal_lora_diffusion_teacher_v13.yaml \
            --output_dir "$OUT" \
            --manifest "$MANIFEST" \
            --test_ride_idx "$RIDE" \
            --assignment_index 4 \
            --num_segments 1 \
            --seed 42 \
            >"$OUT/eval.log" 2>&1
    ) &
    PIDS+=($!)
done

echo "Launched ${#PIDS[@]} processes: ${PIDS[*]}"
FAIL=0
for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        echo "Process $PID FAILED"
        FAIL=1
    fi
done

echo "All processes finished on $(date)"
echo
echo "=== Output tree ==="
ls -lhR "$BASE_OUTPUT"

if [[ $FAIL -ne 0 ]]; then
    echo "WARN: one or more processes failed (check per-GPU eval.log)"
    exit 1
fi
