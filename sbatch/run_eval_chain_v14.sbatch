#!/bin/bash
#SBATCH --job-name=eval-chain-v14-4gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# v14 independent video evaluation — 4 GPUs, 3 rides per GPU,
# actual + counterfactual (noisy_mode=dataset_plus_cf) each.
#
# Per-GPU ride composition:
#   ride1 = training (in-distribution, seen during v14 training)
#   ride2 = held-out eval (same training-pool cities, NOT seen)
#   ride3 = OOD (city outside training distribution)
#
#   GPU 0: Madrid-train   + Madrid-held    + Wellington
#   GPU 1: Rome-train     + Rome-held      + Brighton
#   GPU 2: Stockholm-train+ Stockholm-held + Tokyo
#   GPU 3: Rome-train*    + Rome-held      + Vancouver
#   (*rank 3 re-uses Rome for the training slot per user request —
#    3 training-pool cities spread across 4 GPUs.)
#
# Totals: 12 distinct rides, 7 distinct cities (Madrid/Rome/Stockholm
# + Wellington/Brighton/Tokyo/Vancouver), 24 videos
# (6 per GPU = 3 rides * {normal, counterfactual}).
#
# Checkpoint: MODEL_ASSIGNMENTS[5] → logs/v14_balanced_weunz/causal_lora_step0006600.pt
# Manifest:   logs/v14_balanced_weunz/.ride_manifest.pt (WEU rides: Madrid/Rome/Stockholm)
# Disk-based fallback is used for OOD zarrs absent from the manifest
# (see utils/eval_chain.py :: _load_ride_entry_from_disk).
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
BASE_OUTPUT="eval/eval_chain_v14_${TS}"
CONFIG="configs/causal_lora_diffusion_teacher_v14.yaml"
MANIFEST="logs/v14_balanced_weunz/.ride_manifest.pt"
# Disk-fallback roots: encoded_root is the superset (contains both WEU and
# OOD cities), caption_root covers all rides we list below.
ENCODED_ROOT="/projects/u6ex/fbots/frodobots_encoded"
CAPTION_ROOT="/projects/u6ex/fbots/frodobots_captions/train"

mkdir -p "$BASE_OUTPUT"

echo "Base output: $BASE_OUTPUT"
echo "Config:      $CONFIG"
echo "Manifest:    $MANIFEST"
echo "Starting v14 eval on $(date)"
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)"
nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader

# Pick the latest v14 checkpoint and patch MODEL_ASSIGNMENTS[5].ckpt to match.
LATEST_CKPT=$(ls -t logs/v14_balanced_weunz/causal_lora_step*.pt | head -1)
echo "Latest v14 checkpoint: $LATEST_CKPT"
python3 - <<PYEOF
import re
path = 'utils/eval_chain.py'
with open(path) as f:
    s = f.read()
pat = r'("ckpt":\s*")logs/v14_balanced_weunz/causal_lora_step[0-9]+\.pt(",\s*\n\s*"label":\s*"v14_balanced_weunz")'
new_ckpt = '${LATEST_CKPT}'
s2, n = re.subn(pat, r'\g<1>' + new_ckpt + r'\g<2>', s)
if n != 1:
    raise SystemExit(f'Expected 1 patch, got {n}')
with open(path, 'w') as f:
    f.write(s2)
print(f'Patched eval_chain.py v14 ckpt -> {new_ckpt}')
PYEOF

# Per-rank zarr lists: (training, held-out eval, OOD).
#   GPU 0 — Madrid / Madrid / Wellington
#   GPU 1 — Rome   / Rome   / Brighton
#   GPU 2 — Stockholm / Stockholm / Tokyo
#   GPU 3 — Rome   / Rome   / Vancouver
RIDES_GPU0=(20240228112655.zarr 20240216101235.zarr 20240408220222.zarr)
RIDES_GPU1=(20240321110634.zarr 20240314084739.zarr 20240408112440.zarr)
RIDES_GPU2=(20240327115148.zarr 20240319133048.zarr 20240209152725.zarr)
RIDES_GPU3=(20240315164317.zarr 20240314090046.zarr 20240318183737.zarr)

TAG_GPU0="madrid_madrid_wellington"
TAG_GPU1="rome_rome_brighton"
TAG_GPU2="stockholm_stockholm_tokyo"
TAG_GPU3="rome_rome_vancouver"

PIDS=()
for GPU in 0 1 2 3; do
    case $GPU in
        0) RIDES=("${RIDES_GPU0[@]}"); TAG=$TAG_GPU0 ;;
        1) RIDES=("${RIDES_GPU1[@]}"); TAG=$TAG_GPU1 ;;
        2) RIDES=("${RIDES_GPU2[@]}"); TAG=$TAG_GPU2 ;;
        3) RIDES=("${RIDES_GPU3[@]}"); TAG=$TAG_GPU3 ;;
    esac
    OUT="$BASE_OUTPUT/gpu${GPU}_${TAG}"
    mkdir -p "$OUT"
    echo "Launching GPU $GPU  tag=$TAG  rides=${RIDES[*]}  out=$OUT"
    (
        CUDA_VISIBLE_DEVICES=$GPU \
        WORLD_SIZE=1 \
        LOCAL_RANK=0 \
        python utils/eval_chain.py \
            --config "$CONFIG" \
            --output_dir "$OUT" \
            --manifest "$MANIFEST" \
            --assignment_index 5 \
            --eval_zarrs "${RIDES[@]}" \
            --noisy_mode dataset_plus_cf \
            --encoded_root "$ENCODED_ROOT" \
            --caption_root "$CAPTION_ROOT" \
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
