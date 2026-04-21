#!/bin/bash
#SBATCH --job-name=eval-causal-chain-v14-4gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# v14 CAUSAL-CHAIN rolling-context eval — 4 GPUs × 7 chained videos each.
# Each rank commits one gen chunk per video into a growing causal clean_x,
# producing one "final_causal_rollout" 24-frame mp4 per GPU plus per-video
# debug mp4s.
#
# Per-GPU config:
#   GPU 0  Madrid held-out    offset=100  mode=dataset         tag=madrid_held_dataset
#   GPU 1  Madrid held-out    offset=100  mode=counterfactual  tag=madrid_held_cf
#   GPU 2  Brighton (OOD)     offset=<auto turn-heavy>  mode=dataset  tag=brighton_turny
#   GPU 3  Tokyo (OOD)        offset=<auto turn-heavy>  mode=dataset  tag=tokyo_turny
#
# GPU 0 and GPU 1 run the *same* ride and *same* window (offset=100, matching
# the previous eval_chain run's gpu0 seg2) but with opposite noisy-action
# conditioning, for A/B comparison. GPU 2 and GPU 3 are OOD so we skip CF.
#
# Pre-flight: utils/find_turn_heavy_offsets.py scans sliding 21-latent windows
# on Brighton + Tokyo using the ss_vae and picks the offset with max sum|z7|.
#
# Checkpoint:  MODEL_ASSIGNMENTS[5] → logs/v14_balanced_weunz/causal_lora_step*.pt
# Manifest:    logs/v14_balanced_weunz/.ride_manifest.pt (Madrid/Rome/Stockholm)
# Disk fallback: /projects/u6ex/fbots/frodobots_encoded (superset with OOD cities)
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
BASE_OUTPUT="eval/eval_causal_chain_v14_${TS}"
CONFIG="configs/causal_lora_diffusion_teacher_v14.yaml"
MANIFEST="logs/v14_balanced_weunz/.ride_manifest.pt"
ENCODED_ROOT="/projects/u6ex/fbots/frodobots_encoded"
CAPTION_ROOT="/projects/u6ex/fbots/frodobots_captions/train"
MOTION_ROOT="/projects/u6ex/fbots/frodobots_motion"
SS_VAE_CKPT="action_query/checkpoints/ss_vae_8free.pt"

mkdir -p "$BASE_OUTPUT"

echo "Base output: $BASE_OUTPUT"
echo "Config:      $CONFIG"
echo "Manifest:    $MANIFEST"
echo "Starting v14 causal-chain eval on $(date)"
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)"
nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader

# ---- Auto-patch latest v14 ckpt into MODEL_ASSIGNMENTS[5] ----
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

# ---- Pre-flight: find turn-heavy offsets for Brighton + Tokyo ----
echo
echo "=== Pre-flight: turn-heavy window detection ==="
BRIGHTON_ZARR="20240408112440.zarr"
TOKYO_ZARR="20240209152725.zarr"
PREFLIGHT_LOG="$BASE_OUTPUT/turn_heavy_preflight.log"
PREFLIGHT_ENV="$BASE_OUTPUT/turn_heavy_offsets.env"

python -u -m utils.find_turn_heavy_offsets \
    --zarrs "$BRIGHTON_ZARR" "$TOKYO_ZARR" \
    --encoded_root "$ENCODED_ROOT" \
    --caption_root "$CAPTION_ROOT" \
    --motion_root "$MOTION_ROOT" \
    --ss_vae_checkpoint "$SS_VAE_CKPT" \
    --window 21 --stride 25 --min_offset 100 \
    --output_format env \
    --env_names BRIGHTON_OFFSET TOKYO_OFFSET \
    > "$PREFLIGHT_ENV" 2> "$PREFLIGHT_LOG"

echo "Pre-flight offsets:"
cat "$PREFLIGHT_ENV"
# shellcheck disable=SC1090
source "$PREFLIGHT_ENV"
: "${BRIGHTON_OFFSET:?pre-flight did not set BRIGHTON_OFFSET (see $PREFLIGHT_LOG)}"
: "${TOKYO_OFFSET:?pre-flight did not set TOKYO_OFFSET (see $PREFLIGHT_LOG)}"
echo "BRIGHTON_OFFSET=$BRIGHTON_OFFSET  TOKYO_OFFSET=$TOKYO_OFFSET"

# ---- Per-rank configuration ----
ZARR_GPU0="20240216101235.zarr"; OFF_GPU0=100;               MODE_GPU0="dataset";        TAG_GPU0="madrid_held_dataset"
ZARR_GPU1="20240216101235.zarr"; OFF_GPU1=100;               MODE_GPU1="counterfactual"; TAG_GPU1="madrid_held_cf"
ZARR_GPU2="$BRIGHTON_ZARR";      OFF_GPU2=$BRIGHTON_OFFSET;  MODE_GPU2="dataset";        TAG_GPU2="brighton_turny"
ZARR_GPU3="$TOKYO_ZARR";         OFF_GPU3=$TOKYO_OFFSET;     MODE_GPU3="dataset";        TAG_GPU3="tokyo_turny"

PIDS=()
for GPU in 0 1 2 3; do
    case $GPU in
        0) ZARR=$ZARR_GPU0; OFF=$OFF_GPU0; MODE=$MODE_GPU0; TAG=$TAG_GPU0 ;;
        1) ZARR=$ZARR_GPU1; OFF=$OFF_GPU1; MODE=$MODE_GPU1; TAG=$TAG_GPU1 ;;
        2) ZARR=$ZARR_GPU2; OFF=$OFF_GPU2; MODE=$MODE_GPU2; TAG=$TAG_GPU2 ;;
        3) ZARR=$ZARR_GPU3; OFF=$OFF_GPU3; MODE=$MODE_GPU3; TAG=$TAG_GPU3 ;;
    esac
    OUT="$BASE_OUTPUT/gpu${GPU}_${TAG}"
    mkdir -p "$OUT"
    echo "Launching GPU $GPU  zarr=$ZARR  offset=$OFF  mode=$MODE  tag=$TAG  out=$OUT"
    (
        CUDA_VISIBLE_DEVICES=$GPU \
        WORLD_SIZE=1 \
        LOCAL_RANK=0 \
        python utils/eval_causal_chain.py \
            --config "$CONFIG" \
            --output_dir "$OUT" \
            --manifest "$MANIFEST" \
            --assignment_index 5 \
            --rank_zarr "$ZARR" \
            --rank_offset "$OFF" \
            --rank_mode "$MODE" \
            --rank_tag "$TAG" \
            --encoded_root "$ENCODED_ROOT" \
            --caption_root "$CAPTION_ROOT" \
            --num_causal_videos 7 \
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
