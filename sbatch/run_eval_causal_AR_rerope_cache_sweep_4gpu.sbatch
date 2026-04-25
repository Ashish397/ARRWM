#!/bin/bash
#SBATCH --job-name=eval-causal-AR-rerope-cache-sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# AR-rerope cache-size sweep: same configuration as GPU 1 of
# ``run_eval_causal_AR_rerope_4gpu.sbatch`` (mode=ar, cache_refresh=
# append, --infinity_rope) replicated across 4 GPUs with progressively
# LARGER KV cache budgets, to visualise how AR rollout quality
# degrades when the cache rolls more aggressively.
#
# ``--cache_chunks`` is in units of 3-frame chunks (BASE_CHUNK_FRAMES=
# 3 in utils/eval_causal_AR.py), so the requested cache sizes in frames
# map as:
#
#   GPU 0   --cache_chunks 3   →   9 frames KV cache  (smallest, rolls 5×)
#   GPU 1   --cache_chunks 4   →  12 frames KV cache  (rolls 4×)
#   GPU 2   --cache_chunks 5   →  15 frames KV cache  (rolls 3×)
#   GPU 3   --cache_chunks 6   →  18 frames KV cache  (largest, rolls 2×)
#
# With AR_INITIAL_CHUNKS=1 (1 real seed chunk) + AR_GEN_CHUNKS=7
# (7 generated chunks) = 8 committed chunks total = 24 frames committed,
# every GPU here rolls its KV cache (the previous baseline run with
# cache_chunks=12 / 36 frames never rolled because 12 > 8).
#
# All four runs use the LongLive-style Block-Relativistic (Infinity)
# RoPE patch from utils/infinity_rope.py via --infinity_rope, so
# rotations remain bounded across rolls. Differences between videos
# isolate the effect of cache size on AR generation quality.
#
# Same ride / offset / seed / counterfactual actions as
# run_eval_causal_AR_rerope_4gpu.sbatch so videos can be diffed
# directly against gpu1_ar_append_rerope/ from that run.
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
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

TS=$(date +%Y%m%d_%H%M%S)
BASE_OUTPUT="eval/eval_AR_rerope_cache_sweep_${TS}"
CONFIG="configs/action_ode_distill.yaml"
MANIFEST="logs/v14_balanced_weunz/.ride_manifest.pt"
ENCODED_ROOT="/projects/u6ex/fbots/frodobots_encoded"
CAPTION_ROOT="/projects/u6ex/fbots/frodobots_captions/train"
MOTION_ROOT="/projects/u6ex/fbots/frodobots_motion"
SS_VAE_CKPT="action_query/checkpoints/ss_vae_8free.pt"

RIDE_ZARR="20240216101235.zarr"
RIDE_OFFSET=100

STUDENT_CKPT="${STUDENT_CKPT:-/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_ode_distill_E/action_ode_step0001000.pt}"
if [ ! -f "$STUDENT_CKPT" ]; then
    echo "ERROR: STUDENT_CKPT does not exist: $STUDENT_CKPT" >&2
    echo "  Override via: STUDENT_CKPT=/path/to/checkpoint.pt sbatch ..." >&2
    exit 1
fi

DENOISING_STEPS="${DENOISING_STEPS:-4}"
SEED="${SEED:-42}"
AR_INITIAL_CHUNKS="${AR_INITIAL_CHUNKS:-1}"
AR_GEN_CHUNKS="${AR_GEN_CHUNKS:-7}"

# Per-GPU cache_chunks in 3-frame units (frames = 3 * cache_chunks).
GPU0_CACHE_CHUNKS=3   # 9 frames
GPU1_CACHE_CHUNKS=4   # 12 frames
GPU2_CACHE_CHUNKS=5   # 15 frames
GPU3_CACHE_CHUNKS=6   # 18 frames

mkdir -p "$BASE_OUTPUT"

echo "Base output:      $BASE_OUTPUT"
echo "Config:           $CONFIG"
echo "Manifest:         $MANIFEST"
echo "Student ckpt:     $STUDENT_CKPT"
echo "Ride:             $RIDE_ZARR  offset=$RIDE_OFFSET"
echo "Denoising steps:  $DENOISING_STEPS"
echo "AR init/gen:      $AR_INITIAL_CHUNKS / $AR_GEN_CHUNKS  (= $((AR_INITIAL_CHUNKS + AR_GEN_CHUNKS)) committed chunks)"
echo "Cache sweep:      GPU0=${GPU0_CACHE_CHUNKS}c/9f  GPU1=${GPU1_CACHE_CHUNKS}c/12f  GPU2=${GPU2_CACHE_CHUNKS}c/15f  GPU3=${GPU3_CACHE_CHUNKS}c/18f"
echo "Seed:             $SEED"
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)"
nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader

PIDS=()
for GPU in 0 1 2 3; do
    case $GPU in
        0) CC=$GPU0_CACHE_CHUNKS ;;
        1) CC=$GPU1_CACHE_CHUNKS ;;
        2) CC=$GPU2_CACHE_CHUNKS ;;
        3) CC=$GPU3_CACHE_CHUNKS ;;
    esac
    FRAMES=$((CC * 3))
    TAG="ar_rerope_cache${FRAMES}f"
    OUT="$BASE_OUTPUT/gpu${GPU}_${TAG}"
    mkdir -p "$OUT"
    echo "Launching GPU $GPU  cache_chunks=$CC (${FRAMES} frames)  tag=$TAG  out=$OUT"

    (
        CUDA_VISIBLE_DEVICES=$GPU \
        WORLD_SIZE=1 \
        LOCAL_RANK=0 \
        python utils/eval_causal_AR.py \
            --config "$CONFIG" \
            --output_dir "$OUT" \
            --manifest "$MANIFEST" \
            --student_ckpt "$STUDENT_CKPT" \
            --mode ar \
            --rank_zarr "$RIDE_ZARR" \
            --rank_offset "$RIDE_OFFSET" \
            --rank_mode counterfactual \
            --rank_tag "$TAG" \
            --encoded_root "$ENCODED_ROOT" \
            --caption_root "$CAPTION_ROOT" \
            --motion_root "$MOTION_ROOT" \
            --ss_vae_checkpoint "$SS_VAE_CKPT" \
            --num_causal_videos 7 \
            --denoising_steps "$DENOISING_STEPS" \
            --cache_chunks "$CC" \
            --chunks_per_step 1 \
            --ar_initial_chunks "$AR_INITIAL_CHUNKS" \
            --ar_gen_chunks "$AR_GEN_CHUNKS" \
            --ar_cache \
            --ar_cache_refresh append \
            --infinity_rope \
            --seed "$SEED" \
            > "$OUT/eval.log" 2>&1
    ) &
    PIDS+=($!)
done

echo "Launched ${#PIDS[@]} processes: ${PIDS[*]}"
FAIL=0
for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        echo "Process $PID FAILED (check per-GPU eval.log)"
        FAIL=1
    fi
done

echo "All processes finished on $(date)"
echo
echo "=== Output tree ==="
ls -lhR "$BASE_OUTPUT"
echo
echo "=== Per-GPU summary (cache sweep) ==="
for GPU in 0 1 2 3; do
    LOG=$(ls "$BASE_OUTPUT/gpu${GPU}_"*/eval.log 2>/dev/null | head -1)
    if [ -n "$LOG" ]; then
        echo "--- GPU $GPU ($(basename $(dirname $LOG))) ---"
        grep -E "rollout done|streaming rollout done|infinity_rope=|cache_chunks=" "$LOG" | tail -5 || true
    fi
done

if [[ $FAIL -ne 0 ]]; then
    echo "WARN: one or more processes failed (check per-GPU eval.log)"
    exit 1
fi
