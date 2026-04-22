#!/bin/bash
#SBATCH --job-name=eval-causal-AR-4gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# ODE-distilled student, 4-GPU eval on the SAME ride used by
# run_eval_causal_chain_v14.sbatch (Madrid held-out
# 20240216101235.zarr @ offset=100) so AR vs chain can be compared
# head-to-head on identical ground truth.
#
#   GPU 0  AR mode      rank_mode=counterfactual
#   GPU 1  AR mode      rank_mode=dataset
#   GPU 2  chain mode   rank_mode=dataset
#   GPU 3  chain mode   rank_mode=counterfactual
#
# All 4 GPUs:
#   * same zarr + offset (Madrid held-out, matches v14-chain gpu0/1)
#   * same student checkpoint (ODE distill 10h, latest_*.pt picked
#     automatically from logs/action_ode_distill_10h/)
#   * same denoising budget (4 steps, the student's training pool)
#
# AR-specific knobs. Chain-equivalent seed + AR rollout:
#   --cache_chunks 12        max that fits at training-mode memory
#   --chunks_per_step 1      matches training num_frame_per_block=3
#   --ar_initial_chunks 1    exactly one real chunk (3 frames) of seed,
#                            matching eval_causal_chain's CONTEXT_FRAMES
#                            seed_lat. generate_ar always pushes this
#                            block through cache-refresh so the KV cache
#                            starts with the SAME 3 real frames chain
#                            uses — then AR-generates on top.
#   --ar_gen_chunks 7        main AR rollout (21 frames = chain's NUM_FRAMES)
#   --ar_cache               ON (default) — after the real seed, the cache
#                            is populated by the model's own generated
#                            outputs (realistic deployment). With
#                            ar_initial_chunks=1 no bootstrap chunks are
#                            generated; we go straight from real seed
#                            to main AR rollout. Total output = 3 real
#                            seed + 21 AR-gen = 24 frames (matches chain).
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
BASE_OUTPUT="eval/eval_causal_AR_${TS}"
CONFIG="configs/action_ode_distill.yaml"
MANIFEST="logs/v14_balanced_weunz/.ride_manifest.pt"
ENCODED_ROOT="/projects/u6ex/fbots/frodobots_encoded"
CAPTION_ROOT="/projects/u6ex/fbots/frodobots_captions/train"
MOTION_ROOT="/projects/u6ex/fbots/frodobots_motion"
SS_VAE_CKPT="action_query/checkpoints/ss_vae_8free.pt"

# Same ride + offset as run_eval_causal_chain_v14.sbatch (gpu0/gpu1)
# — Madrid held-out, makes AR vs chain directly comparable.
RIDE_ZARR="20240216101235.zarr"
RIDE_OFFSET=100

# Pin the ODE student checkpoint explicitly. The 1000-step snapshot
# the user wants lives in logs/action_ode_distill_E/ (NOT
# logs/action_ode_distill_10h/, which only ever ran to step 300).
# Override with STUDENT_CKPT=... sbatch ... if pointing to a different
# step; the script will fail-fast below if the file doesn't exist.
STUDENT_CKPT="${STUDENT_CKPT:-/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_ode_distill_E/action_ode_step0001000.pt}"
if [ ! -f "$STUDENT_CKPT" ]; then
    echo "ERROR: STUDENT_CKPT does not exist: $STUDENT_CKPT" >&2
    echo "  Override via: STUDENT_CKPT=/path/to/checkpoint.pt sbatch ..." >&2
    exit 1
fi

DENOISING_STEPS="${DENOISING_STEPS:-4}"
SEED="${SEED:-42}"

mkdir -p "$BASE_OUTPUT"

echo "Base output:      $BASE_OUTPUT"
echo "Config:           $CONFIG"
echo "Manifest:         $MANIFEST"
echo "Student ckpt:     $STUDENT_CKPT"
echo "Ride:             $RIDE_ZARR  offset=$RIDE_OFFSET"
echo "Denoising steps:  $DENOISING_STEPS"
echo "Seed:             $SEED"
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)"
nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader

# ---- Per-GPU configuration ----
# Each row:   gpu_idx  eval_mode  rank_mode           tag
GPU0_MODE="ar";    GPU0_RM="counterfactual"; GPU0_TAG="ar_cf"
GPU1_MODE="ar";    GPU1_RM="dataset";        GPU1_TAG="ar_ds"
GPU2_MODE="chain"; GPU2_RM="dataset";        GPU2_TAG="chain_ds"
GPU3_MODE="chain"; GPU3_RM="counterfactual"; GPU3_TAG="chain_cf"

PIDS=()
for GPU in 0 1 2 3; do
    case $GPU in
        0) MODE=$GPU0_MODE; RM=$GPU0_RM; TAG=$GPU0_TAG ;;
        1) MODE=$GPU1_MODE; RM=$GPU1_RM; TAG=$GPU1_TAG ;;
        2) MODE=$GPU2_MODE; RM=$GPU2_RM; TAG=$GPU2_TAG ;;
        3) MODE=$GPU3_MODE; RM=$GPU3_RM; TAG=$GPU3_TAG ;;
    esac
    OUT="$BASE_OUTPUT/gpu${GPU}_${TAG}"
    mkdir -p "$OUT"
    echo "Launching GPU $GPU  eval_mode=$MODE  rank_mode=$RM  tag=$TAG  out=$OUT"

    # AR-mode extras; chain mode ignores these flags (eval_causal_AR.py
    # branches on --mode so the AR knobs are inert when --mode chain).
    AR_ARGS=()
    if [ "$MODE" = "ar" ]; then
        AR_ARGS=(
            --cache_chunks 12
            --chunks_per_step 1
            --ar_initial_chunks 1
            --ar_gen_chunks 7
            --ar_cache
        )
    fi

    (
        CUDA_VISIBLE_DEVICES=$GPU \
        WORLD_SIZE=1 \
        LOCAL_RANK=0 \
        python utils/eval_causal_AR.py \
            --config "$CONFIG" \
            --output_dir "$OUT" \
            --manifest "$MANIFEST" \
            --student_ckpt "$STUDENT_CKPT" \
            --mode "$MODE" \
            --rank_zarr "$RIDE_ZARR" \
            --rank_offset "$RIDE_OFFSET" \
            --rank_mode "$RM" \
            --rank_tag "$TAG" \
            --encoded_root "$ENCODED_ROOT" \
            --caption_root "$CAPTION_ROOT" \
            --motion_root "$MOTION_ROOT" \
            --ss_vae_checkpoint "$SS_VAE_CKPT" \
            --num_causal_videos 7 \
            --denoising_steps "$DENOISING_STEPS" \
            --seed "$SEED" \
            "${AR_ARGS[@]}" \
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

if [[ $FAIL -ne 0 ]]; then
    echo "WARN: one or more processes failed (check per-GPU eval.log)"
    exit 1
fi
