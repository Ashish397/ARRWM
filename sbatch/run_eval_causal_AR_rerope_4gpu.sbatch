#!/bin/bash
#SBATCH --job-name=eval-causal-AR-rerope-4gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# AR-rerope head-to-head: validates the LongLive-style Block-Relativistic
# (Infinity) RoPE patch installed by ``utils/infinity_rope.py``. The
# patch is enabled via ``--infinity_rope`` (default on; see
# ``infinity_rope`` in ``configs/action_ode_distill.yaml``).
#
#   GPU 0  ar (cache=append) --no-infinity_rope    rank=counterfactual
#          ↳ buggy baseline: caches pre-rotated K at absolute global
#            positions. Stutters / staggers at chunk boundaries on long
#            rollouts because rolled cache slots keep stale rotations
#            and Q rotation grows unbounded.
#
#   GPU 1  ar (cache=append) --infinity_rope       rank=counterfactual
#          ↳ rerope fix: stores un-roped K, rotates Q/cached-K with
#            window-relative indices at attention time. Compare
#            head-to-head against GPU 0 — same ride, same seed, same
#            actions, only RoPE strategy differs.
#
#   GPU 2  ar_refresh per_pass                     rank=counterfactual
#          ↳ quality reference: re-runs the full 12-frame train forward
#            every denoise step (4×/chunk, no kv_cache). Slowest path
#            but the closest to chain numerically — sets the upper-bound
#            on what cached AR can hope to look like.
#
#   GPU 3  chain                                   rank=dataset
#          ↳ chain baseline: 7 separate 21-frame teacher-forced rollouts.
#            Same ride/offset as the others; uses dataset (not CF)
#            actions to give a clean quality anchor.
#
# All GPUs run on the same ride+offset as
# ``sbatch/run_eval_causal_AR_chain_4gpu.sbatch`` so per-GPU outputs
# are directly comparable. After completion, inspect:
#
#   eval/eval_AR_rerope_<TS>/gpu0_ar_append_norerope/...mp4
#   eval/eval_AR_rerope_<TS>/gpu1_ar_append_rerope/...mp4
#   eval/eval_AR_rerope_<TS>/gpu2_ar_refresh_perpass/...mp4
#   eval/eval_AR_rerope_<TS>/gpu3_chain_baseline/...mp4
#
# and the per-GPU eval.log files for wall times and peak memory.
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
BASE_OUTPUT="eval/eval_AR_rerope_${TS}"
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
FIFO_SIZE="${FIFO_SIZE:-3}"
AR_INITIAL_CHUNKS="${AR_INITIAL_CHUNKS:-1}"
AR_GEN_CHUNKS="${AR_GEN_CHUNKS:-7}"
CACHE_CHUNKS="${CACHE_CHUNKS:-12}"

mkdir -p "$BASE_OUTPUT"

echo "Base output:      $BASE_OUTPUT"
echo "Config:           $CONFIG"
echo "Manifest:         $MANIFEST"
echo "Student ckpt:     $STUDENT_CKPT"
echo "Ride:             $RIDE_ZARR  offset=$RIDE_OFFSET"
echo "Denoising steps:  $DENOISING_STEPS"
echo "FIFO size:        $FIFO_SIZE"
echo "AR init/gen:      $AR_INITIAL_CHUNKS / $AR_GEN_CHUNKS"
echo "Cache chunks:     $CACHE_CHUNKS"
echo "Seed:             $SEED"
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)"
nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader

# Per-GPU configuration. ``MODE`` selects the Python entrypoint, ``RM``
# is the rank action mode, ``TAG`` is the output-subdir + filename tag,
# ``EXTRA`` carries mode-specific CLI flags (e.g. infinity_rope toggle).
GPU0_MODE="ar";          GPU0_RM="counterfactual"; GPU0_TAG="ar_append_norerope"
GPU1_MODE="ar";          GPU1_RM="counterfactual"; GPU1_TAG="ar_append_rerope"
GPU2_MODE="ar_refresh";  GPU2_RM="counterfactual"; GPU2_TAG="ar_refresh_perpass"
GPU3_MODE="chain";       GPU3_RM="dataset";        GPU3_TAG="chain_baseline"

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
    echo "Launching GPU $GPU  mode=$MODE  rank_mode=$RM  tag=$TAG  out=$OUT"

    (
        if [ "$MODE" = "ar_refresh" ]; then
            CUDA_VISIBLE_DEVICES=$GPU \
            WORLD_SIZE=1 \
            LOCAL_RANK=0 \
            python utils/eval_causal_AR_chain.py \
                --config "$CONFIG" \
                --output_dir "$OUT" \
                --manifest "$MANIFEST" \
                --student_ckpt "$STUDENT_CKPT" \
                --rank_zarr "$RIDE_ZARR" \
                --rank_offset "$RIDE_OFFSET" \
                --rank_mode "$RM" \
                --rank_tag "$TAG" \
                --encoded_root "$ENCODED_ROOT" \
                --caption_root "$CAPTION_ROOT" \
                --motion_root "$MOTION_ROOT" \
                --ss_vae_checkpoint "$SS_VAE_CKPT" \
                --fifo_size "$FIFO_SIZE" \
                --ar_initial_chunks "$AR_INITIAL_CHUNKS" \
                --ar_gen_chunks "$AR_GEN_CHUNKS" \
                --denoising_steps "$DENOISING_STEPS" \
                --refresh_mode per_pass \
                --seed "$SEED" \
                > "$OUT/eval.log" 2>&1
        elif [ "$MODE" = "ar" ]; then
            # GPU 0: --no-infinity_rope (buggy baseline). GPU 1:
            # --infinity_rope (rerope fix). All other AR knobs identical
            # so the only difference between the two videos is the RoPE
            # strategy.
            if [ "$GPU" = "0" ]; then
                ROPE_FLAG="--no-infinity_rope"
            else
                ROPE_FLAG="--infinity_rope"
            fi
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
                --rank_mode "$RM" \
                --rank_tag "$TAG" \
                --encoded_root "$ENCODED_ROOT" \
                --caption_root "$CAPTION_ROOT" \
                --motion_root "$MOTION_ROOT" \
                --ss_vae_checkpoint "$SS_VAE_CKPT" \
                --num_causal_videos 7 \
                --denoising_steps "$DENOISING_STEPS" \
                --cache_chunks "$CACHE_CHUNKS" \
                --chunks_per_step 1 \
                --ar_initial_chunks "$AR_INITIAL_CHUNKS" \
                --ar_gen_chunks "$AR_GEN_CHUNKS" \
                --ar_cache \
                --ar_cache_refresh append \
                $ROPE_FLAG \
                --seed "$SEED" \
                > "$OUT/eval.log" 2>&1
        else
            CUDA_VISIBLE_DEVICES=$GPU \
            WORLD_SIZE=1 \
            LOCAL_RANK=0 \
            python utils/eval_causal_AR.py \
                --config "$CONFIG" \
                --output_dir "$OUT" \
                --manifest "$MANIFEST" \
                --student_ckpt "$STUDENT_CKPT" \
                --mode chain \
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
                > "$OUT/eval.log" 2>&1
        fi
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
echo "=== Per-GPU wall times (rerope summary) ==="
for GPU in 0 1 2 3; do
    LOG=$(ls "$BASE_OUTPUT/gpu${GPU}_"*/eval.log 2>/dev/null | head -1)
    if [ -n "$LOG" ]; then
        echo "--- GPU $GPU ($(basename $(dirname $LOG))) ---"
        grep -E "rollout done|streaming rollout done|infinity_rope=" "$LOG" | tail -5 || true
    fi
done

if [[ $FAIL -ne 0 ]]; then
    echo "WARN: one or more processes failed (check per-GPU eval.log)"
    exit 1
fi
