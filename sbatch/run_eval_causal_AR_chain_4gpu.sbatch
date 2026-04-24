#!/bin/bash
#SBATCH --job-name=eval-causal-AR-chain-4gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# AR_refresh head-to-head with the plain AR + chain baselines from
# run_eval_causal_AR_4gpu.sbatch, on the SAME ride + offset used by
# the v14 chain eval (20240216101235.zarr @ offset=100).
#
#   GPU 0  ar_refresh   rank_mode=counterfactual
#   GPU 1  ar_refresh   rank_mode=dataset
#   GPU 2  ar           rank_mode=dataset          (baseline)
#   GPU 3  chain        rank_mode=dataset          (baseline)
#
# The ar_refresh rollout re-computes context K/V from raw latents at
# every denoise step (chain-style freshness) while keeping the AR
# rollout structure (one new 3-frame chunk committed per streaming
# step, noise seed per chunk). FIFO=3 chunks by default. See
# utils/eval_causal_AR_chain.py for the full rationale.
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
BASE_OUTPUT="eval/eval_AR_chain_${TS}"
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

mkdir -p "$BASE_OUTPUT"

echo "Base output:      $BASE_OUTPUT"
echo "Config:           $CONFIG"
echo "Manifest:         $MANIFEST"
echo "Student ckpt:     $STUDENT_CKPT"
echo "Ride:             $RIDE_ZARR  offset=$RIDE_OFFSET"
echo "Denoising steps:  $DENOISING_STEPS"
echo "FIFO size:        $FIFO_SIZE"
echo "AR init/gen:      $AR_INITIAL_CHUNKS / $AR_GEN_CHUNKS"
echo "Seed:             $SEED"
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)"
nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader

# Per-GPU configuration: mode + rank_mode + tag.
GPU0_MODE="ar_refresh"; GPU0_RM="counterfactual"; GPU0_TAG="ar_refresh_cf"
GPU1_MODE="ar_refresh"; GPU1_RM="dataset";        GPU1_TAG="ar_refresh_ds"
GPU2_MODE="ar";         GPU2_RM="dataset";        GPU2_TAG="ar_ds_baseline"
GPU3_MODE="chain";      GPU3_RM="dataset";        GPU3_TAG="chain_ds_baseline"

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
                --seed "$SEED" \
                > "$OUT/eval.log" 2>&1
        else
            # AR / chain baselines via the existing script.
            AR_ARGS=()
            if [ "$MODE" = "ar" ]; then
                AR_ARGS=(
                    --cache_chunks 12
                    --chunks_per_step 1
                    --ar_initial_chunks "$AR_INITIAL_CHUNKS"
                    --ar_gen_chunks "$AR_GEN_CHUNKS"
                    --ar_cache
                )
            fi
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

if [[ $FAIL -ne 0 ]]; then
    echo "WARN: one or more processes failed (check per-GPU eval.log)"
    exit 1
fi
