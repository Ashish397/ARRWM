#!/bin/bash
# Local (single-GPU, RTX 5090) port of
# sbatch/run_eval_AR_chain_mem_probe.sbatch. Runs the AR_refresh chain
# and the AR baseline serially on GPU 0 for the same Brighton ride.
set -e -o pipefail

cd /home/ashish/ARRWM

source /home/ashish/miniconda3/etc/profile.d/conda.sh
conda activate flash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

TS=$(date +%Y%m%d_%H%M%S)
BASE_OUTPUT="/home/ashish/ARRWM/eval/eval_AR_chain_mem_${TS}"
CONFIG="configs/action_ode_distill_local.yaml"
ENCODED_ROOT="/home/ashish/frodobots/frodobots_encoded"
CAPTION_ROOT="/home/ashish/frodobots/frodobots_captions/train"
MOTION_ROOT="/home/ashish/frodobots/frodobots_motion"
SS_VAE_CKPT="action_query/checkpoints/ss_vae_8free.pt"
RIDE_ZARR="${RIDE_ZARR:-20240408152948.zarr}"
RIDE_OFFSET="${RIDE_OFFSET:-0}"
STUDENT_CKPT="${STUDENT_CKPT:-/home/ashish/action_ode_step0001000.pt}"
DENOISING_STEPS=4
SEED=42
FIFO_SIZE=3
AR_INITIAL_CHUNKS=1
AR_GEN_CHUNKS=7

mkdir -p "$BASE_OUTPUT"
echo "Base output:  $BASE_OUTPUT"
echo "Student ckpt: $STUDENT_CKPT"
echo "Ride:         $RIDE_ZARR  offset=$RIDE_OFFSET"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader

# --- Run 1: AR_refresh (eval_causal_AR_chain.py) ---
OUT0="$BASE_OUTPUT/gpu0_ar_refresh_ds"
mkdir -p "$OUT0"
echo
echo "=== Run 1: AR_refresh (eval_causal_AR_chain.py) ==="
CUDA_VISIBLE_DEVICES=0 \
WORLD_SIZE=1 \
LOCAL_RANK=0 \
python utils/eval_causal_AR_chain.py \
    --config "$CONFIG" \
    --output_dir "$OUT0" \
    --student_ckpt "$STUDENT_CKPT" \
    --rank_zarr "$RIDE_ZARR" \
    --rank_offset "$RIDE_OFFSET" \
    --rank_mode dataset \
    --rank_tag ar_refresh_ds \
    --encoded_root "$ENCODED_ROOT" \
    --caption_root "$CAPTION_ROOT" \
    --motion_root "$MOTION_ROOT" \
    --ss_vae_checkpoint "$SS_VAE_CKPT" \
    --fifo_size "$FIFO_SIZE" \
    --ar_initial_chunks "$AR_INITIAL_CHUNKS" \
    --ar_gen_chunks "$AR_GEN_CHUNKS" \
    --denoising_steps "$DENOISING_STEPS" \
    --seed "$SEED" \
    > "$OUT0/eval.log" 2>&1 || { echo "RUN 1 FAILED"; tail -30 "$OUT0/eval.log"; exit 1; }
echo "--- AR_refresh log tail ---"
grep -E "rollout done|peak_alloc|cache" "$OUT0/eval.log" || tail -15 "$OUT0/eval.log"

# --- Run 2: AR baseline (eval_causal_AR.py) ---
OUT1="$BASE_OUTPUT/gpu0_ar_ds"
mkdir -p "$OUT1"
echo
echo "=== Run 2: AR baseline (eval_causal_AR.py) ==="
CUDA_VISIBLE_DEVICES=0 \
WORLD_SIZE=1 \
LOCAL_RANK=0 \
python utils/eval_causal_AR.py \
    --config "$CONFIG" \
    --output_dir "$OUT1" \
    --student_ckpt "$STUDENT_CKPT" \
    --mode ar \
    --rank_zarr "$RIDE_ZARR" \
    --rank_offset "$RIDE_OFFSET" \
    --rank_mode dataset \
    --rank_tag ar_ds \
    --encoded_root "$ENCODED_ROOT" \
    --caption_root "$CAPTION_ROOT" \
    --motion_root "$MOTION_ROOT" \
    --ss_vae_checkpoint "$SS_VAE_CKPT" \
    --num_causal_videos 7 \
    --denoising_steps "$DENOISING_STEPS" \
    --seed "$SEED" \
    --cache_chunks 12 \
    --chunks_per_step 1 \
    --ar_initial_chunks "$AR_INITIAL_CHUNKS" \
    --ar_gen_chunks "$AR_GEN_CHUNKS" \
    --ar_cache \
    > "$OUT1/eval.log" 2>&1 || { echo "RUN 2 FAILED"; tail -30 "$OUT1/eval.log"; exit 1; }
echo "--- AR baseline log tail ---"
grep -E "rollout done|peak_alloc|cache" "$OUT1/eval.log" || tail -15 "$OUT1/eval.log"

echo
echo "Both runs complete."
echo "Outputs:"
ls -la "$BASE_OUTPUT"/*/ 2>/dev/null
