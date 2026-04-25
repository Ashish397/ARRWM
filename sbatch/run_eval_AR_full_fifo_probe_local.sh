#!/bin/bash
# Local (single-GPU, RTX 5090) port of
# sbatch/run_eval_AR_full_fifo_probe.sbatch. Runs the append baseline
# (cache_chunks=12) and full_fifo variant (cache_chunks=3) serially on
# GPU 0 for the same Brighton ride.
set -e -o pipefail

cd /home/ashish/ARRWM

source /home/ashish/miniconda3/etc/profile.d/conda.sh
conda activate flash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

TS=$(date +%Y%m%d_%H%M%S)
BASE_OUTPUT="/home/ashish/ARRWM/eval/eval_AR_full_fifo_${TS}"
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
AR_INITIAL_CHUNKS=1
AR_GEN_CHUNKS=7

mkdir -p "$BASE_OUTPUT"
echo "Base output: $BASE_OUTPUT"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader

run_ar () {
    local TAG=$1
    local REFRESH=$2
    local CACHE_CHUNKS=$3
    local RM=$4
    local OUT="$BASE_OUTPUT/gpu0_${TAG}"
    mkdir -p "$OUT"
    echo
    echo "=== Run: refresh=$REFRESH  cache_chunks=$CACHE_CHUNKS  rm=$RM ==="
    echo "   out=$OUT"
    CUDA_VISIBLE_DEVICES=0 \
    WORLD_SIZE=1 \
    LOCAL_RANK=0 \
    python utils/eval_causal_AR.py \
        --config "$CONFIG" \
        --output_dir "$OUT" \
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
        --seed "$SEED" \
        --cache_chunks "$CACHE_CHUNKS" \
        --chunks_per_step 1 \
        --ar_initial_chunks "$AR_INITIAL_CHUNKS" \
        --ar_gen_chunks "$AR_GEN_CHUNKS" \
        --ar_cache \
        --ar_cache_refresh "$REFRESH" \
        > "$OUT/eval.log" 2>&1 || { echo "$TAG FAILED"; tail -30 "$OUT/eval.log"; return 1; }
    grep -E "rollout done|emitted chunks|peak_alloc|cache-fill strategy|cache_refresh" "$OUT/eval.log" | head -20 || tail -15 "$OUT/eval.log"
}

run_ar append_baseline append   12 dataset
run_ar full_fifo       full_fifo 3 dataset

echo
echo "=== Results summary ==="
for d in "$BASE_OUTPUT"/gpu0_*; do
    echo "-- $d --"
    grep -E "rollout done|peak_alloc|cache-fill strategy|cache_refresh" "$d/eval.log" | head -10 || true
done
