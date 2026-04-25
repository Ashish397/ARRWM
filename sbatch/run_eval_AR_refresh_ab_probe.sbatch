#!/bin/bash
#SBATCH --job-name=eval-AR-refresh-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=2
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# A/B probe for the two AR_refresh recaching strategies in
# utils/eval_causal_AR_chain.py, on the same ride / seed / student:
#
#   GPU 0: --refresh_mode per_pass   (original AR_refresh — 4 big
#          12-frame _forward_train calls per chunk. Looks clean but
#          ~5x slower than AR baseline.)
#
#   GPU 1: --refresh_mode per_chunk  (Recommendation #1 — LongLive-
#          inspired: rebuild a real kv_cache once per chunk via N
#          sequential 3-frame refresh forwards, then 4 × 3-frame
#          _forward_inference denoise passes reusing the cached
#          context K/V. ~3-5x faster than per_pass. Shares numerics
#          with eval_causal_AR.py --ar_cache_refresh full_fifo so
#          must be visually verified.)
#
# Both runs emit peak_alloc/peak_reserved and wall-time in their
# "rollout done" lines. Compare the two mp4s side-by-side to decide
# if per_chunk preserves per_pass's visual quality.

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
BASE_OUTPUT="eval/eval_AR_refresh_ab_${TS}"
CONFIG="configs/action_ode_distill.yaml"
MANIFEST="logs/v14_balanced_weunz/.ride_manifest.pt"
ENCODED_ROOT="/projects/u6ex/fbots/frodobots_encoded"
CAPTION_ROOT="/projects/u6ex/fbots/frodobots_captions/train"
MOTION_ROOT="/projects/u6ex/fbots/frodobots_motion"
SS_VAE_CKPT="action_query/checkpoints/ss_vae_8free.pt"
RIDE_ZARR="20240216101235.zarr"
RIDE_OFFSET=100
STUDENT_CKPT="${STUDENT_CKPT:-/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_ode_distill_E/action_ode_step0001000.pt}"
DENOISING_STEPS=4
SEED=42
FIFO_SIZE=3
AR_INITIAL_CHUNKS=1
AR_GEN_CHUNKS=7

mkdir -p "$BASE_OUTPUT"
echo "Base output:  $BASE_OUTPUT"
echo "Student ckpt: $STUDENT_CKPT"
echo "Ride:         $RIDE_ZARR  offset=$RIDE_OFFSET"
nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader

run_ar_refresh() {
    local gpu_idx="$1"
    local refresh_mode="$2"
    local out_sub="$3"
    local tag="$4"

    local out_dir="$BASE_OUTPUT/$out_sub"
    mkdir -p "$out_dir"
    (
        CUDA_VISIBLE_DEVICES="$gpu_idx" \
        WORLD_SIZE=1 \
        LOCAL_RANK=0 \
        python utils/eval_causal_AR_chain.py \
            --config "$CONFIG" \
            --output_dir "$out_dir" \
            --manifest "$MANIFEST" \
            --student_ckpt "$STUDENT_CKPT" \
            --rank_zarr "$RIDE_ZARR" \
            --rank_offset "$RIDE_OFFSET" \
            --rank_mode dataset \
            --rank_tag "$tag" \
            --encoded_root "$ENCODED_ROOT" \
            --caption_root "$CAPTION_ROOT" \
            --motion_root "$MOTION_ROOT" \
            --ss_vae_checkpoint "$SS_VAE_CKPT" \
            --refresh_mode "$refresh_mode" \
            --fifo_size "$FIFO_SIZE" \
            --ar_initial_chunks "$AR_INITIAL_CHUNKS" \
            --ar_gen_chunks "$AR_GEN_CHUNKS" \
            --denoising_steps "$DENOISING_STEPS" \
            --seed "$SEED" \
            > "$out_dir/eval.log" 2>&1
    ) &
    echo $!
}

PID0=$(run_ar_refresh 0 per_pass   gpu0_per_pass   ar_refresh_per_pass)
PID1=$(run_ar_refresh 1 per_chunk  gpu1_per_chunk  ar_refresh_per_chunk)

FAIL=0
wait $PID0 || FAIL=1
wait $PID1 || FAIL=1

echo
echo "============================================================"
echo "=== per_pass (GPU 0) log tail ==="
grep -E "rollout done|peak_alloc|per_pass\]|per_chunk\]" \
    "$BASE_OUTPUT/gpu0_per_pass/eval.log" || true

echo
echo "=== per_chunk (GPU 1) log tail ==="
grep -E "rollout done|peak_alloc|per_pass\]|per_chunk\]" \
    "$BASE_OUTPUT/gpu1_per_chunk/eval.log" || true
echo "============================================================"

echo
echo "Raw rollout mp4s:"
find "$BASE_OUTPUT" -name '*_rollout_raw.mp4' -print

if [[ $FAIL -ne 0 ]]; then
    echo "WARN: one or more processes failed"
    exit 1
fi
