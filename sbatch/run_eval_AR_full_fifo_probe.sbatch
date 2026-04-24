#!/bin/bash
#SBATCH --job-name=eval-AR-full-fifo-probe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=2
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# A/B probe for the new Variant A cache-refresh strategy.
# GPU 0: AR with ar_cache_refresh=append        (baseline, cache_chunks=12)
# GPU 1: AR with ar_cache_refresh=full_fifo     (new,      cache_chunks=3)

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
BASE_OUTPUT="eval/eval_AR_full_fifo_${TS}"
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
AR_INITIAL_CHUNKS=1
AR_GEN_CHUNKS=7

mkdir -p "$BASE_OUTPUT"
echo "Base output: $BASE_OUTPUT"
nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader

run_ar () {
    local GPU=$1
    local TAG=$2
    local REFRESH=$3
    local CACHE_CHUNKS=$4
    local RM=$5
    local OUT="$BASE_OUTPUT/gpu${GPU}_${TAG}"
    mkdir -p "$OUT"
    echo "Launching GPU $GPU  refresh=$REFRESH  cache_chunks=$CACHE_CHUNKS  rm=$RM  out=$OUT"
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
            > "$OUT/eval.log" 2>&1
    )
}

run_ar 0 append_baseline append   12 dataset &
PID0=$!
run_ar 1 full_fifo       full_fifo 3 dataset &
PID1=$!

FAIL=0
wait $PID0 || FAIL=1
wait $PID1 || FAIL=1

echo
echo "=== Results summary ==="
for d in "$BASE_OUTPUT"/gpu*; do
    echo "-- $d --"
    grep -E "rollout done|emitted chunks|emitted chunk at|peak_alloc|cache-fill strategy" "$d/eval.log" || true
done

if [[ $FAIL -ne 0 ]]; then
    echo "WARN: one or more processes failed"
    exit 1
fi
