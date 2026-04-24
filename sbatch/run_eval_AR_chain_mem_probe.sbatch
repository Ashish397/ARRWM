#!/bin/bash
#SBATCH --job-name=eval-AR-chain-mem-probe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=2
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# Quick memory probe: one AR baseline + one AR_refresh on the same
# ride, same student, same seed. Both pipelines now emit
# peak_alloc/peak_reserved alongside the "rollout done" line.

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
BASE_OUTPUT="eval/eval_AR_chain_mem_${TS}"
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

# GPU 0: AR_refresh (dataset)
OUT0="$BASE_OUTPUT/gpu0_ar_refresh_ds"
mkdir -p "$OUT0"
(
    CUDA_VISIBLE_DEVICES=0 \
    WORLD_SIZE=1 \
    LOCAL_RANK=0 \
    python utils/eval_causal_AR_chain.py \
        --config "$CONFIG" \
        --output_dir "$OUT0" \
        --manifest "$MANIFEST" \
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
        > "$OUT0/eval.log" 2>&1
) &
PID0=$!

# GPU 1: AR baseline (dataset)
OUT1="$BASE_OUTPUT/gpu1_ar_ds"
mkdir -p "$OUT1"
(
    CUDA_VISIBLE_DEVICES=1 \
    WORLD_SIZE=1 \
    LOCAL_RANK=0 \
    python utils/eval_causal_AR.py \
        --config "$CONFIG" \
        --output_dir "$OUT1" \
        --manifest "$MANIFEST" \
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
        > "$OUT1/eval.log" 2>&1
) &
PID1=$!

FAIL=0
wait $PID0 || FAIL=1
wait $PID1 || FAIL=1
echo "--- AR_refresh log tail ---"
grep -E "rollout done|peak_alloc" "$OUT0/eval.log" || true
echo "--- AR baseline log tail ---"
grep -E "rollout done|peak_alloc" "$OUT1/eval.log" || true

if [[ $FAIL -ne 0 ]]; then
    echo "WARN: one or more processes failed"
    exit 1
fi
