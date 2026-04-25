#!/bin/bash
# Local (single-GPU, 32GB) port of sbatch/eval_rolling_staircase_ode.sbatch.
# One Brighton ride on GPU 0; all paths point at /home/ashish local data.
set -e -o pipefail

cd /home/ashish/ARRWM

source /home/ashish/miniconda3/etc/profile.d/conda.sh
conda activate flash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STUDENT_CKPT="${STUDENT_CKPT:-/home/ashish/Downloads/action_ode_step0001000.pt}"
CONFIG="${CONFIG:-configs/longlive_phase1_rolling_staircase.yaml}"
ODE_CONFIG="${ODE_CONFIG:-configs/action_ode_distill_local.yaml}"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
CKPT_STEM="$(basename "$STUDENT_CKPT" .pt)"
# When DISABLE_RENOISE=0 (the Phase-1 default), carryover slots are
# renoised to the staircase ladder each rolling step. When DISABLE_RENOISE=1
# (ablation), carryover slots pass through at t=0 and only the trailing
# slot starts from fresh pure noise.
DISABLE_RENOISE="${DISABLE_RENOISE:-1}"
_TAG="renoise$([ "$DISABLE_RENOISE" = "1" ] && echo "Off" || echo "On")"
# Reflect the Phase-1 YAML's ``rollout_num_slots`` value in the outdir so
# 4-slot / 2-slot / 1-slot runs don't collide. Grepped directly from the
# YAML (first unindented match) — the pipeline reads the YAML as the
# authoritative switch.
NS=$(grep -E "^rollout_num_slots:" "$CONFIG" 2>/dev/null | head -1 | awk '{print $2}')
NS=${NS:-4}
OUTDIR="${OUTDIR:-/home/ashish/ARRWM/eval/eval_rolling_staircase_${CKPT_STEM}_ns${NS}_${_TAG}_${RUN_STAMP}}"
mkdir -p "$OUTDIR"
echo "OUTDIR=$OUTDIR"
echo "STUDENT_CKPT=$STUDENT_CKPT"
echo "CONFIG=$CONFIG"
echo "ODE_CONFIG=$ODE_CONFIG"

# Brighton ride (present in local /home/ashish/frodobots/frodobots_encoded/).
RIDE_ZARR="${RIDE_ZARR:-20240408152948.zarr}"
RIDE_TAG="${RIDE_TAG:-brighton}"
RIDE_OFFSET="${RIDE_OFFSET:-0}"

MAX_RIDE_FRAMES="${MAX_RIDE_FRAMES:-120}"
# Hard cap on the number of generated chunks (one commit per rolling
# step in the sequential-rollout mode). 20 gen chunks = 60 gen latents
# -> ~4 s of video at fps=8 + 3 prime chunks.
MAX_ROLLING_STEPS="${MAX_ROLLING_STEPS:-20}"
SEED="${SEED:-42}"
FPS="${FPS:-8}"

out_gpu="${OUTDIR}/gpu0_${RIDE_TAG}"
mkdir -p "$out_gpu"
log_gpu="${out_gpu}/stdout.log"
echo "[GPU 0] ride=${RIDE_ZARR} tag=${RIDE_TAG} offset=${RIDE_OFFSET} -> ${out_gpu}"

DISABLE_RENOISE_FLAG=""
if [ "$DISABLE_RENOISE" = "1" ]; then
    DISABLE_RENOISE_FLAG="--disable_renoise"
fi

CUDA_VISIBLE_DEVICES=0 \
    python utils/eval_rolling_staircase.py \
        --config "$CONFIG" \
        --ode_config "$ODE_CONFIG" \
        --student_ckpt "$STUDENT_CKPT" \
        --max_ride_frames "$MAX_RIDE_FRAMES" \
        --max_rolling_steps "$MAX_ROLLING_STEPS" \
        --seed "$SEED" \
        --fps "$FPS" \
        --encoded_root /home/ashish/frodobots/frodobots_encoded \
        --caption_root /home/ashish/frodobots/frodobots_captions/train \
        --motion_root /home/ashish/frodobots/frodobots_motion \
        --ss_vae_checkpoint action_query/checkpoints/ss_vae_8free.pt \
        --dtype bfloat16 \
        --output_dir "$out_gpu" \
        --ride_tag "$RIDE_TAG" \
        --rank_zarr "$RIDE_ZARR" \
        --rank_offset "$RIDE_OFFSET" \
        $DISABLE_RENOISE_FLAG \
        2>&1 | tee "$log_gpu"

echo "Done on $(date). mp4 under $OUTDIR"
ls -la "$OUTDIR"/*/*.mp4 2>/dev/null || true
