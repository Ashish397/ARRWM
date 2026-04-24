#!/bin/bash
#SBATCH --job-name=eval-roll-stair-ode
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=1:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# Rolling-staircase INFERENCE visualisation for the ODE-distilled
# student. Runs the SAME pipeline Phase-1 training wraps the generator
# with (no loss / no teacher / no fake-score), dumps one annotated mp4
# per GPU, one mp4 per held-out ride.
#
# Use this to eyeball how the ODE init behaves when placed inside the
# Phase-1 rolling-staircase wrapper — if the Phase-1 vis looks uglier
# than what you see here, that means training is regressing away from
# the ODE and you should pull the learning rate down / add more GT
# context / etc.
#
# Layout:
#   4 GPUs, 4 rides, 1 ride per GPU, 1 process per GPU.
#   Each process calls utils/eval_rolling_staircase.py with
#   CUDA_VISIBLE_DEVICES bound to its own GPU so there's no contention.
#
# Outputs land in $OUTDIR/gpuN_<ride_tag>/... — rename $OUTDIR to stamp
# different runs apart. mp4s are drawn in the same v14 / ODE-distill
# eval style (ride id + chunk index + action-z bars + phase/t tag), so
# you can flip between these and the Phase-1 training vis mp4s without
# adjusting your eyes.
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
mkdir -p "$CACHE_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Student checkpoint -------------------------------------------------
# Defaults to the ODE init (step 1000) as requested; override at submit
# time with `STUDENT_CKPT=... sbatch ...` to point at any Phase-1
# snapshot written by train.py (keys: generator / action_projection /
# action_token_projection / step).
STUDENT_CKPT="${STUDENT_CKPT:-/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_ode_distill_E/action_ode_step0001000.pt}"
CONFIG="${CONFIG:-configs/longlive_phase1_rolling_staircase.yaml}"
ODE_CONFIG="${ODE_CONFIG:-configs/action_ode_distill.yaml}"

# ---- Output dir ----------------------------------------------------------
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
CKPT_STEM="$(basename "$STUDENT_CKPT" .pt)"
OUTDIR="${OUTDIR:-/scratch/u6ex/as1748.u6ex/ARRWM/eval/eval_rolling_staircase_${CKPT_STEM}_${RUN_STAMP}}"
mkdir -p "$OUTDIR" logs
echo "OUTDIR=$OUTDIR"
echo "STUDENT_CKPT=$STUDENT_CKPT"
echo "CONFIG=$CONFIG"
echo "ODE_CONFIG=$ODE_CONFIG"

# ---- Per-GPU ride selection ---------------------------------------------
# 4 held-out rides from the WEU pool. Pick any 4 .zarr basenames that
# (a) are longer than ~120 latent frames and (b) you want to eyeball.
# Override at submit time with `RIDES_GPU0=... RIDES_GPU1=... sbatch ...`.
# Default set is hand-picked to cover straight / turny / urban /
# suburban so you see how the model generalises across regimes.
RIDE_ZARR_0="${RIDE_ZARR_0:-20240216101235.zarr}"
RIDE_TAG_0="${RIDE_TAG_0:-madrid_turny}"
RIDE_OFFSET_0="${RIDE_OFFSET_0:-0}"

RIDE_ZARR_1="${RIDE_ZARR_1:-20240309135856.zarr}"
RIDE_TAG_1="${RIDE_TAG_1:-brighton_seafront}"
RIDE_OFFSET_1="${RIDE_OFFSET_1:-0}"

RIDE_ZARR_2="${RIDE_ZARR_2:-20240420091612.zarr}"
RIDE_TAG_2="${RIDE_TAG_2:-suburban_straight}"
RIDE_OFFSET_2="${RIDE_OFFSET_2:-0}"

RIDE_ZARR_3="${RIDE_ZARR_3:-20240321123057.zarr}"
RIDE_TAG_3="${RIDE_TAG_3:-urban_mixed}"
RIDE_OFFSET_3="${RIDE_OFFSET_3:-0}"

MAX_RIDE_FRAMES="${MAX_RIDE_FRAMES:-240}"     # ~77 rolling steps @ npb=3.
MAX_ROLLING_STEPS="${MAX_ROLLING_STEPS:-120}" # Hard wall-clock cap.
SEED="${SEED:-42}"
FPS="${FPS:-8}"

COMMON_ARGS=(
    --config "$CONFIG"
    --ode_config "$ODE_CONFIG"
    --student_ckpt "$STUDENT_CKPT"
    --max_ride_frames "$MAX_RIDE_FRAMES"
    --max_rolling_steps "$MAX_ROLLING_STEPS"
    --seed "$SEED"
    --fps "$FPS"
    --encoded_root /projects/u6ex/fbots/frodobots_encoded_weu
    --caption_root /projects/u6ex/fbots/frodobots_captions/train
    --motion_root /projects/u6ex/fbots/frodobots_motion
    --ss_vae_checkpoint action_query/checkpoints/ss_vae_8free.pt
    --dtype bfloat16
)

# ---- Launch 4 processes, one per GPU ------------------------------------
# We don't use torchrun: every process is independent (no DDP, no
# synchronisation) — the only shared resource is the filesystem. Each
# one pins itself to a single CUDA device via CUDA_VISIBLE_DEVICES so
# we don't accidentally fight for memory on GPU 0.
for i in 0 1 2 3; do
    zarr_var="RIDE_ZARR_$i"; zarr="${!zarr_var}"
    tag_var="RIDE_TAG_$i";   tag="${!tag_var}"
    off_var="RIDE_OFFSET_$i"; off="${!off_var}"

    out_gpu="${OUTDIR}/gpu${i}_${tag}"
    mkdir -p "$out_gpu"
    log_gpu="${out_gpu}/stdout.log"

    echo "[GPU $i] ride=${zarr} tag=${tag} offset=${off} -> ${out_gpu}"

    CUDA_VISIBLE_DEVICES=$i \
        python utils/eval_rolling_staircase.py \
            "${COMMON_ARGS[@]}" \
            --output_dir "$out_gpu" \
            --ride_tag "$tag" \
            --rank_zarr "$zarr" \
            --rank_offset "$off" \
        > "$log_gpu" 2>&1 &
done

wait
echo "All 4 rides done on $(date). mp4s under $OUTDIR"
ls -la "$OUTDIR"/*/*.mp4 2>/dev/null || true
