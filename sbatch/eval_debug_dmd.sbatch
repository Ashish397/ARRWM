#!/bin/bash
#SBATCH --job-name=eval-debug-dmd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=1:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# DMD-teacher sanity visualiser.
#
# For each ride, produces 4 side-by-side mp4s after rolling `num_blocks`
# chunks through the exact same Phase-1-style rolling setup (warmup
# prime + per-block causal denoise + commit):
#
#   1. student_hi       — student, pass t_hi -> 0 (default 750 -> 0),
#                         one forward per chunk, BEFORE renoising.
#                         Captures the student's raw per-step x0 at
#                         the noise level the Phase-1 staircase's slot
#                         3 operates at.
#
#   2. student_lo       — same student, same left-context, but the
#                         noisy input is at t_lo (default 250 -> 0).
#                         Captures the student's x0 at the noise level
#                         slot 1 operates at.
#
#   3. teacher_raw      — v14 LoRA teacher (bidirectional) seeing
#                         [ 4 GT chunks, student_hi.prev_commit,
#                           noisy GT chunk @ t=teacher_t ]
#                         as context; records teacher's x0 prediction
#                         for the target chunk. This is literally what
#                         the Phase-1 DMD's real_score sees for slot 0.
#
#   4. teacher_cleaned  — same as teacher_raw but the student's prev
#                         commit is first "cleaned up" by a separate
#                         teacher forward (no grad) on
#                         [ 4 GT chunks, noisy(prev_commit, cleanup_t) ]
#                         so we can see whether the raw student commit
#                         is degrading what the teacher predicts.
#
# Use cases:
#
#   * If student_hi / student_lo look OK but teacher_raw doesn't, the
#     DMD direction is pulling the student away from its own sane
#     predictions — tune dmd_loss_weight / min_score_timestep /
#     ts_schedule.
#
#   * If teacher_raw and teacher_cleaned diverge meaningfully, the
#     student's rolled context is poisoning the teacher's scores, so
#     the student's roll quality (not the teacher) is the bottleneck.
#
#   * If student_hi visibly drifts off-ride after a few blocks, the
#     student itself is the problem — train longer ODE first / check
#     the student load path / check RoPE.
#
# Layout:
#   4 GPUs, 4 rides, 1 ride per GPU, 1 process per GPU. Each process
#   writes 4 mp4s to its own $OUTDIR/gpuN_<tag>/ subdir, plus one
#   combined stats.json.
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

# ---- Checkpoints --------------------------------------------------------
STUDENT_CKPT="${STUDENT_CKPT:-/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_ode_distill_E/action_ode_step0001000.pt}"
V14_TEACHER_CKPT="${V14_TEACHER_CKPT:-/scratch/u6ex/as1748.u6ex/ARRWM/logs/v14_balanced_weunz/causal_lora_step0006600.pt}"
CONFIG="${CONFIG:-configs/longlive_phase1_rolling_staircase.yaml}"
ODE_CONFIG="${ODE_CONFIG:-configs/action_ode_distill.yaml}"

# ---- Debug-DMD knobs ----------------------------------------------------
DEBUG_NUM_BLOCKS="${DEBUG_NUM_BLOCKS:-10}"         # Request: "up to 10 blocks rolling".
DEBUG_STUDENT_T_HI="${DEBUG_STUDENT_T_HI:-750}"    # "pass 750 -> 0".
DEBUG_STUDENT_T_LO="${DEBUG_STUDENT_T_LO:-250}"    # "pass 250 -> 0".
DEBUG_TEACHER_T="${DEBUG_TEACHER_T:-750}"          # Teacher target-chunk noise.
DEBUG_CLEANUP_T="${DEBUG_CLEANUP_T:-750}"          # Teacher prev-commit cleanup noise.
DEBUG_TEACHER_NUM_GT_CHUNKS="${DEBUG_TEACHER_NUM_GT_CHUNKS:-4}"

# ---- LoRA config (v14) --------------------------------------------------
# Rank + alpha must match the v14 training config, or the strict load
# fails. 256/256 matches causal_lora_diffusion_teacher_v14.yaml.
V14_LORA_RANK="${V14_LORA_RANK:-256}"
V14_LORA_ALPHA="${V14_LORA_ALPHA:-256}"
V14_LORA_DROPOUT="${V14_LORA_DROPOUT:-0.0}"

# ---- Output dir ---------------------------------------------------------
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
CKPT_STEM="$(basename "$STUDENT_CKPT" .pt)"
OUTDIR="${OUTDIR:-/scratch/u6ex/as1748.u6ex/ARRWM/eval/eval_debug_dmd_${CKPT_STEM}_${RUN_STAMP}}"
mkdir -p "$OUTDIR" logs
echo "OUTDIR=$OUTDIR"
echo "STUDENT_CKPT=$STUDENT_CKPT"
echo "V14_TEACHER_CKPT=$V14_TEACHER_CKPT"
echo "CONFIG=$CONFIG"
echo "ODE_CONFIG=$ODE_CONFIG"
echo "DEBUG: num_blocks=$DEBUG_NUM_BLOCKS t_hi=$DEBUG_STUDENT_T_HI t_lo=$DEBUG_STUDENT_T_LO"
echo "       teacher_t=$DEBUG_TEACHER_T cleanup_t=$DEBUG_CLEANUP_T gt_chunks=$DEBUG_TEACHER_NUM_GT_CHUNKS"

# ---- Per-GPU ride selection --------------------------------------------
# 4 held-out rides from the WEU pool. Same defaults as
# eval_rolling_staircase_ode.sbatch so you can directly compare the two
# runs side by side on the same content.
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

# At npb=3 frames/chunk + 9 primed frames, 10 blocks needs 9 + 30 = 39
# latent frames. Give ourselves a floor of 80 to match the training-time
# min_ride_length and avoid near-end-of-ride truncations.
MAX_RIDE_FRAMES="${MAX_RIDE_FRAMES:-120}"
SEED="${SEED:-42}"
FPS="${FPS:-8}"

COMMON_ARGS=(
    --debug_dmd
    --config "$CONFIG"
    --ode_config "$ODE_CONFIG"
    --student_ckpt "$STUDENT_CKPT"
    --v14_teacher_checkpoint "$V14_TEACHER_CKPT"
    --v14_lora_rank "$V14_LORA_RANK"
    --v14_lora_alpha "$V14_LORA_ALPHA"
    --v14_lora_dropout "$V14_LORA_DROPOUT"
    --debug_num_blocks "$DEBUG_NUM_BLOCKS"
    --debug_student_t_hi "$DEBUG_STUDENT_T_HI"
    --debug_student_t_lo "$DEBUG_STUDENT_T_LO"
    --debug_teacher_t "$DEBUG_TEACHER_T"
    --debug_cleanup_t "$DEBUG_CLEANUP_T"
    --debug_teacher_num_gt_chunks "$DEBUG_TEACHER_NUM_GT_CHUNKS"
    --max_ride_frames "$MAX_RIDE_FRAMES"
    --seed "$SEED"
    --fps "$FPS"
    --encoded_root /projects/u6ex/fbots/frodobots_encoded_weu
    --caption_root /projects/u6ex/fbots/frodobots_captions/train
    --motion_root /projects/u6ex/fbots/frodobots_motion
    --ss_vae_checkpoint action_query/checkpoints/ss_vae_8free.pt
    --dtype bfloat16
)

# ---- Launch 4 processes, one per GPU -----------------------------------
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
echo "All 4 rides done on $(date). 4 mp4s x 4 GPUs = 16 mp4s under $OUTDIR"
ls -la "$OUTDIR"/*/*.mp4 2>/dev/null || true
