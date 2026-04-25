#!/bin/bash
#SBATCH --job-name=collapse-sweep-phase-a
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=2:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# Phase A of the rollout-collapse calibration sweep.
#
# Drives utils/collapse_sweep_phase_a.py on 4 GPUs / 4 processes /
# single node. Each process processes its slice of the rides list
# (rides[rank::world_size]) -- no DDP, no collectives, just file-system
# sharding. Output structure:
#
#   $OUTDIR/
#     rides_full.json                # global rides list (rank 0 only)
#     rank_0/<ride_dir>/metrics.csv  # per-step metrics
#     rank_0/<ride_dir>/rollout.mp4  # decoded slot-0 commits + GT
#     rank_0/<ride_dir>/meta.json    # ride metadata snapshot
#     rank_0/summary.json            # rank-level rollup
#     rank_1/...
#     rank_2/...
#     rank_3/...
#
# After the run completes, hand-label `<ride_dir>/rollout.mp4` for the
# visible-collapse step (or run a CoTracker/SSIM auto-label) and feed
# the metrics CSVs into the ROC analysis stage. Phase A's job ends
# once every rank has emitted its summary.json without errors.
#
# Per-process memory budget on H100 80GB:
#   - ODE student (bf16):                ~25 GiB
#   - Wan VAE (fp32):                    ~3 GiB
#   - v14 teacher (LoRA-merged, bf16):   ~25 GiB  (only with REAL_SCORE=1)
#   - KV caches + activations / ride:    ~5 GiB
# Total without M4: ~33 GiB; with M4: ~58 GiB. Both fit.
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
# Defaults to the ODE init at step 1000 (= what Phase-1 starts from).
# Override at submit time:
#   STUDENT_CKPT=/path/to/checkpoint.pt sbatch ...
STUDENT_CKPT="${STUDENT_CKPT:-/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_ode_distill_E/action_ode_step0001000.pt}"
CONFIG="${CONFIG:-configs/longlive_phase1_rolling_staircase.yaml}"
ODE_CONFIG="${ODE_CONFIG:-configs/action_ode_distill.yaml}"

# ---- Output dir ----------------------------------------------------------
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
CKPT_STEM="$(basename "$STUDENT_CKPT" .pt)"
OUTDIR="${OUTDIR:-/scratch/u6ex/as1748.u6ex/ARRWM/eval/collapse_sweep_phase_a_${CKPT_STEM}_${RUN_STAMP}}"
mkdir -p "$OUTDIR" logs
echo "OUTDIR=$OUTDIR"
echo "STUDENT_CKPT=$STUDENT_CKPT"
echo "CONFIG=$CONFIG"
echo "ODE_CONFIG=$ODE_CONFIG"

# ---- Sweep knobs --------------------------------------------------------
# AUTO_N rides total split across 4 ranks → AUTO_N/4 rides per GPU.
# AUTO_MIN_FRAMES is the minimum n_latent_frames per ride (set so every
# ride has enough room to actually collapse before the action stream
# runs out -- 300 latent frames ≈ 100 rolling steps at npb=3).
AUTO_N="${AUTO_N:-32}"
AUTO_MIN_FRAMES="${AUTO_MIN_FRAMES:-300}"
MAX_ROLLING_STEPS="${MAX_ROLLING_STEPS:-}"   # default: roll to natural ride end
MAX_RIDE_FRAMES="${MAX_RIDE_FRAMES:-}"        # default: full ride
SEED="${SEED:-42}"
FPS="${FPS:-8}"

# Real-score (M4). Off by default to keep the sweep cheap. Flip on for
# the full diagnostic; +25-30% wall-clock per step.
REAL_SCORE="${REAL_SCORE:-0}"
V14_CKPT="${V14_CKPT:-/scratch/u6ex/as1748.u6ex/ARRWM/logs/v14_balanced_weunz/causal_lora_step0006600.pt}"
REAL_TARGET_T="${REAL_TARGET_T:-625}"

# Optional explicit rides JSON (overrides AUTO_*).
RIDES_JSON="${RIDES_JSON:-}"

# Optional manifest (faster auto-discovery than globbing encoded_root).
MANIFEST="${MANIFEST:-}"

# mp4 decode toggle. Default ON; flip to 0 to skip the ~30s/ride decode.
DECODE_MP4="${DECODE_MP4:-1}"

# ---- Build COMMON_ARGS --------------------------------------------------
COMMON_ARGS=(
    --config "$CONFIG"
    --ode_config "$ODE_CONFIG"
    --student_ckpt "$STUDENT_CKPT"
    --output_dir "$OUTDIR"
    --auto_n "$AUTO_N"
    --auto_min_frames "$AUTO_MIN_FRAMES"
    --seed "$SEED"
    --fps "$FPS"
    --encoded_root /projects/u6ex/fbots/frodobots_encoded_weu
    --caption_root /projects/u6ex/fbots/frodobots_captions/train
    --motion_root /projects/u6ex/fbots/frodobots_motion
    --ss_vae_checkpoint action_query/checkpoints/ss_vae_8free.pt
    --dtype bfloat16
)
if [[ -n "$MAX_ROLLING_STEPS" ]]; then
    COMMON_ARGS+=(--max_rolling_steps "$MAX_ROLLING_STEPS")
fi
if [[ -n "$MAX_RIDE_FRAMES" ]]; then
    COMMON_ARGS+=(--max_ride_frames "$MAX_RIDE_FRAMES")
fi
if [[ -n "$RIDES_JSON" ]]; then
    COMMON_ARGS+=(--rides_json "$RIDES_JSON")
fi
if [[ -n "$MANIFEST" ]]; then
    COMMON_ARGS+=(--manifest "$MANIFEST")
fi
if [[ "$REAL_SCORE" == "1" ]]; then
    COMMON_ARGS+=(--enable_real_score
                  --v14_teacher_checkpoint "$V14_CKPT"
                  --real_target_t "$REAL_TARGET_T")
fi
if [[ "$DECODE_MP4" == "0" ]]; then
    COMMON_ARGS+=(--no_decode_mp4)
fi

# ---- Launch 4 processes, one per GPU ------------------------------------
# We use plain backgrounded subprocesses (mirroring
# eval_rolling_staircase_ode.sbatch). Each process pins to its own
# CUDA device via CUDA_VISIBLE_DEVICES and slices the rides list via
# RANK / WORLD_SIZE env vars. No torchrun -- there are no collectives.
WORLD_SIZE=4
for i in 0 1 2 3; do
    log_gpu="${OUTDIR}/rank_${i}_stdout.log"
    echo "[GPU $i] launching rank=$i log=$log_gpu"
    CUDA_VISIBLE_DEVICES=$i \
    RANK=$i WORLD_SIZE=$WORLD_SIZE LOCAL_RANK=0 \
        python utils/collapse_sweep_phase_a.py \
            "${COMMON_ARGS[@]}" \
        > "$log_gpu" 2>&1 &
done

wait
echo "All 4 ranks finished on $(date). Output: $OUTDIR"
echo "Rank summaries:"
ls -la "$OUTDIR"/rank_*/summary.json 2>/dev/null || true
echo "Per-ride metrics CSVs (count):"
find "$OUTDIR" -name 'metrics.csv' | wc -l
