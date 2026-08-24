#!/bin/bash
#SBATCH --job-name=fp-depth-offline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
#
# FP-DEPTH STUDY -- OFFLINE ARMS. Fingerprints ROLLOUT LATENTS THAT ALREADY
# EXIST ON DISK, so it needs one GPU for ~1h and backfills while the online
# 8-node arm queues. Nothing here generates a rollout.
#
# Submit:
#   sbatch --export=ALL,ODE_CKPT=<ode student .pt>[,MODE=all,TAG=...] \
#     sbatch/run_fp_depth_offline.sbatch
#
# MODE (default "flowrec"):
#   flowrec   PRIMARY   texture_abc recordings: 26 model arms x 40 committed
#                       chunks, GT-PAIRABLE.
#   tf        CONTROL   teacher-forced GT-context chunks at world-time
#                       positions 1..6.
#   long36    SECONDARY compass-steered 37-deep rollouts, REFERENCE-FREE.
#   latdump   MINOR     trainer rollout_latent_dump_dir, depth<=4, no GT.
#   all                 every one of the above, in that order.
#
# ---------------------------------------------------------------------
# WHY flowrec IS THE PRIMARY ARM
#
#   * DEPTH 0..39. The thesis is about drift COMPOUNDING with depth, and
#     39 is far past the >=10 the ship/no-ship criterion needs. (The old
#     latdump arm topped out at 4, which could neither ship nor falsify.)
#   * GT PAIRING IS VALID. rank0_ride.json records rank_mode="dataset",
#     i.e. the rollout was driven by the ride's OWN GT actions along the
#     real trajectory (utils/eval_causal_AR.py: noisy_fa_ar =
#     clean_fa_ar.clone()), NOT synthetic compass steering. So the
#     position-matched GT chunk is the correct target, which restores
#     per-position m calibration, the true-drift yardstick and a properly
#     PAIRED variance-contraction control.
#   * 26 INDEPENDENT REPLICATIONS. The arms are DIFFERENT MODELS, so the
#     study runs the full depth curve WITHIN each arm and then asks how
#     many reproduce it. Pooling them would blend between-model differences
#     into what reads as within-model spread; the script refuses to.
#   * FREE NOISE FLOOR. The _s43/_s44/_s45 directories are ODE_FLOW_SEED
#     replicates of the SAME arm, so the across-seed scatter is a measured
#     floor. Any depth effect smaller than it is not an effect.
#
# THE FLOOR IS NOT ZERO, AND IT MOVES WITH THE KNOBS. Measured against the
# shipped _dmd_fp_perturb with a literal x0_hat=0 teacher (2026-08-24), at
# scale 0.05/0.15/0.30: channel_rot -0.673/-0.610/-0.488, patch_shuffle
# -0.657/-0.591/-0.460, hf_scramble -0.567/-0.490/-0.354 (above the others --
# it preserves the low frequencies). So raw s is ORDINAL ONLY and only within
# one (perturb, scale, timestep) setting; the script REFUSES to pool them and
# prints the applicable floor in its own PROBE SETTINGS block.
#
# ALIGNMENT TRAP, handled in code, not assumed: load_latent_chunk applies
# _LATENT_HEAD_DROP internally while eval_causal_AR slices the raw zarr.
# The study measures |rollout_k - GT(k+shift)| over the first 10 chunks and
# REFUSES TO RUN unless shift 0 is the best match (verified 2026-08-24:
# shift 0 wins over -3..+3 aggregated across all 26 arms; a single chunk
# cannot resolve it, the aggregate can).
#
# THE long36 ARM IS REFERENCE-FREE BY CONSTRUCTION: those rollouts used
# CONSTANT COMPASS ACTIONS (|z|=0.5), so the model was deliberately steered
# off the real trajectory and position-matched GT is NOT a valid target.
# --flowrec-no-gt suppresses every GT-referenced quantity at the source so
# nobody can difference against GT there later.
# ---------------------------------------------------------------------
set -euo pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM
: "${ODE_CKPT:?submit with --export=ALL,ODE_CKPT=<ode student .pt>}"

MODE=${MODE:-flowrec}
TAG=${TAG:-offline}
ARRD=/lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM_data
FLOWREC_GLOB=${FLOWREC_GLOB:-"$ARRD/eval/texture_abc_*"}
TF_GLOB=${TF_GLOB:-"$ARRD/analysis/eval_final/flow_viz/flow_tf_gt0/r08_TF_c*"}
LONG36_GLOB=${LONG36_GLOB:-"$ARRD/analysis/eval_final/flow_viz/flow_long36_*/r08_*_s0"}
DUMP_DIR=${DUMP_DIR:-analysis/drift_probe_v1}
# r08 = analysis/eval_final/phaseA_windows.json[8]; the TF recordings ship no
# rank0_ride.json, so the pairing is stated here as an explicit claim.
TF_ZARR=${TF_ZARR:-/projects/u6ex/fbots/frodobots_encoded/20240322072501.zarr}
TF_OFFSET=${TF_OFFSET:-1380}

TEACHER_CKPT=${TEACHER_CKPT:-/scratch/u6ex/as1748.u6ex/ARRWM/logs/v14e_pca8_raw/causal_lora_step0005000.pt}
CONFIG=${CONFIG:-configs/action_forcing_phase3_dmd.yaml}
FP_SEEDS=${FP_SEEDS:-2}
# TIMESTEP SWEEP -- see the long note in sbatch/run_fp_depth_study.sh. t was
# CHOSEN, NOT MEASURED, so the default sweeps three values; each is reported
# separately against its own floor and cross-compared at the end. This is why
# the walltime is 4h rather than 2h. PROBE_TS=500 for a single-t run.
PROBE_TS=${PROBE_TS:-"250 500 750"}
PERTURB=${PERTURB:-hf_scramble}
STRIDE=${STRIDE:-1}
MAX_ARMS=${MAX_ARMS:-0}

source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR HF_HUB_CACHE=$CACHE_DIR
export HUGGINGFACE_HUB_CACHE=$CACHE_DIR TRANSFORMERS_CACHE=$CACHE_DIR
# 14e lineage: LOAD-BEARING. Without it zarr_dataset defaults to ss_vae and
# the study encodes a DIFFERENT physical action than the teacher was trained
# on -- silent corruption, not an error.
export ARRWM_ACTION_ENCODER=pca_raw
export TMPDIR=/tmp
unset LOCALDIR APPTAINER_CACHEDIR
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

test -f "$ODE_CKPT"
test -f "$TEACHER_CKPT"
mkdir -p logs
JID=${SLURM_JOB_ID:-manual}

COMMON=(
  --config "$CONFIG"
  --ode-ckpt "$ODE_CKPT"
  --teacher-ckpt "$TEACHER_CKPT"
  --encoded-root /projects/u6ex/fbots/frodobots_encoded_weunz
  --caption-root /projects/u6ex/fbots/frodobots_captions/train
  --motion-root /projects/u6ex/fbots/frodobots_motion
  --ss-vae-checkpoint action_query/checkpoints/ss_vae_8free.pt
  --action-dims 0 1
  --denoising-step-list 1000 625 357.142857 208.333333
  --probe-timestep $PROBE_TS
  --perturb "$PERTURB"
  --fp-seeds "$FP_SEEDS"
  --auc-min 0.8
  --deep-depth-min 10
  --verdict-metric s
  --seed 1234
)

run_flowrec() {
  local out=logs/fp_depth_flowrec_${TAG}_${JID}.jsonl
  echo "[fp] PRIMARY flowrec -> $out"
  python analysis/dmd_fp_depth_study.py "${COMMON[@]}" \
    --from-flowrec $FLOWREC_GLOB \
    --teacher-frames 21 \
    --depth-bands 0 3 10 20 30 40 \
    --flowrec-depth-stride "$STRIDE" \
    --max-arms "$MAX_ARMS" \
    --out "$out"
}

run_tf() {
  local out=logs/fp_depth_tfctrl_${TAG}_${JID}.jsonl
  echo "[fp] CONTROL teacher-forced GT-context -> $out"
  # Separates WORLD-TIME POSITION from ROLL DEPTH: these are depth-0 chunks
  # at positions 1..6, exposure-bias-free. If s moves across them, a
  # depth trend in the AR arm is partly a scene/time-of-day effect.
  python analysis/dmd_fp_depth_study.py "${COMMON[@]}" \
    --from-flowrec $TF_GLOB \
    --flowrec-zarr "$TF_ZARR" --flowrec-offset "$TF_OFFSET" \
    --teacher-frames 21 \
    --depth-bands 0 1 \
    --out "$out"
}

run_long36() {
  local out=logs/fp_depth_long36_${TAG}_${JID}.jsonl
  echo "[fp] SECONDARY long36 (REFERENCE-FREE) -> $out"
  python analysis/dmd_fp_depth_study.py "${COMMON[@]}" \
    --from-flowrec $LONG36_GLOB \
    --flowrec-no-gt \
    --teacher-frames 21 \
    --depth-bands 0 3 10 20 30 37 \
    --out "$out"
}

run_latdump() {
  local out=logs/fp_depth_latdump_${TAG}_${JID}.jsonl
  echo "[fp] MINOR latdump cross-check (depth<=4, no GT) -> $out"
  python analysis/dmd_fp_depth_study.py "${COMMON[@]}" \
    --from-latents "$DUMP_DIR" \
    --offline-window-frames 9 \
    --num-gt 64 \
    --ride-scan-limit 200 \
    --out "$out"
}

case "$MODE" in
  flowrec) run_flowrec ;;
  tf)      run_tf ;;
  long36)  run_long36 ;;
  latdump) run_latdump ;;
  all)     run_flowrec; run_tf; run_long36; run_latdump ;;
  *) echo "unknown MODE=$MODE" >&2; exit 2 ;;
esac

echo "[fp] done. Re-report any dump without a GPU:"
echo "  python analysis/dmd_fp_depth_study.py --report-only logs/fp_depth_*_${JID}.jsonl"
