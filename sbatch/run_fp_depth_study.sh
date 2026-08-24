#!/bin/bash
# FP-DEPTH STUDY (online) -- does the DMD teacher's self-fingerprint rank
# off-manifold distance? Go/no-go gate on docs/DMD_MANIFOLD_GATE.md's
# fingerprint gate. Design + criterion: analysis/dmd_fp_depth_study.py.
#
# SINGLE GPU, srun --overlap onto an ALREADY-RUNNING holder (the established
# pattern here; see sbatch/run_smoke_gate.sh + sbatch/_fgan_holder1n.sh).
# Nothing in this file submits anything -- it attaches to $HOLDER.
#
#   HOLDER=<jobid> [NODE=nid010188] [TAG=fpdepth] \
#   ODE_CKPT=<ode student .pt> [STUDENT_CKPT=<phase-3 snapshot .pt>] \
#     bash sbatch/run_fp_depth_study.sh
#
# The study rolls the student to depth 18 on >=64 windows and fingerprints
# GT / rollout / variance-matched-GT at each depth, so it is NOT a smoke:
# budget roughly (windows x depths x 5 fingerprints x FP_SEEDS) teacher
# forwards plus one 18-chunk AR rollout per window. Start with
# NUM_WINDOWS=8 to time it before committing to 64.
#
# Judge by the VERDICT block the script prints, and read the PROBE SETTINGS
# block above it: raw s is measured against an oblivious-teacher floor that
# moves with the perturbation mode, its scale AND the timestep, so it is
# ORDINAL ONLY within one setting. The script refuses to pool settings. It states its criterion
# before the numbers and prints
#   PROBE CANNOT RANK OFF-MANIFOLD DISTANCE -- GATE NOT VIABLE
# on a null, and
#   PROBE READS VARIANCE CONTRACTION, NOT OFF-MANIFOLD DISTANCE
# if the variance-matched control reproduces the effect. Either is a real
# answer; neither is a failed run.
set -euo pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM
: "${HOLDER:?set HOLDER to a running holder job ID}"
: "${ODE_CKPT:?set ODE_CKPT to the ODE-distilled student the DMD stage inits from}"

NODE=${NODE:-nid010188}
TAG=${TAG:-fpdepth}
RUNSTAMP=${RUNSTAMP:-$(date +%H%M%S)}
TEACHER_CKPT=${TEACHER_CKPT:-/scratch/u6ex/as1748.u6ex/ARRWM/logs/v14e_pca8_raw/causal_lora_step0005000.pt}
CONFIG=${CONFIG:-configs/action_forcing_phase3_dmd.yaml}
NUM_WINDOWS=${NUM_WINDOWS:-64}
DEPTHS=${DEPTHS:-"2 6 10 14 18"}
FP_SEEDS=${FP_SEEDS:-2}
# TIMESTEP SWEEP. t is the noise level the teacher is asked to restore the
# perturbation from, and it was CHOSEN, NOT MEASURED -- the single biggest
# unswept design choice here. The campaign has already been burnt once by a
# timestep artefact (the "35-step collapse" that was flat over 800 steps).
# Each value is measured from the SAME draws at the SAME points and reported
# SEPARATELY against its own oblivious-teacher floor, with a CROSS-SETTING
# block at the end. Cost scales linearly; set PROBE_TS=500 for a single-t run.
PROBE_TS=${PROBE_TS:-"250 500 750"}
PERTURB=${PERTURB:-hf_scramble}
RIDE_SCAN=${RIDE_SCAN:-600}

OUT=logs/fp_depth_${TAG}_h${HOLDER}_${RUNSTAMP}.jsonl
ERR=logs/fp_depth_${TAG}_h${HOLDER}_${RUNSTAMP}.err

source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR HF_HUB_CACHE=$CACHE_DIR
export HUGGINGFACE_HUB_CACHE=$CACHE_DIR TRANSFORMERS_CACHE=$CACHE_DIR
# 14e lineage: LOAD-BEARING. Without it zarr_dataset defaults to ss_vae and
# the study encodes a DIFFERENT physical action than the teacher was
# trained on -- silent z2/z7 corruption, not an error.
export ARRWM_ACTION_ENCODER=pca_raw
# TMPDIR hygiene: an agent session's TMPDIR (/local/user/<id>) does not
# exist on the compute nodes and dies at rank init. Pin it.
export TMPDIR=/tmp
unset LOCALDIR APPTAINER_CACHEDIR
export OMP_NUM_THREADS=8
export PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Per-TAG rendezvous port so two studies on the same holder cannot collide.
# Single process, but the port is still reserved for the pattern's sake and
# so a torchrun-wrapped variant of this file needs no second convention.
_TAGHASH=$(printf '%s' "$TAG" | cksum | cut -d' ' -f1)
export MASTER_ADDR=$NODE
export MASTER_PORT=$((29500 + (HOLDER + _TAGHASH) % 16000))

test -f "$ODE_CKPT"
test -f "$TEACHER_CKPT"
mkdir -p logs

echo "[fp-depth] holder=$HOLDER node=$NODE tag=$TAG port=$MASTER_PORT"
echo "[fp-depth] windows=$NUM_WINDOWS depths=[$DEPTHS] out=$OUT"

STUDENT_ARG=()
[ -n "${STUDENT_CKPT:-}" ] && STUDENT_ARG=(--student-ckpt "$STUDENT_CKPT")

srun --jobid="$HOLDER" --overlap --nodelist="$NODE" --nodes=1 \
     --ntasks-per-node=1 --gpus-per-node=1 --gpu-bind=none \
     --export=ALL,CUDA_VISIBLE_DEVICES=0 \
  python analysis/dmd_fp_depth_study.py \
    --config "$CONFIG" \
    --ode-ckpt "$ODE_CKPT" \
    --teacher-ckpt "$TEACHER_CKPT" \
    "${STUDENT_ARG[@]}" \
    --encoded-root /projects/u6ex/fbots/frodobots_encoded_weunz \
    --caption-root /projects/u6ex/fbots/frodobots_captions/train \
    --motion-root /projects/u6ex/fbots/frodobots_motion \
    --ss-vae-checkpoint action_query/checkpoints/ss_vae_8free.pt \
    --action-dims 0 1 \
    --denoising-step-list 1000 625 357.142857 208.333333 \
    --ride-scan-limit "$RIDE_SCAN" \
    --num-windows "$NUM_WINDOWS" \
    --depths $DEPTHS \
    --teacher-frames 21 \
    --ar-initial-chunks 3 \
    --cache-chunks 7 \
    --probe-timestep $PROBE_TS \
    --perturb "$PERTURB" \
    --fp-seeds "$FP_SEEDS" \
    --auc-min 0.8 \
    --deep-depth-min 10 \
    --verdict-metric s \
    --seed 1234 \
    --out "$OUT" \
    2>&1 | tee "$ERR"

echo "[fp-depth] done. records=$OUT  log=$ERR"
echo "[fp-depth] re-report without a GPU:"
echo "  python analysis/dmd_fp_depth_study.py --report-only $OUT"
