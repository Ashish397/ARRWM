#!/bin/bash
# PIXDINO SMOKE + pix_gan_weight CALIBRATION PROBE.
#
#   HOLDER=<jobid> ARM=pixdino_online PORTOFF=3 STEPS=25 PIXW=0.0 \
#       bash sbatch/run_pixdino_smoke.sh
#
# Normally invoked by writing a one-liner that calls this into
# ``logs/.holder_cmd_<jobid>.sh``; the holder polls for that file and runs
# it AS the batch job, which slurmctld owns. Nothing agent-side is
# load-bearing (an srun client launched from the agent's shell gets
# SIGTERM'd when the turn ends -- observed).
#
# WHAT IT ANSWERS, in one 25-step run:
#   1. does it BOOT (encoder built, decode callback installed, param
#      groups split);
#   2. do the CONNECTIONS fire -- and specifically does
#      ``ladd_pix_wan_projector_calls`` stay at 0, which is the whole
#      claim of the feature;
#   3. MEMORY peak WITH the step it occurred at, and s/step;
#   4. NaN check;
#   5. the CALIBRATION READ. Run this at PIXW=0.0 and take
#      ``surrogate_grad_ratio_unweighted`` -- the share the surrogate
#      term WOULD have at weight 1.0. At weight 0 the applied term has no
#      gradient at all, so this is the ONLY key that can answer, and the
#      real arms' PIXW = 0.10 / r puts it in the 5-20 % band.
set -uo pipefail
: "${HOLDER:?set HOLDER (the holder jobid)}"
ARM=${ARM:-pixdino_online}
PORTOFF=${PORTOFF:-0}
STEPS=${STEPS:-25}
PIXW=${PIXW:-0.0}
SRC=sbatch/${ARM}.sbatch
test -f "$SRC" || { echo "no such arm: $SRC"; exit 2; }

cd /scratch/u6ex/as1748.u6ex/ARRWM
NODES=$(scontrol show hostnames "$(squeue -j "$HOLDER" -h -o %N)")
NNODE=$(echo "$NODES" | wc -l); NODELIST=$(echo "$NODES" | paste -sd,)
MASTER_ADDR=$(echo "$NODES" | head -1)
MASTER_PORT=$((29500 + (HOLDER + PORTOFF) % 16000))
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=logs/pixdino_smoke_${ARM}_h${HOLDER}_${STAMP}.log

# Same launcher swap as sbatch/run_smoke_on_holder.sh: --overlap is
# required to share the allocation, and the rdzv id/port must be unique
# per concurrent run or two smokes collide on one rendezvous.
GEN=/tmp/pixdino_${ARM}_${HOLDER}_${PORTOFF}.sh
sed -e 's/^srun torchrun \\/srun --jobid='"$HOLDER"' --overlap --nodelist='"$NODELIST"' --nodes='"$NNODE"' --ntasks-per-node=1 --gpus-per-node=4 --gpu-bind=none torchrun \\/' \
    -e 's/--nnodes=\$SLURM_NNODES/--nnodes='"$NNODE"'/' \
    -e 's/--rdzv_id=\$SLURM_JOB_ID/--rdzv_id='"${HOLDER}${PORTOFF}"'/' \
    -e 's|--rdzv_endpoint=\${MASTER_ADDR}:\${MASTER_PORT}|--rdzv_endpoint='"$MASTER_ADDR:$MASTER_PORT"'|' \
    -e 's/^#SBATCH.*//' \
    "$SRC" > "$GEN"

echo "[PIXDINO-SMOKE] arm=$ARM holder=$HOLDER nodes=$NODELIST port=$MASTER_PORT"
echo "[PIXDINO-SMOKE] steps=$STEPS PIXW=$PIXW log=$LOG"
# ---------------------------------------------------------------------
# DEXTRA -- CORRECTED 2026-08-26. The first version was
#     "mem_snapshot_every=1 sample_interval=1000 holdout_eval_interval=1000"
# and THREE of those three were wrong. Recorded in full because the
# failure mode is the repo's endemic one and the lesson is not the typo:
#
#  1. ``sample_interval=1000`` -- REAL key, WRONG value, and the one that
#     did the damage. ``configs/action_forcing_phase1.yaml:100`` sets
#     ``sample_interval: 15`` and the arms inherit it; 1000 overrode it,
#     so a 90-step probe fired ONE sample event and emitted ONE mp4. A
#     TEXTURE experiment whose probe produces no images is not
#     observable -- the probe was measuring something nobody could look
#     at. Now a variable, defaulting to the arms' own 15.
#  2. ``mem_snapshot_every`` -- NO SUCH KEY. The real gate is
#     ``memory_audit_enabled`` (trainer ``_mem_step_snapshot``:17450).
#     So the memory instrumentation I added was itself a silent no-op,
#     and the absence of the ``6a_after_deferred_disc`` snapshot -- which
#     I nearly read as evidence about the deferred flush -- meant
#     nothing at all.
#  3. ``holdout_eval_interval`` -- NO SUCH KEY anywhere in the repo.
#     Dropped rather than guessed at.
#
# ``OmegaConf.from_dotlist`` CREATES unknown keys silently, so all three
# were accepted without a murmur. Anything added here must be grepped
# for a real read site first.
# ---------------------------------------------------------------------
# SAMPLE_EVERY: video cadence. Default 15 = the arms' own value, so a
# probe is visually comparable to the arm it is probing. Raise it only
# deliberately, and never to a value larger than STEPS.
SAMPLE_EVERY=${SAMPLE_EVERY:-15}
if [ "$SAMPLE_EVERY" -gt "$STEPS" ]; then
  echo "[PIXDINO-SMOKE] WARNING: SAMPLE_EVERY=$SAMPLE_EVERY > STEPS=$STEPS" \
       "-- this probe would emit NO videos. Clamping to $((STEPS / 4 + 1))." >&2
  SAMPLE_EVERY=$((STEPS / 4 + 1))
fi
# TELEM=1 (set below): the calibration probe fires EVERY step, so the
# ratio comes from a distribution rather than one sample -- a single
# sample cannot distinguish "small" from "noisy".
# memory_audit_enabled=true: the REAL memory-audit gate, so the peak is
# attributable to a labelled point inside the step.
echo "[PIXDINO-SMOKE] sample_interval=$SAMPLE_EVERY (videos ON)"
SLURM_JOB_ID=$HOLDER SLURM_NNODES=$NNODE SLURM_JOB_NODELIST=$NODELIST \
  MAXSTEPS=$STEPS CKPT_EVERY=1000 PIXW=$PIXW TELEM=1 \
  DEXTRA="memory_audit_enabled=true sample_interval=$SAMPLE_EVERY" \
  bash "$GEN" > "$LOG" 2>&1
RC=$?
echo "[PIXDINO-SMOKE] training exit=$RC"
bash sbatch/check_pixdino_smoke.sh "$LOG"
exit $RC
