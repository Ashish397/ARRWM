#!/bin/bash
# Run a smk_*.sbatch's training command on an ALREADY-ALLOCATED holder.
#   HOLDER=<jobid> SMOKE=<tag> PORTOFF=<int> bash sbatch/run_smoke_on_holder.sh
# Reuses the smoke script verbatim by overriding only the srun launcher bits:
# --overlap is required to share the allocation; the rdzv port must be unique
# per concurrent smoke or two holders collide on the same rendezvous.
set -euo pipefail
: "${HOLDER:?set HOLDER}"; : "${SMOKE:?set SMOKE (e.g. v6, v6b, amfr, 6allon)}"
PORTOFF=${PORTOFF:-0}
SRC=sbatch/${SMOKE}.sbatch; [ -f "$SRC" ] || SRC=sbatch/smk_${SMOKE}.sbatch
test -f "$SRC" || { echo "no such smoke: $SRC"; exit 2; }
# ===================== COLLISION GUARD (2026-08-26) =====================
# REFUSE to launch into a holder that already has a live job step.
#
# WHY THIS EXISTS. The holder loop runs ONE command at a time: it moves
# logs/.holder_cmd_<jobid>.sh to logs/.holder_running_<jobid>.sh and runs
# it. Writing a NEW .holder_cmd_ while a command is executing does NOT
# queue safely -- the loop fires it as soon as the file appears on a poll
# where the previous job is still alive on the nodes, two torchruns then
# contend for the same allocation, and BOTH die on
# RendezvousConnectionError. Measured 2026-08-26: this destroyed a matched
# baseline arm (carncommit_off on 6144623) which logged ZERO training
# steps, while its treatment partner survived -- leaving a paired smoke
# that was no longer an experiment. The surviving half looked healthy.
#
# NOTE the guard canNOT test for .holder_running_<jobid>.sh: by the time
# this script runs, the loop has ALREADY created that file for US. It must
# ask slurm what STEPS are live instead. The holder itself is the .batch
# (and .extern) step; anything else is somebody's running job.
#
# HOLDER_FORCE=1 overrides, for the case where you have positively
# confirmed the other step is dead. Do not set it habitually.
# The `|| true` is LOAD-BEARING, do not remove it. This script runs under
# `set -euo pipefail`. On a CLEAN holder every line is .batch/.extern, so
# `grep -v` matches nothing and exits 1; pipefail propagates that as the
# pipeline status even though `wc -l` succeeded, and `set -e` then aborts
# the whole script. Without `|| true` this guard fails CLOSED on an idle
# holder -- it kills every launch it is supposed to permit, silently, with
# exit 1 and no output. Measured: it blocked the carncommit_off relaunch at
# 17:59:02 and would have blocked every smoke on every holder.
_LIVE=$(squeue -j "$HOLDER" -h -s -o "%i" 2>/dev/null | grep -vE '\.(batch|extern)$' | wc -l) || true
if [ "${_LIVE:-0}" -gt 0 ] && [ "${HOLDER_FORCE:-0}" != "1" ]; then
  echo "[HOLDERSMOKE] REFUSING to launch: holder $HOLDER already has ${_LIVE} live job step(s)." >&2
  squeue -j "$HOLDER" -s -o "%.18i %.30j %.10M %N" >&2
  echo "[HOLDERSMOKE] Another job is on these nodes. Launching now would kill BOTH" >&2
  echo "[HOLDERSMOKE] via RendezvousConnectionError (see the comment in this script)." >&2
  echo "[HOLDERSMOKE] Wait for it to finish, or pick a genuinely free holder." >&2
  echo "[HOLDERSMOKE] Override with HOLDER_FORCE=1 ONLY if you have confirmed it is dead." >&2
  exit 3
fi
# =======================================================================
NODES=$(scontrol show hostnames "$(squeue -j "$HOLDER" -h -o %N)")
NNODE=$(echo "$NODES" | wc -l); NODELIST=$(echo "$NODES" | paste -sd,)
MASTER_ADDR=$(echo "$NODES" | head -1)
MASTER_PORT=$((29500 + (HOLDER + PORTOFF) % 16000))
LOG=logs/holdersmoke_${SMOKE}_h${HOLDER}.log
# Take the script's own env + override block, but swap its launcher line.
sed -e 's/^srun torchrun \\/srun --jobid='"$HOLDER"' --overlap --nodelist='"$NODELIST"' --nodes='"$NNODE"' --ntasks-per-node=1 --gpus-per-node=4 --gpu-bind=none torchrun \\/' \
    -e 's/--nnodes=\$SLURM_NNODES/--nnodes='"$NNODE"'/' \
    -e 's/--rdzv_id=\$SLURM_JOB_ID/--rdzv_id='"$HOLDER$PORTOFF"'/' \
    -e 's|--rdzv_endpoint=\${MASTER_ADDR}:\${MASTER_PORT}|--rdzv_endpoint='"$MASTER_ADDR:$MASTER_PORT"'|' \
    -e 's/^#SBATCH.*//' \
    "$SRC" > /tmp/holdersmoke_${SMOKE}_${HOLDER}.sh
echo "[HOLDERSMOKE] smoke=$SMOKE holder=$HOLDER nodes=$NODELIST port=$MASTER_PORT log=$LOG"
SLURM_JOB_ID=$HOLDER SLURM_NNODES=$NNODE SLURM_JOB_NODELIST=$NODELIST \
  bash /tmp/holdersmoke_${SMOKE}_${HOLDER}.sh > "$LOG" 2>&1
echo "[HOLDERSMOKE] exit=$? log=$LOG"
