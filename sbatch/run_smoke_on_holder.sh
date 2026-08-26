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
