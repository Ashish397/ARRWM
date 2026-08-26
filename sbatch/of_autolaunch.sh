#!/bin/bash
# Waits for ANY of my pending holders to start, then writes the OF smoke
# command into that holder's command-file loop. The holder (a batch job
# owned by slurmctld) executes it, so the run does NOT depend on the agent
# session staying alive -- the failure mode that killed earlier attempts.
#
# Peer-confirmed hazard this respects: relaunching onto nodes where a prior
# attempt is still draining causes foreign-GB OOM. We only ever write to a
# holder with NO running step, and the holder loop runs one command at a time.
cd /scratch/u6ex/as1748.u6ex/ARRWM
CANDIDATES="$*"
while true; do
  for J in $CANDIDATES; do
    ST=$(squeue -j "$J" -h -o "%T" 2>/dev/null)
    [ "$ST" = "RUNNING" ] || continue
    # only if the holder is idle (batch step only, no torchrun step)
    NSTEP=$(squeue -s -j "$J" -h 2>/dev/null | grep -vc "\.batch")
    [ "$NSTEP" -eq 0 ] || continue
    [ -f "logs/.holder_cmd_$J.sh" ] && continue
    [ -f "logs/.holder_running_$J.sh" ] && continue
    cat > "logs/.holder_cmd_$J.sh" <<INNER
cd /scratch/u6ex/as1748.u6ex/ARRWM
HOLDER=$J PORTOFF=7811 RUNSTAMP=auto MAXSTEPS=60 \
  OF_EXTRA="dmd_loss_start_step=0 dmd_loss_warmup_steps=5" \
  bash sbatch/run_of_smoke.sh
INNER
    echo "$(date) armed OF smoke on holder $J"
    exit 0
  done
  sleep 30
done
