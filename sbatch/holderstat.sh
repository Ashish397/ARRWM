#!/bin/bash
# Holder status derived from SLURM, never inferred from log presence.
# A log file proves a job ONCE EXISTED, not that it is running -- a stale log
# from a dead job reads identically to a live one. Liveness comes from a
# non-.batch job step; the log is only consulted to NAME what is live.
cd /scratch/u6ex/as1748.u6ex/ARRWM
out="[$(date +%H:%M)]"
for jid in $(squeue -u "$USER" -h -o "%i" 2>/dev/null | sort); do
  st=$(squeue -j "$jid" -h -o "%T" 2>/dev/null)
  tl=$(squeue -j "$jid" -h -o "%L" 2>/dev/null)
  if [ "$st" != "RUNNING" ]; then out="$out | $jid:${st:0:4}/$tl"; continue; fi
  live=$(squeue -j "$jid" -h -s -o "%i" 2>/dev/null | grep -vE '\.(batch|extern)$' | wc -l) || true
  if [ "${live:-0}" -eq 0 ]; then
    busy=""; [ -f "logs/.holder_running_$jid.sh" ] && busy="(cmd running, no srun)"
    out="$out | $jid:IDLE$busy/$tl"; continue
  fi
  # something IS live -- name it from the most recently WRITTEN log
  f=$(ls -t logs/holdersmoke_*_h${jid}.log 2>/dev/null | head -1)
  nm="?"; prog=""
  if [ -n "$f" ]; then
    nm=$(basename "$f" | sed 's/holdersmoke_//;s/_h[0-9]*\.log//')
    prog=$(grep -oE 'step=[0-9]+/[0-9]+' "$f" 2>/dev/null | tail -1)
    # stale-log guard: if the log has not been touched in 5 min but a step is
    # live, say so rather than reporting a number that may be from a dead run
    if [ -n "$(find "$f" -mmin +5 2>/dev/null)" ]; then prog="${prog:-?}(log stale)"; fi
  fi
  out="$out | $jid:$nm ${prog:-running}/$tl"
done
echo "$out"
