#!/bin/bash
# One dispatch pass (run every minute from the workstation). Arms still waiting on a PENDING holder are moved
# onto any RUNNING holder that is idle (no command queued or running); the pending holder, which holds no node,
# is then cancelled. Running holders are never cancelled. State: $L/.dispatch_map (arm jobid per line).
R=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler; L=$R/_logs/ours30s; T=$R/ARRWM/sbatch/u6qf/fleet_cmd_template_v3.sh
M=$L/.dispatch_map
declare -A ST; while read -r j s; do ST[$j]=$s; done < <(squeue -u $USER -h -o "%i %t")
idle=(); for j in "${!ST[@]}"; do [ "${ST[$j]}" = R ] || continue; [ -e $L/.holder_cmd_$j.sh ] || [ -e $L/.holder_running_$j.sh ] || idle+=($j); done
echo "$(date -u +%H:%M) states: $(for j in "${!ST[@]}"; do echo -n "$j=${ST[$j]} "; done) idle: ${idle[*]:-none}"
while read -r arm j; do
  [ "${ST[$j]:-gone}" = PD ] || continue
  [ ${#idle[@]} -gt 0 ] || break
  mv $L/.holder_cmd_$j.sh $L/.dispatch_stage_$arm.sh 2>/dev/null || continue
  h=${idle[0]}; idle=("${idle[@]:1}")
  sed -e "s|__ARMS__|$arm|" -e "s|__WINS__||" $T > $L/.holder_cmd_$h.sh
  sed -i "s|^$arm .*|$arm $h|" $M
  s=$(squeue -j $j -h -o %t 2>/dev/null)
  if [ "$s" = PD ]; then scancel $j; echo "moved $arm: pending $j cancelled -> running holder $h"; else echo "moved $arm -> $h; $j now '$s' (left alone, will idle)"; fi
  rm -f $L/.dispatch_stage_$arm.sh
done < $M
