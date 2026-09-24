#!/bin/bash
# Dispatch the four-context YUME v3 delta, one action per idle holder.
set -euo pipefail

R=${ARRWM_REMOTE_ROOT:-/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler}
A=${ARRWM_CODE_ROOT:-$R/ARRWM}
P=${PANEL32_STAGE:-$R/panel32_v3_stage}
L=$R/panel32_stage/logs/holders
STATE=$P/dispatch/yume_delta
mkdir -p "$STATE"
test -s "$P/reuse_yume_v2_to_v3.json"

mapfile -t candidates < <(
  awk '/PANEL32_HOLDER_READY/ {
         for (i=1; i<=NF; i++) if ($i ~ /^job=/) {
           sub(/^job=/, "", $i); print $i
         }
       }' "$L"/6841741_*.out 2>/dev/null | sort -nu
)
holders=()
for holder in "${candidates[@]}"; do
  squeue -j "$holder" -h -t R -o '%i' | grep -q . || continue
  test ! -e "$L/.holder_cmd_${holder}.sh" || continue
  test ! -e "$L/.holder_running_${holder}.sh" || continue
  holders+=("$holder")
done
test "${#holders[@]}" -ge 9

actions=(F FR R BR B BL L FL N)
for index in "${!actions[@]}"; do
  holder=${holders[$index]}
  action=${actions[$index]}
  tmp="$L/.holder_cmd_${holder}.tmp.$$"
  printf '%s\n' '#!/bin/bash' 'set -euo pipefail' \
    "export PANEL32_STAGE='$P'" \
    "export PANEL32_MANIFEST='$A/grids/eval/panel32_locked_v3.json'" \
    "export PANEL32_PROVENANCE='$P/sources/panel32_source_provenance.json'" \
    "bash '$A/sbatch/u6qf/panel32_holder_sequence.sh' 'yume:$action'" >"$tmp"
  chmod +x "$tmp"
  mv "$tmp" "$L/.holder_cmd_${holder}.sh"
done

printf '%s holders=9 tasks=9\n' "$(date -Is)" | tee "$STATE/dispatch.log"

