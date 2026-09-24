#!/bin/bash
# Distribute all corrected-context DMD/ODE tasks over currently idle holders.
set -euo pipefail

R=${ARRWM_REMOTE_ROOT:-/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler}
A=${ARRWM_CODE_ROOT:-$R/ARRWM}
P=${PANEL32_STAGE:-$R/panel32_v3_stage}
L=$R/panel32_stage/logs/holders
STATE=$P/dispatch/arrwm_delta
mkdir -p "$STATE"
test -f "$P/logs/seed_setup/COMPLETE"

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
test "${#holders[@]}" -gt 0

tasks=()
for arm in recoverybase nocarn nocommit noaux nogan meanenergy vartv; do
  for action in F FR R BR B BL L FL N; do tasks+=("ours:$arm:$action"); done
done
for arm in local_kl pointwise_mse; do
  for action in F FR R BR B BL L FL N; do tasks+=("ode:$arm:$action"); done
done

for holder in "${holders[@]}"; do
  tmp="$L/.holder_cmd_${holder}.tmp.$$"
  printf '%s\n' '#!/bin/bash' 'set -euo pipefail' \
    "export PANEL32_STAGE='$P'" >"$tmp"
done
for index in "${!tasks[@]}"; do
  holder=${holders[$((index % ${#holders[@]}))]}
  IFS=: read -r kind arm action <<<"${tasks[$index]}"
  printf "bash '%s/sbatch/u6qf/panel32_v3_arrwm_delta_task.sh' '%s' '%s' '%s'\n" \
    "$A" "$kind" "$arm" "$action" >>"$L/.holder_cmd_${holder}.tmp.$$"
done
for holder in "${holders[@]}"; do
  tmp="$L/.holder_cmd_${holder}.tmp.$$"
  chmod +x "$tmp"
  mv "$tmp" "$L/.holder_cmd_${holder}.sh"
done

printf '%s holders=%s tasks=%s\n' \
  "$(date -Is)" "${#holders[@]}" "${#tasks[@]}" | tee "$STATE/dispatch.log"

