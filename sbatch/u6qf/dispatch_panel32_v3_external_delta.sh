#!/bin/bash
# Distribute the 45 corrected-context external generation tasks over idle holders.
set -euo pipefail

R=${ARRWM_REMOTE_ROOT:-/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler}
A=${ARRWM_CODE_ROOT:-$R/ARRWM}
P=${PANEL32_STAGE:-$R/panel32_v3_stage}
L=$R/panel32_stage/logs/holders
STATE=$P/dispatch/external_delta
EXCLUDE_HOLDERS=${EXCLUDE_HOLDERS:-}
mkdir -p "$STATE"

mapfile -t candidates < <(
  awk '/PANEL32_HOLDER_READY/ {
         for (i=1; i<=NF; i++) if ($i ~ /^job=/) {
           sub(/^job=/, "", $i); print $i
         }
       }' "$L"/6841741_*.out 2>/dev/null | sort -nu
)
holders=()
for holder in "${candidates[@]}"; do
  [[ " $EXCLUDE_HOLDERS " == *" $holder "* ]] && continue
  squeue -j "$holder" -h -t R -o '%i' | grep -q . || continue
  test ! -e "$L/.holder_cmd_${holder}.sh" || continue
  test ! -e "$L/.holder_running_${holder}.sh" || continue
  holders+=("$holder")
done
test "${#holders[@]}" -gt 0

tasks=()
for model in lingbot dreamx matrixgame2 minwm minwm_ode; do
  for action in F FR R BR B BL L FL N; do
    tasks+=("$model:$action")
  done
done

# Build every command in a temporary file first; publish only after all tasks
# have been assigned exactly once.
for holder in "${holders[@]}"; do
  tmp="$L/.holder_cmd_${holder}.tmp.$$"
  printf '%s\n' '#!/bin/bash' 'set -euo pipefail' \
    "export PANEL32_STAGE='$P'" >"$tmp"
done
for index in "${!tasks[@]}"; do
  holder=${holders[$((index % ${#holders[@]}))]}
  IFS=: read -r model action <<<"${tasks[$index]}"
  printf "bash '%s/sbatch/u6qf/panel32_v3_external_delta_action.sh' '%s' '%s'\n" \
    "$A" "$model" "$action" >>"$L/.holder_cmd_${holder}.tmp.$$"
done
for holder in "${holders[@]}"; do
  tmp="$L/.holder_cmd_${holder}.tmp.$$"
  # Do not consume an allocation with an empty command.
  if [ "$(wc -l <"$tmp")" -le 3 ]; then
    rm "$tmp"
    continue
  fi
  chmod +x "$tmp"
  mv "$tmp" "$L/.holder_cmd_${holder}.sh"
done

printf '%s holders=%s tasks=%s exclude=%s\n' \
  "$(date -Is)" "${#holders[@]}" "${#tasks[@]}" "$EXCLUDE_HOLDERS" \
  | tee "$STATE/dispatch.log"
