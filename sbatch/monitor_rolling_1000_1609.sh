#!/usr/bin/env bash
# Cancel only these three 2026-09-16 rolling allocations, and only after
# their own full step-1000 checkpoint has finished writing.
set -u

arr=/scratch/u6qf/as1748.u6qf/ARRWM_straggler/ARRWM
state="$arr/logs/rolling1000_monitor_1609"
mkdir -p "$state"
exec >>"$state/events.log" 2>&1
printf '%s monitor start pid=%s\n' "$(date -u +%FT%TZ)" "$$"

targets=(
  '6590718|phase3_rolling_definitive_4n_basev2_stat_mean_only_gbs16_1609'
  '6591554|phase3_rolling_definitive_4n_basev2_no_carn_gbs16_1609'
  '6593747|phase3_rolling_definitive_4n_basev2_stat_nonmean_only_gbs16_1609'
)

while :; do
  remaining=0
  for target in "${targets[@]}"; do
    IFS='|' read -r job arm <<<"$target"
    marker="$state/$job.done"
    [[ -e "$marker" ]] && continue
    remaining=$((remaining + 1))
    run="$arr/logs/dmd10k_${arm}/dmd10k_${arm}_j${job}"
    ckpt="$run/phase1_step0001000.pt"
    if [[ -f "$ckpt" && ! -L "$ckpt" ]]; then
      bytes1=$(stat -c %s "$ckpt" 2>/dev/null || printf 0)
      age=$(( $(date +%s) - $(stat -c %Y "$ckpt" 2>/dev/null || date +%s) ))
      if (( bytes1 >= 17000000000 && age >= 60 )); then
        sleep 20
        bytes2=$(stat -c %s "$ckpt" 2>/dev/null || printf 0)
        if [[ "$bytes1" == "$bytes2" ]]; then
          printf '%s checkpoint ready job=%s arm=%s bytes=%s\n' "$(date -u +%FT%TZ)" "$job" "$arm" "$bytes2"
          if squeue -h -j "$job" | grep -q .; then
            scancel "$job"
            printf '%s cancelled job=%s after checkpoint=%s\n' "$(date -u +%FT%TZ)" "$job" "$ckpt"
          else
            printf '%s already stopped job=%s checkpoint=%s\n' "$(date -u +%FT%TZ)" "$job" "$ckpt"
          fi
          printf '%s\n' "$ckpt" >"$marker"
          remaining=$((remaining - 1))
          continue
        fi
      fi
    fi
    # Record an unexpected stop without mistaking it for a successful save.
    if ! squeue -h -j "$job" | grep -q .; then
      printf '%s ALERT job=%s left queue without stable step-1000 checkpoint\n' "$(date -u +%FT%TZ)" "$job"
      printf 'missing %s\n' "$ckpt" >"$state/$job.stopped_without_1000"
      printf '%s\n' "$ckpt" >"$marker"
      remaining=$((remaining - 1))
    fi
  done
  (( remaining == 0 )) && break
  sleep 30
done
printf '%s monitor complete\n' "$(date -u +%FT%TZ)"
