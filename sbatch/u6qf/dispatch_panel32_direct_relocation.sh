#!/bin/bash
# Run the exact direct-relocation calculation on all 30 retained holders.
set -euo pipefail

R=${ARRWM_REMOTE_ROOT:-/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler}
CODE=${ARRWM_CODE_ROOT:-$R/ARRWM}
P=${PANEL32_STAGE_ROOT:-$R/panel32_v2_stage}
OUT=${PANEL32_EVAL_OUT:-$P/eval_final}
HOLDER_LOGS="$R/panel32_stage/logs/holders"
SHARDS=${PANEL32_DIRECT_RELOCATION_SHARDS:-30}
case "$SHARDS" in *[!0-9]*|'') echo "invalid shard count: $SHARDS" >&2; exit 2 ;; esac
if [ "$SHARDS" -lt 1 ]; then
  echo "shard count must be positive" >&2
  exit 2
fi
mkdir -p "$OUT/direct_relocation_shards/logs"

mapfile -t running < <(squeue -h -u "$(id -un)" -t R -n panel32-hold -o '%A' | sort -nu)
if [ "${#running[@]}" -ne "$SHARDS" ]; then
  echo "expected exactly $SHARDS retained running holders, found ${#running[@]}" >&2
  exit 1
fi

for shard in "${!running[@]}"; do
  holder=${running[$shard]}
  command="$HOLDER_LOGS/.holder_cmd_${holder}.sh"
  active="$HOLDER_LOGS/.holder_running_${holder}.sh"
  test ! -e "$command" || { echo "holder $holder already has queued command" >&2; exit 1; }
  test ! -e "$active" || { echo "holder $holder already executing command" >&2; exit 1; }
  temporary="$HOLDER_LOGS/.holder_cmd_${holder}.tmp.$$"
  cat >"$temporary" <<EOF
#!/bin/bash
set -euo pipefail
export PANEL32_STAGE_ROOT='$P'
export PANEL32_EVAL_OUT='$OUT'
export PANEL32_EVAL_CONFIG='$P/panel32_eval_config.json'
export PANEL32_LOCKED_MANIFEST='$CODE/grids/eval/panel32_locked_v2.json'
source '$CODE/sbatch/u6qf/panel32_eval_env.sh'
export OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32
cd '$CODE'
"\$PY" grids/eval/panel32_direct_relocation_shard.py \
  --out '$OUT' --shard-index '$shard' --shard-count '$SHARDS' \
  >'$OUT/direct_relocation_shards/logs/shard_${shard}.log' 2>&1
EOF
  chmod +x "$temporary"
  mv "$temporary" "$command"
  printf '%s holder=%s shard=%s/%s\n' "$(date -Is)" "$holder" "$shard" "$SHARDS"
done | tee "$OUT/direct_relocation_shards/dispatch.log"

echo "DIRECT_RELOCATION_DISPATCHED shards=$SHARDS $(date -Is)"
