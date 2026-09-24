#!/bin/bash
# Run inside one Slurm task that owns all four GPUs and all 64 CPUs on a node.
# Slurm's per-task GPU binding on u6qf can alias physical devices, so this
# parent performs physical-UUID validation once and then assigns local GPU
# ordinals explicitly to four child workers.
set -euo pipefail

MODEL=${1:?usage: frodo8_node_local.sh MODEL}
case "$MODEL" in
  lingbot|dreamx|matrixgame2|minwm|minwm_ode) ;;
  *) echo "unsupported Frodo8 model: $MODEL" >&2; exit 64 ;;
esac

R=${ARRWM_REMOTE_ROOT:-/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler}
A=${ARRWM_CODE_ROOT:-$R/ARRWM}
S=${ALIGNED32_STAGE:-$R/aligned32_stage}
PY=${ARRWM_PYTHON:-$R/miniforge3/envs/arrwm/bin/python}
RUN_ID=${FRODO8_RUN_ID:?FRODO8_RUN_ID must be exported by frodo8_on_holder.sh}
LOG_ROOT=${FRODO8_LOG_ROOT:-$S/logs/frodo8/$RUN_ID/$MODEL}
mkdir -p "$LOG_ROOT"

# The encompassing srun owns every GPU on the node.  Rebuild a deterministic
# local 0..3 view, then prove that each isolated child resolves to a distinct
# physical UUID before any model process is started.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
UUID_FILE="$LOG_ROOT/gpu_uuid_preflight.tsv"
: >"$UUID_FILE"
uuids=()
for gpu in 0 1 2 3; do
  uuid=$(CUDA_VISIBLE_DEVICES="$gpu" "$PY" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
props = torch.cuda.get_device_properties(0)
uuid = str(getattr(props, "uuid", ""))
assert uuid and uuid.lower() != "none", props
print(uuid)
PY
)
  uuids+=("$uuid")
  printf '%s\t%s\n' "$gpu" "$uuid" >>"$UUID_FILE"
done
unique=$(printf '%s\n' "${uuids[@]}" | sort -u | wc -l)
if [ "$unique" -ne 4 ]; then
  echo "FATAL: local CUDA ordinals do not resolve to four unique physical GPUs" >&2
  cat "$UUID_FILE" >&2
  exit 74
fi
echo "FRODO8_GPU_UUID_PREFLIGHT_PASS model=$MODEL $(tr '\n' ' ' <"$UUID_FILE")"

pids=()
for shard in 0 1 2 3; do
  (
    export FRODO8_SHARD="$shard"
    export CUDA_VISIBLE_DEVICES="$shard"
    bash "$A/sbatch/u6qf/frodo8_model_worker.sh" "$MODEL"
  ) >"$LOG_ROOT/node_worker_${shard}.out" 2>"$LOG_ROOT/node_worker_${shard}.err" &
  pids+=("$!")
done

rc=0
for pid in "${pids[@]}"; do
  wait "$pid" || rc=1
done
if [ "$rc" -ne 0 ]; then
  echo "one or more Frodo8 node-local workers failed" >&2
  tail -100 "$LOG_ROOT"/node_worker_*.err >&2 || true
  exit 70
fi
echo "FRODO8_NODE_LOCAL_COMPLETE model=$MODEL workers=4 $(date -Is)"
