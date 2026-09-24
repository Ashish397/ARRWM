#!/bin/bash
# Generate one panel32 action with four one-GPU ARRWM workers in a holder.
set -euo pipefail

ACTION=${1:?usage: ours_panel32_holder_action.sh ACTION PANEL_MANIFEST SOURCE_PROVENANCE BUNDLE_INDEX}
PANEL_MANIFEST=${2:?usage: ours_panel32_holder_action.sh ACTION PANEL_MANIFEST SOURCE_PROVENANCE BUNDLE_INDEX}
SOURCE_PROVENANCE=${3:?usage: ours_panel32_holder_action.sh ACTION PANEL_MANIFEST SOURCE_PROVENANCE BUNDLE_INDEX}
BUNDLE_INDEX=${4:?usage: ours_panel32_holder_action.sh ACTION PANEL_MANIFEST SOURCE_PROVENANCE BUNDLE_INDEX}
case "$ACTION" in F|FR|R|BR|B|BL|L|FL|N) ;; *) echo "bad action: $ACTION" >&2; exit 2 ;; esac

R=${ARRWM_REMOTE_ROOT:-/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler}
A=${ARRWM_CODE_ROOT:-$R/ARRWM}
PY=${ARRWM_PYTHON:-$R/miniforge3/envs/arrwm/bin/python}
PANEL_ROOT=${PANEL32_STAGE:-$R/panel32_stage}
OUT=${OURS_PANEL32_OUT:-$PANEL_ROOT/fleet30s/ours}
CKPT=${OURS_PANEL32_CKPT:-$R/ckpts/recovery_base/phase1_step0001000.pt.lean.pt}
TAG=${OURS_PANEL32_TAG:-ours}
MODEL_LABEL=${OURS_PANEL32_MODEL_LABEL:-ARRWM four-step DMD}
CARN_COMMIT=${OURS_PANEL32_CARN_COMMIT:-auto}
WAN=${OURS_WAN_MODEL_ROOT:-$R/frodobots/Wan2.1-T2V-1.3B}
LOG=${OURS_PANEL32_LOG:-$PANEL_ROOT/logs/ours/$ACTION}
mkdir -p "$OUT" "$LOG"

mkdir -p "$PANEL_ROOT/locks"
exec 7>"$PANEL_ROOT/locks/generate_ours_${TAG}_${ACTION}.lock"
flock 7

export PATH="$R/miniforge3/envs/arrwm/bin:$PATH"
export PYTHONPATH="$A${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${HF_HOME:-$R/frodobots/hf_cache}
export TORCH_HOME=${TORCH_HOME:-$R/torch_home}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

test -x "$PY"
test -f "$PANEL_MANIFEST"
test -f "$SOURCE_PROVENANCE"
test -f "$BUNDLE_INDEX"
test -f "$CKPT"
mapfile -t UUIDS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits)
test "${#UUIDS[@]}" -eq 4
test "$(printf '%s\n' "${UUIDS[@]}" | sort -u | wc -l)" -eq 4

pids=()
read -r -a CONTEXT_SHARDS <<<"${PANEL32_CONTEXT_SHARDS:-0 1 2 3}"
CONTEXT_SHARD_COUNT=${PANEL32_CONTEXT_SHARD_COUNT:-4}
test "${#CONTEXT_SHARDS[@]}" -eq 4
for gpu in 0 1 2 3; do
  shard=${CONTEXT_SHARDS[$gpu]}
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PY" "$A/interactive/external_models/ours_panel32.py" \
      --panel-manifest "$PANEL_MANIFEST" \
      --source-provenance "$SOURCE_PROVENANCE" \
      --bundle-index "$BUNDLE_INDEX" \
      --checkpoint "$CKPT" --output "$OUT" --action "$ACTION" \
      --tag "$TAG" --model-label "$MODEL_LABEL" \
      --carn-commit "$CARN_COMMIT" \
      --shard-index "$shard" --shard-count "$CONTEXT_SHARD_COUNT" \
      --wan-model-root "$WAN"
  ) >"$LOG/gpu${gpu}.out" 2>"$LOG/gpu${gpu}.err" &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
if [ "$rc" -ne 0 ]; then
  tail -100 "$LOG"/gpu*.err >&2 || true
  exit 1
fi
touch "$LOG/COMPLETE"
echo "OURS_PANEL32_ACTION_COMPLETE action=$ACTION context_shards=${CONTEXT_SHARDS[*]} $(date -Is)"
