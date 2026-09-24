#!/bin/bash
# Generate one mixed-panel action for one ODE arm with four one-GPU workers.
# Payload only: run inside an existing four-GPU Isambard holder allocation.
set -euo pipefail

ARM=${1:?usage: ode_panel32_holder_action.sh ARM ACTION PANEL_MANIFEST SOURCE_PROVENANCE BUNDLE_INDEX}
ACTION=${2:?usage: ode_panel32_holder_action.sh ARM ACTION PANEL_MANIFEST SOURCE_PROVENANCE BUNDLE_INDEX}
PANEL_MANIFEST=${3:?usage: ode_panel32_holder_action.sh ARM ACTION PANEL_MANIFEST SOURCE_PROVENANCE BUNDLE_INDEX}
SOURCE_PROVENANCE=${4:?usage: ode_panel32_holder_action.sh ARM ACTION PANEL_MANIFEST SOURCE_PROVENANCE BUNDLE_INDEX}
BUNDLE_INDEX=${5:?usage: ode_panel32_holder_action.sh ARM ACTION PANEL_MANIFEST SOURCE_PROVENANCE BUNDLE_INDEX}
case "$ACTION" in F|FR|R|BR|B|BL|L|FL|N) ;; *) echo "bad action: $ACTION" >&2; exit 2 ;; esac

R=${ARRWM_REMOTE_ROOT:-/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler}
A=${ARRWM_CODE_ROOT:-$R/ARRWM}
PY=${ARRWM_PYTHON:-$R/miniforge3/envs/arrwm/bin/python}
PANEL_ROOT=${PANEL32_STAGE:-$R/panel32_stage}
CONFIG=${ODE_PANEL32_CONFIG:-$A/configs/ar_eval_dmd_student.yaml}
TEACHER=${ODE_PANEL32_TEACHER_CKPT:-$A/logs/v14e_pca8_raw/causal_lora_step0005000.pt}
WAN_ROOT=${ODE_PANEL32_WAN_ROOT:-$R/frodobots}

case "$ARM" in
  local_kl)
    MODEL_ID=kl4rung
    OBJECTIVE=local_kl
    CKPT=${ODE_PANEL32_KL_CKPT:-$A/logs/ode14e_pilot/run3_flip2_rollkl10k/action_ode_step0000400.pt}
    ;;
  pointwise_mse)
    MODEL_ID=mse4rung
    OBJECTIVE=pointwise_mse
    CKPT=${ODE_PANEL32_MSE_CKPT:-$A/logs/ode14e_pilot/run3_flip2_roll10k/action_ode_step0000400.pt}
    ;;
  *) echo "bad ODE arm: $ARM (expected local_kl or pointwise_mse)" >&2; exit 2 ;;
esac

OUT=${ODE_PANEL32_OUT:-$PANEL_ROOT/fleet30s/ours_$MODEL_ID}
LOG=${ODE_PANEL32_LOG:-$PANEL_ROOT/logs/ours_$MODEL_ID/$ACTION}
RUNNER=$A/grids/eval/ode_panel32_runner.py
mkdir -p "$OUT" "$LOG"

mkdir -p "$PANEL_ROOT/locks"
exec 7>"$PANEL_ROOT/locks/generate_ode_${MODEL_ID}_${ACTION}.lock"
flock 7

export PATH="$R/miniforge3/envs/arrwm/bin:$PATH"
export PYTHONPATH="$A:$A/action-forcing${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${HF_HOME:-$R/frodobots/hf_cache}
export TORCH_HOME=${TORCH_HOME:-$R/torch_home}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export EVAL_SPAN_MATCH_TRAINING=1

# A holder may have run a probe earlier.  None of its optional sampler hooks
# may leak into this matched fleet.
unset ODE_ACTION_CFG ODE_EMDHEAD ODE_EMDREMAP ODE_EMDREMAP_ASENS
unset ODE_FLOW_DET ODE_FLOW_REC ODE_FLOW_SEED ODE_GEXCL ODE_GEXCL_R0
unset ODE_KLTS_CKPT ODE_KLTS_FIXED ODE_REPNUDGE_ETA ODE_SAMPLER
unset ODE_SL_KMU ODE_SL_KSTD ODE_STAT_LOCK ODE_VRFM ODE_VRFM_ZDIM
unset AF_SNAPSHOT_STEPS AF_EVAL_STEPS

test -x "$PY"
test -f "$RUNNER"
test -f "$PANEL_MANIFEST"
test -f "$SOURCE_PROVENANCE"
test -f "$BUNDLE_INDEX"
test -s "$CKPT"
test -s "$TEACHER"
test -f "$CONFIG"
test -d "$WAN_ROOT/Wan2.1-T2V-1.3B"

mapfile -t GPU_UUIDS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits)
NGPU=${#GPU_UUIDS[@]}
NUNIQUE=$(printf '%s\n' "${GPU_UUIDS[@]}" | sort -u | wc -l)
if [ "$NGPU" -ne 4 ] || [ "$NUNIQUE" -ne 4 ]; then
  echo "ODE panel32 requires four distinct visible GPUs; found $NGPU / $NUNIQUE" >&2
  exit 1
fi

COMMON=(
  --panel-manifest "$PANEL_MANIFEST"
  --source-provenance "$SOURCE_PROVENANCE"
  --bundle-index "$BUNDLE_INDEX"
  --checkpoint "$CKPT"
  --teacher-checkpoint "$TEACHER"
  --config "$CONFIG"
  --wan-model-root "$WAN_ROOT"
  --output "$OUT"
  --model-id "$MODEL_ID"
  --objective "$OBJECTIVE"
  --action "$ACTION"
)

read -r -a CONTEXT_SHARDS <<<"${PANEL32_CONTEXT_SHARDS:-0 1 2 3}"
CONTEXT_SHARD_COUNT=${PANEL32_CONTEXT_SHARD_COUNT:-4}
test "${#CONTEXT_SHARDS[@]}" -eq 4
COMMON+=(--shard-count "$CONTEXT_SHARD_COUNT")

echo "ODE_PANEL32_START arm=$ARM action=$ACTION holder=${SLURM_JOB_ID:-none} host=$(hostname) gpus=4 $(date -Is)"
CUDA_VISIBLE_DEVICES=0 "$PY" "$RUNNER" "${COMMON[@]}" --shard-index "${CONTEXT_SHARDS[0]}" --dry-run \
  | tee "$LOG/dry_run_${SLURM_JOB_ID:-manual}.log"

pids=()
for gpu in 0 1 2 3; do
  shard=${CONTEXT_SHARDS[$gpu]}
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export WORLD_SIZE=1 RANK=0 LOCAL_RANK=0
    "$PY" "$RUNNER" "${COMMON[@]}" --shard-index "$shard"
  ) >"$LOG/gpu${gpu}.out" 2>"$LOG/gpu${gpu}.err" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if [ "$failed" -ne 0 ]; then
  tail -100 "$LOG"/gpu*.err >&2 || true
  exit 1
fi
touch "$LOG/COMPLETE"
echo "ODE_PANEL32_ACTION_COMPLETE arm=$ARM action=$ACTION context_shards=${CONTEXT_SHARDS[*]} $(date -Is)"
