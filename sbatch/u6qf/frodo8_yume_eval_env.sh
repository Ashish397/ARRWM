#!/bin/bash
# Shared, fail-closed scope for the 8-context FrodoBots + YUME pilot.
# Source this file; it intentionally submits no jobs.

R=${ARRWM_REMOTE_ROOT:-/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler}
CODE=${ARRWM_CODE_ROOT:-${AF_ROOT:-$R/ARRWM}}
PY=${ARRWM_PYTHON:-$R/miniforge3/envs/arrwm/bin/python3.10}
ALIGNED=${FRODO8_ALIGNED_ROOT:-$R/aligned32_stage/fleet30s_aligned32}
YUME=${YUME_OUT:-$ALIGNED/yume5b}
DREAMX30S_DIR=${DREAMX30S_DIR:-$ALIGNED/dreamx}
OUT=${FRODO8_EVAL_OUT:-$CODE/grids/eval/out_frodo8_yume_aligned32_20260923}
UID_JSON=$CODE/experiments/e1/scene_shortlist/frodo8_manifest.json
FRODO8_UIDS=u31,u04,a20,m30,m38,m89,m128,b36
FRODO8_MODELS=lingbot,dreamx,minwm,matrixgame2,yume5b,minwm_ode,ours_kl4rung,ours_mse4rung,ours_recovery_base,ours_no_commit,ours_no_aux,ours_no_gan,ours_no_carn,ours_stat_mean_only,ours_stat_nonmean_only,ours_base_v2_BROKEN

export R CODE PY ALIGNED YUME DREAMX30S_DIR OUT UID_JSON FRODO8_UIDS FRODO8_MODELS
export PATH="$R/miniforge3/envs/arrwm/bin:$PATH"
export TORCH_HOME="$R/torch_home"
export HF_HOME="$R/hf_cache"
export ARR="$CODE"
export OURS30S_DIR="$R/_logs/ours30s/out"
export OURS_ODE_DIR="$R/iclrv3_localfleet/experiments/e1/rec/long40/.motion_check"
export MINWM_ODE_DIR="$R/aligned32_stage/minwm_ode"
export MINWM_ODE_SWAP_YAW=0
export FLEET30S_ALIGNED32_DIR="$ALIGNED"
export YUME30S_DIR="$YUME"
export SEED65_DIR="$R/aligned32_stage/seed_stream_aligned"
export FLEET30S_UIDS="$FRODO8_UIDS"
export FLEET30S_MODELS="$FRODO8_MODELS"
export PYTHONPATH="$CODE/grids/eval:$CODE/code_release${PYTHONPATH:+:$PYTHONPATH}"

require_four_distinct_cuda_ordinals() {
  local label=${1:-frodo8_eval}
  local uuid unique gpu
  local -a uuids=()
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
  done
  unique=$(printf '%s\n' "${uuids[@]}" | sort -u | wc -l)
  if [ "$unique" -ne 4 ]; then
    echo "$label requires four distinct CUDA ordinals; got: ${uuids[*]}" >&2
    return 74
  fi
  echo "FRODO8_EVAL_GPU_UUID_PREFLIGHT_PASS label=$label uuids=${uuids[*]}"
}
