#!/bin/bash
# Shared, fail-closed environment for the locked mixed-dataset panel32 run.
# Source this file; it submits and launches nothing.

R=${ARRWM_REMOTE_ROOT:-/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler}
CODE=${ARRWM_CODE_ROOT:-${AF_ROOT:-$R/ARRWM}}
PY=${ARRWM_PYTHON:-$R/miniforge3/envs/arrwm/bin/python3.10}
PANEL=${PANEL32_STAGE_ROOT:-$R/panel32_stage}
LOCKED_MANIFEST=${PANEL32_LOCKED_MANIFEST:-$CODE/grids/eval/panel32_locked_v1.json}
CONFIG=${PANEL32_EVAL_CONFIG:-$PANEL/panel32_eval_config.json}
OUT=${PANEL32_EVAL_OUT:-$PANEL/eval_final}
PANEL32_MODELS=lingbot,dreamx,matrixgame2,minwm,minwm_ode,yume5b,ours_recovery_base,ours_no_carn,ours_no_commit,ours_no_aux,ours_no_gan,ours_stat_mean_only,ours_stat_nonmean_only,ours_kl4rung,ours_mse4rung

export R CODE PY PANEL LOCKED_MANIFEST CONFIG OUT PANEL32_MODELS
export PATH="$R/miniforge3/envs/arrwm/bin:$PATH"
export TORCH_HOME="$R/torch_home"
export HF_HOME="$R/iclrv3_hf_cache"
export HF_HUB_OFFLINE=1
export ARR="$CODE"
export PANEL32_EVAL_CONFIG="$CONFIG"
export FLEET30S_MODELS="$PANEL32_MODELS"
export PYTHONPATH="$CODE/grids/eval:$CODE/code_release${PYTHONPATH:+:$PYTHONPATH}"

require_four_distinct_cuda_ordinals() {
  local label=${1:-panel32_eval}
  local uuid unique gpu
  local -a uuids=()
  for gpu in 0 1 2 3; do
    uuid=$(CUDA_VISIBLE_DEVICES="$gpu" "$PY" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
properties = torch.cuda.get_device_properties(0)
uuid = str(getattr(properties, "uuid", ""))
assert uuid and uuid.lower() != "none", properties
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
  echo "PANEL32_EVAL_GPU_UUID_PREFLIGHT_PASS label=$label uuids=${uuids[*]}"
}
