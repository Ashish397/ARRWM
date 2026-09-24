#!/bin/bash
# One-context GPU integration smoke for the final DMD and local-KL ODE paths.
set -euo pipefail

R=${ARRWM_REMOTE_ROOT:-/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler}
A=${ARRWM_CODE_ROOT:-$R/ARRWM}
P=$R/panel32_stage
PY=$R/miniforge3/envs/arrwm/bin/python
M=$A/grids/eval/panel32_locked_v1.json
S=$P/sources/panel32_source_provenance.json
B=$P/ours_seed_bundles/panel32_seed_bundles.json
WAN=$R/frodobots/Wan2.1-T2V-1.3B
SMOKE=$P/smoke
mkdir -p "$SMOKE/ours" "$SMOKE/ode"

export PATH="$R/miniforge3/envs/arrwm/bin:$PATH"
export PYTHONPATH="$A:$A/action-forcing${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=$R/frodobots/hf_cache
export TORCH_HOME=$R/torch_home
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export EVAL_SPAN_MATCH_TRAINING=1
export CUDA_VISIBLE_DEVICES=0

"$PY" "$A/interactive/external_models/ours_panel32.py" \
  --panel-manifest "$M" --source-provenance "$S" --bundle-index "$B" \
  --checkpoint "$R/ckpts/recovery_base/phase1_step0001000.pt.lean.pt" \
  --output "$SMOKE/ours" --action F --tag smokerecoverybase \
  --model-label "ARRWM recoverybase smoke" --carn-commit auto \
  --shard-index 0 --shard-count 32 --wan-model-root "$WAN"

ODE_COMMON=(
  --panel-manifest "$M" --source-provenance "$S" --bundle-index "$B"
  --checkpoint "$A/logs/ode14e_pilot/run3_flip2_rollkl10k/action_ode_step0000400.pt"
  --teacher-checkpoint "$A/logs/v14e_pca8_raw/causal_lora_step0005000.pt"
  --config "$A/configs/ar_eval_dmd_student.yaml" --wan-model-root "$R/frodobots"
  --output "$SMOKE/ode" --model-id smoke_kl4rung --objective local_kl
  --action F --shard-index 0 --shard-count 32
)
"$PY" "$A/grids/eval/ode_panel32_runner.py" "${ODE_COMMON[@]}" --dry-run
"$PY" "$A/grids/eval/ode_panel32_runner.py" "${ODE_COMMON[@]}"

echo "PANEL32_ARRWM_SMOKE_COMPLETE $(date -Is)"
