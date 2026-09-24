#!/bin/bash
# Bind the generic evaluation payloads to the final v2 source/generation tree.
set -euo pipefail
R=${ARRWM_REMOTE_ROOT:-/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler}
A=${ARRWM_CODE_ROOT:-$R/ARRWM}
P=${PANEL32_STAGE:-$R/panel32_v2_stage}
export PANEL32_STAGE_ROOT="$P"
export PANEL32_EVAL_CONFIG="$P/panel32_eval_config.json"
export PANEL32_EVAL_OUT="$P/eval_final"
export PANEL32_LOCKED_MANIFEST="$A/grids/eval/panel32_locked_v2.json"
exec bash "$A/sbatch/u6qf/panel32_eval_sequence.sh" "$@"
