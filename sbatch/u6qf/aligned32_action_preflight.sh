#!/bin/bash
#SBATCH --job-name=a32-actions
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler/aligned32_stage/logs/%x_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler/aligned32_stage/logs/%x_%j.err
set -euo pipefail

R=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler
CODE="$R/ARRWM"
PY="$R/miniforge3/envs/arrwm/bin/python3.10"
OUT="${PREFLIGHT_OUT:-$CODE/grids/eval/out_iclr_aligned32_20260923/preflight}"
MODELS="${PREFLIGHT_MODELS:-lingbot,dreamx,minwm,matrixgame2,minwm_ode,ours_recovery_base}"
UIDS="${PREFLIGHT_UIDS:-u31,u37,u48,u00}"
export PATH="$R/miniforge3/envs/arrwm/bin:$PATH"

export ARR="$CODE"
export OURS30S_DIR="$R/_logs/ours30s/out"
export OURS_ODE_DIR="$R/iclrv3_localfleet/experiments/e1/rec/long40/.motion_check"
export MINWM_ODE_DIR="$R/aligned32_stage/minwm_ode"
export MINWM_ODE_SWAP_YAW=0
export FLEET30S_ALIGNED32_DIR="$R/aligned32_stage/fleet30s_aligned32"
export SEED65_DIR="$R/aligned32_stage/seed_stream_aligned"
export TORCH_HOME="$R/torch_home"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$CODE/grids/eval:$CODE/code_release:${PYTHONPATH:-}"

cd "$CODE"
rm -f "$OUT/ACTION_PREFLIGHT_COMPLETE"
"$PY" grids/eval/preflight_action_conventions.py \
  --out "$OUT" --models "$MODELS" --uids "$UIDS"
test -f "$OUT/ACTION_PREFLIGHT_COMPLETE"
echo "ALIGNED32_ACTION_PREFLIGHT_COMPLETE $(date -Is)"
