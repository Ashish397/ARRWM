#!/bin/bash
# One immutable, fail-closed gate over every regenerated external fleet and
# the regenerated minWM ODE fleet before any metric producer is allowed to run.
#SBATCH --job-name=a32-fleet-gate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler/aligned32_stage/logs/%x_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler/aligned32_stage/logs/%x_%j.err
set -euo pipefail

R=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler
CODE="$R/ARRWM"
PY="$R/miniforge3/envs/arrwm/bin/python3.10"
OUT="$CODE/grids/eval/out_iclr_aligned32_20260923"
export PATH="$R/miniforge3/envs/arrwm/bin:$PATH"
mkdir -p "$OUT/logs"
rm -f "$OUT/logs/FLEET_GATE_COMPLETE"

"$PY" "$CODE/grids/eval/validate_aligned32_fleets.py" \
  --root "$R/aligned32_stage/fleet30s_aligned32" \
  --uids "$CODE/experiments/e1/scene_shortlist/e1_32_windows.json" \
  --seed-frame32 "$R/aligned32_stage/seed_frame32" \
  --seed65-dir "$R/aligned32_stage/seed_stream_aligned" \
  --minwm-ode-dir "$R/aligned32_stage/minwm_ode" \
  --matrix-runner "$CODE/code_release/baselines/matrixgame_runner.py" \
  --matrix-config "$R/aligned32_stage/Matrix-Game-2/configs/inference_yaml/inference_universal.yaml" \
  --matrix-checkpoint "$R/aligned32_stage/Matrix-Game-2/Matrix-Game-2.0/base_distilled_model/base_distill.safetensors" \
  >"$OUT/logs/fleet_gate_validation.json"

touch "$OUT/logs/FLEET_GATE_COMPLETE"
echo "ALIGNED32_FLEET_GATE_COMPLETE $(date -Is)"
