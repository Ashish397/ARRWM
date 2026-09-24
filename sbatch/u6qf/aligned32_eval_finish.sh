#!/bin/bash
# Merge, score, and fail-closed validate the four aligned evaluation shards.
#SBATCH --job-name=a32-finish
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler/aligned32_stage/logs/%x_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler/aligned32_stage/logs/%x_%j.err
set -euo pipefail

R=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler
CODE="$R/ARRWM"
PY="$R/miniforge3/envs/arrwm/bin/python3.10"
OUT="$CODE/grids/eval/out_iclr_aligned32_20260923"
export PATH="$R/miniforge3/envs/arrwm/bin:$PATH"
rm -f "$OUT/logs/EVALUATION_COMPLETE"
for shard in 0 1 2 3; do test -f "$OUT/logs/shard${shard}/COMPLETE"; done

export ARR="$CODE"
export OURS30S_DIR="$R/_logs/ours30s/out"
export OURS_ODE_DIR="$R/iclrv3_localfleet/experiments/e1/rec/long40/.motion_check"
export MINWM_ODE_DIR="$R/aligned32_stage/minwm_ode"
export MINWM_ODE_SWAP_YAW=0
export FLEET30S_ALIGNED32_DIR="$R/aligned32_stage/fleet30s_aligned32"
export SEED65_DIR="$R/aligned32_stage/seed_stream_aligned"
export PYTHONPATH="$CODE/grids/eval:$CODE/code_release:${PYTHONPATH:-}"
export OMP_NUM_THREADS=32

cd "$CODE"
"$PY" grids/eval/final_v2_summarize.py --out "$OUT" >"$OUT/logs/cpu_summary.log" 2>&1
"$PY" grids/eval/final_v2_relocation.py --out "$OUT" >"$OUT/logs/relocation.log" 2>&1
"$PY" grids/eval/final_v3_quality_summary.py --out "$OUT" >"$OUT/logs/quality_summary.log" 2>&1
"$PY" grids/eval/validate_action_conventions.py --out "$OUT" >"$OUT/logs/action_conventions.log" 2>&1
"$PY" grids/eval/final_v3_validate.py --out "$OUT" >"$OUT/logs/final_validation.log" 2>&1
"$PY" grids/eval/render_iclr_rollout_samples.py \
  --manifest "$OUT/video_manifest.csv" \
  --seed-frame32 "$R/aligned32_stage/seed_frame32" \
  --figures "$CODE/iclr/Figures" >"$OUT/logs/rollout_figures.log" 2>&1
touch "$OUT/logs/EVALUATION_COMPLETE"
echo "ALIGNED32_EVALUATION_COMPLETE $(date -Is)"
