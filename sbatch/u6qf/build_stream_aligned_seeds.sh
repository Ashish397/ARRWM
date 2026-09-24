#!/bin/bash
#SBATCH --job-name=a32-seeds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler/aligned32_stage/logs/%x_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler/aligned32_stage/logs/%x_%j.err
set -euo pipefail

R=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler
CODE="$R/ARRWM"
PY="$R/miniforge3/envs/arrwm/bin/python3.10"
export PATH="$R/miniforge3/envs/arrwm/bin:$PATH"

"$PY" "$CODE/grids/eval/build_stream_aligned_seeds.py" \
  --windows "$CODE/experiments/e1/scene_shortlist/e1_32_windows.json" \
  --ours-dir "$R/_logs/ours30s/out/recovery_base" \
  --video-out "$R/aligned32_stage/seed_stream_aligned" \
  --frame-out "$R/aligned32_stage/seed_frame32"

touch "$R/aligned32_stage/seed_stream_aligned/COMPLETE"
echo "STREAM_ALIGNED_SEEDS_COMPLETE $(date -Is)"
