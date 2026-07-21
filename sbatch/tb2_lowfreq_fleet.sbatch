#!/bin/bash
#SBATCH --job-name=tb2_lf_fleet
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=05:00:00
#SBATCH --output=logs/slurm/tb2_lf_fleet_%j.out

# Low-frequency-collapse ensemble ingredients FLEET-WIDE (~3,400 videos):
# mangle (PAL4VST + Qwen melt) + geometry (depth/LSD/epipolar), 4 shards.
# yume excluded: A_yume is being repopulated by the yume-full job.
source /lus/lfs1aip2/scratch/u6ex/as1748.u6ex/miniforge3/bin/activate arrwm
cd /lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM
export PYTHONPATH=/lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TB2_EXCLUDE=yume
export TB2_NSHARD=4

for s in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$s TB2_SHARD=$s bash -c '
    TB2_MANGLE_OUT=analysis/testbench_v2/out/fleet_mangle.csv \
      python analysis/testbench_v2/mangle_metrics.py
    TB2_GEOM_OUT=analysis/testbench_v2/out/fleet_geometry.csv \
      python analysis/testbench_v2/geometry_metrics.py
  ' &
done
wait
python - <<'PYEOF'
import pandas as pd, glob
for stem in ('fleet_mangle', 'fleet_geometry'):
    shards = sorted(glob.glob(f'analysis/testbench_v2/out/{stem}.shard*.csv'))
    if shards:
        df = pd.concat(map(pd.read_csv, shards)).drop_duplicates('vid')
        df.to_csv(f'analysis/testbench_v2/out/{stem}.csv', index=False)
        print(stem, len(df), 'rows from', len(shards), 'shards')
PYEOF
