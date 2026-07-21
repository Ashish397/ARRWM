#!/bin/bash
#SBATCH --job-name=tb2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/tb2_gpu_%j.out

# Testbench V2 stage 2: CSD/DINO/Gram/MS-SWD/RAFT/pyiqa features, 4 shards
# in parallel (one per GPU). Merge shards afterwards:
#   python - <<'EOF'
#   import pandas as pd, glob
#   pd.concat(map(pd.read_csv, glob.glob('analysis/testbench_v2/out/gpu_features.shard*.csv'))) \
#     .to_csv('analysis/testbench_v2/out/gpu_features.csv', index=False)
#   EOF
source /lus/lfs1aip2/scratch/u6ex/as1748.u6ex/miniforge3/bin/activate arrwm
cd /lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM
export PYTHONPATH=/lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export TB2_CSD_CKPT=/path/to/csd_vitl14.pth   # else falls back to CLIP ViT-L

export TB2_NSHARD=4
for s in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$s TB2_SHARD=$s \
    python analysis/testbench_v2/gpu_metrics.py &
done
wait
python - <<'EOF'
import pandas as pd, glob
shards = sorted(glob.glob('analysis/testbench_v2/out/gpu_features.shard*.csv'))
pd.concat(map(pd.read_csv, shards)).to_csv(
    'analysis/testbench_v2/out/gpu_features.csv', index=False)
print('merged', len(shards), 'shards')
EOF
