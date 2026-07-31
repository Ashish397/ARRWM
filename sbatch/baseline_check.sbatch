#!/bin/bash
#SBATCH --job-name=af_baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/af_baseline_%j.out
# Verify the code release still computes what it did before cleanup.
# MODE=save writes the reference; MODE=check compares against it.
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
cd /scratch/u6ex/as1748.u6ex/ARRWM
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export AF_ROOT=/scratch/u6ex/as1748.u6ex/ARRWM/code_release
export DATA_ROOT=/tmp/data WAN_MODELS=/tmp/wan HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache
export BASELINE_JSON=${BASELINE_JSON:-/scratch/u6ex/as1748.u6ex/ARRWM/tools/baseline.json}
python tools/release_baseline.py --${MODE:-check}
echo "BASELINE_EXIT=$?"
