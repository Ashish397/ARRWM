#!/bin/bash
#SBATCH --job-name=secant-decomp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm

export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache

echo "=== Secant decomposition over all matched LMDB windows (both actual + cf sides) ==="

python noise_model/run_secant_decomp.py \
    --out_dir noise_model/secant_out \
    --n_save  200

echo "=== Done ==="
echo "Outputs:"
ls -lh noise_model/secant_out/
