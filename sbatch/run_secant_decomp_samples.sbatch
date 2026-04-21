#!/bin/bash
#SBATCH --job-name=secant-samples
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm

export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache

echo "=== Secant decomp — collecting 200 e_noise samples for Approach 4 training ==="
echo "NOTE: This writes to a SEPARATE directory (secant_out_samples) to avoid"
echo "      overwriting the fitted model in secant_out."

python noise_model/run_secant_decomp.py \
    --out_dir noise_model/secant_out_samples \
    --n_save  200

echo "=== Done ==="
echo "Outputs:"
ls -lh noise_model/secant_out_samples/
