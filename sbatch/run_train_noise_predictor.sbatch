#!/bin/bash
#SBATCH --job-name=noise-predictor
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm

echo "=== Training ContentNoisePredictor (Approach 4) ==="
echo "Data: noise_model/secant_out/{gt_samples,e_noise_samples_actual}.pt"

python noise_model/train_noise_predictor.py \
    --data_dir   noise_model/secant_out \
    --out_dir    noise_model/secant_out \
    --epochs     300 \
    --batch_size 32 \
    --lr         1e-3 \
    --hidden_dim 64 \
    --n_layers   3

echo "=== Done ==="
ls -lh noise_model/secant_out/noise_predictor.pt \
       noise_model/secant_out/noise_predictor_diagnostics.json 2>/dev/null || true
