#!/bin/bash
#SBATCH --job-name=collect-paired
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm

echo "=== Collecting paired (gt_target, e_noise) samples for Approach 4 ==="
echo "NOTE: Reads same window order as secant_decomp run to reproduce alignment."
echo "      Outputs gt_samples.pt to noise_model/secant_out/ (NO overwrite of existing files)"

python noise_model/collect_paired_samples.py \
    --secant_out_dir noise_model/secant_out \
    --out_dir        noise_model/secant_out \
    --n_save         200

echo "=== Done ==="
ls -lh noise_model/secant_out/gt_samples.pt noise_model/secant_out/paired_meta.json 2>/dev/null || true
