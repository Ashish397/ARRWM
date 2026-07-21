#!/bin/bash
#SBATCH --job-name=mb-sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
set -e
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache TMPDIR=/tmp
for cfg in "23|0|0.01" "23|0|0.001" "17:23|0|0.01" "11:17|0|0.01" "23|1|0.001" "17|0|0.003"; do
  L=$(echo $cfg | cut -d'|' -f1); P=$(echo $cfg | cut -d'|' -f2); T=$(echo $cfg | cut -d'|' -f3)
  echo "===== CONFIG layers=$L pool=$P tail=$T ====="
  MB_LAYERS="$L" MB_POOL="$P" MB_TAIL="$T" \
    MB_OUT=analysis/eval_final/mangle_bank_L${L//:/-}_P${P}_T${T}.csv \
    PYTHONPATH=$PWD python utils/mangle_bank.py 2>/dev/null | grep -E "ranking|r08_B :|r08_BL :|r01_R :|REAL"
done
echo "MB-SWEEP done"
