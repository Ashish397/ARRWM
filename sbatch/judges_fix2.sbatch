#!/bin/bash
#SBATCH --job-name=judges-fix2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
set -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache TMPDIR=/tmp
R8="pca8_8node:pca4:pca2:16node:4node:noatok:noadaln:minwm"
L=/scratch/u6ex/as1748.u6ex/ARRWM/logs
echo "=== videolisa (PYTHONPATH cleared) ==="
(cd third_party/VideoLISA && PYTHONPATH= VL_RUNS=$R8 python videolisa_judge.py) > $L/fix2_vlisa_${SLURM_JOB_ID}.log 2>&1 || echo "VLISA FAILED"
echo "=== nsgvd (full windows for new models) ==="
(cd third_party/NSG-VD && PYTHONPATH= NV_RUNS=$R8 python /scratch/u6ex/as1748.u6ex/ARRWM/utils/nsgvd_runner.py) > $L/fix2_nsgvd_${SLURM_JOB_ID}.log 2>&1 || echo "NSGVD FAILED"
for f in vlisa nsgvd; do
  echo "##### $f #####"
  grep -aE "VALIDATION|r01R|r08B|r08BL|REAL|mean=" $L/fix2_${f}_${SLURM_JOB_ID}.log | tail -14
  grep -am2 -aE "Traceback|Error" $L/fix2_${f}_${SLURM_JOB_ID}.log
done
echo "JUDGES-FIX2 done"
