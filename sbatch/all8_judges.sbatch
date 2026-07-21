#!/bin/bash
#SBATCH --job-name=all8-judges
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
set -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH TMPDIR=/tmp
R8="pca8_8node:pca4:pca2:16node:4node:noatok:noadaln:minwm"
L=/scratch/u6ex/as1748.u6ex/ARRWM/logs
echo "=== qwen ===";    VJ_RUNS=$R8 python utils/vlm_judge.py  > $L/all8_qwen_${SLURM_JOB_ID}.log 2>&1 || echo "QWEN FAILED"
echo "=== d3 ===";      D3_RUNS=$R8 python utils/d3_runner.py  > $L/all8_d3_${SLURM_JOB_ID}.log 2>&1 || echo "D3 FAILED"
echo "=== videolisa ==="; (cd third_party/VideoLISA && VL_RUNS=$R8 python videolisa_judge.py) > $L/all8_vlisa_${SLURM_JOB_ID}.log 2>&1 || echo "VLISA FAILED"
echo "=== nsgvd ===";   (cd third_party/NSG-VD && NV_RUNS=$R8 python /scratch/u6ex/as1748.u6ex/ARRWM/utils/nsgvd_runner.py) > $L/all8_nsgvd_${SLURM_JOB_ID}.log 2>&1 || echo "NSGVD FAILED"
for f in qwen d3 vlisa nsgvd; do
  echo "##### $f results #####"
  grep -aE "VALIDATION|r01R|r08B|r08BL|r01_R|r08_B|r08_BL|REAL|ranking|mean=" $L/all8_${f}_${SLURM_JOB_ID}.log 2>/dev/null | tail -22
  echo "--- errors (if any) ---"
  grep -am2 -E "Traceback|Error" $L/all8_${f}_${SLURM_JOB_ID}.log 2>/dev/null
done
echo "ALL8-JUDGES done"
