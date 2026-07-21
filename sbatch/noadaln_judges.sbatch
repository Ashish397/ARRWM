#!/bin/bash
#SBATCH --job-name=noadaln-judges
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH TMPDIR=/tmp
R7="pca8_8node:pca4:pca2:16node:4node:noatok:noadaln"
echo "=== qwen ==="; VJ_RUNS=$R7 python utils/vlm_judge.py 2>&1 | grep -E "qwen|VALIDATION|r0|REAL" | tail -20
echo "=== d3 ==="; D3_RUNS=$R7 python utils/d3_runner.py 2>&1 | grep -E "d3|ranking|r0" | tail -14
echo "=== videolisa ==="; cd third_party/VideoLISA && VL_RUNS=$R7 python videolisa_judge.py 2>&1 | grep -E "vlisa|VALIDATION|r0|REAL" | tail -30; cd ../..
echo "=== nsgvd ==="; cd third_party/NSG-VD && NV_RUNS=$R7 python /scratch/u6ex/as1748.u6ex/ARRWM/utils/nsgvd_runner.py 2>&1 | tail -12
echo "NOADALN-JUDGES done"
