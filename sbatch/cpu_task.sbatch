#!/bin/bash
#SBATCH --job-name=cpu_task
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --time=01:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/cpu_%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/cpu_%x_%j.err
set -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM/code_release
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export DATA_ROOT=/projects/u6ex/fbots WAN_MODELS=/scratch/u6ex/as1748.u6ex/frodobots
export AF_ROOT=/scratch/u6ex/as1748.u6ex/ARRWM PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM/code_release
export OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32
echo "TAG=${CPU_TAG} start=$(date -u +%H:%M:%S)"
eval "${CPU_CMD}"; RC=$?
echo "TAG=${CPU_TAG} CPU_EXIT=${RC} end=$(date -u +%H:%M:%S)"
exit $RC
