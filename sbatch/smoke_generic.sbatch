#!/bin/bash
# Generic release smoke: run one command from inside code_release on a GPU node.
#   SMK_CMD  : the command to run (required)
#   SMK_TAG  : label, used in the log name
# Everything else (env contract, data roots) is set here so the smokes exercise
# the release exactly as a reader would, via DATA_ROOT / WAN_MODELS / AF_ROOT.
#SBATCH --job-name=smk
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/smk_%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/smk_%x_%j.err

set -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM/code_release

source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm

CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR HF_HUB_CACHE=$CACHE_DIR
export HUGGINGFACE_HUB_CACHE=$CACHE_DIR TRANSFORMERS_CACHE=$CACHE_DIR

export DATA_ROOT=/projects/u6ex/fbots
export WAN_MODELS=/scratch/u6ex/as1748.u6ex/frodobots
export AF_ROOT=/scratch/u6ex/as1748.u6ex/ARRWM
export PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM/code_release
export ARRWM_ACTION_ENCODER=pca_raw
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

echo "SMOKE_TAG=${SMK_TAG}"
echo "SMOKE_CMD=${SMK_CMD}"
echo "node=$(hostname)  gpus=${SLURM_GPUS_ON_NODE}  start=$(date -u +%H:%M:%S)"
echo "-----------------------------------------------------------------"

eval "${SMK_CMD}"
RC=$?

echo "-----------------------------------------------------------------"
echo "SMOKE_TAG=${SMK_TAG} SMOKE_EXIT=${RC} end=$(date -u +%H:%M:%S)"
exit $RC
