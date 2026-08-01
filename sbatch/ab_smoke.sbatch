#!/bin/bash
# A/B smoke: run the SAME recipe under the pre-cleanup repo and under
# code_release, on identical node count and seed, so any divergence in the
# logged loss is attributable to the cleanup and nothing else.
#   AB_REPO   : absolute path of the tree to run (main repo or code_release)
#   AB_CONFIG : absolute path of the matched config
#   AB_TAG    : label used for the logdir
#SBATCH --job-name=ab_smoke
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

set -e -o pipefail
cd "${AB_REPO}"

source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm

CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR HF_HUB_CACHE=$CACHE_DIR
export HUGGINGFACE_HUB_CACHE=$CACHE_DIR TRANSFORMERS_CACHE=$CACHE_DIR

LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/ab_${AB_TAG}
mkdir -p "$LOGDIR" logs

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 16000))

export NCCL_CROSS_NIC=1 NCCL_SOCKET_IFNAME=hsn NCCL_DEBUG=WARN NCCL_IB_TIMEOUT=50
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ARRWM_ACTION_ENCODER=pca_raw
export WANDB_MODE=offline          # keep the comparison free of network effects

echo "AB_TAG=${AB_TAG}  REPO=${AB_REPO}  CONFIG=${AB_CONFIG}"
echo "Job $SLURM_JOB_ID  Nodes: $SLURM_NNODES  GPUs: $(( SLURM_NNODES * SLURM_GPUS_ON_NODE ))"

srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=4 \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  train.py \
  --config_path "${AB_CONFIG}" \
  --logdir "$LOGDIR" \
  --wandb-save-dir wandb

echo "AB_SMOKE_DONE ${AB_TAG} $(date)"
