#!/bin/bash
# Inner launcher for the ODE-F smoke, invoked under an `srun` allocation:
#   srun --nodes=2 --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=128 \
#        --exclusive --time=01:00:00 bash sbatch/smoke_ode_F_srun.sh
# Validates the critic-dim fix (2-dim critic + shape-filtered 14d critic8 load),
# the dual-CD path, and the Madrid causal-chain eval before the full run starts.
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR HF_HUB_CACHE=$CACHE_DIR HUGGINGFACE_HUB_CACHE=$CACHE_DIR TRANSFORMERS_CACHE=$CACHE_DIR
mkdir -p logs/action_ode_distill_F_smoke

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 16000))
export NCCL_CROSS_NIC=1 NCCL_SOCKET_IFNAME=hsn NCCL_DEBUG=WARN NCCL_IB_TIMEOUT=50
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=4 \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  action-forcing/train.py \
  --config configs/action_ode_distill_F_smoke.yaml

echo "ODE-F SMOKE done $(date)"
