#!/bin/bash
#SBATCH --job-name=action_ode_distill_10h
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Usage:
#   sbatch sbatch/run_action_ode_distill_8n.sbatch [config] [extra overrides...]
#
# 8 nodes x 4 GPUs = 32 ranks. One torchrun per node (srun --ntasks-per-node=1),
# torchrun fans out to 4 local workers. Rendezvous via c10d on the first node.

set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM

# Capture + clear positional args before sourcing conda's activate script.
# `source .../bin/activate` inherits the parent script's positional args,
# and miniforge's activate errors out with
#   "ArgumentError: activate does not accept more than one argument"
# when more than one arg (i.e. our forwarded overrides) is visible to it.
_orig_args=("$@")
set --
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
set -- "${_orig_args[@]}"

export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache
export HF_HUB_CACHE=$HF_HOME
export HUGGINGFACE_HUB_CACHE=$HF_HOME
export TRANSFORMERS_CACHE=$HF_HOME
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
# NCCL: try native IB first (fast). Uncomment NCCL_IB_DISABLE=1 only if
# IB handshake hangs. TORCH_NCCL_BLOCKING_WAIT gives us sane error msgs.
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=WARN
# Avoid HF-hub collisions during first-rank downloads across nodes.
export TOKENIZERS_PARALLELISM=false

CONFIG="${1:-configs/action_ode_distill.yaml}"
shift || true
EXTRA_ARGS=("$@")

# -----------------------------------------------------------------------------
# Rendezvous: first node of the allocation becomes the coordinator.
# -----------------------------------------------------------------------------
MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$(python3 -c "import socket,sys; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
export MASTER_ADDR="$MASTER_NODE"
export MASTER_PORT

mkdir -p logs

echo "=== $(date) job=$SLURM_JOB_ID ==="
echo "Nodes:       $SLURM_JOB_NUM_NODES   Node list: $SLURM_JOB_NODELIST"
echo "Master:      $MASTER_ADDR:$MASTER_PORT"
echo "Config:      $CONFIG"
echo "Overrides:   ${EXTRA_ARGS[*]:-<none>}"

# One torchrun per node. torchrun internally spawns --nproc_per_node workers.
# --rdzv_backend=c10d with shared endpoint gives us collective bootstrap
# without needing an external etcd.
srun --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 \
    bash -c '
        set -e
        echo "[rank-node=$SLURM_NODEID host=$(hostname)] torchrun start"
        nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
        torchrun \
            --nnodes='"$SLURM_JOB_NUM_NODES"' \
            --nproc_per_node=4 \
            --node_rank=$SLURM_NODEID \
            --rdzv_id='"$SLURM_JOB_ID"' \
            --rdzv_backend=c10d \
            --rdzv_endpoint='"$MASTER_ADDR:$MASTER_PORT"' \
            action-forcing/train.py \
                --config '"$CONFIG"' \
                '"${EXTRA_ARGS[*]}"'
    '

echo "=== $(date) action_ode_distill finished ==="
