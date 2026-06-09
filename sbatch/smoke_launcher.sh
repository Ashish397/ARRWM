#!/bin/bash
# Smoke launcher: runs the g1 config for SMOKE_MAX_STEPS steps on whatever
# node count SLURM gives this step. Invoked via:
#   srun --jobid=<alloc> --overlap --ntasks-per-node=1 bash sbatch/smoke_launcher.sh
# Derives the override flags directly from the g1 sbatch so it always tracks
# the latest g1 config; only max_steps / run_name / wandb are swapped.
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm

CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR HF_HUB_CACHE=$CACHE_DIR HUGGINGFACE_HUB_CACHE=$CACHE_DIR TRANSFORMERS_CACHE=$CACHE_DIR
export NCCL_CROSS_NIC=1 NCCL_SOCKET_IFNAME=hsn NCCL_DEBUG=WARN NCCL_IB_TIMEOUT=50 OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
export ARRWM_ROPE_DEBUG=1
export WANDB_MODE=offline

LOGDIR=logs/smoke
WANDB_SAVE_DIR=wandb
mkdir -p "$LOGDIR"
SMOKE_MAX_STEPS=${SMOKE_MAX_STEPS:-25}
G1=${SMOKE_SBATCH:-sbatch/train_action_forcing_phase3_dmd_one_step_E2_4_ns_allsup_sav2_channel_base_flash_transition_v8g1.sbatch}

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 16000))

# Pull g1's override block, swap max_steps + run_name, strip line-continuations.
# Unique run_name per smoke so auto_resume can't short-circuit by resuming
# a stale smoke checkpoint (which silently skips training). Also force
# auto_resume=false belt-and-braces.
SMOKE_RUN="smoke_${SMOKE_TAG:-run}_${SLURM_JOB_ID}_${SMOKE_MAX_STEPS}"
OVERRIDES=$(awk '/^  --override/{f=1;next} f{print}' "$G1" | sed -n '1,/run_name=/p' \
  | sed "s/max_steps=200/max_steps=${SMOKE_MAX_STEPS}/" \
  | sed "s/run_name=.*/run_name=${SMOKE_RUN}/" \
  | tr -d '\\')
# Optional extra overrides (e.g. vis_save_local=true to force the 7-chunk
# eval on under WANDB_MODE=offline). Space-separated key=value pairs.
OVERRIDES="$OVERRIDES auto_resume=false ${SMOKE_EXTRA:-}"
echo "[smoke] run_name=$SMOKE_RUN auto_resume=false"

echo "[smoke] nodes=$SLURM_NNODES master=$MASTER_ADDR:$MASTER_PORT max_steps=$SMOKE_MAX_STEPS"
torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=4 \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  trainer/causal_action_forcing_train.py \
  --config configs/action_forcing_phase3_dmd.yaml \
  --override $OVERRIDES
echo "[smoke] DONE rc=$?"
