#!/bin/bash
# INTERACTIVE rollout-ODE smoke. Runs inside an existing salloc (JOBID=$ALLOC).
#   ALLOC=<jobid> TAG=t1 STEPS=3 bash sbatch/iact_roll_smoke.sh [extra=overrides ...]
# Deliberately NOT the sbatch driver: no probe/scorecard tail, tiny step count,
# fresh logfile per attempt so successive tries stay separable.
set -x
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=$PWD:$PWD/action-forcing TMPDIR=/tmp
export AF_SNAPSHOT_STEPS="0,15,18,19,-1" AF_EVAL_STEPS=20
export ODE_ROLLOUT=1 ODE_ALLDIR=1
export AF_CKPT_DEBUG=${AF_CKPT_DEBUG:-0}

ALLOC=${ALLOC:?set ALLOC=<salloc jobid>}
TAG=${TAG:-t1}
STEPS=${STEPS:-3}
# squeue, not `scontrol show job | grep NodeList`: that output also contains
# ReqNodeList=/ExcNodeList= and a mis-anchored pattern silently yields
# "(null)" as the rendezvous host, which fails as an opaque c10d timeout.
NODES=$(squeue -j $ALLOC -h -o "%D" | tr -d ' ')
MASTER_ADDR=$(scontrol show hostnames "$(squeue -j $ALLOC -h -o '%N')" | head -n1)
[ -n "$MASTER_ADDR" ] || { echo "alloc $ALLOC has no nodes yet"; exit 1; }
# Port and rdzv_id must be unique PER ATTEMPT, not per allocation: two runs in
# the same held allocation otherwise collide on the c10d store (observed:
# fix1 died with RendezvousConnectionError against dbg1's stale store).
TAGSUM=$(printf %s "$TAG" | cksum | cut -d' ' -f1)
MASTER_PORT=$((20000 + (ALLOC + TAGSUM) % 20000))
export MASTER_ADDR MASTER_PORT
export NCCL_CROSS_NIC=1 NCCL_SOCKET_IFNAME=hsn OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Surface the real frame: without this an assert deep in the rollout shows up
# only as a rank-0 SIGABRT with no Python traceback.
export TORCH_SHOW_CPP_STACKTRACES=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1

LOG=logs/iact_roll_${TAG}.log
LOGDIR=logs/ode14e_pilot/iact_${TAG}
rm -rf "$LOGDIR"; mkdir -p "$LOGDIR"

# --overlap: since Slurm 20.11 a step does NOT share resources with existing
# steps by default. The node HOLDER (hold_nodes_2node.sbatch) keeps its
# allocation alive with a foreground `sleep`, which is itself a step -- so
# without --overlap this srun blocks forever waiting for resources that the
# sleep is holding, and looks identical to "still queued". Harmless for a
# plain salloc allocation, which has no competing step.
srun --overlap --jobid=$ALLOC --nodes=$NODES --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=64 \
  torchrun --nnodes=$NODES --nproc_per_node=4 \
    --rdzv_id=iact${ALLOC}_${TAG} --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  action-forcing/train.py \
    --config configs/action_ode_distill_F.yaml \
    chunked_lmdb=true ode_chunked_supervision=true \
    cd_teacher_loss_enabled=false cd_student_loss_enabled=false \
    ode_rollout=true ode_rollout_commit=teacher \
    alldir_batches=true ode_curriculum=true ode_curriculum_epochs=10 \
    clean_root=/projects/u6ex/fbots/frodobots_lmdb/v14e_pilot_dir8n \
    cf_root=/projects/u6ex/fbots/frodobots_lmdb/v14e_pilot_dir8n \
    clean_only=false require_cf=false lambda_cf=1.0 \
    random_steps="[0,15,18,19]" eval_inference_steps=20 \
    generator_ckpt=/scratch/u6ex/as1748.u6ex/ARRWM/logs/v14e_pca8_raw/causal_lora_step0005000.pt \
    total_steps=$STEPS save_interval=$STEPS eval_interval=$STEPS \
    ckpt_skip_optimizer=true ckpt_local_stage=true \
    logdir=$LOGDIR config_name=iact_${TAG} wandb_name=iact_${TAG} \
    "$@" > $LOG 2>&1
RC=$?
echo "=== EXIT $RC ==="
exit $RC
