#!/bin/bash
#SBATCH --job-name=ode14e-pilot
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# 14e ODE pilot training on a chained pilot LMDB (action-multiplicity study).
# Env: PILOT_VARIANT=gt|flip2|dir4|dir8 (required), PILOT_STEPS (def 500;
# use 20 for the 2-node smoke). Pinned 20-step rung grid end-to-end.
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
CACHE=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache
export HF_HOME=$CACHE PYTHONPATH=$PWD:$PWD/action-forcing TMPDIR=/tmp
export AF_SNAPSHOT_STEPS="0,15,18,19,-1" AF_EVAL_STEPS=20
V=${PILOT_VARIANT:?set PILOT_VARIANT}
STEPS=${PILOT_STEPS:-500}
LAMCF=1.0; [ "$V" = "gt" ] && LAMCF=0.0
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1)
MASTER_PORT=29511

srun torchrun \
  --nnodes=$SLURM_NNODES --nproc_per_node=4 \
  --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  action-forcing/train.py \
  --config configs/action_ode_distill_F.yaml \
  chunked_lmdb=true \
  clean_root=/projects/u6ex/fbots/frodobots_lmdb/v14e_pilot_${V} \
  cf_root=/projects/u6ex/fbots/frodobots_lmdb/v14e_pilot_${V} \
  clean_only=false require_cf=false lambda_cf=${LAMCF} \
  random_steps="[0,15,18,19]" \
  generator_ckpt=/scratch/u6ex/as1748.u6ex/ARRWM/logs/v14e_pca8_raw/causal_lora_step0005000.pt \
  max_steps=${STEPS} checkpoint_interval=100 \
  log_dir=/scratch/u6ex/as1748.u6ex/ARRWM/logs/ode14e_pilot \
  config_name=ode14e_pilot_${V}
echo "ODE14E-PILOT ${V} done $(date)"
