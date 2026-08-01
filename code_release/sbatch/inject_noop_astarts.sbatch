#!/bin/bash
#SBATCH --job-name=inj-noopA
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# No-op (static command) rollouts from the 32 PHASE-A starts. Same
# inject_eval machinery as phase B, but IE_WINDOWS=phaseA_windows.json;
# 1 node / 4 shards (only 32 gens per run).
set -e -o pipefail
cd ${AF_ROOT}
source ${CONDA_ROOT}/bin/activate; conda activate arrwm
export HF_HOME=${HF_HOME} PYTHONPATH=${AF_ROOT}:$PYTHONPATH TMPDIR=/tmp
echo "inject-noopA: config=$IE_CONFIG out=$IE_OUT ckpt=$IE_CKPT"
srun --ntasks=4 --ntasks-per-node=4 --gpus-per-node=4 bash -c '
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  export IE_SHARD=$SLURM_PROCID IE_NSHARDS=4
  export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_${USER}_${SLURM_PROCID} TRITON_CACHE_DIR=/tmp/tr_${USER}_${SLURM_PROCID}
  python evaluation/inject_eval.py > logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_shard${SLURM_PROCID}.log 2>&1
'
echo "inj-noopA done $(date)"
