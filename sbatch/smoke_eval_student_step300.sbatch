#!/bin/bash
#SBATCH --job-name=af-smoke-step300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# Renders clean.mp4, cf.mp4, sidebyside.mp4 for the student checkpoint
# ``latest_0000300.pt`` written by af_trainer.ode.

set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM

# Protect `conda activate` from any positional args that may reach this
# script. See run_action_ode_distill_8n.sbatch for the full explanation.
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
export OMP_NUM_THREADS=4

echo "=== $(date) host=$(hostname) job=$SLURM_JOB_ID ==="
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader

OUT=/scratch/u6ex/as1748.u6ex/ARRWM/eval/action_ode_distill_step0000300
mkdir -p "$OUT"

python3 action-forcing/bin/smoke_eval.py \
    --ckpt /scratch/u6ex/as1748.u6ex/ARRWM/logs/v14_balanced_weunz/causal_lora_step0006600.pt \
    --student-ckpt /scratch/u6ex/as1748.u6ex/ARRWM/logs/action_ode_distill_10h/latest_0000300.pt \
    --config configs/action_ode_distill.yaml \
    --pair-index 0 \
    --output "$OUT"

echo "=== $(date) smoke_eval step-300 done; outputs in $OUT ==="
ls -lh "$OUT"
