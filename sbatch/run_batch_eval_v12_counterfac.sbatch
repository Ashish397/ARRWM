#!/bin/bash
#SBATCH --job-name=v12-batch-eval-cf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm

export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export NUM_GPUS=4

mkdir -p /projects/u6ex/fbots/frodobots_lmdb_counterfac /projects/u6ex/fbots/frodobots_noise_counterfac logs

echo "Starting v12 counterfactual batch eval on $(date)"

for GPU in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$GPU GPU_RANK=$GPU python3 batch_eval_v12_counterfac.py \
        > logs/batch_eval_cf_gpu${GPU}.log 2>&1 &
done

wait
echo "Counterfactual batch eval completed on $(date)"
