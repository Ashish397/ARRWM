#!/bin/bash
#SBATCH --job-name=rec-tf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=$PWD:$PWD/action-forcing TMPDIR=/tmp
FR_RUN=tf_gt0 FR_CKPT=none FR_CONFIG=configs/ar_eval_dmd_student.yaml \
  FR_RUNGS="1000,625,357.142857,208.333333" python utils/flow_record_tf.py || echo "FAIL tf"
echo "REC-TF done $(date)"
