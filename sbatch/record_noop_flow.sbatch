#!/bin/bash
#SBATCH --job-name=rec-noop
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Stationary/no-op branch for the flow trees: teacher 48-step + 20-step
# chains and teacher-init 4-rung ladder, action (0,0), window r08.
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=$PWD:$PWD/action-forcing TMPDIR=/tmp
CKPT=logs/v14e_pca8_raw/causal_lora_step0005000.pt
CFG=configs/causal_lora_diffusion_teacher_v14e.yaml
FR_NOOP=1 FR_RUN=14e8noop FR_CONFIG=$CFG FR_CKPT=$CKPT FR_STEPS=48 \
  FR_BLOCKS=0,1,2,3,4,5 FR_NSEEDS=4 python utils/flow_record.py || echo "FAIL 48"
FR_NOOP=1 FR_RUN=14e8s20noop FR_CONFIG=$CFG FR_CKPT=$CKPT FR_STEPS=20 \
  FR_BLOCKS=0,1,2,3,4,5 FR_NSEEDS=4 python utils/flow_record.py || echo "FAIL 20"
FR_NOOP=1 FR_RUN=pilot_gt0noop FR_CONFIG=configs/ar_eval_dmd_student.yaml FR_CKPT=none \
  FR_RUNGS="1000,625,357.142857,208.333333" FR_CHUNKS=6 FR_NSEEDS=2 FR_VIDEO=1 \
  python utils/flow_record_ode_student.py || echo "FAIL gt0noop"
echo "REC-NOOP done $(date)"
