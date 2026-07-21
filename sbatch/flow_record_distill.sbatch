#!/bin/bash
#SBATCH --job-name=flow-rec-dist
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# ODE-distillation flow diagnosis, window r08, 8 dirs x 4 seeds, pinned noise
# shared with the existing v14e recordings:
#   1) 14d teacher (v14d_critic8_natural @2150) — dense 48-step ODE via
#      utils/flow_record.py (identical protocol to the 14e recordings)
#   2) three ODE-F students (latest ckpts) — their real 4-step re-noise
#      sampler via utils/flow_record_ode_student.py + the env-gated hook in
#      utils/eval_causal_AR.py (initial chunk noise IDENTICAL to teacher's)
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH TMPDIR=/tmp
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_${USER}_frd TRITON_CACHE_DIR=/tmp/tr_${USER}_frd

echo "=== recording 14d teacher (dense 48-step) ==="
FR_RUN=14d FR_CONFIG=configs/causal_lora_diffusion_teacher_v14d.yaml \
  FR_CKPT=logs/v14d_critic8_natural/causal_lora_step0002150.pt \
  python utils/flow_record.py

declare -A SCFG=(
  [odeF]=configs/action_ode_distill_F.yaml
  [odeF_teacherCD]=configs/action_ode_distill_F_teacherCD.yaml
  [odeF_noCD]=configs/action_ode_distill_F_noCD.yaml
)
declare -A SCKPT=(
  [odeF]=logs/action_ode_distill_F/action_ode_step0000200.pt
  [odeF_teacherCD]=logs/action_ode_distill_F_teacherCD/action_ode_step0000500.pt
  [odeF_noCD]=logs/action_ode_distill_F_noCD/action_ode_step0001000.pt
)
for run in odeF odeF_teacherCD odeF_noCD; do
  echo "=== recording student $run ==="
  FR_RUN=$run FR_CONFIG=${SCFG[$run]} FR_CKPT=${SCKPT[$run]} \
    python utils/flow_record_ode_student.py
done
echo "FLOW-REC-DIST done $(date)"
