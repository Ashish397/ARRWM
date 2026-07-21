#!/bin/bash
#SBATCH --job-name=flow-rec-dmd3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Flow recording for the two heavily-trained phase-3 DMD students + their
# reference teachers, window r08, 8 dirs x 4 seeds, pinned noise:
#   teachers (dense 48-step, BOTH blocks 0+1 for compounding analysis):
#     v14  = v14_balanced_weunz@6600        (stat_wave's frozen merged teacher)
#     14d2 = v14d_critic8_natural@2150      (af_all's teacher; re-record w/ blk1)
#   students (their real 4-step sampler, both chunks, via ar_eval config —
#   load_checkpoint strict-overlays the full generator so merge base is moot):
#     afall_freal_cd  = af_all_freal_cd_j5313179/phase1_step0000500.pt
#     statwave_Freal  = stat_wave_dmd3_fwd_Freal_j5302282/phase1_step0000200.pt
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH TMPDIR=/tmp
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_${USER}_fr3 TRITON_CACHE_DIR=/tmp/tr_${USER}_fr3

echo "=== recording v14 teacher (dense, blocks 0+1) ==="
FR_RUN=14 FR_CONFIG=configs/causal_lora_diffusion_teacher_v14.yaml \
  FR_CKPT=logs/v14_balanced_weunz/causal_lora_step0006600.pt \
  FR_BLOCKS=0,1 python utils/flow_record.py

echo "=== recording 14d teacher (dense, blocks 0+1) ==="
FR_RUN=14d2 FR_CONFIG=configs/causal_lora_diffusion_teacher_v14d.yaml \
  FR_CKPT=logs/v14d_critic8_natural/causal_lora_step0002150.pt \
  FR_BLOCKS=0,1 python utils/flow_record.py

P3=logs/action_forcing_phase3_dmd_one_step
echo "=== recording student afall_freal_cd ==="
FR_RUN=afall_freal_cd FR_CONFIG=configs/ar_eval_dmd_student.yaml \
  FR_CKPT=$P3/af_all_freal_cd_j5313179/phase1_step0000500.pt \
  python utils/flow_record_ode_student.py

echo "=== recording student statwave_Freal ==="
FR_RUN=statwave_Freal FR_CONFIG=configs/ar_eval_dmd_student.yaml \
  FR_CKPT=$P3/stat_wave_dmd3_fwd_Freal_j5302282/phase1_step0000200.pt \
  python utils/flow_record_ode_student.py

echo "FLOW-REC-DMD3 done $(date)"
