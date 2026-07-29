#!/bin/bash
#SBATCH --job-name=reprobe-fx
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:30:00
#SBATCH --output=logs/reprobe-fx_%j.out
#SBATCH --error=logs/reprobe-fx_%j.err

# Re-record the two DMD students through the FIXED serve path (infinity_rope
# wrap now inside flow_record_ode_student.py). Same protocol as the original
# flow_record_dmd3.sbatch recordings (own denoising_step_list, FR_CHUNKS=2
# default) so the delta vs flow_afall_freal_cd / flow_statwave_Freal isolates
# the RoPE probe artifact from real student divergence.
set -x
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=$PWD:$PWD/action-forcing TMPDIR=/tmp

P3=logs/action_forcing_phase3_dmd_one_step

echo "=== re-probe afall_freal_cd (fixed RoPE) ==="
FR_RUN=afall_freal_cd_fx FR_CONFIG=configs/ar_eval_dmd_student.yaml \
  FR_CKPT=$P3/af_all_freal_cd_j5313179/phase1_step0000500.pt \
  FR_VIDEO=1 FR_NSEEDS=4 \
  python utils/flow_record_ode_student.py || echo "PROBE-FAIL afall_fx"

echo "=== re-probe statwave_Freal (fixed RoPE) ==="
FR_RUN=statwave_Freal_fx FR_CONFIG=configs/ar_eval_dmd_student.yaml \
  FR_CKPT=$P3/stat_wave_dmd3_fwd_Freal_j5302282/phase1_step0000200.pt \
  FR_VIDEO=1 FR_NSEEDS=4 \
  python utils/flow_record_ode_student.py || echo "PROBE-FAIL statwave_fx"

echo "=== scorecards vs each student's own teacher (per flow_diverge_dmd3 docs) ==="
FDD_PAIRS="afall_freal_cd_fx=14d2" FDD_FIGNAME=flow_afall_fx_scorecard \
  python utils/flow_diverge_dmd3.py || echo "SCORE-FAIL afall_fx"
FDD_PAIRS="statwave_Freal_fx=14" FDD_FIGNAME=flow_statwave_fx_scorecard \
  python utils/flow_diverge_dmd3.py || echo "SCORE-FAIL statwave_fx"

MC_RUNS=afall_freal_cd_fx:statwave_Freal_fx python utils/motion_check.py || true
echo "REPROBE-FX done $(date)"
