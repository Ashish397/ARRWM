#!/bin/bash
#SBATCH --job-name=probe-seeddepth
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# EXPOSURE-BIAS TEST (audit A1). Training always gives a 21-frame/7-chunk CLEAN
# teacher context; serving starts from ONE real chunk, so every served chunk runs
# at a context depth (3..18) the student was never trained at, and all but the
# first chunk is its OWN output. Serve the SAME checkpoint with a 1-chunk seed
# and a 7-chunk (trained-depth, all-REAL) seed. If the stripes and the weak
# motion largely vanish with the deep real seed, the failure is exposure bias,
# not the objective -- which changes what we build next.
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=$PWD:$PWD/action-forcing TMPDIR=/tmp
CKPT=logs/ode14e_pilot/run3_flip2_c10mse/action_ode_step0002800.pt
for SC in 1 7; do
  FR_SEED_CHUNKS=$SC FR_RUN=pilot8_seed${SC} FR_CONFIG=configs/ar_eval_dmd_student.yaml \
    FR_CKPT=$CKPT FR_RUNGS="1000,625,357.142857,208.333333" FR_VIDEO=1 FR_CHUNKS=6 FR_NSEEDS=2 \
    python utils/flow_record_ode_student.py || echo "PROBE-FAIL seed$SC"
  FDD_NSEEDS=2 FDD_PAIRS="pilot8_seed${SC}=14e8" FDD_FIGNAME=flow_pilot8_seed${SC}_scorecard \
    python utils/flow_diverge_dmd3.py || true
done
MC_RUNS=TEACHER48:pilot8_seed1:pilot8_seed7 python utils/motion_check.py || true
MT_BASE=TEACHER48 python utils/motion_table.py || true
TM_RUNS=pilot8_seed1:pilot8_seed7 TM_OUT=analysis/eval_final/flow_viz/teacher_match_seeddepth.csv \
  python utils/teacher_match.py || true
echo "PROBE-SEEDDEPTH done $(date)"
