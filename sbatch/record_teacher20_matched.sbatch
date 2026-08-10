#!/bin/bash
#SBATCH --job-name=rec-teach20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# SEED-MATCHED dense-teacher reference for the match metric. FR_CKPT=none
# = teacher weights; 20-rung shift-5 schedule = the teacher's real sampler;
# same window/seed stream as every student probe, so student and reference
# share their NOISE. Without this the metric ceiling is ~0.85 (a model vs
# itself at a different seed) and the 0.95 target is unreachable by
# construction. Also records the 4-rung teacher-init at matched seeds.
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=$PWD:$PWD/action-forcing TMPDIR=/tmp
R20="1000.000000,989.583333,978.260870,965.909091,952.380952,937.500000,921.052632,902.777778,882.352941,859.375000,833.333333,803.571429,769.230769,729.166667,681.818182,625.000000,555.555556,468.750000,357.142857,208.333333"
FR_RUN=teacher20_matched FR_CONFIG=configs/ar_eval_dmd_student.yaml FR_CKPT=none \
  FR_RUNGS="$R20" FR_VIDEO=1 FR_CHUNKS=6 FR_NSEEDS=2 \
  python utils/flow_record_ode_student.py || echo "REC-FAIL teacher20"
TM_RUNS=pilot4_alldir8n2:pilot_gt0:pilot2_flip2:pilot5_a8n_rm10:pilot5_a8n_rmauto2 \
  TM_REF_RUN=teacher20_matched python utils/teacher_match.py || true
echo "REC-TEACH20 done $(date)"
