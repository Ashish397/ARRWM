#!/bin/bash
#SBATCH --job-name=vista-sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Vista deformation sweep: find a turn setting that steers WITHOUT deformation.
# Full-lock (speed8/angle780) turned but deformed; 2.0/260 didn't move.
set -e
ARR=/scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache TMPDIR=/tmp
cd $ARR/third_party/Vista
for combo in "2.0:390" "4.0:390" "4.0:520" "2.0:780" "6.0:650"; do
  SPD=${combo%%:*}; ANG=${combo##*:}
  export VS_WINDOWS="8" VS_DIRS="R" VS_ROUNDS=4 VS_STEPS=50
  export VS_SPEED=$SPD VS_ANGLE=$ANG
  export VS_OUT=$ARR/logs/eval_final/baseline_smoke/vista_sweep/s${SPD}_a${ANG}
  PYTHONPATH= python vista_runner.py
  echo "[sweep] done speed=$SPD angle=$ANG"
done
echo "VISTA-SWEEP done $(date)"
