#!/bin/bash
#SBATCH --job-name=cm-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --gpus-per-node=4
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH TMPDIR=/tmp
export ARRWM_ACTION_ENCODER=pca_raw
MAN=analysis/eval_final/manifest_unseen.pt

# 6 runs split across the 4 GPUs
declare -A G=( [0]="pca8_8node pca4" [1]="pca2 16node" [2]="4node noadaln" [3]="noatok" )

run_phase () {  # $1=phase(A|B) $2=recon
  local ph="$1" recon="$2"
  for g in 0 1 2 3; do
    dirs=""
    for run in ${G[$g]}; do dirs+="${run}:logs/eval_final/${ph}/${run}/control_test,"; done
    dirs="${dirs%,}"
    [ -z "$dirs" ] && continue
    CUDA_VISIBLE_DEVICES=$g CM_DIRS="$dirs" CM_RECON="$recon" CM_MANIFEST="$MAN" CM_STRIDE=200 \
      CM_OUT="analysis/eval_final/phase${ph}_metrics_g${g}.csv" \
      TORCHINDUCTOR_CACHE_DIR="/tmp/ti_${USER}_$g" TRITON_CACHE_DIR="/tmp/tr_${USER}_$g" \
      python utils/chunk_metrics.py > "logs/cm-eval_${SLURM_JOB_ID}_${ph}_gpu${g}.log" 2>&1 &
  done
  wait
  python -c "
import pandas as pd, glob
parts=[pd.read_csv(f) for f in sorted(glob.glob('analysis/eval_final/phase${ph}_metrics_g*.csv')) if __import__('os').path.getsize(f)>50]
d=pd.concat(parts, ignore_index=True); d.to_csv('analysis/eval_final/phase${ph}_metrics.csv', index=False)
print('phase${ph}: runs', sorted(d.run.unique()), 'rows', len(d), '| iqa', int(d.musiq.notna().sum()), '| recon', int(d.mae.notna().sum()))
"
}

echo "=== Phase A (no recon: injected dirs are counterfactual) ==="
run_phase A ""
echo "=== Phase B (recon=static vs real continuation) ==="
run_phase B "static"
echo "CM-EVAL done $(date)"
