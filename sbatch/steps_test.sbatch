#!/bin/bash
#SBATCH --job-name=steps-test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Denoise-steps ablation: window 0, four cardinal commands (F,B,R,L), at
# 40/20/15/10 steps. Compare against the 48-step phase-A reference
# (logs/eval_final/A/pca8_8node, same window+branches).
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH TMPDIR=/tmp
export IE_CONFIG=configs/causal_lora_diffusion_teacher_v14e.yaml
export IE_CKPT=logs/v14e_pca8_raw/causal_lora_step0005000.pt
export IE_PHASE=A IE_ONLY=0 IE_BRANCHES=F,B,R,L
export IE_WINDOWS=analysis/eval_final/phaseA_windows.json
export IE_MANIFEST=analysis/eval_final/manifest_unseen.pt
for st in 40 20 15 10; do
  IE_STEPS=$st IE_OUT=logs/eval_final/steps_test/s${st} python utils/inject_eval.py
  echo "steps=$st done $(date)"
done
echo "STEPS-TEST done $(date)"
