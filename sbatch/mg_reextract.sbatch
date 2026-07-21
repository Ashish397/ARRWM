#!/bin/bash
#SBATCH --job-name=mg-reextract
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --gpus-per-node=4
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Re-extract all metrics for the regenerated matrixgame BR/BL videos.
# All extractors are resume-safe: stale rows/members were purged, so each
# fills exactly the 64 missing (window, BR/BL) entries; world grove
# re-encodes the full model (npz was rebuilt from scratch by design).
set -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH TMPDIR=/tmp ARRWM_ACTION_ENCODER=pca_raw

CUDA_VISIBLE_DEVICES=0 HH_MODELS=matrixgame HH_OUT=analysis/eval_final/headtohead_matrixgame.csv \
  python utils/headtohead_extract.py > logs/mgre_${SLURM_JOB_ID}_hh.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 SA_RUNS=matrixgame SA_OUT=analysis/eval_final/suiteA_comp2.csv \
  python utils/suite_scoreA.py > logs/mgre_${SLURM_JOB_ID}_suite.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 bash -c '
  EN_RUNS=matrixgame EN_OUT=analysis/eval_final/edge_nov_g2.csv python utils/edge_novelty.py &&
  CJ_RUNS=matrixgame CJ_OUT=analysis/eval_final/center_jump_g2.csv python utils/center_jump.py &&
  SD_RUNS=matrixgame SD_OUT=analysis/eval_final/spawn_det_g2.csv python utils/spawn_detect.py
' > logs/mgre_${SLURM_JOB_ID}_nov.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 bash -c '
  WG_RUN=matrixgame python utils/world_grove.py &&
  TY_MODELS=matrixgame python utils/ty_extract.py
' > logs/mgre_${SLURM_JOB_ID}_grove.log 2>&1 &
wait
echo "MG-REEXTRACT done $(date)"
