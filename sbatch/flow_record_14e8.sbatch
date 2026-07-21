#!/bin/bash
#SBATCH --job-name=flow-rec-14e8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# 14e (pca8@5000) dense 48-step teacher over EIGHT chained blocks, window r08,
# 8 dirs x 4 seeds, pinned noise — the 14e analog of trajs_14d8_w8.npz, to
# compare deep-chunk AR drift of the two teacher lineages before the 14e
# LMDB generation bakes 14e's drift into deep-chunk distillation targets.
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH TMPDIR=/tmp
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_${USER}_f14e TRITON_CACHE_DIR=/tmp/tr_${USER}_f14e

FR_RUN=14e8 FR_CONFIG=configs/causal_lora_diffusion_teacher_v14e.yaml \
  FR_CKPT=logs/v14e_pca8_raw/causal_lora_step0005000.pt \
  FR_BLOCKS=0,1,2,3,4,5,6,7 python utils/flow_record.py
echo "FLOW-REC-14E8 done $(date)"
