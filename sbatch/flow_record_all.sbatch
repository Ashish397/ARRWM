#!/bin/bash
#SBATCH --job-name=flow-rec-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Record denoising trajectories (8 dirs x 4 seeds, window r08, identical noise
# across runs) for every v14e ablation ckpt @ step 5000, one subprocess per run
# so trainer state can't leak between models. Skips runs already on disk.
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH TMPDIR=/tmp
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_${USER}_fr TRITON_CACHE_DIR=/tmp/tr_${USER}_fr

CFG=configs/causal_lora_diffusion_teacher
declare -A CONFIGS=(
  [pca8]=${CFG}_v14e.yaml
  [pca4]=${CFG}_v14e_pca4.yaml
  [pca2]=${CFG}_v14e_pca2.yaml
  [16node]=${CFG}_v14e_16node.yaml
  [4node]=${CFG}_v14e_4node.yaml
  [noatok]=${CFG}_v14e_noatok.yaml
  [noadaln]=${CFG}_v14e_noadaln.yaml
)
declare -A CKPTS=(
  [pca8]=logs/v14e_pca8_raw/causal_lora_step0005000.pt
  [pca4]=logs/v14e_pca4/causal_lora_step0005000.pt
  [pca2]=logs/v14e_pca2/causal_lora_step0005000.pt
  [16node]=logs/v14e_16node/causal_lora_step0005000.pt
  [4node]=logs/v14e_4node/causal_lora_step0005000.pt
  [noatok]=logs/v14e_noatok/causal_lora_step0005000.pt
  [noadaln]=logs/v14e_noadaln/causal_lora_step0005000.pt
)

for run in pca8 pca4 pca2 16node 4node noatok noadaln; do
  echo "=== recording $run ==="
  FR_RUN=$run FR_CONFIG=${CONFIGS[$run]} FR_CKPT=${CKPTS[$run]} \
    python utils/flow_record.py
done
echo "FLOW-REC-ALL done $(date)"
