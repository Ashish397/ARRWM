#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# $1=config  $2=logdir  $3=out_dir  $4=checkpoint(optional)
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
CONFIG="$1"; LOGDIR="$2"; OUTDIR="$3"; CKPT="${4:-}"
set --
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR HF_HUB_CACHE=$CACHE_DIR HUGGINGFACE_HUB_CACHE=$CACHE_DIR TRANSFORMERS_CACHE=$CACHE_DIR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
echo "compare-video  config=$CONFIG  logdir=$LOGDIR  out=$OUTDIR  job=$SLURM_JOB_ID  $(date)"
python utils/make_compare_video.py \
  --config "$CONFIG" --logdir "$LOGDIR" --out_dir "$OUTDIR" \
  --windows_per_ride 2 --seed 0 --save_gt \
  ${CKPT:+--checkpoint "$CKPT"}
echo "Done $(date)"
