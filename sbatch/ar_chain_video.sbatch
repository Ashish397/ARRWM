#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# $1=assignment_index $2=config $3=manifest $4=rank_zarr $5=rank_offset $6=rank_tag $7=out_dir
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
AIDX="$1"; CONFIG="$2"; MANIFEST="$3"; ZARR="$4"; OFFSET="$5"; TAG="$6"; OUTDIR="$7"
set --
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR HF_HUB_CACHE=$CACHE_DIR HUGGINGFACE_HUB_CACHE=$CACHE_DIR TRANSFORMERS_CACHE=$CACHE_DIR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export WORLD_SIZE=1 LOCAL_RANK=0 CUDA_VISIBLE_DEVICES=0

echo "AR-chain  aidx=$AIDX  zarr=$ZARR off=$OFFSET tag=$TAG  out=$OUTDIR  job=$SLURM_JOB_ID  $(date)"
python utils/eval_causal_chain.py \
  --assignment_index "$AIDX" \
  --config "$CONFIG" \
  --manifest "$MANIFEST" \
  --rank_zarr "$ZARR" \
  --rank_offset "$OFFSET" \
  --rank_mode dataset \
  --rank_tag "$TAG" \
  --encoded_root /projects/u6ex/fbots/frodobots_encoded_weu \
  --caption_root /projects/u6ex/fbots/frodobots_captions/train \
  --seed 42 \
  --output_dir "$OUTDIR"
echo "Done $(date)"
