#!/bin/bash
# Byte-identical-OFF proof for the reverse-noiser de-drift diff.
# Runs the PRE-DIFF (git HEAD) copy of utils/eval_causal_AR.py with the exact
# arm-"base" invocation. Its rollout mp4 must be byte-identical to the
# NOTE: recreate the pre-diff copy first (it is deleted after use):
#   git show HEAD:utils/eval_causal_AR.py > utils/.eval_causal_AR_predif.py
# It MUST live at repo-depth 1 in a REAL dir (logs/ is a symlink into
# ARRWM_data, which breaks the script's _REPO_ROOT = parents[1] lookup).
# working-tree base arm's.
set -u -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache
export HF_HUB_CACHE=$HF_HOME
export HUGGINGFACE_HUB_CACHE=$HF_HOME
export TRANSFORMERS_CACHE=$HF_HOME
export TMPDIR=/tmp
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM
PRE=/scratch/u6ex/as1748.u6ex/ARRWM/utils/.eval_causal_AR_predif.py
CKPT=logs/dmd10k_carntx_match/dmd10k_carntx_match_h6117225_224923/eval_step0200.pt
out=eval/dedrift_smoke_predif
mkdir -p "$out"
CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 LOCAL_RANK=0 python "$PRE" \
  --student_ckpt "$CKPT" \
  --config configs/ar_eval_dmd_student.yaml \
  --manifest "" \
  --rank_zarr 20240216101235.zarr \
  --rank_offset 100 \
  --rank_mode dataset \
  --rank_tag madrid \
  --encoded_root /projects/u6ex/fbots/frodobots_encoded_weunz \
  --caption_root /projects/u6ex/fbots/frodobots_captions/train \
  --motion_root /projects/u6ex/fbots/frodobots_motion \
  --ss_vae_checkpoint action_query/checkpoints/ss_vae_8free.pt \
  --seed 42 --denoising_steps 4 --mode ar \
  --ar_initial_chunks 3 --ar_gen_chunks 15 --cache_chunks 7 \
  --no-ar_cache --infinity_rope --carn_seam_affine_lambda 0.0 \
  --label dedrift_base --output_dir "$out" > logs/dedrift_smoke_predif.log 2>&1
echo "predif exit=$?"
echo "=== md5 comparison (must match) ==="
md5sum eval/dedrift_smoke_base/rank0_dedrift_base_madrid/*_rollout_raw.mp4 \
       eval/dedrift_smoke_predif/rank0_dedrift_base_madrid/*_rollout_raw.mp4
