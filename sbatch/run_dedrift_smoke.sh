#!/bin/bash
# Reverse-noiser de-drift SMOKE for utils/eval_causal_AR.py.
#
# Proves the new --reverse_noiser_* commit-time de-drift path (a) is inert
# when the flag is unset and (b) runs + produces finite, sane output when
# pointed at a real reverse-trained ForwardNoiser.
#
# 3 arms, IDENTICAL seed / student / ride / geometry, ONE flag apart:
#   base      : no --reverse_noiser_checkpoint  (byte-identical-off reference)
#   carnctrl  : gansig_carn_ctrl j6135660 fn_rev_step0175.pt
#               (forward_noiser_reverse=true + chain_levels=true -> the FN was
#                conditioned on the INPUT's rollout level, so level_auto)
#   carntxmt  : dmd10k_carntx_match h6117225 fn_rev_step0125.pt
#               (reverse=true + chain_levels=false + carn_recurse=false ->
#                FN(student chunk @L) -> clean GT, applied at level 1 in
#                training; this checkpoint is the SAME RUN as the student,
#                so it is the matched-pair arm)
#
# carn_seam_affine_lambda is 0.0 in every arm: the de-drift and the seam
# affine are not a validated stack, so this smoke exercises the de-drift alone.
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
export MASTER_PORT=${MASTER_PORT:-44911}

CKPT=logs/dmd10k_carntx_match/dmd10k_carntx_match_h6117225_224923/eval_step0200.pt
FN_CTRL=/scratch/u6ex/as1748.u6ex/ARRWM_data/logs/dmd10k_gansig_carn_ctrl/dmd10k_gansig_carn_ctrl_j6135660/fn_rev_step0175.pt
FN_MATCH=/scratch/u6ex/as1748.u6ex/ARRWM_data/logs/dmd10k_carntx_match/dmd10k_carntx_match_h6117225_224923/fn_rev_step0125.pt
test -f "$CKPT" || { echo "MISSING student $CKPT"; exit 2; }
test -f "$FN_CTRL" || { echo "MISSING $FN_CTRL"; exit 2; }
test -f "$FN_MATCH" || { echo "MISSING $FN_MATCH"; exit 2; }

GEN_CHUNKS=${GEN_CHUNKS:-15}

run_arm () {  # $1=tag  $2..=extra flags
  local tag="$1"; shift
  local out="eval/dedrift_smoke_${tag}"
  mkdir -p "$out"
  echo "=== ARM $tag $(date) ==="
  CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 LOCAL_RANK=0 python utils/eval_causal_AR.py \
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
    --seed 42 \
    --denoising_steps 4 \
    --mode ar \
    --ar_initial_chunks 3 \
    --ar_gen_chunks "$GEN_CHUNKS" \
    --cache_chunks 7 \
    --no-ar_cache \
    --infinity_rope \
    --carn_seam_affine_lambda 0.0 \
    --label "dedrift_${tag}" \
    --output_dir "$out" \
    "$@" > "logs/dedrift_smoke_${tag}.log" 2>&1
  echo "=== ARM $tag exit=$? $(date) log=logs/dedrift_smoke_${tag}.log ==="
}

run_arm base
run_arm carnctrl \
  --reverse_noiser_checkpoint "$FN_CTRL" \
  --reverse_noiser_dedrift_level_auto \
  --reverse_noiser_dedrift_steps 1 \
  --reverse_noiser_dedrift_alpha0 1.0 \
  --reverse_noiser_dedrift_min_level 1
run_arm carntxmt \
  --reverse_noiser_checkpoint "$FN_MATCH" \
  --reverse_noiser_dedrift_level 1 \
  --reverse_noiser_dedrift_steps 1 \
  --reverse_noiser_dedrift_alpha0 1.0 \
  --reverse_noiser_dedrift_min_level 1

echo "ALL DEDRIFT SMOKE ARMS DONE $(date)"
for t in base carnctrl carntxmt; do
  echo "--- $t [AR][STATS] ---"
  grep -h "\[AR\]\[STATS\]\|\[AR\]\[DEDRIFT\]" "logs/dedrift_smoke_${t}.log" | head -40
done
