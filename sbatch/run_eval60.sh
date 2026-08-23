#!/bin/bash
# The canonical fixed-route 60 s evaluation, standalone.
#
# Identical invocation to the tail of sbatch/run_full_carn_probe.sh (same
# Madrid route, seed 42, 4 denoising steps, 3 seed chunks, 100 generated
# chunks, cache_chunks=7, infinity-RoPE, inference CARN lambda 0.5) so its
# output is directly comparable to every arm scored so far. Split out because
# fullcarn_bidir_kl_strict03 finished training but lost its chained eval, and
# re-running the eval must never mean re-running the training.
#
# Usage: ARM=<label> CKPT=<eval_step0200.pt> [EVAL_CHUNKS=100] \
#        bash sbatch/run_eval60.sh
set -euo pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
: "${ARM:?set ARM to the run label}"
: "${CKPT:?set CKPT to the evaluation checkpoint}"
EVAL_CHUNKS=${EVAL_CHUNKS:-100}
STEPTAG=${STEPTAG:-step200}

source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache
export HF_HUB_CACHE=$HF_HOME
export HUGGINGFACE_HUB_CACHE=$HF_HOME
export TRANSFORMERS_CACHE=$HF_HOME
export TMPDIR=/tmp
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

test -f "$CKPT"
OUT="eval/${ARM}_${STEPTAG}_carn05_60s"
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 LOCAL_RANK=0 python utils/eval_causal_AR.py \
  --student_ckpt "$CKPT" \
  --config configs/ar_eval_dmd_student.yaml \
  --manifest logs/v14_balanced_weunz/.ride_manifest.pt \
  --rank_zarr 20240216101235.zarr \
  --rank_offset 100 \
  --rank_mode dataset \
  --rank_tag madrid60 \
  --encoded_root /projects/u6ex/fbots/frodobots_encoded_weunz \
  --caption_root /projects/u6ex/fbots/frodobots_captions/train \
  --motion_root /projects/u6ex/fbots/frodobots_motion \
  --ss_vae_checkpoint action_query/checkpoints/ss_vae_8free.pt \
  --seed 42 \
  --denoising_steps 4 \
  --mode ar \
  --ar_initial_chunks 3 \
  --ar_gen_chunks "$EVAL_CHUNKS" \
  --cache_chunks 7 \
  --no-ar_cache \
  --infinity_rope \
  --carn_seam_affine_lambda 0.5 \
  --label "${ARM}_${STEPTAG}_carn05" \
  --output_dir "$OUT" \
  > "logs/eval_${ARM}_${STEPTAG}_60s.err" 2>&1
echo "eval done: $OUT"
