#!/bin/bash
# One srun task: render CLIPS[$SLURM_PROCID] for the current config ($CFG_*).
# Pinned to the task's local GPU. Skips if already done. Re-times to ~4s.
set -u
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm

ci=${SLURM_PROCID}
line=$(sed -n "$((ci + 1))p" "$CLIPS_FILE")
[ -z "$line" ] && exit 0
read -r z s <<< "$line"
d="$OUT/${CFG_LABEL}_c${ci}"
if [ -f "$d/timed.mp4" ]; then echo "skip $CFG_LABEL c$ci"; exit 0; fi

export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID}
export WORLD_SIZE=1 LOCAL_RANK=0 SMOOTH_DECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR HF_HUB_CACHE=$CACHE_DIR HUGGINGFACE_HUB_CACHE=$CACHE_DIR TRANSFORMERS_CACHE=$CACHE_DIR

python utils/eval_causal_chain.py --assignment_index "$CFG_A" --config "$CFG_CONFIG" --manifest "$CFG_MANIFEST" \
  --rank_zarr "$z" --rank_offset "$s" --rank_mode dataset --rank_tag "c$ci" \
  --encoded_root "$ENC" --caption_root "$CAP" --seed 42 --output_dir "$d" > "$d.log" 2>&1 \
  || { echo "FAIL $CFG_LABEL c$ci (see $d.log)"; exit 1; }
raw=$(ls "$d"/rank0_*/final_causal_rollout_*_raw.mp4 2>/dev/null | head -1)
[ -f "$raw" ] && ffmpeg -y -loglevel error -i "$raw" -vf "setpts=N/(24*TB)" -r 24 -an "$d/timed.mp4" && echo "done $CFG_LABEL c$ci"
