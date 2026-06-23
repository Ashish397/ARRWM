#!/bin/bash
#SBATCH --job-name=loo-dims-videos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Watchable AR causal-chain rollouts for the dimension study: v14, loo_f3,
# probe2dim, critic8 -- each on a GPU, over 3 held-out clips. SMOOTH_DECODE,
# re-timed to ~4s/24fps. Saves _raw/_annotated/_gt mp4s under eval/loo_dims/.
set -u -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR HF_HUB_CACHE=$CACHE_DIR HUGGINGFACE_HUB_CACHE=$CACHE_DIR TRANSFORMERS_CACHE=$CACHE_DIR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
ENC=/projects/u6ex/fbots/frodobots_encoded_weu
CAP=/projects/u6ex/fbots/frodobots_captions/train
OUT=eval/loo_dims; mkdir -p "$OUT"

# model table: gpu | assignment | label | config | manifest
GPU=(0 1 2 3)
A=(5 6 11 10)
LAB=(v14 loo_f3 probe2dim critic8)
CFG=(configs/causal_lora_diffusion_teacher_v14.yaml
     configs/causal_lora_diffusion_teacher_v14_loo_f3.yaml
     configs/causal_lora_diffusion_teacher_v14_loo_probe2dim.yaml
     configs/causal_lora_diffusion_teacher_v14_loo_critic8_noprobe.yaml)
MAN=(logs/v14_balanced_weunz/.ride_manifest.pt
     logs/v14_loo_f3/.ride_manifest.pt
     logs/v14_loo_probe2dim/.ride_manifest.pt
     logs/v14_loo_critic8_noprobe/.ride_manifest.pt)

# 3 held-out clips (zarr offset). All models held these out.
CLIP_Z=(20240216101235.zarr 20240314084739.zarr 20240319133048.zarr)
CLIP_O=(30 30 30)

render_model() {  # $1 = model index
  local m=$1
  for ci in "${!CLIP_Z[@]}"; do
    local d="$OUT/${LAB[$m]}_c${ci}"
    if [ -f "$d/timed.mp4" ]; then echo "skip ${LAB[$m]} c$ci"; continue; fi
    CUDA_VISIBLE_DEVICES=${GPU[$m]} WORLD_SIZE=1 LOCAL_RANK=0 SMOOTH_DECODE=1 \
      python utils/eval_causal_chain.py --assignment_index "${A[$m]}" --config "${CFG[$m]}" --manifest "${MAN[$m]}" \
        --rank_zarr "${CLIP_Z[$ci]}" --rank_offset "${CLIP_O[$ci]}" --rank_mode dataset --rank_tag "c$ci" \
        --encoded_root "$ENC" --caption_root "$CAP" --seed 42 --output_dir "$d" > "$d.log" 2>&1 \
      || { echo "FAIL ${LAB[$m]} c$ci (see $d.log)"; continue; }
    local raw; raw=$(ls "$d"/rank0_*/final_causal_rollout_*_annotated.mp4 2>/dev/null | head -1)
    [ -f "$raw" ] && ffmpeg -y -loglevel error -i "$raw" -vf "setpts=N/(24*TB)" -r 24 -an "$d/timed.mp4" && echo "done ${LAB[$m]} c$ci"
  done
}

echo "=== rendering 4 models x 3 clips (one model per GPU) $(date) ==="
for m in 0 1 2 3; do render_model "$m" & done
wait
echo "=== outputs under $OUT ==="
ls -R "$OUT"/*/timed.mp4 2>/dev/null
echo "Done $(date)"
