#!/bin/bash
#SBATCH --job-name=loo-dims-vid20-1n
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Dimensionality study, 20-diverse-clip videos on ONE node (queues faster than
# 5 nodes). 4 GPUs = one model per GPU, each looping all 20 clips sequentially:
# GPU0 v14, GPU1 loo_f3, GPU2 probe2dim, GPU3 critic8. SMOOTH_DECODE, skip/resume.
set -u -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR HF_HUB_CACHE=$CACHE_DIR HUGGINGFACE_HUB_CACHE=$CACHE_DIR TRANSFORMERS_CACHE=$CACHE_DIR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
ENC=/projects/u6ex/fbots/frodobots_encoded_weu
CAP=/projects/u6ex/fbots/frodobots_captions/train
OUT=eval/loo_dims_20; mkdir -p "$OUT"
CLIPS_FILE="$OUT/clips.txt"
python3 -c "import json;[print(w['zarr_path'].split('/')[-1], w['start']) for w in json.load(open('paper_assets/diverse_clips.json'))['windows'][:20]]" > "$CLIPS_FILE"
mapfile -t CLIPS < "$CLIPS_FILE"
echo "clips: ${#CLIPS[@]}"

# model table (one per GPU): label assignment config manifest
LABELS=(v14 loo_f3 probe2dim critic8)
ASSIGN=(5 6 11 10)
CONFIGS=(configs/causal_lora_diffusion_teacher_v14.yaml
         configs/causal_lora_diffusion_teacher_v14_loo_f3.yaml
         configs/causal_lora_diffusion_teacher_v14_loo_probe2dim.yaml
         configs/causal_lora_diffusion_teacher_v14_loo_critic8_noprobe.yaml)
MANIFS=(logs/v14_balanced_weunz/.ride_manifest.pt
        logs/v14_loo_f3/.ride_manifest.pt
        logs/v14_loo_probe2dim/.ride_manifest.pt
        logs/v14_loo_critic8_noprobe/.ride_manifest.pt)

render_one() {  # gpu label assign config manifest zarr offset clipidx
  local gpu=$1 lab=$2 a=$3 cfg=$4 man=$5 z=$6 s=$7 ci=$8
  local d="$OUT/${lab}_c${ci}"
  [ -f "$d/timed.mp4" ] && { echo "skip $lab c$ci"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu WORLD_SIZE=1 LOCAL_RANK=0 SMOOTH_DECODE=1 \
    python utils/eval_causal_chain.py --assignment_index "$a" --config "$cfg" --manifest "$man" \
      --rank_zarr "$z" --rank_offset "$s" --rank_mode dataset --rank_tag "c$ci" \
      --encoded_root "$ENC" --caption_root "$CAP" --seed 42 --output_dir "$d" > "$d.log" 2>&1 \
    || { echo "FAIL $lab c$ci (see $d.log)"; return 1; }
  local raw; raw=$(ls "$d"/rank0_*/final_causal_rollout_*_annotated.mp4 2>/dev/null | head -1)
  [ -f "$raw" ] && ffmpeg -y -loglevel error -i "$raw" -vf "setpts=N/(24*TB)" -r 24 -an "$d/timed.mp4" && echo "done $lab c$ci"
}

echo "=== 4 models in parallel (one per GPU), each over ${#CLIPS[@]} clips $(date) ==="
for m in 0 1 2 3; do
  ( for ci in "${!CLIPS[@]}"; do read -r z s <<< "${CLIPS[$ci]}"
      render_one "$m" "${LABELS[$m]}" "${ASSIGN[$m]}" "${CONFIGS[$m]}" "${MANIFS[$m]}" "$z" "$s" "$ci"
    done ) &
done
wait

echo "=== stitch 2x2 grids (TL v14, TR loo_f3, BL probe2dim, BR critic8) $(date) ==="
G="$OUT/grids"; mkdir -p "$G"
for ci in "${!CLIPS[@]}"; do
  a="$OUT/v14_c${ci}/timed.mp4";       b="$OUT/loo_f3_c${ci}/timed.mp4"
  c="$OUT/probe2dim_c${ci}/timed.mp4"; d="$OUT/critic8_c${ci}/timed.mp4"
  [ -f "$a" ] && [ -f "$b" ] && [ -f "$c" ] && [ -f "$d" ] && \
    ffmpeg -y -loglevel error -i "$a" -i "$b" -i "$c" -i "$d" \
      -filter_complex "[0:v][1:v][2:v][3:v]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0" "$G/grid_c${ci}.mp4"
done
echo "grids: $(ls "$G"/*.mp4 2>/dev/null | wc -l) / ${#CLIPS[@]}"
echo "Done $(date)"
