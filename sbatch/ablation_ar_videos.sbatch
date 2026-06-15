#!/bin/bash
#SBATCH --job-name=ablation-ar-videos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Render AR rollouts for v14 + 4 LOO configs on the 20 diverse clips.
# Phase 1: 4 LOO configs, one per GPU (parallel). Phase 2: v14, 4 clips at a time.
# Each output re-timed to ~4s (24 fps). Skip/resume on existing timed.mp4.
# Reuses the existing AR machinery (eval_causal_chain.py) UNCHANGED.
set -u -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR HF_HUB_CACHE=$CACHE_DIR HUGGINGFACE_HUB_CACHE=$CACHE_DIR TRANSFORMERS_CACHE=$CACHE_DIR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
ENC=/projects/u6ex/fbots/frodobots_encoded_weu
CAP=/projects/u6ex/fbots/frodobots_captions/train
OUT=eval/ar_ablation; mkdir -p "$OUT"

# --- clip list: prefer diverse_clips.json; wait up to 30 min; fallback to interesting top-20 ---
CJ=paper_assets/diverse_clips.json
for i in $(seq 1 60); do [ -f "$CJ" ] && break; echo "waiting for $CJ ($i)"; sleep 30; done
[ -f "$CJ" ] || CJ=paper_assets/interesting_windows.json
mapfile -t CLIPS < <(python3 -c "import json;[print(w['zarr_path'].split('/')[-1], w['start']) for w in json.load(open('$CJ'))['windows'][:20]]")
echo "using clip list $CJ : ${#CLIPS[@]} clips"

# config table: assignment | label | config | manifest
LOO_A=(6 7 8 9)
LOO_L=(loo_f3 loo_tokens loo_adaln loo_f2)
LOO_C=(configs/causal_lora_diffusion_teacher_v14_loo_f3.yaml configs/causal_lora_diffusion_teacher_v14_loo_tokens.yaml configs/causal_lora_diffusion_teacher_v14_loo_adaln.yaml configs/causal_lora_diffusion_teacher_v14_loo_f2.yaml)
LOO_M=(logs/v14_loo_f3/.ride_manifest.pt logs/v14_loo_tokens/.ride_manifest.pt logs/v14_loo_adaln/.ride_manifest.pt logs/v14_loo_f2/.ride_manifest.pt)

render_one() {  # gpu assignment config manifest zarr start clipidx label
  local gpu=$1 a=$2 cfg=$3 man=$4 z=$5 s=$6 ci=$7 lab=$8
  local d="$OUT/${lab}_c${ci}"; local timed="$d/timed.mp4"
  if [ -f "$timed" ]; then echo "skip $lab c$ci (exists)"; return 0; fi
  CUDA_VISIBLE_DEVICES=$gpu WORLD_SIZE=1 LOCAL_RANK=0 \
    python utils/eval_causal_chain.py --assignment_index "$a" --config "$cfg" --manifest "$man" \
      --rank_zarr "$z" --rank_offset "$s" --rank_mode dataset --rank_tag "c$ci" \
      --encoded_root "$ENC" --caption_root "$CAP" --seed 42 --output_dir "$d" \
      > "$d.log" 2>&1 || { echo "FAIL $lab c$ci (see $d.log)"; return 1; }
  local raw; raw=$(ls "$d"/rank0_*/final_causal_rollout_*_raw.mp4 2>/dev/null | head -1)
  [ -f "$raw" ] && ffmpeg -y -loglevel error -i "$raw" -vf "setpts=N/(24*TB)" -r 24 -an "$timed" && echo "done $lab c$ci"
}

echo "=== Phase 1: 4 LOO configs, one per GPU ==="
for gi in 0 1 2 3; do
  ( for ci in "${!CLIPS[@]}"; do read -r z s <<< "${CLIPS[$ci]}"
      render_one "$gi" "${LOO_A[$gi]}" "${LOO_C[$gi]}" "${LOO_M[$gi]}" "$z" "$s" "$ci" "${LOO_L[$gi]}"
    done ) &
done
wait
echo "=== Phase 2: v14, 4 clips at a time ==="
for ci in "${!CLIPS[@]}"; do read -r z s <<< "${CLIPS[$ci]}"
  render_one "$((ci % 4))" 5 configs/causal_lora_diffusion_teacher_v14.yaml logs/v14_balanced_weunz/.ride_manifest.pt "$z" "$s" "$ci" v14 &
  if (( (ci + 1) % 4 == 0 )); then wait; fi
done
wait

echo "=== Phase 3: per-clip 2x2 grid of the 4 LOO configs (TL loo_adaln, TR loo_tokens, BL loo_f3, BR loo_f2) ==="
G="$OUT/grids"; mkdir -p "$G"
for ci in "${!CLIPS[@]}"; do
  a="$OUT/loo_adaln_c${ci}/timed.mp4"; b="$OUT/loo_tokens_c${ci}/timed.mp4"
  c="$OUT/loo_f3_c${ci}/timed.mp4";   d="$OUT/loo_f2_c${ci}/timed.mp4"
  if [ -f "$a" ] && [ -f "$b" ] && [ -f "$c" ] && [ -f "$d" ]; then
    ffmpeg -y -loglevel error -i "$a" -i "$b" -i "$c" -i "$d" \
      -filter_complex "[0:v][1:v][2:v][3:v]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0" "$G/grid_c${ci}.mp4" \
      && echo "grid c$ci"
  fi
done

echo "=== Phase 4: AR-rollout quality scoring (PSNR/SSIM/LPIPS/MUSIQ + per-position drift, per config) ==="
CUDA_VISIBLE_DEVICES=0 python utils/score_ar_rollouts.py || echo "AR scoring failed (non-fatal)"
echo "Done $(date)"
