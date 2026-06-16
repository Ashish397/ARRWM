#!/bin/bash
#SBATCH --job-name=ablation-ar-smooth
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# 5 nodes x 4 GPUs = 20 GPUs. One config per "pass": all 20 clips render at once
# (one clip per GPU). 5 config passes (v14 + 4 LOO) => ~5 x one-clip-time total.
# SMOOTH_DECODE=1 => seam-free single-pass decode. Skip/resume per (config,clip).
set -u -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm

export ENC=/projects/u6ex/fbots/frodobots_encoded_weu
export CAP=/projects/u6ex/fbots/frodobots_captions/train
export OUT=eval/ar_ablation_smooth
mkdir -p "$OUT"
export CLIPS_FILE="$OUT/clips.txt"
python3 -c "import json;[print(w['zarr_path'].split('/')[-1], w['start']) for w in json.load(open('paper_assets/diverse_clips.json'))['windows'][:20]]" > "$CLIPS_FILE"
NCLIP=$(wc -l < "$CLIPS_FILE"); echo "clips: $NCLIP"

# config table (label assignment config manifest) -- 5 passes
LABELS=(v14 loo_adaln loo_tokens loo_f3 loo_f2)
ASSIGN=(5 8 7 6 9)
CONFIGS=(configs/causal_lora_diffusion_teacher_v14.yaml configs/causal_lora_diffusion_teacher_v14_loo_adaln.yaml configs/causal_lora_diffusion_teacher_v14_loo_tokens.yaml configs/causal_lora_diffusion_teacher_v14_loo_f3.yaml configs/causal_lora_diffusion_teacher_v14_loo_f2.yaml)
MANIFS=(logs/v14_balanced_weunz/.ride_manifest.pt logs/v14_loo_adaln/.ride_manifest.pt logs/v14_loo_tokens/.ride_manifest.pt logs/v14_loo_f3/.ride_manifest.pt logs/v14_loo_f2/.ride_manifest.pt)

for k in "${!LABELS[@]}"; do
  export CFG_LABEL="${LABELS[$k]}" CFG_A="${ASSIGN[$k]}" CFG_CONFIG="${CONFIGS[$k]}" CFG_MANIFEST="${MANIFS[$k]}"
  echo "=== pass $k: $CFG_LABEL  ($NCLIP clips across 20 GPUs)  $(date) ==="
  srun --ntasks=20 --ntasks-per-node=4 --gpus-per-node=4 bash utils/ar_render_task.sh || echo "pass $CFG_LABEL had task failures (resume will catch them)"
done

echo "=== stitch 2x2 LOO grids + AR scoring (head node) ==="
G="$OUT/grids"; mkdir -p "$G"
for ci in $(seq 0 $((NCLIP - 1))); do
  a="$OUT/loo_adaln_c${ci}/timed.mp4"; b="$OUT/loo_tokens_c${ci}/timed.mp4"
  c="$OUT/loo_f3_c${ci}/timed.mp4";    d="$OUT/loo_f2_c${ci}/timed.mp4"
  [ -f "$a" ] && [ -f "$b" ] && [ -f "$c" ] && [ -f "$d" ] && \
    ffmpeg -y -loglevel error -i "$a" -i "$b" -i "$c" -i "$d" \
      -filter_complex "[0:v][1:v][2:v][3:v]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0" "$G/grid_c${ci}.mp4"
done
AR_OUT="$OUT" CUDA_VISIBLE_DEVICES=0 python utils/score_ar_rollouts.py || echo "scoring failed (non-fatal)"
echo "Done $(date)"
