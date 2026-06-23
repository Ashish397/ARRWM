#!/bin/bash
#SBATCH --job-name=loo-dims-videos-20
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Dimensionality study, FULL 20-diverse-clip video set: v14, loo_f3, probe2dim,
# critic8. 5 nodes x 4 GPUs = 20 GPUs: one config per pass (all 20 clips render
# at once, one clip per GPU). 4 config passes. SMOOTH_DECODE. Skip/resume.
set -u -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm

export ENC=/projects/u6ex/fbots/frodobots_encoded_weu
export CAP=/projects/u6ex/fbots/frodobots_captions/train
export OUT=eval/loo_dims_20
mkdir -p "$OUT"
export CLIPS_FILE="$OUT/clips.txt"
python3 -c "import json;[print(w['zarr_path'].split('/')[-1], w['start']) for w in json.load(open('paper_assets/diverse_clips.json'))['windows'][:20]]" > "$CLIPS_FILE"
NCLIP=$(wc -l < "$CLIPS_FILE"); echo "clips: $NCLIP"

# config table (label assignment config manifest) -- 4 passes
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

for k in "${!LABELS[@]}"; do
  export CFG_LABEL="${LABELS[$k]}" CFG_A="${ASSIGN[$k]}" CFG_CONFIG="${CONFIGS[$k]}" CFG_MANIFEST="${MANIFS[$k]}"
  echo "=== pass $k: $CFG_LABEL  ($NCLIP clips across 20 GPUs)  $(date) ==="
  srun --ntasks=20 --ntasks-per-node=4 --gpus-per-node=4 bash utils/ar_render_task.sh || echo "pass $CFG_LABEL had task failures (resume will catch them)"
done

echo "=== stitch 2x2 grids (TL v14, TR loo_f3, BL probe2dim, BR critic8) $(date) ==="
G="$OUT/grids"; mkdir -p "$G"
for ci in $(seq 0 $((NCLIP - 1))); do
  a="$OUT/v14_c${ci}/timed.mp4";       b="$OUT/loo_f3_c${ci}/timed.mp4"
  c="$OUT/probe2dim_c${ci}/timed.mp4"; d="$OUT/critic8_c${ci}/timed.mp4"
  [ -f "$a" ] && [ -f "$b" ] && [ -f "$c" ] && [ -f "$d" ] && \
    ffmpeg -y -loglevel error -i "$a" -i "$b" -i "$c" -i "$d" \
      -filter_complex "[0:v][1:v][2:v][3:v]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0" "$G/grid_c${ci}.mp4"
done
echo "=== grids ==="; ls "$G"/*.mp4 2>/dev/null | wc -l
echo "Done $(date)"
