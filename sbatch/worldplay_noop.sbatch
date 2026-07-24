#!/bin/bash
#SBATCH --job-name=wplay-noop
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# WorldPlay DISTILLED production: 32 windows x 8 dirs, 8s (189f, 48 latents),
# few_step=4, no CFG, sp=1 per video, 4 videos in parallel (one per GPU).
set -e
ARR=/scratch/u6ex/as1748.u6ex/ARRWM
WP=$ARR/third_party/HY-WorldPlay
MODEL=$WP/weights/HunyuanVideo-1.5
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache TMPDIR=/tmp

cd $ARR
ALLW=$(seq -s: 0 31)
PYTHONPATH=$ARR SW_WINDOWS="$ALLW" python utils/decode_seed_windows.py
python utils/wplay_poses.py
python - <<'EOF'
import imageio, numpy as np, os
from PIL import Image
ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
for wi in range(32):
    p = f"{ARR}/analysis/eval_final/seed65/seed65_r{wi:02d}_f0.png"
    if not os.path.exists(p):
        r = imageio.get_reader(f"{ARR}/analysis/eval_final/seed65/seed65_r{wi:02d}.mp4")
        Image.fromarray(np.asarray(r.get_data(0))).save(p)
        r.close()
print("32 seed pngs ready")
EOF

cd $WP
export PYTHONPATH=$WP/_compat:$WP
OUTROOT=$ARR/logs/eval_final/A_worldplay_nullact
mkdir -p $OUTROOT
PROMPT="a first-person view from a small delivery robot driving on a sidewalk in a residential street, houses and parked cars, daytime, clear weather."

runshard() {  # gpu windows...
  local GPU=$1; shift
  for W in "$@"; do
    WW=$(printf "%02d" $W)
    for D in NOOP; do
      [ -f $OUTROOT/worldplay_r${WW}_${D}.mp4 ] && continue
      RUNOUT=$OUTROOT/run_r${WW}_${D}
      mkdir -p $RUNOUT
      T0=$(date +%s)
      CUDA_VISIBLE_DEVICES=$GPU $WP/_venv/bin/python -m torch.distributed.run \
        --nproc_per_node=1 --master_port=$((29650+GPU)) hyvideo/generate.py \
        --prompt "$PROMPT" --resolution 480p \
        --model_path $MODEL \
        --action_ckpt $WP/weights/ar_distilled_action_model/model.safetensors \
        --model_type ar --few_step true --num_inference_steps 4 \
        --image_path $ARR/analysis/eval_final/seed65/seed65_r${WW}_f0.png \
        --pose $WP/poses_ours/pose_${D}.json \
        --video_length 189 --width 832 --height 480 \
        --seed 1 --sr false --rewrite false \
        --transformer_resident_ar_rollout true \
        --output_path $RUNOUT > $ARR/logs/wplaynoop_g${GPU}.log 2>&1 || { echo "[wplay-noop] FAIL r${WW}_${D} (gpu $GPU)"; continue; }
      T1=$(date +%s)
      mv $RUNOUT/gen.mp4 $OUTROOT/worldplay_r${WW}_${D}.mp4
      echo "[wplay-noop] saved worldplay_r${WW}_${D}.mp4 gen_time=$((T1-T0))s"
    done
  done
}
runshard 0 0 1 2 3 4 5 6 7 &
runshard 1 8 9 10 11 12 13 14 15 &
runshard 2 16 17 18 19 20 21 22 23 &
runshard 3 24 25 26 27 28 29 30 31 &
wait
echo "WPLAY-FULL done $(date)"
