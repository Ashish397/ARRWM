#!/bin/bash
#SBATCH --job-name=wplay-arl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=14:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# WorldPlay NON-DISTILLED (ar_rl, 50 steps CFG6) production shard.
# sp=4 (one clip at a time across 4 GPUs, as smoked: ~690s/clip).
# WP_SHARD 0..3 -> windows [8*SHARD, 8*SHARD+7], all 8 dirs = 64 clips (~12.2h).
set -e
ARR=/scratch/u6ex/as1748.u6ex/ARRWM
WP=$ARR/third_party/HY-WorldPlay
MODEL=$WP/weights/HunyuanVideo-1.5
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache TMPDIR=/tmp
SHARD=${WP_SHARD:-0}

cd $ARR
ALLW=$(seq -s: 0 31)
PYTHONPATH=$ARR SW_WINDOWS="$ALLW" python utils/decode_seed_windows.py
python utils/wplay_poses.py

cd $WP
export PYTHONPATH=$WP/_compat:$WP
OUTROOT=$ARR/logs/eval_final/A_worldplay_arl
mkdir -p $OUTROOT
PROMPT="a first-person view from a small delivery robot driving on a sidewalk in a residential street, houses and parked cars, daytime, clear weather."

for W in $(seq $((8*SHARD)) $((8*SHARD+7))); do
  WW=$(printf "%02d" $W)
  for D in F FR R BR B BL L FL; do
    [ -f $OUTROOT/worldplay_r${WW}_${D}.mp4 ] && continue
    RUNOUT=$OUTROOT/run_r${WW}_${D}
    mkdir -p $RUNOUT
    T0=$(date +%s)
    $WP/_venv/bin/python -m torch.distributed.run \
      --nproc_per_node=4 --master_port=$((29700+SHARD)) hyvideo/generate.py \
      --prompt "$PROMPT" --resolution 480p \
      --model_path $MODEL \
      --action_ckpt $WP/weights/ar_rl_model/diffusion_pytorch_model.safetensors \
      --model_type ar \
      --image_path $ARR/analysis/eval_final/seed65/seed65_r${WW}_f0.png \
      --pose $WP/poses_ours/pose_${D}.json \
      --video_length 189 --width 832 --height 480 \
      --num_inference_steps 50 --seed 1 --sr false --rewrite false \
      --transformer_resident_ar_rollout true \
      --output_path $RUNOUT > $ARR/logs/wplayarl_s${SHARD}.log 2>&1 || { echo "[wplay-arl] FAIL r${WW}_${D}"; continue; }
    T1=$(date +%s)
    mv $RUNOUT/gen.mp4 $OUTROOT/worldplay_r${WW}_${D}.mp4
    echo "[wplay-arl] saved worldplay_r${WW}_${D}.mp4 gen_time=$((T1-T0))s"
  done
done
echo "WPLAY-ARL shard $SHARD done $(date)"
