#!/bin/bash
#SBATCH --job-name=yume-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
set -e
ARR=/scratch/u6ex/as1748.u6ex/ARRWM
YU=$ARR/third_party/YUME
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache TMPDIR=/tmp
export TOKENIZERS_PARALLELISM=false

# 1) Yume-5B-720P (~35GB: DiT fp32 + umt5 + Wan2.2 VAE) + InternVL3-2B captioner
cd $YU
if [ ! -f Yume-5B-720P/diffusion_pytorch_model.safetensors ]; then
  hf download stdstu123/Yume-5B-720P --local-dir Yume-5B-720P --exclude "demo.mp4"
fi
if [ ! -d InternVL3-2B-Instruct ]; then
  hf download OpenGVLab/InternVL3-2B-Instruct --local-dir InternVL3-2B-Instruct
fi
ls Yume-5B-720P/

# 2) seed frame jpgs (one dir per window; their loader consumes a jpg folder)
cd $ARR
PYTHONPATH=$ARR SW_WINDOWS="8:1" python utils/decode_seed_windows.py
python - <<'EOF'
import imageio, numpy as np, os
from PIL import Image
ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
YU = f"{ARR}/third_party/YUME"
for wi in (8, 1):
    d = f"{YU}/jpg_r{wi:02d}"
    os.makedirs(d, exist_ok=True)
    r = imageio.get_reader(f"{ARR}/analysis/eval_final/seed65/seed65_r{wi:02d}.mp4")
    Image.fromarray(np.asarray(r.get_data(0))).save(f"{d}/seed.jpg", quality=95)
    r.close()
print("jpg dirs OK")
EOF

# 3) smoke: 2 windows x {F,R}, 8 segments = 256 frames @16fps, 4-step distilled
cd $YU
export PYTHONPATH=$YU/_shims:$YU
OUTROOT=$ARR/logs/eval_final/baseline_smoke/yume_v2
mkdir -p $OUTROOT
PROMPT="First-person view moving along a residential street sidewalk, houses and parked cars, daytime."
for W in 8 1; do
  WW=$(printf "%02d" $W)
  for D in F R; do
    [ -f $OUTROOT/yume_r${WW}_${D}.mp4 ] && continue
    RUNOUT=$OUTROOT/run_r${WW}_${D}
    mkdir -p $RUNOUT
    T0=$(date +%s)
    $YU/_venv/bin/python -m torch.distributed.run --nproc_per_node 1 --master_port 29718 \
      fastvideo/sample/sample_5b.py \
      --seed 43 \
      --gradient_checkpointing \
      --train_batch_size=1 \
      --max_sample_steps=600000 \
      --mixed_precision="bf16" \
      --allow_tf32 \
      --video_output_dir="$RUNOUT" \
      --caption_path="./caption_ours_${D}.txt" \
      --test_data_dir="./val" \
      --num_euler_timesteps 4 \
      --rand_num_img 0.6 \
      --jpg_dir="./jpg_r${WW}/" \
      --prompt "$PROMPT"
    T1=$(date +%s)
    # cumulative saves: the highest step index holds the full rollout
    LAST=$(ls -1 $RUNOUT/*.mp4 | sort -t_ -k99 | python -c "
import sys, re
fs = [l.strip() for l in sys.stdin]
fs.sort(key=lambda f: int(re.search(r'_(\d+)\.mp4$', f).group(1)))
print(fs[-1])")
    cp "$LAST" $OUTROOT/yume_r${WW}_${D}.mp4
    echo "[yume] saved yume_r${WW}_${D}.mp4 gen_time=$((T1-T0))s from $LAST"
  done
done
echo "YUME-SMOKE done $(date)"
