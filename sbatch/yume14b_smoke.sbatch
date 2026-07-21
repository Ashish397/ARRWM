#!/bin/bash
#SBATCH --job-name=yume14b-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
set -e
ARR=/scratch/u6ex/as1748.u6ex/ARRWM
YU=$ARR/third_party/YUME
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache TMPDIR=/tmp
export TOKENIZERS_PARALLELISM=false

# weights pre-staged on login node (Yume-I2V-540P = full non-distilled 14B, Yume 1.0)
cd $YU
ls Yume-I2V-540P/ | head

# seed jpg dirs (shared with 5B smoke; no-op if present)
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
    p = f"{d}/seed.jpg"
    if not os.path.exists(p):
        r = imageio.get_reader(f"{ARR}/analysis/eval_final/seed65/seed65_r{wi:02d}.mp4")
        Image.fromarray(np.asarray(r.get_data(0))).save(p, quality=95)
        r.close()
print("jpg dirs OK")
EOF

# smoke: 2 windows x {F,R}, 8 segments, FULL 14B @ 50 euler steps, 544x960
cd $YU
export PYTHONPATH=$YU/_shims:$YU
OUTROOT=$ARR/logs/eval_final/baseline_smoke/yume14b
mkdir -p $OUTROOT
for W in 8 1; do
  WW=$(printf "%02d" $W)
  for D in F R; do
    [ -f $OUTROOT/yume14b_r${WW}_${D}.mp4 ] && continue
    RUNOUT=$OUTROOT/run_r${WW}_${D}
    mkdir -p $RUNOUT
    T0=$(date +%s)
    $YU/_venv/bin/python -m torch.distributed.run --nproc_per_node 1 --master_port 29719 \
      fastvideo/sample/sample.py \
      --seed 42 \
      --gradient_checkpointing \
      --train_batch_size=1 \
      --max_sample_steps=600000 \
      --mixed_precision="bf16" \
      --allow_tf32 \
      --video_output_dir="$RUNOUT" \
      --caption_path="./caption_ours14_${D}.txt" \
      --test_data_dir="./val" \
      --num_euler_timesteps 50 \
      --rand_num_img 0.6 \
      --jpg_dir="./jpg_r${WW}/" \
      --t5_cpu
    T1=$(date +%s)
    LAST=$(ls -1 $RUNOUT/*.mp4 | python -c "
import sys, re
fs = [l.strip() for l in sys.stdin]
fs.sort(key=lambda f: int(re.search(r'_(\d+)\.mp4$', f).group(1)))
print(fs[-1])")
    cp "$LAST" $OUTROOT/yume14b_r${WW}_${D}.mp4
    echo "[yume14b] saved yume14b_r${WW}_${D}.mp4 gen_time=$((T1-T0))s from $LAST"
  done
done
echo "YUME14B-SMOKE done $(date)"
