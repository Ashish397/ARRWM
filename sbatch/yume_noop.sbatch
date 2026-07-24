#!/bin/bash
#SBATCH --job-name=yume-noop
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Yume-5B production: 32 windows x 8 dirs, 8s (4 segments x 32f @16fps = 128f).
# One torchrun per direction (model loaded once, 4-way data parallel over 32 seeds);
# InternVL auto-caption (no --prompt) appends scene description per seed.
set -e
ARR=/scratch/u6ex/as1748.u6ex/ARRWM
YU=$ARR/third_party/YUME
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache TMPDIR=/tmp
export TOKENIZERS_PARALLELISM=false

# seeds: all 32 window frames as jpgs named rNN.jpg
cd $ARR
ALLW=$(seq -s: 0 31)
PYTHONPATH=$ARR SW_WINDOWS="$ALLW" python utils/decode_seed_windows.py
python - <<'EOF'
import imageio, numpy as np, os
from PIL import Image
ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
d = f"{ARR}/third_party/YUME/jpg_all32"
os.makedirs(d, exist_ok=True)
for wi in range(32):
    p = f"{d}/r{wi:02d}.jpg"
    if not os.path.exists(p):
        r = imageio.get_reader(f"{ARR}/analysis/eval_final/seed65/seed65_r{wi:02d}.mp4")
        Image.fromarray(np.asarray(r.get_data(0))).save(p, quality=95)
        r.close()
print("32 seed jpgs ready")
EOF

cd $YU
export PYTHONPATH=$YU/_shims:$YU
OUTROOT=$ARR/logs/eval_final/A_yume_nullact
mkdir -p $OUTROOT
for D in NOOP; do
  RUNOUT=$OUTROOT/run_$D
  mkdir -p $RUNOUT
  # skip direction if all 32 finals already exist
  DONE=$(ls $OUTROOT/yume_r*_${D}.mp4 2>/dev/null | wc -l)
  [ "$DONE" -ge 32 ] && { echo "[yume-noop] $D complete, skip"; continue; }
  T0=$(date +%s)
  $YU/_venv/bin/python -m torch.distributed.run --nproc_per_node 4 --master_port 29720 \
    fastvideo/sample/sample_5b.py \
    --seed 43 \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --max_sample_steps=600000 \
    --mixed_precision="bf16" \
    --allow_tf32 \
    --video_output_dir="$RUNOUT" \
    --caption_path="./caption_prod_${D}.txt" \
    --test_data_dir="./val" \
    --num_euler_timesteps 4 \
    --rand_num_img 0.6 \
    --jpg_dir="./jpg_all32/"
  T1=$(date +%s)
  echo "[yume-noop] $D torchrun done in $((T1-T0))s"
  # collect: per window, the highest-step cumulative file = full 4-segment rollout
  python - <<PYEOF
import glob, re, os, shutil
RUNOUT = "$RUNOUT"; OUTROOT = "$OUTROOT"; D = "$D"
for wi in range(32):
    cands = glob.glob(f"{RUNOUT}/r{wi:02d}_*.mp4")
    if not cands:
        print(f"[collect] MISSING r{wi:02d}_{D}"); continue
    cands.sort(key=lambda f: int(re.search(r"_(\d+)\.mp4$", f).group(1)))
    dst = f"{OUTROOT}/yume_r{wi:02d}_{D}.mp4"
    if not os.path.exists(dst):
        shutil.copy(cands[-1], dst)
print(f"[collect] {D}: {len(glob.glob(f'{OUTROOT}/yume_r*_{D}.mp4'))}/32")
PYEOF
done
echo "YUME-FULL done $(date)"
