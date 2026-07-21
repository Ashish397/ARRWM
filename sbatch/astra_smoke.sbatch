#!/bin/bash
#SBATCH --job-name=astra-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Astra (ICLR26, Wan1.3B AR-denoising WM) smoke: r08+r01 x {F=cam1, R=cam3},
# 8s (40 latent frames = 160f @20fps), real-video seeding (13 frames = 4 latents).
set -e
ARR=/scratch/u6ex/as1748.u6ex/ARRWM
AS=$ARR/third_party/Astra
WAN=/scratch/u6ex/as1748.u6ex/frodobots/Wan2.1-T2V-1.3B
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache TMPDIR=/tmp

cd $ARR
PYTHONPATH=$ARR SW_WINDOWS="8:1" python utils/decode_seed_windows.py
python - <<'PYEOF'
import imageio, numpy as np, os
from PIL import Image
ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
for wi in (8, 1):
    p = f"{ARR}/analysis/eval_final/seed65/seed65_r{wi:02d}_f0.png"
    if not os.path.exists(p):
        r = imageio.get_reader(f"{ARR}/analysis/eval_final/seed65/seed65_r{wi:02d}.mp4")
        Image.fromarray(np.asarray(r.get_data(0))).save(p)
        r.close()
print("seed pngs OK")
PYEOF

cd $AS
export PYTHONPATH=$AS/_shims:$AS
OUTROOT=$ARR/logs/eval_final/baseline_smoke/astra
mkdir -p $OUTROOT
PROMPT="A first-person view from a small delivery robot on a sidewalk in a residential street. Houses with fences and gardens line the road, cars are parked along the curb, and trees cast soft shadows in bright daytime light. The camera moves smoothly through the scene."

for W in 8 1; do
  WW=$(printf "%02d" $W)
  for CT in 1 3; do
    [ "$CT" = "1" ] && D=F || D=R
    OUTP=$OUTROOT/astra_r${WW}_${D}.mp4
    [ -f $OUTP ] && continue
    T0=$(date +%s)
    $AS/_venv/bin/python scripts/infer_demo.py \
      --condition_image $ARR/analysis/eval_final/seed65/seed65_r${WW}_f0.png \
      --condition_video $ARR/analysis/eval_final/seed65/seed65_r${WW}.mp4 \
      --start_frame 0 \
      --initial_condition_frames 4 \
      --frames_per_generation 8 \
      --total_frames_to_generate 40 \
      --dit_path $AS/models/Astra/checkpoints/diffusion_pytorch_model.ckpt \
      --wan_model_path $WAN \
      --prompt "$PROMPT" \
      --modality_type sekai \
      --cam_type $CT \
      --output_path $OUTP
    T1=$(date +%s)
    echo "[astra] saved astra_r${WW}_${D}.mp4 gen_time=$((T1-T0))s"
  done
done
echo "ASTRA-SMOKE done $(date)"
