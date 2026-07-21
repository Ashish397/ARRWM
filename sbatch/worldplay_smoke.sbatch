#!/bin/bash
#SBATCH --job-name=wplay-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
set -e
ARR=/scratch/u6ex/as1748.u6ex/ARRWM
WP=$ARR/third_party/HY-WorldPlay
MODEL=$WP/weights/HunyuanVideo-1.5
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache TMPDIR=/tmp

mkdir -p $MODEL/text_encoder $MODEL/vision_encoder

# 1) HunyuanVideo-1.5 base: vae, scheduler, 480p_i2v transformer
if [ ! -d $MODEL/transformer/480p_i2v ]; then
  hf download tencent/HunyuanVideo-1.5 --local-dir $MODEL \
    --include "vae/*" "scheduler/*" "transformer/480p_i2v/*"
fi

# 2) text encoders: Qwen2.5-VL (symlink to cached snapshot), byt5-small
if [ ! -e $MODEL/text_encoder/llm ]; then
  QSNAP=$(ls -d $HF_HOME/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/*/ | head -1)
  ln -s "$QSNAP" $MODEL/text_encoder/llm
fi
if [ ! -d $MODEL/text_encoder/byt5-small ]; then
  hf download google/byt5-small --local-dir $MODEL/text_encoder/byt5-small
fi

# 3) Glyph-SDXL-v2 (ModelScope direct; only 3 files needed)
GL=$MODEL/text_encoder/Glyph-SDXL-v2
mkdir -p $GL/assets $GL/checkpoints
for f in assets/color_idx.json assets/multilingual_10-lang_idx.json checkpoints/byt5_model.pt; do
  if [ ! -s $GL/$f ]; then
    curl -fL --retry 3 -o $GL/$f \
      "https://modelscope.cn/models/AI-ModelScope/Glyph-SDXL-v2/resolve/master/$f"
  fi
done
ls -la $GL/assets $GL/checkpoints

# 4) vision encoder: FLUX.1-Redux is gated -> google/siglip-so400m-patch14-384
#    rearranged into the expected {image_encoder, feature_extractor} layout
if [ ! -d $MODEL/vision_encoder/siglip/image_encoder ]; then
  MODEL=/scratch/u6ex/as1748.u6ex/ARRWM/third_party/HY-WorldPlay/weights/HunyuanVideo-1.5 python - <<'EOF'
import os
from transformers import SiglipVisionModel, SiglipImageProcessor
root = os.environ["MODEL"] + "/vision_encoder/siglip"
m = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384")
m.save_pretrained(root + "/image_encoder")
p = SiglipImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
p.save_pretrained(root + "/feature_extractor")
print("siglip substitute saved")
EOF
fi

# 5) WorldPlay action ckpt: ar_rl_model (best non-distilled, 50-step CFG6)
if [ ! -f $WP/weights/ar_rl_model/diffusion_pytorch_model.safetensors ]; then
  hf download tencent/HY-WorldPlay --local-dir $WP/weights --include "ar_rl_model/*"
fi
ls -la $WP/weights/ar_rl_model/

# 5b) siglip image_encoder weights (login-node extraction OOMs; do it here)
IE=$MODEL/vision_encoder/siglip/image_encoder/model.safetensors
if [ ! -s $IE ]; then
  python - <<'PYEOF'
import os
from safetensors import safe_open
from safetensors.torch import save_file
W = "/scratch/u6ex/as1748.u6ex/ARRWM/third_party/HY-WorldPlay/weights"
tensors = {}
with safe_open(f"{W}/_siglip_raw/model.safetensors", framework="pt") as f:
    for k in f.keys():
        if k.startswith("vision_model."):
            tensors[k] = f.get_tensor(k)
out = f"{W}/HunyuanVideo-1.5/vision_encoder/siglip/image_encoder/model.safetensors"
save_file(tensors, out)
print("siglip image_encoder written:", os.path.getsize(out)/1e6, "MB,", len(tensors), "tensors")
PYEOF
fi

# 6) seed frames
cd $ARR
PYTHONPATH=$ARR SW_WINDOWS="8:1" python utils/decode_seed_windows.py
python - <<'EOF'
import imageio, numpy as np
from PIL import Image
ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
for wi in (8, 1):
    r = imageio.get_reader(f"{ARR}/analysis/eval_final/seed65/seed65_r{wi:02d}.mp4")
    Image.fromarray(np.asarray(r.get_data(0))).save(f"{ARR}/analysis/eval_final/seed65/seed65_r{wi:02d}_f0.png")
    r.close()
print("seed pngs OK")
EOF

# 7) smoke: 2 windows x {F=w-63, R=right-63}, 253 frames (64 latents), 50 steps CFG
cd $WP
export PYTHONPATH=$WP/_compat:$WP
OUTROOT=$ARR/logs/eval_final/baseline_smoke/worldplay
mkdir -p $OUTROOT
PROMPT="a first-person view from a small delivery robot driving on a sidewalk in a residential street, houses and parked cars, daytime, clear weather."
for W in 8 1; do
  for D in F R; do
    if [ "$D" = "F" ]; then POSE="w-47"; else POSE="right-47"; fi
    WW=$(printf "%02d" $W)
    [ -f $OUTROOT/worldplay_r${WW}_${D}.mp4 ] && continue
    T0=$(date +%s)
    $WP/_venv/bin/python -m torch.distributed.run --nproc_per_node=4 --master_port=29612 hyvideo/generate.py \
      --prompt "$PROMPT" --resolution 480p \
      --model_path $MODEL \
      --action_ckpt $WP/weights/ar_rl_model/diffusion_pytorch_model.safetensors \
      --model_type ar --image_path $ARR/analysis/eval_final/seed65/seed65_r${WW}_f0.png \
      --pose "$POSE" --video_length 189 --width 832 --height 480 \
      --num_inference_steps 50 --seed 1 --sr false --rewrite false \
      --transformer_resident_ar_rollout true \
      --output_path $OUTROOT/r${WW}_${D}
    T1=$(date +%s)
    mv $OUTROOT/r${WW}_${D}/gen.mp4 $OUTROOT/worldplay_r${WW}_${D}.mp4
    echo "[wplay] saved worldplay_r${WW}_${D}.mp4 gen_time=$((T1-T0))s"
  done
done
echo "WPLAY-SMOKE done $(date)"
