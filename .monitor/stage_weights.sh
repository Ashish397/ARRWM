#!/bin/bash
# Stage all baseline + VLM weights with stall-resistant retries.
# hf downloads resume from .incomplete files, so timeout-kill + retry converges.
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate && conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache
export HF_HUB_DOWNLOAD_TIMEOUT=30
export HF_HUB_DISABLE_XET=1
T=/scratch/u6ex/as1748.u6ex/ARRWM/third_party

dl() {
  local n=0
  until timeout 600 hf download "$@" > /dev/null 2>&1; do
    n=$((n+1)); echo "[stage][retry $n] hf download $1"
    [ $n -ge 30 ] && { echo "[stage][GIVEUP] $1"; return 1; }
    sleep 5
  done
  echo "[stage][OK] $1"
}

# Matrix-Game 2.0 (~12GB, partially done)
cd $T/Matrix-Game/Matrix-Game-2
[ -f Matrix-Game-2.0/base_distilled_model/base_distill.safetensors ] || dl Skywork/Matrix-Game-2.0 --local-dir Matrix-Game-2.0 \
  --include "base_distilled_model/*" "Wan2.1_VAE.pth" "models_clip*" "xlm-roberta-large/*" "*.json"

# Vista (~10GB)
cd $T/Vista && mkdir -p ckpts
[ -f ckpts/vista.safetensors ] || dl OpenDriveLab/Vista vista.safetensors --local-dir ckpts/

# HY-WorldPlay
WP=$T/HY-WorldPlay; MODEL=$WP/weights/HunyuanVideo-1.5
mkdir -p $MODEL/text_encoder $MODEL/vision_encoder
[ -d $MODEL/transformer/480p_i2v ] || dl tencent/HunyuanVideo-1.5 --local-dir $MODEL --include "vae/*" "scheduler/*" "transformer/480p_i2v/*"
[ -d $MODEL/text_encoder/byt5-small ] || dl google/byt5-small --local-dir $MODEL/text_encoder/byt5-small
[ -f $WP/weights/ar_rl_model/diffusion_pytorch_model.safetensors ] || dl tencent/HY-WorldPlay --local-dir $WP/weights --include "ar_rl_model/*"
GL=$MODEL/text_encoder/Glyph-SDXL-v2; mkdir -p $GL/assets $GL/checkpoints
for f in assets/color_idx.json assets/multilingual_10-lang_idx.json checkpoints/byt5_model.pt; do
  [ -s $GL/$f ] || curl -fL --retry 10 --retry-all-errors -C - --max-time 900 -sS -o $GL/$f \
    "https://modelscope.cn/models/AI-ModelScope/Glyph-SDXL-v2/resolve/master/$f" && echo "[stage][OK] glyph $f"
done
if [ ! -d $MODEL/vision_encoder/siglip/image_encoder ]; then
  dl google/siglip-so400m-patch14-384
  MODEL=$MODEL python - <<'EOF'
import os
from transformers import SiglipVisionModel, SiglipImageProcessor
root = os.environ["MODEL"] + "/vision_encoder/siglip"
SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384").save_pretrained(root + "/image_encoder")
SiglipImageProcessor.from_pretrained("google/siglip-so400m-patch14-384").save_pretrained(root + "/feature_extractor")
print("[stage][OK] siglip substitute")
EOF
fi

# Yume (~40GB)
cd $T/YUME
[ -f Yume-5B-720P/diffusion_pytorch_model.safetensors ] || dl stdstu123/Yume-5B-720P --local-dir Yume-5B-720P --exclude "demo.mp4"
[ -d InternVL3-2B-Instruct/.cache ] || dl OpenGVLab/InternVL3-2B-Instruct --local-dir InternVL3-2B-Instruct

# VLM judges (into HF cache)
dl OpenGVLab/InternVL3-8B
dl ByteDance/Sa2VA-4B

echo "=== STAGED ==="
du -sh $T/Matrix-Game/Matrix-Game-2/Matrix-Game-2.0 $T/Vista/ckpts $WP/weights \
  $T/YUME/Yume-5B-720P $T/YUME/InternVL3-2B-Instruct \
  $HF_HOME/hub/models--OpenGVLab--InternVL3-8B $HF_HOME/hub/models--ByteDance--Sa2VA-4B 2>/dev/null
