#!/bin/bash
# Auto-release held smoke jobs as their weight repos finish (fetch_weights.log
# emits "[fetch] REPO DONE <repo>"). Also: siglip rearrange for WorldPlay and
# vlm-bench submission once its two VLMs are staged. Exits when all handled.
ARR=/scratch/u6ex/as1748.u6ex/ARRWM
T=$ARR/third_party
LOG=$ARR/.monitor/fetch_weights.log
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm

done_repo() { grep -q "REPO DONE $1" $LOG 2>/dev/null; }
rel() {  # rel <jobname>
  local jid=$(squeue --me -h -o "%i %j %R" | awk -v n="$1" '$2==n && $0 ~ /JobHeldUser/ {print $1}')
  [ -n "$jid" ] && scontrol release $jid && echo "[release] $1 ($jid)"
}

mg2=0; mg2b=0; vista=0; wplay=0; yume5=0; yume14=0; vlm=0
for i in $(seq 1 720); do
  if [ $mg2 = 0 ] && done_repo "Skywork/Matrix-Game-2.0"; then rel mg2-smoke; mg2=1; fi
  if [ $mg2b = 0 ] && done_repo "Skywork/Matrix-Game-2.0"; then rel mg2base-smoke; mg2b=1; fi
  if [ $vista = 0 ] && done_repo "OpenDriveLab/Vista"; then rel vista-smoke; vista=1; fi
  if [ $wplay = 0 ] && done_repo "tencent/HunyuanVideo-1.5" && done_repo "google/byt5-small" \
     && done_repo "tencent/HY-WorldPlay" && done_repo "google/siglip-so400m-patch14-384"; then
    MODEL=$T/HY-WorldPlay/weights/HunyuanVideo-1.5
    if [ ! -d $MODEL/vision_encoder/siglip/image_encoder ]; then
      MODEL=$MODEL RAW=$T/HY-WorldPlay/weights/_siglip_raw python - <<'EOF'
import os
from transformers import SiglipVisionModel, SiglipImageProcessor
root = os.environ["MODEL"] + "/vision_encoder/siglip"
raw = os.environ["RAW"]
SiglipVisionModel.from_pretrained(raw).save_pretrained(root + "/image_encoder")
SiglipImageProcessor.from_pretrained(raw).save_pretrained(root + "/feature_extractor")
print("[release] siglip rearranged")
EOF
    fi
    QSNAP=$(ls -d /scratch/u6ex/as1748.u6ex/frodobots/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/*/ | head -1)
    [ -e $MODEL/text_encoder/llm ] || ln -s "$QSNAP" $MODEL/text_encoder/llm
    rel wplay-smoke; wplay=1
  fi
  if [ $yume5 = 0 ] && done_repo "stdstu123/Yume-5B-720P" && done_repo "OpenGVLab/InternVL3-2B-Instruct"; then
    rel yume-smoke; yume5=1
  fi
  if [ $yume14 = 0 ] && done_repo "stdstu123/Yume-I2V-540P"; then rel yume14b-smoke; yume14=1; fi
  if [ $vlm = 0 ] && done_repo "OpenGVLab/InternVL3-8B" && done_repo "ByteDance/Sa2VA-4B"; then
    cd $ARR
    JV=$(sbatch --parsable --export=ALL,VB_INTERNVL=$T/_vlm/InternVL3-8B,VB_SA2VA=$T/_vlm/Sa2VA-4B sbatch/vlm_bench.sbatch 2>/dev/null) \
      && echo "[release] vlm-bench resubmitted as $JV" || echo "[release] vlm-bench sbatch missing - submit manually"
    vlm=1
  fi
  if [ $mg2 = 1 ] && [ $vista = 1 ] && [ $wplay = 1 ] && [ $yume5 = 1 ] && [ $yume14 = 1 ] && [ $vlm = 1 ]; then
    echo "[release] all jobs released"; exit 0
  fi
  sleep 60
done
echo "[release] daemon timeout - remaining: mg2=$mg2 vista=$vista wplay=$wplay yume5=$yume5 yume14=$yume14 vlm=$vlm"
