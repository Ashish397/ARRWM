#!/bin/bash
# Fetch the official public YUME-5B checkpoint into the shared Isambard path.
# Run once on a network-enabled login/data-transfer node; this submits no job.
set -euo pipefail

R=${ARRWM_REMOTE_ROOT:-/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler}
PY_ROOT=${ARRWM_PYTHON_ROOT:-$R/miniforge3/envs/arrwm}
HF=${HF_CLI:-$PY_ROOT/bin/hf}
MODEL_ID=${YUME_MODEL_ID:-stdstu123/Yume-5B-720P}
REVISION=${YUME_MODEL_REVISION:-main}
TARGET=${YUME_MODEL_DIR:-$R/yume/Yume-5B-720P}
export HF_HOME=${HF_HOME:-$R/frodobots/hf_cache}

test -x "$HF"
mkdir -p "$TARGET" "$HF_HOME"
"$HF" download "$MODEL_ID" --revision "$REVISION" --local-dir "$TARGET"

# The released sampler uses the UMT5 tokenizer by repository ID.  Pre-cache
# only its tokenizer/config files; the 11.4 GB text-encoder weights above are
# loaded from the YUME checkpoint directory.
"$HF" download google/umt5-xxl \
  --include config.json tokenizer_config.json special_tokens_map.json spiece.model tokenizer.json

for asset in \
  diffusion_pytorch_model.safetensors \
  Wan2.2_VAE.pth \
  models_t5_umt5-xxl-enc-bf16.pth \
  config.json; do
  test -s "$TARGET/$asset"
done
test -f "$TARGET/config.yaml"

sha256sum \
  "$TARGET/diffusion_pytorch_model.safetensors" \
  "$TARGET/Wan2.2_VAE.pth" \
  "$TARGET/models_t5_umt5-xxl-enc-bf16.pth" \
  >"$TARGET/ARRWM_SHA256SUMS"
printf 'model_id=%s\nrevision=%s\ndownloaded_at=%s\n' \
  "$MODEL_ID" "$REVISION" "$(date -Is)" >"$TARGET/ARRWM_DOWNLOAD_PROVENANCE"
echo "YUME5B_ASSETS_READY target=$TARGET $(date -Is)"
