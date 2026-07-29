#!/usr/bin/env bash
# Waits for (a) the two model downloads to finish and (b) the GPU to free (<4GB used),
# then runs the matched melt-VLM benchmark. Safe to launch in background now.
set -u
LOG="${1:-/tmp/dl.log}"
cd "$(dirname "$0")/../.."
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate flash

echo "[wait] waiting for downloads to finish..."
while ! grep -qE "DONE nvidia/Cosmos-Reason1-7B|FAIL nvidia/Cosmos-Reason1-7B" "$LOG" 2>/dev/null; do sleep 30; done
echo "[wait] downloads settled:"; grep -E "DONE|FAIL" "$LOG"

echo "[wait] waiting for GPU to free (<4000 MiB used)..."
while true; do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "${USED:-99999}" -lt 4000 ] && break
  sleep 30
done
echo "[wait] GPU free (${USED} MiB used). Launching benchmark."
bash grids/eval/run_melt_local.sh
