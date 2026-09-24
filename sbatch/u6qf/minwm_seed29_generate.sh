#!/bin/bash
# Run inside a u6qf one-node/four-GPU holder. A one-clip smoke test gates the
# full fleet, so a bad two-block seed never launches all 288 generations.
set -euo pipefail

R=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler
B="$R/minwm_seed29"
MW="$B/minWM"
ARR="$B/ARRWM"
PY="$R/miniforge3/envs/arrwm/bin/python3.10"
HF="$R/miniforge3/envs/arrwm/bin/hf"
WEIGHTS="$MW/ckpts/Wan21/Action2V/dmd/model.pt"
BASE="$MW/Wan21/wan_models/Wan2.1-T2V-1.3B"
LOG="$B/logs"
OUT="$B/ARRWM/logs/eval_final/fleet30s/minwm_seed29"
mkdir -p "$LOG" "$OUT"

# Download inside the durable holder so the transfer survives agent sessions.
# Hugging Face local-dir downloads resume the partial staging already present.
export PATH="$R/miniforge3/envs/arrwm/bin:$PATH"
export HF_HOME="$R/frodobots/hf_cache"
if [ ! -s "$WEIGHTS" ]; then
  "$HF" download MIN-Lab/minWM --local-dir "$MW/ckpts" \
    --include 'Wan21/Action2V/dmd/*'
fi
if [ ! -s "$BASE/models_t5_umt5-xxl-enc-bf16.pth" ] || \
   [ ! -s "$BASE/diffusion_pytorch_model.safetensors" ] || \
   [ ! -s "$BASE/Wan2.1_VAE.pth" ]; then
  "$HF" download Wan-AI/Wan2.1-T2V-1.3B --local-dir "$BASE"
fi
test "$(stat -c %s "$WEIGHTS")" = 5959605031
test -s "$BASE/models_t5_umt5-xxl-enc-bf16.pth"
test -s "$BASE/diffusion_pytorch_model.safetensors"
test -s "$BASE/Wan2.1_VAE.pth"

export AF_ROOT="$ARR"
export HF_HOME="$R/frodobots/hf_cache"
export TMPDIR=/tmp
export PYTHONPATH="$ARR"
export MW_CPU_T5=1 MW_SEED_LAT=8 MW_CHUNK_DECODE=8
export MW_SEED_FMT="$ARR/analysis/eval_final/seed65_e1/seed65_{wi}.mp4"

# Validation: eight seed latents must decode from 29 pixels, and one generated
# four-latent block must add exactly 16 pixels.
SMOKE="$ARR/logs/eval_final/minwm_seed29_smoke"
rm -rf "$SMOKE"
mkdir -p "$SMOKE"
(
  cd "$MW"
  CUDA_VISIBLE_DEVICES=0 MW_NUMLAT=12 MW_WINDOWS=u31 MW_DIRS=F \
    MW_TAG=_seed29_smoke MW_SAVE_LAT=1 MW_OUT="$SMOKE" \
    "$PY" minwm_runner.py
) >"$LOG/smoke.log" 2>&1

"$PY" - <<'PY'
import json, subprocess
from pathlib import Path
p = Path('/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler/minwm_seed29/ARRWM/logs/eval_final/minwm_seed29_smoke')
v = next(p.glob('*.mp4'))
j = json.loads(Path(str(v) + '.json').read_text())
assert j['seed_frames'] == 29, j
assert j['seed_latents'] == 8, j
assert j['generated_frames'] == 16, j
n = int(subprocess.check_output([
    'ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
    '-show_entries', 'stream=nb_read_frames', '-of', 'default=nw=1:nk=1', str(v)
], text=True).strip())
assert n == 45, (v, n)
print('SMOKE_VALIDATED seed_pixels=29 seed_latents=8 generated_pixels=16 total_pixels=45')
PY

# Four independent model processes, one per GPU. Each gets eight of the 32
# contexts and all nine actions.
mapfile -t UIDS < <("$PY" - <<PY
import json
for row in json.load(open('$ARR/experiments/e1/scene_shortlist/e1_32_windows.json')):
    print(row['uid'])
PY
)

pids=()
cd "$MW"
for gpu in 0 1 2 3; do
  windows=()
  for ((i=gpu; i<${#UIDS[@]}; i+=4)); do windows+=("${UIDS[$i]}"); done
  joined=$(IFS=,; echo "${windows[*]}")
  echo "gpu=$gpu windows=$joined" | tee "$LOG/shard${gpu}.log"
  CUDA_VISIBLE_DEVICES="$gpu" MW_NUMLAT=128 MW_WINDOWS="$joined" \
    MW_DIRS=F,FR,R,BR,B,BL,L,FL,NOOP MW_TAG=_seed29 MW_SAVE_LAT=1 \
    MW_OUT="$OUT" "$PY" minwm_runner.py >>"$LOG/shard${gpu}.log" 2>&1 &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
test "$rc" -eq 0

"$PY" - <<'PY'
import json, subprocess
from pathlib import Path
p = Path('/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler/minwm_seed29/ARRWM/logs/eval_final/fleet30s/minwm_seed29')
videos = sorted(p.glob('*.mp4'))
sidecars = sorted(p.glob('*.mp4.json'))
assert len(videos) == 288, len(videos)
assert len(sidecars) == 288, len(sidecars)
for sidecar in sidecars:
    row = json.loads(sidecar.read_text())
    assert row['seed_frames'] == 29, (sidecar, row['seed_frames'])
    assert row['seed_latents'] == 8, (sidecar, row['seed_latents'])
    assert row['generated_frames'] == 480, (sidecar, row['generated_frames'])
for video in videos:
    n = int(subprocess.check_output([
        'ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_read_frames', '-of', 'default=nw=1:nk=1', str(video)
    ], text=True).strip())
    assert n == 509, (video, n)
print('FLEET_VALIDATED videos=288 seed_pixels=29 generated_pixels=480 total_pixels=509')
PY

echo "MINWM_SEED29_COMPLETE $(date -Is)"
