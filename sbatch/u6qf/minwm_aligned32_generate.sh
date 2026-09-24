#!/bin/bash
#SBATCH --job-name=minwm-a32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler/minwm_seed29/logs_aligned32/%x_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler/minwm_seed29/logs_aligned32/%x_%j.err
# Regenerate the ICLR minWM DMD fleet with the generation boundary aligned to
# real-video frame 32 and minWM's yaw convention mapped to ours.
set -euo pipefail

R=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler
B="$R/minwm_seed29"
MW="$B/minWM"
ARR="$B/ARRWM"
PY="$R/miniforge3/envs/arrwm/bin/python3.10"
LOG="$B/logs_aligned32"
OUT="$R/aligned32_stage/fleet30s_aligned32/minwm"
test -f "$R/aligned32_stage/seed_stream_aligned/COMPLETE"
mkdir -p "$LOG" "$OUT"

export PATH="$R/miniforge3/envs/arrwm/bin:$PATH"
export AF_ROOT="$ARR" HF_HOME="$R/frodobots/hf_cache" TMPDIR=/tmp
export PYTHONPATH="$ARR"
export MW_CPU_T5=1 MW_SEED_LAT=8 MW_SEED_START=4 MW_CHUNK_DECODE=8
export MW_SEED_FMT="$R/aligned32_stage/seed_stream_aligned/seed65_{wi}.mp4"

# One generated block gates the fleet. A 29-frame context beginning at frame 4
# ends at frame 32, so generated frame zero has the same real-time origin as
# our nine-latent-frame checkpoints.
SMOKE="$ARR/logs/eval_final/minwm_aligned32_smoke"
rm -rf "$SMOKE"
mkdir -p "$SMOKE"
(
  cd "$MW"
  CUDA_VISIBLE_DEVICES=0 MW_NUMLAT=12 MW_WINDOWS=u31 MW_DIRS=L \
    MW_TAG=_aligned32_smoke MW_OUT="$SMOKE" "$PY" minwm_runner.py
) >"$LOG/smoke.log" 2>&1

"$PY" - <<'PY'
import json, subprocess
from pathlib import Path
p = Path('/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler/minwm_seed29/ARRWM/logs/eval_final/minwm_aligned32_smoke')
v = next(p.glob('*.mp4'))
j = json.loads(Path(str(v) + '.json').read_text())
assert j['seed_frames'] == 29 and j['seed_latents'] == 8, j
assert j['seed_start_frame'] == 4 and j['generated_frames'] == 16, j
n = int(subprocess.check_output([
    'ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
    '-show_entries', 'stream=nb_read_frames', '-of', 'default=nw=1:nk=1', str(v)
], text=True).strip())
assert n == 45, (v, n)
print('SMOKE_VALIDATED boundary=real_frame_32 direction=L total_frames=45')
PY

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
    MW_DIRS=F,FR,R,BR,B,BL,L,FL,NOOP MW_TAG=_aligned32 MW_SAVE_LAT=1 \
    MW_OUT="$OUT" "$PY" minwm_runner.py >>"$LOG/shard${gpu}.log" 2>&1 &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
test "$rc" -eq 0

"$PY" - <<'PY'
import json, subprocess
from pathlib import Path
p = Path('/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler/aligned32_stage/fleet30s_aligned32/minwm')
videos = sorted(p.glob('*.mp4'))
sidecars = sorted(p.glob('*.mp4.json'))
assert len(videos) == 288 and len(sidecars) == 288, (len(videos), len(sidecars))
for sidecar in sidecars:
    row = json.loads(sidecar.read_text())
    assert row['seed_start_frame'] == 4, (sidecar, row['seed_start_frame'])
    assert row['seed_frames'] == 29 and row['generated_frames'] == 480, (sidecar, row)
for video in videos:
    n = int(subprocess.check_output([
        'ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_read_frames', '-of', 'default=nw=1:nk=1', str(video)
    ], text=True).strip())
    assert n == 509, (video, n)
print('FLEET_VALIDATED videos=288 boundary=real_frame_32 generated_frames=480')
PY

echo "MINWM_ALIGNED32_COMPLETE $(date -Is)"
