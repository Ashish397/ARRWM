#!/bin/bash
#SBATCH --job-name=mwode-a32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler/aligned32_stage/logs/%x_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler/aligned32_stage/logs/%x_%j.err
set -euo pipefail

R=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler
B="$R/minwm_seed29"
MW="$B/minWM"
ARR="$B/ARRWM"
PY="$R/miniforge3/envs/arrwm/bin/python3.10"
LOG="$R/aligned32_stage/logs/minwm_ode"
OUT="$R/aligned32_stage/minwm_ode"
SEEDS="$R/aligned32_stage/seed_stream_aligned"
test -f "$SEEDS/COMPLETE"
mkdir -p "$LOG" "$OUT"

export PATH="$R/miniforge3/envs/arrwm/bin:$PATH"
export AF_ROOT="$ARR" HF_HOME="$R/frodobots/hf_cache" TMPDIR=/tmp
export PYTHONPATH="$ARR"
export MW_CPU_T5=1 MW_SEED_LAT=4 MW_SEED_START=20 MW_CHUNK_DECODE=8
export MW_SEED_FMT="$SEEDS/seed65_{wi}.mp4"
export MW_CKPT="$MW/ckpts/Wan21/Action2V/causal_ode/model.pt"
export MW_STAGE=ode
export MW_MODEL_LABEL="minWM Wan2.1-1.3B Action2V causal ODE"

SMOKE="$R/aligned32_stage/smoke_aligned32/minwm_ode"
rm -rf "$SMOKE"
mkdir -p "$SMOKE"
(
  cd "$MW"
  CUDA_VISIBLE_DEVICES=0 MW_NUMLAT=12 MW_WINDOWS=u31 MW_DIRS=L \
    MW_TAG=_ode_smoke MW_OUT="$SMOKE" "$PY" minwm_runner.py
) >"$LOG/smoke.log" 2>&1

"$PY" - "$SMOKE" <<'PY'
import json, subprocess, sys
from pathlib import Path
p = Path(sys.argv[1])
v = next(p.glob('*.mp4'))
j = json.loads(Path(str(v) + '.json').read_text())
assert j['model'] == 'minWM Wan2.1-1.3B Action2V causal ODE', j
assert j['stage'] == 'ode' and j['config'] == 'causal_ode_camera.yaml', j
assert j['seed_start_frame'] == 20 and j['seed_frames'] == 13, j
assert j['seed_latents'] == 4 and j['generated_frames'] == 32, j
assert j['generation_boundary_real_frame'] == 32, j
assert j['yaw_adapter'] == 'released_minwm_yaw_negated_to_common_convention', j
n = int(subprocess.check_output([
    'ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
    '-show_entries', 'stream=nb_read_frames', '-of', 'default=nw=1:nk=1', str(v)
], text=True).strip())
assert n == 45, (v, n)
print('MINWM_ODE_SMOKE_VALIDATED boundary=real_frame_32 direction=L frames=45')
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
  CUDA_VISIBLE_DEVICES="$gpu" MW_NUMLAT=124 MW_WINDOWS="$joined" \
    MW_DIRS=F,FR,R,BR,B,BL,L,FL,NOOP MW_TAG=_ode MW_OUT="$OUT" \
    "$PY" minwm_runner.py >>"$LOG/shard${gpu}.log" 2>&1 &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
test "$rc" -eq 0

"$PY" - "$OUT" <<'PY'
import json, subprocess, sys
from pathlib import Path
p = Path(sys.argv[1])
videos = sorted(p.glob('minwm_ode_*.mp4'))
assert len(videos) == 288, len(videos)
for video in videos:
    sidecar = Path(str(video) + '.json')
    row = json.loads(sidecar.read_text())
    assert row['model'] == 'minWM Wan2.1-1.3B Action2V causal ODE', sidecar
    assert row['stage'] == 'ode' and row['config'] == 'causal_ode_camera.yaml', sidecar
    assert row['seed_start_frame'] == 20 and row['seed_frames'] == 13, sidecar
    assert row['seed_latents'] == 4 and row['generated_frames'] == 480, sidecar
    assert row['generation_boundary_real_frame'] == 32, sidecar
    assert row['yaw_adapter'] == 'released_minwm_yaw_negated_to_common_convention', sidecar
    n = int(subprocess.check_output([
        'ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_read_frames', '-of', 'default=nw=1:nk=1', str(video)
    ], text=True).strip())
    assert n == 493, (video, n)
print('MINWM_ODE_FLEET_VALIDATED videos=288 boundary=real_frame_32 generated=480')
PY

echo "MINWM_ODE_ALIGNED32_COMPLETE $(date -Is)"
