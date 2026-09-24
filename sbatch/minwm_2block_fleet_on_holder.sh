#!/bin/bash
set -euo pipefail
: "${ALLOC:?set ALLOC to the minWM holder job id}"
ROOT=/scratch/u6ex/as1748.u6ex/ARRWM
OUT="$ROOT/logs/eval_final/fleet30s/minwm_seed29"
mkdir -p "$OUT" "$ROOT/logs/minwm_seed29"

srun --jobid="$ALLOC" --overlap --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --gpus-per-node=4 --cpus-per-task=4 --gpu-bind=none bash -lc '
set -euo pipefail
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
ROOT=/scratch/u6ex/as1748.u6ex/ARRWM
cd "$ROOT/third_party/minWM"
export AF_ROOT="$ROOT" HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache
export TMPDIR=/tmp PYTHONPATH="$ROOT" MW_CPU_T5=1 MW_SEED_LAT=8 MW_NUMLAT=128
export MW_DIRS=F,FR,R,BR,B,BL,L,FL,NOOP MW_TAG=_seed29 MW_SAVE_LAT=1
export MW_SEED_FMT="$ROOT/analysis/eval_final/seed65_e1/seed65_{wi}.mp4"
export MW_OUT="$ROOT/logs/eval_final/fleet30s/minwm_seed29"

mapfile -t UID < <(python - <<PY
import json
for row in json.load(open("$ROOT/experiments/e1/scene_shortlist/e1_32_windows.json")):
    print(row["uid"])
PY
)
pids=()
for local_gpu in 0 1 2 3; do
  shard=$local_gpu
  windows=()
  for ((i=shard; i<${#UID[@]}; i+=4)); do windows+=("${UID[$i]}"); done
  joined=$(IFS=,; echo "${windows[*]}")
  log="$ROOT/logs/minwm_seed29/shard${shard}.log"
  echo "shard=$shard gpu=$local_gpu windows=$joined" | tee "$log"
  CUDA_VISIBLE_DEVICES="$local_gpu" MW_WINDOWS="$joined" python minwm_runner.py >>"$log" 2>&1 &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
exit "$rc"
'

python - <<'PY'
import json
from pathlib import Path
p = Path('/scratch/u6ex/as1748.u6ex/ARRWM/logs/eval_final/fleet30s/minwm_seed29')
videos = sorted(p.glob('*.mp4'))
sidecars = sorted(p.glob('*.mp4.json'))
assert len(videos) == 288, len(videos)
assert len(sidecars) == 288, len(sidecars)
for sidecar in sidecars:
    row = json.loads(sidecar.read_text())
    assert row['seed_frames'] == 29, (sidecar, row['seed_frames'])
    assert row['seed_latents'] == 8, (sidecar, row['seed_latents'])
    assert row['generated_frames'] == 480, (sidecar, row['generated_frames'])
print('validated', len(videos), 'videos with 29 seed + 480 generated frames')
PY
