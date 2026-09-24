#!/bin/bash
#SBATCH --job-name=ext-a32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler/aligned32_stage/logs/%x_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler/aligned32_stage/logs/%x_%j.err
# Smoke-gated, frame-32-aligned ICLR fleet for one external model.
set -euo pipefail

MODEL=${1:?usage: sbatch external_aligned32_generate.sbatch MODEL}
case "$MODEL" in lingbot|dreamx|matrixgame2) ;; *) echo "bad model: $MODEL"; exit 2;; esac

R=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler
A="$R/ARRWM"
S="$R/aligned32_stage"
PY="$R/miniforge3/envs/arrwm/bin/python"
SEEDS="$S/seed_frame32"
OUT="$S/fleet30s_aligned32/$MODEL"
LOG="$S/logs/$MODEL"
test -f "$S/seed_stream_aligned/COMPLETE"
mkdir -p "$OUT" "$LOG"
export PATH="$R/miniforge3/envs/arrwm/bin:$PATH"
export AF_ROOT="$A" PYTHONPATH="$A" HF_HOME="$R/frodobots/hf_cache"
export EVAL_REAL_FRAME=32

mapfile -t UIDS < <("$PY" - <<PY
import json
for row in json.load(open('$A/experiments/e1/scene_shortlist/e1_32_windows.json')):
    print(row['uid'])
PY
)

run_one() {
  local model=$1 windows=$2 dirs=$3 out=$4
  case "$model" in
    lingbot)
      cd "$S/lingbot-world-v2"
      # The causal-fast pipeline rounds latent length down to a multiple of
      # its four-latent chunk. 493 yields 1 conditioning + 492 generated
      # frames; evaluation consumes only the first 480 generated frames.
      LB_WINDOWS="$windows" LB_DIRS="$dirs" LB_OUT="$out" \
        LB_SEED_DIR="$SEEDS" LB_FRAMES=493 LB_OFFLOAD=0 \
        LB_CKPT="$S/lingbot-world-v2/weights/lingbot-world-v2-1.3b-causal-fast" \
        LB_ASSETS="$S/lingbot-world-v2/weights/assets" \
        "$PY" "$A/code_release/baselines/lingbot_runner.py"
      ;;
    dreamx)
      cd "$S/DreamX-World"
      # Native three-latent blocks require a multiple of three. 123 yields
      # one conditioning frame plus 488 generated frames; scoring uses 480.
      DX_WINDOWS="$windows" DX_DIRS="$dirs" DX_OUT="$out" DX_SEED_DIR="$SEEDS" \
        DX_NLAT=123 DX_PYTHON="$PY" \
        "$PY" "$A/code_release/baselines/dreamx_runner.py"
      ;;
    matrixgame2)
      cd "$S/Matrix-Game-2"
      MG_WINDOWS="$windows" MG_DIRS="${dirs//N/NOOP}" MG_NUMLAT=189 \
        MG_OUT="$out" MG_SEED_FMT="$SEEDS/seed65_{wi}_f0.png" \
        MG_VENDOR_ROOT="$S/Matrix-Game-2" PYTHONPATH="$S/Matrix-Game-2" \
        "$PY" "$A/code_release/baselines/matrixgame_runner.py"
      ;;
  esac
}

# A single forward clip must decode, start from the canonical frame-32 PNG,
# and cover at least 30 generated seconds before the four full shards launch.
SMOKE="$S/smoke_aligned32/$MODEL"
rm -rf "$SMOKE"; mkdir -p "$SMOKE"
run_one "$MODEL" u31 F "$SMOKE" >"$LOG/smoke.log" 2>&1
"$PY" - "$MODEL" "$SMOKE" <<'PY'
import cv2, json, subprocess, sys
from pathlib import Path
model, root = sys.argv[1], Path(sys.argv[2])
videos = list(root.glob('*.mp4'))
assert videos, root
v = videos[0]
probe = json.loads(subprocess.check_output([
    'ffprobe','-v','error','-select_streams','v:0',
    '-show_entries','stream=nb_frames,r_frame_rate','-of','json',str(v)
], text=True))['streams'][0]
n = int(probe['nb_frames']); num, den = map(int, probe['r_frame_rate'].split('/')); fps = num / den
minimum = 481 if model != 'matrixgame2' else 751
assert n >= minimum, (model, n, minimum, fps)
cap = cv2.VideoCapture(str(v))
stats = []
for index in (0, int(round(6 * fps))):
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    assert ok, (model, v, index)
    stats.append((index, float(frame.mean()), float(frame.std()), int(frame.max())))
cap.release()
# Catch failed/black rendering before a 288-video fleet is launched.  This is
# a pipeline-health gate, not a visual-quality threshold.
assert all(mean > 3 and std > 3 and peak > 10 for _, mean, std, peak in stats), (model, stats)
print(f'SMOKE_VALIDATED model={model} frames={n} fps={fps} frame_stats={stats}')
PY

pids=()
for gpu in 0 1 2 3; do
  windows=()
  for ((i=gpu; i<${#UIDS[@]}; i+=4)); do windows+=("${UIDS[$i]}"); done
  joined=$(IFS=,; echo "${windows[*]}")
  echo "gpu=$gpu windows=$joined" | tee "$LOG/shard${gpu}.log"
  (CUDA_VISIBLE_DEVICES="$gpu" run_one "$MODEL" "$joined" F,FR,R,BR,B,BL,L,FL,N "$OUT") \
    >>"$LOG/shard${gpu}.log" 2>&1 &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
test "$rc" -eq 0

"$PY" - "$MODEL" "$OUT" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
model, root = sys.argv[1], Path(sys.argv[2])
# DreamX retains both vendor files and stable symlinks; count stable names only.
prefix = {'lingbot':'lingbot_', 'dreamx':'dreamx_', 'matrixgame2':'matrixgame_'}[model]
videos = sorted(p for p in root.glob(f'{prefix}*.mp4') if not p.is_symlink() or model == 'dreamx')
if model == 'dreamx':
    videos = sorted(root.glob('dreamx_*.mp4'))
assert len(videos) == 288, (model, len(videos))
minimum = 481 if model != 'matrixgame2' else 751
for v in videos:
    n = int(subprocess.check_output([
        'ffprobe','-v','error','-count_frames','-select_streams','v:0',
        '-show_entries','stream=nb_read_frames','-of','default=nw=1:nk=1',str(v)
    ], text=True).strip())
    assert n >= minimum, (v, n, minimum)
    sidecar = Path(str(v) + '.json')
    assert sidecar.exists(), sidecar
    meta = json.loads(sidecar.read_text())
    assert meta['generation_boundary_real_frame'] == 32, (sidecar, meta)
    seed = Path(meta['seed_image'])
    h = hashlib.sha256(seed.read_bytes()).hexdigest()
    assert h == meta['seed_sha256'], (sidecar, h, meta['seed_sha256'])
print(f'FLEET_VALIDATED model={model} videos=288 boundary=real_frame_32')
PY

echo "EXTERNAL_ALIGNED32_COMPLETE model=$MODEL $(date -Is)"
