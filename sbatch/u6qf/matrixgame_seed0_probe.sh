#!/bin/bash
# Probe the released Matrix-Game seed-0 stochastic path on cases that became
# black under the previous non-vendor seed before replacing the full fleet.
#SBATCH --job-name=mg-a32-probe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler/aligned32_stage/logs/%x_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler/aligned32_stage/logs/%x_%j.err
set -euo pipefail

R=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler
A="$R/ARRWM"
S="$R/aligned32_stage"
PY="$R/miniforge3/envs/arrwm/bin/python"
VENDOR="$S/Matrix-Game-2"
OUT="$S/matrix_seed0_probe"
export PATH="$R/miniforge3/envs/arrwm/bin:$PATH"
export AF_ROOT="$A" HF_HOME="$R/frodobots/hf_cache" EVAL_REAL_FRAME=32
export MG_NUMLAT=189 MG_SEED=0 MG_CAM=0.1 MG_KDIM=4
export MG_VENDOR_ROOT="$VENDOR" PYTHONPATH="$VENDOR" MG_OUT="$OUT"
export MG_SEED_FMT="$S/seed_frame32/seed65_{wi}_f0.png"
mkdir -p "$OUT"

cd "$VENDOR"
pids=()
for spec in "0:u31:F" "1:a00:F" "2:u31:L" "3:m42:B"; do
  IFS=: read -r gpu uid direction <<<"$spec"
  (CUDA_VISIBLE_DEVICES="$gpu" MG_WINDOWS="$uid" MG_DIRS="$direction" \
    "$PY" "$A/code_release/baselines/matrixgame_runner.py") \
    >"$OUT/${uid}_${direction}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

"$PY" - "$OUT" <<'PY'
import cv2, json, sys
from pathlib import Path
root = Path(sys.argv[1])
report = {}
for path in sorted(root.glob("*.mp4")):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    samples = []
    for seconds in range(0, 31, 2):
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(n - 1, round(seconds * fps)))
        ok, frame = cap.read()
        assert ok, (path, seconds)
        samples.append({"s": seconds, "mean": float(frame.mean()),
                        "std": float(frame.std()), "max": int(frame.max())})
    cap.release()
    report[path.name] = samples
print(json.dumps(report, indent=2))
PY
