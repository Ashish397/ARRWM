#!/bin/bash
# DreamX aligned-frame fleet, four independent workers per node (one per GPU).
# Two node shards give eight workers over the 32 evaluation contexts.  Model
# loading is staggered until the preceding worker has released its temporary
# 20-GB checkpoint/state-dict allocations; generation then runs concurrently.
#SBATCH --job-name=dx4-a32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --mem=460000M
#SBATCH --time=04:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler/aligned32_stage/logs/%x_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler/aligned32_stage/logs/%x_%j.err
set -euo pipefail

NODE_SHARD=${1:?usage: sbatch dreamx_aligned32_4gpu_node.sh NODE_SHARD_0_OR_1}
case "$NODE_SHARD" in 0|1) ;; *) echo "bad node shard: $NODE_SHARD"; exit 2 ;; esac

R=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler
A="$R/ARRWM"
S="$R/aligned32_stage"
PY="$R/miniforge3/envs/arrwm/bin/python"
OUT="$S/fleet30s_aligned32/dreamx"
LOG="$S/logs/dreamx_4gpu_${SLURM_JOB_ID}"
DIRS=F,FR,R,BR,B,BL,L,FL,N

mkdir -p "$OUT" "$LOG"
export PATH="$R/miniforge3/envs/arrwm/bin:$PATH"
export AF_ROOT="$A" PYTHONPATH="$A" HF_HOME="$R/frodobots/hf_cache"
export EVAL_REAL_FRAME=32
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MALLOC_ARENA_MAX=2

mapfile -t ALL_UIDS < <("$PY" - <<PY
import json
for row in json.load(open('$A/experiments/e1/scene_shortlist/e1_32_windows.json')):
    print(row['uid'])
PY
)

# Node 0 owns even source-list positions, node 1 odd positions.  Within the
# node, round-robin the 16 owned contexts across its four GPUs (four each).
GPU_WINDOWS=("" "" "" "")
owned=0
for ((i=0; i<${#ALL_UIDS[@]}; i++)); do
  if (( i % 2 == NODE_SHARD )); then
    gpu=$((owned % 4))
    if [ -n "${GPU_WINDOWS[$gpu]}" ]; then
      GPU_WINDOWS[$gpu]="${GPU_WINDOWS[$gpu]},${ALL_UIDS[$i]}"
    else
      GPU_WINDOWS[$gpu]="${ALL_UIDS[$i]}"
    fi
    owned=$((owned + 1))
  fi
done
test "$owned" -eq 16

# Quarantine only incomplete files owned by this node.  Complete vendor files
# are resumable inputs and will be skipped by the released DreamX loop.
PARTIAL_ARCHIVE="$S/archive_dreamx_partials_${SLURM_JOB_ID}"
for joined in "${GPU_WINDOWS[@]}"; do
  for uid in ${joined//,/ }; do
    for direction in ${DIRS//,/ }; do
      vendor="$OUT/${uid}_${direction}_seed_frame32_seed65_${uid}_f0.mp4"
      stable="$OUT/dreamx_${uid}_${direction}.mp4"
      if [ -f "$vendor" ]; then
        frames=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames \
          -of default=nw=1:nk=1 "$vendor" 2>/dev/null || true)
        if [ "$frames" != 489 ]; then
          mkdir -p "$PARTIAL_ARCHIVE"
          mv "$vendor" "$PARTIAL_ARCHIVE/"
          [ ! -e "$stable" ] && [ ! -L "$stable" ] || mv "$stable" "$PARTIAL_ARCHIVE/"
          [ ! -e "$stable.json" ] || mv "$stable.json" "$PARTIAL_ARCHIVE/"
        fi
      fi
    done
  done
done

cd "$S/DreamX-World"
pids=()
for gpu in 0 1 2 3; do
  joined=${GPU_WINDOWS[$gpu]}
  shard_log="$LOG/gpu${gpu}.log"
  echo "gpu=$gpu windows=$joined" | tee "$shard_log"
  (CUDA_VISIBLE_DEVICES="$gpu" DX_WINDOWS="$joined" DX_DIRS="$DIRS" \
    DX_OUT="$OUT" DX_SEED_DIR="$S/seed_frame32" DX_NLAT=123 DX_PYTHON="$PY" \
    "$PY" "$A/code_release/baselines/dreamx_runner.py") >>"$shard_log" 2>&1 &
  pid=$!
  pids+=("$pid")

  # Do not overlap the large transient checkpoint-loading phase.  Once this
  # marker appears the live bf16 model is on its GPU and dead staging arenas
  # have been trimmed; its generation may overlap the next worker's load.
  while ! grep -qE '^Loaded [0-9]+ items from ' "$shard_log"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" || true
      echo "DreamX worker gpu=$gpu exited during model load" >&2
      tail -100 "$shard_log" >&2
      exit 1
    fi
    sleep 5
  done
  echo "gpu=$gpu model-loaded; starting next GPU worker" | tee -a "$shard_log"
done

rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
test "$rc" -eq 0

"$PY" - "$OUT" "${GPU_WINDOWS[@]}" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
windows = [u for group in sys.argv[2:] for u in group.split(',') if u]
directions = "F,FR,R,BR,B,BL,L,FL,N".split(",")
assert len(windows) == 16 and len(set(windows)) == 16, windows
for uid in windows:
    for direction in directions:
        video = root / f"dreamx_{uid}_{direction}.mp4"
        sidecar = Path(str(video) + ".json")
        assert video.exists() and sidecar.exists(), (video, sidecar)
        frames = int(subprocess.check_output([
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames", "-of", "default=nw=1:nk=1",
            str(video),
        ], text=True).strip())
        assert frames == 489, (video, frames)
        meta = json.loads(sidecar.read_text())
        assert meta["generation_boundary_real_frame"] == 32, (sidecar, meta)
print("DREAMX_4GPU_NODE_VALIDATED", len(windows) * len(directions))
PY

echo "DREAMX_ALIGNED32_4GPU_COMPLETE node_shard=$NODE_SHARD $(date -Is)"
