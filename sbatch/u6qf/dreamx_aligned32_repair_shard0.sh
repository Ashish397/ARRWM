#!/bin/bash
# Repair one DreamX aligned-frame shard after the original four-process job
# exceeded its shared host-memory cgroup. Existing vendor outputs are skipped
# by DreamX, so each shard is safely resumable.
#SBATCH --job-name=dream-repair
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
# DreamX briefly materialises multiple checkpoint/state-dict copies while
# loading.  The steady-state process is much smaller, but the loading peak can
# exceed 120 GB and is enforced by Slurm's process-group memory cgroup.
#SBATCH --mem=460000M
#SBATCH --time=08:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler/aligned32_stage/logs/%x_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler/aligned32_stage/logs/%x_%j.err
set -euo pipefail

R=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler
A="$R/ARRWM"
S="$R/aligned32_stage"
PY="$R/miniforge3/envs/arrwm/bin/python"
OUT="$S/fleet30s_aligned32/dreamx"
SHARD=${1:?usage: sbatch dreamx_aligned32_repair_shard0.sh SHARD_INDEX}
case "$SHARD" in
  0) WINDOWS=u31,u04,a20,m30,m38,m89,m128,b36 ;;
  1) WINDOWS=u37,u01,n05,m32,m42,m93,b00,b40 ;;
  2) WINDOWS=u48,a00,n12,m33,m45,m103,b01,b102 ;;
  3) WINDOWS=u00,a19,n14,m36,m83,m111,b02,b103 ;;
  *) echo "bad shard: $SHARD"; exit 2 ;;
esac
DIRS=F,FR,R,BR,B,BL,L,FL,N

export PATH="$R/miniforge3/envs/arrwm/bin:$PATH"
export AF_ROOT="$A" PYTHONPATH="$A" HF_HOME="$R/frodobots/hf_cache"
export EVAL_REAL_FRAME=32
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# The vendor resume check is pathname-only. Quarantine a file left without a
# complete 489-frame MP4 index by an earlier killed writer; otherwise it would
# be skipped as though it were valid. Shards own disjoint UID sets.
PARTIAL_ARCHIVE="$S/archive_dreamx_partials_${SLURM_JOB_ID}"
for uid in ${WINDOWS//,/ }; do
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

cd "$S/DreamX-World"
DX_WINDOWS="$WINDOWS" DX_DIRS="$DIRS" DX_OUT="$OUT" \
  DX_SEED_DIR="$S/seed_frame32" DX_NLAT=123 DX_PYTHON="$PY" \
  "$PY" "$A/code_release/baselines/dreamx_runner.py"

"$PY" - "$OUT" "$WINDOWS" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
windows = sys.argv[2].split(",")
directions = "F,FR,R,BR,B,BL,L,FL,N".split(",")
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
print("DREAMX_REPAIR_VALIDATED", len(windows) * len(directions))
PY

echo "DREAMX_ALIGNED32_REPAIR_COMPLETE shard=$SHARD $(date -Is)"
