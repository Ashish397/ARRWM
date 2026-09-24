#!/usr/bin/env bash
# Bring the matched v3 rolling-400 render back after its Slurm job succeeds.
set -euo pipefail

job_id=6645866
remote=as1748.u6qf@u6qf.aip2.isambard
remote_dir=/scratch/u6qf/as1748.u6qf/ARRWM_straggler/_logs/ours30s/out/base_v3_roll400
local_dir=/home/ashish/ARRWM/logs/eval_final/ours30s/base_v3_roll400
deadline=$((SECONDS + 8 * 3600))

while (( SECONDS < deadline )); do
  state=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$remote" \
    "sacct -j $job_id -X --format=State -P --noheader | head -n 1" 2>/dev/null | tr -d '[:space:]' || true)
  case "$state" in
    COMPLETED) break ;;
    FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*)
      echo "Render job $job_id ended in state $state" >&2
      exit 1 ;;
  esac
  sleep 60
done
(( SECONDS < deadline )) || { echo "Timed out waiting for job $job_id" >&2; exit 1; }

mkdir -p "$local_dir"
rsync -av --ignore-existing --include='*.mp4' --include='*.mp4.json' --exclude='*' \
  "$remote:$remote_dir/" "$local_dir/"

count=$(find "$local_dir" -maxdepth 1 -type f -name '*.mp4' | wc -l)
test "$count" -eq 3
for direction in F L B; do
  video="$local_dir/base_v3_roll400_m42_${direction}.mp4"
  test -s "$video"
  ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames \
    -of default=noprint_wrappers=1 "$video"
done
echo "Copied and verified $count v3 rolling-400 videos in $local_dir"
