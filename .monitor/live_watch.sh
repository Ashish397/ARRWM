#!/bin/bash
# Live progress monitor: samples every 3 min, EXITS (-> notifies Claude) on:
#  - download stall (no byte delta across 2 samples while staging runs)
#  - any smoke job leaving the queue (completed/failed) since last report
#  - 30-min heartbeat regardless
T=/scratch/u6ex/as1748.u6ex/ARRWM/third_party
HF=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache

bytes() {
  du -sb $T/Matrix-Game/Matrix-Game-2/Matrix-Game-2.0 $T/Vista/ckpts \
    $T/HY-WorldPlay/weights $T/YUME/Yume-5B-720P $T/YUME/InternVL3-2B-Instruct \
    $HF/hub/models--OpenGVLab--InternVL3-8B $HF/hub/models--ByteDance--Sa2VA-4B 2>/dev/null \
    | awk '{s+=$1} END {print s+0}'
}
jobs_sig() { squeue --me -h -o "%i:%T" | sort | tr '\n' ' '; }

B0=$(bytes); J0=$(jobs_sig); SAME=0
echo "[watch] start bytes=$((B0/1024/1024))MB jobs: $J0"
for i in $(seq 1 10); do
  sleep 180
  B1=$(bytes); J1=$(jobs_sig)
  DELTA=$(( (B1-B0)/1024/1024 ))
  echo "[watch] t+$((i*3))min staged=$((B1/1024/1024))MB (+${DELTA}MB) jobs: $J1"
  STAGING_ALIVE=$(pgrep -f "stage_weights.sh" | head -1)
  if [ -n "$STAGING_ALIVE" ] && [ "$B1" = "$B0" ]; then
    SAME=$((SAME+1))
    if [ $SAME -ge 2 ]; then echo "[watch] ALERT: staging stalled (no bytes in 6 min)"; exit 2; fi
  else
    SAME=0
  fi
  if [ "$J1" != "$J0" ]; then echo "[watch] ALERT: job state change"; exit 3; fi
  B0=$B1
done
echo "[watch] 30-min heartbeat"
exit 0
