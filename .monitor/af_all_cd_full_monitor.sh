#!/bin/bash
# Watch the af-all-freal-cd FULL run (5313179, 8 nodes, 500 steps). The 2-node
# smoke (5313117) went GREEN, validating the whole new stack on 14d. This
# tracks the full to completion: report key crossings + new-feature liveness
# (CD loss, IQA agreement EMA, depth ratchet). Counts summed per-file (bc).
ID=5313179
BASE=af-all-freal-cd
cd /scratch/u6ex/as1748.u6ex/ARRWM
STATUS=/tmp/af_all_cd_full_status.txt
: > "$STATUS"
SEEN=0; SAW_RUN=0
START=$(date +%s); HEARTBEAT=7200; i=0
csum() { grep -hcE "$1" "$2" "$3" 2>/dev/null | paste -sd+ | bc 2>/dev/null; }
while true; do
  i=$((i+1)); NOW=$(date +%s); EL=$((NOW-START)); problem=""
  E="logs/${BASE}_${ID}.err"; O="logs/${BASE}_${ID}.out"
  ST=$(squeue -j $ID -h -o "%T" 2>/dev/null); [ -z "$ST" ] && ST="GONE"
  [ "$ST" != "GONE" ] && SEEN=1
  STEP=$(grep -hoE "step=[0-9]+/500" "$E" "$O" 2>/dev/null | grep -oE "[0-9]+" | sort -n | tail -1)
  DEPTH=$(grep -hoE "streaming_tp_depth[\": ]+[0-9.]+" "$E" "$O" 2>/dev/null | grep -oE "[0-9.]+$" | sort -n | tail -1)
  CD=$(csum "cd_loss_raw" "$E" "$O")
  IQA=$(csum "streaming_iqa_agree" "$E" "$O")
  DN=$(csum "Training complete|Training completed" "$E" "$O")
  OOM=$(csum "CUDA out of memory|OutOfMemoryError" "$E" "$O")
  ERR=$(grep -hcE "Traceback|RuntimeError|ValueError|KeyError|AssertionError|IndexError|size mismatch|FileNotFoundError" "$E" 2>/dev/null)
  HG=$(grep -hcE "Watchdog caught|operation timed out|ProcessGroupNCCL.*[Tt]imed out" "$E" 2>/dev/null)
  MT=$(stat -c %Y "$E" 2>/dev/null || echo 0); IDLE=$((NOW-MT))
  echo "[m$i +${EL}s] $ST ${STEP:-_}/500 depth=${DEPTH:-_} cd=${CD:-0} iqa=${IQA:-0} dn=${DN:-0} oom=${OOM:-0} err=${ERR:-0} hg=${HG:-0} idle=${IDLE}s" >> "$STATUS"
  [ "${OOM:-0}" -gt 0 ] 2>/dev/null && problem="OOM"
  [ "${ERR:-0}" -gt 0 ] 2>/dev/null && problem="ERR"
  [ "${HG:-0}" -gt 0 ] 2>/dev/null && problem="HANG"
  if [ "$ST" = "RUNNING" ] && [ "$MT" -gt 0 ] && [ "$IDLE" -gt 1500 ] && [ -n "${STEP:-}" ]; then problem="STALL>25min(step=${STEP})"; fi
  [ -n "$problem" ] && { echo "AF_ALL_CD_FULL_PROBLEM $problem (step=${STEP:-_} depth=${DEPTH:-_})" >> "$STATUS"; break; }
  if [ "$ST" = "GONE" ] && [ "$SEEN" -eq 1 ]; then
    if [ "${DN:-0}" -gt 0 ] && [ "${ERR:-0}" -eq 0 ]; then
      echo "AF_ALL_CD_FULL_GREEN (complete; last_step=${STEP:-_}/500 depth=${DEPTH:-_} cd=${CD:-0} iqa=${IQA:-0})" >> "$STATUS"
    elif [ "${STEP:-0}" -ge 480 ] 2>/dev/null; then
      echo "AF_ALL_CD_FULL_WALLTIME_OK (reached ${STEP}/500 before SIGTERM, no err -> ran full length)" >> "$STATUS"
    else
      echo "AF_ALL_CD_FULL_GONE_EARLY (terminated at step ${STEP:-_}/500 -> investigate)" >> "$STATUS"
    fi
    break
  fi
  [ "$EL" -ge "$HEARTBEAT" ] && { echo "AF_ALL_CD_FULL_HEARTBEAT el=${EL}s ${STEP:-_}/500 depth=${DEPTH:-_} cd=${CD:-0} iqa=${IQA:-0}" >> "$STATUS"; break; }
  sleep 120
done
