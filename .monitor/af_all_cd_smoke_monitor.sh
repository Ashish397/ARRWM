#!/bin/bash
# Watch the af-all-freal-cd SMOKE (5313060, 2 nodes, 40 steps). Validates the
# whole new stack on 14d: LoRA-merge student init, frozen-14d freal teacher,
# CD loss, af-all IQA (MUSIQ/NIQE) depth gate, GAN+flash off. On clean
# completion the 8-node full can be submitted. Counts summed per-file (bc).
ID=5313117
BASE=af-all-freal-cd-smoke
cd /scratch/u6ex/as1748.u6ex/ARRWM
STATUS=/tmp/af_all_cd_smoke_status.txt
: > "$STATUS"
SEEN=0
START=$(date +%s); HEARTBEAT=3600; i=0
csum() { grep -hcE "$1" "$2" "$3" 2>/dev/null | paste -sd+ | bc 2>/dev/null; }
while true; do
  i=$((i+1)); NOW=$(date +%s); EL=$((NOW-START)); problem=""
  E="logs/${BASE}_${ID}.err"; O="logs/${BASE}_${ID}.out"
  ST=$(squeue -j $ID -h -o "%T" 2>/dev/null); [ -z "$ST" ] && ST="GONE"
  [ "$ST" != "GONE" ] && SEEN=1
  STEP=$(grep -hoE "step=[0-9]+/40" "$E" "$O" 2>/dev/null | grep -oE "[0-9]+" | sort -n | tail -1)
  # new-feature liveness signals
  CDN=$(csum "cd_loss_raw|causal-CD loss ENABLED|student generator INITIALIZED from" "$E" "$O")
  IQA=$(csum "IQA depth-gate ENABLED|streaming_iqa_agree" "$E" "$O")
  MRG=$(csum "student generator INITIALIZED from merged v14 LoRA" "$E" "$O")
  DN=$(csum "Training complete|Training completed" "$E" "$O")
  OOM=$(csum "CUDA out of memory|OutOfMemoryError" "$E" "$O")
  ERR=$(grep -hcE "Traceback|RuntimeError|ValueError|KeyError|AssertionError|IndexError|size mismatch|FileNotFoundError" "$E" 2>/dev/null)
  HG=$(grep -hcE "Watchdog caught|operation timed out|ProcessGroupNCCL.*[Tt]imed out" "$E" 2>/dev/null)
  MT=$(stat -c %Y "$E" 2>/dev/null || echo 0); IDLE=$((NOW-MT))
  echo "[m$i +${EL}s] $ST ${STEP:-_}/40 merge=${MRG:-0} cd=${CDN:-0} iqa=${IQA:-0} dn=${DN:-0} oom=${OOM:-0} err=${ERR:-0} hg=${HG:-0} idle=${IDLE}s" >> "$STATUS"
  [ "${OOM:-0}" -gt 0 ] 2>/dev/null && problem="OOM"
  [ "${ERR:-0}" -gt 0 ] 2>/dev/null && problem="ERR"
  [ "${HG:-0}" -gt 0 ] 2>/dev/null && problem="HANG"
  if [ "$ST" = "RUNNING" ] && [ "$MT" -gt 0 ] && [ "$IDLE" -gt 1500 ] && [ -n "${STEP:-}" ]; then problem="STALL>25min(step=${STEP})"; fi
  [ -n "$problem" ] && { echo "AF_ALL_CD_SMOKE_PROBLEM $problem" >> "$STATUS"; break; }
  if [ "$ST" = "GONE" ] && [ "$SEEN" -eq 1 ]; then
    if [ "${DN:-0}" -gt 0 ] && [ "${ERR:-0}" -eq 0 ]; then
      echo "AF_ALL_CD_SMOKE_GREEN (complete @40, no errors; merge=${MRG:-0} cd=${CDN:-0} iqa=${IQA:-0}) -> submit full" >> "$STATUS"
    else
      echo "AF_ALL_CD_SMOKE_GONE_NO_COMPLETE (terminated without completion -> investigate; merge=${MRG:-0} cd=${CDN:-0} iqa=${IQA:-0})" >> "$STATUS"
    fi
    break
  fi
  [ "$EL" -ge "$HEARTBEAT" ] && { echo "AF_ALL_CD_SMOKE_HEARTBEAT el=${EL}s ${STEP:-_}/40" >> "$STATUS"; break; }
  sleep 120
done
