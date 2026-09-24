#!/bin/bash
# Resume a disjoint subset of the minWM 29-frame fleet on one four-GPU node.
set -euo pipefail
: "${MW_ASSIGNED_UIDS:?comma-separated context IDs are required}"
: "${MW_WORKER_ID:?worker identifier is required}"

R=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler
B="$R/minwm_seed29"
MW="$B/minWM"
ARR="$B/ARRWM"
PY="$R/miniforge3/envs/arrwm/bin/python3.10"
OUT="$ARR/logs/eval_final/fleet30s/minwm_seed29"
LOG="$B/logs/parallel_${MW_WORKER_ID}"
mkdir -p "$LOG" "$OUT"

export AF_ROOT="$ARR"
export HF_HOME="$R/frodobots/hf_cache"
export TMPDIR=/tmp
export PYTHONPATH="$ARR"
export MW_CPU_T5=1 MW_SEED_LAT=8 MW_CHUNK_DECODE=8 MW_NUMLAT=128
export MW_SEED_FMT="$ARR/analysis/eval_final/seed65_e1/seed65_{wi}.mp4"
export MW_DIRS=F,FR,R,BR,B,BL,L,FL,NOOP MW_TAG=_seed29 MW_SAVE_LAT=1
export MW_OUT="$OUT"

IFS=, read -ra UIDS <<< "$MW_ASSIGNED_UIDS"
cd "$MW"
pids=()
for gpu in 0 1 2 3; do
  assigned=()
  for ((i=gpu; i<${#UIDS[@]}; i+=4)); do assigned+=("${UIDS[$i]}"); done
  joined=$(IFS=,; echo "${assigned[*]}")
  [ -n "$joined" ] || continue
  echo "gpu=$gpu contexts=$joined" | tee "$LOG/gpu${gpu}.log"
  CUDA_VISIBLE_DEVICES="$gpu" MW_WINDOWS="$joined" \
    "$PY" minwm_runner.py >>"$LOG/gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done

rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
if [ "$rc" -ne 0 ]; then
  touch "$LOG/FAILED"
  exit 1
fi
touch "$LOG/COMPLETE"
echo "WORKER_COMPLETE id=$MW_WORKER_ID contexts=$MW_ASSIGNED_UIDS $(date -Is)"
