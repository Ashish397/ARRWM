#!/bin/bash
# One task / one GPU worker for the fixed FrodoBots-eight generation pilot.
# A node-local parent launches this file four times with explicit
# FRODO8_SHARD=0..3 and CUDA_VISIBLE_DEVICES=0..3.  Each worker owns two
# disjoint contexts, so no two workers can write the same clip.
set -euo pipefail

MODEL=${1:?usage: frodo8_model_worker.sh MODEL}
case "$MODEL" in
  lingbot|dreamx|matrixgame2|minwm|minwm_ode) ;;
  *) echo "unsupported Frodo8 model: $MODEL" >&2; exit 64 ;;
esac

SHARD=${FRODO8_SHARD:-${SLURM_LOCALID:-}}
case "$SHARD" in 0|1|2|3) ;; *) echo "worker needs FRODO8_SHARD/SLURM_LOCALID 0..3" >&2; exit 64 ;; esac
VISIBLE=${CUDA_VISIBLE_DEVICES:-}
if [ -z "$VISIBLE" ] || [[ "$VISIBLE" == *,* ]]; then
  echo "Frodo8 worker requires exactly one Slurm-isolated GPU; CUDA_VISIBLE_DEVICES=$VISIBLE" >&2
  exit 64
fi

R=${ARRWM_REMOTE_ROOT:-/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler}
A=${ARRWM_CODE_ROOT:-$R/ARRWM}
S=${ALIGNED32_STAGE:-$R/aligned32_stage}
PY=${ARRWM_PYTHON:-$R/miniforge3/envs/arrwm/bin/python}
PY310=${MINWM_PYTHON:-$R/miniforge3/envs/arrwm/bin/python3.10}
MANIFEST=${FRODO8_MANIFEST:-$A/experiments/e1/scene_shortlist/frodo8_manifest.json}
SOURCE_MANIFEST=${FRODO32_MANIFEST:-$A/experiments/e1/scene_shortlist/e1_32_windows.json}
SEED_FRAME32=${FRODO8_SEED_FRAME32:-$S/seed_frame32}
SEED_STREAM=${FRODO8_SEED_STREAM:-$S/seed_stream_aligned}
FLEET_ROOT=${FRODO8_FLEET_ROOT:-$S/fleet30s_aligned32}
RUN_ID=${FRODO8_RUN_ID:-manual_$(date +%Y%m%dT%H%M%S)}
LOG_ROOT=${FRODO8_LOG_ROOT:-$S/logs/frodo8/$RUN_ID/$MODEL}
MARKER_ROOT=${FRODO8_MARKER_ROOT:-$S/logs/frodo8/$RUN_ID/markers/$MODEL}
ARCHIVE=${FRODO8_ARCHIVE:-$S/archive_frodo8_resume/$RUN_ID}

case "$MODEL" in
  minwm_ode) OUT=${FRODO8_MINWM_ODE_DIR:-$S/minwm_ode} ;;
  *) OUT="$FLEET_ROOT/$MODEL" ;;
esac

mkdir -p "$OUT" "$LOG_ROOT" "$MARKER_ROOT" "$ARCHIVE"
export PATH="$R/miniforge3/envs/arrwm/bin:$PATH"
export HF_HOME=${HF_HOME:-$R/frodobots/hf_cache}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MALLOC_ARENA_MAX=2
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export EVAL_REAL_FRAME=32

mapfile -t ALL_UIDS < <("$PY" - "$MANIFEST" <<'PY'
import json, sys
for row in json.load(open(sys.argv[1]))["contexts"]:
    print(row["uid"])
PY
)
test "${#ALL_UIDS[@]}" -eq 8
UIDS=("${ALL_UIDS[$SHARD]}" "${ALL_UIDS[$((SHARD + 4))]}")
JOINED=$(IFS=,; echo "${UIDS[*]}")
DIRS=F,FR,R,BR,B,BL,L,FL,N

# Prevent accidental duplicate workers for the same model/shard.  The lock is
# held for preparation, model loading, generation, and shard validation.
exec 9>"$S/logs/frodo8/${MODEL}_shard${SHARD}.lock"
if ! flock -n 9; then
  echo "another Frodo8 $MODEL shard $SHARD worker is active" >&2
  exit 73
fi

PREP="$LOG_ROOT/prepare_shard${SHARD}.json"
"$PY" "$A/grids/eval/frodo8_generation.py" prepare \
  --manifest "$MANIFEST" --source-manifest "$SOURCE_MANIFEST" \
  --model "$MODEL" --output "$OUT" --uids "$JOINED" \
  --seed-frame32 "$SEED_FRAME32" --seed-stream "$SEED_STREAM" \
  --archive "$ARCHIVE" >"$PREP"

RETAINED=$("$PY" - "$PREP" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["retained"])
PY
)
echo "[frodo8] model=$MODEL shard=$SHARD uids=$JOINED retained=$RETAINED gpu=${CUDA_VISIBLE_DEVICES:-unset}"

# DreamX workers load in GPU order.  Empty shards still advance the marker so
# a fully resumed shard cannot leave the next worker waiting.  Actual
# generation overlaps once each preceding worker has loaded and trimmed its
# temporary CPU checkpoint objects.
if [ "$MODEL" = dreamx ] && [ "$SHARD" -gt 0 ]; then
  PREV=$((SHARD - 1))
  deadline=$((SECONDS + 3600))
  until [ -f "$MARKER_ROOT/loaded_${PREV}" ]; do
    if [ -f "$MARKER_ROOT/failed_${PREV}" ]; then
      echo "DreamX preceding shard $PREV failed during load" >&2
      touch "$MARKER_ROOT/failed_${SHARD}"
      exit 70
    fi
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo "timed out waiting for DreamX shard $PREV load marker" >&2
      touch "$MARKER_ROOT/failed_${SHARD}"
      exit 70
    fi
    sleep 5
  done
fi

if [ "$RETAINED" -eq 18 ]; then
  echo "[frodo8] model=$MODEL shard=$SHARD already complete"
  [ "$MODEL" != dreamx ] || touch "$MARKER_ROOT/loaded_${SHARD}"
else
  RUN_LOG="$LOG_ROOT/gpu${SHARD}.log"
  case "$MODEL" in
    lingbot)
      cd "$S/lingbot-world-v2"
      export AF_ROOT="$A" PYTHONPATH="$A"
      LB_WINDOWS="$JOINED" LB_DIRS="$DIRS" LB_OUT="$OUT" \
        LB_SEED_DIR="$SEED_FRAME32" LB_FRAMES=493 LB_OFFLOAD=0 \
        LB_CKPT="$S/lingbot-world-v2/weights/lingbot-world-v2-1.3b-causal-fast" \
        LB_ASSETS="$S/lingbot-world-v2/weights/assets" \
        "$PY" "$A/code_release/baselines/lingbot_runner.py" >"$RUN_LOG" 2>&1
      ;;
    dreamx)
      cd "$S/DreamX-World"
      export AF_ROOT="$A" PYTHONPATH="$A"
      (DX_WINDOWS="$JOINED" DX_DIRS="$DIRS" DX_OUT="$OUT" \
        DX_SEED_DIR="$SEED_FRAME32" DX_NLAT=123 DX_PYTHON="$PY" \
        "$PY" "$A/code_release/baselines/dreamx_runner.py") >"$RUN_LOG" 2>&1 &
      dream_pid=$!
      while ! grep -qE '^Loaded [0-9]+ items from ' "$RUN_LOG"; do
        if ! kill -0 "$dream_pid" 2>/dev/null; then
          wait "$dream_pid" || true
          touch "$MARKER_ROOT/failed_${SHARD}"
          tail -100 "$RUN_LOG" >&2 || true
          exit 70
        fi
        sleep 5
      done
      touch "$MARKER_ROOT/loaded_${SHARD}"
      wait "$dream_pid" || {
        touch "$MARKER_ROOT/failed_${SHARD}"
        exit 70
      }
      ;;
    matrixgame2)
      cd "$S/Matrix-Game-2"
      export AF_ROOT="$A" PYTHONPATH="$S/Matrix-Game-2" MG_VENDOR_ROOT="$S/Matrix-Game-2"
      MG_WINDOWS="$JOINED" MG_DIRS="${DIRS//N/NOOP}" MG_NUMLAT=189 \
        MG_OUT="$OUT" MG_SEED_FMT="$SEED_FRAME32/seed65_{wi}_f0.png" \
        MG_SEED=0 MG_CAM=0.1 MG_KDIM=4 \
        "$PY" "$A/code_release/baselines/matrixgame_runner.py" >"$RUN_LOG" 2>&1
      ;;
    minwm|minwm_ode)
      B="$R/minwm_seed29"
      MW="$B/minWM"
      ARR="$B/ARRWM"
      cd "$MW"
      export AF_ROOT="$ARR" PYTHONPATH="$ARR" TMPDIR=/tmp
      export MW_CPU_T5=1 MW_CHUNK_DECODE=8
      export MW_SEED_FMT="$SEED_STREAM/seed65_{wi}.mp4"
      unset MW_CKPT
      if [ "$MODEL" = minwm ]; then
        export MW_SEED_LAT=8 MW_SEED_START=4 MW_STAGE=dmd
        export MW_MODEL_LABEL="minWM Wan2.1-1.3B Action2V 4-step DMD"
        MW_NUMLAT=128 MW_TAG=_aligned32
      else
        export MW_SEED_LAT=4 MW_SEED_START=20 MW_STAGE=ode
        export MW_CKPT="$MW/ckpts/Wan21/Action2V/causal_ode/model.pt"
        export MW_MODEL_LABEL="minWM Wan2.1-1.3B Action2V causal ODE"
        MW_NUMLAT=124 MW_TAG=_ode
      fi
      MW_WINDOWS="$JOINED" MW_DIRS=F,FR,R,BR,B,BL,L,FL,NOOP \
        MW_NUMLAT="$MW_NUMLAT" MW_TAG="$MW_TAG" MW_OUT="$OUT" \
        "$PY310" "$A/third_party/minWM/minwm_runner.py" >"$RUN_LOG" 2>&1
      ;;
  esac
fi

"$PY" "$A/grids/eval/frodo8_generation.py" validate \
  --manifest "$MANIFEST" --source-manifest "$SOURCE_MANIFEST" \
  --model "$MODEL" --output "$OUT" --uids "$JOINED" \
  --seed-frame32 "$SEED_FRAME32" --seed-stream "$SEED_STREAM" \
  --report "$LOG_ROOT/validated_shard${SHARD}.json"
touch "$LOG_ROOT/COMPLETE_shard${SHARD}"
echo "FRODO8_SHARD_COMPLETE model=$MODEL shard=$SHARD uids=$JOINED $(date -Is)"
