#!/bin/bash
# Launch one model's FrodoBots-eight pilot on an already-running four-GPU
# holder.  This script never submits or cancels a job.  Isambard's ordinary
# overlapping step leaves one holder GPU unavailable, so an external-launcher
# step inherits the complete node allocation.  Its node-local parent validates
# four unique physical UUIDs and starts one explicit process per GPU.
set -euo pipefail

MODEL=${1:?usage: ALLOC=<holder-job-id> frodo8_on_holder.sh MODEL}
case "$MODEL" in
  lingbot|dreamx|matrixgame2|minwm|minwm_ode) ;;
  *) echo "unsupported Frodo8 model: $MODEL" >&2; exit 64 ;;
esac

ALLOC=${ALLOC:-${HOLDER:-${SLURM_JOB_ID:-}}}
if [ -z "$ALLOC" ]; then
  echo "set ALLOC (or HOLDER) to an already-running four-GPU holder job" >&2
  exit 64
fi

R=${ARRWM_REMOTE_ROOT:-/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler}
A=${ARRWM_CODE_ROOT:-$R/ARRWM}
S=${ALIGNED32_STAGE:-$R/aligned32_stage}
PY=${ARRWM_PYTHON:-$R/miniforge3/envs/arrwm/bin/python}
MANIFEST=${FRODO8_MANIFEST:-$A/experiments/e1/scene_shortlist/frodo8_manifest.json}
SOURCE_MANIFEST=${FRODO32_MANIFEST:-$A/experiments/e1/scene_shortlist/e1_32_windows.json}
SEED_FRAME32=${FRODO8_SEED_FRAME32:-$S/seed_frame32}
SEED_STREAM=${FRODO8_SEED_STREAM:-$S/seed_stream_aligned}
FLEET_ROOT=${FRODO8_FLEET_ROOT:-$S/fleet30s_aligned32}
RUN_ID=${FRODO8_RUN_ID:-frodo8_${MODEL}_h${ALLOC}_$(date +%Y%m%dT%H%M%S)}
LOG_ROOT=${FRODO8_LOG_ROOT:-$S/logs/frodo8/$RUN_ID/$MODEL}
MARKER_ROOT=${FRODO8_MARKER_ROOT:-$S/logs/frodo8/$RUN_ID/markers/$MODEL}
export ARRWM_REMOTE_ROOT="$R" ARRWM_CODE_ROOT="$A" ALIGNED32_STAGE="$S"
export ARRWM_PYTHON="$PY" FRODO8_MANIFEST="$MANIFEST" FRODO32_MANIFEST="$SOURCE_MANIFEST"
export FRODO8_SEED_FRAME32="$SEED_FRAME32" FRODO8_SEED_STREAM="$SEED_STREAM"
export FRODO8_FLEET_ROOT="$FLEET_ROOT" FRODO8_RUN_ID="$RUN_ID"
export FRODO8_LOG_ROOT="$LOG_ROOT" FRODO8_MARKER_ROOT="$MARKER_ROOT"
export PATH="$R/miniforge3/envs/arrwm/bin:$PATH"

mkdir -p "$LOG_ROOT" "$MARKER_ROOT"
"$PY" "$A/grids/eval/frodo8_generation.py" verify-manifest \
  --manifest "$MANIFEST" --source-manifest "$SOURCE_MANIFEST" \
  >"$LOG_ROOT/manifest_check.json"
test -f "$SEED_STREAM/COMPLETE"

STATE=$(squeue -h -j "$ALLOC" -o %T | head -n 1)
if [ "$STATE" != RUNNING ]; then
  echo "holder $ALLOC is not RUNNING (state=${STATE:-missing})" >&2
  exit 69
fi
NODE=${FRODO8_NODE:-$(squeue -h -j "$ALLOC" -o %N | head -n 1)}
if [ -z "$NODE" ] || [[ "$NODE" == *","* ]] || [[ "$NODE" == *"["* ]]; then
  echo "set FRODO8_NODE to one node in holder $ALLOC (reported nodes=$NODE)" >&2
  exit 64
fi

echo "[frodo8] holder=$ALLOC node=$NODE model=$MODEL run=$RUN_ID start=$(date -Is)" | tee "$LOG_ROOT/launch.log"
srun --jobid="$ALLOC" --overlap --external-launcher \
  --nodes=1 --nodelist="$NODE" --ntasks=1 --ntasks-per-node=1 \
  --kill-on-bad-exit=1 \
  --output="$LOG_ROOT/srun_node.out" --error="$LOG_ROOT/srun_node.err" \
  bash "$A/sbatch/u6qf/frodo8_node_local.sh" "$MODEL"

case "$MODEL" in
  minwm_ode) OUT=${FRODO8_MINWM_ODE_DIR:-$S/minwm_ode} ;;
  *) OUT="$FLEET_ROOT/$MODEL" ;;
esac
"$PY" "$A/grids/eval/frodo8_generation.py" validate \
  --manifest "$MANIFEST" --source-manifest "$SOURCE_MANIFEST" \
  --model "$MODEL" --output "$OUT" \
  --seed-frame32 "$SEED_FRAME32" --seed-stream "$SEED_STREAM" \
  --report "$LOG_ROOT/VALIDATED_72.json" | tee -a "$LOG_ROOT/launch.log"
touch "$LOG_ROOT/COMPLETE"
echo "FRODO8_MODEL_COMPLETE holder=$ALLOC node=$NODE model=$MODEL $(date -Is)" | tee -a "$LOG_ROOT/launch.log"
