#!/bin/bash
# Run one model/action over panel32 as four independent one-GPU workers.
set -euo pipefail

MODEL=${1:?usage: panel32_external_holder_action.sh MODEL ACTION MANIFEST PROVENANCE}
ACTION=${2:?usage: panel32_external_holder_action.sh MODEL ACTION MANIFEST PROVENANCE}
MANIFEST=${3:?usage: panel32_external_holder_action.sh MODEL ACTION MANIFEST PROVENANCE}
PROVENANCE=${4:?usage: panel32_external_holder_action.sh MODEL ACTION MANIFEST PROVENANCE}
case "$MODEL" in lingbot|dreamx|matrixgame2|minwm|minwm_ode) ;; *) echo "bad model: $MODEL" >&2; exit 2;; esac
case "$ACTION" in F|FR|R|BR|B|BL|L|FL|N) ;; *) echo "bad action: $ACTION" >&2; exit 2;; esac

R=${ARRWM_REMOTE_ROOT:-/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler}
A=${ARRWM_CODE_ROOT:-$R/ARRWM}
PANEL_ROOT=${PANEL32_STAGE:-$R/panel32_stage}
PY=${ARRWM_PYTHON:-$R/miniforge3/envs/arrwm/bin/python}
PY310=${MINWM_PYTHON:-$R/miniforge3/envs/arrwm/bin/python3.10}
INPUTS=${PANEL32_INPUTS:-$PANEL_ROOT/model_inputs}
OUT=${PANEL32_EXTERNAL_OUT:-$PANEL_ROOT/fleet30s/$MODEL}
LOG=${PANEL32_EXTERNAL_LOG:-$PANEL_ROOT/logs/$MODEL/$ACTION}
PLAN="$INPUTS/plan_${MODEL}_${ACTION}.json"
mkdir -p "$INPUTS" "$OUT" "$LOG/markers"

# A queued sequence and a newly freed holder may converge on the same future
# action.  Serialize at model/action granularity so the second caller resumes
# from the first caller's complete outputs instead of writing concurrently.
mkdir -p "$PANEL_ROOT/locks"
exec 7>"$PANEL_ROOT/locks/generate_${MODEL}_${ACTION}.lock"
flock 7

export PATH="$R/miniforge3/envs/arrwm/bin:$PATH"
export PYTHONPATH="$A${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${HF_HOME:-$R/frodobots/hf_cache}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-14}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-14}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-14}
export EVAL_REAL_FRAME=32
PROMPT='A first-person view of an outdoor environment.'

test -f "$MANIFEST"
test -f "$PROVENANCE"
mapfile -t UUIDS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits)
test "${#UUIDS[@]}" -eq 4
test "$(printf '%s\n' "${UUIDS[@]}" | sort -u | wc -l)" -eq 4

exec 8>"$INPUTS/.materialize.lock"
flock 8
"$PY" "$A/grids/eval/panel32_model_inputs.py" \
  --manifest "$MANIFEST" --source-provenance "$PROVENANCE" \
  --model "$MODEL" --action "$ACTION" --stage "$INPUTS" --output "$OUT" \
  --prompt "$PROMPT" --materialize >"$LOG/plan.stdout.json"
flock -u 8
test -f "$PLAN"

worker() {
  local shard=$1
  local joined disk_action
  joined=$("$PY" - "$PLAN" "$shard" <<'PY'
import json, sys
rows=json.load(open(sys.argv[1]))['items'][int(sys.argv[2])::4]
assert len(rows)==8, len(rows)
print(','.join(row['context_id'] for row in rows))
PY
)
  disk_action=$("$PY" - "$PLAN" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))['disk_action'])
PY
)
  local runlog="$LOG/gpu${shard}.log"

  if [ "$MODEL" = dreamx ] && [ "$shard" -gt 0 ]; then
    local prev=$((shard - 1)) deadline=$((SECONDS + 3600))
    until [ -f "$LOG/markers/loaded_${prev}" ]; do
      if [ -f "$LOG/markers/failed_${prev}" ] || [ "$SECONDS" -ge "$deadline" ]; then
        touch "$LOG/markers/failed_${shard}"; return 70
      fi
      sleep 5
    done
  fi

  case "$MODEL" in
    lingbot)
      cd "$R/aligned32_stage/lingbot-world-v2"
      # LingBot rounds its latent length down to a four-latent chunk.  A
      # request for 481 pixel frames therefore returns only 477 frames
      # (29.75 generated seconds).  493 is the smallest native chunk-aligned
      # request that covers the common 30.0-second evaluation endpoint.
      AF_ROOT="$A" LB_WINDOWS="$joined" LB_DIRS="$disk_action" \
        LB_OUT="$OUT" LB_SEED_DIR="$INPUTS/frame32" LB_FRAMES=493 \
        LB_CAP="$PROMPT" LB_OFFLOAD=0 \
        LB_CKPT="$R/aligned32_stage/lingbot-world-v2/weights/lingbot-world-v2-1.3b-causal-fast" \
        LB_ASSETS="$R/aligned32_stage/lingbot-world-v2/weights/assets" \
        "$PY" "$A/code_release/baselines/lingbot_runner.py" >"$runlog" 2>&1
      ;;
    dreamx)
      cd "$R/aligned32_stage/DreamX-World"
      (AF_ROOT="$A" DX_WINDOWS="$joined" DX_DIRS="$disk_action" \
        DX_OUT="$OUT" DX_SEED_DIR="$INPUTS/frame32" DX_NLAT=123 \
        DX_CAP="$PROMPT" DX_PYTHON="$PY" \
        "$PY" "$A/code_release/baselines/dreamx_runner.py") >"$runlog" 2>&1 &
      local pid=$!
      while ! grep -qE '^Loaded [0-9]+ items from ' "$runlog"; do
        if ! kill -0 "$pid" 2>/dev/null; then
          wait "$pid" || true; touch "$LOG/markers/failed_${shard}"; return 70
        fi
        sleep 5
      done
      touch "$LOG/markers/loaded_${shard}"
      wait "$pid" || { touch "$LOG/markers/failed_${shard}"; return 70; }
      ;;
    matrixgame2)
      cd "$R/aligned32_stage/Matrix-Game-2"
      PYTHONPATH="$R/aligned32_stage/Matrix-Game-2" AF_ROOT="$A" \
        MG_VENDOR_ROOT="$R/aligned32_stage/Matrix-Game-2" \
        MG_WINDOWS="$joined" MG_DIRS="$disk_action" MG_NUMLAT=189 \
        MG_OUT="$OUT" MG_SEED_FMT="$INPUTS/frame32/seed65_{wi}_f0.png" \
        MG_SEED=0 MG_CAM=0.1 MG_KDIM=4 \
        "$PY" "$A/code_release/baselines/matrixgame_runner.py" >"$runlog" 2>&1
      ;;
    minwm|minwm_ode)
      cd "$R/minwm_seed29/minWM"
      local stage=dmd seed_lat=8 seed_start=4 total_lat=128 tag=_aligned32 label='minWM Wan2.1-1.3B Action2V 4-step DMD'
      local ckpt=()
      if [ "$MODEL" = minwm_ode ]; then
        stage=ode; seed_lat=4; seed_start=20; total_lat=124; tag=_ode
        label='minWM Wan2.1-1.3B Action2V causal ODE'
        ckpt=(MW_CKPT="$R/minwm_seed29/minWM/ckpts/Wan21/Action2V/causal_ode/model.pt")
      fi
      env AF_ROOT="$R/minwm_seed29/ARRWM" TMPDIR=/tmp MW_CPU_T5=1 \
        MW_CHUNK_DECODE=8 MW_SEED_FMT="$INPUTS/streams/seed65_{wi}.avi" \
        MW_SEED_LAT="$seed_lat" MW_SEED_START="$seed_start" MW_STAGE="$stage" \
        MW_MODEL_LABEL="$label" MW_WINDOWS="$joined" MW_DIRS="$disk_action" \
        MW_NUMLAT="$total_lat" MW_TAG="$tag" MW_OUT="$OUT" MW_CAP="$PROMPT" \
        "${ckpt[@]}" "$PY310" "$A/third_party/minWM/minwm_runner.py" >"$runlog" 2>&1
      ;;
  esac
}

pids=()
for shard in 0 1 2 3; do
  (export CUDA_VISIBLE_DEVICES="$shard"; worker "$shard") &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
if [ "$rc" -ne 0 ]; then
  tail -n 100 -- "$LOG"/gpu*.log >&2 || true
  exit 1
fi

# Attach the common source/input/action lineage only after every worker exits.
"$PY" - "$A" "$PLAN" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from grids.eval.panel32_model_inputs import attach_sidecar_provenance
plan=json.load(open(sys.argv[2]))
for item in plan['items']:
    video=Path(item['stable_output'])
    sidecar=Path(str(video)+'.json')
    if not video.is_file() or not sidecar.is_file():
        raise SystemExit(f'missing output or sidecar: {video}')
    attach_sidecar_provenance(plan, item, sidecar)
print(f"attached panel32 provenance to {len(plan['items'])} outputs")
PY
touch "$LOG/COMPLETE"
echo "PANEL32_EXTERNAL_ACTION_COMPLETE model=$MODEL action=$ACTION contexts=32 $(date -Is)"
