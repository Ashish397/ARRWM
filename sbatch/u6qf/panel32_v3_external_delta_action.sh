#!/bin/bash
# Generate the four author-corrected v3 contexts for one external model/action.
set -euo pipefail

MODEL=${1:?usage: panel32_v3_external_delta_action.sh MODEL ACTION}
ACTION=${2:?usage: panel32_v3_external_delta_action.sh MODEL ACTION}
case "$MODEL" in lingbot|dreamx|matrixgame2|minwm|minwm_ode) ;; *) exit 2;; esac
case "$ACTION" in F|FR|R|BR|B|BL|L|FL|N) ;; *) exit 2;; esac

R=${ARRWM_REMOTE_ROOT:-/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler}
A=${ARRWM_CODE_ROOT:-$R/ARRWM}
P=${PANEL32_STAGE:-$R/panel32_v3_stage}
PY=${ARRWM_PYTHON:-$R/miniforge3/envs/arrwm/bin/python}
PY310=${MINWM_PYTHON:-$R/miniforge3/envs/arrwm/bin/python3.10}
MANIFEST=$A/grids/eval/panel32_locked_v3.json
PROVENANCE=$P/sources/panel32_source_provenance.json
INPUTS=$P/model_inputs
OUT=$P/fleet30s/$MODEL
LOG=$P/logs/$MODEL/$ACTION
PLAN=$INPUTS/plan_${MODEL}_${ACTION}.json
CONTEXTS=(ego4d-8ed9e028 ego4d-a07fc4f3 ego4d-de54e5c6 ego4d-3abe265d)
PROMPT='A first-person view of an outdoor environment.'
mkdir -p "$OUT" "$LOG/markers" "$P/locks"

exec 7>"$P/locks/generate_${MODEL}_${ACTION}.lock"
flock 7
export PATH="$R/miniforge3/envs/arrwm/bin:$PATH"
export PYTHONPATH="$A${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${HF_HOME:-$R/frodobots/hf_cache}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-14}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-14}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-14}
export EVAL_REAL_FRAME=32

exec 8>"$INPUTS/.materialize.lock"
flock 8
"$PY" "$A/grids/eval/panel32_model_inputs.py" \
  --manifest "$MANIFEST" --source-provenance "$PROVENANCE" \
  --model "$MODEL" --action "$ACTION" --stage "$INPUTS" --output "$OUT" \
  --prompt "$PROMPT" --materialize >"$LOG/plan.stdout.json"
flock -u 8

mapfile -t UUIDS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits)
test "${#UUIDS[@]}" -eq 4
test "$(printf '%s\n' "${UUIDS[@]}" | sort -u | wc -l)" -eq 4

disk_action=$(
  "$PY" - "$PLAN" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))['disk_action'])
PY
)

worker() {
  local shard=$1 joined=${CONTEXTS[$1]} runlog="$LOG/gpu${1}.log"
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
      local stage=dmd seed_lat=8 seed_start=4 total_lat=128 tag=_aligned32
      local label='minWM Wan2.1-1.3B Action2V 4-step DMD' ckpt=()
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
  (export CUDA_VISIBLE_DEVICES="$shard"; worker "$shard") & pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
if [ "$rc" -ne 0 ]; then tail -n 100 -- "$LOG"/gpu*.log >&2 || true; exit 1; fi

"$PY" - "$A" "$PLAN" "${CONTEXTS[@]}" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from grids.eval.panel32_model_inputs import attach_sidecar_provenance
plan=json.load(open(sys.argv[2])); wanted=set(sys.argv[3:])
rows=[row for row in plan['items'] if row['context_id'] in wanted]
assert len(rows)==4
for item in rows:
    video=Path(item['stable_output']); sidecar=Path(str(video)+'.json')
    if not video.is_file() or not sidecar.is_file():
        raise SystemExit(f'missing output or sidecar: {video}')
    attach_sidecar_provenance(plan,item,sidecar)
print('attached v3 provenance to four corrected outputs')
PY
touch "$LOG/COMPLETE"
echo "PANEL32_V3_EXTERNAL_DELTA_COMPLETE model=$MODEL action=$ACTION $(date -Is)"
