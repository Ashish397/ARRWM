#!/bin/bash
# Run one mixed-panel action inside an existing four-GPU holder allocation.
# This is a payload, not a submission script.  Invoke it through
# yume5b_panel32_on_holder.sh so the external-launcher step sees all four GPUs.
set -euo pipefail

ACTION=${1:?usage: yume5b_panel32_holder_action.sh ACTION PANEL_MANIFEST SOURCE_PROVENANCE}
PANEL_MANIFEST=${2:?usage: yume5b_panel32_holder_action.sh ACTION PANEL_MANIFEST SOURCE_PROVENANCE}
SOURCE_PROVENANCE=${3:?usage: yume5b_panel32_holder_action.sh ACTION PANEL_MANIFEST SOURCE_PROVENANCE}
case "$ACTION" in F|FR|R|BR|B|BL|L|FL|N) ;; *) echo "bad action: $ACTION" >&2; exit 2 ;; esac

R=${ARRWM_REMOTE_ROOT:-/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler}
A=${AF_ROOT:-$R/ARRWM}
YU=${YUME_ROOT:-$A/third_party/YUME}
MODEL_DIR=${YUME_MODEL_DIR:-$R/yume/Yume-5B-720P}
PANEL_ROOT=${PANEL32_STAGE:-$R/panel32_stage}
PY=${YUME_PYTHON:-$R/miniforge3/envs/arrwm/bin/python}
RUNNER=$A/grids/eval/yume5b_panel32_runner.py
STAGE=${YUME_PANEL_STAGE:-$PANEL_ROOT/yume5b/work}
OUT=${YUME_PANEL_OUT:-$PANEL_ROOT/yume5b/outputs}
PROMPT=${YUME_PANEL_SCENE_PROMPT:-A first-person view of an outdoor environment.}
BASE_SEED=${YUME_PANEL_BASE_SEED:-43}
ROLLOUT_SEGMENTS=17
LOG=$STAGE/logs
mkdir -p "$LOG" "$OUT"

mkdir -p "$PANEL_ROOT/locks"
exec 7>"$PANEL_ROOT/locks/generate_yume5b_${ACTION}.lock"
flock 7

export PATH="$R/miniforge3/envs/arrwm/bin:$PATH"
export PYTHONPATH="$YU:$A${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${HF_HOME:-$R/frodobots/hf_cache}
export TMPDIR=${TMPDIR:-/tmp}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export YUME_MODEL_DIR="$MODEL_DIR"
export YUME_SKIP_CAPTIONER=1
export YUME_SCENE_PROMPT="$PROMPT"
export YUME_ACTION_PREFIX=""

case "$ACTION" in
  F)  export YUME_ACTION_KEYS=W    YUME_ACTION_MOUSE=· YUME_ACTION_DISTANCE=4 YUME_ACTION_TURN=0 YUME_ACTION_ROTATION=0 ;;
  FR) export YUME_ACTION_KEYS=W    YUME_ACTION_MOUSE=→ YUME_ACTION_DISTANCE=4 YUME_ACTION_TURN=4 YUME_ACTION_ROTATION=4 ;;
  R)  export YUME_ACTION_KEYS=None YUME_ACTION_MOUSE=→ YUME_ACTION_DISTANCE=0 YUME_ACTION_TURN=4 YUME_ACTION_ROTATION=4 ;;
  BR) export YUME_ACTION_KEYS=S    YUME_ACTION_MOUSE=→ YUME_ACTION_DISTANCE=4 YUME_ACTION_TURN=4 YUME_ACTION_ROTATION=4 ;;
  B)  export YUME_ACTION_KEYS=S    YUME_ACTION_MOUSE=· YUME_ACTION_DISTANCE=4 YUME_ACTION_TURN=0 YUME_ACTION_ROTATION=0 ;;
  BL) export YUME_ACTION_KEYS=S    YUME_ACTION_MOUSE=← YUME_ACTION_DISTANCE=4 YUME_ACTION_TURN=4 YUME_ACTION_ROTATION=4 ;;
  L)  export YUME_ACTION_KEYS=None YUME_ACTION_MOUSE=← YUME_ACTION_DISTANCE=0 YUME_ACTION_TURN=4 YUME_ACTION_ROTATION=4 ;;
  FL) export YUME_ACTION_KEYS=W    YUME_ACTION_MOUSE=← YUME_ACTION_DISTANCE=4 YUME_ACTION_TURN=4 YUME_ACTION_ROTATION=4 ;;
  N)  export YUME_ACTION_KEYS=None YUME_ACTION_MOUSE=· YUME_ACTION_DISTANCE=0 YUME_ACTION_TURN=0 YUME_ACTION_ROTATION=0 ;;
esac

test -x "$PY"
test -f "$RUNNER"
test -f "$PANEL_MANIFEST"
test -f "$SOURCE_PROVENANCE"
test -f "$YU/fastvideo/sample/sample_5b.py"
for asset in diffusion_pytorch_model.safetensors Wan2.2_VAE.pth models_t5_umt5-xxl-enc-bf16.pth config.json; do
  test -s "$MODEL_DIR/$asset"
done

# Fail closed if the actual vendor checkout still injects the release's
# city/person prior or if a caller silently changes the panel-wide prompt.
"$PY" - "$YU/fastvideo/sample/sample_5b.py" "$PROMPT" <<'PY'
from pathlib import Path
import sys

sample = Path(sys.argv[1]).read_text(encoding="utf-8")
prompt = sys.argv[2]
expected = "A first-person view of an outdoor environment."
assert prompt == expected, ("YUME panel prompt mismatch", prompt, expected)
for forbidden in ("This video depicts a city walk", "Person moves", "Person stands"):
    assert forbidden not in sample, ("hidden vendor prompt prefix remains", forbidden)
assert 'os.environ.get("YUME_ACTION_PREFIX", "")' in sample
print("YUME_PANEL32_VENDOR_PROMPT_PREFLIGHT_OK")
PY

mapfile -t GPU_UUIDS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits)
NGPU=${#GPU_UUIDS[@]}
NUNIQUE=$(printf '%s\n' "${GPU_UUIDS[@]}" | sort -u | wc -l)
if [ "$NGPU" -ne 4 ] || [ "$NUNIQUE" -ne 4 ]; then
  echo "YUME panel32 requires four distinct visible GPU UUIDs; found $NGPU GPUs / $NUNIQUE UUIDs" >&2
  printf 'visible UUID: %s\n' "${GPU_UUIDS[@]}" >&2
  exit 1
fi

COMMON=(
  --action "$ACTION"
  --panel-manifest "$PANEL_MANIFEST"
  --source-provenance "$SOURCE_PROVENANCE"
  --repo-root "$A"
  --remote-root "$R"
  --yume-root "$YU"
  --model-dir "$MODEL_DIR"
  --stage "$STAGE"
  --output "$OUT"
  --scene-prompt "$PROMPT"
  --base-seed "$BASE_SEED"
)

echo "YUME_PANEL32_START action=$ACTION holder=${SLURM_JOB_ID:-none} host=$(hostname) gpus=$NGPU unique_uuids=$NUNIQUE $(date -Is)"
nvidia-smi -L
"$PY" "$RUNNER" dry-run "${COMMON[@]}" | tee "$LOG/${ACTION}_dry_run_${SLURM_JOB_ID:-manual}.log"
"$PY" "$RUNNER" prepare "${COMMON[@]}" | tee "$LOG/${ACTION}_prepare_${SLURM_JOB_ID:-manual}.log"
if "$PY" "$RUNNER" complete "${COMMON[@]}" >"$LOG/${ACTION}_preexisting_${SLURM_JOB_ID:-manual}.log" 2>&1; then
  echo "YUME_PANEL32_ALREADY_COMPLETE action=$ACTION $(date -Is)"
  exit 0
fi

readarray -t PLAN < <("$PY" - "$STAGE/manifest_${ACTION}.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
print(data["input_root"])
print(data["vendor_output"])
print(len(data["work_items"]))
PY
)
INPUT=${PLAN[0]}
VENDOR=${PLAN[1]}
ITEMS=${PLAN[2]}
if [ "$ITEMS" -lt 4 ] || [ "$ITEMS" -gt 32 ] || [ $((ITEMS % 4)) -ne 0 ]; then
  echo "staged work must contain 4..32 items balanced across four ranks; found $ITEMS" >&2
  exit 1
fi
echo "YUME_PANEL32_GENERATE action=$ACTION staged=$ITEMS ranks=4 contexts_per_rank=$((ITEMS / 4)) prompt_sha=$(printf %s "$PROMPT" | sha256sum | cut -d' ' -f1)"

cd "$YU"
"$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
  fastvideo/sample/sample_5b.py \
  --seed "$BASE_SEED" \
  --gradient_checkpointing \
  --train_batch_size 1 \
  --max_sample_steps 600000 \
  --mixed_precision bf16 \
  --allow_tf32 \
  --t5_cpu \
  --video_output_dir "$VENDOR" \
  --test_data_dir ./val \
  --num_euler_timesteps 4 \
  --rand_num_img 0.6 \
  --jpg_dir "$INPUT" \
  --rollout_segments "$ROLLOUT_SEGMENTS" \
  --save_final_only \
  2>&1 | tee "$LOG/${ACTION}_torchrun_${SLURM_JOB_ID:-manual}.log"

cd "$A"
"$PY" "$RUNNER" finalize "${COMMON[@]}" | tee "$LOG/${ACTION}_finalize_${SLURM_JOB_ID:-manual}.log"
"$PY" "$RUNNER" validate "${COMMON[@]}" | tee "$LOG/${ACTION}_validate_${SLURM_JOB_ID:-manual}.log"
echo "YUME_PANEL32_COMPLETE action=$ACTION contexts=32 generated_seconds=30 $(date -Is)"
