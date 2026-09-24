#!/bin/bash
# Payload for ONE common action inside a four-GPU u6qf allocation step.
# Launch it through yume5b_frodo8_on_holder.sh from the login node.  Within
# the resulting single srun step, torchrun gives one rank to each GPU and
# distributes Frodo8 as two contexts per rank; this payload must not create a
# nested srun.
set -euo pipefail

ACTION=${1:?usage: yume5b_frodo8_holder_action.sh F|FR|R|BR|B|BL|L|FL|N}
case "$ACTION" in F|FR|R|BR|B|BL|L|FL|N) ;; *) echo "bad action: $ACTION" >&2; exit 2 ;; esac

R=${ARRWM_REMOTE_ROOT:-/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler}
A=${AF_ROOT:-$R/ARRWM}
YU=${YUME_ROOT:-$A/third_party/YUME}
MODEL_DIR=${YUME_MODEL_DIR:-$R/yume/Yume-5B-720P}
S=${ALIGNED32_STAGE:-$R/aligned32_stage}
PY=${YUME_PYTHON:-$R/miniforge3/envs/arrwm/bin/python}
RUNNER=$A/code_release/baselines/yume5b_frodo8_runner.py
STAGE=${YUME_FRODO_STAGE:-$S/yume5b_frodo8}
OUT=${YUME_OUT:-$S/fleet30s_aligned32/yume5b}
SEEDS=${YUME_SEED_DIR:-$S/seed_frame32}
LOG=$STAGE/logs
mkdir -p "$LOG" "$OUT"

export PATH="$R/miniforge3/envs/arrwm/bin:$PATH"
export PYTHONPATH="$YU:$A${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${HF_HOME:-$R/frodobots/hf_cache}
export TMPDIR=${TMPDIR:-/tmp}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export YUME_MODEL_DIR="$MODEL_DIR"
export YUME_SKIP_CAPTIONER=1
export YUME_SCENE_PROMPT="A first-person view from a small delivery robot travelling along a street or sidewalk in a real outdoor environment."

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
test -f "$YU/fastvideo/sample/sample_5b.py"
for asset in diffusion_pytorch_model.safetensors Wan2.2_VAE.pth models_t5_umt5-xxl-enc-bf16.pth config.json; do
  test -s "$MODEL_DIR/$asset"
done
for uid in u31 u04 a20 m30 m38 m89 m128 b36; do
  test -f "$SEEDS/seed65_${uid}_f0.png"
done

mapfile -t GPU_UUIDS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits)
NGPU=${#GPU_UUIDS[@]}
NUNIQUE=$(printf '%s\n' "${GPU_UUIDS[@]}" | sort -u | wc -l)
if [ "$NGPU" -ne 4 ] || [ "$NUNIQUE" -ne 4 ]; then
  echo "YUME Frodo8 requires four distinct visible GPU UUIDs; found $NGPU GPUs / $NUNIQUE UUIDs" >&2
  printf 'visible UUID: %s\n' "${GPU_UUIDS[@]}" >&2
  exit 1
fi
echo "YUME_FRODO8_START action=$ACTION holder=${SLURM_JOB_ID:-none} host=$(hostname) gpus=$NGPU unique_uuids=$NUNIQUE $(date -Is)"
nvidia-smi -L

COMMON=(--action "$ACTION" --repo-root "$A" --yume-root "$YU" --model-dir "$MODEL_DIR" --seed-dir "$SEEDS" --stage "$STAGE" --output "$OUT")
"$PY" "$RUNNER" prepare "${COMMON[@]}" | tee "$LOG/${ACTION}_prepare_${SLURM_JOB_ID:-manual}.log"
if "$PY" "$RUNNER" complete "${COMMON[@]}" >"$LOG/${ACTION}_preexisting_${SLURM_JOB_ID:-manual}.log" 2>&1; then
  echo "YUME_FRODO8_ALREADY_COMPLETE action=$ACTION $(date -Is)"
  exit 0
fi

INPUT=$STAGE/inputs/$ACTION
VENDOR=$STAGE/vendor/$ACTION
test -d "$INPUT"
test -d "$VENDOR"
ITEMS=$(find "$INPUT" -mindepth 1 -maxdepth 1 \( -type f -o -type l \) | wc -l)
if [ "$ITEMS" -ne 4 ] && [ "$ITEMS" -ne 8 ]; then
  echo "staged work must be four or eight balanced items, found $ITEMS" >&2
  exit 1
fi
echo "YUME_FRODO8_GENERATE action=$ACTION staged=$ITEMS ranks=4 contexts_per_rank=$((ITEMS / 4))"

cd "$YU"
"$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
  fastvideo/sample/sample_5b.py \
  --seed 43 \
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
  --rollout_segments 17 \
  --save_final_only \
  2>&1 | tee "$LOG/${ACTION}_torchrun_${SLURM_JOB_ID:-manual}.log"

cd "$A"
"$PY" "$RUNNER" finalize "${COMMON[@]}" | tee "$LOG/${ACTION}_finalize_${SLURM_JOB_ID:-manual}.log"
"$PY" "$RUNNER" validate "${COMMON[@]}" | tee "$LOG/${ACTION}_validate_${SLURM_JOB_ID:-manual}.log"
echo "YUME_FRODO8_COMPLETE action=$ACTION videos=8 generated_seconds=30 $(date -Is)"
