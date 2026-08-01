#!/bin/bash
# Push the remaining release smokes into an already-running held allocation,
# bypassing the queue. Each step is independent and logs its own exit code, so
# one failure does not stop the rest.
#
#   bash tools/run_in_hold.sh <hold_jobid>
#
# Results land under verification/smoke/.
set -u
JOB="${1:?usage: run_in_hold.sh <hold_jobid>}"
ARR=/scratch/u6ex/as1748.u6ex/ARRWM
REL=$ARR/code_release
CKPT=$ARR/logs/v14e_pca8_raw/causal_lora_step0005000.pt
OUT=$ARR/verification/smoke
mkdir -p "$OUT"

run() {  # run <tag> <command>
  local tag="$1"; shift
  echo "=== $tag  $(date -u +%H:%M:%S) ==="
  srun --overlap --jobid="$JOB" --ntasks=1 bash -lc "
    source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
    cd $REL
    export DATA_ROOT=/projects/u6ex/fbots WAN_MODELS=/scratch/u6ex/as1748.u6ex/frodobots
    export AF_ROOT=$ARR PYTHONPATH=$REL ARRWM_ACTION_ENCODER=pca_raw
    export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache
    export HF_HUB_CACHE=\$HF_HOME TRANSFORMERS_CACHE=\$HF_HOME
    export WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    $*
  " > "$OUT/$tag.log" 2>&1
  echo "    exit=$? -> $OUT/$tag.log"
}

# 1. eval pipeline head: seeds, injects actions, generates, reads back
run inject "IE_CONFIG=configs/causal_lora_diffusion_teacher_v14e.yaml \
  IE_CKPT=$CKPT IE_PHASE=A \
  IE_WINDOWS=$ARR/analysis/eval_final/phaseA_windows.json \
  IE_MANIFEST=$ARR/analysis/eval_final/manifest_unseen.pt \
  IE_OUT=$OUT/inject IE_STEPS=8 \
  python evaluation/inject_eval.py"

# 2. inference / rollout
run infer "python inference.py --config_path configs/causal_lora_diffusion_teacher_v14e.yaml"

# 3. CoTracker motion extraction (the input the action basis is fitted on)
run motion "python preprocessing/pre_encode_motion.py"

# 4. argument surfaces of the remaining preprocessing entry points
run preproc_args "python preprocessing/pre_encode_text.py --help >/dev/null && \
  python preprocessing/ride_level_caption.py --help >/dev/null && \
  python preprocessing/pre_encode_local.py --help >/dev/null && echo ALL_PREPROC_ARGS_OK"

# 5. numeric baseline, on a compute node
run baseline "cd $ARR && python tools/release_baseline.py --check"

echo
echo "=== summary ==="
for f in "$OUT"/*.log; do
  printf "  %-16s %s\n" "$(basename "$f" .log)" "$(tail -1 "$f" | cut -c1-90)"
done
