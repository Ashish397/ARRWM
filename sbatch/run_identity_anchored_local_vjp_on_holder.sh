#!/bin/bash
set -euo pipefail

cd /lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM
source /lus/lfs1aip2/scratch/u6ex/as1748.u6ex/miniforge3/etc/profile.d/conda.sh
conda activate arrwm

BASE=${BASE:-/scratch/u6ex/as1748.u6ex/ARRWM_data/gan_aligned_discrimination/surrogate_field_2708/local_vjp_granular_audit_r0_h6158256}
BANK="$BASE/bank"
RESULTS=${RESULTS:-$BASE/results_identity_anchor}
UPDATES=${UPDATES:-1200}
CHECKPOINTS=${CHECKPOINTS:-0,1,3,10,30,100,300,600,900,1200}
GRANULARITY=${GRANULARITY:-granular}
STAGE_LIST=${STAGE_LIST:-1 2 3 6 7 9 10 11 13 14 15}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-}
read -r -a STAGES <<<"$STAGE_LIST"
mkdir -p "$RESULTS"
checkpoint_args=()
if [ -n "$CHECKPOINT_DIR" ]; then
  checkpoint_args=(--checkpoint-dir "$CHECKPOINT_DIR")
fi

for wave_start in 0 4 8; do
  pids=()
  wave_end=$((wave_start + 3))
  if [ "$wave_end" -ge "${#STAGES[@]}" ]; then
    wave_end=$((${#STAGES[@]} - 1))
  fi
  for item in $(seq "$wave_start" "$wave_end"); do
    stage=${STAGES[$item]}
    gpu=$((item - wave_start))
    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
      python analysis/gan_tuning/local_vjp_audit.py fit-stage \
        --bank-dir "$BANK" --stage-index "$stage" --device cuda \
        --granularity "$GRANULARITY" --identity-anchor \
        "${checkpoint_args[@]}" \
        --updates "$UPDATES" --checkpoints "$CHECKPOINTS" \
        --output "$RESULTS/stage${stage}.json" \
        >"$BASE/fit_identity_stage${stage}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
done

date -u +'%Y-%m-%dT%H:%M:%SZ' >"$RESULTS/COMPLETE"
