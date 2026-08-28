#!/bin/bash
# Run inside an already-running four-GPU holder batch command.  This script
# never submits, cancels, or alters the holder allocation itself.
set -euo pipefail

cd /lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM
source /lus/lfs1aip2/scratch/u6ex/as1748.u6ex/miniforge3/etc/profile.d/conda.sh
conda activate arrwm

OUT=${OUT:-/scratch/u6ex/as1748.u6ex/ARRWM_data/gan_aligned_discrimination/surrogate_field_2708/local_vjp_audit_r0_h6158256}
PAIRED=${PAIRED:-/scratch/u6ex/as1748.u6ex/ARRWM_data/gan_aligned_discrimination/surrogate_field_2708/paired_decoder_pullback_r0_h6158256/paired_vgg_r0.npz}
BANK="$OUT/bank"
RESULTS=${RESULTS:-$OUT/results}
UPDATES=${UPDATES:-300}
CHECKPOINTS=${CHECKPOINTS:-0,1,3,10,30,100,300}
GRANULARITY=${GRANULARITY:-macro}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-}
NUM_STAGES=6
if [ "$GRANULARITY" = fine ]; then
  NUM_STAGES=9
elif [ "$GRANULARITY" = granular ]; then
  NUM_STAGES=17
fi
mkdir -p "$BANK" "$RESULTS"
checkpoint_args=()
if [ -n "$CHECKPOINT_DIR" ]; then
  checkpoint_args=(--checkpoint-dir "$CHECKPOINT_DIR")
fi

PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q testing/test_local_vjp_audit.py >"$OUT/tests_full.log" 2>&1

if [ "${SKIP_EXTRACT:-0}" != 1 ]; then
  pids=()
  for shard in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$shard PYTHONPATH=. OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
      python analysis/gan_tuning/local_vjp_audit.py extract-shard \
        --paired-bank "$PAIRED" --output-dir "$BANK" --device cuda \
        --granularity "$GRANULARITY" \
        --shard-index "$shard" --num-shards 4 \
        >"$OUT/extract_shard${shard}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
fi

if [ -n "${STAGE_LIST:-}" ]; then
  read -r -a STAGE_IDS <<<"$STAGE_LIST"
else
  mapfile -t STAGE_IDS < <(seq 0 $((NUM_STAGES - 1)))
fi
for wave_start in $(seq 0 4 $((${#STAGE_IDS[@]} - 1))); do
  wave_end=$((wave_start + 3))
  if [ "$wave_end" -ge "${#STAGE_IDS[@]}" ]; then
    wave_end=$((${#STAGE_IDS[@]} - 1))
  fi
  pids=()
  for item in $(seq "$wave_start" "$wave_end"); do
    stage=${STAGE_IDS[$item]}
    gpu=$((item - wave_start))
    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
      python analysis/gan_tuning/local_vjp_audit.py fit-stage \
        --bank-dir "$BANK" --stage-index "$stage" --device cuda \
        --granularity "$GRANULARITY" \
        "${checkpoint_args[@]}" \
        --updates "$UPDATES" --checkpoints "$CHECKPOINTS" \
        --output "$RESULTS/stage${stage}.json" \
        >"$OUT/fit_stage${stage}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
done

date -u +'%Y-%m-%dT%H:%M:%SZ' >"$OUT/COMPLETE"
