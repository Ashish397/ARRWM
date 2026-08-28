#!/bin/bash
# Compute-matched offline controller benchmark inside an existing holder.
# Never submits or cancels an allocation and never launches a live GAN.
set -euo pipefail

: "${HOLDER:?set HOLDER to the dedicated allocation}"
ROOT=/scratch/u6ex/as1748.u6ex/ARRWM
TARGETS=${TARGETS:-/scratch/u6ex/as1748.u6ex/ARRWM_data/gan_aligned_discrimination/surrogate_field_2708/targets_2crop_h6156714}
MODE=${MODE:-smoke}
OUT=${OUT:-/scratch/u6ex/as1748.u6ex/ARRWM_data/gan_aligned_discrimination/surrogate_rl_h${HOLDER}_${MODE}}
NODE_INDEX=${NODE_INDEX:-0}

if [[ "$MODE" != smoke && "$MODE" != decisive ]]; then
  echo "[SURROGATE-RL] MODE must be smoke or decisive, got $MODE" >&2
  exit 11
fi

cd "$ROOT"
node_expr=$(squeue -j "$HOLDER" -h -o %N)
mapfile -t nodes < <(scontrol show hostnames "$node_expr")
test "${#nodes[@]}" -gt 0
node=${nodes[$NODE_INDEX]:-${nodes[0]}}
mkdir -p "$OUT" logs
log=logs/surrogate_rl_h${HOLDER}_${MODE}_$(date +%Y%m%d_%H%M%S).log

# Python processes start at different times. Snapshot every imported local
# source so another live session cannot change the experiment between policy
# training and evaluation.
snapshot=$(mktemp -d /tmp/arrwm_surrogate_rl.XXXXXX)
trap 'rm -rf -- "$snapshot"' EXIT
mkdir -p "$snapshot/analysis/gan_tuning" "$snapshot/model"
cp -- analysis/gan_tuning/surrogate_rl_controller_benchmark.py \
  "$snapshot/analysis/gan_tuning/"
cp -- model/latent_gradient_surrogate.py model/latent_texture_critic.py \
  "$snapshot/model/"
sha256sum \
  analysis/gan_tuning/surrogate_rl_controller_benchmark.py \
  model/latent_gradient_surrogate.py model/latent_texture_critic.py \
  > "$OUT/source_sha256.txt"

echo "[SURROGATE-RL] holder=$HOLDER node=$node mode=$MODE out=$OUT" | tee "$log"
srun --jobid="$HOLDER" --overlap --nodelist="$node" --nodes=1 --ntasks=1 \
  --gpus-per-node=4 --gpu-bind=none bash -lc '
set -euo pipefail
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
SNAPSHOT='"$(printf %q "$snapshot")"'
ROOT='"$(printf %q "$ROOT")"'
TARGETS='"$(printf %q "$TARGETS")"'
OUT='"$(printf %q "$OUT")"'
MODE='"$(printf %q "$MODE")"'
export PYTHONPATH="$SNAPSHOT" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$SNAPSHOT"
script=analysis/gan_tuning/surrogate_rl_controller_benchmark.py

python "$script" --targets-dir "$TARGETS" validate \
  --output "$OUT/isolation.json" > "$OUT/isolation.log" 2>&1

if [ "$MODE" = smoke ]; then
  CUDA_VISIBLE_DEVICES=0 python "$script" --targets-dir "$TARGETS" train-policy \
    --rotation 0 --policy-index 0 --meta-episodes-per-bank 1 --substeps 2 \
    --batch-size 3 --width 16 --blocks 1 --max-steps 3 \
    --output "$OUT/r0_policy0.json" > "$OUT/r0_policy0.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 python "$script" --targets-dir "$TARGETS" \
    evaluate-rotation --rotation 0 --policy-dir "$OUT" \
    --methods fixed,hand,bandit,oracle --student-seeds 27082026 \
    --substeps 2 --batch-size 3 --width 16 --blocks 1 --max-steps 3 \
    --output "$OUT/rotation0.json" > "$OUT/rotation0.log" 2>&1
  python "$script" --targets-dir "$TARGETS" summarize \
    --inputs "$OUT/rotation0.json" --output "$OUT/SUMMARY.json" \
    > "$OUT/summary.log" 2>&1
  exit 0
fi

# Fit three independently seeded policies inside each nested rotation. Each
# fit uses only the two train ranks and one validation rank in that fold. Run
# in waves of four to respect the four GPUs on the node.
jobs=()
for rotation in 0 1 2; do
  for index in 0 1 2; do jobs+=("$rotation:$index"); done
done
for offset in 0 4 8; do
  pids=()
  labels=()
  for slot in 0 1 2 3; do
    position=$((offset + slot))
    if [ "$position" -ge "${#jobs[@]}" ]; then continue; fi
    IFS=: read -r rotation index <<< "${jobs[$position]}"
    CUDA_VISIBLE_DEVICES="$slot" python "$script" --targets-dir "$TARGETS" \
      train-policy --rotation "$rotation" --policy-index "$index" \
      --meta-episodes-per-bank 3 --substeps 24 --batch-size 6 \
      --width 96 --blocks 6 --max-steps 9 \
      --output "$OUT/r${rotation}_policy${index}.json" \
      > "$OUT/r${rotation}_policy${index}.log" 2>&1 &
    pids+=("$!")
    labels+=("r${rotation}_policy${index}")
  done
  rc=0
  for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
  if [ "$rc" -ne 0 ]; then
    for label in "${labels[@]}"; do tail -n 100 "$OUT/${label}.log" || true; done
    exit 31
  fi
done

# Each rotation gets a GPU and uses its three already-frozen nested policies
# (one per reported seed). Evaluation never updates them.
pids=()
for rotation in 0 1 2; do
  CUDA_VISIBLE_DEVICES="$rotation" python "$script" --targets-dir "$TARGETS" \
    evaluate-rotation --rotation "$rotation" --policy-dir "$OUT" \
    --methods fixed,hand,bandit,oracle \
    --substeps 24 --batch-size 6 --width 96 --blocks 6 --max-steps 9 \
    --output "$OUT/rotation${rotation}.json" \
    > "$OUT/rotation${rotation}.log" 2>&1 &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
if [ "$rc" -ne 0 ]; then
  tail -n 120 "$OUT"/rotation*.log || true
  exit 41
fi

python "$script" --targets-dir "$TARGETS" summarize \
  --inputs "$OUT/rotation0.json" "$OUT/rotation1.json" "$OUT/rotation2.json" \
  --output "$OUT/SUMMARY.json" > "$OUT/summary.log" 2>&1
' >> "$log" 2>&1

echo "[SURROGATE-RL] complete summary=$OUT/SUMMARY.json" | tee -a "$log"
