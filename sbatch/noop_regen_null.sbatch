#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
#
# Regenerate the no-op (static) rollouts for ALL 7 v14e ablations on the 32
# Phase-A windows, with a NON-ZERO null action vector.
#
# Why: z=(0,0) is not zero motion. The pca_raw action encoder is mean-centred
# (P = (flat - pca_mean) @ pca_comp.T), so z=0 commands the DATASET-MEAN
# motion, whose flow field has a radial expansion of +0.147 -- a mild forward
# creep. Inverting for flat=0 gives z=(-0.0234,-0.0013), which matches the
# median encoding of the real stationary continuations (-0.0230,-0.0014).
# Whether the MODEL zeroes out at the encoder's inverse is a separate
# question, hence the candidate sweep.
#
# Required env (pass with --export):
#   IE_NULL_Z   "throttle,steer"  e.g. "-0.05,0"
#   NULL_TAG    output suffix     e.g. "m0p05"
# Writes logs/eval_final/B_Astarts_null${NULL_TAG}/<run>/control_test/, leaving
# the original zero-vector set untouched. inject_eval skips any rollout whose
# mp4 already exists, so re-running resumes rather than duplicating.
set -e -o pipefail
ARR=/scratch/u6ex/as1748.u6ex/ARRWM
cd $ARR
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=$ARR:$PYTHONPATH TMPDIR=/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

: "${IE_NULL_Z:?must pass IE_NULL_Z}"
: "${NULL_TAG:?must pass NULL_TAG}"
echo "[regen] IE_NULL_Z=$IE_NULL_Z  tag=$NULL_TAG  start $(date)"

RUNS=(
  "pca8_8node:causal_lora_diffusion_teacher_v14e.yaml:v14e_pca8_raw"
  "pca4:causal_lora_diffusion_teacher_v14e_pca4.yaml:v14e_pca4"
  "pca2:causal_lora_diffusion_teacher_v14e_pca2.yaml:v14e_pca2"
  "16node:causal_lora_diffusion_teacher_v14e_16node.yaml:v14e_16node"
  "4node:causal_lora_diffusion_teacher_v14e_4node.yaml:v14e_4node"
  "noatok:causal_lora_diffusion_teacher_v14e_noatok.yaml:v14e_noatok"
  "noadaln:causal_lora_diffusion_teacher_v14e_noadaln.yaml:v14e_noadaln"
)

for ENTRY in "${RUNS[@]}"; do
  RUN=${ENTRY%%:*}; REST=${ENTRY#*:}; CFG=${REST%%:*}; LOGD=${REST#*:}
  OUT=$ARR/logs/eval_final/B_Astarts_null${NULL_TAG}/${RUN}
  mkdir -p "$OUT/control_test"   # inject_eval appends control_test to IE_OUT
  DONE=$(find "$OUT/control_test" -maxdepth 1 -name "step05000_r??_static_raw.mp4" 2>/dev/null | wc -l)
  if [ "$DONE" -ge 32 ]; then echo "[regen] $RUN complete, skip"; continue; fi
  T0=$(date +%s)
  # 4 shards, one per GPU, over the 32 windows
  for S in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$S IE_SHARD=$S IE_NSHARDS=4 \
    IE_CONFIG=configs/$CFG IE_CKPT=logs/$LOGD/causal_lora_step0005000.pt \
    IE_PHASE=B IE_WINDOWS=analysis/eval_final/phaseA_windows.json \
    IE_MANIFEST=analysis/eval_final/manifest_unseen.pt \
    IE_OUT="$OUT" IE_NULL_Z="$IE_NULL_Z" \
      python utils/inject_eval.py \
        > $ARR/logs/regen_${NULL_TAG}_${RUN}_g${S}.log 2>&1 &
  done
  wait
  T1=$(date +%s)
  echo "[regen] $RUN done in $((T1-T0))s -> $(find "$OUT/control_test" -maxdepth 1 -name "step05000_r??_static_raw.mp4" | wc -l)/32"
done

echo "[regen] ALL DONE tag=$NULL_TAG $(date)"
