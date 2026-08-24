#!/bin/bash
# NOISE-FLOOR PROTOCOL for the texture battery.
#
# WHY: docs/TEXABC_REFERENCE_TABLE.md measured (on the texture_abc instrument)
# that C_late fft-anisotropy and angular entropy are NOISE-DOMINATED across
# ODE_FLOW_SEED (sd up to +-0.34), while HF power and Laplacian kurtosis are the
# reliable axes (>=2.8 sigma). Our `seed_log_dist` votes on hv_anisotropy +
# haar_HL_LH_ratio and deliberately EXCLUDES hf_power (measured 0/29 agreement
# with the researcher's ranking in the late rollout).
#
# Those two facts are not contradictory -- hf_power can be a PRECISE measurement
# of something that does not track the eye, and anisotropy a NOISY measurement of
# something that does. But it means our verdict rests on a noisy axis, and
# **haar_HL_LH_ratio's noise floor has never been measured at all** even though
# it is our strongest voter (49/59 overall, 29/29 in the late rollout).
#
# This runs the same checkpoint at N ODE_FLOW_SEEDs and reports, per statistic,
# mean +- sd and the between-arm gap in sigma. `--seed` is a no-op here: chunk
# noise comes from ODE_FLOW_SEED (utils/eval_causal_AR.py:977).
#
# Usage: ARM=<name> CKPT=<path> SEEDS="1234 43 44" bash sbatch/seed_noise_probe.sh
set -uo pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
: "${ARM:?}" ; : "${CKPT:?}"
SEEDS=${SEEDS:-"1234 43 44"}
OUT=${OUT:-analysis/seed_noise}
mkdir -p "$OUT"

for s in $SEEDS; do
  tag="${ARM}_s${s}"
  if [ ! -d "eval/${tag}_step200_carn05_60s" ]; then
    echo "[seed] eval $tag $(date)"
    EVAL_SEED=$s ARM="$tag" CKPT="$CKPT" bash sbatch/run_eval60.sh \
      > "$OUT/${tag}_eval.log" 2>&1
    echo "[seed] eval $tag exit=$? $(date)"
  fi
  source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
  OMP_NUM_THREADS=4 python analysis/rollout_quality.py \
    "${tag}=eval/${tag}_step200_carn05_60s" --win 5 --no-orb \
    --json "$OUT/${tag}.json" > "$OUT/${tag}_score.txt" 2>&1
  echo "[seed] score $tag exit=$?"
done
echo "[seed] done $ARM $(date)"
