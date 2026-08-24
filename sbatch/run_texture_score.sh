#!/bin/bash
# A1/A11 — score rollout videos with the anisotropic texture battery.
#
# WHY THIS RUNS ON A HOLDER: long python jobs are killed erratically on the
# login node (SIGKILL/137, observed repeatedly, not memory — peak RSS is only
# ~1.4 GB against 124 GB free). Compute nodes do not have that problem.
#
# One ARM PER PROCESS: two videos in a single process also gets killed, and a
# fresh process per arm is the cheapest robust workaround.
#
# Usage: ARMS="a b c" bash sbatch/run_texture_score.sh
set -uo pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
OUT=${OUT:-analysis/texture_score}
mkdir -p "$OUT"
WIN=${WIN:-5}

for arm in ${ARMS:?set ARMS to a space-separated list of arm names}; do
  d="eval/${arm}_step200_carn05_60s"
  [ -d "$d" ] || { echo "[skip] no eval dir for $arm"; continue; }
  echo "[score] $arm $(date)"
  python analysis/rollout_quality.py "${arm}=${d}" \
      --win "$WIN" --no-orb --json "$OUT/${arm}.json" \
      > "$OUT/${arm}.txt" 2>&1
  echo "[score] $arm exit=$? $(date)"
done

echo "[score] merging"
python - <<'PY'
import glob, json, os
out = os.environ.get("OUT", "analysis/texture_score")
rows = {}
for f in sorted(glob.glob(os.path.join(out, "*.json"))):
    try:
        rows.update(json.load(open(f)))
    except Exception as e:
        print("bad json", f, e)
if not rows:
    raise SystemExit("no results")
hdr = f"{'arm':28s} {'seed_end':>8s} {'anisoXseed':>10s} {'HLLHXseed':>9s} {'logdist':>8s}"
print(hdr); print("-" * len(hdr))
for lab, p in sorted(rows.items(), key=lambda kv: kv[1]["summary"].get("seed_log_dist", 9e9)):
    s = p["summary"]
    print(f"{lab:28s} {p['seed_end']:8d} "
          f"{s.get('hv_anisotropy_x_seed', float('nan')):10.3f} "
          f"{s.get('haar_HL_LH_ratio_x_seed', float('nan')):9.3f} "
          f"{s.get('seed_log_dist', float('nan')):8.4f}")
print("\nlower seed_log_dist = closer to the reality in the seed context")
print("(voters: hv_anisotropy + haar_HL_LH_ratio only; hf_power is REPORTED "
      "but never votes -- measured 0/29 agreement in the late rollout)")
json.dump(rows, open(os.path.join(out, "_merged.json"), "w"), indent=1)
PY
echo "[score] done $(date)"
