#!/bin/bash
# Extract the shipped zip somewhere clean and run it from THERE.
#
# This is the check neither agent had done: everything else was run inside the
# repo, where a file the zip omits is still on disk and a path that resolves by
# accident still resolves. Run under an allocation; needs a GPU for stages 4-6.
#
#   bash tools/verify_from_zip.sh 2>&1 | tee verification/from_zip.txt
set -u

ZIP=/scratch/u6ex/as1748.u6ex/ARRWM/code_release.zip
WORK=/scratch/u6ex/as1748.u6ex/_zipcheck
CONDA=/lus/lfs1aip2/scratch/u6ex/as1748.u6ex/miniforge3
RAW=/projects/u6ex/fbots/frodobots_raw/frodobots_data
MANIFEST=/scratch/u6ex/as1748.u6ex/ARRWM/logs/abh_main/.ride_manifest.pt

pass=0; fail=0
step () {  # step <name> <cmd...>
  local name="$1"; shift
  printf '%-46s ' "$name"
  local out; out=$("$@" 2>&1); local rc=$?
  if [ $rc -eq 0 ]; then echo "PASS"; pass=$((pass+1))
  else echo "FAIL (rc=$rc)"; echo "$out" | tail -6 | sed 's/^/      /'; fail=$((fail+1)); fi
}

echo "=== extracting to a clean directory ==="
rm -rf "$WORK"; mkdir -p "$WORK"
unzip -q "$ZIP" -d "$WORK" || { echo "unzip failed"; exit 1; }
cd "$WORK/code_release" || exit 1
echo "extracted $(find . -type f | wc -l) files to $WORK/code_release"

source "$CONDA/bin/activate" && conda activate arrwm || exit 1
export AF_ROOT="$PWD" PYTHONPATH="$PWD" MPLBACKEND=Agg
export DATA_ROOT=/projects/u6ex/fbots
export WAN_MODELS=/scratch/u6ex/as1748.u6ex/frodobots
export HF_HOME=${HF_HOME:-/scratch/u6ex/as1748.u6ex/hf_cache}
echo "AF_ROOT=$AF_ROOT"
echo

echo "=== 1. the tree is self-contained ==="
step "pytest (the shipped suite)"        python -m pytest tests -q
step "fresh-clone gate"                  python /scratch/u6ex/as1748.u6ex/ARRWM/tools/fresh_clone_check.py
echo

echo "=== 2. figures regenerate from shipped data ==="
export WG_OUT="$WORK/out/wedges" RC_OUT="$WORK/out"
mkdir -p "$WORK/out"
step "figures/wedge_plots.py"            python figures/wedge_plots.py
step "figures/response_curves_eval.py"   python figures/response_curves_eval.py
echo

echo "=== 3. preprocessing, from raw video ==="
step "preprocessing/build_rides_csv.py"  python preprocessing/build_rides_csv.py \
        --data_root "$RAW" --out "$WORK/rides.csv"
head -2 "$WORK/rides.csv" > "$WORK/one.csv" 2>/dev/null
step "preprocessing/pre_encode_direct.py" python preprocessing/pre_encode_direct.py \
        --rides_csv "$WORK/one.csv" --output_root "$WORK/enc" \
        --vae_path "$WAN_MODELS/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
echo

echo "=== 4. training starts from the extracted tree ==="
mkdir -p "$WORK/logs/smoke"
[ -f "$MANIFEST" ] && ln -sf "$MANIFEST" "$WORK/logs/smoke/.ride_manifest.pt" \
  && echo "(reusing a prebuilt ride manifest; a cold run rescans for ~30 min)"
export ARRWM_ACTION_ENCODER=pca_raw
step "train.py, a few steps" timeout 1500 python train.py \
        --config_path configs/causal_lora_diffusion_teacher_v14e.yaml \
        --logdir "$WORK/logs/smoke"
echo

echo "======================================================"
echo "  $pass passed, $fail failed"
[ $fail -eq 0 ] && echo "  the zip runs standalone" || echo "  see failures above"
echo "======================================================"
exit $fail
