#!/bin/bash
# Validate the regenerated fleets and build the immutable evaluation manifest.
#SBATCH --job-name=a32-setup
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler/aligned32_stage/logs/%x_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler/aligned32_stage/logs/%x_%j.err
set -euo pipefail

R=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler
CODE="$R/ARRWM"
PY="$R/miniforge3/envs/arrwm/bin/python3.10"
OUT="$CODE/grids/eval/out_iclr_aligned32_20260923"
ALIGNED="$R/aligned32_stage/fleet30s_aligned32"
export PATH="$R/miniforge3/envs/arrwm/bin:$PATH"
mkdir -p "$OUT/logs"
test -f "$OUT/logs/FLEET_GATE_COMPLETE"
rm -f "$OUT/logs/SETUP_COMPLETE"

export ARR="$CODE"
export OURS30S_DIR="$R/_logs/ours30s/out"
export OURS_ODE_DIR="$R/iclrv3_localfleet/experiments/e1/rec/long40/.motion_check"
export MINWM_ODE_DIR="$R/aligned32_stage/minwm_ode"
export MINWM_ODE_SWAP_YAW=0
export FLEET30S_ALIGNED32_DIR="$ALIGNED"
export SEED65_DIR="$R/aligned32_stage/seed_stream_aligned"
export PYTHONPATH="$CODE/grids/eval:$CODE/code_release:${PYTHONPATH:-}"

cd "$CODE"
"$PY" grids/eval/validate_aligned32_fleets.py \
  --root "$ALIGNED" \
  --uids "$CODE/experiments/e1/scene_shortlist/e1_32_windows.json" \
  --seed-frame32 "$R/aligned32_stage/seed_frame32" \
  --seed65-dir "$SEED65_DIR" \
  --minwm-ode-dir "$MINWM_ODE_DIR" \
  --matrix-runner "$CODE/code_release/baselines/matrixgame_runner.py" \
  --matrix-config "$R/aligned32_stage/Matrix-Game-2/configs/inference_yaml/inference_universal.yaml" \
  --matrix-checkpoint "$R/aligned32_stage/Matrix-Game-2/Matrix-Game-2.0/base_distilled_model/base_distill.safetensors" \
  >"$OUT/logs/fleet_validation.json"
"$PY" grids/eval/final_v2_iclr.py --out "$OUT" --stage manifest \
  >"$OUT/logs/manifest.log" 2>&1
"$PY" grids/eval/final_v2_iclr.py --out "$OUT" --stage decode \
  >"$OUT/logs/decode.log" 2>&1
"$PY" grids/eval/final_v2_iclr.py --out "$OUT" --stage enrich \
  >"$OUT/logs/enrich.log" 2>&1

# Decode both ends of the conditioning prefix for every one of the 4,320
# submitted videos and compare them with the canonical source-video span.
# This checks the actual media, independently of filenames and sidecars.
"$PY" grids/eval/validate_manifest_context_alignment.py \
  --manifest "$OUT/video_manifest.csv" \
  --seed65-dir "$SEED65_DIR" \
  --workers 4 \
  --output "$OUT/context_alignment_rows.csv" \
  >"$OUT/logs/context_alignment.json"

"$PY" - "$OUT/video_manifest.csv" <<'PY'
import sys
import pandas as pd
p = sys.argv[1]
d = pd.read_csv(p)
assert len(d) == 4320 and d.local_video.all()
assert (d.groupby('model').size() == 288).all()
assert d.decoded_frames.notna().all()
assert (d.decoded_frames == d.container_frames).all()
assert (d.generated_duration_s >= 30).all()
print('ALIGNED32_MANIFEST_VALIDATED rows=4320 models=15')
PY
touch "$OUT/logs/SETUP_COMPLETE"
echo "ALIGNED32_EVAL_SETUP_COMPLETE $(date -Is)"
