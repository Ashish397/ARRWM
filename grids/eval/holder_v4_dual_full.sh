#!/bin/bash
# Full dual-reference style and geometry run inside a durable u6qf holder.
set -u
BASE=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler
ROOT="$BASE/eval30s_check/ARR"
PY="$BASE/miniforge3/envs/arrwm/bin/python3.10"
OUT="$ROOT/grids/eval/out_dual_reference_full"
MANIFEST="$ROOT/grids/eval/v4_manifest_u6qf.csv"
export HF_HOME="$BASE/iclrv3_hf_cache"
export TORCH_HOME="$BASE/torch_home"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p "$OUT"
cd "$ROOT"

# DINO is substantially faster; GPU 0 starts geometry shard 0 when style ends.
(
  CUDA_VISIBLE_DEVICES=0 "$PY" grids/eval/final_v4_dual_reference.py \
    --manifest "$MANIFEST" --out "$OUT/style_full.csv" --metric style \
    > "$OUT/style_full.log" 2>&1 || exit 1
  CUDA_VISIBLE_DEVICES=0 "$PY" grids/eval/final_v4_dual_reference.py \
    --manifest "$MANIFEST" --out "$OUT/geometry_shard0.csv" --metric geometry \
    --shard-index 0 --shard-count 4 > "$OUT/geometry_shard0.log" 2>&1
) & p0=$!

for shard in 1 2 3; do
  CUDA_VISIBLE_DEVICES="$shard" "$PY" grids/eval/final_v4_dual_reference.py \
    --manifest "$MANIFEST" --out "$OUT/geometry_shard${shard}.csv" --metric geometry \
    --shard-index "$shard" --shard-count 4 > "$OUT/geometry_shard${shard}.log" 2>&1 &
  eval "p${shard}=\$!"
done

status=0
for p in "$p0" "$p1" "$p2" "$p3"; do wait "$p" || status=1; done
if [ "$status" -eq 0 ]; then
  "$PY" - <<'PY' || status=2
from pathlib import Path
import pandas as pd
out = Path("grids/eval/out_dual_reference_full")
s = pd.read_csv(out / "style_full.csv")
g = pd.concat([pd.read_csv(out / f"geometry_shard{i}.csv") for i in range(4)],
              ignore_index=True)
key = ["scene", "model", "window_start_s", "condition"]
assert len(s) == 21600 and not s.duplicated(key).any(), len(s)
assert len(g) == 21600 and not g.duplicated(key).any(), len(g)
assert set(s.condition) == {"baseline"} and set(g.condition) == {"baseline"}
assert s.groupby(["model", "window_end_s"]).size().eq(288).all()
assert g.groupby(["model", "window_end_s"]).size().eq(288).all()
g.sort_values(key).to_csv(out / "geometry_full.csv", index=False)
print("validated", len(s), len(g))
PY
fi
for f in "$OUT"/*.log; do echo "== $f =="; tail -3 "$f"; done
echo "dual_reference_full_exit=$status"
exit "$status"
