#!/usr/bin/env bash
# Matched melt-VLM benchmark on the 84 labelled videos, all three candidates + AUC table.
# Run from repo root with the GPU free:  bash grids/eval/run_melt_local.sh
set -e
cd "$(dirname "$0")/../.."          # repo root
export PYTHONPATH="$PWD:$PYTHONPATH"

for M in qwen25vl7b internvl3_8b cosmos_reason1_7b; do
  echo "=================  $M  ================="
  MELT_MODEL=$M python grids/eval/melt_vlm_bench.py
done

echo "=================  matched AUC table  ================="
python grids/eval/melt_auc_table.py
