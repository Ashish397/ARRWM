#!/bin/bash
# One of four disjoint all-model evaluation shards on a four-GPU node.
#SBATCH --job-name=a32-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler/aligned32_stage/logs/%x_%A_%a.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler/aligned32_stage/logs/%x_%A_%a.err
set -euo pipefail

R=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler
CODE="$R/ARRWM"
PY="$R/miniforge3/envs/arrwm/bin/python3.10"
OUT="$CODE/grids/eval/out_iclr_aligned32_20260923"
export PATH="$R/miniforge3/envs/arrwm/bin:$PATH"
SHARD=${SLURM_ARRAY_TASK_ID:?array index required}
LOG="$OUT/logs/shard${SHARD}"
mkdir -p "$LOG"
rm -f "$LOG/COMPLETE" "$LOG/FAILED"
test -f "$OUT/logs/SETUP_COMPLETE"

export ARR="$CODE"
export OURS30S_DIR="$R/_logs/ours30s/out"
export OURS_ODE_DIR="$R/iclrv3_localfleet/experiments/e1/rec/long40/.motion_check"
export MINWM_ODE_DIR="$R/aligned32_stage/minwm_ode"
export MINWM_ODE_SWAP_YAW=0
export FLEET30S_ALIGNED32_DIR="$R/aligned32_stage/fleet30s_aligned32"
export SEED65_DIR="$R/aligned32_stage/seed_stream_aligned"
export HF_HOME="$R/iclrv3_hf_cache"
export TORCH_HOME="$R/torch_home"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$CODE/grids/eval:$CODE/code_release:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

cd "$CODE"
pids=()
names=()
"$PY" grids/eval/final_v2_cpu.py --out "$OUT" \
  --shard-index "$SHARD" --shard-count 4 >"$LOG/cpu.log" 2>&1 &
pids+=("$!"); names+=(cpu)
CUDA_VISIBLE_DEVICES=0 "$PY" grids/eval/final_v2_style.py --out "$OUT" \
  --shard-index "$SHARD" --shard-count 4 >"$LOG/style.log" 2>&1 &
pids+=("$!"); names+=(style)
CUDA_VISIBLE_DEVICES=1 "$PY" grids/eval/final_v2_control.py --out "$OUT" \
  --shard-index "$SHARD" --shard-count 4 >"$LOG/control.log" 2>&1 &
pids+=("$!"); names+=(control)
CUDA_VISIBLE_DEVICES=2 "$PY" grids/eval/final_v3_quality_gpu.py --out "$OUT" \
  --metric geometry --shard-index "$SHARD" --shard-count 4 \
  >"$LOG/geometry.log" 2>&1 &
pids+=("$!"); names+=(geometry)
CUDA_VISIBLE_DEVICES=3 "$PY" grids/eval/final_v3_quality_gpu.py --out "$OUT" \
  --metric conjuration --shard-index "$SHARD" --shard-count 4 \
  >"$LOG/conjuration.log" 2>&1 &
pids+=("$!"); names+=(conjuration)

rc=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || { echo "FAILED ${names[$i]}"; rc=1; }
done
if [ "$rc" -ne 0 ]; then touch "$LOG/FAILED"; exit 1; fi
touch "$LOG/COMPLETE"
echo "ALIGNED32_EVAL_SHARD_COMPLETE shard=$SHARD $(date -Is)"
