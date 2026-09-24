#!/bin/bash
# Matched 30-second gate for v3c's own stationary step-200 parent.
# Two E1 windows x nine commands, matching v3/v3b/no-aux gates.
#SBATCH --job-name=v3cstat-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=2
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/u6qf/as1748.u6qf/ARRWM_straggler/_logs/ours30s/v3cstat_eval_%j.out
#SBATCH --error=/scratch/u6qf/as1748.u6qf/ARRWM_straggler/_logs/ours30s/v3cstat_eval_%j.err

set -euo pipefail
R=/scratch/u6qf/as1748.u6qf/ARRWM_straggler
A=$R/ARRWM
RUN=$A/logs/dmd10k_phase3_stationary_definitive_8n_base_v3c_gbs32_1709/dmd10k_phase3_stationary_definitive_8n_base_v3c_gbs32_1709_j6647725
CKPT=$RUN/phase1_step0000200.pt
OUT=$R/_logs/ours30s/out/base_v3c_stationary200
export PATH=$R/miniforge3/envs/arrwm/bin:$PATH
export PYTHONPATH=$A:$A/action-forcing
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=$R/hf_cache
export TORCH_HOME=$R/torch_home
cd "$A"

test -s "$CKPT"
python interactive/engine.py --save_lean_ckpt --ckpt "$CKPT"
test -s "$CKPT.lean.pt"
mkdir -p "$OUT"

pids=()
for spec in 0:m42 1:u31; do
  gpu=${spec%%:*}
  win=${spec#*:}
  CUDA_VISIBLE_DEVICES=$gpu python interactive/external_models/ours_fleet_30s.py \
    --ckpt "$CKPT.lean.pt" --tag base_v3c_stationary200 --out "$OUT" \
    --bundle experiments/e1/seed_bundle_e1 \
    --wan_model_path "$R/frodobots/Wan2.1-T2V-1.3B/" \
    --windows "$win" \
    > "$R/_logs/ours30s/v3cstat_eval_${SLURM_JOB_ID}_${win}.log" 2>&1 &
  pids+=("$!")
done

rc=0
for pid in "${pids[@]}"; do
  wait "$pid" || rc=1
done
test "$rc" -eq 0
count=$(find "$OUT" -maxdepth 1 -type f -name 'base_v3c_stationary200_*.mp4.json' | wc -l)
echo "base-v3c stationary gate: $count/18 clips complete"
test "$count" -eq 18
