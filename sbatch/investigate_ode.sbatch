#!/bin/bash
#SBATCH --job-name=invest-ode
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# ODE-step collapse investigation (controlled A/B):
#  1) weight drift step0 vs step250 (did training destroy the teacher init?)
#  2) probe the STEP-0 checkpoint (= pure teacher init) through the IDENTICAL
#     serve path -> bisects training-damage vs serve-path bug
set -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=$PWD:$PWD/action-forcing TMPDIR=/tmp

echo "=== 1) weight drift ==="
python - <<'PY'
import numpy, torch
P='logs/ode14e_pilot/run_gt_cap'
a = torch.load(f'{P}/action_ode_step0000000.pt', map_location='cpu', mmap=True, weights_only=False)
b = torch.load(f'{P}/action_ode_step0000250.pt', map_location='cpu', mmap=True, weights_only=False)
ga, gb = a['generator'], b['generator']
import math
tot_n, tot_d = 0.0, 0.0
worst = []
for k in ga:
    ta, tb = ga[k].float(), gb[k].float()
    dn, dd = (tb-ta).norm().item()**2, ta.norm().item()**2
    tot_n += dn; tot_d += dd
    worst.append(((dn/max(dd,1e-12))**0.5, k))
worst.sort(reverse=True)
print(f"GLOBAL generator rel-drift: {(tot_n/max(tot_d,1e-12))**0.5:.4f}")
for r, k in worst[:5]: print(f"  worst {r:.4f} {k[-60:]}")
for head in ('action_projection','action_token_projection'):
    pa, pb = a[head], b[head]
    n = sum(((pb[k].float()-pa[k].float()).norm().item()**2) for k in pa)
    d = sum((pa[k].float().norm().item()**2) for k in pa)
    print(f"{head} rel-drift: {(n/max(d,1e-12))**0.5:.4f}")
PY

echo "=== 2) probe STEP-0 (pure teacher init) ==="
FR_RUN=pilot_gt0 FR_CONFIG=configs/ar_eval_dmd_student.yaml \
  FR_CKPT=logs/ode14e_pilot/run_gt_cap/action_ode_step0000000.pt \
  FR_RUNGS="1000,625,357.142857,208.333333" FR_VIDEO=1 FR_CHUNKS=6 FR_NSEEDS=2 \
  python utils/flow_record_ode_student.py
FDD_NSEEDS=2 FDD_PAIRS="pilot_gt0=14e8" FDD_FIGNAME=flow_pilot_gt0_scorecard \
  python utils/flow_diverge_dmd3.py
MC_RUNS=pilot_gt0 python utils/motion_check.py
echo "INVEST-ODE done"
