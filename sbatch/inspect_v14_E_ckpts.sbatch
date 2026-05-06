#!/bin/bash
#SBATCH --job-name=inspect-v14-E
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm

srun --ntasks=1 python - <<'PYEOF'
import torch

paths = [
    ('v14_LoRA',  '/scratch/u6ex/as1748.u6ex/ARRWM/logs/v14_balanced_weunz/causal_lora_step0006600.pt'),
    ('E_step1000','/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_ode_distill_E/action_ode_step0001000.pt'),
]

for label, path in paths:
    print(f'=== {label} ({path}) ===', flush=True)
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        print(f'  top keys: {list(ckpt.keys())}', flush=True)
        for k in ('action_critic', 'state_probe', 'action_critic_optimizer',
                  'state_probe_optimizer', 'lora', 'generator',
                  'action_projection', 'action_token_projection'):
            if k in ckpt:
                v = ckpt[k]
                if isinstance(v, dict):
                    leaves = [t for t in v.values() if hasattr(t, 'numel')]
                    if leaves:
                        n = sum(t.numel() for t in leaves)
                        sample_keys = list(v.keys())[:3]
                        print(f'  {k}: {len(v)} tensors, {n/1e6:.2f}M params, '
                              f'sample: {sample_keys}', flush=True)
                    else:
                        print(f'  {k}: dict with {len(v)} entries '
                              f'(non-tensor; sample keys: {list(v.keys())[:3]})',
                              flush=True)
                else:
                    print(f'  {k}: {type(v)}', flush=True)
            else:
                print(f'  {k}: MISSING', flush=True)
        # If action_critic / state_probe present, dump per-tensor shapes for the
        # first few to confirm shape-compat with the v18+ defaults.
        for k in ('action_critic', 'state_probe'):
            if k in ckpt and isinstance(ckpt[k], dict):
                print(f'  --- {k} tensor shapes (first 8) ---', flush=True)
                for i, (kk, vv) in enumerate(list(ckpt[k].items())[:8]):
                    if hasattr(vv, 'shape'):
                        print(f'    {kk}: {tuple(vv.shape)}', flush=True)
                    else:
                        print(f'    {kk}: {type(vv)}', flush=True)
    except Exception as e:
        print(f'  ERROR: {type(e).__name__}: {e}', flush=True)
    print(flush=True)

print('DONE', flush=True)
PYEOF
