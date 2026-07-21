#!/bin/bash
#SBATCH --job-name=minwm-sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM/third_party/minWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache TMPDIR=/tmp
for st in 0.16 0.32; do
  MW_STEPT=$st MW_TAG="_t${st}" MW_WINDOWS="8:1" MW_DIRS="F:B" python minwm_runner.py
done
cd /scratch/u6ex/as1748.u6ex/ARRWM
python - <<'PY'
import sys; sys.path.insert(0,"utils")
exec(open("utils/minwm_calib.py").read().split('print(f"{')[0])   # reuse robust()
import os
for st in ("0.16","0.32"):
    for wi in (8,1):
        for d in ("F","B"):
            p=f"logs/eval_final/minwm_smoke/minwm_t{st}_r{wi:02d}_{d}.mp4"
            if os.path.exists(p):
                rec,rot=robust(p); print(f"t{st} r{wi:02d}_{d}: rec={rec:.1f}% rot={rot:.1f}")
PY
echo "MINWM-SWEEP done"
