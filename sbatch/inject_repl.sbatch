#!/bin/bash
#SBATCH --job-name=inject-repl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH TMPDIR=/tmp
echo "replace: phase=$IE_PHASE out=$IE_OUT slots=$IE_ONLY"
# purge stale JSONL lines for the replaced slots (old night windows)
python - <<'PYC'
import glob, json, os, shutil
slots={int(x) for x in os.environ["IE_ONLY"].replace(":",",").split(",")}
for f in glob.glob(os.path.join(os.environ["IE_OUT"],"control_test","metrics_r*.jsonl")):
    keep=[]
    for ln in open(f):
        try: d=json.loads(ln)
        except Exception: continue
        if d.get("rank") not in slots: keep.append(ln)
    shutil.copy(f,f+".night_bak"); open(f,"w").writelines(keep)
    print(f"cleaned {f}")
PYC
srun --ntasks=4 --gpus-per-node=4 bash -c '
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  export IE_SHARD=$SLURM_PROCID IE_NSHARDS=4
  export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_${USER}_${SLURM_PROCID} TRITON_CACHE_DIR=/tmp/tr_${USER}_${SLURM_PROCID}
  python utils/inject_eval.py > logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_shard${SLURM_PROCID}.log 2>&1
'
echo "INJECT-REPL done $(date)"
