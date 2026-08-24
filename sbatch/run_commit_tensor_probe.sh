#!/bin/bash
# A23 launch-gate (b) probe driver — decode BOTH commit tensors from ONE
# checkpoint and run the texture battery on each.
#
# Runs on a HOLDER (compute node). The login node SIGKILLs long python jobs.
#
# Usage (from a holder command file, where ALLOC is exported):
#     HOLDER=$ALLOC bash sbatch/run_commit_tensor_probe.sh
# or standalone against any idle holder:
#     HOLDER=<jobid> bash sbatch/run_commit_tensor_probe.sh
#
# Env overrides: CKPT, TAG, CHUNKS, FLASH_T, NNODE.
set -uo pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
: "${HOLDER:?set HOLDER to a running holder job id}"

CKPT=${CKPT:-logs/dmd10k_ganfix_strict_rerun/dmd10k_ganfix_strict_rerun_h6106414_083234/eval_step0200.pt}
TAG=${TAG:-ganfix_strict_rerun_step200}
CHUNKS=${CHUNKS:-100}
FLASH_T=${FLASH_T:-60}
OUT=analysis/commit_tensor_probe/${TAG}
NODES=${NODES:-$(scontrol show hostnames "$(squeue -j "$HOLDER" -h -o %N)" | head -1)}

test -f "$CKPT" || { echo "missing ckpt $CKPT"; exit 2; }
mkdir -p "$OUT" logs

srun --jobid="$HOLDER" --overlap --nodelist="$NODES" --nodes=1 --ntasks=1 \
     --cpus-per-task=16 --gpus-per-node=4 --gpu-bind=none \
     --export=ALL,CUDA_VISIBLE_DEVICES=0 \
  bash -c "
    source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
    conda activate arrwm
    cd /scratch/u6ex/as1748.u6ex/ARRWM
    export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache
    export HF_HUB_CACHE=\$HF_HOME
    export HUGGINGFACE_HUB_CACHE=\$HF_HOME
    export TRANSFORMERS_CACHE=\$HF_HOME
    export TMPDIR=/tmp
    export OMP_NUM_THREADS=8
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export WORLD_SIZE=1 LOCAL_RANK=0
    python analysis/compare_commit_tensors.py \
      --student_ckpt '$CKPT' \
      --ar_gen_chunks $CHUNKS \
      --flash_t $FLASH_T \
      --out '$OUT'
  " > "logs/a23_commit_probe_${TAG}.log" 2>&1

rc=$?
echo "[a23] probe exit=$rc  log=logs/a23_commit_probe_${TAG}.log  out=$OUT"
tail -40 "logs/a23_commit_probe_${TAG}.log"
exit $rc
