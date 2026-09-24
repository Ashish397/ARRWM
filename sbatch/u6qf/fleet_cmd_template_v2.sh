#!/bin/bash
# Holder command v2: run one or more DMD arms' 30 s fleets on the 4 GPUs of the held node.
# Filled in: ARMS="<tag>[,<tag>...]" WINS="<uid,...>" (default all 32) ; ALLOC is set by the holder.
# v2 changes: makes the lean checkpoint on the compute node when missing (login node kills it),
# and lets Slurm bind one GPU per task (--gpus-per-task=1) instead of overriding CUDA_VISIBLE_DEVICES,
# which left task 3 with "No CUDA GPUs are available" under ConstrainDevices.
ARMS="__ARMS__"; WINS="__WINS__"
R=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler; A=$R/ARRWM; L=$R/_logs/ours30s
export PATH=$R/miniforge3/envs/arrwm/bin:$PATH
export PYTHONPATH=$A:$A/action-forcing PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=$R/hf_cache TORCH_HOME=$R/torch_home
cd $A
[ -z "$WINS" ] && WINS=$(python3 -c "import json; print(','.join(x['uid'] for x in json.load(open('experiments/e1/scene_shortlist/e1_32_windows.json'))))")
for ARM in ${ARMS//,/ }; do
  CKPT=$R/ckpts/$ARM/phase1_step0001000.pt.lean.pt
  if [ ! -s $CKPT ]; then
    echo "[cmd] making lean for $ARM $(date)"
    srun --overlap --jobid=$ALLOC --nodes=1 --ntasks=1 --cpus-per-task=16 bash -c "CUDA_VISIBLE_DEVICES= python interactive/engine.py --save_lean_ckpt --ckpt $R/ckpts/$ARM/phase1_step0001000.pt 2>&1 | grep -E 'Lean|rror' | tail -2"
    [ -s $CKPT ] || { echo "[cmd] lean FAILED for $ARM"; continue; }
  fi
  echo "[cmd] fleet $ARM windows $WINS $(date)"
  srun --overlap --jobid=$ALLOC --nodes=1 --ntasks=4 --ntasks-per-node=4 --gpus-per-task=1 --cpus-per-task=16 --exact bash -c '
    W=$(python3 -c "w=\"'"$WINS"'\".split(\",\"); k=int(\"$SLURM_PROCID\"); n=(len(w)+3)//4; print(\",\".join(w[k*n:(k+1)*n]))")
    echo "[task $SLURM_PROCID] CVD=$CUDA_VISIBLE_DEVICES $(nvidia-smi -L 2>/dev/null | head -1) windows $W $(date)"
    [ -z "$W" ] && exit 0
    python interactive/external_models/ours_fleet_30s.py --ckpt '"$CKPT"' --tag '"$ARM"' --out '"$R"'/_logs/ours30s/out/'"$ARM"' \
      --bundle experiments/e1/seed_bundle_e1 --wan_model_path '"$R"'/frodobots/Wan2.1-T2V-1.3B/ --windows $W \
      >> '"$L"'/'"$ARM"'_task$SLURM_PROCID.log 2>&1
    echo "[task $SLURM_PROCID] exit $? $(date)"'
done
echo "[cmd] all arms done $(date)"
