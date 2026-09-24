#!/bin/bash
# Holder command: run one DMD arm's 30 s fleet on the 4 GPUs of the held node, 8 windows per GPU.
# Filled in by make_cmd: ARM=<tag> CKPT=<lean ckpt> ; ALLOC is set by the holder.
ARM=__ARM__; CKPT=__CKPT__
R=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler; A=$R/ARRWM; L=$R/_logs/ours30s
export PATH=$R/miniforge3/envs/arrwm/bin:$PATH   # env python + ffmpeg (FrameSink pipes to "ffmpeg")
export PYTHONPATH=$A:$A/action-forcing PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=$R/hf_cache TORCH_HOME=$R/torch_home
cd $A
srun --overlap --jobid=$ALLOC --nodes=1 --ntasks=4 --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=16 --exact bash -c '
  W=$(python3 -c "import json; w=[x[\"uid\"] for x in json.load(open(\"experiments/e1/scene_shortlist/e1_32_windows.json\"))]; k=int(\"$SLURM_PROCID\"); print(\",\".join(w[k*8:(k+1)*8]))")
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  echo "[task $SLURM_PROCID] gpu $CUDA_VISIBLE_DEVICES windows $W $(date)"
  python interactive/external_models/ours_fleet_30s.py --ckpt '"$CKPT"' --tag '"$ARM"' --out '"$R"'/_logs/ours30s/out/'"$ARM"' \
    --bundle experiments/e1/seed_bundle_e1 --wan_model_path '"$R"'/frodobots/Wan2.1-T2V-1.3B/ --windows $W \
    > '"$L"'/'"$ARM"'_task$SLURM_PROCID.log 2>&1
  echo "[task $SLURM_PROCID] exit $? $(date)"'
