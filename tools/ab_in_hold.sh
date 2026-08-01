#!/bin/bash
# Run both A/B sides sequentially inside one held allocation, so they share
# hardware exactly and the only difference is the tree under test.
JOB="${1:?usage: ab_in_hold.sh <hold_jobid>}"
ARR=/scratch/u6ex/as1748.u6ex/ARRWM
side() {  # side <tag> <repo> <config>
  local tag="$1" repo="$2" cfg="$3"
  mkdir -p "$ARR/logs/abh_$tag"
  ln -sf "$ARR/logs/v14e_noatok/.ride_manifest.pt" "$ARR/logs/abh_$tag/.ride_manifest.pt"
  srun --overlap --jobid="$JOB" --ntasks=1 bash -lc "
    source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
    cd $repo
    export DATA_ROOT=/projects/u6ex/fbots WAN_MODELS=/scratch/u6ex/as1748.u6ex/frodobots
    export AF_ROOT=$ARR PYTHONPATH=$repo ARRWM_ACTION_ENCODER=pca_raw
    export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache
    export HF_HUB_CACHE=\$HF_HOME TRANSFORMERS_CACHE=\$HF_HOME HF_HUB_OFFLINE=1
    export WANDB_MODE=offline TMPDIR=/tmp
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
    torchrun --nnodes=1 --nproc_per_node=\$(python -c 'import torch;print(torch.cuda.device_count())') \
      --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
      train.py --config_path $cfg --logdir $ARR/logs/abh_$tag --wandb-save-dir wandb
  " > "$ARR/logs/abh_$tag.log" 2>&1
  echo "  $tag exit=$? steps=$(grep -cE '\| step +[0-9]+ \| loss' "$ARR/logs/abh_$tag.log" 2>/dev/null)"
}
echo "=== A: pre-cleanup repo ==="; side main "$ARR" "$ARR/_ab_main.yaml"
echo "=== B: code_release ===";    side rel  "$ARR/code_release" "$ARR/_ab_release.yaml"
