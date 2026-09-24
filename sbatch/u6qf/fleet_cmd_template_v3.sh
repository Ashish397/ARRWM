#!/bin/bash
# Holder command v3: one process per GPU, launched directly from the holder's batch shell (which owns all
# of the job's GPUs). srun overlap steps on u6qf only expose 3 of 4 GPUs, so no srun here.
# Each GPU k works through ARMS in order on its share of WINS. Clips that already exist are skipped.
ARMS="__ARMS__"; WINS="__WINS__"
R=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler; A=$R/ARRWM; L=$R/_logs/ours30s
export PATH=$R/miniforge3/envs/arrwm/bin:$PATH
export PYTHONPATH=$A:$A/action-forcing PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=$R/hf_cache TORCH_HOME=$R/torch_home
cd $A
[ -z "$WINS" ] && WINS=$(python3 -c "import json; print(','.join(x['uid'] for x in json.load(open('experiments/e1/scene_shortlist/e1_32_windows.json'))))")
echo "[cmd] batch shell CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; nvidia-smi -L
NG=$(nvidia-smi -L | grep -c "^GPU")
echo "[cmd] $NG GPUs visible; arms $ARMS; windows $WINS $(date)"
for k in $(seq 0 $((NG-1))); do
  W=$(python3 -c "w='$WINS'.split(','); n=-(-len(w)//$NG); print(','.join(w[$k*n:($k+1)*n]))")
  [ -z "$W" ] && continue
  (
    for ARM in ${ARMS//,/ }; do
      echo "[gpu $k] $ARM windows $W start $(date)"
      CUDA_VISIBLE_DEVICES=$k python interactive/external_models/ours_fleet_30s.py \
        --ckpt $R/ckpts/$ARM/phase1_step0001000.pt.lean.pt --tag $ARM --out $R/_logs/ours30s/out/$ARM \
        --bundle experiments/e1/seed_bundle_e1 --wan_model_path $R/frodobots/Wan2.1-T2V-1.3B/ --windows $W \
        >> $L/${ARM}_g$k.log 2>&1
      echo "[gpu $k] $ARM exit $? $(date)"
    done
  ) &
done
wait
echo "[cmd] all done $(date)"
