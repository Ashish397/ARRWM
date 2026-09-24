#!/bin/bash
# Flagship recovery_base on u6qf, one holder: (1) 30 s fleet, 32 windows x 9 commands, one process per GPU from the
# batch shell; (2) eval suite on those 288 clips at 6/15/30 s (bit-exact decode, ORB descriptors saved) + windowed
# CoTracker. Status: $L/flagship_status.txt
R=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler; A=$R/ARRWM; L=$R/_logs/ours30s; E=$R/eval30s_check
ARM=recovery_base; CKPT=$R/ckpts/$ARM/phase1_step0001000.pt.lean.pt; S=$L/flagship_status.txt
export PATH=$R/miniforge3/envs/arrwm/bin:$PATH
say() { echo "$(date -u +%H:%M) $*" >> $S; }
say "start on $(hostname); $(nvidia-smi -L | grep -c ^GPU) GPUs"
cd $A; export PYTHONPATH=$A:$A/action-forcing PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=$R/hf_cache TORCH_HOME=$R/torch_home
WINS=$(python3 -c "import json; print(','.join(x['uid'] for x in json.load(open('experiments/e1/scene_shortlist/e1_32_windows.json'))))")
for k in 0 1 2 3; do
  W=$(python3 -c "w='$WINS'.split(','); print(','.join(w[$k*8:($k+1)*8]))")
  ( CUDA_VISIBLE_DEVICES=$k python interactive/external_models/ours_fleet_30s.py --ckpt $CKPT --tag $ARM --out $L/out/$ARM \
      --bundle experiments/e1/seed_bundle_e1 --wan_model_path $R/frodobots/Wan2.1-T2V-1.3B/ --windows $W >> $L/${ARM}_g$k.log 2>&1
    echo "gen g$k rc=$?" >> $S ) &
done; wait
n=$(ls $L/out/$ARM/*.mp4.json 2>/dev/null | wc -l); say "generation finished: $n clips"
[ "$n" -ge 288 ] || { say "ERROR generation incomplete"; exit 1; }
# (2) eval: three horizons in parallel (GPUs 0-2 shared, 288 clips each), windowed CoTracker on GPU 3
unset PYTHONPATH; export OMP_NUM_THREADS=6
for H in 6 15 30; do ( EVAL_DECODE_BITEXACT=1 bash $E/run_eval30s.sh ours_$ARM all $H $E/out_flagship/h$H > $E/out_flagship_h$H.runlog 2>&1; say "eval h$H: $(grep -c 'rc=0' $E/out_flagship/h$H/logs/timings.txt) of 5 instruments rc=0" ) & done
( cd $E/ARR/grids/eval; mkdir -p $E/out_flagship/cot30w; CUDA_VISIBLE_DEVICES=3 ARR=$E/ARR OURS30S_DIR=$L/out TORCH_HOME=$R/torch_home FLEET=30s EVAL_DECODE_BITEXACT=1 FLEET30S_MODELS=ours_$ARM \
    EVAL_OUT_DIR=$E/out_flagship/cot30w python fleet30s_cotracker.py > $E/out_flagship/cot30w/run.log 2>&1; say "cot30w rc=$?" ) &
wait; say "ALLDONE"
