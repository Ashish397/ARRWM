#!/bin/bash
# Run BOTH edge2 suites, one per GPU, inside ONE srun step (so the two
# python processes get distinct devices instead of both landing on GPU 0).
#   ALLOC=<jobid> bash utils/.probe_ar_edge2_both.sh
set -euo pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
ALLOC=${ALLOC:?}
L=/scratch/u6ex/as1748.u6ex/ARRWM/logs

read -r -d '' PAYLOAD <<'EOF' || true
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
cd /scratch/u6ex/as1748.u6ex/ARRWM
export PYTHONPATH=$PWD:$PWD/action-forcing
export TMPDIR=/tmp
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache
export HF_HUB_CACHE=$HF_HOME HUGGINGFACE_HUB_CACHE=$HF_HOME
export ARRWM_ACTION_ENCODER=pca_raw
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=/scratch/u6ex/as1748.u6ex/ARRWM/logs
RUNGS=1000,625,357.142857,208.333333
STU=/scratch/u6ex/as1748.u6ex/ARRWM/logs/ode14e_pilot/run3_flip2_roll10k/action_ode_step0000200.pt

CUDA_VISIBLE_DEVICES=0 python utils/.probe_ar_edge2.py \
  --suite main --student_ckpt "$STU" --rungs "$RUNGS" \
  --windows 3 --chunks 3 --seed_chunks 3 --eval_steps 20 \
  --probe_t 1000,980,800,625,500,357,208,100,50 \
  --out_json $L/probe_ar_edge2_main.json > $L/probe_edge2_main.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=1 python utils/.probe_ar_edge2.py \
  --suite control --student_ckpt "$STU" --rungs "$RUNGS" \
  --windows 3 --chunks 3 --seed_chunks 3 --degrade 0.05 \
  --probe_t 1000,625,357,208 \
  --out_json $L/probe_ar_edge2_control.json > $L/probe_edge2_control.log 2>&1 &
P2=$!
wait $P1; R1=$?
wait $P2; R2=$?
echo "=== main rc=$R1  control rc=$R2 ==="
EOF

srun --overlap --jobid="$ALLOC" -N1 -n1 --gpus=2 --cpus-per-task=32 \
  bash -c "$PAYLOAD"
