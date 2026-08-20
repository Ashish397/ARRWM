#!/bin/bash
# Render the AR comparison videos (+ optionally re-run the control suite on
# the second GPU) inside ONE srun step.
#   ALLOC=<jobid> bash utils/.probe_ar_video.sh
set -euo pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
ALLOC=${ALLOC:?}
CTRL2=${CTRL2:-1}          # 1 = also run the strengthened control suite
DEGRADE=${DEGRADE:-0.2}
CHUNKS=${CHUNKS:-6}
WINDOWS=${WINDOWS:-2}

read -r -d '' PAYLOAD <<EOF || true
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
cd /scratch/u6ex/as1748.u6ex/ARRWM
export PYTHONPATH=\$PWD:\$PWD/action-forcing
export TMPDIR=/tmp
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache
export HF_HUB_CACHE=\$HF_HOME HUGGINGFACE_HUB_CACHE=\$HF_HOME
export ARRWM_ACTION_ENCODER=pca_raw
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=/scratch/u6ex/as1748.u6ex/ARRWM/logs
RUNGS=1000,625,357.142857,208.333333
STU=/scratch/u6ex/as1748.u6ex/ARRWM/logs/ode14e_pilot/run3_flip2_roll10k/action_ode_step0000200.pt

CUDA_VISIBLE_DEVICES=0 python utils/.probe_ar_video.py \
  --student_ckpt "\$STU" --rungs "\$RUNGS" \
  --windows $WINDOWS --chunks $CHUNKS --seed_chunks 3 \
  --render_t 1000,980,625,208.333333 --extra_gen 1 --eval_steps 20 \
  --fps 16 > \$L/probe_ar_video.log 2>&1 &
P1=\$!
if [ "$CTRL2" = "1" ]; then
CUDA_VISIBLE_DEVICES=1 python utils/.probe_ar_edge2.py \
  --suite control --student_ckpt "\$STU" --rungs "\$RUNGS" \
  --windows 3 --chunks 3 --seed_chunks 3 --degrade $DEGRADE \
  --probe_t 1000,980,625,357,208,100 \
  --out_json \$L/probe_ar_edge2_control2.json \
  > \$L/probe_edge2_control2.log 2>&1 &
P2=\$!
fi
wait \$P1; R1=\$?
if [ "$CTRL2" = "1" ]; then wait \$P2; R2=\$?; else R2=na; fi
echo "=== video rc=\$R1  control2 rc=\$R2 ==="
EOF

srun --overlap --jobid="$ALLOC" -N1 -n1 --gpus=2 --cpus-per-task=32 \
  bash -c "$PAYLOAD"
