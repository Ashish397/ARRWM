#!/bin/bash
# Driver for utils/.probe_ar_edge2.py  (the falsification suite).
#   ALLOC=<jobid> SUITE=main bash utils/.probe_ar_edge2.sh
set -euo pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM

SUITE=${SUITE:-main}
STU_CKPT=${STU_CKPT:-/scratch/u6ex/as1748.u6ex/ARRWM/logs/ode14e_pilot/run3_flip2_roll10k/action_ode_step0000200.pt}
TEA_CKPT=${TEA_CKPT:-/scratch/u6ex/as1748.u6ex/ARRWM/logs/v14e_pca8_raw/causal_lora_step0005000.pt}
CFG=${CFG:-configs/action_ode_distill_F.yaml}
CHUNKS=${CHUNKS:-3}
SEEDCH=${SEEDCH:-3}
WINDOWS=${WINDOWS:-3}
PROBE_T=${PROBE_T:-1000,980,800,625,500,357,208,100,50}
EVAL_STEPS=${EVAL_STEPS:-20}
DEGRADE=${DEGRADE:-0.05}
# LOAD-BEARING: the DMD arms override the rung ladder; the ODE config's own
# ladder is [1000,625,312.5,178.57] and is NOT what the arms roll with.
RUNGS=${RUNGS:-1000,625,357.142857,208.333333}
OUT=${OUT:-/scratch/u6ex/as1748.u6ex/ARRWM/logs/probe_ar_edge2_${SUITE}.json}

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
python utils/.probe_ar_edge2.py \
  --suite $SUITE \
  --student_ckpt "$STU_CKPT" \
  --teacher_ckpt "$TEA_CKPT" \
  --config "$CFG" \
  --chunks $CHUNKS --seed_chunks $SEEDCH --windows $WINDOWS \
  --probe_t "$PROBE_T" --eval_steps $EVAL_STEPS --degrade $DEGRADE \
  --rungs "$RUNGS" \
  --out_json "$OUT"
EOF

ALLOC=${ALLOC:-}
if [ "$ALLOC" = "none" ] || [ -z "$ALLOC" ]; then
  bash -c "$PAYLOAD"
else
  srun --overlap --jobid="$ALLOC" -N1 -n1 --gpus=1 --cpus-per-task=16 \
    bash -c "$PAYLOAD"
fi
