#!/bin/bash
# GO/NO-GO probe for the DMD AR-teacher head.
#
#   Does the v14e teacher, served AR through a KV cache on the STUDENT's
#   OWN rolled context, actually beat the student on MAE-to-GT?
#
# This is the decision the whole dmd_ar_head idea rests on. It runs
# OUTSIDE the DMD machinery (no critic, no eq.-8 normalizer, no MAE gate,
# no 42f window) so the answer is not entangled with any of them. See the
# module docstring in utils/.probe_ar_teacher_edge.py for the exact
# serving contract (it mirrors sbatch/train_dmd10k_stat.sbatch).
#
# USAGE — inside an existing allocation (a `hold-2n-6h` holder):
#   ALLOC=<jobid> bash utils/.probe_ar_teacher_edge.sh
#
# USAGE — on a node you are already sitting on (no srun):
#   ALLOC=none bash utils/.probe_ar_teacher_edge.sh
#
# Knobs (env):
#   STU_CKPT   student checkpoint (default = the msear/msedual arms' ODE init)
#   TEA_CKPT   teacher LoRA       (default = v14e_pca8_raw step 5000)
#   CFG        ODE config         (default = configs/action_ode_distill_F.yaml)
#   CHUNKS     student chunks to roll (default 3 = the AR head's band)
#   SEEDCH     real seed chunks   (default 3 = dmd_context_clean_frames 9 / npb)
#   WINDOWS    real contexts to probe (default 4)
#   PROBE_T    rungs for the single-shot comparison (default = the arm's ladder)
#   NOISE_SRC  student|gt  what gets noised to PROBE_T (default student —
#              exactly what the DMD AR head feeds its scorers)
#   EVAL_STEPS teacher full-sampler chain length (default 20; 0 = skip)
#   OUT        json output path
#
# NOTE: the ride manifest is ~21 GB; the first load takes minutes.
# NOTE: needs 1 GPU and ~2x1.3B bf16 of weights (two ODERegression builds).
set -euo pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM

STU_CKPT=${STU_CKPT:-/scratch/u6ex/as1748.u6ex/ARRWM/logs/ode14e_pilot/run3_flip2_roll10k/action_ode_step0000200.pt}
TEA_CKPT=${TEA_CKPT:-/scratch/u6ex/as1748.u6ex/ARRWM/logs/v14e_pca8_raw/causal_lora_step0005000.pt}
CFG=${CFG:-configs/action_ode_distill_F.yaml}
CHUNKS=${CHUNKS:-3}
SEEDCH=${SEEDCH:-3}
WINDOWS=${WINDOWS:-4}
PROBE_T=${PROBE_T:-1000,625,357,208}
NOISE_SRC=${NOISE_SRC:-student}
EVAL_STEPS=${EVAL_STEPS:-20}
OUT=${OUT:-/scratch/u6ex/as1748.u6ex/ARRWM/logs/probe_ar_teacher_edge.json}

for f in "$STU_CKPT" "$TEA_CKPT" "$CFG"; do
  [ -e "$f" ] || { echo "[probe] MISSING: $f" >&2; exit 2; }
done

read -r -d '' PAYLOAD <<EOF || true
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
cd /scratch/u6ex/as1748.u6ex/ARRWM
export PYTHONPATH=\$PWD:\$PWD/action-forcing
export TMPDIR=/tmp
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache
export HF_HUB_CACHE=\$HF_HOME HUGGINGFACE_HUB_CACHE=\$HF_HOME
# LOAD-BEARING: the 14e lineage is pca_raw ([0,1] = throttle,steer).
# Without it the action stream is silently the wrong encoding.
export ARRWM_ACTION_ENCODER=pca_raw
python utils/.probe_ar_teacher_edge.py \
  --student_ckpt "$STU_CKPT" \
  --teacher_ckpt "$TEA_CKPT" \
  --config "$CFG" \
  --chunks $CHUNKS \
  --seed_chunks $SEEDCH \
  --windows $WINDOWS \
  --probe_t "$PROBE_T" \
  --noise_source "$NOISE_SRC" \
  --eval_steps $EVAL_STEPS \
  --out_json "$OUT"
EOF

ALLOC=${ALLOC:-}
if [ -z "$ALLOC" ]; then
  echo "[probe] set ALLOC=<holder jobid>  (or ALLOC=none to run here)" >&2
  echo "[probe] RUNNING holders:" >&2
  squeue -u "$USER" -h -o '%i %j %T' | awk '$3=="RUNNING"' >&2 || true
  exit 2
fi

if [ "$ALLOC" = "none" ]; then
  bash -c "$PAYLOAD"
else
  # --overlap: the holder's own step keeps its resources; we ride along.
  srun --overlap --jobid="$ALLOC" -N1 -n1 --gpus=1 --cpus-per-task=16 \
    bash -c "$PAYLOAD"
fi
