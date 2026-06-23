#!/bin/bash
#SBATCH --job-name=arc-v14c-cont
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=2
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# ONE long-lived job that HOLDS a node for the whole training run and runs the
# causal-chain eval in-process at every +100 step milestone. Queues ONCE (no
# per-milestone requeue), so it stays in sync with critic8-backward.
# Progress -> eval/ar_chain/v14c_progress.log ; videos -> eval/ar_chain/v14c_s<step>_<city>/
set -u
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR HF_HUB_CACHE=$CACHE_DIR HUGGINGFACE_HUB_CACHE=$CACHE_DIR TRANSFORMERS_CACHE=$CACHE_DIR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8

TRAIN_JOB=5295255
LOGDIR=logs/v14c_critic8_backward
TRAINLOG=logs/v14c-critic8-backward-64gpu_5295255.err
PROG=eval/ar_chain/v14c_progress.log
MARKDIR=eval/ar_chain/.v14c_markers
mkdir -p "$MARKDIR" eval/ar_chain
NCV=7   # must be >=7 (NUM_ACTION_CHUNKS) or the final causal rollout + ROLLOUT_METRICS are skipped
CFG=configs/causal_lora_diffusion_teacher_v14c.yaml
MAN=$LOGDIR/.ride_manifest.pt
ENC=/projects/u6ex/fbots/frodobots_encoded_weunz
CAP=/projects/u6ex/fbots/frodobots_captions/train
A=12
# cities to render (one per GPU, parallel = same wall-clock). stockholm is the
# v14b stationary-command->still-moving-forward diagnostic clip.
CITIES="${CITIES:-madrid stockholm}"
declare -A ZARR=( [madrid]=20240216101235.zarr [rome]=20240314084739.zarr [romeb]=20240314090046.zarr [stockholm]=20240319133048.zarr )

ckpt(){ printf "$LOGDIR/causal_lora_step%07d.pt" "$1"; }
train_alive(){ squeue -h -j "$TRAIN_JOB" -o "%T" 2>/dev/null | grep -qE "PENDING|RUNNING|CONFIGURING|COMPLETING"; }
cur_step(){ grep -E "step [0-9]+ \| loss" "$TRAINLOG" 2>/dev/null | tail -1 | grep -oE "step [0-9]+" | grep -oE "[0-9]+"; }
say(){ echo "[cont $(date +%H:%M:%S)] $*" | tee -a "$PROG"; }

cs=$(cur_step); cs=${cs:-0}
M=$(( (cs/100) * 100 )); [ "$M" -lt 100 ] && M=100   # earliest available 100-milestone (incl. current)
say "CONTINUOUS eval job on $(hostname) | current step ~$cs | first milestone $M | NCV=$NCV cities=$CITIES"

while true; do
  # wait for milestone M's checkpoint to be fully written, or training to end
  while true; do
    if [ -f "$(ckpt $M)" ] && { [ -f "$(ckpt $((M+50)))" ] || ! train_alive; }; then break; fi
    if ! train_alive && [ ! -f "$(ckpt $M)" ]; then say "training ended; no ckpt for $M. exiting."; exit 0; fi
    sleep 45
  done
  if [ -f "$MARKDIR/s${M}.done" ]; then M=$((M+100)); continue; fi

  CK=$(ckpt $M); t0=$(date +%s)
  say "step $M: running causal-chain eval (NCV=$NCV, cities=$CITIES)"
  gpu=0
  for city in $CITIES; do
    z="${ZARR[$city]}"; d="eval/ar_chain/v14c_s${M}_${city}"
    CUDA_VISIBLE_DEVICES=$gpu WORLD_SIZE=1 LOCAL_RANK=0 \
      python utils/eval_causal_chain.py --assignment_index "$A" --config "$CFG" --manifest "$MAN" \
        --ckpt_override "$CK" --num_causal_videos "$NCV" \
        --rank_zarr "$z" --rank_offset 100 --rank_mode dataset --rank_tag "$city" \
        --encoded_root "$ENC" --caption_root "$CAP" --seed 42 --output_dir "$d" > "$d.log" 2>&1 &
    gpu=$((gpu+1))
  done
  wait
  touch "$MARKDIR/s${M}.done"
  dur=$(( $(date +%s) - t0 ))
  say "step $M DONE in ${dur}s | videos: eval/ar_chain/v14c_s${M}_*/"
  grep -h "ROLLOUT_METRICS" eval/ar_chain/v14c_s${M}_*.log 2>/dev/null | sed 's/^/[cont][metrics] /' | tee -a "$PROG" >/dev/null
  M=$(( M + 100 ))
done
