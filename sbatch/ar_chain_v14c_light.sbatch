#!/bin/bash
#SBATCH --job-name=arc-v14c-light
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:45:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Light causal-chain eval: a short chain (NCV videos, default 3) on CITIES
# (default just madrid) so one eval ~= the wall-clock of 100 training steps,
# keeping it in sync with the critic8-backward run.
#   Usage: sbatch sbatch/ar_chain_v14c_light.sbatch <CKPT_PATH> <STEP_TAG>
#   Env:   NCV=3  CITIES="madrid"   (override for more videos/cities; cities run
#          in parallel across the 4 GPUs at no extra wall-clock.)
set -u -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR HF_HUB_CACHE=$CACHE_DIR HUGGINGFACE_HUB_CACHE=$CACHE_DIR TRANSFORMERS_CACHE=$CACHE_DIR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8

CKARG="${1:?need checkpoint path or 'latest' as arg1}"
if [ "$CKARG" = "latest" ]; then
  CKPT=$(ls -1 logs/v14c_critic8_backward/causal_lora_step*.pt 2>/dev/null | sort -V | tail -1)
  : "${CKPT:?no checkpoint found in logs/v14c_critic8_backward}"
  STEP=$(basename "$CKPT" | grep -oE '[0-9]+'); STEP=$((10#$STEP))
  echo "resolved 'latest' -> $CKPT (step $STEP)"
else
  CKPT="$CKARG"; STEP="${2:-x}"
fi
NCV="${NCV:-7}"   # >=7 required for ROLLOUT_METRICS
CITIES="${CITIES:-madrid stockholm}"
ENC=/projects/u6ex/fbots/frodobots_encoded_weunz
CAP=/projects/u6ex/fbots/frodobots_captions/train
CFG=configs/causal_lora_diffusion_teacher_v14c.yaml
MAN=logs/v14c_critic8_backward/.ride_manifest.pt
A=12

# city -> zarr lookup
declare -A ZARR=( [madrid]=20240216101235.zarr [rome]=20240314084739.zarr [romeb]=20240314090046.zarr [stockholm]=20240319133048.zarr )

echo "v14c LIGHT ar_chain | ckpt=$CKPT | step=$STEP | NCV=$NCV | cities=$CITIES"
gpu=0
for city in $CITIES; do
  z="${ZARR[$city]}"
  d="eval/ar_chain/v14c_s${STEP}_${city}"
  CUDA_VISIBLE_DEVICES=$gpu WORLD_SIZE=1 LOCAL_RANK=0 \
    python utils/eval_causal_chain.py --assignment_index "$A" --config "$CFG" --manifest "$MAN" \
      --ckpt_override "$CKPT" --num_causal_videos "$NCV" \
      --rank_zarr "$z" --rank_offset 100 --rank_mode dataset --rank_tag "$city" \
      --encoded_root "$ENC" --caption_root "$CAP" --seed 42 --output_dir "$d" > "$d.log" 2>&1 \
    && echo "done v14c s${STEP} ${city}" || echo "FAIL v14c s${STEP} ${city} (see $d.log)" &
  gpu=$((gpu+1))
done
wait
echo "=== ROLLOUT_METRICS v14c s${STEP} ==="
grep -h "ROLLOUT_METRICS" eval/ar_chain/v14c_s${STEP}_*.log 2>/dev/null
echo "Done $(date)"
