#!/bin/bash
#SBATCH --job-name=ar-chain-v14c
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# v14c AR-chain rollouts on the SAME 4 city clips as eval/ar_chain (offset 100,
# seed 42). Reuses MODEL_ASSIGNMENTS index 12 (v14b == critic8 arch) and points
# at an arbitrary mid-training v14c checkpoint via --ckpt_override.
#   Usage: sbatch sbatch/ar_chain_v14c.sbatch <CKPT_PATH> <STEP_TAG>
# Outputs -> eval/ar_chain/v14c_s<STEP>_<city>.
set -u -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR HF_HUB_CACHE=$CACHE_DIR HUGGINGFACE_HUB_CACHE=$CACHE_DIR TRANSFORMERS_CACHE=$CACHE_DIR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8

CKPT="${1:?need checkpoint path as arg1}"
STEP="${2:-x}"
ENC=/projects/u6ex/fbots/frodobots_encoded_weunz
CAP=/projects/u6ex/fbots/frodobots_captions/train
CFG=configs/causal_lora_diffusion_teacher_v14c.yaml
MAN=logs/v14c_critic8_backward/.ride_manifest.pt
A=12

CITY=(madrid rome romeb stockholm)
ZARR=(20240216101235.zarr 20240314084739.zarr 20240314090046.zarr 20240319133048.zarr)

echo "v14c ar_chain | ckpt=$CKPT | step=$STEP"
for ci in 0 1 2 3; do
  d="eval/ar_chain/v14c_s${STEP}_${CITY[$ci]}"
  CUDA_VISIBLE_DEVICES=$ci WORLD_SIZE=1 LOCAL_RANK=0 \
    python utils/eval_causal_chain.py --assignment_index "$A" --config "$CFG" --manifest "$MAN" \
      --ckpt_override "$CKPT" \
      --rank_zarr "${ZARR[$ci]}" --rank_offset 100 --rank_mode dataset --rank_tag "${CITY[$ci]}" \
      --encoded_root "$ENC" --caption_root "$CAP" --seed 42 --output_dir "$d" > "$d.log" 2>&1 \
    && echo "done v14c s${STEP} ${CITY[$ci]}" || echo "FAIL v14c s${STEP} ${CITY[$ci]} (see $d.log)" &
done
wait
echo "=== rollout metrics (command vs rendered-judge vs critic) for v14c s${STEP} ==="
grep -h "ROLLOUT_METRICS" eval/ar_chain/v14c_s${STEP}_*.log 2>/dev/null
echo "Done $(date)"
