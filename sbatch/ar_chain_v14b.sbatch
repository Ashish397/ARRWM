#!/bin/bash
#SBATCH --job-name=ar-chain-v14b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# v14b AR-chain rollouts on the SAME 4 city clips as eval/ar_chain (offset 100,
# seed 42), so v14b slots in alongside v14/f3/tok/probe2dim/critic8. 1 node,
# 4 GPUs = one city per GPU. Outputs -> eval/ar_chain/v14b_<city>.
set -u -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR HF_HUB_CACHE=$CACHE_DIR HUGGINGFACE_HUB_CACHE=$CACHE_DIR TRANSFORMERS_CACHE=$CACHE_DIR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
ENC=/projects/u6ex/fbots/frodobots_encoded_weunz
CAP=/projects/u6ex/fbots/frodobots_captions/train
CFG=configs/causal_lora_diffusion_teacher_v14b.yaml
MAN=logs/v14b_critic8_curated/.ride_manifest.pt
A=12

CITY=(madrid rome romeb stockholm)
ZARR=(20240216101235.zarr 20240314084739.zarr 20240314090046.zarr 20240319133048.zarr)

for ci in 0 1 2 3; do
  d="eval/ar_chain/v14b_${CITY[$ci]}"
  CUDA_VISIBLE_DEVICES=$ci WORLD_SIZE=1 LOCAL_RANK=0 \
    python utils/eval_causal_chain.py --assignment_index "$A" --config "$CFG" --manifest "$MAN" \
      --rank_zarr "${ZARR[$ci]}" --rank_offset 100 --rank_mode dataset --rank_tag "${CITY[$ci]}" \
      --encoded_root "$ENC" --caption_root "$CAP" --seed 42 --output_dir "$d" > "$d.log" 2>&1 \
    && echo "done v14b ${CITY[$ci]}" || echo "FAIL v14b ${CITY[$ci]} (see $d.log)" &
done
wait
echo "=== rollout metrics (command vs rendered-judge vs critic) for v14b ==="
grep -h "ROLLOUT_METRICS" eval/ar_chain/v14b_*.log 2>/dev/null
echo "Done $(date)"
