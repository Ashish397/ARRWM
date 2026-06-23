#!/bin/bash
#SBATCH --job-name=ar-chain-dims
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Make probe2dim + critic8 AR-chain rollouts for the SAME 4 city clips as the
# existing eval/ar_chain set (v14/f3/tok), so they slot in alongside: same
# zarrs, offset 100, seed 42, rank_tag=<city>, default decode. 1 node, 4 GPUs,
# 8 renders (2 models x 4 cities), 4 at a time. Outputs -> eval/ar_chain/<model>_<city>.
set -u -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR HF_HUB_CACHE=$CACHE_DIR HUGGINGFACE_HUB_CACHE=$CACHE_DIR TRANSFORMERS_CACHE=$CACHE_DIR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
ENC=/projects/u6ex/fbots/frodobots_encoded_weu
CAP=/projects/u6ex/fbots/frodobots_captions/train

# cities: name -> zarr (all offset 100, matching the existing ar_chain set)
CITY=(madrid rome romeb stockholm)
ZARR=(20240216101235.zarr 20240314084739.zarr 20240314090046.zarr 20240319133048.zarr)

# models: label -> assignment, config, manifest
MLAB=(probe2dim critic8)
MA=(11 10)
MCFG=(configs/causal_lora_diffusion_teacher_v14_loo_probe2dim.yaml
      configs/causal_lora_diffusion_teacher_v14_loo_critic8_noprobe.yaml)
MMAN=(logs/v14_loo_probe2dim/.ride_manifest.pt
      logs/v14_loo_critic8_noprobe/.ride_manifest.pt)

render_one() {  # gpu label assign config manifest zarr city
  local gpu=$1 lab=$2 a=$3 cfg=$4 man=$5 z=$6 city=$7
  local d="eval/ar_chain/${lab}_${city}"
  CUDA_VISIBLE_DEVICES=$gpu WORLD_SIZE=1 LOCAL_RANK=0 \
    python utils/eval_causal_chain.py --assignment_index "$a" --config "$cfg" --manifest "$man" \
      --rank_zarr "$z" --rank_offset 100 --rank_mode dataset --rank_tag "$city" \
      --encoded_root "$ENC" --caption_root "$CAP" --seed 42 --output_dir "$d" > "$d.log" 2>&1 \
    && echo "done $lab $city" || echo "FAIL $lab $city (see $d.log)"
}

# build 8 tasks, run 4 at a time (one per GPU)
i=0
for mi in 0 1; do
  for ci in 0 1 2 3; do
    gpu=$((i % 4))
    render_one "$gpu" "${MLAB[$mi]}" "${MA[$mi]}" "${MCFG[$mi]}" "${MMAN[$mi]}" "${ZARR[$ci]}" "${CITY[$ci]}" &
    i=$((i + 1))
    (( i % 4 == 0 )) && wait
  done
done
wait
echo "=== rollout metrics (command vs rendered-judge vs critic) for the new ablations ==="
grep -h "ROLLOUT_METRICS" eval/ar_chain/probe2dim_*.log eval/ar_chain/critic8_*.log 2>/dev/null
echo "Done $(date)"
