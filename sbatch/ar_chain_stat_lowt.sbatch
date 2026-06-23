#!/bin/bash
#SBATCH --job-name=ar-chain-statlowt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# AR causal-chain rollout for the stat_wave_dmd3 LOW-t-skew DMD student
# (step-200 ckpt). Same 4 city clips / offset 100 / seed 42 as eval/ar_chain
# (v14/v14b/critic8/...), so it slots in alongside them. 1 node, 4 GPUs = one
# city per GPU. Uses eval_causal_AR.py (student overlay onto the v14 teacher
# base) at the student's trained 4-step schedule. Outputs -> eval/ar_chain/statlowt_<city>.
set -u -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR HF_HUB_CACHE=$CACHE_DIR HUGGINGFACE_HUB_CACHE=$CACHE_DIR TRANSFORMERS_CACHE=$CACHE_DIR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
ENC=/projects/u6ex/fbots/frodobots_encoded_weunz
CAP=/projects/u6ex/fbots/frodobots_captions/train
CFG=configs/ar_eval_dmd_student.yaml
MAN=logs/v14_balanced_weunz/.ride_manifest.pt
MOTION=/projects/u6ex/fbots/frodobots_motion
CKPT=logs/action_forcing_phase3_dmd_one_step/stat_wave_dmd3_lowt_j5290667/phase1_step0000200.pt
LABEL=statlowt

CITY=(madrid rome romeb stockholm)
ZARR=(20240216101235.zarr 20240314084739.zarr 20240314090046.zarr 20240319133048.zarr)

for ci in 0 1 2 3; do
  d="eval/ar_chain/${LABEL}_${CITY[$ci]}"
  CUDA_VISIBLE_DEVICES=$ci WORLD_SIZE=1 LOCAL_RANK=0 \
    python utils/eval_causal_AR.py --student_ckpt "$CKPT" --config "$CFG" --manifest "$MAN" \
      --rank_zarr "${ZARR[$ci]}" --rank_offset 100 --rank_mode dataset --rank_tag "${CITY[$ci]}" \
      --encoded_root "$ENC" --caption_root "$CAP" --motion_root "$MOTION" --seed 42 --denoising_steps 4 --mode chain \
      --label "$LABEL" --output_dir "$d" > "$d.log" 2>&1 \
    && echo "done ${LABEL} ${CITY[$ci]}" || echo "FAIL ${LABEL} ${CITY[$ci]} (see $d.log)" &
done
wait
echo "=== rollout metrics for ${LABEL} ==="
grep -h "ROLLOUT_METRICS" eval/ar_chain/${LABEL}_*.log 2>/dev/null
echo "Done $(date)"
