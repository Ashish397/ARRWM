#!/bin/bash
#SBATCH --job-name=noop-tracks
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Stationarity + animation from learned point tracks, RANSAC pipeline unchanged.
#   TRACKER=bootstapir (default) | cotracker
# Rank 0 also renders the demo videos so the tracker can be eyeballed.
# BootsTAPIR: third_party/tapnet + checkpoints/bootstapir_checkpoint_v2.pt
# CoTracker3: torch.hub cache (scaled_offline.pth). Neither needs network.
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache TMPDIR=/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TRACKER="${TRACKER:-bootstapir}"
echo "noop-tracks: TRACKER=$TRACKER"
srun --ntasks=4 --ntasks-per-node=4 --gpus-per-node=4 bash -c '
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  export TB2_SHARD=$SLURM_PROCID TB2_NSHARD=4
  export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
  L=logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_shard${SLURM_PROCID}.log
  # rank 0 first renders the demo videos (a few minutes on a GPU)
  if [ "$SLURM_PROCID" = "0" ]; then
    for A in "pca8 4node real --scene 30" "pca8 real --scene 26" \
             "astra real --scene 5" "worldplay matrixgame real --scene 12"; do
      python analysis/testbench_v2/noop_excursion_video.py --systems $A \
        --tracker '"$TRACKER"' --device cuda --grid-step 16 >> $L 2>&1
    done
  fi
  python analysis/testbench_v2/noop_align_eval.py --tracker '"$TRACKER"' \
      --device cuda --out noop_align_${TRACKER}.csv >> $L 2>&1
'
echo "noop-tracks done $(date)"
