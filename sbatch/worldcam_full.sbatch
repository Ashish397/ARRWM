#!/bin/bash
#SBATCH --job-name=wcam-full
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=14:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# WorldCam production: 32 windows x 8 dirs, 8s videos (AR 52 -> 241f @30fps).
# WC_SHARD=0 -> windows 0-15, WC_SHARD=1 -> windows 16-31 (submit twice).
set -e
ARR=/scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache TMPDIR=/tmp
SHARD=${WC_SHARD:-0}

cd $ARR
ALLW=$(seq -s: 0 31)
PYTHONPATH=$ARR SW_WINDOWS="$ALLW" python utils/decode_seed_windows.py

cd $ARR/third_party/WorldCam
declare -A W
if [ "$SHARD" = "0" ]; then
  W[0]="0:1:2:3"; W[1]="4:5:6:7"; W[2]="8:9:10:11"; W[3]="12:13:14:15"
else
  W[0]="16:17:18:19"; W[1]="20:21:22:23"; W[2]="24:25:26:27"; W[3]="28:29:30:31"
fi
PIDS=()
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g WC_WINDOWS="${W[$g]}" WC_DIRS="F:FR:R:BR:B:BL:L:FL" \
    WC_STEPT=0.02 WC_STEPR=1.5 WC_ARSTEPS=52 WC_FPS=30 \
    WC_OUT=$ARR/logs/eval_final/A_worldcam \
    PYTHONPATH= python worldcam_runner.py > $ARR/logs/wcamfull_s${SHARD}_g${g}.log 2>&1 &
  PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait $p; done
echo "WCAM-FULL shard $SHARD done $(date)"
