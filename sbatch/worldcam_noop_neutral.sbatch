#!/bin/bash
#SBATCH --job-name=wcam-noopnp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# WorldCam null-action: identity camera trajectory (WC_STEPT=0 WC_STEPR=0 WC_PREFIX=0 WC_NEG='' WC_CAP="$NEUTRAL"),
# 32 A-scene starts, one shard (1 dir x 32 windows over 4 GPUs).
set -e
ARR=/scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache TMPDIR=/tmp
cd $ARR
NEUTRAL="a first-person view from a small delivery robot on a sidewalk in a residential street, houses and parked cars, daytime, clear weather."
ALLW=$(seq -s: 0 31)
PYTHONPATH=$ARR SW_WINDOWS="$ALLW" python utils/decode_seed_windows.py
cd $ARR/third_party/WorldCam
declare -A W=( [0]="0:1:2:3:4:5:6:7" [1]="8:9:10:11:12:13:14:15" [2]="16:17:18:19:20:21:22:23" [3]="24:25:26:27:28:29:30:31" )
PIDS=()
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g WC_WINDOWS="${W[$g]}" WC_DIRS="F" \
    WC_STEPT=0 WC_STEPR=0 WC_PREFIX=0 WC_NEG='' WC_CAP="$NEUTRAL" WC_ARSTEPS=52 WC_FPS=30 \
    WC_OUT=$ARR/logs/eval_final/A_worldcam_noop_neutral \
    PYTHONPATH= python worldcam_runner.py > $ARR/logs/wcamnoopnp_g${g}.log 2>&1 &
  PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait $p; done
echo "videos: $(ls $ARR/logs/eval_final/A_worldcam_noop_neutral/*.mp4 2>/dev/null | wc -l)/32"
echo "WCAM-NOOP done $(date)"
