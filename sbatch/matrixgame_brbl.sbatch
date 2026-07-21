#!/bin/bash
#SBATCH --job-name=mg2-brbl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Matrix-Game 2.0 (distilled) production: 32 windows x 8 dirs, 8s (51 latents = 201f @25fps)
set -e
ARR=/scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache TMPDIR=/tmp

cd $ARR
ALLW=$(seq -s: 0 31)
PYTHONPATH=$ARR SW_WINDOWS="$ALLW" python utils/decode_seed_windows.py

cd $ARR/third_party/Matrix-Game/Matrix-Game-2
declare -A W
W[0]="0:1:2:3:4:5:6:7"; W[1]="8:9:10:11:12:13:14:15"
W[2]="16:17:18:19:20:21:22:23"; W[3]="24:25:26:27:28:29:30:31"
PIDS=()
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g MG_WINDOWS="${W[$g]}" MG_DIRS="BR:BL" \
    MG_NUMLAT=51 MG_OUT=$ARR/logs/eval_final/A_matrixgame \
    PYTHONPATH= python matrixgame_runner.py > $ARR/logs/mg2full_g${g}.log 2>&1 &
  PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait $p; done
echo "MG2-FULL done $(date)"
