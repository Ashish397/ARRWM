#!/bin/bash
#SBATCH --job-name=base-recon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
set -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM/third_party
for repo in cvlab-kaist/WorldCam stdstu12/YUME OpenDriveLab/Vista SkyworkAI/Matrix-Game Tencent-Hunyuan/HY-WorldPlay; do
  d=$(basename $repo)
  [ -d "$d" ] || git clone --depth 1 https://github.com/$repo.git "$d" 2>&1 | tail -1
  echo "########## $repo ##########"
  ls "$d" | head -18
  echo "--- inference/action/weights refs from README ---"
  grep -iE -m 20 "inference|weights|checkpoint|huggingface|action|keyboard|camera|pose|trajectory|command|fps|resolution|frames" "$d"/README.md 2>/dev/null | head -20
  echo
done
echo "BASE-RECON done"
