#!/bin/bash
#SBATCH --job-name=det-recon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
set -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM/third_party
for repo in chenhaoxing/DeMamba wuwuwuyue/DeCoF Rohit-Kundu/UNITE justin-crchang/3DCNNDetection.github.io; do
  d=$(basename $repo)
  [ -d "$d" ] || git clone --depth 1 https://github.com/$repo.git "$d" 2>&1 | tail -1
  echo "########## $repo ##########"
  ls "$d" 2>/dev/null | head -20
  echo "----- README (weights/inference sections) -----"
  cat "$d"/README.md 2>/dev/null | head -120
  echo "----- weight/ckpt links -----"
  grep -riE "drive.google|huggingface.co|modelscope|\.pth|\.ckpt|\.pt\b|weight" "$d"/README.md 2>/dev/null | head -12
  echo
done
echo "DET-RECON done"
