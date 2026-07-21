#!/bin/bash
#SBATCH --job-name=det-recon2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
set -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM/third_party
for repo in ZSHsh98/NSG-VD SparkleXFantasy/MM-Det SparkleXFantasy/MM-Det-Plus Zig-HS/D3; do
  d=$(basename $repo)
  [ -d "$d" ] || git clone --depth 1 https://github.com/$repo.git "$d" 2>&1 | tail -1
  echo "########## $repo ##########"
  find "$d" -maxdepth 2 -name "*.pth" -o -maxdepth 2 -name "*.ckpt" -o -maxdepth 2 -name "*.pt" 2>/dev/null | head -5
  ls "$d" | head -15
  echo "----- README head -----"
  head -100 "$d"/README.md 2>/dev/null
  echo "----- weight/infer refs -----"
  grep -riE "drive.google|huggingface|\.pth|\.ckpt|demo|inference|test\.py" "$d"/README.md 2>/dev/null | head -10
  echo
done
echo "DET-RECON2 done"
