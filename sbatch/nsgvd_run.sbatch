#!/bin/bash
#SBATCH --job-name=nsgvd-run
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM/third_party/NSG-VD
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache TMPDIR=/tmp
pip install -q torchsde torchdiffeq moviepy loguru hydra-core tabulate seaborn ninja scikit-learn 2>&1 | tail -1 || true
# guided-diffusion checkpoint (2.1GB) at their expected relative path
if [ ! -f pretrained/guided_diffusion/256x256_diffusion_uncond.pt ]; then
  mkdir -p pretrained/guided_diffusion
  wget -q https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt \
    -O pretrained/guided_diffusion/256x256_diffusion_uncond.pt
  echo "diffusion ckpt downloaded: $(du -h pretrained/guided_diffusion/256x256_diffusion_uncond.pt | cut -f1)"
fi
python /scratch/u6ex/as1748.u6ex/ARRWM/utils/nsgvd_runner.py
echo "NSGVD-RUN done $(date)"
