#!/bin/bash
#SBATCH --job-name=iqa-pca
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate && conda activate arrwm
python utils/no_ref_iqa.py \
  "VAE-mode=eval/ar_chain/v14d_stream_b1_w7_madrid/stream_b1_w7_madrid_cont_decode.mp4" \
  "PCA-mode=eval/ar_chain/v14d_stream_b1_w7_PCAMODE_madrid/stream_b1_w7_madrid_cont_decode.mp4"
