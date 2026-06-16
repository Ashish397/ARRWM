#!/bin/bash
#SBATCH --job-name=encode-rear-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Study 1 (AOO) rear/reverse-camera encode SMOKE: 30 weu rides, uid_s_1001.
# Reuses pre_encode_direct.py VERBATIM (identical Wan-VAE path + params as the
# forward weu zarrs) so rear latents are drop-in 1:1 alignable with forward.
# Study 3: the encode path already uses the VAE causal feat_cache within each
# block (_fast_encode) + a 1-frame block overlap (stream_blocks) -> no extra
# caching change is correct here; matching forward exactly is what we want.
set -u -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

CSV=data/rear_encode/weu_rear_smoke.csv
OUT=/projects/u6ex/fbots/frodobots_encoded_weu_rear

echo "rear-encode smoke start $(date)  csv=$CSV  out=$OUT"
python utils/pre_encode_direct.py \
  --rides_csv "$CSV" \
  --output_root "$OUT" \
  --vae_path /scratch/u6ex/as1748.u6ex/frodobots/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth \
  --device cuda \
  --model_dtype float16 \
  --latent_dtype float16 \
  --scale 832x480 \
  --encode_stride 600 \
  --zarr_latent_chunk 32
echo "rear-encode smoke done $(date)"
