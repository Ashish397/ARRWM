#!/bin/bash
#SBATCH --job-name=rear-align-viz
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# Build wall-clock alignment maps for the rear smoke rides, then render a few
# side-by-side FORWARD|REVERSE overlay videos (z2/z7 forward, flipped reverse)
# so we can VISUALLY confirm the two views are aligned. Runs on a compute node
# (no login-node data access). Submit with --dependency=afterok:<encode_jobid>.
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
export OMP_NUM_THREADS=4

echo "== building alignment maps (30 smoke rides) =="
python utils/build_rear_alignment.py \
  --forward_root /projects/u6ex/fbots/frodobots_encoded_weu \
  --rear_root /projects/u6ex/fbots/frodobots_encoded_weu_rear \
  --csv data/rear_encode/weu_rear_smoke.csv \
  --tol 0.15 \
  --rear_shift_sec 0.8

echo "== rendering overlay videos for a few good-overlap rides =="
mkdir -p eval/dual_view
for RT in 20240221112028 20240221112909 20240307112454 20240216103439; do
  echo "-- viz $RT --"
  python utils/visualise_dual_view.py --ride_ts "$RT" --start 0 --n_latents 90 --fps 20 \
    || echo "viz failed for $RT (continuing)"
done
echo "done $(date); outputs in eval/dual_view/"
ls -la eval/dual_view/ || true
