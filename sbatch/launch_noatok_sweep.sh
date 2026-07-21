#!/bin/bash
set -e; cd /scratch/u6ex/as1748.u6ex/ARRWM
NVID=${NVID:-16}
declare -A RANGES=( [A]="100 700" [B]="800 1400" [C]="1500 2100" [D]="2200 2800" )
for tag in A B C D; do
  read lo hi <<< "${RANGES[$tag]}"
  out="logs/v14e_noatok_offline_sweep_${tag}"; rm -rf "$out"
  sbatch --export=ALL,OCE_MIN=${lo},OCE_MAX=${hi},OCE_STRIDE=100,OCE_NVID=${NVID},OCE_OUT=${out} \
    --job-name=oce-sw4-${tag} sbatch/oce_sweep4.sbatch
done
echo "submitted 4 sweep jobs (4 GPUs each = 16 GPUs total), NVID=$NVID"
