#!/bin/bash
#SBATCH --job-name=pilots-v2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# FIXED pilot rerun (post 3-agent review): label-leak clean_x shift, chunked
# supervision flag ON, CD losses OFF, fresh run2_* logdirs, per-branch
# clean_x, critic-mask pin, interleaved fixed-RoPE probes (pilot2_*).
bash /scratch/u6ex/as1748.u6ex/ARRWM/sbatch/_train_pilots_seq.sh
