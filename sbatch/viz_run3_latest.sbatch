#!/bin/bash
#SBATCH --job-name=viz-run3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Videos + flow-tree ellipse diagram from the newest run3_flip2_{roll,rolled,
# rollkl} checkpoints. Shares logs/.viz_run3.lock with the holder command
# files (.holder_cmd_6015261/2.sh), so whichever allocation lands first does
# the work exactly once and the others skip.
bash /scratch/u6ex/as1748.u6ex/ARRWM/logs/.viz_run3_cmd.sh
