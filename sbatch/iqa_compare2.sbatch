#!/bin/bash
#SBATCH --job-name=iqa-cmp2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate && conda activate arrwm
python utils/no_ref_iqa.py  14_s1000=.tmp/iqa_vids/14_step1000.mp4 14d_s1000=.tmp/iqa_vids/14d_step1000.mp4 14_s1200=.tmp/iqa_vids/14_step1200.mp4 14d_s1200=.tmp/iqa_vids/14d_step1200.mp4 14_s1400=.tmp/iqa_vids/14_step1400.mp4 14d_s1400=.tmp/iqa_vids/14d_step1400.mp4 14_s1600=.tmp/iqa_vids/14_step1600.mp4 14d_s1600=.tmp/iqa_vids/14d_step1600.mp4 14_s1800=.tmp/iqa_vids/14_step1800.mp4 14d_s1800=.tmp/iqa_vids/14d_step1800.mp4 14_s2000=.tmp/iqa_vids/14_step2000.mp4 14d_s2000=.tmp/iqa_vids/14d_step2000.mp4
