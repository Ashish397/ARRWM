#!/bin/bash
#SBATCH --job-name=iqa-cmp
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
python utils/no_ref_iqa.py  14_s50=.tmp/iqa_vids/14_step50.mp4 14d_s50=.tmp/iqa_vids/14d_step50.mp4 14_s100=.tmp/iqa_vids/14_step100.mp4 14d_s100=.tmp/iqa_vids/14d_step100.mp4 14_s150=.tmp/iqa_vids/14_step150.mp4 14d_s150=.tmp/iqa_vids/14d_step150.mp4 14_s500=.tmp/iqa_vids/14_step500.mp4 14d_s500=.tmp/iqa_vids/14d_step500.mp4 14_s1000=.tmp/iqa_vids/14_step1000.mp4 14d_s1000=.tmp/iqa_vids/14d_step1000.mp4 14_s1050=.tmp/iqa_vids/14_step1050.mp4 14d_s1050=.tmp/iqa_vids/14d_step1050.mp4
