#!/bin/bash
#SBATCH --job-name=nc-nullinv
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# No Critic's 32 no-op rollouts under the NULL-INVERSE command, matching the
# B_Astarts_nullinv campaign Table 15 / Figure 28 were built from. z=(0,0)
# would command the DATASET-MEAN motion (a forward creep) = the plain
# B_Astarts campaign, which is the wrong one.
set -e -o pipefail
ARR=/scratch/u6ex/as1748.u6ex/ARRWM
cd $ARR
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=$ARR:$PYTHONPATH TMPDIR=/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export IE_CONFIG=configs/causal_lora_diffusion_teacher_v14e_nocritic.yaml
export IE_CKPT=logs/v14e_nocritic/causal_lora_step0005000.pt
export IE_PHASE=B IE_NULL_Z="-0.0234,-0.0013"
export IE_WINDOWS=analysis/eval_final/phaseA_windows.json
export IE_MANIFEST=analysis/eval_final/manifest_unseen.pt
export IE_OUT=logs/eval_final/B_Astarts_nullinv/nocritic
srun --ntasks=4 --gpus-per-node=4 bash -c '
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID IE_SHARD=$SLURM_PROCID IE_NSHARDS=4
  export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_$SLURM_PROCID TRITON_CACHE_DIR=/tmp/tr_$SLURM_PROCID
  python utils/inject_eval.py
'
echo "NC-NULLINV done count=$(ls $IE_OUT/control_test/*_raw.mp4 2>/dev/null | wc -l) $(date)"
