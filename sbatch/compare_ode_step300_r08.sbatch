#!/usr/bin/env bash
# u6qf one-action step-300 probe against existing u6ex nine-action r08 videos.
# The packaged seed/prompt is byte-identical to the original u6ex r08 input.
#SBATCH --job-name=ode1a-r08-300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=2
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/u6qf/as1748.u6qf/ARRWM_straggler/_logs/%x_%j.out
#SBATCH --error=/scratch/u6qf/as1748.u6qf/ARRWM_straggler/_logs/%x_%j.err
set -eo pipefail

root=/scratch/u6qf/as1748.u6qf/ARRWM_straggler
arr=$root/ARRWM
cd "$arr"
source "$root/miniforge3/envs/arrwm/bin/activate"
set -u

export HF_HOME="$root/frodobots/hf_cache"
export TORCH_HOME="$root/torch_home"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$arr:$arr/action-forcing"
export TMPDIR=/tmp ARR_ROOT="$arr"
export FR_CONFIG="$arr/configs/ar_eval_dmd_student_u6qf.yaml"
export FR_SEED_PT="$arr/analysis/eval_final/r08_ode_seed_prompt.pt"
export FR_OUT="$arr/analysis/eval_final/ode_step300_r08"
export FR_RUNGS='1000,625,357.142857,208.333333'
export FR_VIDEO=1 FR_CHUNKS=6 FR_NSEEDS=1 FR_FPS=16 FR_SEED_CHUNKS=3
export FR_WINDOW=8 FR_DIRS=F,L,B,BR
test -s "$FR_SEED_PT"

ckpts=(
    "$arr/logs/ode14e_straggler/run_roll10k1a_strict1/action_ode_step0000300.pt"
    "$arr/logs/ode14e_straggler/run_rollkl10k1a_strict1/action_ode_step0000300.pt"
)
runs=(mse1_s300 kl1_s300)
for ckpt in "${ckpts[@]}"; do test -s "$ckpt"; done

pids=()
for gpu in 0 1; do
    CUDA_VISIBLE_DEVICES="$gpu" FR_RUN="${runs[$gpu]}" FR_CKPT="${ckpts[$gpu]}" \
      python utils/flow_record_ode_student_seed_override.py \
      >"$root/_logs/ode_r08_s300_${runs[$gpu]}_${SLURM_JOB_ID}.log" 2>&1 &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
