#!/bin/bash
#SBATCH --job-name=kl1v9-8ctx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/u6qf/as1748.u6qf/ARRWM_straggler/_logs/%x_%j.out
#SBATCH --error=/scratch/u6qf/as1748.u6qf/ARRWM_straggler/_logs/%x_%j.err
set -eo pipefail

ROOT=/scratch/u6qf/as1748.u6qf/ARRWM_straggler
ARR=$ROOT/ARRWM
OUT=$ARR/analysis/eval_final/kl_1v9_8contexts
MANIFEST=$ARR/analysis/.dmd_smoke_manifest.pt
CFG=$ARR/configs/ar_eval_dmd_student_u6qf.yaml
mkdir -p "$OUT" "$ROOT/_logs"
cd "$ARR"
source "$ROOT/miniforge3/envs/arrwm/bin/activate"
set -u

export HF_HOME=$ROOT/frodobots/hf_cache
export TORCH_HOME=$ROOT/torch_home
export PYTHONPATH=$ARR:$ARR/action-forcing
export TMPDIR=/tmp
export ARR_ROOT=$ARR
export FR_ZARR_ROOT=$ROOT/DATA/frodobots_encoded_weunz
export FR_CONFIG=$CFG
export FR_MANIFEST=$MANIFEST
export FR_OUT=$OUT
export FR_RUNGS='1000,625,357.142857,208.333333'
export FR_VIDEO=1 FR_CHUNKS=29 FR_NSEEDS=1 FR_FPS=16 FR_SEED_CHUNKS=3
export FR_DIRS=L,FL,B,BR

KL9=$ARR/logs/ode14e_pilot/run3_flip2_rollkl910k/action_ode_step0000500.pt
KL1=$ARR/logs/ode14e_straggler/run_rollkl10k1a_strict1/action_ode_step0000500.pt
test -s "$KL9"
test -s "$KL1"

# Eight distinct real starting contexts. Offsets are centered, chunk-aligned,
# and leave at least the three real seed chunks available in every ride.
INDICES=(0 1 2 3 4 5 6 7)
OFFSETS=(36 285 744 663 744 744 45 42)

worker() {
  local gpu=$1 run ckpt lo
  if (( gpu < 2 )); then
    run=kl_9action_8ctx
    ckpt=$KL9
  else
    run=kl_1action_8ctx
    ckpt=$KL1
  fi
  lo=$(( (gpu % 2) * 4 ))
  for ((j=lo; j<lo+4; j++)); do
    CUDA_VISIBLE_DEVICES=$gpu \
      FR_RUN=$run FR_CKPT=$ckpt \
      FR_MANIFEST_INDEX=${INDICES[$j]} FR_OFFSET=${OFFSETS[$j]} \
      FR_WINDOW_LABEL=$j \
      python utils/flow_record_ode_student.py \
      >"$ROOT/_logs/kl1v9_${run}_ctx${j}_${SLURM_JOB_ID}.log" 2>&1
  done
}

for gpu in 0 1 2 3; do
  worker "$gpu" &
done
wait

FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
MC=$OUT/.motion_check
for ctx in 0 1 2 3 4 5 6 7; do
  inputs=()
  filters=()
  n=0
  for dir in L FL B BR; do
    for model in kl_9action_8ctx kl_1action_8ctx; do
      inputs+=( -i "$MC/$model/r$(printf '%02d' "$ctx")_${dir}_s0.mp4" )
      label="${model/kl_/KL } | $dir"
      filters+=( "[$n:v]scale=416:240,drawtext=fontfile=$FONT:text='$label':x=8:y=8:fontsize=18:fontcolor=white:borderw=2[v$n]" )
      n=$((n+1))
    done
  done
  filter=$(IFS=';'; echo "${filters[*]}")
  filter+=";[v0][v1][v2][v3][v4][v5][v6][v7]xstack=inputs=8:layout=0_0|416_0|0_240|416_240|0_480|416_480|0_720|416_720[v]"
  ffmpeg -y "${inputs[@]}" -filter_complex "$filter" -map '[v]' \
    -an -r 16 -c:v libx264 -crf 18 -pix_fmt yuv420p \
    "$OUT/seed_$(printf '%02d' "$ctx")_L_FL_B_BR_kl9_vs_kl1.mp4"
done

echo "KL 1-action vs 9-action, eight-context comparison complete: $OUT"
