#!/bin/bash
#SBATCH --job-name=astra-noopnp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Astra production: 32 windows x 8 dirs, 8s (~140s/clip); cam_type map:
# F=1 L=2 R=3 FL=4 FR=5 B=8 BL=9 BR=10 (8/9/10 = our added backward branches)
set -e
ARR=/scratch/u6ex/as1748.u6ex/ARRWM
AS=$ARR/third_party/Astra
WAN=/scratch/u6ex/as1748.u6ex/frodobots/Wan2.1-T2V-1.3B
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache TMPDIR=/tmp

cd $ARR
ALLW=$(seq -s: 0 31)
PYTHONPATH=$ARR SW_WINDOWS="$ALLW" python utils/decode_seed_windows.py

cd $AS
export PYTHONPATH=$AS/_shims:$AS
OUTROOT=$ARR/logs/eval_final/A_astra_noop_neutral
mkdir -p $OUTROOT
PROMPT="A first-person view from a small delivery robot on a sidewalk in a residential street. Houses with fences and gardens line the road, cars are parked along the curb, and trees cast soft shadows in bright daytime light."

runshard() {  # gpu windows...
  local GPU=$1; shift
  for W in "$@"; do
    WW=$(printf "%02d" $W)
    for PAIR in NOOP:11; do
      D=${PAIR%%:*}; CT=${PAIR##*:}
      OUTP=$OUTROOT/astra_r${WW}_${D}.mp4
      [ -f $OUTP ] && continue
      T0=$(date +%s)
      CUDA_VISIBLE_DEVICES=$GPU $AS/_venv/bin/python scripts/infer_demo.py \
        --condition_image $ARR/analysis/eval_final/seed65/seed65_r${WW}_f0.png \
        --condition_video $ARR/analysis/eval_final/seed65/seed65_r${WW}.mp4 \
        --start_frame 0 --initial_condition_frames 4 \
        --frames_per_generation 8 --total_frames_to_generate 40 \
        --dit_path $AS/models/Astra/checkpoints/diffusion_pytorch_model.ckpt \
        --wan_model_path $WAN \
        --prompt "$PROMPT" \
        --modality_type sekai --cam_type $CT \
        --output_path $OUTP > $ARR/logs/astranoopnp_g${GPU}.log 2>&1 || { echo "[astra-noopnp] FAIL r${WW}_${D}"; continue; }
      T1=$(date +%s)
      echo "[astra-noopnp] saved astra_r${WW}_${D}.mp4 gen_time=$((T1-T0))s"
    done
  done
}
runshard 0 0 1 2 3 4 5 6 7 &
runshard 1 8 9 10 11 12 13 14 15 &
runshard 2 16 17 18 19 20 21 22 23 &
runshard 3 24 25 26 27 28 29 30 31 &
wait
echo "ASTRA-NOOPNP done $(date)"
