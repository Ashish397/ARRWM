#!/bin/bash
#SBATCH --job-name=flow-rec-comp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Record denoising trajectories of the DENSE-FLOW comparators on window r08,
# 8 compass dirs, pinned noise (matched across dirs), for the cross-flow-model
# divergence comparison:
#   astra    50-step FlowMatch, one 8-frame chunk (--total_frames_to_generate 8)
#   worldcam 64-step diffusion-forcing ladder (stage=8 micro-steps/AR step),
#            20 AR steps so >=1 frame completes its full noise->clean lifetime
# (few-step DMD samplers minwm/matrixgame re-noise between steps -> no flow
# path; yume/worldplay are 4-point Euler -> excluded from the dense set)
set -e
ARR=/scratch/u6ex/as1748.u6ex/ARRWM
FVOUT=$ARR/analysis/eval_final/flow_viz
AS=$ARR/third_party/Astra
WAN=/scratch/u6ex/as1748.u6ex/frodobots/Wan2.1-T2V-1.3B
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache TMPDIR=/tmp

cd $ARR
PYTHONPATH=$ARR SW_WINDOWS="8" python utils/decode_seed_windows.py

# ---------------- astra: 8 dirs x 1 chunk x 50 steps ----------------
cd $AS
export PYTHONPATH=$AS/_shims:$AS
PROMPT="A first-person view from a small delivery robot on a sidewalk in a residential street. Houses with fences and gardens line the road, cars are parked along the curb, and trees cast soft shadows in bright daytime light. The camera moves smoothly through the scene."
SCRATCH=$FVOUT/.flowcomp_scratch
mkdir -p $SCRATCH
for PAIR in F:1 FR:5 R:3 BR:10 B:8 BL:9 L:2 FL:4; do
  D=${PAIR%%:*}; CT=${PAIR##*:}
  RECD=$FVOUT/flow_astra/r08_${D}
  [ -f $RECD/chunk0.npz ] && { echo "[comp] astra $D exists"; continue; }
  ASTRA_FLOW_REC=$RECD ASTRA_FLOW_SEED=1234 \
    $AS/_venv/bin/python scripts/infer_demo.py \
    --condition_image $ARR/analysis/eval_final/seed65/seed65_r08_f0.png \
    --condition_video $ARR/analysis/eval_final/seed65/seed65_r08.mp4 \
    --start_frame 0 --initial_condition_frames 4 \
    --frames_per_generation 8 --total_frames_to_generate 8 \
    --dit_path $AS/models/Astra/checkpoints/diffusion_pytorch_model.ckpt \
    --wan_model_path $WAN \
    --prompt "$PROMPT" \
    --modality_type sekai --cam_type $CT \
    --output_path $SCRATCH/astra_rec_r08_${D}.mp4
  echo "[comp] astra $D recorded"
done

# ---------------- worldcam: 8 dirs x 20 AR steps, full ladder ----------------
cd $ARR/third_party/WorldCam
WC_WINDOWS="8" WC_DIRS="F:FR:R:BR:B:BL:L:FL" \
  WC_STEPT=0.02 WC_STEPR=1.5 WC_ARSTEPS=20 WC_FPS=30 \
  WC_OUT=$SCRATCH WC_FLOW_REC_BASE=$FVOUT/flow_worldcam \
  PYTHONPATH= python worldcam_runner.py
echo "FLOW-REC-COMP done $(date)"
