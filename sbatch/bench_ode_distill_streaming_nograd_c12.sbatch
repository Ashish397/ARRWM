#!/bin/bash
#SBATCH --job-name=bench_ode_distill_streaming_nograd_c12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
#
# One-off: peak VRAM for streaming inference at
#   chunks_per_step = 2   (6 latent frames denoised per step)
#   cache_chunks    = 12  (36 past frames retained => local_attn = 42 frames)
#   denoising_steps = 4
# under torch.inference_mode() (no autograd graph).
#
# Timed portion = exactly 1 streaming step, preceded by enough warmup
# (cache_chunks + chunks_per_step = 14 base chunks) that the KV cache
# is fully populated and the eviction branch has fired.
#
# Output: logs/bench_ode_distill_streaming_nograd_c12_<jobid>/nograd_cache12.json

set -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM

_orig_args=("$@")
set --
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
set -- "${_orig_args[@]}"

export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache
export HF_HUB_CACHE=$HF_HOME
export HUGGINGFACE_HUB_CACHE=$HF_HOME
export TRANSFORMERS_CACHE=$HF_HOME
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

STUDENT_CKPT="${1:-/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_ode_distill_10h/latest_0000300.pt}"
TEACHER_CKPT="${2:-/scratch/u6ex/as1748.u6ex/ARRWM/logs/v14_balanced_weunz/causal_lora_step0006600.pt}"
CONFIG="${3:-configs/action_ode_distill.yaml}"

CHUNKS_PER_STEP=2
CACHE_CHUNKS=12
DENOISING_STEPS=4
WARMUP_CHUNKS=$(( CACHE_CHUNKS + CHUNKS_PER_STEP ))   # 14
TIMED_CHUNKS=$CHUNKS_PER_STEP                         # 2 => 1 timed step

OUT_DIR="/scratch/u6ex/as1748.u6ex/ARRWM/logs/bench_ode_distill_streaming_nograd_c12_${SLURM_JOB_ID:-local}"
mkdir -p "$OUT_DIR"
echo "=== $(date) job=${SLURM_JOB_ID:-local} ==="
echo "Student ckpt:     $STUDENT_CKPT"
echo "Teacher ckpt:     $TEACHER_CKPT"
echo "Config:           $CONFIG"
echo "cache_chunks:     $CACHE_CHUNKS"
echo "chunks_per_step:  $CHUNKS_PER_STEP"
echo "denoising_steps:  $DENOISING_STEPS"
echo "warmup_chunks:    $WARMUP_CHUNKS"
echo "timed_chunks:     $TIMED_CHUNKS"
echo "Output dir:       $OUT_DIR"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true

python action-forcing/bin/bench_streaming_inference.py \
    --ckpt "$TEACHER_CKPT" \
    --student-ckpt "$STUDENT_CKPT" \
    --config "$CONFIG" \
    --denoising_steps "$DENOISING_STEPS" \
    --chunks_per_step "$CHUNKS_PER_STEP" \
    --cache_chunks "$CACHE_CHUNKS" \
    --total_chunks "$TIMED_CHUNKS" \
    --warmup_chunks "$WARMUP_CHUNKS" \
    --no_grad \
    --label "nograd_cache${CACHE_CHUNKS}_cps${CHUNKS_PER_STEP}" \
    --output "$OUT_DIR/nograd_cache${CACHE_CHUNKS}.json"

echo
echo "=== $(date) nograd c12 bench finished ==="
