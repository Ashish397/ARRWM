#!/bin/bash
#SBATCH --job-name=latent-boundary-gate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=2
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# GATE: is loo_f3 a safe v15 base? Render ONE diverse clip with v14 (GPU0) and
# loo_f3 (GPU1), both with LATENT_BOUNDARY_DIAG=1. Compare the logged
# boundary_rms/interior_rms ratio in LATENT space on the committed rollout:
#   * both ratios ~1            -> latents continuous; seam was decode-only -> loo_f3 SAFE base
#   * loo_f3 ratio >> v14 ratio -> loo_f3 model output discontinuous at chunk
#                                  boundaries -> real AR regression -> NOT a safe base
set -u -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR HF_HUB_CACHE=$CACHE_DIR HUGGINGFACE_HUB_CACHE=$CACHE_DIR TRANSFORMERS_CACHE=$CACHE_DIR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export LATENT_BOUNDARY_DIAG=1
ENC=/projects/u6ex/fbots/frodobots_encoded_weu
CAP=/projects/u6ex/fbots/frodobots_captions/train
OUT=eval/latent_boundary_gate; mkdir -p "$OUT"

# one diverse clip (first in the curated list)
read -r Z S < <(python3 -c "import json;w=json.load(open('paper_assets/diverse_clips.json'))['windows'][0];print(w['zarr_path'].split('/')[-1], w['start'])")
echo "gate clip: $Z $S"

run_one() {  # gpu assignment config manifest label
  local gpu=$1 a=$2 cfg=$3 man=$4 lab=$5
  CUDA_VISIBLE_DEVICES=$gpu WORLD_SIZE=1 LOCAL_RANK=0 \
    python utils/eval_causal_chain.py --assignment_index "$a" --config "$cfg" --manifest "$man" \
      --rank_zarr "$Z" --rank_offset "$S" --rank_mode dataset --rank_tag gate \
      --encoded_root "$ENC" --caption_root "$CAP" --seed 42 --output_dir "$OUT/$lab" \
      > "$OUT/$lab.log" 2>&1 || echo "FAIL $lab (see $OUT/$lab.log)"
}

run_one 0 5 configs/causal_lora_diffusion_teacher_v14.yaml      logs/v14_balanced_weunz/.ride_manifest.pt v14    &
run_one 1 6 configs/causal_lora_diffusion_teacher_v14_loo_f3.yaml logs/v14_loo_f3/.ride_manifest.pt        loo_f3 &
wait

echo "=== LATENT_BOUNDARY_DIAG results ==="
grep -h "LATENT_BOUNDARY_DIAG" "$OUT"/v14.log "$OUT"/loo_f3.log || echo "no diag lines found"
echo "Done $(date)"
