#!/bin/bash
# Autonomous eval driver for the two dimension ablations. Waits for the two
# training jobs, evals each (independent swap + quality) vs their baselines
# (probe2dim vs v14 ; critic8 vs loo_f3), then prints the head-to-heads.
set -u
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate 2>/dev/null
conda activate arrwm 2>/dev/null

CRITIC_JOB=5266612
PROBE_JOB=5266613
SWAP=sbatch/eval_action_swap_generic.sbatch
QUAL=sbatch/eval_quality_generic.sbatch

echo "[loo-eval] start $(date)"
# 1) wait for both training jobs to leave the queue
for J in "$CRITIC_JOB" "$PROBE_JOB"; do
  while squeue -j "$J" -h -o "%T" 2>/dev/null | grep -qE "PENDING|RUNNING|CONFIGURING|COMPLETING"; do sleep 120; done
  echo "[loo-eval] training job $J left queue $(date)"
done
sleep 15

# 2) per run: (config, logdir, tag). Verify a checkpoint exists, then eval.
declare -A CFG=(
  [critic8]=configs/causal_lora_diffusion_teacher_v14_loo_critic8_noprobe.yaml
  [probe2dim]=configs/causal_lora_diffusion_teacher_v14_loo_probe2dim.yaml )
declare -A DIR=(
  [critic8]=logs/v14_loo_critic8_noprobe
  [probe2dim]=logs/v14_loo_probe2dim )

EVAL_JOBS=()
for tag in critic8 probe2dim; do
  ckpt=$(ls -t "${DIR[$tag]}"/causal_lora_step*.pt 2>/dev/null | head -1)
  if [ -z "$ckpt" ]; then echo "[loo-eval] WARN: no checkpoint for $tag (${DIR[$tag]}); skipping its eval"; continue; fi
  echo "[loo-eval] $tag latest ckpt: $ckpt"
  s=$(sbatch --parsable "$SWAP" "${CFG[$tag]}" "${DIR[$tag]}" "paper_assets/swap_${tag}.json")
  q=$(sbatch --parsable "$QUAL" "${CFG[$tag]}" "${DIR[$tag]}" "paper_assets/qual_${tag}.json")
  echo "[loo-eval] submitted $tag swap=$s qual=$q"
  EVAL_JOBS+=("$s" "$q")
done

# 3) wait for all eval jobs
for J in "${EVAL_JOBS[@]}"; do
  while squeue -j "$J" -h -o "%T" 2>/dev/null | grep -qE "PENDING|RUNNING|CONFIGURING|COMPLETING"; do sleep 60; done
done
echo "[loo-eval] all eval jobs done $(date)"
sleep 10

# 4) collate
echo ""
echo "############### CONTROLLABILITY (independent swap judge) ###############"
python utils/analyze_swap_stratified.py 2>&1 || echo "swap analyze failed"
echo ""
echo "############### VISUAL QUALITY (probe2dim vs v14 ; critic8 vs loo_f3) ###############"
python - <<'PY' 2>&1 || echo "quality collate failed"
import json, os
PA='paper_assets'
def s(tag):
    p=f'{PA}/qual_{tag}.json'
    if not os.path.exists(p): return None
    d=json.load(open(p)).get('summary',{})
    return d
rows=[('v14',s('v14')),('probe2dim',s('probe2dim')),('loo_f3',s('loo_f3')),('critic8',s('critic8'))]
ks=['PSNR','SSIM','LPIPS','MUSIQ_imaging','CLIP_consistency','FVD_r3d18_indicative']
print(f"{'run':<12}"+''.join(f'{k:>13}' for k in ks))
for name,d in rows:
    if not d: print(f"{name:<12}  -- missing --"); continue
    print(f"{name:<12}"+''.join(f"{(d.get(k) if isinstance(d.get(k),(int,float)) else float('nan')):>13.3f}" for k in ks))
print("\nCompare: probe2dim vs v14  (does removing the 6 probe dims hurt?)")
print("         critic8   vs loo_f3 (does adding the 6 critic dims @0.5x help?)")
PY
echo "[loo-eval] done $(date)"
