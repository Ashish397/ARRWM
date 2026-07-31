#!/bin/bash
# Launch the final action-injection (A) or static-render (B) eval for all 7 runs.
# Usage: bash sbatch/launch_inject.sh A     (or B)
set -e
cd ${AF_ROOT}
PHASE="${1:-A}"
EF=analysis/eval_final
WIN="$EF/phase${PHASE}_windows.json"
MAN="$EF/manifest_unseen.pt"
[ -f "$WIN" ] || { echo "missing $WIN — run select_unseen_windows first"; exit 1; }
[ -f "$MAN" ] || { echo "missing $MAN"; exit 1; }

# run -> config:logdir
declare -A MAP=(
  [pca8_8node]="causal_lora_diffusion_teacher_v14e.yaml:v14e_pca8_raw"
  [pca4]="causal_lora_diffusion_teacher_v14e_pca4.yaml:v14e_pca4"
  [pca2]="causal_lora_diffusion_teacher_v14e_pca2.yaml:v14e_pca2"
  [16node]="causal_lora_diffusion_teacher_v14e_16node.yaml:v14e_16node"
  [4node]="causal_lora_diffusion_teacher_v14e_4node.yaml:v14e_4node"
  [noatok]="causal_lora_diffusion_teacher_v14e_noatok.yaml:v14e_noatok"
  [noadaln]="causal_lora_diffusion_teacher_v14e_noadaln.yaml:v14e_noadaln"
)
for run in "${!MAP[@]}"; do
  cfg="configs/${MAP[$run]%%:*}"; dir="logs/${MAP[$run]##*:}"
  ck="$dir/causal_lora_step0005000.pt"
  out="logs/eval_final/${PHASE}/${run}"
  mkdir -p "$out"
  jid=$(sbatch --parsable --job-name="inj${PHASE}-${run}" \
    --export=ALL,IE_CONFIG="$cfg",IE_CKPT="$ck",IE_PHASE="$PHASE",IE_WINDOWS="$WIN",IE_MANIFEST="$MAN",IE_OUT="$out" \
    sbatch/inject_eval.sbatch)
  echo "submitted inj${PHASE}-${run} = $jid  (out=$out)"
done
