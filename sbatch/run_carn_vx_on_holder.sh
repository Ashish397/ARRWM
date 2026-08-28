#!/bin/bash
# CARN_VX_2708 matched-arm launcher on an existing two-node holder.
#
# Every arm is generated from sbatch/carncommit_long.sbatch, the exact source
# of dmd10k_carncommit_long_{off,on}.  Mode-specific values are injected as
# LAST-WINS dotlist overrides immediately before run_name.  This preserves the
# requested OFF base (DMD-band clean GAN, frozen action critic, one pair mode)
# while making the CARN consumer explicit and auditable.
set -euo pipefail
: "${CARNVX_MODE:?set CARNVX_MODE=tx_plus_former|tx_minus_latter|tx_both|aux_minus|commit_aux|commit_aux_flash60|commit_aux_staged|aux_tx_minus|aux_tx_minus_ac_online|aux_tx_minus_ac_frozen055|commit_aux_tx_minus|commit_aux_tx_both}"
PORTOFF=${PORTOFF:-2708}
RUNSTAMP=${RUNSTAMP:-$(date +%H%M%S)}
MAXSTEPS=${MAXSTEPS:-300}

# Campaign-wide Flash-era routing contract.  The temporary no-Flash treatment
# moved every consumer onto a differentiable ladder endpoint and proved both
# visually worse and materially more memory hungry.  Restore the dedicated
# t=60 Flash x0 for anti-collapse/action guidance and the legacy R2 domain.
# LADD remains on the base recipe's clean DMD-scored band; Flash is an
# auxiliary/CARN domain here, not a replacement GAN fake source.
COMMON="sample_interval=15 checkpoint_interval=200 flash_dmd_enabled=true flash_dmd_gan_t=60 anti_collapse_apply_to_flash_rung=true anti_collapse_apply_to_ladder_endpoint=false pix_finish_grad_enabled=false gen_aux_losses_x0_source=flash forward_noiser_train_source=flash forward_noiser_rollout2_source=legacy ladd_fake_sample_source=dmd ladd_disc_force_clean=true reverse_noiser_dedrift_apply_to_commit=false reverse_noiser_dedrift_apply_to_flash=false reverse_noiser_internalize_weight=0.0 reverse_noiser_dedrift_apply_to_train_score=false forward_noiser_apply_decoupled=false forward_noiser_apply_gt_former=false forward_noiser_apply_gt_both=false forward_noiser_apply_in_aux=false forward_noiser_cycle_enabled=false ladd_gt_transition_carn_former=false ladd_gt_transition_carn_latter_reverse=false ladd_gt_transition_match=false ladd_adjacent_chunks_enabled=false"

case "$CARNVX_MODE" in
  tx_plus_former)
    DARM="carnvx_tx_plus_former_flash60_${RUNSTAMP}"
    CARNVX_EXTRA="$COMMON ladd_gt_vs_fake_enabled=false ladd_gt_transition_enabled=true ladd_fake_backbone_grad_scale=0.2 forward_noiser_reverse=false reverse_noiser_dedrift_enabled=false ladd_gt_transition_carn_former=true ladd_gt_transition_carn_steps=1 ladd_gt_transition_gen_detach_former=true"
    ;;
  tx_minus_latter)
    DARM="carnvx_tx_minus_latter_flash60_${RUNSTAMP}"
    CARNVX_EXTRA="$COMMON ladd_gt_vs_fake_enabled=false ladd_gt_transition_enabled=true ladd_fake_backbone_grad_scale=0.2 forward_noiser_reverse=true reverse_noiser_dedrift_enabled=false ladd_gt_transition_carn_latter_reverse=true ladd_gt_transition_carn_steps=1 ladd_gt_transition_gen_detach_former=true"
    ;;
  tx_both)
    # Both signs require two distinct learned maps. Cycle mode supplies
    # F(R1->R2) for +former and G(R2->R1) for -latter.
    DARM="carnvx_tx_both_flash60_${RUNSTAMP}"
    CARNVX_EXTRA="$COMMON ladd_gt_vs_fake_enabled=false ladd_gt_transition_enabled=true ladd_fake_backbone_grad_scale=0.2 forward_noiser_reverse=false forward_noiser_cycle_enabled=true reverse_noiser_dedrift_enabled=false ladd_gt_transition_carn_former=true ladd_gt_transition_carn_latter_reverse=true ladd_gt_transition_carn_steps=1 ladd_gt_transition_gen_detach_former=true"
    ;;
  aux_minus)
    DARM="carnvx_aux_minus_flash60_${RUNSTAMP}"
    CARNVX_EXTRA="$COMMON forward_noiser_train_source=rollout ladd_gt_vs_fake_enabled=true ladd_gt_transition_enabled=false ladd_fake_backbone_grad_scale=0.2 forward_noiser_reverse=true reverse_noiser_dedrift_enabled=true reverse_noiser_dedrift_apply_to_train_score=false reverse_noiser_internalize_weight=${CARNVX_AUX_WEIGHT:-0.25}"
    ;;
  commit_aux)
    DARM="carnvx_commit_aux_minus_flash60_${RUNSTAMP}"
    CARNVX_EXTRA="$COMMON forward_noiser_train_source=rollout ladd_gt_vs_fake_enabled=true ladd_gt_transition_enabled=false ladd_fake_backbone_grad_scale=0.2 forward_noiser_reverse=true reverse_noiser_dedrift_enabled=true reverse_noiser_dedrift_apply_to_train_score=false reverse_noiser_internalize_weight=${CARNVX_AUX_WEIGHT:-0.25} reverse_noiser_dedrift_apply_to_commit=true"
    ;;
  commit_aux_flash60)
    # Matched follow-up to the no-Flash commit+aux arm. Restore the complete
    # known-good Flash-era auxiliary routing bundle, not the LADD fake: LADD
    # deliberately remains on its tuned clean DMD-scored band.  R1=rollout and
    # R2=legacy reproduce the completed aux-minus arm's CARN domains; the
    # frozen action critic and anti-collapse anchor return to the dedicated
    # t=60 x0. KV commit remains the ladder endpoint through the pipeline's
    # default-true flash_dmd_commit_ladder_endpoint contract.
    DARM="carnvx_commit_aux_minus_flash60_tuned_${RUNSTAMP}"
    CARNVX_EXTRA="$COMMON forward_noiser_train_source=rollout ladd_gt_vs_fake_enabled=true ladd_gt_transition_enabled=false ladd_fake_backbone_grad_scale=0.2 forward_noiser_reverse=true reverse_noiser_dedrift_enabled=true reverse_noiser_dedrift_apply_to_train_score=false reverse_noiser_internalize_weight=${CARNVX_AUX_WEIGHT:-0.25} reverse_noiser_dedrift_apply_to_commit=true"
    ;;
  commit_aux_staged)
    # Upgraded combination: reproduce the clean aux-minus arm exactly while
    # the R2->R1 corrector learns, then introduce only a quarter-dose into KV
    # memory over a long ramp.  Aux keeps its calibrated 0.25 L1 weight and
    # full de-drift pseudo-target; only the recurrent commit is attenuated.
    DARM="carnvx_commit_aux_minus_staged_flash60_${RUNSTAMP}"
    CARNVX_EXTRA="$COMMON forward_noiser_train_source=rollout ladd_gt_vs_fake_enabled=true ladd_gt_transition_enabled=false ladd_fake_backbone_grad_scale=0.2 forward_noiser_reverse=true reverse_noiser_dedrift_enabled=true reverse_noiser_dedrift_apply_to_train_score=false reverse_noiser_internalize_weight=${CARNVX_AUX_WEIGHT:-0.25} reverse_noiser_dedrift_apply_to_commit=true reverse_noiser_commit_alpha=${CARNVX_COMMIT_ALPHA:-0.25} reverse_noiser_commit_start_step=${CARNVX_COMMIT_START:-100} reverse_noiser_commit_ramp_steps=${CARNVX_COMMIT_RAMP:-100}"
    ;;
  aux_tx_minus)
    # Two pair modes means ten discriminator backwards per generator step.
    # Scale 0.1 preserves the one-mode total backbone kick (5 * 0.2 = 1).
    DARM="carnvx_aux_minus_tx_minus_flash60_${RUNSTAMP}"
    CARNVX_EXTRA="$COMMON forward_noiser_train_source=rollout ladd_gt_vs_fake_enabled=true ladd_gt_transition_enabled=true ladd_fake_backbone_grad_scale=0.1 forward_noiser_reverse=true reverse_noiser_dedrift_enabled=true reverse_noiser_dedrift_apply_to_train_score=false reverse_noiser_internalize_weight=${CARNVX_AUX_WEIGHT:-0.25} ladd_gt_transition_carn_latter_reverse=true ladd_gt_transition_carn_steps=1 ladd_gt_transition_gen_detach_former=true"
    ;;
  aux_tx_minus_ac_online)
    # Exact aux-minus + tx-minus treatment from m5z2cfxl, with the action
    # critic changed from a frozen readout to the already-proven online
    # recipe.  ``action_teacher_mode=all`` is REQUIRED: freeze=false with the
    # teacher off disables the action-critic loss path rather than making it
    # online.  The 0.5 regression weight / two updates match the completed
    # gansig_gtvf_dmd_criticon arm; generator guidance itself stays at 0.3.
    DARM="carnvx_aux_minus_tx_minus_ac_online_flash60_${RUNSTAMP}"
    CARNVX_EXTRA="$COMMON forward_noiser_train_source=rollout ladd_gt_vs_fake_enabled=true ladd_gt_transition_enabled=true ladd_fake_backbone_grad_scale=0.1 forward_noiser_reverse=true reverse_noiser_dedrift_enabled=true reverse_noiser_dedrift_apply_to_train_score=false reverse_noiser_internalize_weight=${CARNVX_AUX_WEIGHT:-0.25} ladd_gt_transition_carn_latter_reverse=true ladd_gt_transition_carn_steps=1 ladd_gt_transition_gen_detach_former=true action_critic_aux_enabled=true action_critic_freeze=false action_teacher_mode=all action_critic_z_loss_weight=0.5 critic_updates_per_step=2 generator_action_z_guidance_weight=0.3"
    ;;
  aux_tx_minus_ac_frozen055)
    # Exact m5z2cfxl treatment with the frozen v14e critic retained and only
    # its generator-side guidance ceiling raised from 0.3 to 0.55.
    DARM="carnvx_aux_minus_tx_minus_ac_frozen055_flash60_${RUNSTAMP}"
    CARNVX_EXTRA="$COMMON forward_noiser_train_source=rollout ladd_gt_vs_fake_enabled=true ladd_gt_transition_enabled=true ladd_fake_backbone_grad_scale=0.1 forward_noiser_reverse=true reverse_noiser_dedrift_enabled=true reverse_noiser_dedrift_apply_to_train_score=false reverse_noiser_internalize_weight=${CARNVX_AUX_WEIGHT:-0.25} ladd_gt_transition_carn_latter_reverse=true ladd_gt_transition_carn_steps=1 ladd_gt_transition_gen_detach_former=true action_critic_aux_enabled=true action_critic_freeze=true action_teacher_mode=off generator_action_z_guidance_weight=0.55"
    ;;
  commit_aux_tx_minus)
    DARM="carnvx_commit_aux_minus_tx_minus_flash60_${RUNSTAMP}"
    CARNVX_EXTRA="$COMMON forward_noiser_train_source=rollout ladd_gt_vs_fake_enabled=true ladd_gt_transition_enabled=true ladd_fake_backbone_grad_scale=0.1 forward_noiser_reverse=true reverse_noiser_dedrift_enabled=true reverse_noiser_dedrift_apply_to_train_score=false reverse_noiser_internalize_weight=${CARNVX_AUX_WEIGHT:-0.25} reverse_noiser_dedrift_apply_to_commit=true ladd_gt_transition_carn_latter_reverse=true ladd_gt_transition_carn_steps=1 ladd_gt_transition_gen_detach_former=true"
    ;;
  commit_aux_tx_both)
    # Full sign-symmetric stack: GT-vs-fake aux target + transition pair,
    # with the cycle reverse net also correcting committed AR memory.
    DARM="carnvx_commit_aux_minus_tx_both_flash60_${RUNSTAMP}"
    CARNVX_EXTRA="$COMMON forward_noiser_train_source=rollout ladd_gt_vs_fake_enabled=true ladd_gt_transition_enabled=true ladd_fake_backbone_grad_scale=0.1 forward_noiser_reverse=false forward_noiser_cycle_enabled=true reverse_noiser_dedrift_enabled=true reverse_noiser_dedrift_apply_to_train_score=false reverse_noiser_internalize_weight=${CARNVX_AUX_WEIGHT:-0.25} reverse_noiser_dedrift_apply_to_commit=true ladd_gt_transition_carn_former=true ladd_gt_transition_carn_latter_reverse=true ladd_gt_transition_carn_steps=1 ladd_gt_transition_gen_detach_former=true"
    ;;
  *)
    echo "unknown CARNVX_MODE=$CARNVX_MODE" >&2
    exit 2
    ;;
esac

echo "[CARNVX-CONFIG] mode=$CARNVX_MODE"
echo "[CARNVX-CONFIG] source=sbatch/carncommit_long.sbatch"
echo "[CARNVX-CONFIG] run_name=dmd10k_${DARM}_j<holder>"
echo "[CARNVX-CONFIG] max_steps=$MAXSTEPS last_wins=[$CARNVX_EXTRA]"

# Static review path: resolve the exact final override block without touching
# Slurm, conda, W&B, a holder command file, or a GPU.
if [ "${PRINT_ONLY:-0}" = "1" ]; then
  exit 0
fi

: "${HOLDER:?set HOLDER to a running two-node holder job ID}"
if ! [[ "$MAXSTEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[CARNVX] FATAL: MAXSTEPS must be a positive integer; got $MAXSTEPS" >&2
  exit 4
fi
export DARM CARNVX_EXTRA MAXSTEPS
# Keep the inherited source script's early provenance line consistent with
# the authoritative last-wins treatment. The explicit dotlist flag remains
# the runtime source of truth.
case "$CARNVX_MODE" in
  commit_aux|commit_aux_flash60|commit_aux_staged|commit_aux_tx_minus|commit_aux_tx_both)
    export COMMITDEDRIFT=true
    ;;
  *)
    export COMMITDEDRIFT=false
    ;;
esac

# Refuse to collide with another torchrun on the allocation.  Pending holders
# are armed via their command file and only invoke this after becoming RUNNING.
LIVE=$(squeue -j "$HOLDER" -h -s -o "%i" 2>/dev/null | grep -vE '\.(batch|extern)$' | wc -l) || true
if [ "${LIVE:-0}" -gt 0 ]; then
  echo "[CARNVX] holder $HOLDER already has $LIVE live step(s); refusing collision" >&2
  exit 3
fi
NODE_EXPR=$(squeue -j "$HOLDER" -h -o %N)
if [ -z "$NODE_EXPR" ] || [ "$NODE_EXPR" = "(null)" ]; then
  echo "[CARNVX] FATAL: holder $HOLDER is not RUNNING or has no nodes" >&2
  exit 5
fi
NODES=$(scontrol show hostnames "$NODE_EXPR")
NNODE=$(echo "$NODES" | wc -l)
if [ "$NNODE" -ne 2 ]; then
  echo "[CARNVX] FATAL: expected a two-node holder; $HOLDER has $NNODE" >&2
  exit 6
fi
NODELIST=$(echo "$NODES" | paste -sd,)
MASTER_ADDR=$(echo "$NODES" | head -1)
MASTER_PORT=$((29500 + (HOLDER + PORTOFF) % 16000))
SRC=sbatch/carncommit_long.sbatch
TMP0=$(mktemp "/tmp/carnvx_${CARNVX_MODE}_${HOLDER}_${PORTOFF}_inject.XXXXXX")
TMP1=$(mktemp "/tmp/carnvx_${CARNVX_MODE}_${HOLDER}_${PORTOFF}_run.XXXXXX")
trap 'rm -f "$TMP0" "$TMP1"' EXIT
N_RUN_NAME=$(grep -c '^[[:space:]]*run_name=dmd10k_' "$SRC")
if [ "$N_RUN_NAME" -ne 1 ]; then
  echo "[CARNVX] FATAL: expected one run_name injection site; got $N_RUN_NAME" >&2
  exit 7
fi
awk '
  /^[[:space:]]*run_name=dmd10k_/ { print "    " ENVIRON["CARNVX_EXTRA"] " \\" }
  { print }
' "$SRC" > "$TMP0"
sed -e 's/^srun torchrun \\/srun --jobid='"$HOLDER"' --overlap --nodelist='"$NODELIST"' --nodes='"$NNODE"' --ntasks-per-node=1 --gpus-per-node=4 --gpu-bind=none torchrun \\/' \
    -e 's/--nnodes=\$SLURM_NNODES/--nnodes='"$NNODE"'/' \
    -e 's/--rdzv_id=\$SLURM_JOB_ID/--rdzv_id='"$HOLDER$PORTOFF"'/' \
    -e 's|--rdzv_endpoint=\${MASTER_ADDR}:\${MASTER_PORT}|--rdzv_endpoint='"$MASTER_ADDR:$MASTER_PORT"'|' \
    -e 's/^#SBATCH.*//' \
    "$TMP0" > "$TMP1"

LOG=logs/holdersmoke_carnvx_${CARNVX_MODE}_h${HOLDER}_${RUNSTAMP}.log
echo "[CARNVX] mode=$CARNVX_MODE holder=$HOLDER nodes=$NODELIST port=$MASTER_PORT"
echo "[CARNVX] base=carncommit_long_off overrides=[$CARNVX_EXTRA]"
echo "[CARNVX] log=$LOG steps=$MAXSTEPS"
{
  echo "[CARNVX-CONFIG] mode=$CARNVX_MODE"
  echo "[CARNVX-CONFIG] source=$SRC"
  echo "[CARNVX-CONFIG] run_name=dmd10k_${DARM}_j${HOLDER}"
  echo "[CARNVX-CONFIG] max_steps=$MAXSTEPS last_wins=[$CARNVX_EXTRA]"
} > "$LOG"
set +e
SLURM_JOB_ID=$HOLDER SLURM_NNODES=$NNODE SLURM_JOB_NODELIST=$NODELIST \
  bash "$TMP1" >> "$LOG" 2>&1
RC=$?
set -e
echo "[CARNVX] exit=$RC log=$LOG"
exit "$RC"
