#!/bin/bash
# Run an ordered sequence of panel32 generation tasks inside one four-GPU holder.
set -euo pipefail

R=${ARRWM_REMOTE_ROOT:-/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler}
A=${ARRWM_CODE_ROOT:-$R/ARRWM}
PANEL_ROOT=${PANEL32_STAGE:-$R/panel32_stage}
MANIFEST=${PANEL32_MANIFEST:-$A/grids/eval/panel32_locked_v1.json}
PROVENANCE=${PANEL32_PROVENANCE:-$PANEL_ROOT/sources/panel32_source_provenance.json}
BUNDLES=${PANEL32_BUNDLES:-$PANEL_ROOT/ours_seed_bundles/panel32_seed_bundles.json}

test "$#" -gt 0
for task in "$@"; do
  IFS=: read -r kind name action extra <<<"$task"
  echo "PANEL32_SEQUENCE_START task=$task holder=${SLURM_JOB_ID:-none} $(date -Is)"
  case "$kind" in
    external)
      test -n "$name" && test -n "$action"
      bash "$A/sbatch/u6qf/panel32_external_holder_action.sh" \
        "$name" "$action" "$MANIFEST" "$PROVENANCE"
      ;;
    yume)
      test -n "$name"
      YUME_PANEL_OUT="$PANEL_ROOT/fleet30s/yume5b" \
      YUME_PANEL_STAGE="$PANEL_ROOT/yume5b/work" \
        bash "$A/sbatch/u6qf/yume5b_panel32_holder_action.sh" \
          "$name" "$MANIFEST" "$PROVENANCE"
      ;;
    ours)
      test -n "$name" && test -n "$action"
      case "$name" in
        recoverybase) ckpt=recovery_base; commit=auto ;;
        nocarn) ckpt=no_carn; commit=off ;;
        nocommit) ckpt=no_commit; commit=off ;;
        noaux) ckpt=no_aux; commit=auto ;;
        nogan) ckpt=no_gan; commit=auto ;;
        meanenergy) ckpt=stat_mean_only; commit=auto ;;
        vartv) ckpt=stat_nonmean_only; commit=auto ;;
        *) echo "bad ours arm: $name" >&2; exit 2 ;;
      esac
      OURS_PANEL32_CKPT="$R/ckpts/$ckpt/phase1_step0001000.pt.lean.pt" \
      OURS_PANEL32_OUT="$PANEL_ROOT/fleet30s/ours_$name" \
      OURS_PANEL32_TAG="$name" \
      OURS_PANEL32_MODEL_LABEL="ARRWM DMD $name" \
      OURS_PANEL32_CARN_COMMIT="$commit" \
      OURS_PANEL32_LOG="$PANEL_ROOT/logs/ours_$name/$action" \
        bash "$A/sbatch/u6qf/ours_panel32_holder_action.sh" \
          "$action" "$MANIFEST" "$PROVENANCE" "$BUNDLES"
      ;;
    ode)
      test -n "$name" && test -n "$action"
      bash "$A/sbatch/u6qf/ode_panel32_holder_action.sh" \
        "$name" "$action" "$MANIFEST" "$PROVENANCE" "$BUNDLES"
      ;;
    smoke)
      test "$name" = arrwm
      bash "$A/sbatch/u6qf/panel32_arrwm_smoke.sh"
      ;;
    *)
      echo "bad task kind: $kind (task=$task)" >&2
      exit 2
      ;;
  esac
  echo "PANEL32_SEQUENCE_DONE task=$task holder=${SLURM_JOB_ID:-none} $(date -Is)"
done
