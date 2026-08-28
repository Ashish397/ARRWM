#!/bin/bash
# Launch the 2026-08-27 GAN sample-budget arms inside an existing 2-node
# holder.  Nothing is submitted by this file: it is intended to be called by
# the holder's command file after the allocation becomes RUNNING.
#
#   HOLDER=615xxxx GAN_ARM=vgg_5x PORTOFF=27081 \
#       bash sbatch/run_gan_crop_arm_on_holder.sh
#
# The source launchers contain a final, hard-coded pixel block *after*
# DEXTRA.  Consequently DEXTRA cannot change the knobs these arms need.  This
# launcher inserts one validated LAST-WINS block immediately before run_name;
# the generated block and its source launcher are printed into the run log.
set -euo pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

T0_ACTIVE_ARM=vgg_surrogate_directgrad_currentteacher_8targets_t0rungs_active
PIXCOND_T0_ACTIVE_ARM=vgg_surrogate_directgrad_currentteacher_8targets_t0rungs_rgbmaxmin_active
GATE_V2_ARM=vgg_surrogate_directgrad_gatev2_t0rungs
DECODER_SHAPED_ARM=vgg_decoder_shaped_exactmidlow12_t0rungs
: "${GAN_ARM:?set GAN_ARM=aligned_discrimination_capture|vgg_5x|rn50_5x|vgg_surrogate_teacher_d5x_targets4perclass_fit24|vgg_surrogate_directgrad_teacher_d5x_targets4perclass_fit24|vgg_surrogate_directgrad_currentteacher_8targets_refresh_fit24|vgg_surrogate_directgrad_currentteacher_8targets_refresh_fit24_alignedflash|vgg_surrogate_directgrad_currentteacher_16targets_refresh_fit24_liveflash|$T0_ACTIVE_ARM|$PIXCOND_T0_ACTIVE_ARM|$GATE_V2_ARM|$DECODER_SHAPED_ARM}"
PORTOFF=${PORTOFF:-27081}
RUNSTAMP=${RUNSTAMP:-$(date +%H%M%S)}
MAXSTEPS=${MAXSTEPS:-200}
SAMPLE_EVERY=${SAMPLE_EVERY:-15}
SURROGATE_STAGE=${SURROGATE_STAGE:-calibrate}

# One discriminator update over K=2,F=5 has exactly the same number of
# per-sample logits per training step as the parent K=1,F=2,updates=5:
#     2*5*1 == 1*2*5 == 10.
# L=3 adds temporal diversity; split=1 caps the graph-on VAE transient; x
# stratification makes the two spatial draws non-overlapping in origin band.
BALANCED_5X="gan_updates_per_step=1 ladd_pixel_crops_per_row=2 ladd_pixel_lat_frames=3 ladd_pixel_frames_per_crop=5 ladd_pixel_decode_split=1 ladd_pixel_crop_stratify_x=1 ladd_pixel_crop_y_lo_frac=0.0 ladd_pixel_crop_y_hi_frac=1.0"
# The detached D pass can afford the fivefold evidence above.  The graph-on
# generator route keeps the proven parent geometry; full D5x/G5x OOMed in the
# ordinary checkpointed VAE backward with both split=2 and split=1.
DIRECT_G1X="ladd_pixel_g_crops_per_row=1 ladd_pixel_g_lat_frames=2 ladd_pixel_g_frames_per_crop=2 ladd_pixel_g_decode_split=0 ladd_pixel_g_crop_stratify_x=0 ladd_pixel_post_d_memory_release=true"

case "$GAN_ARM" in
  aligned_discrimination_capture)
    # Dataset-collection arm for the fair cross-backbone benchmark.  The
    # generator follows the selected CARN-commit base recipe, while every
    # candidate adversarial generator route is exactly zero.  The VGG disc is
    # merely the vehicle that reaches the common post-decode pixel boundary;
    # DINO/VGG/RN50/PixGAN are all evaluated later on these same saved pixels.
    : "${CAPTURE_DIR:?set CAPTURE_DIR for aligned_discrimination_capture}"
    SRC=sbatch/pixvgg_online.sbatch
    DARM="aligned_discrimination_capture_carncommit_${RUNSTAMP}"
    ARM_KIND=benchmark
    ARM_STAGE=capture
    DEFAULT_TELEM=0
    DEFAULT_TRIPWIRE=0
    CAPTURE_MAX=${CAPTURE_MAX:-12}
    GAN_ARM_EXTRA="$BALANCED_5X $DIRECT_G1X ladd_disc_loss_weight=0.0 surrogate_critic_enabled=false pix_gan_weight=0.0 gan_disc_start_step=0 gan_critic_warmup_steps=0 ladd_disc_micro_batch_groups=1 ladd_r1_num_samples=0 reverse_noiser_dedrift_enabled=true reverse_noiser_dedrift_apply_to_commit=true reverse_noiser_dedrift_level=1 reverse_noiser_dedrift_min_level=1 reverse_noiser_dedrift_steps=1 reverse_noiser_dedrift_alpha0=1.0 reverse_noiser_dedrift_alpha_decay=0.5 ladd_discrimination_capture_dir=$CAPTURE_DIR ladd_discrimination_capture_every=1 ladd_discrimination_capture_max_records_per_rank=$CAPTURE_MAX"
    ;;
  vgg_5x)
    SRC=sbatch/pixvgg_online.sbatch
    DARM="pixvgg_direct_D5x_G1x_${RUNSTAMP}"
    ARM_KIND=direct
    ARM_STAGE=direct
    DEFAULT_TELEM=0
    DEFAULT_TRIPWIRE=0
    GAN_ARM_EXTRA="$BALANCED_5X $DIRECT_G1X surrogate_critic_enabled=false ladd_disc_loss_weight=1.0"
    ;;
  rn50_5x)
    SRC=sbatch/pixrn50_online.sbatch
    DARM="pixrn50_direct_D5x_G1x_${RUNSTAMP}"
    ARM_KIND=direct
    ARM_STAGE=direct
    DEFAULT_TELEM=0
    DEFAULT_TRIPWIRE=0
    GAN_ARM_EXTRA="$BALANCED_5X $DIRECT_G1X surrogate_critic_enabled=false ladd_disc_loss_weight=1.0"
    ;;
  $DECODER_SHAPED_ARM)
    SRC=sbatch/pixvgg_online.sbatch
    ARM_KIND=decoder_shaped
    ARM_STAGE=$SURROGATE_STAGE
    DEFAULT_TELEM=1
    DEFAULT_TRIPWIRE=0
    BUNDLE=${DECODER_PULLBACK_BUNDLE:-/scratch/u6ex/as1748.u6ex/ARRWM_data/gan_aligned_discrimination/surrogate_field_2708/local_vjp_granular_audit_r0_h6158256/exact_residual_ladder_2808/decoder_shaped_all_residual_seed0.pt}
    [ -f "$BUNDLE" ] || {
      echo "[GAN-CROP] FATAL: decoder pullback bundle missing: $BUNDLE" >&2
      exit 18
    }
    # An exact audit cannot coexist with the full DMD generator graph: the
    # exact WAN VJP alone peaks at ~11.1 GiB and the live trainer has less
    # than 1 GiB headroom after the hybrid route.  The bundle is instead
    # gated by the ride-disjoint + strict Cartesian offline audits.  Keeping
    # this zero is a runtime requirement, not a relaxation of that gate.
    DECODER_COMMON="$BALANCED_5X ladd_disc_loss_weight=0.0 surrogate_critic_enabled=false surrogate_decoder_shaped_enabled=true surrogate_decoder_shaped_bundle=$BUNDLE surrogate_decoder_shaped_audit_every=0 surrogate_teacher_backbone=ladd_pixel pix_r1_gamma=0.0 pix_crop_lat=[24,32] pix_lat_frames_per_crop=3 gan_disc_start_step=0 gan_critic_warmup_steps=0 log_interval=5 flash_dmd_enabled=false flash_dmd_gan_t=0 pix_finish_grad_enabled=true exit_exclude_last_rung=true pix_flash_grad_select_enabled=false ladd_fake_sample_source=dmd ladd_disc_force_clean=true gen_aux_losses_x0_source=ladder_endpoint forward_noiser_train_source=ladder_endpoint forward_noiser_rollout2_source=ladder_endpoint"
    case "$SURROGATE_STAGE" in
      calibrate)
        if [ -n "${PIXW:-}" ] && [ "${PIXW}" != "0" ] && [ "${PIXW}" != "0.0" ]; then
          echo "[GAN-CROP] FATAL: decoder calibration requires PIXW=0.0; got PIXW=${PIXW}" >&2
          exit 19
        fi
        PIXW=0.0
        DARM="pixvgg_decoderpullback_currentpixel_allresidual_q1p99_t0rungs_noflash_cal_w0_${RUNSTAMP}"
        GAN_ARM_EXTRA="$DECODER_COMMON pix_gan_weight=0.0"
        ;;
      active)
        : "${SURROGATE_APPROVED:?set SURROGATE_APPROVED=YES after reviewing calibration}"
        : "${SURROGATE_CALIBRATION_RUN:?name the reviewed decoder calibration run}"
        : "${SURROGATE_Q1_MIN:?set the reviewed strict Cartesian q1}"
        : "${PIXW:?set PIXW from the unweighted shared-gradient ratio}"
        [ "$SURROGATE_APPROVED" = "YES" ] || exit 20
        awk -v x="$SURROGATE_Q1_MIN" 'BEGIN { exit !(x+0 >= 0.99) }' || {
          echo "[GAN-CROP] FATAL: all-residual decoder strict Cartesian q1 ${SURROGATE_Q1_MIN} is below 0.99" >&2
          exit 21
        }
        awk -v x="$PIXW" 'BEGIN { exit !(x+0 > 0) }' || {
          echo "[GAN-CROP] FATAL: decoder active stage requires PIXW > 0" >&2
          exit 22
        }
        PIXW_TAG=${PIXW//./p}
        DARM="pixvgg_decoderpullback_currentpixel_allresidual_q1p99_t0rungs_noflash_active_w${PIXW_TAG}_${RUNSTAMP}"
        GAN_ARM_EXTRA="$DECODER_COMMON pix_gan_weight=${PIXW}"
        ;;
      *)
        echo "[GAN-CROP] FATAL: SURROGATE_STAGE=$SURROGATE_STAGE; use calibrate|active" >&2
        exit 23
        ;;
    esac
    ;;
  vgg_surrogate_teacher_d5x_targets4perclass_fit24|vgg_surrogate_directgrad_teacher_d5x_targets4perclass_fit24|vgg_surrogate_directgrad_currentteacher_8targets_refresh_fit24|vgg_surrogate_directgrad_currentteacher_8targets_refresh_fit24_alignedflash|vgg_surrogate_directgrad_currentteacher_8targets_refresh_fit24_liveflash|vgg_surrogate_directgrad_currentteacher_16targets_refresh_fit24_liveflash|$T0_ACTIVE_ARM|$PIXCOND_T0_ACTIVE_ARM|$GATE_V2_ARM)
    SRC=sbatch/pixvgg_online.sbatch
    ARM_KIND=surrogate
    ARM_STAGE=$SURROGATE_STAGE
    DEFAULT_TELEM=25
    DEFAULT_TRIPWIRE=0

    # The surrogate teacher still pays a graph-on VAE decode on refresh
    # steps.  It amortises that graph; it does not eliminate teacher decode
    # gradients.  Four crop windows keeps the proven DINO distillation
    # geometry while the shallower VGG field gets 24 fitting substeps.
    SURROGATE_MODE_EXTRA=""
    SURROGATE_NAME="teacherD5x_targets4perclass_fit24"
    SURROGATE_CROPS_PER_STEP=4
    DYNAMICS_OVERRIDE_ARM=0
    if [ "$GAN_ARM" = "vgg_surrogate_directgrad_teacher_d5x_targets4perclass_fit24" ] || [ "$GAN_ARM" = "vgg_surrogate_directgrad_currentteacher_8targets_refresh_fit24" ] || [ "$GAN_ARM" = "vgg_surrogate_directgrad_currentteacher_8targets_refresh_fit24_alignedflash" ] || [ "$GAN_ARM" = "vgg_surrogate_directgrad_currentteacher_8targets_refresh_fit24_liveflash" ] || [ "$GAN_ARM" = "vgg_surrogate_directgrad_currentteacher_16targets_refresh_fit24_liveflash" ] || [ "$GAN_ARM" = "$T0_ACTIVE_ARM" ] || [ "$GAN_ARM" = "$PIXCOND_T0_ACTIVE_ARM" ] || [ "$GAN_ARM" = "$GATE_V2_ARM" ]; then
      # First-order synthetic-gradient student. It predicts the normalized
      # teacher field directly; the teacher-gradient RMS EMA restores the
      # physical scale at generator consumption. Unlike the scalar-potential
      # control, fitting this arm does not require a double backward.
      SURROGATE_MODE_EXTRA="surrogate_gradient_mode=direct surrogate_direct_width=96 surrogate_direct_num_blocks=6 surrogate_direct_head_init_std=1.0e-3 surrogate_direct_teacher_rms_beta=0.95 surrogate_teacher_target_microbatch=1"
      SURROGATE_NAME="directgrad_teacherD5x_targets4perclass_fit24"
    fi
    if [ "$GAN_ARM" = "vgg_surrogate_directgrad_currentteacher_8targets_refresh_fit24" ] || [ "$GAN_ARM" = "vgg_surrogate_directgrad_currentteacher_8targets_refresh_fit24_alignedflash" ] || [ "$GAN_ARM" = "vgg_surrogate_directgrad_currentteacher_8targets_refresh_fit24_liveflash" ] || [ "$GAN_ARM" = "vgg_surrogate_directgrad_currentteacher_16targets_refresh_fit24_liveflash" ] || [ "$GAN_ARM" = "$T0_ACTIVE_ARM" ] || [ "$GAN_ARM" = "$PIXCOND_T0_ACTIVE_ARM" ] || [ "$GAN_ARM" = "$GATE_V2_ARM" ]; then
      # Current real and fake teacher-gradient targets only. Capacity one per
      # class prevents the 24 fitting substeps from mixing fields labelled by
      # older versions of the online discriminator. The matched-norm tangent
      # control is sparse telemetry only and is never applied to the generator.
      SURROGATE_MODE_EXTRA="$SURROGATE_MODE_EXTRA surrogate_cache_capacity=1 surrogate_param_control_enabled=true"
      SURROGATE_NAME="directgrad_currentteacher_8targets_refresh_fit24"
    fi
    if [ "$GAN_ARM" = "vgg_surrogate_directgrad_currentteacher_8targets_refresh_fit24_alignedflash" ]; then
      # Parent DMD source is the scientifically useful control. This arm
      # aligns the trained teacher's fake distribution with the flash t=60
      # fake on which teacher targets are queried and generator guidance is
      # measured. Still weight zero.
      SURROGATE_MODE_EXTRA="$SURROGATE_MODE_EXTRA ladd_fake_sample_source=flash"
      SURROGATE_NAME="directgrad_currentteacher_8targets_refresh_fit24_alignedflash"
    fi
    if [ "$GAN_ARM" = "vgg_surrogate_directgrad_currentteacher_8targets_refresh_fit24_liveflash" ] || [ "$GAN_ARM" = "vgg_surrogate_directgrad_currentteacher_16targets_refresh_fit24_liveflash" ]; then
      # The flash slab is a mixed CopySlices buffer: earlier block groups
      # carry generator graphs while the deliberately skipped trailing group
      # does not. Select the latest contiguous mask-live frames instead of
      # the historical (and, for multi-block calls, detached) final frames.
      SURROGATE_MODE_EXTRA="$SURROGATE_MODE_EXTRA pix_flash_grad_select_enabled=true"
      SURROGATE_NAME="directgrad_currentteacher_8targets_refresh_fit24_liveflash"
    fi
    if [ "$GAN_ARM" = "vgg_surrogate_directgrad_currentteacher_16targets_refresh_fit24_liveflash" ]; then
      # Sixteen logical targets per refresh: eight current reals plus eight
      # current fakes. Target graphs remain serialized at microbatch one, so
      # this doubles independent field evidence rather than peak graph width.
      SURROGATE_CROPS_PER_STEP=8
      SURROGATE_NAME="directgrad_currentteacher_16targets_refresh_fit24_liveflash"
    fi
    if [ "$GAN_ARM" = "$T0_ACTIVE_ARM" ] || [ "$GAN_ARM" = "$PIXCOND_T0_ACTIVE_ARM" ] || [ "$GAN_ARM" = "$GATE_V2_ARM" ]; then
      # Researcher-authorised dynamics arm (2026-08-27): remove the extra
      # t=60 Flash-DMD forward and query the surrogate on graph-live final
      # ladder x0 estimates instead.  Keep the simpler four-real/four-fake
      # target budget; the 16-target arm added overlapping crops rather than
      # independent rides and did not improve the field.  Every former
      # Flash-facing auxiliary consumer is made explicit: LADD keeps the
      # DMD-scored clean-x0 band, while the frozen action critic and latent
      # surrogate consume the final ladder endpoint.
      if [ "$GAN_ARM" != "$GATE_V2_ARM" ]; then
        [ "$SURROGATE_STAGE" = "active" ] || {
          echo "[GAN-CROP] FATAL: $GAN_ARM requires SURROGATE_STAGE=active" >&2
          exit 15
        }
      fi
      DYNAMICS_OVERRIDE_ARM=1
      SURROGATE_MODE_EXTRA="$SURROGATE_MODE_EXTRA flash_dmd_enabled=false flash_dmd_gan_t=0 pix_finish_grad_enabled=true exit_exclude_last_rung=true pix_flash_grad_select_enabled=false ladd_fake_sample_source=dmd ladd_disc_force_clean=true gen_aux_losses_x0_source=ladder_endpoint forward_noiser_train_source=ladder_endpoint forward_noiser_rollout2_source=ladder_endpoint"
      SURROGATE_NAME="directgrad_currentteacher_8targets_t0rungs_noflash"
    fi
    if [ "$GAN_ARM" = "$GATE_V2_ARM" ]; then
      # Candidate selected for the honest feature-field screen: fit only the
      # fake field consumed by G, optimise direction rather than discarded
      # raw magnitude, and expose the student to the VAE's temporal coupling
      # plus the VGG head's clip-global pooled statistics. The 1e-3 Adam LR
      # is the measured fast-tracking setting; exact per-crop origins and the
      # distinct live audit stream are enforced in the shared implementation.
      SURROGATE_MODE_EXTRA="$SURROGATE_MODE_EXTRA surrogate_critic_lr=1.0e-3 surrogate_direct_loss_mode=cosine surrogate_direct_real_loss_weight=0.0 surrogate_direct_fake_loss_weight=1.0 surrogate_direct_temporal_mixing=true surrogate_direct_temporal_blocks=2 surrogate_direct_global_context=true"
      SURROGATE_NAME="gatev2_temporalglobal_cosine_fakeonly_lr1e3_t0rungs_noflash"
      # Unlike the two historical researcher-override arms, gate-v2 exists
      # specifically to earn the honest >=0.50 authorization. Never inherit
      # the t0 block's bypass merely because it shares the no-Flash source.
      DYNAMICS_OVERRIDE_ARM=0
    fi
    if [ "$GAN_ARM" = "$PIXCOND_T0_ACTIVE_ARM" ]; then
      # Detached rendered-state conditioning. Each latent cell receives RGB
      # maxima and minima from its exact four decoded frames and spatial
      # pixel footprint (six channels total). The VAE decode is no_grad and
      # batch-serialized; only the synthetic field remains a generator
      # gradient. This arm otherwise matches the no-Flash t=0-rung arm.
      SURROGATE_MODE_EXTRA="$SURROGATE_MODE_EXTRA surrogate_pixel_condition_enabled=true surrogate_pixel_condition_decode_batch=1"
      SURROGATE_NAME="directgrad_currentteacher_8targets_t0rungs_noflash_rgbmaxmin"
    fi
    SURROGATE_COMMON="$BALANCED_5X ladd_disc_loss_weight=0.0 surrogate_critic_enabled=true surrogate_teacher_backbone=ladd_pixel surrogate_critic_lr=2.0e-4 surrogate_value_loss_weight=1.0 surrogate_grad_loss_weight=1.0 surrogate_grad_loss_normalize=true surrogate_grad_check_every=20 surrogate_distill_substeps=24 surrogate_critic_head_init_std=0.02 pix_teacher_refresh_every=2 surrogate_cache_capacity=8 surrogate_teacher_use_checkpoint=true pix_crop_lat=[24,32] pix_crops_per_step=$SURROGATE_CROPS_PER_STEP pix_frames_per_crop=2 pix_lat_frames_per_crop=2 pix_decode_border_trim=8 pix_decode_batch=4 pix_real_pool_windows=2560 pix_real_pool_refresh=8 pix_real_pool_warm_updates=32 pix_r1_gamma=0.0 $SURROGATE_MODE_EXTRA"

    case "$SURROGATE_STAGE" in
      calibrate)
        # A calibration run may measure an unweighted generator gradient but
        # must never apply it.  Refuse even an accidentally inherited weight.
        if [ -n "${PIXW:-}" ] && [ "${PIXW}" != "0" ] && [ "${PIXW}" != "0.0" ]; then
          echo "[GAN-CROP] FATAL: calibration requires PIXW=0.0; got PIXW=${PIXW}" >&2
          exit 4
        fi
        PIXW=0.0
        DARM="pixvgg_surrogate_${SURROGATE_NAME}_cal_w0_${RUNSTAMP}"
        GAN_ARM_EXTRA="$SURROGATE_COMMON pix_gan_weight=0.0"
        ;;
      active)
        # Activation is deliberately inconvenient.  The previous DINO
        # surrogate reached value corr=0.803 but held-out grad cosine=0.0075;
        # a non-zero weight before checking the VGG field would be an invalid
        # experiment, not an optimistic one.
        : "${SURROGATE_APPROVED:?set SURROGATE_APPROVED=YES after reviewing calibration}"
        : "${SURROGATE_CALIBRATION_RUN:?name the reviewed calibration run}"
        : "${PIXW:?set PIXW from 0.10 / surrogate_*_grad_ratio_unweighted}"
        [ "$SURROGATE_APPROVED" = "YES" ] || {
          echo "[GAN-CROP] FATAL: SURROGATE_APPROVED must be literal YES" >&2
          exit 5
        }
        if [ "$GAN_ARM" = "$GATE_V2_ARM" ]; then
          : "${SURROGATE_Q1_MIN:?set the minimum reviewed unseen-crop sample-q1 cosine}"
          : "${SURROGATE_AUDIT_COUNT:?set the number of reviewed unseen-crop audits}"
          awk -v x="$SURROGATE_AUDIT_COUNT" 'BEGIN { exit !(x+0 >= 2) }' || {
            echo "[GAN-CROP] FATAL: gate-v2 requires at least two reviewed live audits; got ${SURROGATE_AUDIT_COUNT}" >&2
            exit 17
          }
          awk -v x="$SURROGATE_Q1_MIN" 'BEGIN { exit !(x+0 >= 0.50) }' || {
            echo "[GAN-CROP] FATAL: minimum per-sample q1 cosine ${SURROGATE_Q1_MIN} is below activation gate 0.50" >&2
            exit 6
          }
        elif [ "$DYNAMICS_OVERRIDE_ARM" -eq 1 ]; then
          : "${SURROGATE_COS:?set the reviewed held-out gradient cosine}"
          [ "${SURROGATE_DYNAMICS_OVERRIDE:-}" = "YES" ] || {
            echo "[GAN-CROP] FATAL: $GAN_ARM requires literal SURROGATE_DYNAMICS_OVERRIDE=YES" >&2
            exit 16
          }
          echo "[GAN-CROP] RESEARCH OVERRIDE: activating below the historical cosine gate (recorded cosine=${SURROGATE_COS}); this tests coupled training dynamics"
        else
          : "${SURROGATE_COS:?set the reviewed held-out gradient cosine}"
          awk -v x="$SURROGATE_COS" 'BEGIN { exit !(x+0 >= 0.50) }' || {
            echo "[GAN-CROP] FATAL: held-out cosine ${SURROGATE_COS} is below activation gate 0.50" >&2
            exit 6
          }
        fi
        awk -v x="$PIXW" 'BEGIN { exit !(x+0 > 0) }' || {
          echo "[GAN-CROP] FATAL: active stage requires PIXW > 0; got ${PIXW}" >&2
          exit 7
        }
        PIXW_TAG=${PIXW//./p}
        ACTIVE_TAG=active
        [ "$DYNAMICS_OVERRIDE_ARM" -eq 1 ] && ACTIVE_TAG=active_researchoverride
        DARM="pixvgg_surrogate_${SURROGATE_NAME}_${ACTIVE_TAG}_w${PIXW_TAG}_${RUNSTAMP}"
        GAN_ARM_EXTRA="$SURROGATE_COMMON pix_gan_weight=${PIXW}"
        ;;
      *)
        echo "[GAN-CROP] FATAL: SURROGATE_STAGE=$SURROGATE_STAGE; use calibrate|active" >&2
        exit 8
        ;;
    esac
    ;;
  *)
    echo "[GAN-CROP] FATAL: unknown GAN_ARM=$GAN_ARM" >&2
    exit 2
    ;;
esac

# The original direct 5x graph was the tight path. Attempt 1 additionally ran the
# parent's every-step parameter-gradient probe (two extra full DiT backwards)
# and OOMed. Attempt 2 removed it but showed that decode_split=2 still OOMs
# during checkpointed VAE recomputation on the ordinary backward; split=1
# still failed at the same boundary. Direct arms now route the
# detached D update through D5x and the graph-on generator guidance through the
# proven G1x geometry. Optional diagnostics stay off; crop-plan, D-update, loss
# and memory counters remain live. The surrogate calibration keeps a sparse
# parameter probe because its unweighted ratio is required to set PIXW and its
# ordinary G route carries no decoder graph.
TELEM=${TELEM:-$DEFAULT_TELEM}
TRIPWIRE_EVERY=${TRIPWIRE_EVERY:-$DEFAULT_TRIPWIRE}

echo "[GAN-CROP-CONFIG] arm=$GAN_ARM kind=$ARM_KIND stage=$ARM_STAGE"
echo "[GAN-CROP-CONFIG] source=$SRC"
echo "[GAN-CROP-CONFIG] run_name=dmd10k_${DARM}_j<holder>"
echo "[GAN-CROP-CONFIG] last_wins=[$GAN_ARM_EXTRA]"
echo "[GAN-CROP-CONFIG] diagnostics=gan_grad_telemetry_every=$TELEM texture_tripwire_every=$TRIPWIRE_EVERY"

# Used by the static review test.  This resolves every arm without touching
# Slurm, conda, W&B, or a GPU.
if [ "${PRINT_ONLY:-0}" = "1" ]; then
  exit 0
fi

: "${HOLDER:?set HOLDER to a running two-node holder job ID}"
if ! [[ "$MAXSTEPS" =~ ^[1-9][0-9]*$ && "$SAMPLE_EVERY" =~ ^[1-9][0-9]*$ ]]; then
  echo "[GAN-CROP] FATAL: MAXSTEPS and SAMPLE_EVERY must be positive integers" >&2
  exit 13
fi
if [ "$SAMPLE_EVERY" -gt "$MAXSTEPS" ]; then
  echo "[GAN-CROP] FATAL: SAMPLE_EVERY=$SAMPLE_EVERY exceeds MAXSTEPS=$MAXSTEPS; smoke would produce no texture video" >&2
  exit 14
fi

# Refuse to collide with another torchrun in the allocation.  A holder's
# batch/extern steps are ignored; any other live child step is a collision.
LIVE=$(squeue -j "$HOLDER" -h -s -o "%i" 2>/dev/null \
  | grep -vE '\.(batch|extern)$' | wc -l) || true
if [ "${LIVE:-0}" -gt 0 ]; then
  echo "[GAN-CROP] FATAL: holder $HOLDER already has $LIVE live child step(s)" >&2
  exit 9
fi

NODE_EXPR=$(squeue -j "$HOLDER" -h -o %N)
if [ -z "$NODE_EXPR" ] || [ "$NODE_EXPR" = "(null)" ]; then
  echo "[GAN-CROP] FATAL: holder $HOLDER is not RUNNING or has no nodes" >&2
  exit 10
fi
NODES=$(scontrol show hostnames "$NODE_EXPR")
NNODE=$(echo "$NODES" | wc -l)
if [ "$NNODE" -ne 2 ]; then
  echo "[GAN-CROP] FATAL: expected a 2-node holder; $HOLDER has $NNODE" >&2
  exit 11
fi
NODELIST=$(echo "$NODES" | paste -sd,)
MASTER_ADDR=$(echo "$NODES" | head -1)
MASTER_PORT=$((29500 + (HOLDER + PORTOFF) % 16000))

export GAN_ARM_EXTRA
TMP_INJECT=$(mktemp "/tmp/gancrop_${GAN_ARM}_${HOLDER}_inject.XXXXXX")
TMP_RUN=$(mktemp "/tmp/gancrop_${GAN_ARM}_${HOLDER}_run.XXXXXX")
trap 'rm -f "$TMP_INJECT" "$TMP_RUN"' EXIT

# Exactly one run_name assignment exists in each reviewed source launcher.
# Inserting immediately above it makes every arm override last-wins without
# relying on the parents' earlier DEXTRA site.
N_RUN_NAME=$(grep -c '^[[:space:]]*run_name=dmd10k_' "$SRC")
if [ "$N_RUN_NAME" -ne 1 ]; then
  echo "[GAN-CROP] FATAL: expected one run_name injection site in $SRC; got $N_RUN_NAME" >&2
  exit 12
fi
awk '
  /^[[:space:]]*run_name=dmd10k_/ { print "    " ENVIRON["GAN_ARM_EXTRA"] " \\" }
  { print }
' "$SRC" > "$TMP_INJECT"

sed -e 's/^srun torchrun \\/srun --jobid='"$HOLDER"' --overlap --nodelist='"$NODELIST"' --nodes='"$NNODE"' --ntasks-per-node=1 --gpus-per-node=4 --gpu-bind=none torchrun \\/' \
    -e 's/--nnodes=\$SLURM_NNODES/--nnodes='"$NNODE"'/' \
    -e 's/--rdzv_id=\$SLURM_JOB_ID/--rdzv_id='"$HOLDER$PORTOFF"'/' \
    -e 's|--rdzv_endpoint=\${MASTER_ADDR}:\${MASTER_PORT}|--rdzv_endpoint='"$MASTER_ADDR:$MASTER_PORT"'|' \
    -e 's/^#SBATCH.*//' \
    "$TMP_INJECT" > "$TMP_RUN"

LOG="logs/gan_crop_${GAN_ARM}_${ARM_STAGE}_h${HOLDER}_${RUNSTAMP}.log"
echo "[GAN-CROP] holder=$HOLDER nodes=$NODELIST port=$MASTER_PORT steps=$MAXSTEPS"
echo "[GAN-CROP] log=$LOG sample_interval=$SAMPLE_EVERY"

# Persist the authoritative injected block in the experiment's own log.  The
# source launcher's historical PIXVGG/PIXRN echo describes its parent defaults
# and is necessarily stale after a last-wins injection; proof-of-fire counters
# below remain the authority for realised runtime values.
{
  echo "[GAN-CROP-CONFIG] arm=$GAN_ARM kind=$ARM_KIND stage=$ARM_STAGE"
  echo "[GAN-CROP-CONFIG] source=$SRC"
  echo "[GAN-CROP-CONFIG] run_name=dmd10k_${DARM}_j${HOLDER}"
  echo "[GAN-CROP-CONFIG] last_wins=[$GAN_ARM_EXTRA]"
  echo "[GAN-CROP-CONFIG] diagnostics=gan_grad_telemetry_every=$TELEM texture_tripwire_every=$TRIPWIRE_EVERY"
} > "$LOG"

set +e
SLURM_JOB_ID=$HOLDER SLURM_NNODES=$NNODE SLURM_JOB_NODELIST=$NODELIST \
  DARM=$DARM MAXSTEPS=$MAXSTEPS CKPT_EVERY=${CKPT_EVERY:-50} TELEM=$TELEM \
  PIXW=${PIXW:-0.0} \
  DEXTRA="memory_audit_enabled=true sample_interval=$SAMPLE_EVERY texture_tripwire_every=$TRIPWIRE_EVERY ${DEXTRA:-}" \
  bash "$TMP_RUN" >> "$LOG" 2>&1
RC=$?
set -e

echo "[GAN-CROP] training exit=$RC log=$LOG"
if [ "$RC" -eq 0 ] && [ "$GAN_ARM" != "aligned_discrimination_capture" ]; then
  bash sbatch/check_gan_crop_arm.sh "$LOG" "$GAN_ARM" "$ARM_STAGE" || RC=$?
fi
exit "$RC"
