#!/bin/bash
# Read proof-of-fire metrics for a run produced by
# run_gan_crop_arm_on_holder.sh.  W&B-only metrics are read from
# wandb-summary.json; ABSENT is never treated as zero.
set -euo pipefail

LOG=${1:?usage: check_gan_crop_arm.sh <log> <arm> [stage] [wandb-run-dir]}
ARM=${2:?set arm name from run_gan_crop_arm_on_holder.sh}
STAGE=${3:-calibrate}
WBDIR=${4:-}
test -f "$LOG" || { echo "no such log: $LOG" >&2; exit 2; }

if [ -z "$WBDIR" ]; then
  RID=$(grep -oE 'run-[0-9]{8}_[0-9]{6}-[a-z0-9]+' "$LOG" | head -1 || true)
  [ -n "$RID" ] && WBDIR="wandb/wandb/$RID"
fi
if [ -z "$WBDIR" ] || [ ! -d "$WBDIR" ]; then
  echo "FATAL: could not identify this run's W&B directory from $LOG" >&2
  exit 2
fi

python3 - "$LOG" "$WBDIR" "$ARM" "$STAGE" <<'PY'
import json
import math
import os
import re
import sys

log_path, wbdir, arm, stage = sys.argv[1:]
summary_path = os.path.join(wbdir, "files", "wandb-summary.json")
if not os.path.exists(summary_path):
    raise SystemExit(f"FATAL: no summary at {summary_path}")
with open(summary_path, encoding="utf-8") as fh:
    summary = json.load(fh)
with open(log_path, errors="replace") as fh:
    log = fh.read()

def get(key):
    for candidate in (f"gen/train/{key}", f"train/{key}", f"gen/{key}", key):
        if candidate in summary:
            return summary[candidate], candidate
    return None, None

passed = failed = warned = 0

def emit(state, label, key, value, note=""):
    global passed, failed, warned
    if state == "PASS":
        passed += 1
    elif state == "WARN":
        warned += 1
    else:
        failed += 1
    shown = "ABSENT" if value is None else repr(value)
    print(f"{state:4} {label}: {key}={shown}" + (f" -- {note}" if note else ""))

def check(label, key, pred, note="", warn=False):
    value, actual = get(key)
    if value is None:
        emit("WARN" if warn else "FAIL", label, actual or f"train/{key}", None,
             "key absent; absent is not zero")
        return None
    try:
        ok = bool(pred(value))
    except Exception:
        ok = False
    emit("PASS" if ok else ("WARN" if warn else "FAIL"), label,
         actual, value, note)
    return value

print("=== GAN CROP ARM CONNECTION REVIEW ===")
print(f"arm={arm} stage={stage} log={log_path}")
print(f"summary={summary_path} wandb_step={summary.get('_step')}")

for label, pattern in (
    ("feature source built", r"\[LADD-PIXFEAT\] built feature source:"),
    ("decode callback installed", r"\[LADD-PIXFEAT\] decode callback INSTALLED:"),
):
    found = bool(re.search(pattern, log))
    emit("PASS" if found else "FAIL", label, os.path.basename(log_path), found)

check("pixel branch ran", "ladd_pix_disc_forwards", lambda x: x > 0)
check("encoder ran", "ladd_pix_src_forwards", lambda x: x > 0)
check("WAN projector stayed out", "ladd_pix_wan_projector_calls", lambda x: x == 0)
check("pretrained basis", "ladd_pix_vgg_pretrained", lambda x: x == 1)
check("pooled readout installed", "ladd_pix_pooled_readout", lambda x: x == 1)
check("pooled readout ran", "ladd_pix_pooled_calls", lambda x: x > 0)
check("dense head stayed out", "ladd_pix_dense_head_calls", lambda x: x == 0)
prefix = "d_" if arm in ("vgg_5x", "rn50_5x") else ""
check("fivefold D logits realised", f"ladd_pix_{prefix}logits_per_sample",
      lambda x: x == 10, "must be K=2 x frames=5")
check("D K=2 realised", f"ladd_pix_{prefix}crops_per_row_used", lambda x: x == 2)
check("D L=3 realised", f"ladd_pix_{prefix}lat_frames_used", lambda x: x == 3)
check("D five frames realised", f"ladd_pix_{prefix}frames_per_crop_used", lambda x: x == 5)
check("D latent request not clamped", f"ladd_pix_{prefix}lat_frames_clamped", lambda x: x == 0)
check("D frame request not clamped", f"ladd_pix_{prefix}frames_per_crop_clamped", lambda x: x == 0)
check("D decode split fired", f"ladd_pix_{prefix}decode_split_active", lambda x: x == 1)
check("D horizontal stratification fired", f"ladd_pix_{prefix}crop_stratify_active", lambda x: x == 1)
check("D lower-row bias stayed off", f"ladd_pix_{prefix}crop_band_active", lambda x: x == 0)
expected_dim = 1040 if arm == "rn50_5x" else 1440
expected_taps = 1 if arm == "rn50_5x" else 2
check("statistic width", "ladd_pix_stat_dim", lambda x: x == expected_dim)
check("tap count", "ladd_pix_n_taps", lambda x: x == expected_taps)
check("deferred D closure flushed", "ladd_disc_deferred_flush_events", lambda x: x > 0)
check("GT-vs-fake pairs built", "ladd_n_pairs_gt", lambda x: x > 0)
check("D not skipped", "r3gan_disc_skipped_gt", lambda x: x == 0)

if arm in ("vgg_5x", "rn50_5x"):
    check("D detached decoder ran", "ladd_pix_d_decode_nograd", lambda x: x > 0)
    check("G decoder gradient ran", "ladd_pix_g_decode_grad", lambda x: x > 0)
    check("G parent logits realised", "ladd_pix_g_logits_per_sample", lambda x: x == 2)
    check("G K=1 realised", "ladd_pix_g_crops_per_row_used", lambda x: x == 1)
    check("G L=2 realised", "ladd_pix_g_lat_frames_used", lambda x: x == 2)
    check("G two frames realised", "ladd_pix_g_frames_per_crop_used", lambda x: x == 2)
    check("G split stayed off", "ladd_pix_g_decode_split_active", lambda x: x == 0)
    check("G stratification stayed off", "ladd_pix_g_crop_stratify_active", lambda x: x == 0)
    check("post-D allocator release fired", "ladd_pix_post_d_release_events", lambda x: x > 0)
elif "decoder_shaped" in arm:
    check("direct decoder gradient stayed off", "ladd_pix_decode_grad", lambda x: x == 0)
    check("decoder-shaped route served generator", "surrogate_consumed", lambda x: x == 1)
    check("current teacher used", "surrogate_decoder_current_teacher", lambda x: x == 1)
    check("decoder state detached", "surrogate_decoder_state_detached", lambda x: x == 1)
    check("all 12 residual stages exact", "surrogate_exact_residual_stage_count",
          lambda x: x == 12)
    check("all-residual Q1~0.99 route active", "surrogate_exact_all_residual",
          lambda x: x == 1)
    check("LADD teacher identity active", "surrogate_decoder_teacher_ladd_pixel", lambda x: x == 1)
    check("pixel cotangent nonzero", "surrogate_decoder_pixel_cotangent_rms", lambda x: x > 0)
    # The exact WAN VJP requires another ~11.1 GiB and cannot coexist with
    # the full generator graph.  Transfer is gated offline on ride-disjoint
    # and strict Cartesian banks; live checks prove current-teacher routing,
    # nonzero signal, consumption, and (for the active arm) application.
    exact_audit, exact_audit_key = get("surrogate_decoder_exact_audit")
    emit("PASS" if exact_audit is None else "FAIL",
         "co-resident exact audit stayed disabled",
         exact_audit_key or "train/surrogate_decoder_exact_audit", exact_audit,
         "offline strict Cartesian q1 is the activation gate")
    if stage == "calibrate":
        check("weight-zero probe did not apply G term", "pix_g_applied",
              lambda x: x == 0, warn=True)
        ratio, ratio_key = get("surrogate_param_grad_ratio_unweighted")
        if ratio is None:
            ratio, ratio_key = get("surrogate_grad_ratio_unweighted")
        if ratio is None or not math.isfinite(float(ratio)) or float(ratio) <= 0:
            emit("FAIL", "unweighted calibration ratio available",
                 ratio_key or "train/surrogate_{param_,}grad_ratio_unweighted", ratio)
        else:
            emit("PASS", "unweighted calibration ratio available", ratio_key, ratio,
                 f"candidate PIXW for 10% share = {0.10 / float(ratio):.6g}")
    else:
        check("active decoder-shaped G term applied", "pix_g_applied", lambda x: x == 1)
    if "flash" in arm.lower():
        check("Flash x0 source active", "pix_fake_source_ladder", lambda x: x == 0)
        live = check("Flash buffer has live frames",
                     "pix_flash_grad_frames", lambda x: x > 0)
        total = check("Flash buffer frame count measured",
                      "pix_flash_grad_frames_total", lambda x: x > 0)
        check("Flash liveness selector active",
              "pix_flash_grad_select_active", lambda x: x == 1)
        if live is not None and total is not None:
            emit("PASS" if 0 < float(live) < float(total) else "FAIL",
                 "detached Flash tail excluded",
                 "train/pix_flash_grad_frames[/_total]", (live, total),
                 "live must be a strict subset on the grouped Flash roll")
    else:
        check("ladder x0 source active", "pix_fake_source_ladder", lambda x: x == 1)
else:
    check("direct decoder gradient stayed off", "ladd_pix_decode_grad", lambda x: x == 0,
          "teacher refresh decode gradients bypass this LADD counter")
    check("surrogate distillation ran", "surrogate_distill_ran", lambda x: x == 1)
    check("surrogate served generator route", "surrogate_consumed", lambda x: x == 1)
    check("LADD pixel teacher was queried", "ladd_pix_teacher_calls", lambda x: x > 0)
    check("teacher refreshed", "surrogate_n_teacher_refresh", lambda x: x > 0)
    check("24 fitting substeps reached runtime", "surrogate_distill_substeps", lambda x: x == 24)
    if "gatev2" in arm:
        check("unused real field targets disabled",
              "surrogate_real_targets_disabled", lambda x: x == 1)
        check("fake-only target weight realised",
              "surrogate_real_target_weight", lambda x: x == 0)
    else:
        check("real pool filled", "surrogate_pool_size", lambda x: x > 0)
    empty, empty_key = get("surrogate_pool_EMPTY")
    emit("PASS" if empty is None else "FAIL", "real pool never empty alarmed",
         empty_key or "train/surrogate_pool_EMPTY", empty,
         "absence is the intended healthy regime for this alarm key")
    cosine = check("diagnostic global gradient cosine measured", "surrogate_check_cos_sim",
                   lambda x: math.isfinite(float(x)), warn=True)
    sample_median = check(
        "unseen-crop per-sample median cosine measured",
        "surrogate_check_cos_sample_median",
        lambda x: math.isfinite(float(x)), warn=True,
    )
    sample_q1 = check(
        "unseen-crop per-sample q1 cosine measured",
        "surrogate_check_cos_sample_q1",
        lambda x: math.isfinite(float(x)), warn=True,
    )
    check(
        "unseen-crop audit has multiple samples",
        "surrogate_check_cos_sample_n", lambda x: float(x) > 1, warn=True,
    )
    check(
        "unseen-crop audit aggregated all eight ranks",
        "surrogate_check_distributed_ranks", lambda x: float(x) == 8,
        warn=True,
    )
    rolling_q1 = check(
        "final-two unseen audits have a finite q1 minimum",
        "surrogate_check_cos_sample_q1_rolling2_min",
        lambda x: math.isfinite(float(x)), warn=True,
    )
    rolling_count = check(
        "two unseen audits available for sustained gate",
        "surrogate_check_cos_sample_q1_rolling2_count",
        lambda x: float(x) >= 2, warn=True,
    )
    relerr = check("unseen-crop gradient relative error measured", "surrogate_check_rel_err",
                  lambda x: math.isfinite(float(x)), warn=True)
    check("audit did not reuse fitted tensor batch",
          "surrogate_check_reuses_fit_batch", lambda x: x == 0)
    check("audit used independent crop RNG stream",
          "surrogate_check_distinct_crop_stream", lambda x: x == 1)
    check("audit exact crop origins do not overlap fit origins",
          "surrogate_check_origin_overlap_frac", lambda x: x == 0, warn=True)
    if rolling_q1 is not None and rolling_count is not None:
        gate_pass = float(rolling_q1) >= 0.50 and float(rolling_count) >= 2
        dynamics_override = (
            "t0rungs" in arm and "gatev2" not in arm and stage == "active"
        )
        emit(
            "PASS" if gate_pass else ("WARN" if dynamics_override else "FAIL"),
            "ROBUST LIVE TRANSFER GATE sample-q1 cosine >= 0.50",
            "train/surrogate_check_cos_sample_q1_rolling2_min", rolling_q1,
            (
                "researcher-authorised active dynamics arm; the live crop "
                "probe is disjoint but the ride-disjoint bank remains the gate"
                if dynamics_override else
                "failure means keep PIXW=0; global cosine cannot override it"
            ),
        )
    if stage == "calibrate":
        check("weight-zero probe did not apply G term", "pix_g_applied", lambda x: x == 0,
              warn=True)
        ratio, ratio_key = get("surrogate_param_grad_ratio_unweighted")
        if ratio is None:
            ratio, ratio_key = get("surrogate_grad_ratio_unweighted")
        if ratio is None or not math.isfinite(float(ratio)) or float(ratio) <= 0:
            emit("FAIL", "unweighted calibration ratio available",
                 ratio_key or "train/surrogate_{param_,}grad_ratio_unweighted", ratio)
        else:
            emit("PASS", "unweighted calibration ratio available", ratio_key, ratio,
                 f"candidate PIXW for 10% share = {0.10 / float(ratio):.6g}")
        if "currentteacher" in arm or "gatev2" in arm:
            check("matched-norm tangent control reached parameters",
                  "surrogate_param_control_grad_norm", lambda x: x > 0)
        if "liveflash" in arm:
            check("flash liveness selector active",
                  "pix_flash_grad_select_active", lambda x: x == 1)
            live = check("flash buffer has live frames",
                         "pix_flash_grad_frames", lambda x: x > 0)
            total = check("flash buffer frame count measured",
                          "pix_flash_grad_frames_total", lambda x: x > 0)
            if live is not None and total is not None:
                emit("PASS" if 0 < float(live) < float(total) else "FAIL",
                     "detached flash tail excluded",
                     "train/pix_flash_grad_frames[/_total]",
                     (live, total),
                     "live must be a strict subset on the 9-frame grouped roll")
    else:
        check("active surrogate G term applied", "pix_g_applied", lambda x: x == 1)
        if "t0rungs" in arm:
            check("ladder x0 source active", "pix_fake_source_ladder", lambda x: x == 1)
            live = check("ladder x0 buffer has live frames",
                         "pix_finish_grad_frames", lambda x: x > 0)
            total = check("ladder x0 frame count measured",
                          "pix_finish_grad_frames_total", lambda x: x > 0)
            if live is not None and total is not None:
                emit("PASS" if 0 < float(live) <= float(total) else "FAIL",
                     "ladder x0 liveness mask valid",
                     "train/pix_finish_grad_frames[/_total]", (live, total))
            flash, flash_key = get("pix_flash_grad_select_active")
            emit("PASS" if flash is None else "FAIL",
                 "Flash selector absent", flash_key or
                 "train/pix_flash_grad_select_active", flash,
                 "Flash-DMD is disabled in this arm")
        if "rgbmaxmin" in arm:
            check("detached pixel conditioning active",
                  "surrogate_pixel_condition_active", lambda x: x == 1)
            check("pixel conditioning has RGB max+min channels",
                  "surrogate_pixel_condition_channels", lambda x: x == 6)
            check("pixel conditioning cut from decoder graph",
                  "surrogate_pixel_condition_detached", lambda x: x == 1)
            check("Wan temporal groups aligned 4:1",
                  "surrogate_pixel_condition_temporal_scale", lambda x: x == 4)
            check("Wan spatial rows aligned 8:1",
                  "surrogate_pixel_condition_spatial_scale_h", lambda x: x == 8)
            check("Wan spatial columns aligned 8:1",
                  "surrogate_pixel_condition_spatial_scale_w", lambda x: x == 8)
            check("pixel conditioning VAE decode ran",
                  "surrogate_pixel_condition_decode_calls", lambda x: x > 0)
            check("pixel conditioning projection learned",
                  "surrogate_pixel_condition_weight_norm", lambda x: x > 0)
            check("pixel conditioning projection received gradients",
                  "surrogate_pixel_condition_grad_norm", lambda x: x > 0)
            check("pixel conditioning changes the predicted field",
                  "surrogate_pixel_condition_field_delta_rms", lambda x: x > 0)

crash = re.search(r"Traceback|CUDA out of memory|RuntimeError", log)
emit("PASS" if crash is None else "FAIL", "no crash", os.path.basename(log_path),
     crash.group(0) if crash else "clean")
nan_keys = [k for k, v in summary.items()
            if isinstance(v, float) and math.isnan(v) and "r1" not in k]
emit("PASS" if not nan_keys else "FAIL", "no unexpected NaNs", "wandb-summary", nan_keys)

print(f"=== {passed} PASS / {failed} FAIL / {warned} WARN ===")
raise SystemExit(1 if failed else 0)
PY
