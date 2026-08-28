#!/bin/bash
# CONNECTION PROOF CHECKER for a pixdino run.
#
#   bash sbatch/check_pixdino_smoke.sh <log> [wandb-run-dir]
#
# If the wandb dir is omitted it is discovered from the log (wandb prints
# its run id at startup), else the newest wandb/wandb/run-* is used.
#
# WHY THIS WAS REWRITTEN (2026-08-26)
# ===================================
# The first version grepped the LOG for keys that are WANDB-ONLY --
# ``train/ladd_pix_*`` are emitted through the wandb dict
# (trainer/causal_action_forcing_train.py:7073-7075) and never printed to
# stdout. It reported 12 FAILs on a run whose evidence it could not see:
# it called the grid jitter "LOCKED" while jitter_x=4 / jitter_y=-1 were
# moving, and reported surrogate_distill_ran ABSENT while it was 1.
#
# A checker that cannot see its own evidence is worse than none. It makes
# a working build look broken, and -- the real danger -- an ABSENT key and
# a genuinely-zero key look identical to it, so it can pass a broken build
# for exactly the reason it failed a working one.
#
# So: every check reads wandb-summary.json, and every line PRINTS THE KEY
# IT READ AND THE FILE IT CAME FROM. ABSENT is reported as ABSENT and is
# never silently treated as 0. Checks that CANNOT be decided from the run
# (lagged counters, warmup windows) report WARN, not FAIL.
set -uo pipefail
LOG=${1:?usage: check_pixdino_smoke.sh <log> [wandb-run-dir]}
test -f "$LOG" || { echo "no such log: $LOG"; exit 2; }
WBDIR=${2:-}
if [ -z "$WBDIR" ]; then
  RID=$(grep -oE "run-[0-9]{8}_[0-9]{6}-[a-z0-9]+" "$LOG" | head -1)
  [ -n "$RID" ] && WBDIR="wandb/wandb/$RID"
fi
[ -n "${WBDIR:-}" ] && [ -d "$WBDIR" ] || WBDIR=$(ls -dt wandb/wandb/run-* 2>/dev/null | head -1)

python3 - "$LOG" "$WBDIR" <<'PY'
import json, os, re, sys
log_path, wbdir = sys.argv[1], sys.argv[2]
sum_path = os.path.join(wbdir, "files", "wandb-summary.json")
print("=== PIXDINO CONNECTION PROOFS ===")
print(f"log:   {log_path}")
print(f"wandb: {sum_path}")
if not os.path.exists(sum_path):
    print("FATAL: no wandb summary -- nothing can be verified. Exiting.")
    sys.exit(2)
S = json.load(open(sum_path))
LOG = open(log_path, errors="replace").read()
SUMNAME = "wandb-summary.json"

def get(key):
    """(value, actual_key) or (None, None). ABSENT != 0.0, ever."""
    for k in (f"gen/train/{key}", f"train/{key}", f"gen/{key}", key):
        if k in S:
            return S[k], k
    return None, None

npass = nfail = nwarn = 0
def emit(label, state, where, val, note=""):
    global npass, nfail, nwarn
    if state == "PASS": npass += 1
    elif state == "WARN": nwarn += 1
    else: nfail += 1
    print(f"  {state:4}  {label}")
    print(f"        read {where} = {'ABSENT' if val is None else val}"
          + (f"\n        {note}" if note else ""))

def ck_key(label, key, pred, note="", warn_only=False):
    v, k = get(key)
    if v is None:
        emit(label, "WARN" if warn_only else "FAIL", f"train/{key} [{SUMNAME}]",
             None, "key never logged -- ABSENT is not 0")
        return None
    ok = bool(pred(v))
    emit(label, "PASS" if ok else ("WARN" if warn_only else "FAIL"),
         f"{k} [{SUMNAME}]", v, note)
    return v

step = S.get("_step")
print(f"\n  RUN REACHED wandb _step = {step}")
print("  warmups: gan_disc_start_step=20  gan_critic_warmup_steps=20")
print("           gan_warmup_steps=25 (full GAN at 45)  pix_real_pool_warm_updates=32")
if isinstance(step, (int, float)) and step < 46:
    print("  *** THIS RUN IS SHORTER THAN ITS OWN WARMUPS. Cumulative counters")
    print("      below cannot distinguish 'never fired' from 'fired once, after")
    print("      the last log'. Any FAIL here is INCONCLUSIVE, not a defect. ***")

print("\n--- 1. build (these ARE printed to stderr, so the log is the source) ---")
for label, pat in [
    ("encoder built", r"\[LADD-PIXFEAT\] built feature source: [^\n]*"),
    ("decode callback installed", r"\[LADD-PIXFEAT\] decode callback INSTALLED[^\n]*"),
    ("encoder param group split", r"\[LADD-PIXFEAT\] encoder param group[^\n]*"),
]:
    m = re.search(pat, LOG)
    emit(label, "PASS" if m else "FAIL", os.path.basename(log_path),
         (m.group(0)[:140] if m else None))

print("\n--- 2. the features come from the PIXEL encoder ---")
ck_key("disc took the pixel branch", "ladd_pix_disc_forwards", lambda v: v > 0)
ck_key("encoder actually ran", "ladd_pix_src_forwards", lambda v: v > 0,
       "the encoder's OWN counter, inside model/ladd_pixel_features.py")
ck_key("WAN taps NOT used", "ladd_pix_wan_projector_calls", lambda v: v == 0,
       "MUST be exactly 0 -- non-zero means some call site still reads the "
       "WAN projector")
gh, _ = get("ladd_pix_grid_h"); gw, _ = get("ladd_pix_grid_w")
emit("DINO token grid (expect 13 x 17)",
     "PASS" if (gh or 0) > 0 and (gw or 0) > 0 else "FAIL",
     f"train/ladd_pix_grid_h,_w [{SUMNAME}]", f"{gh} x {gw}")
ck_key("images scored", "ladd_pix_images", lambda v: v > 0)
ck_key("counters read AFTER the deferred flush", "ladd_pix_logs_after_flush",
       lambda v: v == 1,
       "provenance. 1 = the post-flush publish. ABSENT = the STALE pre-flush "
       "publish, in which case every counter in this section lags one logged "
       "GAN step and a 0 proves nothing.")

print("\n--- 3. did the D-update actually run ---")
ck_key("deferred closure FLUSHED", "ladd_disc_deferred_flush_events",
       lambda v: v > 0,
       "unconditional key. Every OTHER deferred-update key is gated on the "
       "fake-score feature-backbone override, which a PIXEL arm never "
       "installs -- so on this config they are all suppressed and cannot "
       "answer this question.")
ck_key("deferred closures executed", "ladd_disc_deferred_flush_closures",
       lambda v: v > 0)
ck_key("gt_vs_fake pairs built", "ladd_n_pairs_gt", lambda v: v > 0)
ck_key("disc not skipped", "r3gan_disc_skipped_gt", lambda v: v == 0)
ck_key("D-updates counted", "r3gan_disc_updates_total_gt", lambda v: v > 0,
       "assembled INSIDE _ladd_run_pair_mode, i.e. before the deferred "
       "flush -- lags one logged GAN step by construction, so 0 is "
       "inconclusive on a short run", warn_only=True)
ck_key("gt_vs_fake D-loss", "r3gan_d_loss_gt", lambda v: v != 0,
       "healthy 0.25-0.55. Same lag as above.", warn_only=True)

print("\n--- 4. anti-lattice (phase jitter) ---")
jy, _ = get("ladd_pix_jitter_y"); jx, _ = get("ladd_pix_jitter_x")
emit("jitter values present", "PASS" if jy is not None and jx is not None
     else "FAIL", f"train/ladd_pix_jitter_y,_x [{SUMNAME}]", f"y={jy} x={jx}",
     "the summary holds ONE value per key, so this shows presence, not "
     "movement. Movement is guaranteed by construction (jitter_for is a "
     "function of the step) and is unit-tested; the old checker's "
     "'phase LOCKED' FAIL was reading the wrong file, not a finding.")

print("\n--- 5. surrogate route ---")
ck_key("distillation ran", "surrogate_distill_ran", lambda v: v == 1)
ck_key("critic served the generator", "surrogate_consumed", lambda v: v == 1)
ck_key("G-term applied", "pix_g_applied", lambda v: v == 1, warn_only=True)
ck_key("score_pixels served the surrogate", "ladd_pix_teacher_calls",
       lambda v: v > 0)
ck_key("teacher refreshed", "surrogate_n_teacher_refresh", lambda v: v > 0,
       "read this, never a raw forward count (2x under checkpointing)")
w, _ = get("pix_g_warmup_skipped")
if w == 1:
    print("        NOTE pix_g_warmup_skipped=1 -- the G-term was inside its")
    print("        warmup at the last logged step. pix_g_applied=0 is EXPECTED")
    print("        there and is not evidence of a defect.")

print("\n--- 6. gradient REACHES the generator (student-chunk probe) ---")
ck_key("probe anchored at the student chunk",
       "surrogate_grad_probe_site_is_chunk", lambda v: v == 1, warn_only=True)
sev, _ = get("surrogate_grad_severed_from_chunk")
emit("gradient NOT severed", "PASS" if sev is None else "FAIL",
     f"train/surrogate_grad_severed_from_chunk [{SUMNAME}]", sev,
     "absent = good. 1 = the surrogate term does not reach the generator.")
unav, _ = get("surrogate_grad_probe_unavailable")
if unav == 1:
    r1, _ = get("surrogate_grad_probe_reason_no_tensor")
    r2, _ = get("surrogate_grad_probe_reason_base_unreachable")
    if r1 == 1:
        reason, st = ("no_tensor: fake_lat absent or requires_grad=False", "FAIL")
    elif r2 == 1:
        reason, st = ("base_unreachable: the REST of generator_loss does not "
                      "reach the chunk. Normal INSIDE the warmup window; a "
                      "defect outside it.", "WARN")
    else:
        reason, st = ("*** NO REASON KEY -- the alarm cannot say what tripped "
                      "it. That is itself a defect (fixed 2026-08-26). ***",
                      "FAIL")
    emit("probe produced a reading", st,
         f"train/surrogate_grad_probe_unavailable [{SUMNAME}]", 1, reason)
ck_key("grad norm non-zero", "surrogate_grad_norm_unweighted", lambda v: v > 0,
       "an EXACT 0.0 is the defect the probe was moved to the chunk to catch")

print("\n--- 7. CALIBRATION READ (requires PIXW=0.0) ---")
# PREFERRED anchor: the PARAMETERS. On these arms the surrogate's fake is
# flash_dmd_gan_x0 -- a different generator sub-graph from the tensor DMD
# scores (One-Forcing divergence 3) -- so no single tensor lies on both
# routes and the same-quantity chunk ratio is undefined by construction,
# not by defect. The parameters are where the two gradients actually add.
r, kr = get("surrogate_param_grad_ratio_unweighted")
src = "PARAMETER-SPACE (preferred)"
if r is None:
    r, kr = get("surrogate_grad_ratio_unweighted")
    src = "student-chunk (fallback)"
if r is None:
    print("      NO READING. Neither anchor produced a ratio:")
    print("        train/surrogate_param_grad_ratio_unweighted  ABSENT")
    print("        train/surrogate_grad_ratio_unweighted        ABSENT")
    print(f"      in {SUMNAME}. PIXW CANNOT be set from this run.")
    sev, _ = get("surrogate_param_grad_severed")
    if sev == 1:
        print("      train/surrogate_param_grad_severed = 1 -- the surrogate")
        print("      term reaches NO generator parameter. That IS a defect.")
elif r > 0:
    print(f"      anchor: {src}")
    print(f"      {kr} = {r}")
    print(f"      => PIXW = 0.10 / {r:.6g} = {0.10/r:.4g}")
    print(f"         5-20 % band: {0.05/r:.4g} .. {0.20/r:.4g}")
    nr, _ = get("surrogate_param_n_reached_sur")
    np_, _ = get("surrogate_param_n_params")
    if nr is not None:
        print(f"      surrogate reached {nr} / {np_} trainable generator params")
else:
    print(f"      {kr} = {r} -- cannot solve for PIXW.")

print("\n--- 8. memory / speed / NaN ---")
for pat, label in [(r"alloc[=: ]+([0-9.]+)", "alloc GB"),
                   (r"step_peak[=: ]+([0-9.]+)", "step_peak GB"),
                   (r"s/step[=: ]+([0-9.]+)", "s/step")]:
    m = re.findall(pat, LOG)
    print(f"      {label}: last 3 = {m[-3:] if m else 'none found in log'}")
crash = re.search(r"Traceback|CUDA out of memory|RuntimeError", LOG)
emit("no crash", "PASS" if crash is None else "FAIL",
     os.path.basename(log_path), crash.group(0) if crash else "clean")
nan_keys = [k for k, v in S.items()
            if isinstance(v, float) and v != v and "r1" not in k]
emit("no unexpected NaN in logged metrics", "PASS" if not nan_keys else "FAIL",
     SUMNAME, nan_keys or "none",
     "r1_* NaN is the did-not-fire regime flag and is excluded on purpose")

print(f"\n=== {npass} PASS / {nfail} FAIL / {nwarn} WARN ===")
print("WARN = cannot be decided from THIS run (lagged counter, warmup window,")
print("       or advisory). Not evidence of a defect.")
sys.exit(1 if nfail else 0)
PY
