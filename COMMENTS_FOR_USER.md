# LOGGING DEFECT: the encoder param-group line always cries wolf (2026-08-26)

`trainer/causal_action_forcing_train.py:3020-3029` prints, UNCONDITIONALLY:

    [LADD-PIXFEAT] encoder param group: n_encoder=%d n_head_adapter=%d
    lr_scale=%.4g encoder_lr=%.3g head_lr=%.3g (0 encoder params => the
    encoder is FROZEN, which contradicts ladd_pixel_encoder_trainable)

The parenthetical is part of the FORMAT STRING, not a conditional branch. So a
perfectly healthy trainable-encoder arm logs, verbatim (holder 6143212,
pixdirect_onlinelr, live):

    n_encoder=174 n_head_adapter=56 lr_scale=0.4 encoder_lr=8e-06
    head_lr=2e-05 (0 encoder params => the encoder is FROZEN, which
    contradicts ladd_pixel_encoder_trainable)

n_encoder=174, trainable_encoder_params=22.06M, lr_scale=0.4 -- everything is
CORRECT, and the line still says the encoder is frozen and contradicts its own
flag. This is the forgeable-zero failure wearing a different hat: an alarm that
fires in every state cannot report any state. It is exactly the kind of line
that produces a false conclusion when someone greps a log at 2am.

NOT FIXED YET, DELIBERATELY. It is a pure print and changes no result, but the
arms record `srcmd5` over the trainer sources in their [ARRWM-PROVENANCE] line.
Editing it now would make the arms of the CURRENTLY RUNNING comparison
(pixdirect_onlinelr / _strong / _gtclean / pixdino_frozen) hash differently from
one another for no scientific reason. Proposed fix at the next natural break:

    if not _enc_params:
        print("... WARNING: 0 encoder params => encoder is FROZEN, which "
              "contradicts ladd_pixel_encoder_trainable=true")
    else:
        print("... n_encoder=%d ...")

i.e. emit the diagnosis ONLY in the state it diagnoses. Sign-off wanted on
whether to fix now (breaking srcmd5 continuity mid-campaign) or at the break.

# COMMENTS FOR USER

Live questions and things you need to know. Anything proposed → signed off →
implemented → **finished** is moved to `RESOLVED_TODOs.md`.
Newest first.

---

# GAN SAMPLE BUDGET: the observability landed, the CAPACITY did NOT (2026-08-26)

Code: `model/ladd_disc.py` (`__init__`, `_pixel_features`),
`model/gan_balance.py` (new), `trainer/causal_action_forcing_train.py`
(`_ladd_install_pixel_decode`, `_ladd_pixel_logs`, `_ladd_penalty_fire_logs`,
`_ladd_run_pair_mode`, `train`), `configs/action_forcing_phase3_dmd.yaml`.
Tests: `testing/test_ladd_pixel_crop_budget.py` (26 passed; 21 of them FAIL on
the pre-change tree), `testing/test_gan_dwins_tripwire.py` (20 passed; the
whole file fails pre-change). Every capacity knob is flag-gated and OFF, and
the off path is the legacy one byte for byte.

**The headline, and it reverses the brief I was given.** I was asked to raise
the disc's sample budget as far as memory allows. Re-derived from live logs,
**memory was never the binding constraint** — even K=4/L=3/kf=8 lands ~76 GB
of 95 on the worst rank. The binding constraint is the G/D balance, and the
`pixdirect_strong` arm proves it: **d_loss 0.0104 / 0.0039 / 0.0024 / 0.0102
at steps 161/171/181/191**, four consecutive below the 0.135 D-wins floor,
still falling at run end, while its **median over steps>=50 reads 0.3632** —
inside the healthy band. `pixdirect_onlinelr`, same encoder lr, coverage NOT
raised, dips twice and recovers (0.1346, 0.0139, then 0.6243), median 0.4560.

## SIGN-OFF 1 — the crop-sampler bias you were asked about is BUILT, default OFF

`ladd_pixel_crop_y_lo_frac` / `_hi_frac`, fractions of the latent height that
the crop ORIGIN may be drawn from. `(0.0, 1.0)` ships and is the historical
full-height uniform draw, asserted byte-identical against a longhand replay of
the old schedule at 5 epochs. Lower two-thirds is `crop_y_lo_frac=0.3333`; the
texture benchmark's own band (`analysis/sharpness/radial_spec.py` CROPS) is
`0.35 / 0.97`. It stays a deterministic function of `pixel_epoch`, so the five
D-updates and both R1 forwards still see one identical crop — tested.
Your three objections all still stand and I have NOT turned it on. Proof of
fire is `train/ladd_pix_crop_band_active` = 1.0, with
`ladd_pix_crop_y0_min` >= the band floor (20 on the 60-row latent).

## SIGN-OFF 2 — do NOT raise the capacity knobs yet, except one

Ranked, with the D-side counterweight each one needs. The D's information per
step goes as (images per disc forward) x (`gan_updates_per_step`): the healthy
arm ran 4.88 x 5 = 24.4, the collapsed arm 9.76 x 5 = 48.8. **n = 2 arms, one
of each, so the exchange rate is INCONCLUSIVE** — what is measured is that 2x
at `gan_updates_per_step=5` collapsed and 1x did not.

| change | images/fwd | latent cells/G-update | rank-0 peak | counterweight |
|---|---|---|---|---|
| **`ladd_pixel_lat_frames` 2 -> 3** | **4.88 (UNCHANGED)** | 8.2 % -> 12.3 % | 57.5 -> ~58.9 GB | **NONE NEEDED** |
| `ladd_pixel_crops_per_row` 1 -> 2 | 9.76 (2x) | -> 23.1 % | ~63.2 GB (measured) | `gan_updates_per_step` 5 -> 2 |
| `ladd_pixel_frames_per_crop` 2 -> 6 | 14.6 (3x) | unchanged | ~60.0 GB | `gan_updates_per_step` 5 -> 2 |

`lat_frames=3` is the one I would ship now. It is the ONLY coverage increase
that adds **no** samples to the discriminator — it changes which latent cells
receive gradient, not how many pictures D scores — and it closes the measured
hole where the oldest latent frame of every chunk is never adversarially
supervised in any step, ever. Asserted as a test.

`frames_per_crop` is the cheapest capacity in the system and nobody has used
it: the VAE returns `4 x lat_frames` pixel frames and the disc keeps **2**, so
at L=3 **ten of twelve decoded frames are thrown away** after the expensive
half is already paid. Raising it costs the ViT batch only (~76 MB/image). But
it still feeds D more pictures, so it is gated behind the balance question.

## The tripwire that would have caught this at step 181, not step 200

`train/gan_dwins_tripped` / `_max_streak` / `_trip_step`, plus a loud
`[GAN-BALANCE]` warning. Fires when d_loss stays <= `gan_dwins_floor` (0.135)
for `gan_dwins_k` (3) consecutive logged steps. k=3 because onlinelr's
RECOVERED transient is 2 observations long, so 3 is the smallest k that
separates the measured recovery from the measured collapse — both replayed
verbatim as regression tests. Exactly-0.0 d_loss rows (the pre-disc-start
sentinel) are rejected, not counted, or it would fire on every run at step 21.
It defaults **ON**: it touches no tensor, optimiser or RNG, so training is
byte-identical, and an alarm that ships off is the failure it exists to catch.

## Three defects fixed unconditionally, because they are how false conclusions get made

1. **`lat_frames` was silently clamped with no counter at all**
   (`L = min(lat_frames, F_lat)`). The only evidence a raise took was the boot
   echo, which prints the REQUEST. Now `ladd_pix_lat_frames_cfg` / `_used` /
   `_avail` / `_clamped`, and the same for `frames_per_crop`, `crops_per_row`,
   `crop_rows`, `crop_cols`.
2. **`ladd_pixel_decode_batch` is INERT** — the trainer writes it into
   `disc.pixel_cfg` and the boot echo prints it, but nothing reads it, so
   lowering it is a silent no-op. Left inert (the surrogate path's key is a
   different one, `pix_decode_batch`); the boot echo now says `(INERT)` and the
   working knob is the new `ladd_pixel_decode_split`.
3. **`gan_updates_per_step` had no counter either.**
   `r3gan_disc_updates_total` counts D PHASES (170 on a 200-step arm), not the
   five inner iterations — so the one lever sanctioned as the D-side
   counterweight could not be proven to have taken. Now
   `train/r3gan_disc_inner_updates_total` / `_last`.

## What I would run, in order

1. A **balance arm** at zero coverage change: `pixdirect_onlinelr` exactly,
   `gan_updates_per_step` 5 -> 2, nothing else. It calibrates the counterweight
   before any capacity is spent on it. Expect d_loss median 0.45-0.60,
   `gan_dwins_tripped=0`, `r3gan_disc_inner_updates_total/`
   `r3gan_disc_updates_total` = 2.0.
2. `ladd_pixel_lat_frames=3` alone (no counterweight). Expect
   `ladd_pix_lat_frames_used=3`, `ladd_pix_lat_frames_clamped=0`,
   `ladd_pix_images/ladd_pix_disc_forwards` STILL 4.88, and d_loss in
   onlinelr's band.
3. Only then the coverage arms, each paired with the counterweight step 1
   calibrated. **I have built no sbatch and submitted nothing.**

---

# SIGN-OFF NEEDED — should the GAN crop sampler be biased toward textured rows? (2026-08-26)

**This is a recipe change, so it is not being made without your word.**

Full context in `analysis/gan_tuning/PIXDIRECT_200_RESULTS.md` §3.3. Short
version: the adversarial dose is spatially tiny, and a measurable fraction of
it lands on nothing worth discriminating.

**What was measured** on the completed `pixdirect_frozen` 200-step arms. The
crop RNG is `manual_seed(step*7919 + 104729)` (`model/ladd_disc.py:1685-1710`),
so the crop schedule is exactly reproducible and could be replayed and
content-scored offline:

| | n crops | crops containing NO road/verge at all |
|---|---|---|
| G-update crops (the ones that actually reach the generator) | 36 | **14 (38.9 %)** |
| all D-update crops | 180 | **71 (39.4 %)** |

Per-crop content coverage over the same draws:

| class | mean | median |
|---|---|---|
| road / verge | 0.336 | 0.304 |
| sky | 0.182 | **0.000** |

**It is NOT the sky problem** — the median sky coverage is exactly zero, so
`ONBOARDING_PIXDIRECT.md` §7's "don't measure texture on sky" hazard is not
what is biting here. The defect is **texture-free draws**: crops that contain
neither road nor verge nor sky in quantity, i.e. flat content with nothing for
a texture discriminator to separate on.

This compounds with the rest of the dose accounting: only **36 of 201**
iterations (17.9 %) carry a live G-term at all, a `24x32` crop is 12.3 % of a
60x104 latent frame at **one** location shared by all rows, and `lat_frames=2`
of `num_frame_per_block=3` means the oldest latent frame of every chunk is
never adversarially supervised — **8.2 % of the chunk's latent cells per
G-update**. Losing ~39 % of those draws to texture-free content is a large
fraction of a small budget.

**Proposed fix (NOT implemented):** sample `y0` from the **lower 2/3** of the
latent instead of uniformly over the full height. Minimal, one line, and it
preserves the reproducible-seed property.

**Why it needs your sign-off rather than just being done:**

1. It changes what the discriminator is trained to call "real", which is a
   training-recipe change, not a telemetry change.
2. It biases the disc toward road/verge statistics specifically. If the goal
   is dataset **style** broadly (your stated objective), a road-weighted disc
   may shift style toward road texture rather than toward the dataset.
3. It is not free of confound with the `pixdirect_gtclean` test already built
   (`sbatch/pixdirect_gtclean.sbatch`, not submitted) — if both land at once,
   neither is single-variable.

**Options, in the order I would rank them:** (a) leave it, and treat the 39 %
as part of the measured dose; (b) lower-2/3 `y0` as a flag-gated,
default-off single-variable arm; (c) content-aware rejection sampling, which
is more work and adds a per-step content pass.

No arm has been built for any of these.

---

# SURROGATE ROUTE FIXED — two arms ready, NOT submitted (2026-08-26)

`analysis/gan_tuning/PIXEL_FEATURE_SOURCE.md` §7c "RESOLVED" has the full
measurement. Short version: the surrogate delivered zero gradient because
**the latent critic got 14 gradient steps in a 90-step run**, not because
of anything structural. The distillation runs only on generator
iterations (`dfake_gen_update_ratio=5`) and only above
`gan_disc_start_step=20`, so 90 steps → 14 calls → 7 teacher refreshes,
which is exactly the `surrogate_n_teacher_refresh = 7` that run reported.
Held-out gradient cosine vs critic steps (CPU, n=1 per point):

| critic steps | 7 | 14 | 30 | 60 | 120 | 250 | 500 |
|---|---|---|---|---|---|---|---|
| realistic teacher | 0.380 | 0.572 | 0.676 | 0.758 | 0.803 | 0.826 | 0.841 |
| worst-case teacher | 0.003 | −0.003 | −0.001 | 0.020 | 0.034 | 0.312 | 0.707 |

Production's 0.0097 is simply what 14 steps looks like.

**Fix (both default-off, byte-identical when off):**
`surrogate_distill_substeps=12` — 12 critic steps per call, replaying the
CACHED teacher targets, so `pix_teacher_refresh_every` and the expensive
decode+DINO half are untouched. At `MAXSTEPS=200` that is 36 calls × 12 =
432 critic steps. Plus `surrogate_critic_head_init_std=0.02`, which
removes an exactly-zero trunk gradient at the shipped zero-init head —
mechanism-motivated; its effect on the end metric is **within noise at
n=1**, so it is honest to call it insurance, not a measured win.

`gan_loss_weight` stays 1.0. `surrogate_critic_lr` stays 2.0e-4.
`real_guidance_scale=0.0`. Nothing else in either arm changed.

Tests: `testing/test_surrogate_gradient_field.py`, 8 passed. It fails on
the pre-fix behaviour — verified by making the flag inert, which collapses
the arm pair to `shipped=0.2567 fixed=0.2567`.

## NEEDS YOUR SIGN-OFF — I have NOT run this

`PIXW`: §7c said to calibrate it from `surrogate_grad_ratio_unweighted` on
a `PIXW=0.0` probe. That reading was taken against a dead critic, so it is
void and must be retaken. Proposed sequence, on a FREE holder only:

```bash
# STEP 1 — PIXW=0.0 calibration probe (does not apply the term)
cat > logs/.holder_cmd_<FREE_HOLDER>.sh <<'EOF'
#!/bin/bash
cd /scratch/u6ex/as1748.u6ex/ARRWM
export PIXW=0.0
HOLDER=<FREE_HOLDER> SMOKE=pixdino_frozen PORTOFF=<unique> bash sbatch/run_smoke_on_holder.sh
EOF
chmod +x logs/.holder_cmd_<FREE_HOLDER>.sh
```

Read from `wandb/wandb/<run>/files/wandb-summary.json` (these are
wandb-only keys, the log grep reports false FAILs):

* `train/surrogate_substeps_achieved` — **must read ~12.0, not 1.0.** If
  it reads 1.0 the flag did not fire and the run proves nothing.
* `train/surrogate_check_cos_sim` — was 0.0097. Should be well clear of
  0.3 by step ~150.
* `train/surrogate_grad_norm_unweighted` — must be > 0. **If cos > 0.6 and
  this is STILL exactly 0, there is a second, plumbing defect** — stop
  there, do not read texture off that run.
* then `train/surrogate_grad_ratio_unweighted` → `PIXW = 0.10 / r`.

STEP 2 is the two real arms at that PIXW (`SMOKE=pixdino_frozen` and
`SMOKE=pixdino_online`, separate holders, unique `PORTOFF` each).

Questions for you: (a) do you want STEP 1 at all, or launch straight at a
guessed PIXW to save a holder? (b) `surrogate_distill_substeps=12` is my
pick from the table — happy to go 24 if you would rather over-train the
critic, it is cheap. (c) keep `head_init_std=0.02` or set it 0.0 to keep
the arm strictly single-variable against the shipped critic?

# FIXING THE CARN — bare-minimum experiment set (2026-08-26)

Source review: `analysis/gan_tuning/CARN_TRANSITION_REVIEW.md`. Everything below is
config-only unless marked CODE.

## What is actually broken (three defects, in order of severity)

**D1 — the noiser is queried OUT OF DOMAIN.** With `forward_noiser_reverse=true` the net is
trained as a **de-CARN denoiser**: `fn_inputs = drifted, fn_targets = cleaner`
(`trainer:11045-11050`). But the branch our decoupled arms land in (`_chain_app`,
`:14201-14203`) is documented as a **forward** FN and its level schedule is derived for an
increment that *adds* drift. Design intent pairs `reverse=true` with `apply_gt_both`, NOT
`apply_gt_former`. So we feed clean GT to a network that has only ever seen drifted input.
**Every CARN arm in this campaign has this.** Fix: `forward_noiser_reverse=false` when using
`apply_gt_former`.

**D2 — realized strength is ~half of the coupled path.** Decoupled levels come from the
*band* index (`_Lf = j+1`, `t = _Lf//2`, cap `j`); with 3 band chunks that yields
`levels=[0,1,1]`, **max 1**, and one of three rows left clean. The coupled path drew from
the *ride-absolute* pool position and sat AT the Req-1 cap: `levels=[0,0,1,2,1]`, max 2.
The formulas are algebraically identical; the realized distributions are not.

**D3 — `carn_recurse=false` is inert** in every decoupled arm. Unread by the decoupled
block (it branches on `chain_levels`), and the training guard at `:10871` is false under
`chain_levels=true`. We are training/applying the chained scheme, not a cumulative jumper.

## The ablation we have been running is NOT deconfounded

`carnpure` vs `gansig_gtvf_dmd` differ by more than the CARN:
* **C1 (GAN-fatal): different discriminator initial weights.** `ForwardNoiser` (23.12M
  params) is constructed at `dmd_action_forcing.py:2758`, BEFORE the LADD disc heads are
  built. ~23M `kaiming_uniform_` draws land between the per-rank seed and the disc init.
* **C2:** an entire extra no-grad rollout per ride setup (`_need_rollout2`, true iff
  FN+`teacher_feat`+`chain_levels`), drawing CUDA RNG.
* **C3:** per-step sliced-Wasserstein `randn` + `randperm` on the default CUDA generator.
* **C4:** extra DDP module + 8-bit optimizer.

**The correct control is `carnpure` with `forward_noiser_apply_gt_former=false`** — same
module, same RNG, same rollout, differing ONLY in whether the transform is applied.
`gansig_gtvf_dmd` is the wrong control and is retired from this comparison.

**Also void:** `gansig_gtvf_carn_slide12` ≡ `carnpure`. Its only differing line,
`forward_noiser_gt_match_frames=12`, is read only inside the `not carn_recurse and not
chain_levels` branch — **inert** under `chain_levels=true`. The two "slide12" runs were
**A/A replicates**; the difference observed between them was noise, not matching.

## BARE MINIMUM experiment set (3 arms + control)

| # | arm | change vs `carnpure` | answers | status |
|---|---|---|---|---|
| 1 | `gansig_carn_ctrl` | `apply_gt_former=false` | matched control (C1-C4 all held) | **RUNNING** 6135660 |
| 2 | `carnpure` | — (as-is) | does applying do anything? | done, 191 steps |
| 3 | `gansig_carn_fwd` | `forward_noiser_reverse=false` | **D1 direction fix** — without this, 1-vs-2 measures an OOD transform | **RUNNING** 6135661 |
| 4 | `carn_strong` | `former_mode=strong` (+ #3) | D2 level-ceiling; run only if 1-vs-3 is null | not built |

**Protocol, non-negotiable given the variance study:** ≥3 seeds per arm, **pin the nodelist
so paired arms share nodes** (cheapest single variance fix), run past step 130, read medians
past step 50, and report the bit-stable structural telemetry (`ride_chunk`, `reset`,
`tf_pairs`, `r1_rate`, `hold_step`, `peak_gb`) as proof the arm ran what it claims.

## Credibility bars (from the n=2 variance study)

Identical configs on different nodes diverge at step 0 and amplify ×250 over 90 steps (no
determinism enforcement, TF32 on, different NCCL reduction order). Noise floors:

| metric | floor | usable at n=1? |
|---|---|---|
| `roll_mae` / `dmd_err` / `gt_dist` | 3.5–5.3% | only if Δ > 15% |
| `gen_loss` / `d_loss` | 14–17% | only if Δ > 45% |
| `gan_dmd_grad_ratio` | **50%** | **no** |
| `r1` | **76%** | **no** |
| `d_real` / `gan_cos` | **95–119%** | **no** |

## CODE items (described, not implemented — need sign-off)

1. **Direction guard**: mirror the existing `reverse`/`cycle_enabled` mutual exclusion
   (`dmd_action_forcing.py:2737-2748`) to reject `reverse=true` + `apply_gt_former=true`.
2. **Fail-fast on inert `carn_recurse`** under `chain_levels=true`.
3. **Port site B's Req-2 gen-side shield** (`carn=True` on the D-update only) into the
   decoupled path, which currently copies site C's naive both-sides placement.

---

# OPEN QUESTIONS (waiting on you)

## FIRST — WHAT THE GANSIG ARMS ACTUALLY DID: **almost nothing yet.** Be clear about this.
You asked. Honest answer: **three of the four never ran at all**, because no holder ever
freed for them — the four extra holders you submitted are still queued behind `Priority`
and have never started.

| arm | what it tests | status |
|---|---|---|
| `gansig_real` | +`gt_vs_fake` (a 2nd pair mode: straight real-GT-vs-generated) | **38 steps only**, on the expiring holder |
| `gansig_of` | `fake_sample_source=dmd` — One-Forcing divergence 3, also fixes the action-lag asymmetry | **never ran** |
| `gansig_wide` | wide real pool (needs `all_pairs`) | **never ran** |
| `gansig_huge` | cross-ride real set: 5→60 rows/step, 1→~33 rides | **never ran** |

What the 38 steps of `gansig_real` bought: both pair modes construct, GAN engages, disc
healthy (`d_loss` 0.52 at step 31, not saturating), peak **69.76 GB at step 30** — so a
second pair mode costs only **+4.9 GB**, not the +20 GB feared. No median `ratio` (needs
steps ≥45). That is the entire empirical yield of the gansig family so far.

So the gansig *analysis* was valuable (it found the action-conditioning defect, the dead
config keys, and the cross-ride mechanism) but the gansig *runs* have produced one memory
number. I would rather say that plainly than let four arm names imply four results.

---

## Q1. `ladd_r1_unified_cadence=true` on the matched arms?  ← LESS URGENT NOW
**Update: making `ladd_r1_gamma=10.0` the default (your w1 directive) partly fixed this by
accident.** R1's effect scales with gamma × fire-rate, so:

| | gamma | fires per D-update | effective R1 dose |
|---|---|---|---|
| the arms, this morning | 1.0 | 0.20 | **0.2** |
| **the arms, now (w1 default)** | **10.0** | 0.20 | **2.0** — 10× stronger than this morning |
| tuning ladder (positional) | 10.0 | 1.00 | 10.0 |

So the arms went from 50× weaker than the ladder to **5× weaker**. The gap is real but no
longer alarming, and the mechanism stands: `ladd_r1_every_n_steps=1` cannot raise the
matched path's rate, because the legacy latch caps it at 1-per-5 regardless of cadence.
**Only `ladd_r1_unified_cadence=true` closes the last 5×**, and it defaults false because
of an old double-fire bug.
> **Still worth asking, but I would now say: leave it off** unless a long arm shows the
> disc saturating (`d_loss` → 0.001, logit gap climbing past ~6). Flipping an
> old-bug-flagged knob across eight arms to chase a 5× on a regulariser that is already
> 10× stronger than this morning is not a good risk. **Overrule me if you disagree.**

*Smoke criterion, corrected:* expected `r1_rate` on `smk_v6fmit` is **0.20**, not 0.545.
0.00 = R1 dead; 1.00 = unified got flipped.

## Q2. Port the CARN application to the unmatched path? (optional, ~40 lines)
Matching's real cost is now down to ~nothing (see `RESOLVED_TODOs.md` R7), but the CARN
noiser is still only *applied* inside the matched block. Porting that block to the
positional path — where the level index is directly correct — would make matching
genuinely optional and cost nothing. Design in §6 of
`analysis/smoke_gate/matching_cost_audit.md`.
> **Want it?** Not urgent; the current config works.

## Q4. One-Forcing divergence 1 is now tested by NO arm — do you want it probed?
Honest admission: I "fixed" something that was not broken. `gansig_of` carried
`ladd_fake_backbone_grad_scale=0.25`, which an audit flagged as a 2.5× over-kick against
the `1/(n_modes × updates)` rule. I corrected it to 0.1 — but the file's header had
stated three times that 0.25 was **deliberate**, the arm's second treatment: the
divergence-1 probe (One-Forcing's "the disc *is* the trainable fake-score backbone").
My revert makes `gansig_of` a clean single-variable test of the sample source, which is
better science — but it means **divergence 1 is now an open hypothesis tested by nothing
in the sweep.** (The stale header that still narrated 2.5× has been corrected either way —
that part genuinely was the config-vs-doc bug class we keep hitting.)
> **Want a dedicated divergence-1 arm** (backbone kick raised deliberately, everything
> else held)? It is one more 2 h holder slot.

## Q3. The ladder's other three axes — I now think TWO of them are obsolete
Originally: an R1-gamma sweep, an update-balance test, and a disc-speed test.
Updated view:
- **R1 sweep — obsolete.** The partial data already chose gamma 10 (it raised the gradient
  ratio 2.5× at fixed weight) and you have since made 10.0 the codebase default. Re-running
  1-vs-10 would only re-derive a decision already taken.
- **Update balance / disc speed — still open but LOW value.** Neither addresses resolution,
  which is the actual complaint.
> **My recommendation: drop these and spend the slots on the disc-timestep arms
> (`w2tclean`, `w2tsamp`, both built) and the pixel PatchGAN instead.**

## Q5. NEW — if SAM2 is not being consumed, do we pay for the pixel critic?
`gantune_w2sam` runs with `gan_pixel_texture_enabled=false`, and the trainer warns the
surrogate's generator-side branch is unreachable without the pixel critic. If the agent
confirms it is training a critic nobody reads, the fix is to enable the pixel critic — but
that adds a VAE decode plus a pixel discriminator to every step, and the first SAM2 attempt
already **OOM'd at 94.6 GB** and had to drop from 512 to 256 resolution.
> **If it turns out unconsumed: relaunch with the pixel critic on (and accept the memory
> risk), or drop the SAM2 route and go straight to the standalone pixel PatchGAN?**

## Q6. Pixel PatchGAN — PLANNED (you asked for this), and you are right that it has issues
`sbatch/gantune_w2pix.sbatch` is **not built yet** — I am writing the plan first because
you flagged concerns and I share them. Here is the experiment as designed, with its known
problems stated up front rather than discovered on a holder.

**The design.** Enable `gan_pixel_texture_enabled=true` on the w2 base with the
from-scratch pixel PatchGAN (`docs/TEXTURE_GAN_DESIGN.md`, the `pix_*` machinery, arm G in
that spec). Decode small latent crops to pixels each step (`pix_crop_lat`,
`pix_crops_per_step`, `pix_frames_per_crop`), run a PatchGAN over them, and add its G-term
alongside the latent LADD disc. Judge on the same GAN-health battery **plus** an actual
sharpness measure on the rendered videos, not on `d_loss` alone.

**The issues, honestly:**
1. **Memory.** It adds a VAE decode *with gradient* plus a discriminator to every step.
   SAM2 — which decodes for its teacher too — already OOM'd at 94.6 GB and had to halve
   resolution. Mitigations exist (`pix_crop_lat`, `pix_decode_batch`,
   `pix_decode_border_trim`, `pix_finish_grad_enabled`) but this is the highest-memory
   thing we would have run.
2. **From-scratch discriminator.** Unlike SAM2/ConvNeXt it has no pretrained prior, so it
   must *learn* what real texture looks like from our own data while the generator moves
   under it — the classic unstable-GAN setup, and the reason WP-SURROGATE (the SAM2 route)
   was built as an alternative in the first place.
3. **Crop statistics.** It sees small crops, so it can be satisfied by locally-plausible
   texture that is globally wrong — the standard PatchGAN failure, and a plausible route to
   the *stippled/halftone* artefacts the stripe forensics already found in this codebase.
4. **`pix_r1_gamma=0.0`** in the current defaults means no gradient penalty on the pixel
   disc — given how much R1 mattered on the latent disc, that likely needs raising.
5. **Confound.** It changes both "where the disc looks" (pixels) and "what the disc is"
   (from-scratch PatchGAN) at once, so a null result would not tell us which half failed.

**Because of (5) I would run it AFTER the ConvNeXt teacher**, which changes only the
feature basis and is therefore interpretable. If ConvNeXt raises sharpness, the pixel
PatchGAN may be unnecessary; if it does not, the PatchGAN is the remaining instrument and
we run it knowing (1)–(4).
> **Confirm the ordering (ConvNeXt first, PatchGAN second) or tell me to build the PatchGAN
> now and I will.**

---

# RESOLVED (21:30): NO TRADE-OFF NEEDED — divergence 3 and a clean disc COMBINE

Answering my own flag below: **you do not have to choose.** Read from the precedence chain
at `trainer:12108-12175`, `ladd_disc_force_clean` is evaluated **before** the band and
overrides it, and `ladd_disc_sample_t` is applied **last** and overrides everything. The two
knobs are orthogonal by construction: `ladd_fake_sample_source=dmd` decides *which tensor is
the fake* (graph-on band slice); `disc_t_int` decides *what noise both halves are corrupted
to*. So divergence 3's alignment benefit and maximum texture visibility can be had together.

**How bad the blinding actually is, measured.** The rung distribution (n=18, identical in
w1 and w2) is roughly uniform: t=208 33%, t=357 17%, t=625 17%, t=1000 33% — **mean 566, and
≥357 on two thirds of steps**. Against the flash path's fixed t=60 that is about a **9×
higher mean corruption level**. So my concern was quantitatively justified, though t=1000
was not the whole story.

**Built: `sbatch/gansig_ofclean.sbatch`** — `fake_sample_source=dmd` + `force_clean=true`,
verified `ladd_real_match_fake_t` is unset (if it were true it would silently re-pin the
timestep to the band rung and undo force_clean — a trap worth knowing about).

**One honest cost.** With force_clean the frozen teacher is conditioned on t=0 while the x0
it is judging came from a rung up to 1000. The code follows the fake's own timestep
deliberately ("conditioned on the level it is actually looking at"), so combining them
breaks that invariant. It is legal, not free, and if the arm behaves oddly this is the first
thing to suspect.

**A confound worth recording before anyone reads the obvious analysis.** On w1/w2 the logit
gap rises steeply with rung (+1.05 → +3.35 across t=208→1000). That is *not* evidence that a
noisier disc helps — in those runs `disc_t` was pinned at 60 and only the *sample's* rung
varied, so it measures sample quality: a worse x0 is easier to separate. `gansig_of` is the
clean natural experiment, because there `disc_t == rung`, so the two effects oppose. If its
gap-vs-t comes out flat or falling, the blinding cancels a known +2.2 of separation — which
would be strong motivation for the disc-timestep arms.

> **For your queued arms: nothing is forced.** Only `v6alt` uses the dmd band; if you want
> it to keep divergence 3 *and* see texture, adding `ladd_disc_force_clean=true` is a
> one-line change plus a resubmit. Say the word — I have not touched it.

---

# EARLIER FLAG (21:15) — possible conflict, now resolved above

`gansig_of` confirmed One-Forcing divergence 3 is live, with four pre-registered
predictions all matching: both pair modes fire (`['gt','gtxn']`), pair counts drop 5 → 2+3
exactly as the 9-frame band predicts, the band keys appear, and `graph_on=1` — meaning the
adversarial and distillation gradients now travel one sub-graph, which is the whole point.
It also removes the real/fake action-lag asymmetry, since the action origin is re-based for
both sides.

**But it changed what noise level the discriminator looks at, and not in our favour.**

| path | disc timestep |
|---|---|
| flash (w1/w2 and every arm so far) | **t = 60** — lightly noised, texture largely intact |
| dmd band (this arm, and `v6alt`) | **t = 1000** on the first reading — essentially pure noise |

If the band's rungs sit high on the ladder for most of training, divergence 3 **improves
the alignment signal while making the disc structurally blinder to texture than it already
was.** That is directly opposed to your style objective. It also rhymes with today's
wavelet result: a disc whose input carries no discriminative content collapses to chance —
high noise is another route to the same place.

I have asked for the full `ladd_disc_t` distribution across the run (not just the first
reading), whether the disc separates better on low-t steps than high-t ones, and — the
actionable one — **whether divergence 3 can be combined with a low/clean disc timestep at
all, or whether the band's timestep overrides `ladd_disc_force_clean` / `ladd_disc_sample_t`
by construction.**

> If they are mutually exclusive, we must choose between better action-alignment and any
> texture pressure — and that affects the queued `v6alt`, `6t` and `carnold` arms. I will
> report the precedence answer before recommending anything.

---

# CORRECTION (21:00) — "w1 has a FLAT gap" was WRONG. ALL THREE saturate late.

I recommended w1 as the codebase default partly on the claim that it was "the only rung
whose logit gap did not trend upward (+1.69 → +1.75, flat)". **That was an artefact of
splitting the run in half and taking two medians.** The full trajectories:

```
w1  gap: 0.23 1.79 0.58 0.76 1.20 1.80 1.87 3.45 1.69 1.72 1.68 1.76 0.90 1.74 4.16 5.41 5.28
    d_loss:0.59 0.18 0.47 0.41 0.30 0.19 0.21 0.06 0.20 0.21 0.23 0.24 0.37 0.19 0.03 0.02 0.01
w2  gap: ... 3.75 5.49 4.21   (d_loss 0.05 0.01 0.04)
w4  gap: ... 3.50 3.76 2.35   (d_loss 0.05 0.06 0.16)
```

**w1 ends at gap +5.28 with `d_loss` 0.01 — that is the saturation signature**, the same
profile I called out as a warning on the 6all arms (d_loss → 0.001, gap > 6). It is not
flat; it is flat *in the middle* and then climbs hard from about step 160.

Three things follow, and one of them is uncomfortable:
1. **All three rungs saturate late.** w1 and w2 both reach gap ~5 with `d_loss` ~0.01-0.04
   by step 180-190. w4 is actually the *least* saturated at the end (gap 2.35, d_loss 0.16)
   — the opposite of what I told you when I called w4 "too hot".
2. **My w1-vs-w2 recommendation rested on a statistic that did not survive inspection.**
   The choice of w1 may still be right — it has the lowest gradient ratio and you approved
   it as the default — but "it doesn't saturate" is not a reason I can stand behind.
3. **The disc saturating by step ~180 is a finding in its own right**, and it applies to the
   *production* arms, which run 500 steps — 2.6× further than anything we measured today.
   Whatever these arms do after step 200 is unobserved.

This also raises the stakes on **Q1**: I argued R1 parity was not worth the risk because the
gamma-10 default already gave a 10× stronger dose. If every rung saturates by step 180 at
200 steps, the arms' 5×-weaker R1 over 500 steps is a materially worse bet than I implied.
> **I am not reversing my Q1 recommendation unilaterally — but you should know the
> evidence behind it weakened. If you want R1 parity (`ladd_r1_unified_cadence=true`) on
> the eight arms, say so and I will flip it.**

*Credit where due: a subagent caught this by reading the full series when I had only
compared half-medians, and it also retracted its own incorrect claim about local wandb
telemetry in the same message.*

---

# MAE GATE: **NO** — the premise is falsified by our own data (18:50)

You said it "shouldn't matter but there is a chance". The data says it would actively harm
us, and the reason is interesting. Full analysis: `analysis/gan_tuning/MAE_GATE.md`.

**The ratio sits at parity**: pooled over 9 completed runs (n=100 logged steps),
**median r = 1.033, and 48% of steps have r < 1.0**. On the face of it that is the
"gate would rebalance hard" case — at the shape already in our configs it would give a
median gate weight of **0.078**, i.e. DMD effectively switched off on half the steps, and a
projected `gan_dmd_grad_ratio` of about **3.7** — far past the 0.985 we already know is
unstable at `gan_loss_weight=4.0`.

**But the real finding is that `r` is not measuring what the gate assumes.** Split by
rollout depth: depth 1 → median r **1.373**, *none* of those steps near parity; depth 3 →
median r **0.9935**, *all* of them near parity. A clean 0%/100% split. `r` is a
**rollout-depth / timestep switch, not a student-mastery signal** — `m_real` and `m_fake`
rise *together* (0.22→0.67 and 0.35→0.66), the lockstep artefact the codebase already flags
in two places. So the gate would zero DMD precisely on the accumulated-drift chunks, which
is where we least want to remove the teacher.

**And my hypothesis is directly falsified.** I argued DMD might be drowning the GAN where
the student has caught up. If so, `gan_dmd_grad_ratio` would be *lower* on the r≈1 steps.
It is not: w2 0.273 deep vs 0.303 shallow; w4 1.135 vs 0.503; v6f 0.101 vs 0.074 — no
consistent sign. Worse for your actual complaint, `tripwire_hf_power` is *higher* on the
gate-closing steps, so the gate would add adversarial share where HF energy is already
highest — the opposite of a texture fix. No knob setting rescues it: the gate is monotone
in r and r is monotone in depth.

Two useful by-products: the gate **is** genuinely consumed (traced to
`dmd_loss = dmd_loss * _gate_w`), and our tuning arms already run it in a deliberately
instrumented-but-inert mode (`min_weight=1.0`), which is why this analysis was possible at
all without a single new run.

---

# BACKBONE COMPARISON: FIRST NUMBERS, AND THE REAL WIN IS THE CADENCE (18:47)

First `t_sep` readings with the teacher properly fed (`updates_per_step` 1→8, lr 2e-4→5e-4):

| teacher | early `t_sep` | same point, SAM2 at updates=1 |
|---|---|---|
| ConvNeXt | **+0.154** | +0.000 … +0.028 |
| DINOv2 | **+0.179** | +0.000 … +0.028 |

**Be careful how you read this — I changed two things at once.** These arms differ from the
SAM2 run in *both* backbone *and* teacher-update cadence, so I cannot yet attribute the
5–10× improvement to the feature basis. What it does show cleanly is that **the cadence fix
works**: a teacher fed 8 updates per generator step separates an order of magnitude sooner
than one fed 1.

To close the three-way comparison honestly I have set SAM2 to the same cadence and
resubmitted all three at full length.

**All submitted as ordinary batch jobs, not holder runs** — the 4 extra holders have been
starved at priority-1 for 4–5 hours with no sign of starting, so waiting on them was
costing us the day. Jobs **6133541** (ConvNeXt), **6133542** (DINOv2), **6133543** (SAM2 at
matched cadence), **6133544** (`w2tclean`), **6133545** (`w2tsamp`), **6133546**
(`smk_v6fmit`, which gates the 8 production arms).

---

# CORRECTION (18:35): MY "SAM2 IS THE WRONG TEACHER" ANALYSIS WAS PREMATURE

I told you SAM2's weak separation (`t_sep` 0.03–0.27) was because a segmentation model is
inductively blind to texture. **That may still be true, but it is not what the data shows,
because the teacher barely trained at all.**

The SAM2 agent's final report found the real bottleneck:
- The teacher heads are **zero-init**, and
- `surrogate_sam2_updates_per_step=1` combined with distillation running on the
  *generator* cadence (`dfake_gen_update_ratio=5`) meant the teacher received about
  **19 updates in 115 steps**. Its `d_loss` moved 1.386 → 1.260 — i.e. it had barely begun
  to learn.

A head with 19 updates from zero-init will show near-zero separation **whatever encoder
sits behind it**. So the low `t_sep` is explained by undertraining, and my backbone
argument is unproven rather than supported.

**This nearly produced a wrong conclusion on your instruction.** The ConvNeXt and DINOv2
arms I launched at your request both carried `surrogate_sam2_updates_per_step=1` (the same
key governs all three backbones — `if backbone in ("sam2","dinov2","convnext")`). They
would have shown the same weak separation, and I would have reported "no pretrained teacher
helps", which would have been an artefact of a cadence flag.

**Fixed:** both relaunched with `updates_per_step=1 → 8` and `lr 2e-4 → 5e-4`. Their
holders only had ~23 min left, so these are short runs (~30 steps) — but ~52 teacher
updates versus SAM2's 19, which is what the comparison actually needs. If `t_sep` still
sits near zero with a properly-fed teacher, *then* the backbone-inductive-bias argument
earns its keep.

**Also confirmed, and it closes Q5:** the surrogate genuinely **was** consumed — proven at
every generator step from 31 to 111 by `surrogate_consumed=1`, `pix_g_applied=1`,
`pix_g_via_gen_gan_loss=1`, and a growing `surrogate_g_weighted` (6.4e-05 → 7.0e-04). The
"never consumed" warning is guarded on the `pixel` backbone and does not apply here. No
relaunch was needed for that reason, and `gan_pixel_texture_enabled` correctly stayed off.
Its cost profile is also now known: **+16 GB memory at matched steps, but only +0.1 s/step**
— the surrogate's price is memory, not time.

---

# NOTED: THE MAE GATE (18:25) — under analysis, and the hypothesis is credible

You asked for a note, so here it is with my reasoning; an agent is testing it against data
now and will build the arm if the data supports it.

**What it is.** `dmd_mae_gate_enabled` (default OFF) computes the teacher's MAE (`m_real`)
and the student's (`m_fake`) and their ratio `r`. The gate weight ramps 0→1 as `r` goes
1→`r_full`, floored at `min_weight`. In words: **when the student has caught up with the
teacher (r→1), DMD is attenuated; when the student is much worse (r→r_full), DMD runs at
full strength.**

**Why it could matter for exactly our problem.** This is a *different lever from
`gan_loss_weight`*. We know the weight knob goes unstable by 4.0 (adversarial gradient
exceeding DMD, and you saw it fail visually). The gate instead attenuates DMD **only where
the student is already teacher-equivalent** — which is precisely where DMD has nothing
left to teach and where the GAN's contribution is currently being drowned. That would raise
the effective adversarial share *without* raising the weight, and it would do so
selectively rather than globally.

That maps onto your observation: the teacher can supply geometry (it knows the scene), but
it cannot supply detail it does not itself have. If DMD dominates the update in exactly the
regions where the student already matches the teacher, texture is where the GAN would have
had something to say and was outvoted.

**The decisive check, which is why I have not just built it:** the gate logs `m_real`,
`m_fake` and `ratio` *even when disabled*, so we may already have the answer in today's
completed 200-step runs. If `r` sits far above `r_full` in practice, the gate is a no-op
and the idea is dead. If `r` hovers near 1, it would substantially rebalance DMD against
the GAN. I have asked for that distribution before anything is built.

There is also an existing interaction to respect: the trainer compares
`gan_loss_weight × ladd_disc_loss_weight` against `dmd_base × gate_w` and raises/warns when
the gated DMD falls below the GAN term (`gan_gate_couple_min_ratio`), so some settings are
already illegal by design. That guard is being read before any value is chosen.

> Verdict and (if supported) `gantune_w2mae` to follow. I will not build it if the ratio
> data says the gate would be inert.

---

# WAVELET FINAL: DEAD IN BOTH CONFIGURATIONS (18:20) — and it explains the smoothing you saw

You called it before the numbers did. `gantune_w2wav2` (LL **and** HH dropped, directional
bands only) died the same way, just more slowly:

| step | d_real | d_fake | separation | d_loss | backbone grad |
|---|---|---|---|---|---|
| 81 | −0.7333 | −0.7371 | **0.004** | 0.6914 | 0.18 |
| 91 | −0.7256 | −0.7260 | **0.0004** | 0.6930 | 0.11 |

`d_loss` = ln(2) again, i.e. chance. It survived past step 51 where the first version
collapsed, so dropping HH *did* help — but only by delaying the collapse to ~step 91.
Plain w2 at the same steps runs a backbone gradient of 96–376; this is 0.1–0.2.

**Why it also went smoother, which is the interesting part:** a discriminator restricted to
directional edge bands is most cheaply satisfied by producing *fewer, cleaner edges*. The
generator's easiest route to matching an edge-statistics critic is to suppress
high-frequency content rather than to get it right. Combine that with a critic supplying
almost no gradient and you get exactly what you observed — smoother, less resolved.

**Conclusion: wavelet-HF is a dead end in both tested configurations.** Not worth a third
attempt without a different mechanism (e.g. much lower disc noise, which `w2tclean` tests
separately).

---

# WHY SAM2 IS NOT RAISING RESOLUTION (18:05) — analysed, and I think it is the wrong teacher

**First, the good news: the surrogate is genuinely alive and consumed.** I was wrong to
suspect a no-op. The trainer's "never consumed" warning is guarded on
`teacher_backbone=="pixel"`; with `sam2` it runs surrogate-only mode, admitted at all three
gates, and the run proves it: `PRETRAINED surrogate teacher built`, plus live per-step
`t_dloss` / `t_sep` / `c_vloss` / `c_corr`. So Q5 is answered — no relaunch needed.

**Now the actual problem. The teacher can barely tell real from fake.**
`t_sep` is the SAM2 teacher's real-vs-fake separation:

```
t_sep:  +0.000  +0.014  +0.028  +0.032  +0.023  +0.076  +0.137  +0.179  +0.031  +0.270
c_corr: +0.000  -0.506  +0.024  -0.099  +0.638  -0.570  +0.714  -0.613  +0.760  +0.638
```

Separation of **0.03–0.27** is nearly nothing — compare the LADD disc on the same run,
which separates by **1.0–2.5**. The latent critic tracks the teacher acceptably
(`c_corr` peaks 0.71–0.76, though it oscillates sign, which is its own concern), so the
distillation is not the bottleneck. **The bottleneck is that the teacher has almost no
signal to distil.**

**My explanation, and it is a design-level objection rather than a tuning one: SAM2 is a
segmentation model.** Its encoder is trained to produce features that identify *what and
where objects are* — and to be **invariant to appearance detail**, because a blurry car and
a sharp car must segment identically. We asked it to judge precisely the property it was
trained to ignore. That predicts exactly what we measured: near-zero separation, and
therefore no texture pressure on the generator. It is the same failure as the latent disc
(good at structure, blind to sharpness) wearing a different hat.

**The fix is already in the tree and is one flag.** `model/pretrained_pixel_disc.py`
provides `dinov2` and `convnext` as drop-in alternatives with *the same ADM heads*, and its
own comment says they exist "so an arm-to-arm delta is attributable to the FEATURE BASIS
alone". **ConvNeXt is the one I would try**: ImageNet-trained CNNs are famously
*texture-biased* classifiers — the opposite inductive bias to a segmenter, and precisely
what a sharpness critic wants. DINOv2 is the second choice (richer appearance features than
a segmenter, more semantic than a CNN).

> **RECOMMENDATION: run `surrogate_teacher_backbone=convnext` next, not more SAM2.**
> Same arm otherwise, so the delta isolates the feature basis. I will build it on request.

*Caveat kept honest:* SAM2 also had its resolution halved (512 → 256) after the first
attempt OOM'd, so its texture sensitivity is additionally handicapped. But a 0.03
separation is far too low to blame on resolution alone.

---

# QUEUED: THE DISC-TIMESTEP AXIS (17:50) — two arms, ready for the next holders

Built and waiting: **`gantune_w2tclean`** and **`gantune_w2tsamp`**. Both are `gantune_w2`
with exactly one thing changed, so each is attributable, and both compare against the
completed 200-step `gantune_w2` baseline.

**Why this axis.** The disc currently scores latents at a single fixed `t=60`. The
trainer's own comment says what that costs:
> "The fixed `flash_t` path leaves D solving **one narrow low-noise classification
> problem**; opt-in sampling exposes the transition critic to the diffusion interval."

That is the mechanism I suspect is behind "the GAN fixes 3D but not resolution": a
discriminator trained at one noise point learns whatever separates real from fake *at that
point*, and nothing else.

| arm | change | question it answers |
|---|---|---|
| `gantune_w2tclean` | `ladd_disc_force_clean=true` (disc sees **clean** latents, t=0) | can the disc discriminate texture at all when noise is not hiding it? |
| `gantune_w2tsamp` | `ladd_disc_sample_t=true`, t ∈ [20, 400], shift 5.0 | does exposing D to a *range* of noise (Diffusion-GAN) beat one point? |

I verified both flags are genuinely consumed (`disc_t_int` is computed from them and used
to noise real and fake identically) rather than dead keys — two config keys turned out to
be dead code today, so I no longer trust a flag until I have traced it.

I chose t ∈ [20, 400] rather than the full [20, 980] default: biased toward the cleaner
half, where texture still exists, while still spanning a range.

## …and this CORRECTS my wavelet explanation again
`trainer:12098-12108`: **wavelet-ON forces `disc_t_int = 0`.** The wavelet disc was already
seeing perfectly clean latents:
> "Wavelet-HF has strict precedence: its sub-bands must see clean latents. Sampling after
> the clean-wavelet branch used to silently undo that invariant and turn HF discrimination
> into broadband-noise discrimination."

So my first explanation — "HF at t=60 is dominated by diffusion noise" — was **wrong twice
over**: wrong because I had run the wrong band configuration, and wrong because the wavelet
path never saw noised input in the first place. The HH common-mode explanation from
`model/wavelet_hf.py` is the correct one and now stands alone.

A useful consequence: `gantune_w2wav2` (running now) is *already* a clean-latent disc test,
so if it works we learn about clean input and directional-HF together, and `w2tclean`
separates the two.

---

# WAVELET, PROPERLY EXPLAINED (17:30) — I RAN THE WRONG VARIANT

My "noised latents make HF meaningless" explanation below was a guess, and it was wrong.
The real answer was already written in this codebase, in `model/wavelet_hf.py`:

> `drop_hh` (2026-08-19): also drop the DIAGONAL high-frequency band. HH is the noisiest,
> least structured sub-band — it carries the checkerboard/diagonal component that **has no
> clean analogue in smoothed WAN latents**. The wavelet-ON discriminator was measured dead
> (**d_real == d_fake, d_loss = log2, common-mode drift**) and **HH is the leading suspect**
> for that common-mode term, so this isolates it while keeping the directional HF bands.

So: the exact failure I measured today — `d_real == d_fake`, `d_loss = ln(2)` — had already
been observed, diagnosed, and a flag (`ladd_wavelet_hf_drop_hh`) added specifically to test
the suspected cause. **I did not set that flag.** I set `drop_ll=true` only, which leaves
the band set as **LH + HL + HH** — including the suspect. I ran the test that was already
known to fail.

**Why HH breaks it, mechanically:** the Haar HH band is the diagonal/checkerboard
component. WAN latents are spatially smooth, so neither real nor generated latents carry
meaningful diagonal structure — HH is close to pure noise on *both* branches. Feeding it
to the disc adds a large common-mode term identical in real and fake, which swamps the
directional bands that *do* differ. The disc's output stops depending on which branch it
is looking at: `d_real == d_fake` exactly, loss pinned at chance. That is precisely the
signature we measured, and it explains the near-zero backbone gradient (0.10 vs plain w2's
96–376) — the loss barely depends on the features at all.

**The correct test** — which the docstring spells out — is `drop_ll=true` **AND**
`drop_hh=true`, leaving only the two DIRECTIONAL bands: LH (vertical edges) and HL
(horizontal edges). That is real edge/texture information with genuine real-vs-fake
content, minus the noise band.

Built as **`sbatch/gantune_w2wav2.sbatch`**, queued for the next free holder. I did not
displace `gantune_w2carn` (which you asked for and is running) to make room.

> **Note the ll_weight warning also in that file:** LL at weight ≥0.5 pushes the disc's
> spectral-norm power iteration into oscillation and *silently hangs the next collective*.
> If we ever put LL back, keep `ll_weight ≤ 0.3`.

---

# EARLIER (SUPERSEDED IN PART): WAVELET-HF DEAD WITH LL OUT — the measurement stands, my explanation did not (17:20)

Your "keep LL out" idea was the right test to run, and it came back negative and clean.
`gantune_w2wav` vs plain `gantune_w2`, same base, same seed schedule, matched steps:

| step | w2wav (wavelet HF, LL dropped) | plain w2 |
|---|---|---|
| 31 | gap 0.089, d_loss 0.651, backbone grad **1.5** | gap 0.234, d_loss 0.594, backbone grad **376** |
| 41 | gap 0.113, d_loss 0.639, backbone grad **0.10** | gap 1.586, d_loss 0.208, backbone grad **96** |
| 51 | gap **0.000**, d_loss **0.6932** | — |

At step 51 `d_real` and `d_fake` are **identical** and `d_loss` is **ln(2) to four
decimals** — a discriminator at exactly chance. The backbone gradient runs 100–1000x
weaker than plain w2 the whole way. The wavelet stage was genuinely active
(`wavelet_hf=True`, params 5.58M → 5.59M), so this is not a silent no-op.

**Conclusion: the wavelet HF stage kills the GAN on its own. LL was not the poison.**
That extends the earlier finding rather than overturning it.

**Likely mechanism, and it matters for the resolution question generally:** the disc reads
latents at **t=60**, i.e. already substantially noised. The high-frequency bands of a
noised latent are *dominated by the diffusion noise itself* — the HF content largely IS
the noise. Strip LL and you hand the discriminator a view that is nearly pure noise in
both branches, so it cannot separate real from fake and sits at chance. This predicts
wavelet-HF can only work at much lower `t`, which is a different recipe, not a flag.

**Consequence for resolution:** frequency-domain tricks on noised latents look like a dead
end. That leaves the **pixel-domain route** — the SAM2 surrogate now running, and the
built-but-undeployed pixel PatchGAN — as the live options.

I stopped the dead run at step 51 rather than let it burn 90 more minutes, and reused its
holder for **`gantune_w2carn`** (the w2 optimum + CARN at its matched optimum) which you
asked for earlier and had no slot.

---

# LIVE STATUS

## S0. w1 IS NOW THE CODEBASE GAN DEFAULT (your directive, 16:55)
Applied in two places:
- **`configs/action_forcing_phase3_dmd.yaml`** — `gan_loss_weight: 0.05 → 1.0` and
  `ladd_r1_gamma: 1.0 → 10.0`, with the measurement recorded in-file so nobody
  re-litigates it. This changes the default for **every** run in the codebase from now on.
- **All 10 CARN arms + 2 smokes** — same two values (they were 0.3 / 1.0). Everything else
  already matched w1: disc start 20, warmup 25, 5 updates/step, backbone scale 0.2,
  `r1_every_n=1`. Arms resubmitted: **6132423–6132432**.

**Note the honest caveat on which rung I made default.** You asked for w1, and w1 is
defensible on the strongest ground available: over 200 steps it was the **only rung whose
logit gap did NOT trend upward** (+1.69 → +1.75, flat), where w2 went +0.61 → +2.51 and
w4 +0.99 → +3.05. On the arms' 500-step horizon that stability matters more than peak
signal. w2 remains the stronger *instantaneous* signal (ratio 0.283 vs 0.156, and the
tightest spread), so if the long arms show the GAN under-contributing rather than
saturating, w2 is the upgrade path.

Reverting is one line in the yaml plus a resubmit, and the tuning arms
(`gantune_w2wav`, `gantune_w2sam`, `gantune_w2carn`) deliberately stay at w2 so the
comparison stays live.

## S1. GAN weight ladder — w=2.0 leading, final verdict ~16:45
Three runs, differing by **exactly one number** (adversarial weight 1.0 / 2.0 / 4.0);
everything else byte-identical, CARN off so the GAN is the only moving part.

| weight | n | ratio median | ratio IQR | d_loss | logit gap | cos |
|---|---|---|---|---|---|---|
| 1.0 | 5 | 0.156 | [0.052, 0.678] | 0.296 | +1.20 | −0.011 |
| **2.0** | 4 | **0.257** | **[0.238, 0.291]** | 0.420 | +0.72 | +0.018 |
| 4.0 | 3 | 1.100 | [0.128, 1.300] | 0.493 | +0.51 | −0.028 |

**w=2.0 is the candidate:** strong signal *and* a tight spread, where w=1 scatters over
13× and w=4 has gone hot (adversarial gradient exceeding the DMD gradient).
**Nice surprise:** higher weight *reduces* saturation (`d_loss` rises, gap falls) —
`gan_loss_weight` never enters the D update at all, so it cannot saturate the disc.
Timing risk: w=4's holder has ~2 min of slack; it may be wall-clocked a few samples short.

## S2. Ten production arms queued: 6131750–6131759
Carrying the matching fix (R7) and your approved `ladd_r1_every_n_steps=1`. Their
scripts *also* now carry the five new code fixes' flags, but **I have not resubmitted
them for that** — those seven flags have never run on GPU and I will not put untested
config on production arms. `smk_v6fmit` is the gate; arms resubmit after it passes.

## S3. Five code fixes implemented, 479 tests green, awaiting GPU validation
New `srcmd5 = 5c8faae419ec9e294cb23733daa34c6e`. Write-up:
`analysis/gan_tuning/CODE_FIXES.md`.
1. **G-side checkpoint recovery** (~30 GiB back) — proven the checkpoint arms and that
   the disc receives **no** generator gradient, each with a mutation control.
   *Caveat:* with the flag ON it is not bit-identical (~1e-7 relative, from kernel
   dispatch, not the checkpoint). OFF is byte-identical.
2. **Action origin tracks the latent slice** — your "fix the fundamentals". Arithmetic
   test proves the disc's action rows now cover exactly the frames of its latent chunk,
   and are wrong on rolls ≥2 with the flag off. New `ladd_act_lag` telemetry reads 0 when
   correct.
3. **R1 telemetry** — `r1_rate`, `r1_fires`, and `hold_step` so the 5-step deferred lag
   is self-describing. Default line asserted byte-identical as an exact string.
4. **Match distance logged** — including the money metric `ladd_match_dist_ratio`
   (chosen ÷ pool median; 1.0 = no better than random). Pool cap now configurable;
   raising it costs **zero** extra disc memory, but only raise it with
   `ladd_lazy_cand_disc=true`.
5. **Backbone grad-scale checker** — report-only, never auto-corrects.

## S4. Queued and armed, waiting purely on holders
All four new holders still PENDING. In launch order once the ladder frees three (~16:45):
1. **`smk_v6fmit`** — the gate for the 10 arms. Acceptance: peak vs v6f's 89.8 GB;
   `[FN-GT-FORMER]` firing with **non-zero levels** (all-zero was the bug);
   `ladd_act_lag=0`; `r1_rate` present; first-ever `ladd_match_dist_ratio` reading.
2. **`gansig_of`** — fixes the real/fake action asymmetry on both sides.
3. **`gansig_huge`** — real set 5 → 60 rows/step/rank, pool 5 → 278, **rides 1 → ~33**
   via `ladd_real_pool_cross_ride` (CPU ring, zero GPU cost), all memory guards on.

---

# ACTIVE CAVEATS

- **Metric noise.** `gan_dmd_grad_ratio` is a single-parameter estimate; at fixed config
  it has spanned 190×. **Never quote a single sample** — I did that twice today and was
  wrong twice. Use the median past step 45.
- **`cos` is ~0 everywhere and that is normal.** Every configuration we have run sits
  between −0.07 and +0.25. There is no "+0.69 healthy reference" — that was a single
  sample I misquoted. Near-zero arguably *is* what we want: +1 would mean the adversarial
  gradient is redundant with DMD, <0 would mean it fights it.
- **Memory baseline correction, and it is broader than one number.** The "49.2 GB" figure
  I quoted repeatedly was a step-5 reading; the true non-CARN baseline is **64.89 GB**.
  That artefact propagated: the claim "stripping CARN bought ~14 GB" is the same error —
  compared like-for-like at GAN-active steps, CARN and non-CARN peaks are roughly equal.
  The **+4.9 GB** cost of a second pair mode survives; the absolute ceilings and the
  "~30 GB margin" claim do not.
- **Peaks keep climbing with step count.** The 69.76 GB real-data reading is from step 30
  and that run never passed step 31; the tuning base climbs 49.4 → 64.89 GB between steps
  15 and 60. **Any peak quoted from an early step understates a 200-step run**, which is
  exactly how I got the 49.2 wrong. Treat every memory number with its step attached.
- **Two arms already died today** (v5b, v6f) at *step 0* with a foreign process holding
  40–55 GB — the known holder launch race, not our budget. But a 90 GB peak on a 95 GB
  card is what turned a survivable race fatal, so the ~65 GB target matters.
- **Silent-failure count is now 12.** Today added: the inert CARN (`[FN-GT-FORMER]` never
  fired in 6 of 8 arms) and dead-code config keys (`ladd_*_n_real` are read and then used
  by nothing but an error check). Both pass our override-scanner and both print happily in
  a config echo. **Standing recommendation (not built, needs sign-off):** every
  config-enabled feature registers a "fired at least once" counter and the run fails loudly
  if any is zero by step 50.
- **D5 abort keys — accepted, and I am watching.** `r3gan_d_*_gtxn` and
  `dmd_mae_gate_m_fake` over the arms' first 100 steps; abort signature is `d_loss`
  collapsing toward 0.001 with the logit gap climbing past ~6 (the 6all profile).
