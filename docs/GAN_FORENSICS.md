# GAN Forensics: rolling DMD + R3GAN/LADD discriminator

Date: 2026-08-21. Data: wandb datastores parsed directly
(`wandb.sdk.internal.datastore` + `wandb_internal_pb2`, history records,
`value_json`). Code read at the current working tree (branch `dmd_one_step`).

Two parts:

* **Part A — forensics**: d_loss / d_real / d_fake / g_loss trajectories for 6
  rolling runs, correlation of disc strength with the known breakdown window,
  and a concrete safe operating point.
* **Part B — redesign**: exactly what the disc sees today under ROLLING, and 5
  candidate rolling-GAN designs ranked by implementation cost vs expected
  effect on the cartoon/HF-amplification failure.

---

## Run inventory

All six runs auto-resumed from a `phase1_step0000200.pt` warmstart in their own
log dir, so wandb history starts at step ~201 and the disc is near-fresh at
step 211 in every run (d_loss ≈ ln2 at the first logged point). Scalars are
logged every ~10 steps; `gan_updates_per_step=5` D-updates fire on every
generator iter (`dfake_gen_update_ratio=5` → gen iter every 5th trainer step).

| run | wandb dir | R1 | wavelet HF | notes |
|---|---|---|---|---|
| rollcarn | `run-20260820_223449-mpcnyduh` | γ=1e6, every=1, on **all 5** D-updates | ON (augment, drop_ll) | disc lobotomized; rated **GOOD** |
| rollgansure | `run-20260820_230813-nuth1j33` | γ=0 (no R1) | OFF (disc input noised t=60) | fastest disc win |
| rollg1e3 | `run-20260821_002059-fmtlav34` | γ=1e3, every=5, once/step | OFF (disc t=60) | plateau, intermittent strain |
| rollcombo2 | `run-20260821_011922-yxhnh783` | γ=1e3, every=5, once/step | ON (disc t=0) | + carn_seam_affine 0.5 |
| rollife2 | `run-20260821_084855-ao4gk96h` | γ=1e3, every=5, once/step | ON (disc t=0) | ~= rollcombo2 |
| rolllong | `run-20260821_162525-4z2988ef` | γ=1e3, every=5, once/step | ON (disc t=0) | 700-step target; **broke ~466** |

Shared GAN config across all six: `gan_loss_weight=1.0`, `gan_lr=2e-5`
(AdamW), `gan_updates_per_step=5`, `flash_dmd_gan_t=60`,
`ladd_gt_transition_enabled=true`, `ladd_gt_transition_match=true` (K=4,
pool M=8), `ladd_disc_loss_weight=1.0`, warmup done long before step 200 so
the gen-side weight (`r3gan_g_weight_gtxn`) is **1.0 throughout**.

Config-matrix takeaway: the ONLY deltas between the good run and the broken
runs are R1 gamma/cadence and wavelet on/off. Wavelet on/off also silently
flips the disc input noise level (see Part B): wavelet ON forces clean disc
inputs (t=0), wavelet OFF noises them at t=60 — a hidden confound between the
arms.

rolllong resume anomaly (worth knowing, not the cause): its step-200 disc
checkpoint didn't match the current head shapes — `r3gan_discriminator
missing=17 shape_dropped=15` (`heads.*.cls.*`) and the r3gan optimizer
restarted fresh. So rolllong's disc *cls layers* started from init at 201.
Its d_loss trajectory nevertheless tracks rollcombo2/rollife2 almost exactly,
so this changed timing at most marginally.

---

# PART A — Forensics

## A.0 Reading the metrics

`d_loss` is relativistic RpGAN: `softplus(D(fake) − D(real))` averaged over
matched pairs and tokens (`model/r3gan.py:231`). Useful dictionary:

| d_loss | mean logit gap D(real)−D(fake) | meaning |
|---|---|---|
| 0.693 (ln 2) | 0.0 | disc cannot separate — dead or balanced |
| 0.40 | ~0.7 | mild separation |
| 0.30 | ~1.05 | disc clearly winning |
| 0.20 | ~1.5 | strong |
| 0.10 | ~2.25 | very strong — the "too strong" suspicion is right |
| 0.04 | ~3.2 | runaway |

`g_loss_raw` = `softplus(D(real) − D(fake))` on the gen side, so g_loss ≈
0.69 means zero gen pressure and g_loss 3–5 means the gen is being pushed with
logit gaps of 3–5 — enormous gradients through 5 transition pairs × per-token
logits.

## A.1 Per-run trajectories (condensed)

### rollcarn — γ=1e6 every update: the disc is dead, and that is the run rated GOOD

d_loss is pinned at 0.6930–0.6933 for 210+ steps; |d_real|, |d_fake| < 1e-3;
r1_grad_sq decays 1.6e-6 → 1e-10 (R1 at γ=1e6 fired on *every one* of the 5
D-updates per gen iter drives the disc to a constant function). g_loss ≈
0.6932 always → the GAN contributes ~**zero gradient** to the generator.
DMD loss stays 0.1–1.0, gen grad norms normal, student MAE 0.34–0.73 with no
trend. **The "good" recipe is functionally GAN-OFF.** Every visually-good
rolling run was pure DMD + stat anchors wearing a GAN costume.

### rollgansure — γ=0: disc wins in ~100 steps, student breaks at ~301–321

| step | d_loss | d_real | d_fake | g_loss | dmd_loss | student MAE |
|---|---|---|---|---|---|---|
| 211 | 0.624 | 0.06 | −0.08 | 0.74 | 0.22 | 0.44 |
| 261 | 0.151 | 1.03 | −0.93 | 1.77 | 0.44 | 0.68 |
| 301 | 0.463 | 1.06 | +0.38 | 1.23 | **8.2** | 0.75 |
| 321 | 0.064 | 1.40 | −1.39 | 1.95 | **5.7** | 1.26 |
| 331 | 0.075 | 1.81 | −1.34 | 3.40 | **14.2** | 0.74 |
| 351 | 0.017 | 2.05 | −2.16 | 2.14 | **16.0** | 0.80 |
| 371 | 0.016 | 1.97 | −2.35 | 2.91 | **19.1** | 0.87 (gen grad spike 318) |

Note the step-301 event: d_fake flips positive (+0.38) — the generator briefly
found an adversarial hole — then the disc re-sharpens and crushes it. From
d_loss < 0.10 sustained (~321) the DMD loss explodes an order of magnitude and
never recovers.

### rollg1e3 — γ=1e3@5: plateau at d_loss 0.2–0.4, strain but no runaway (by 371)

d_loss declines to ~0.33 by 261 then oscillates 0.21–0.42 through 371. DMD
spikes at 301 (6.1), 331 (4.5), 351 (11.0) with partial recovery between. R1
here actually holds the disc near a noisy equilibrium — but the student is
already paying (the spikes), and this is one of the arms rated
cartoon/texture-artifacted. Wavelet OFF (disc noised at t=60).

### rollcombo2 / rollife2 — γ=1e3@5 + wavelet: monotone slide, run ended before the cliff

Both: 0.63 → ~0.30 by 281 → 0.10–0.20 by 351–371 (end of run). No DMD
explosion yet, but d_loss is on exactly the trajectory rolllong followed into
its breakdown. These runs simply stopped at ~375 before the disc finished
winning.

### rolllong — the 700-step run: breakdown is the disc winning at ~466

| step | d_loss | d_real | d_fake | g_loss | dmd_loss | MAE | critic gn |
|---|---|---|---|---|---|---|---|
| 211 | 0.632 | 0.04 | −0.09 | 0.73 | 0.97 | 0.41 | 0.53 |
| 291 | 0.220 | 0.75 | −0.73 | 1.32 | 0.49 | 0.51 | 0.71 |
| 321 | 0.160 | 1.03 | −0.89 | 1.46 | 0.12 | 0.44 | 0.48 |
| 341 | 0.644 | 0.83 | **+0.69** | 1.19 | 0.08 | 0.33 | 0.77 |
| 401 | 0.480 | 0.88 | **+0.17** | 1.34 | 0.21 | 0.45 | 0.69 |
| 441 | 0.472 | 1.69 | **+0.70** | 1.57 | 0.59 | 0.68 | 0.90 |
| 451 | 0.486 | 1.61 | **+0.97** | 1.26 | 2.52 | 0.62 | 1.09 |
| 461 | 0.255 | 1.03 | −0.42 | 1.18 | 1.37 | 0.60 | 0.67 (**r1_grad_sq peak 0.086**) |
| 471 | 0.088 | 1.25 | −1.30 | 1.49 | 2.39 | **1.01** | 1.82 |
| 481 | 0.057 | 1.67 | −1.41 | 2.66 | 6.10 | 0.82 | 2.14 |
| 491 | 0.037 | 1.76 | −1.74 | 1.89 | **14.8** | 0.94 | 2.89 |
| 501 | 0.017 | 2.32 | −2.17 | **4.57** | 8.49 | 0.77 | 2.48 |
| 511 | 0.621 | 2.36 | **+2.18** | 1.42 | 6.55 | 0.78 | 1.52 |
| 521 | 0.046 | 2.06 | −1.31 | 2.61 | **15.9** | 0.84 | 4.72 |
| 531 | 0.019 | 2.21 | −1.93 | 3.04 | **33.5** | 0.88 | 2.67 |
| 541 | 0.008 | 2.65 | −2.49 | 3.44 | **17.2** | 0.62 | 2.20 |
| 551–591 | 0.03–0.29 | 2.2–2.6 | mostly ≪0 | 3.8–5.1 | 1.4–6.0 | 0.51–0.70 | 0.8–1.2 |

Phases:

1. **211–330**: same monotone slide as combo2/ife2 (0.63 → 0.16).
2. **331–461 (the fight)**: R1 (γ=1e3, 1-of-5 updates) starts biting hard —
   r1_grad_sq spikes 0.008 → 0.014 → 0.029 → 0.019 → 0.086 (461), i.e. R1
   penalty values 4–43. Repeated disc-flip events (d_fake goes *positive* at
   341, 401, 441, 451): the generator keeps finding adversarial holes in an
   increasingly sharp decision boundary, the disc keeps re-sharpening. This
   oscillation is the visible "strain" precursor.
3. **466–541 (breakdown)**: the disc wins for good. d_loss sustained < 0.10
   from 471 (min 0.008). Medians pre-466 → post-466:
   d_loss **0.309 → 0.042**, generator_dmd_loss **0.42 → 6.10** (peak 33.5),
   g_loss_raw **1.27 → 3.44** (peak 5.13), student MAE-vs-GT
   **0.486 → 0.769** (peak 1.01), critic grad norm **0.64 → 2.06** (peak 4.7).
   One last generator escape at 511 (d_fake +2.18) is crushed within 10 steps.

## A.2 Cross-run threshold analysis

First step where the (0.6/0.4) EMA of d_loss crosses each threshold, vs first
DMD-loss excursion:

| run | EMA<0.3 | EMA<0.2 | EMA<0.1 | first dmd>2 | first dmd>5 | first critic gn>1.5 |
|---|---|---|---|---|---|---|
| rollcarn | never | never | never | never | never | never |
| rollgansure | 261 | 281 | 341 | 301 | 301 | 321 |
| rollg1e3 | 351 | never | never | 301* | 301* | 351 |
| rollcombo2 | 301 | 331 | never | never | never | never |
| rollife2 | 291 | 331 | never | never | never | never |
| rolllong | 291 | 321 | 501 | 451 | 481 | 471 |

*rollg1e3's dmd excursions are isolated spikes with recovery, matching its
d_loss oscillating around 0.2–0.4 rather than collapsing.

Correlations (rolllong, 39 samples): corr(log d_loss, log dmd_loss) = **−0.73**
at lag 0, decaying at positive lags (−0.60, −0.49, −0.45); corr(d_loss,
g_loss_raw) = −0.66. Disc strength and student damage move together
essentially instantaneously at the 10-step logging resolution — the disc
crossing is not merely a leading indicator, it *is* the mechanism (the GAN
gradient at weight 1.0 with a 2–3-logit gap overwhelms the DMD gradient, the
student's distribution deforms, DMD loss reads the deformation).

### Answers to the posed questions

* **Does breakdown onset track d_loss crossing a threshold?** Yes, cleanly and
  in both runs that were allowed to get there. Strain (dmd spikes, disc-flip
  oscillation) begins once EMA d_loss < ~0.3; irreversible breakdown begins
  once d_loss is *sustained* < ~0.10 (rollgansure ~321, rolllong ~471, i.e.
  exactly the reported 466 window given 10-step logging). d_loss ≈ 0.10 = a
  2.25-logit real/fake gap — the "d_loss 0.10 = too strong" suspicion is
  quantitatively confirmed.
* **Does g_loss explode?** Yes: 0.7–1.3 healthy → 2.6–5.1 in the breakdown
  window (median 3.44 post-466 in rolllong). Since gan weight is 1.0 and
  logits are per-token, this is a huge term against a DMD loss whose healthy
  magnitude is ~0.5.
* **Is there a disc-strength level that is safe?** At the current operating
  point (weight 1.0, lr 2e-5, 5 updates/gen-iter, ride-local real pool):
  **no equilibrating level was observed.** Every learning-disc arm slides
  monotonically toward the cliff; γ=1e3@5 only slows the slide (~100 steps →
  ~250 steps of GAN-active training). The one non-degrading arm (rollg1e3,
  d_loss ~0.2–0.4 plateau) still produced the cartoon/texture rating and
  intermittent dmd spikes. Conversely γ=1e6-every-update is not a working GAN
  either — it is a lobotomy (zero gradient both ways). The system as
  configured has only two attractors: dead disc (fine video, no GAN benefit)
  and winning disc (broken video).

### Why the disc always wins here (this drives the Part B redesign)

The disc's real set is tiny and ride-local: the match pool is bounded to the
active window (`dmd_42f_rolling_sup_new=true` path,
`trainer/causal_action_forcing_train.py:10476-10486`) — measured
`ladd_match_n_cand = 22` candidates per ride, `M=8`, `K=4`, ~7–11 *unique*
real transitions per D-update (`ladd_match_n_real` logs), one ride per rank.
5 D-updates per gen iter × per-token logits × spectral-norm conv heads on
frozen-teacher features is far more capacity+data-advantage than 22 candidate
windows can resist: the disc can effectively memorize this ride's GT texture
statistics, and R1's finite-difference penalty on 6 subsampled reals
(`ladd_r1_num_samples=6`, σ=0.01) only measures smoothness *at* those few
reals. The r1_grad_sq spikes and the repeated d_fake>0 escape/crush cycles in
rolllong 331–461 are the signature of a sharp, nearly-memorizing boundary.

## A.3 Proposed safe operating point

Two levels: a conservative knob-only setting for the next run, and a small
code addition (adaptive freeze) that makes disc strength self-limiting.

### Knob-only (no code change)

```
gan_loss_weight: 0.2          # was 1.0 — g_loss ~1.3 at weight 1.0 vs DMD ~0.5 is not a side dish
gan_updates_per_step: 1        # was 5 — D:G from 5:1 to 1:1
gan_lr: 5e-6                   # was 2e-5
ladd_r1_gamma: 1e3             # keep
ladd_r1_every_n_steps: 5       # keep (debt-based cadence => fires every gen iter)
ladd_r1_once_per_step: true    # keep (moot at 1 update/step)
```

Rationale: rollgansure/rolllong show the disc needs ~5× fewer effective
updates to be roughly balanced against the gen (its slide from ln2 to 0.2 took
~25 gen iters = 125 D-updates); cutting updates 5→1 and lr 4× moves its
timescale ~20× slower, and weight 0.2 caps the damage of any residual win
(g_loss 3.4 × 0.2 = 0.7 ≈ DMD scale). This alone probably converts the
monotone slide into a slow drift — but it does not *guarantee* equilibrium,
which is what the gate is for.

### Adaptive disc-freeze gate (small new code, strongly recommended)

Freeze the D-update (not the G-side term) while the disc is strong; thaw with
hysteresis:

* Maintain `self._gan_dloss_ema` (EMA over per-genstep mean d_loss, β≈0.9).
* At the top of `_run_disc_updates` (trainer ~6650): if `ema < freeze_lo`
  (default **0.45**) set `n_disc_updates=0` this step; resume only when
  `ema > thaw_hi` (default **0.60**). Log a `train/r3gan_disc_frozen` flag.
* **DDP safety (essential)**: the freeze decision must be identical on all
  ranks or the disc DDP allreduce deadlocks/diverges. All-reduce the mean
  d_loss scalar (AVG) once per gen step *before* updating the EMA, so every
  rank computes the same EMA and the same gate. This is one 4-byte allreduce
  per gen iter — free.
* Optionally couple the G-side weight to the same EMA
  (`gen_gan_weight *= clamp((ema − 0.1)/0.5, 0, 1)`) so that even while the
  disc is frozen-but-strong the generator is not hammered; this reuses the
  `_couple_gan_weight_to_gate` pattern (trainer 3840) with a d_loss input
  instead of the MAE gate.

Targets: with hysteresis at 0.45/0.60 the disc lives in the d_loss 0.45–0.65
band — separation ~0.2–1.0 logits, i.e. a real but weak signal, an order of
magnitude below the 2.25-logit danger line. This is the only mechanism in
reach that *provably* cannot enter the post-466 regime: sustained d_loss<0.45
becomes impossible by construction.

Suggested knobs: `gan_disc_freeze_dloss_lo=0.45`,
`gan_disc_thaw_dloss_hi=0.60`, `gan_gate_dloss_couple=true`.

---

# PART B — What the disc sees today, and redesign

## B.1 The current ROLLING gt_transition pipeline, precisely

Code path: `train step → _compute_r3gan_losses` (trainer 7418, dispatches on
`gan_backbone == "ladd_teacher_feat"`) `→ _compute_ladd_losses` (4569)
`→ _ladd_run_pair_mode(pair_mode="gt_transition")` (5059).

**FAKE source** — `flash_dmd_gan_x0`
(`model/dmd_action_forcing.py:_surface_flash_gan_slab`, 10843): the per-block
t=`flash_dmd_gan_t`=60 near-clean generator forward over the current rolling
train slab. Grad flows only through this t=60 refinement pass, not the
high-noise rungs. The slab is `streaming_chunk_size=18` frames = 6 chunks of
`npb=3` latent frames. Under rolling these chunks are *student rollout*
chunks whose content has drifted from GT world position (rolls up to depth 6,
`rolling_random_depth 2–6`, clean-match/drift context handling upstream).

**Pairs**: all consecutive chunk pairs `(i, i+1)` of the slab → **5 fake
transition pairs**, each `cat(chunk_i, chunk_{i+1})` along F = **6 latent
frames** (≈ 21–24 pixel frames) (trainer 5129-5137, 5426-5451). Note: with
npb=3, a "transition pair" already *is* a seam-centered 6-frame window — the
last 3 frames of chunk k and the first 3 of k+1 are the entire two chunks.

**REAL source** — matched GT retrieval, not positional
(`ladd_gt_transition_match=true`, trainer 5900+): candidate pool =
`streaming_state["gt_match_latents"]`, which under rolling
(`dmd_42f_rolling_sup_new=true`) is bounded to the active ride window
(trainer 10476-10486). Measured: **n_cand = 22** candidate GT transitions.
For each fake pair, top-`M=8` nearest GT transitions by mean-L1 on the raw
latent (equalization off in these runs: `mag_norm=""`, `mean_equalize`
unset), then **K=4** sampled fresh per D-update (step+salt-seeded), dedup'd
across the 5 fakes → **7–11 unique real windows** forwarded per update
(`ladd_match_n_real`), hard cap 12 (`ladd_gt_transition_match_max_real`).
Block-diagonal RpGAN: each fake is scored only against its own K matched
reals (`_m_rp`, trainer 6572).

**Positions/actions**: fake pair (i,i+1) carries the ride-action slices at
its *nominal* window position (`cf_state + i*npb`, trainer 5787+); matched
reals carry the actions co-located with wherever in the ride they were
retrieved from. Both are injected through the frozen WAN teacher's per-frame
action modulation (the disc conditions on actions "for free"). Because the
fake retrieval key is raw-latent L1 and the pool is the same ride, fake and
real are approximately world-position-matched but action streams can differ.

**Wavelet transform** (`model/wavelet_hf.py`, disc integration
`model/ladd_disc.py:813-824`): single-level Haar SWT per channel → LL, LH,
HL, HH same-resolution sub-bands; `drop_ll=true` discards LL (so
`ll_weight=0.15` is **inert** in these runs — it only matters when
drop_ll=false); a 1×1 conv adapter (Xavier gain 0.1) maps 48 HF channels back
to 16. **Crucially `ladd_wavelet_hf_augment=true`**: the disc input is
`raw_latent + adapter(HF_bands)` — NOT HF-only. So the intended
"WGSR HF-only disc" is actually a raw-plus-learned-HF-emphasis disc: LL/content
is fully visible through the raw branch, and the adapter is a *learned,
disc-optimized* amplifier of whatever HF channels separate student from GT.
That is a plausible shaping mechanism for the cartoon/texture artifact
(the gen-side gradient of `raw + adapter(HF)` is `1 + adapterᵀ·Haarᵀ` — an
identity plus an adversarially-tuned high-pass filter).

Also: wavelet ON forces disc inputs clean (`disc_t_int=0`, trainer 5630);
wavelet OFF runs used `disc_t_int = flash_t = 60` (noised). So the wavelet
ablation arms simultaneously changed input noise — the two knobs have never
been varied independently.

**Disc architecture** (`model/ladd_disc.py`): frozen WAN teacher
(`real_score`) forward with hooks on 5 evenly-spaced transformer blocks
(auto-selected, trainer 692-712) → per-tap Linear channel-mix (CCM, →512) →
FPN-style cross-scale fusion (CSM) → per-tap spectral-norm 2D conv heads
(kernel 3) with prompt conditioning (cmap 64) → **per-token logits**
concatenated across taps. Teacher params frozen; trainable = CCM+CSM+heads
(~10–15M). Stat head off in these runs.

**Update schedule**: GAN only on gen iters (every 5th step). Per gen iter: 5
D-updates (fresh K-match resample each, flip DiffAugment, R1 by debt-based
cadence — γ=1e3 every-5-steps + once_per_step ⇒ exactly 1 R1-carrying update
per gen iter; finite-difference R1 with σ=0.01 on ≤6 subsampled reals), then
one G-side forward in disc.eval() with `rpgan_g_loss` against the (detached)
matched reals; gen weight = `gan_loss_weight × ladd_disc_loss_weight × warmup`
= 1.0 here.

**Dead knob found**: `ladd_pair_start_seed_boundary: true`
(configs/action_forcing_phase3_dmd.yaml:99) is read nowhere in the codebase —
it does nothing. Remove or implement.

## B.2 Why this construction loses

1. **Real-set entropy ~ 22 windows/ride.** The disc's job is reduced to
   "recognize this ride's GT texture", which a 15M head stack over frozen
   5B-teacher features solves quickly. Nothing about weight/lr changes the
   endgame, only its date (Part A).
2. **The fake is always drift-contaminated.** Under rolling every fake
   transition carries accumulated rollout drift; the disc's easiest
   real/fake feature is "amount of drift texture", which is exactly the thing
   the generator cannot fully remove under teacher forcing of its own
   context — so the gradient keeps pushing global texture statistics
   (→ cartoon) instead of seam consistency.
3. **The learned HF adapter is adversarial pre-emphasis.** With augment=true
   the disc trains its own high-pass filter on top of the raw input; the
   gen-side gradient is HF-amplified by construction.
4. **Weight 1.0 with per-token logits** makes the GAN term the largest single
   gradient source the moment separation exists.

## B.3 Redesign candidates

Evaluated against the code as it exists. "Cost" = new code + validation risk;
"effect" = expected impact on the cartoon/HF-amplification failure and on the
disc-wins instability.

### (v) Frozen-teacher feature matching (non-adversarial) — RANK 1

*Mechanism*: drop the adversarial head entirely for the rolling phase; match
frozen WAN-teacher features between the student's flash slab and its matched
GT transitions. LPIPS-in-teacher-space: perceptual sharpness pressure with no
two-player dynamics, hence **no collapse mode at all** — consistent with the
empirical fact that the best-rated runs were the ones whose adversarial
gradient was zero.

*Grounding*: everything needed exists. `WanFeatureProjector`
(`model/ladd_disc.py:75`) is the frozen tap extractor;
`_compute_aux_teacher_disc_losses` (trainer 3164) *already implements exactly
this loss* (`aux_teacher_disc_feat_weight` path: projector on x0 grad-on vs GT
no-grad, per-block L2) — but only for the LoRA aux teacher's x0, not for the
student slab. The matched-retrieval machinery (`_match_select`) provides the
GT counterparts.

*New code*: one function (~80 lines): slice the 5 fake transitions from
`flash_dmd_gan_x0` (reuse `_slice_pair`), `_match_select`-style retrieval of K
matched GT windows, projector forward on both (fake grad-on, real no-grad),
per-block normalized L2 — with **min-over-K** (or softmin, τ≈0.5) over the K
matched reals per fake so the loss is mode-seeking rather than
regress-to-mean (plain mean-over-K would blur, recreating the problem DMD
already has). Weight knob `rolling_featmatch_weight ≈ 0.1–0.3`. No disc, no
D-updates, no R1, no DDP disc wrapper.

*Cost*: **low**. *Effect on cartoon failure*: high (no adversarial HF
amplifier; teacher features carry full-spectrum perceptual content).
*Risk*: feature matching is a weaker sharpness prior than a healthy GAN — but
we have zero evidence a healthy GAN is achievable in this regime, and strong
evidence the unhealthy one destroys runs.

### (A.3 gate) Adaptive-frozen weak disc — RANK 2 (do this regardless)

The Part A operating point (weight 0.2, 1 update/gen-iter, lr 5e-6, d_loss
freeze/thaw gate 0.45/0.60). Not one of the five listed candidates but it is
the cheapest thing that turns the current design from "guaranteed eventual
breakdown" into "bounded nuisance", and it composes with every candidate
below. Cost: very low (one EMA + one allreduce + one if). Effect: eliminates
the breakdown class; does not by itself fix cartoon shaping (the gradient
direction is unchanged, only its magnitude/duty-cycle).

### (iii) Relative pairing at matched world positions with balanced pools — RANK 3

*Mechanism*: keep the transition-pair formulation but fix its data problem:
(a) **cross-ride real pool** — maintain a small FIFO replay buffer of GT
transition windows (latents + actions) from the last N rides across steps
(e.g. 256 windows, ~tens of MB, they are 6×16×H×W latents), sampled alongside
the ride-local matches; (b) **balance** the buffer by direction/city the same
way the v14d LMDB was balanced; (c) optionally match on a brightness-removed
key (the `m1` mag_norm already implemented at trainer 5284-5310) so retrieval
is texture-, not brightness-driven.

*Grounding*: the matched branch is already parameterized by an arbitrary pool
(`gt_match_latents`/`gt_match_actions`, trainer 10476-10486) — widening it is
a data-plumbing change, not a loss change. The stationary configs already use
whole-ride pools; this extends to cross-ride. The prompt-embed handling needs
care (pool entries need their own prompt embeds or action-only conditioning —
`ladd_gt_transition_action_blind` exists as the escape hatch if per-ride
prompts don't transfer).

*Cost*: medium (buffer plumbing + DDP-consistent sampling + prompt handling).
*Effect*: directly attacks the memorization mechanism (22 → hundreds of
candidates), which should move the disc from "memorize this ride" to "learn
generic transition realism" and materially delay or remove the d_loss slide.
Keeps adversarial sharpness benefits if you still want them. Combine with the
gate.

### (iv) Full-spectrum disc with explicit band weights (LL reweighted, not dropped) + weight ~0.3 — RANK 4

*Mechanism*: make the frequency emphasis *fixed and explicit* instead of
learned-and-adversarial: `wavelet_hf_augment=false`, `drop_ll=false`,
`ll_weight≈0.3` — the disc sees only the wavelet view, with LL present but
downweighted by a constant, and the adapter's ability to become a runaway
high-pass is bounded because it no longer rides on top of a raw passthrough.
Plus `gan_loss_weight=0.3`.

*Grounding*: all knobs exist today (`ladd_wavelet_hf_enabled/augment/drop_ll/
ll_weight`); zero new code for the basic version. The docstring warning that
ll_weight ≥ 0.5 destabilizes spectral-norm power iteration (per-rank `_u`
drift → NCCL hang) bounds the safe range to ≤0.3. A fuller "per-band loss
weights" variant (separate head groups per band with per-band loss weights)
would need the disc forward to keep bands separate through CCM — medium new
code in `ladd_disc.py` — probably not worth it before (iii)/(v) are tried.

*Cost*: low (knob-only) to medium (per-band heads). *Effect*: moderate on the
cartoon *shape* of the failure; does nothing about the disc-wins dynamics, so
it must ship with the gate or the reduced operating point. Note the wavelet
2×2 smoke result on the *stationary* gt_transition GAN
(wavelet-on = dead disc there) — under rolling the wavelet-on arms were NOT
dead (combo2/ife2/rolllong all learned), so that earlier finding doesn't
transfer to rolling; re-verify per-regime.

### (i) Seam-centered windows — RANK 5 (mostly already the case)

With npb=3 the current transition pair already *is* the seam ± 3 latent
frames; "last-3/first-3" is a no-op. The genuinely tighter variant is 1+1
latent frames (`_slice(t,i)[:, -1:]` + `_slice(t,j)[:, :1]`, `t_frames=2`) —
trivially expressible in `_slice_pair` (trainer 5259) with `t_frames`/action
slices adjusted (2-frame action slices at the seam). Cost: low. Effect: low —
it narrows the content confound (less pure-texture surface for the disc to
memorize, more seam-discontinuity signal) but does not change the data
asymmetry or the dynamics; the disc still keys on drift texture visible in
even 2 frames. Worth doing only as a refinement of (iii).

### (ii) Trajectory-level disc on downsampled multi-chunk clips — RANK 6

*Mechanism*: discriminate 4–6-chunk clips (12–18 latent frames) as
trajectories — the honest formulation of "rolling realism" (drift is a
trajectory property, not a pair property).

*Grounding problems in the current code*: the disc's backbone is the frozen
WAN teacher, whose forward is in-distribution only at native spatial size —
spatial downsampling (H/2) is OOD for the projector (the same reason
`wavelet_hf.py` chose SWT over DWT). Temporal length is less constrained
(local_attn window 21 frames covers 18), so a native-resolution 18-frame disc
input is feasible but ~3× the tokens per sample of today's pairs, on top of
the checkpointed teacher recompute that already OOM'd FT_v3 (mitigated by
micro-batching). Downsampling would have to happen at the *feature* level
(token pooling before heads — new module) rather than the input level.
n_pairs drops 5 → 1–2 per slab, shrinking an already tiny effective batch.

*Cost*: high (new pooling module, memory re-validation, action-window
plumbing for 18-frame windows). *Effect*: conceptually the best match to the
actual rolling failure (long-horizon drift), but it inherits the same
memorization/dominance dynamics with even fewer real samples per ride, so it
needs (iii)'s pool work *first* anyway. Do later, if at all.

## B.4 Recommended program

1. **Now (knobs only)**: rolling runs go back to the rollcarn setting *or*
   `gan_enabled=false` — Part A shows these are equivalent, and honest
   configs beat accidental lobotomies. Remove the dead
   `ladd_pair_start_seed_boundary` knob.
2. **Next run**: implement (v) teacher-feature matching (min-over-K) at
   weight 0.1–0.3 as the rolling sharpness term. This is the low-cost,
   collapse-proof replacement for the GAN's intended role.
3. **If adversarial pressure is still wanted**: add the A.3 d_loss
   freeze/thaw gate (with the DDP allreduce), the reduced operating point
   (0.2 / 1 update / 5e-6), and (iii) cross-ride balanced pools — in that
   order. Only then revisit (iv) fixed band weighting; skip (i) except as a
   free refinement; defer (ii).
4. **Instrumentation for any future disc run**: alert on EMA d_loss < 0.3
   (strain) and < 0.15 (abort/freeze); both thresholds are validated by two
   independent breakdowns (rollgansure ~321, rolllong ~471).

---

## Appendix: provenance

* Parsed datastores: `wandb/wandb/run-*/run-*.wandb` for the six runs listed
  in the inventory; metrics used: `gen/train/r3gan_{d_loss,d_real,
  d_fake_detached,g_loss_raw,g_weight,r1,r1_grad_sq,r1_fired,disc_skipped}
  _gtxn`, `gen/train/ladd_match_{n_cand,n_real,pool_m,k}_gtxn`,
  `gen/{generator_dmd_loss,grad_norm,dmdtrain_gradient_norm,
  student_mae_vs_gt,student_pred_rms,streaming_window_avg_mae}`,
  `critic/grad_norm`.
* rolllong identified via `rolllong` in
  `run-20260821_162525-4z2988ef/files/wandb-metadata.json`
  (`run_name=dmd10k_rolllong_h6079764_162340`); effective config = last-wins
  over the override list (γ=1e3@5 once/step supersedes the earlier 1e6@1 in
  the same arg list).
* Key code: `trainer/causal_action_forcing_train.py` — `_compute_ladd_losses`
  4569, `_ladd_run_pair_mode` 5059 (pair construction 5129/5426, disc-t logic
  5630, match branch 5900+, matcher pool 6062, `_match_select` 6100,
  D-loop `_run_disc_updates` 6650, gen-side 6826, weight ramp 6810–6824,
  match-pool publication 10454–10486, GAN call site 9981–10005);
  `model/ladd_disc.py` (projector 75, disc 608, wavelet integration 813,
  forward/token layout 790–915); `model/wavelet_hf.py` (SWT + adapter);
  `model/r3gan.py` (losses 231–300); `model/dmd_action_forcing.py`
  (`_surface_flash_gan_slab` 10843).
