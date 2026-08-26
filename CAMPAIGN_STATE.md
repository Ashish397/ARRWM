# GAN CAMPAIGN — STATE OF PLAY, 2026-08-25 21:05

Objective, in the researcher's words: **"it works for most other things now, it just needs
the resolution and the style"** — and, sharpened later, **"our main objective has to be
style shift to the dataset style"**, with the explicit warning that **optimising sharpness
produces blocky/pixelated output**, so the target is *matching* the dataset's appearance
distribution, never maximising detail.

This file is the single synthesis. `COMMENTS_FOR_USER.md` holds live questions;
`RESOLVED_TODOs.md` holds closed items; the per-topic evidence lives under `analysis/`.

---

## 0. RESEARCHER INSIGHTS — the calls that steered this campaign

Recorded because most of them were right before the measurement existed, and several
corrected a direction I was already committed to.

| # | insight | outcome |
|---|---|---|
| I1 | **"Don't optimise for sharpness — it goes blocky and pixelated."** | **Proven on our own frames.** All seven sharpness metrics *prefer* an aliased frame to a real one — `hf_frac_40` by **151x**, lapvar by 23x. `gantune_w4`'s sky is simultaneously 21x "sharper" and posterised to 4 luma levels (real: 12). The measurement agent's own composite initially failed this test and had to be rebuilt with one-sided penalties. **This single instruction prevented us building a metric that would have rewarded the artifact we are trying to remove.** |
| I2 | **"Our main objective has to be style shift to the dataset style."** | Reframed the whole campaign. Sharpness demoted to one axis; every metric became **two-sided distance to the dataset**, so overshooting is penalised like undershooting. |
| I3 | **"Style must be measured ONLINE by the GAN — quick convergence."** | Directly produced the Gram loss. Everything else measures style *offline, hours later*; the disc measures only its own health. Now running. |
| I4 | **"Matching should not cost that much; if it does, approximate it."** | Correct. Matching costs **0.11 GB**; the 24.7 GB was an artefact of `match_k=8` colliding with a by-fake micro-batcher. Worse, the pool had collapsed so we were paying it for a **uniform random draw**. Fixed: ~90 → ~65 GB, and now a real top-3 match. |
| I5 | **"The MAE gate shouldn't matter, but there's a chance."** | Correctly skeptical. Its ratio turned out to be a **rollout-depth switch, not a mastery signal** (depth 1 median 1.373, depth 3 median 0.9935). Would have zeroed DMD on the drift chunks. Not built. |
| I6 | **"I think the CARN network has it"** (the banding) | **Correct, causally.** The noiser's residual is 27x an equal-energy white-noise control; a zero-init noiser produces zero stripes; amplitude grows 0.25 → 1.27 as it trains. |
| I7 | **"Wavelet makes everything smoother and loses resolution."** | **Correct.** Both wavelet arms drove the disc to *exactly* chance (`d_loss` = ln 2). An edge-band-only critic is cheapest satisfied by suppressing HF — the eye saw it before the metric did. |
| I8 | **"w4 doesn't work; w2 seems better than w1."** | Matched the gradient ratio exactly (w4 at 0.985 = adversarial exceeding DMD). |
| I9 | **"These are projected onto the fake score, right?"** | Yes — `backbone=fake_score`, One-Forcing style. Led directly to understanding that the disc's feature basis moves with the student. |
| I10 | **"v6rep is a waste."** | Agreed and cancelled — a seed replicate is a luxury when the roster is compute-starved. |
| I11 | **"gansig_ofclean is amazing at step 151"** / **"w2tclean is much better."** | Both drove real changes: `ofclean` + CARN + online critic built, and `force_clean` made the codebase default. |
| I12 | **"gansig_of had horizontal banding."** | **Correct and measurable** — 1.55x its own GT in tree crowns. My first check used a *sky* crop and found nothing; the artifact lives in **textured** regions. |

### I14 — "gansig_ofclean is the ONLY one that gets the textures right — I think it is the gt_vs_fake part"

**The most important observation of the campaign so far**, because it is the first time any
arm has been reported as *correct on texture* rather than merely healthy on disc metrics —
and texture/style is the whole objective.

**Why the hypothesis is mechanically credible.** `gt_vs_fake` is the pair mode that shows
the disc a **real GT window against a generated window, frame-matched**. Every other arm
runs `gt_transition` ONLY, whose "real" side is a *2-chunk transition* — a construction
about temporal continuity across a seam. So:

* under `gt_transition` alone, the disc's whole notion of "real" is *how a real clip joins
  to itself across a boundary*;
* `gt_vs_fake` is the only mode that ever asks the disc the plain question **"does this look
  like real footage?"** — which is exactly the question style/texture lives in.

That is a strong prior for the researcher being right, and it reframes today's disc-blindness
finding: the disc may be blind to texture partly because **nobody was asking it about
texture** — only about seams.

**But `ofclean` changes four things at once, so attribution is unproven.** Its candidates:
`gt_vs_fake` (this hypothesis), `force_clean`, `fake_sample_source=dmd`, and the halved
backbone kick. Note `gansig_real` = `gt_vs_fake` on the flash path **without** dmd or
force_clean — it is the arm that isolates this hypothesis, it died at step 38, and it is
requeued as **6135337**. **That is now the highest-value pending run**, because it tests the
researcher's texture hypothesis with one variable.

> **ACTION TAKEN:** priority of `gansig_real` raised to the top of the launch order; a
> `gt_vs_fake`-only arm on the *current* defaults is the cleanest possible test and should
> be built if `real`'s config has drifted from that.
> **Corroborating evidence to gather:** run the paired banding measurement (textured crops,
> against each arm's own `clean_x_real`) on `ofclean` vs `w1` vs `w2tclean`. Plain
> `gansig_of` overshot its GT by 1.55x in tree crowns; if `ofclean` sits near 1.0 and the
> single-mode arms do not, that converts the researcher's eye into a number.

---

### I13 — "In some ways ofclean is better, in some ways w1 is better" (open, and right)

Not a tie, and not yet decidable, because the arm changes **four mechanisms at once**:

| | w1 better | ofclean better |
|---|---|---|
| disc-loss floor | — | never below 0.117 (w1 reaches **0.010** = deep saturation) |
| logit-gap ceiling | — | caps at **+2.41** (w1 reaches **+5.41**) |
| GAN/DMD cosine | −0.002 (orthogonal) | **+0.166** (aligned) — scale-free, so genuinely comparable |
| zero-gradient steps | 0/15 | 0/15 (vs 3/8 in plain `gansig_of` — the `force_clean` fix confirmed) |
| chance-side excursions | max `d_loss` 0.469 | reaches **0.832**, past chance (0.693) |
| texture vs GT | — | plain `of` overshot 1.55x; `ofclean` unmeasured |
| gradient ratio | 0.156 | 5.850 — **NOT comparable**, dmd denominator artefact |

**So: `ofclean`'s disc is better-behaved and better-aligned, but it visits the
collapse-toward-chance side that w1 never does.** Which of its four changes causes which
half is unknown, and that is the experiment still missing.

**The decomposition needed** (each is one flag off `w1`, all cheap):
- `gt_vs_fake` alone → the researcher's instinct is that this is the good part.
- `fake_sample_source=dmd` alone → suspect for the chance-side excursions; it is the one
  that made the ratio incomparable and caused the overshoot in plain `of`.
- `force_clean` alone → **already running as `gantune_w2tclean`.**
- backbone `grad_scale` → **now running as `gantune_w1d1`** (divergence-1 probe).

---

## 1. WHAT WE ESTABLISHED TODAY (high confidence, evidence on disk)

### 1.1 The CARN noiser was never applied in most arms — the day's biggest find
`forward_noiser_apply_gt_former` is only read inside the matched-pool block, so with
`ladd_gt_transition_match=false` the noiser trained, logged, and checkpointed while
**never touching the data**. Log census: the application banner fired **0 times** in
v6rep/v6na/v6alt/v6b/am-on, and 2 times in v6f/6all-fr. Fixed by switching every arm to the
matched base. *(Evidence: `COMMENTS_FOR_USER.md` §C.)*

### 1.2 Matching cost 24.7 GB and bought nothing — now ~0.1 GB and actually works
The cost was an artefact of `match_k=8` colliding with a by-fake micro-batcher (9 disc rows
per group vs 2). Worse, `match_pool=100000` collapsed the pool to the 22 existing
candidates, making top-k the identity — **we were paying 24.7 GB for a uniform random
draw**. Two different algorithms were running under one flag name across arms. Now
`match_k=1, pool=3`, predicted peak ~90 → ~65 GB, with a genuine top-3 match, and the
noiser firing at full strength on 3 of 5 rows instead of ~0.
*(Evidence: `analysis/smoke_gate/matching_cost_audit.md`.)*

### 1.3 GAN weight ladder — measured, and w1 is now the codebase default
Medians past step 45 (n≈15): **w1 ratio 0.156, w2 0.283, w4 0.985**. Higher weight
*reduces* saturation (weight never enters the D update). w4 fails visually — confirmed by
eye. **All three saturate late** (gap → ~5, d_loss → 0.01 by step ~180); my earlier claim
that w1 was flat was an artefact of half-medians and is retracted. `gan_loss_weight=1.0` and
`ladd_r1_gamma=10.0` are now the defaults in `configs/action_forcing_phase3_dmd.yaml`.

### 1.4 Wavelet-HF is dead — twice, with a mechanism
Both configurations reached **exactly chance** (`d_real == d_fake`, `d_loss = ln 2`,
backbone gradient 0.1 vs plain w2's 96–376). Dropping HH delayed collapse from step 51 to
91, no more. Mechanism, from the codebase's own note: the HH diagonal band has no analogue
in smooth WAN latents, so it is common-mode across real and fake and swamps the directional
bands. This also explains the researcher's observation that wavelet output looked
*smoother*: an edge-band critic is most cheaply satisfied by suppressing high frequencies.
**Do not revisit without a different mechanism.**

### 1.5 The disc is structurally blind to texture — this is the core of the resolution problem
It scores transformer features of **latents**, at a fixed **t=60**, projected 1536 → 256,
taps [0,2,4,8,29], with `wavelet_hf=False`. It never sees pixels. A blurry decode and a
sharp decode of the same latent look nearly identical to it. That is precisely the reported
symptom: **good at geometry, blind to style**.

### 1.6 The pretrained-teacher surrogate is alive and consumed, but was amplitude-starved
Proven consumed at every generator step 31–111 (four independent markers). But its weighted
contribution was **~0.1% of the LADD term**, and its teacher got **~19 updates in 115
steps** (zero-init heads, `updates_per_step=1` on the generator cadence). So its weak
separation said nothing about the backbone. Now `updates_per_step=8`, `lr=5e-4`,
`pix_gan_weight=2.0`. Cost profile known: **+16 GB, +0.1 s/step** — memory-bound, not
compute-bound.

### 1.7 First backbone signal (confounded, but promising)
Early `t_sep`: **ConvNeXt +0.154, DINOv2 +0.179** vs SAM2's +0.000…+0.028 at matched steps.
**Confounded** — backbone *and* cadence changed together. All three now resubmitted at
identical cadence for a clean three-way test.

### 1.8 The MAE gate: NO
Median r = 1.033 with 48% of steps below 1.0 — but split by rollout depth it is a **clean
0%/100% switch** (depth 1 median 1.373; depth 3 median 0.9935). `r` measures **rollout
depth, not student mastery**; both errors rise together. Enabling it would zero DMD on
exactly the accumulated-drift chunks and drive the ratio to ~3.7. Premise falsified: the
adversarial ratio is *not* lower on the r≈1 steps.
*(Evidence: `analysis/gan_tuning/MAE_GATE.md`.)*

### 1.9 The disc is conditioned on the wrong actions 3 of 4 rolls
Latents sliced at a rolling offset, action origin pinned at ride start → 3/6/9 chunks of
lag on rolls 2/3/4. Symmetric (no bias) but the "action-conditioned" critic is conditioned
on garbage most of the time. Real fix implemented, flag-gated
(`ladd_action_origin_track_slice`), with alignment telemetry.

### 1.10 Divergence 3 is live — and destabilises the GAN/DMD balance
Confirmed by four pre-registered predictions (both modes fire, pairs 5 → 2+3, band keys
present, `graph_on=1`). **But**: it moves the disc from a fixed t=60 to the rung ladder
(t ∈ {208, 357, 625, 1000}, mean **566** — a ~9× higher corruption level), and its measured
gradient ratio is wildly unstable: **0 at step 31, then 17.1 at step 41** against a healthy
band of 0.05–0.30. That instability is now a first-order concern about the mechanism, not a
telemetry artefact.
**Good news:** `ladd_disc_force_clean` is evaluated *before* the band and overrides it, so
divergence 3 and a clean-input disc **combine** (`gansig_ofclean` built).

### 1.11 Silent-failure count: 13 and counting
Today added: the inert CARN; dead config keys (`ladd_*_n_real`, read then used only by an
error check); `carn_seam_affine_match_mean` never plumbed; `ladd_r1_every_n_steps=1` being a
no-op on the matched path; three arms shipping under one job name. **Every one passed our
existing guards and printed happily in a config echo.**

---

## 2. WHERE I WAS WRONG TODAY (so the record is usable)

1. **"The GAN is 10× too weak"** — measured during warmup ramp. Past the ramp it was in band.
2. **"The ratio isn't scaling with weight"** — small-n artefact; it does scale.
3. **"cos ~0 is alarming, v6f shows +0.69"** — +0.69 was a single sample; the real median is
   +0.167 and near-zero is normal *and arguably desirable* (orthogonal ⇒ complementary).
4. **"w1's gap is flat"** — half-median artefact; all three rungs saturate late.
5. **"SAM2 is the wrong teacher because segmentation is texture-invariant"** — premature; the
   teacher had barely trained.
6. **"HF at t=60 is noise, hence wavelet died"** — wrong twice: wavelet forces t=0 anyway,
   and the real cause was the HH common-mode band.
7. **"49.2 GB baseline"** — a step-5 reading; the true settled peak is 64.89 GB. Every
   headroom argument built on it was ~16 GB optimistic.
8. **Fixed `gansig_of`'s backbone scale that wasn't broken** — 0.25 was a documented
   deliberate treatment; my "fix" removed the only divergence-1 probe from the sweep.

**Pattern:** every error was quoting a single sample, or a statistic computed over the wrong
window, on metrics whose step-to-step scatter spans 20–190×. **Standing rule: medians past
step 45, trajectories not endpoints, and never a lone reading.**

---

## 3. THE BACKLOG — QUALIFIED AND PRIORITISED

Scoring: **Gain** = expected movement on *dataset-style distance*, the actual objective.
**Cost** = holder/queue time. **Confidence** = how sure I am it will tell us something.

### P0 — do these first; everything else depends on them

| # | Item | Gain | Cost | Why it's first |
|---|---|---|---|---|
| P0.1 | **Style-distance measurement harness** (in progress) | **Decisive** | CPU only | We cannot判 judge a single style fix without it. All day we measured disc health, never appearance. Must be two-sided (distance to dataset, not "more sharpness") and must include **blockiness detectors** validated adversarially against a deliberately pixelated frame. |
| P0.2 | **`smk_v6fmit` gate smoke** (queued) | High | 2 nodes, ~1 h | Validates the matched+memory config AND the first GPU execution of three code fixes the 8 production arms enable. Currently queued *alongside* those arms, not ahead. |

### P1 — highest expected gain on style

| # | Item | Gain | Cost | Confidence |
|---|---|---|---|---|
| P1.1 | **Three-way pretrained-teacher comparison** (ConvNeXt / DINOv2 / SAM2, matched cadence, weight 2.0) — queued 6133669/90/91 | **High** | 3 × 2 nodes × 2 h | High. A frozen encoder trained on real photographs is a *learned style prior* — the one mechanism here that cannot be satisfied by blockiness. Early signal already favours it. |
| P1.2 | **Disc-timestep arms** `w2tclean` (t=0) and `w2tsamp` (t∈[20,400]) — queued 6133544/45 | **High** | 2 × 2 nodes × 2 h | High. §1.5 says the disc cannot see texture; these are the cheapest direct test of whether *noise level* is why. Config-only, flags verified consumed. |
| P1.3 | **`gansig_ofclean`** — divergence 3 + clean disc (built, unqueued) | Medium-High | 2 nodes × 2 h | Medium. Newly proven legal. But §1.10's ratio instability (0 → 17.1) must be understood first, or this inherits it. |

### P2 — worth doing, but after P1 reports

| # | Item | Gain | Cost | Verdict |
|---|---|---|---|---|
| P2.1 | **Pixel PatchGAN** `gantune_w2pix` (built) | Potentially high | 2 nodes × 2 h, highest memory | **Hold.** Operates where style lives, but is a from-scratch local critic with no pretrained prior — the mechanism *most* satisfiable by blockiness, and this codebase already produced an 8px stippled artefact once. Run only if P1.1 fails, with the phase-fold tripwire armed. |
| P2.2 | **`gantune_w2style`** (built) | Unknown | 2 nodes × 2 h | Assess when its agent reports whether the mechanism is genuine or synthesised. |
| P2.3 | **R1 parity on the 8 arms** (`ladd_r1_unified_cadence=true`) | Medium | 1 resubmit | Evidence *strengthened* since I advised against it: all rungs saturate by step 180 at 200 steps, and the arms run 500. Still carries an old double-fire bug history. **Researcher's call — Q1.** |
| P2.4 | **`gansig_wide` / `gansig_huge`** (real-set widening) | Medium | 2 × 2 nodes × 2 h | Sound, but addresses disc *memorisation*, not style blindness. Lower priority than P1 for this objective. |

### P3 — deprioritised, with reasons

| Item | Verdict |
|---|---|
| Wavelet-HF, third attempt | **Dead.** §1.4. Two configurations, both at chance, mechanism understood. |
| MAE gate | **Dead.** §1.8. Premise falsified by our own data. |
| Ladder's R1-gamma sweep | **Obsolete** — gamma 10 already chosen and made default. |
| Ladder's update-balance / disc-speed | **Low value** — neither touches style. |
| `w2carn` (w2 + CARN) | **Superseded** by the production arms, which are all CARN-matched now. |
| Divergence-1 probe | **Genuinely untested** (I removed it by accident). Cheap to restore; low priority against P1. |

---

## 3b. LATE ADDITIONS (21:20) — three faults fixed, one ranking dispute recorded

**Fixed before they cost us runs:**
1. `w2tclean` and `w2tsamp` were both queued as **`gt-w1`** (inherited `--job-name`),
   indistinguishable in `squeue`. Third instance of this bug today.
2. All three surrogate arms had `MAXSTEPS=200` against `--time=02:10:00`; at ~40 s/step
   they would have been **cut off around step 180**. Wall raised to 03:30:00.
3. **`ladd_disc_timestep_shift=5.0`** in `w2tsamp` — I set it because it is the code
   default, but `docs/GAN_REDESIGN.md` R7 records shift 5.0 as **"measured-bad → GONE",
   eliminated from every active recipe**, with 0.35 as the replacement. Now 0.35.
All five resubmitted: **6134655** (w2tclean) **6134656** (w2tsamp) **6134657** (w2cnx)
**6134658** (w2dino) **6134659** (w2sam).

**A ranking dispute I am recording rather than resolving.** The arms agent ranks the
disc-timestep pair **8th and 9th** ("the LADD disc stays a latent-feature projection
regardless of timestep"); I rank them **P1.2**. Both arguments are honest: theirs is that
changing *when* the disc looks does not change *that* it looks at projected latent features;
mine is that at t=60 the texture has already been destroyed before projection, so the
projection is not the only bottleneck. **The arms are cheap and the disagreement is
resolvable by running them** — which is the argument for keeping them high. If `w2tclean`
shows no change in style-distance, the agent is right and the whole timestep axis dies.

**Style mechanisms — the honest inventory** (from the same report, verified by grep):
* **No Gram/AdaIN style loss exists in the training path.** Every Gram symbol lives in
  offline eval (`grids/eval/style_shift.py`, `analysis/style_shift/axis2_local_style.py`,
  `utils/style_shift_detect.py`) with zero imports from `trainer/`, `model/`, `pipeline/`.
* **No** HF/anisotropy matching loss; `analysis/texture_stats.py` is only ever consumed under
  `no_grad`. No colour/histogram matching. `MomentDiscriminator` is dead code.
* **LPIPS-VGG does exist** (`lpips_loss_weight`, wired at `trainer:18027`) and has **never
  been used** — zero hits across `sbatch/`, `configs/`, `docs/`. Built as
  `gantune_w2style`. Honest caveat: it is a *paired, spatially-aligned* perceptual distance,
  not an unpaired style match, so it does not ask the dataset-style question directly. Needs
  sign-off against the standing "texture supervision is adversarial only" rule.
* **The only true Axis-2 candidate is code that does not exist**: an unpaired region-level
  Gram/AdaIN loss, ~60 lines, for which the encoder, the unpaired real supply and the
  graph-on crop decode are all already present. **This is the highest-ceiling item on the
  board and the only one that targets style directly rather than by proxy.**

---

## 3h. THE PATCHGAN PROBE MEASURED NOTHING (22:35) — my design error, not a result

`gantune_w2pix` completed all 120 steps cleanly. It produced **no usable measurement**, for
two independent reasons, and I designed the first one in.

**(a) The probe defeated itself.** I ran it at `PIXW=0.0` believing the "unweighted"
telemetry would still publish the pixel critic's amplitude. It does not: the G-term is
folded into `gen_gan_loss` *after* multiplication by the weight, so at weight 0 the folded
term is identically zero and differentiating it yields zero. All nine samples read
`pix_gan_grad_ratio = 0` with the cosine flagged undefined.

**This is not "the pixel critic is negligible like SAM2's 0.1%" — it is no measurement at
all**, and the two look identical in the output. Same failure class as everything else
today: a zero that means "never ran" is indistinguishable from a zero that means "ran and
was tiny". **Fix is config-only**: use a small non-zero weight (e.g. `PIXW=0.01`) and divide
the weighted ratio by it.

**(b) The critic never trained anyway.** Pinned at chance to three decimals after 95
D-updates: `pix_d_loss` median **1.386** (chance = 2·ln2 = 1.3863), real/fake gap **0.004**,
`pix_g_loss` **0.6918** (ln2). Root cause is a schedule mismatch I also missed:
`pix_real_pool_warm_updates=128` is required but only **95** D-updates occurred, so
`pix_a21_warming=1` on *every* sample and the real-support floor was unmet for most of the
run. **The probe was ~30% too short for its own warm-up.**

**Verdict: the PatchGAN is UNTESTED, not retired.** Retiring it on this would be discarding
a candidate on a broken measurement. We also have **no evidence either way on its blockiness
risk**, since the critic never learned anything.

**What it did establish — and this is genuinely useful:** the pixel path is *cheap*.
Peak **63.78 GB at step 105** (vs my ~71 GB prediction, and essentially equal to LADD-only
`gantune_w1`'s 64.89), and **35.6 s/step vs 33.1** — **+7.6%**. The graph-on VAE decode added
no measurable memory. The crop/decode plumbing is healthy (no holdout leak, no repeats, no
band mismatch, 24 unique rides).

**Decision: go to the Gram/AdaIN loss first.** The PatchGAN needs two more runs before it is
even judgeable (a longer probe, then a weight decision), whereas the direct loss uses a
frozen encoder with no warm-up and no adversarial instability. Revisit the PatchGAN later on
the strength of its cheapness.

---

## 3g. THE DISC OSCILLATES BETWEEN BOTH FAILURE MODES (22:35, revised)

**First version of this section said "the disc is dying toward chance". Step 101 falsified
that** — it came in at gap **+2.416**, `d_loss` **0.1120**, the *strongest* separation of the
run, in the saturation direction. Full series:

| step | 31 | 41 | 51 | 61 | 71 | 81 | 91 | 101 |
|---|---|---|---|---|---|---|---|---|
| gap | +0.604 | +1.602 | +0.981 | +1.103 | **+0.012** | +0.600 | +0.109 | **+2.416** |
| `d_loss` | 0.454 | 0.207 | 0.358 | 0.333 | **0.691** | 0.473 | 0.653 | **0.112** |

The half-means do drift as I said (`d_loss` 0.338 → 0.482), but **the dispersion swamps the
drift** — `d_loss` spans 0.112–0.691 (stdev 0.201), gap spans +0.012 to +2.416 (stdev 0.795).
At n=8 that supports no trend claim.

**Correct characterisation: violent oscillation between chance (0.691 = ln 2) and saturation
(0.112), inside a 30-step window.** That is arguably a worse pathology than either endpoint,
and the important part is this: **neither a saturation detector nor a chance detector would
reliably catch it** — sampled every 10 steps, this run looks healthy about half the time.

Standing rule, revised accordingly: judge disc health on the **dispersion of `d_loss` across
a window**, not only its level or its trend. A disc visiting both failure modes is not a disc
that is "fine on average". Instrument variance, not just position.

*This is the second time in one section I stated a trend that the next data point overturned.
n=8 on a metric with this variance is not enough for directional claims, and I should have
said "oscillating, insufficient data" the first time.*

**Independent of all that, the proven mechanism gained a third confirmation:** step 101 is
another `ratio=0, cos=+0.000` — the exact `alpha_t=0` signature. Zeros now at steps 31, 61,
101 = **3 of 8 (38%)**, against ~33% predicted from the measured rung distribution. That
finding is holding up and does not depend on the oscillation question at all.

**Also: do not quote `gansig_of`'s ratio median of 2.82.** It pools four `disc_t` regimes,
two of them structurally zero, and its inflation comes from the DMD denominator collapsing
at low rungs, not from a large GAN gradient. Against w1's 0.156 it is an apples-to-oranges
comparison.

**Verdict on divergence 3, consolidated:** the alignment win is real and confirmed
(`graph_on=1`, action-lag removed). It ships two problems — a third of steps deliver exactly
zero adversarial gradient (proven, `alpha_t = 0`), and the disc degrades toward chance
(unexplained). `force_clean` fixes the first and serves the texture objective; there is **no
evidence it touches the second**. Not promotable on this evidence.

---

## 3f. ROOT CAUSE FOUND (22:05) — at `disc_t=1000` the generator gradient is multiplied by ZERO

Not a statistical claim — it falls out of the scheduler arithmetic, which I verified in
source (`wan/utils/fm_solvers_unipc.py`):

```python
def _sigma_to_alpha_sigma_t(self, sigma):
    return 1 - sigma, sigma                                   # line 272-273
...
noisy_samples = alpha_t * original_samples + sigma_t * noise  # line ~796
```

So `∂(disc input)/∂x0 = alpha_t = 1 − sigma`, and `sigma = t/1000`:

| rung | `alpha_t` | generator-side gradient |
|---|---|---|
| 208 | 0.79 | intact |
| 357 | 0.64 | intact |
| 625 | 0.38 | attenuated |
| **1000** | **0.00** | **annihilated — the student's sample is multiplied by zero** |

At t=1000 the disc input is *exactly* `1.0 × noise`. The student's sample does not enter the
discriminator at all, so the gradient back to the generator is exactly zero. The data matches
with no exceptions:

| step | `disc_t` | `gan_grad_norm` |
|---|---|---|
| 31 | **1000** | **0.000000** |
| 41 | 208 | 0.342 |
| 51 | 357 | 0.045 |
| 61 | **1000** | **0.000000** |

**RETRACTED (22:15): the step-71 half of this was wrong.** I reported that the mechanism
predicted step 71's chance-level collapse (`d_loss=0.6911 ≈ ln 2`, gap +0.012) as another
t=1000 noise death. It is not — step 71 ran at **`disc_t=208`** with a perfectly healthy
`gan_grad_norm=0.159`. Two symptoms in one run were connected that share no cause. The
full table, with the real separation:

| step | `disc_t` | `alpha_t` | `gan_grad_norm` | gap |
|---|---|---|---|---|
| 31 | **1000** | **0.000** | **0.000000** | +0.604 |
| 41 | 208 | 0.792 | 0.342 | +1.602 |
| 51 | 357 | 0.643 | 0.045 | +0.981 |
| 61 | **1000** | **0.000** | **0.000000** | +1.103 |
| 71 | 208 | 0.792 | 0.159 | **+0.012** ← collapse, but gradient fine |

**What survives, and it is the important half:** `gan_grad_norm == 0` ⟺ `disc_t == 1000`,
with perfect separation — zero at both t=1000 steps, non-zero at all three lower rungs. That
is scheduler arithmetic, not correlation.

**What is now a SECOND, unexplained problem:** at step 71 both logits sat at ~+2.03 — the
disc collapsed onto a near-constant output rather than being blinded by noise. It recovered
by step 81 (gap +0.60, `d_loss` 0.4733). Logit levels are swinging +2.03 → −1.49 within ten
steps. Plausibly the generator overpowering the disc; **no evidence, so no mechanism
claimed.**

**Consequence:** `force_clean` fixes problem 1. There is *no reason yet* to think it fixes
problem 2. My earlier framing — one flag solving the arm's troubles — was too tidy.

**Impact.** With the exit rung sampled roughly uniformly over {208, 357, 625, 1000},
divergence 3 **discards ~33% of its adversarial signal outright** and attenuates a further
~17% to 0.38×. The alignment win is real; it is being bought at the cost of a third of the
training signal, silently.

**Fix is one flag, and it solves the texture objective at the same time.**
`ladd_disc_force_clean=true` pins `disc_t=0` ⇒ `alpha_t=1.0` — maximum gradient *and*
maximum texture visibility — while keeping the graph-on band as the fake, since force_clean
is evaluated before the band in the precedence chain. **`gansig_ofclean` submitted as job
6134897.** This also raises `w2tclean` (6134655) from "worth testing" to "addresses two
confirmed defects at once".

*Caveat kept: the mechanism is certain, the ~33% frequency rests on the rung distribution
measured on w1/w2 (n=18).*

---

## 3e. DIVERGENCE 3 HAS A REAL DEFECT (21:55) — intermittent zero adversarial gradient

Established by controlled comparison, not inference:

| arm | fake source | GAN-active samples | `gan_grad_norm == 0` |
|---|---|---|---|
| `gantune_w1` | flash | **17** | **0 (0%)** — min 0.0025 |
| `gansig_real` | flash | 1 | 0 |
| **`gansig_of`** | **dmd** | 4 | **2 — steps 31 and 61** |

`ladd_fake_sample_source` is the only difference. So on roughly **half the sampled steps the
generator receives no adversarial gradient at all**, while the discriminator trains normally
(d_loss 0.33, gap +1.10, R1 live). Mechanism still unexplained — critic warmup ruled out
(ramp is 0.44, non-zero), detached band ruled out (telemetry error key absent). **Intermittent,
dmd-only, never on flash** is now established rather than suspected.

The ratio caveat from §3c is also confirmed quantitatively: `gansig_of`'s `dmd_grad_norm`
tracks the rung (t=1000 → 0.211, t=357 → 0.023, t=208 → 0.020) while its `gan_grad_norm`
(0.342, 0.045) sits inside w1's own 0.0025–0.039 range. **"The GAN is 100× too strong here"
would be a measurement error** — the GAN term is normal; the DMD term collapses.

**Verdict on divergence 3:** the alignment win (`graph_on=1`, action-lag removed) is real and
confirmed, but the *delivered* adversarial signal is intermittent, and its natural metric is
not comparable to the rest of the roster. **I would not promote it on this evidence.**

**Consequence for the queued `v6alt`** (the only production arm using dmd): it remains worth
running — it *is* the divergence-3 test, and 500 steps gives far better statistics than four
samples — but we should now expect it to underperform, and we know to read it on
`gan_grad_norm` stratified by rung rather than on pooled ratio. **No change needed unless you
want it pulled.**

This also raises `w2tclean`'s value: a clean-input disc addresses the texture objective
directly, and per the precedence chain it *combines* with dmd if we later want both.

---

## 3d. GATE SMOKE PASSED (21:45) — the 8 production arms are de-risked

`smk_v6fmit` on holder 6130135, step 61+. **Every prediction from today's fixes confirmed:**

| criterion | predicted | measured | verdict |
|---|---|---|---|
| peak memory (matching fix) | ~65 GB, down from v6f's 89.8 | **65.00 GB** | ✅ exact |
| CARN actually applies | fires, levels NOT all zero | **`ACTIVE: FN applied to the FORMER GT latent of 5 matched real pairs (levels=[0,0,1,2,1])`** | ✅ 3 of 5 rows at full strength — precisely what the matching audit predicted |
| R1 rate on matched path | 0.20 (latch-capped) | **`r1_rate=0.200`**, `r1_fires=8` | ✅ exact — and the new telemetry works |
| deferred-lag visibility | new `hold_step` field | **`hold_step=45`** on the step-51 line | ✅ the 5-step lag is now self-describing |
| GAN health | in band | step 51 `ratio=0.124`, `cos=+0.110`, `d_loss=0.508` | ✅ |
| errors / OOM | none | 0 | ✅ |

**This closes the biggest open risk of the day.** The matched configuration the eight arms
depend on is memory-safe at 65 GB (they previously OOM'd at 90+), the CARN genuinely applies
(this morning it never fired at all), and three code fixes executed on GPU for the first
time without incident.

**One watch item, not a blocker:** at step 61 `r1` jumped 0.0115 → 0.9105 with `cos=−0.555`
and `ratio=0.863`. A strongly negative cosine means the adversarial gradient is *opposing*
DMD on that step. One sample, and the ratio caveat in §3c does not apply here (this is a
flash-path arm), but if it persists across the arms' first 100 steps it is the abort
signature to watch.

**Meanwhile in `gansig_of`: `gan_grad_norm = 0` has now RECURRED** — steps 31 and 61 both
show `ratio=0, cos=+0.000`, against non-zero at 41 and 51. That is **2 of 4 GAN-active
samples with zero adversarial gradient**, i.e. the generator learns nothing from the
discriminator on roughly half its steps under divergence 3. Still unexplained (critic warmup
and a detached band both ruled out). This is now a substantive concern about divergence 3
itself, and it bears on the queued **v6alt**.

---

## 3c. MEASUREMENT VALIDITY (21:35) — `gan_dmd_grad_ratio` is NOT comparable across paths

This would have produced a wrong verdict, so it belongs with the standing rules.

Under divergence 3 (`ladd_fake_sample_source=dmd`) the ratio's **denominator co-varies with
the disc timestep**, which is itself random per step:

| step | `disc_t` | `dmd_grad_norm` | `gan_grad_norm` | ratio |
|---|---|---|---|---|
| 31 | 1000 | 0.2114 | 0.0000 | 0 |
| 41 | 208 | **0.0200** | 0.3423 | **17.1** |

The 17.1 spike is **not the GAN exploding** — `gan_grad_norm=0.342` is unremarkable. It is
`dmd_grad_norm` collapsing 10×, and it tracks the rung: a low exit rung means the student's
x0 is already near target, so DMD has little to correct. Mechanically sensible, and fatal to
naive comparison.

**Consequences:**
1. **Never compare a dmd-path arm's ratio against a flash-path baseline.** w1's denominator
   sits at a fixed `disc_t=60`; `gansig_of`'s swings 10× step to step over four rungs. A
   pooled median is an average over four different regimes.
2. For dmd-path arms use **`gan_grad_norm` alone**, or the ratio **stratified by rung**.
3. This applies to the queued **`v6alt`**, and to `gansig_ofclean` — their GAN strength
   cannot be read off the same scale as the rest of the roster.
4. Reassuring on the saturation front: the step-41 trend did **not** continue —
   `d_loss` recovered 0.2068 → 0.3582 and the gap fell +1.602 → +0.981. The disc is
   oscillating, not winning.

Also closed: `ladd_disc_deferred_updates=2` at steps 21/31/41 (w1 reads 1), and both pair
modes carry real separating values (`d_real_gt=+0.626 / d_fake_gt=−0.605` beside the gtxn
pair). `gt_vs_fake` is genuinely firing, not merely publishing keys. `ladd_disc_t` spans
`1000 → 625 → 1000 → 208`. One open thread: `gan_grad_norm=0` at step 31 is still
unexplained — critic warmup was ruled out (the ramp is 0.44 at that step, non-zero), and a
detached band was ruled out (the telemetry error key is absent).

---

## 4. WHAT I AM DOING NOW (ready for the holders)

1. **A single ranked launch list exists** — `analysis/gan_tuning/RESOLUTION_PLAN.md` plus this
   file's §3. When a holder frees: P0.2 if unrun, then P1.1, P1.2, P1.3, in that order.
2. **Everything in P0–P1 is already built and validated** (`bash -n`, no duplicate last-wins
   keys, job names correct, flags traced to their consumption sites — not merely present).
3. **Understand the divergence-3 ratio instability** before `gansig_ofclean` or `v6alt` are
   trusted; it also determines whether v6alt should be pulled from the production queue.
4. **Standing measurement rule** applied to every future verdict: medians past step 45,
   trajectories not endpoints, two-sided distance-to-dataset, blockiness reported beside any
   sharpness number.
