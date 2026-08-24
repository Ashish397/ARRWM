> **RECONCILED 2026-08-23 with `docs/GAN_REDESIGN.md` — EXECUTION PLAN items
> A3, A12, A13, A14, and the researcher-advisor directives A19, A20, A21,
> A24.** This file is the **build spec for B2** (pixel texture PatchGAN, arm
> G). It previously carried a blanket SUPERSEDED banner; the resolutions that
> superseded it have now been folded in below and are marked
> `[A3]`/`[A14]`/`[A19]`/`[A20]`/`[A21]`/`[A24]` at the point of use.
> `GAN_REDESIGN.md` remains the governing document — on any conflict, its
> EXECUTION PLAN wins, and its section C ("Contradictions a sub-agent could
> trip over") still applies.
>
> **A19–A24 are researcher directives and supersede every earlier resolution in
> this file that they touch** — in particular **A12 is superseded by A19** (the
> mean-vs-sum question is answered: MEAN) and **the B2 half of A13 is
> superseded by A19** (there is no R1/R2 balance question because there is no
> R2). Markers formerly reading `[A12]`/`[A13]` now read `[A19]`.
>
> **LAUNCH GATES — hard prerequisites, not follow-ups:**
> - **A23 (was A2, now hardened).** A2 is *resolved*, and the answer is that
>   training and inference commit **different tensors**: training commits the
>   t=60 flash prediction, `utils/eval_causal_AR.py` commits the t≈208 ladder
>   endpoint. That is not sufficient to launch. B2 may not run until **either**
>   the fake source is inference-parity — built from `finish_denoised_chunk`
>   with flash disabled, gradient path through the generation op — **or** a
>   two-tensor texture-battery comparison (decode both from one checkpoint)
>   shows the two are equivalent. Otherwise 400–600 steps of pixel-adversarial
>   training may perfect a hidden t=60 auxiliary render the user never sees.
>   See §3.1.
> - **A22 — held-out discriminator generalisation test.** A separate mandatory
>   build (specced in `GAN_REDESIGN.md` A22, not here): reserved rides the
>   pixel D never sees as real training examples, evaluated periodically for
>   train-vs-held-out margin and accuracy. It answers *"is D memorising its
>   real crops?"* and is **distinct** from the §8.1 positive control (*"can D
>   recognise obvious corruption?"*) and from the fixed-route eval. A prior
>   holdout list leaked into training once before, so the exclusion must be
>   verified against the real-supply path of §3.4, with telemetry.
>
> *Citations in this file prefer the **SYMBOL**, not the line.*
> `trainer/causal_action_forcing_train.py` *is under active edit and line
> numbers drift by hundreds of lines within a day; every line number that
> could not be re-verified on 2026-08-23 has been replaced by a greppable
> symbol or log key. Grep the symbol.*

# TEXTURE GAN v1 — pixel-space patch critic, separated from the transition critic

> **MEASURED-FALSE CLAIMS FLAGGED 2026-08-23 night (adversarial review; details
> in the WP-PIXGAN report + GAN_REDESIGN_TWO Option C):** §5.3's "scale-free in
> P by construction" is FALSE (measured α=−0.915 vs claimed 0; γ must be
> CALIBRATED on the shipped architecture, never assumed 1.0). §4's "receptive
> field ≈70 px / local question" is FALSE with GroupNorm on (gradient support
> spans the full image; norm being removed by researcher decision). §3.7's
> effective-sample arithmetic depends on §4's locality claim and is unfounded
> until re-derived on the true receptive field.


**Date**: 2026-08-23. **Status**: design, pre-implementation.
**Gate**: `gan_pixel_texture_enabled` (default **false**, byte-identical off).

---

## B2 SCOPE FREEZE — what this arm is NOT `[A19]`

**The B2 hypothesis is deliberately narrow, and it stays that way:**

> ***"Can local decoded-pixel adversarial feedback suppress fabricated
> directional texture?"***

Nothing may be added to this arm that widens that question. The items below are
**escalation paths only** — each is reachable *after* B2 returns a reading, and
none may be folded into the B2 build to give it a better chance:

| FROZEN OUT of B2 | where it lives instead |
|---|---|
| ADA / adaptive augmentation from step 0 | escalation, and only on *measured* D-overfit (the A22 held-out test) |
| timestep / provenance-rung conditioning | `GAN_REDESIGN.md` B4, arms K/L (Point 5); §3.5 already says no noising and no timestep |
| pretrained DINOv2 or SD/VQGAN disc weights | `GAN_REDESIGN.md` B5 — triggered *only* by an §8.1 UNDERTRAINED verdict (§4.1) |
| multi-scale or temporal GAN | escalation on the §8 falsification ("texture problem is not marginal-per-frame") |
| wavelet discriminator | `GAN_REDESIGN.md` B5, gated on the B1 anchor proving too rigid |
| nearest / retrieval-matched reals | **never** — Point 7, and A21 forbids similarity selection outright (§3.4) |
| strict action conditioning | parked with the transition stack (B6); §3.5 says no action/prompt conditioning |
| transition pairing (gt_transition, adjacency) | parked (B6) — off in every arm of this ladder (§1) |

Adding any of these before B2 has been read makes the outcome uninterpretable:
a positive result would not attribute to the pixel-adversarial term, and a
negative one would not falsify it. The B2 build has exactly **two** mandatory
additions and both are *instruments*, not changes to the hypothesis: the §8.1
positive control and the A22 held-out generalisation test.

---

## 0. Why this exists (the diagnostic that authorised it)

`eval/texture_abc_strict03/REPORT.md` (ride 20240115085313, strict03 ckpt,
40-chunk rollout, no mp4 round-trip anywhere):

| finding | number | meaning |
|---|---|---|
| A ≈ B | B/A HF power 0.86–0.89, Laplacian kurtosis 13.4→13.0/14.0, anisotropy ~1.0 | the VAE round-trip only *softens* (~15% HF); texture structure, tail statistics and isotropy survive → **NOT a VAE bottleneck (Case 2 excluded)** |
| C ≫ A in HF | C_early 1.49×, C_late 2.21× HF power; Laplacian var 1.76×→2.05× | the student **fabricates** high-frequency energy in its own latents (decoder is common to B and C, so the artefact enters in the latent) |
| directional collapse | fft fy/fx anisotropy: A 1.10 → C_early 0.93 → C_late **0.36**; angular entropy 0.98 → 0.84 | a strong vertical-stripe artefact appears and **amplifies with rollout depth** (visible as total texture mangling in the C_late strip) |
| contrast wash | luma kurtosis −0.19 (A) → −1.2 (C) | global contrast distribution flattens |

**Verdict: Case 1.** A decoded-RGB discriminator is well-founded. The one
Case-3 element (depth amplification) lives in the *student latent*, not the
decoder — which pixel-space supervision sees directly anyway.

The statistics above are implemented as a reusable module in
`analysis/texture_stats.py` (`texture_battery`, `directional_spectrum`,
`haar_band_power`, `hf_kurtosis`, `battery_delta`). Every readout in §8 and §9
uses that module — no second implementation.

## 1. Separation of concerns (the user's directive)

- **Texture critic (NEW, this doc)** — marginal, per-frame, patch-level,
  pixel-space. Answers only: *"does this image patch look like real footage?"*
  No transitions, no actions, no temporal question.
- **Transition critic (EXISTING gt_transition stack)** — parked. Drift/seam
  correction is CARN's job; the transition GAN is not developed further until
  texture works. **Off in every arm of this ladder** (GAN_REDESIGN standing
  decision 2 / B2).
- Both remain independently flag-gated; the new arm runs texture-only.

## 2. The critical design rule: same decoder on both sides

Real and fake are **both decoded through the frozen Wan VAE decoder**:

- fake = `decode(student chunk latent)` — grad flows through the frozen
  decoder into the latent → generator (decoder params frozen, graph to input).
- real = `decode(GT latent)` — **never raw RGB frames**. The diagnostic
  measured decode softening (~15% HF loss). If reals were raw RGB, the critic
  would reward the generator for out-sharpening the decoder — *exactly the
  fabricated-HF failure mode we are treating*. Same-decoder comparison
  cancels the decoder's transfer function and isolates the student's texture.

## 3. Data path

1. **Fake source**: `flash_dmd_gan_x0` (existing flash-DMD pass, t=60), one
   chunk = 3 latent frames. Later option: committed rollout chunks.
   **GATED on A23** — see the banner. A2 is resolved and the answer is
   adverse: `flash_dmd_gan_x0` is training's committed tensor but **not** the
   tensor inference renders, so this source is only admissible once the
   inference-parity fake (`finish_denoised_chunk`, flash off, gradient path
   through the generation op) is built, or the two-tensor battery comparison
   shows equivalence. Note also that `rollout_viz_source=finish` means the
   training-time sample videos render the t=60 tensor too, so they are not an
   independent parity check.
2. **Latent-space crop, then decode**: sample `pix_crop_lat=(24, 32)` random
   latent crops (→ 192×256 px after 8× decode) from the chunk. Decoding a
   crop instead of the full 60×104 latent cuts decoder activation memory ~8×.
   Trim an 8-px border after decode (decoder edge effects).
3. **Temporal**: decode the full 3-latent chunk (→ 12 RGB frames via
   `seed_first`), then keep **`pix_frames_per_crop=3`** random frames `[A3]`
   (was 2). The critic is 2-D per-frame; frames are extra batch rows.
4. **Reals — the two real-supply invariants** `[A20]` `[A21]`. Reals are GT
   latents decoded through the same frozen decoder (§2), drawn
   **uniform-random**, **never nearest-matched** (Point 7; the scope freeze
   forbids retrieval outright), under the coarse band constraint of §3.6.
   Two requirements govern the supply. They are **separate, and both
   mandatory** — satisfying one does not satisfy the other:

   **(a) Per-update batch diversity — the real-batch invariant** `[A20]`.
   *Every discriminator optimizer update gets one independently sampled
   reconstructed-real image/crop per fake image/crop.* At the §3.7 spec
   (`pix_crops_per_step=4`, `pix_frames_per_crop=3`) that is **12 fakes AND 12
   independently drawn reals** — never 12 fakes scored against 2–4 reused
   reals.
   - `pix_reals_per_fake=1` states the invariant; it is not a budget knob.
     Raising it above 1 is permitted only if every additional real is *also* an
     independent draw. Reusing a real across fakes, or dropping below 1:1, is a
     build bug.
   - **Independent means independent at the source.** Three frames taken from
     one decoded crop, or from one ride window, are **not** three independent
     samples of the real distribution: they share exposure, weather, time of
     day, road surface and the same VAE reconstruction. A real draw is
     independent when its *source frame* is drawn independently.
     Frame-within-crop expansion is a **fake-side** sample-count device
     (§3.7) and it does **not** transfer to the real side.
   - The A6 diversity telemetry is extended to prove this **per D update**
     (§7): real images, unique source frames, unique source windows, unique
     rides, repeat fraction. A non-zero real-side repeat fraction is a bug
     report, not a statistic.

   **(b) Across-training data support — pin the TOTAL support** `[A21]`.
   Batch diversity says nothing about how much real data the critic sees over
   the run: a perfectly 1:1 batch drawn forever from 200 cached crops still
   only ever shows the critic 200 crops. So, separately:
   - **Preferred — no cache at all**: sample fresh GT latent frames from the
     **full training dataset** on each D update. The frames are already on the
     loader's path and the D-loop decode is `no_grad` (§5.2), so this is the
     cheap option as well as the correct one.
   - **If caching is required** (loader coupling, throughput): the cache holds
     **≥ 4,096 distinct source frames as a hard minimum**, **8,192–16,384
     preferred**, and is **continuously refreshed** — the resident set must not
     be the same 4,096 frames for the whole 400–600-step run.
   - Draws are **cross-ride**, **uniform within the nuisance band** (§3.6), and
     **never nearest by texture or latent similarity**.

   **Explicit warning.** The parked transition critic's real supply was ~22
   transition windows. Do **not** let this build congratulate itself for
   replacing "22 transition windows" with "200 repeatedly sampled decoded
   crops" — that is the same failure at a larger constant, and it is exactly
   what a naive crop-expansion of a small window pool produces. The A21 floor
   is what makes it a different design, and it is **measured** (§7,
   `pix_real_support_frames`), never assumed from the loader.

   *[RESOLVED — was "[FLAG — unresolved]"]* the original draft also drew reals
   from the cross-ride latent ring (`_ladd_real_ring` slabs, sized by
   `ladd_real_pool_cross_ride`). That ring is apparatus **parked** with the
   transition critic (GAN_REDESIGN standing decision 2; Point 7 [CC]: "a
   texture critic's reals are simply GT latent frames"), and in A21's terms it
   is a *cache*. B2 therefore builds its **own** real-supply path meeting (a)
   and (b); the ring stays parked. `pix_real_ring_enabled` is **withdrawn as a
   concept** rather than defaulted off, because "the ring is off" must never be
   readable as "the diversity telemetry is optional". If B2's own path turns
   out to need a cache, that cache carries the A21 floor and the §7 telemetry
   **unconditionally** — the telemetry is never gated on a source flag.
5. **No noising, no timestep conditioning** (pixels are clean; the APT t=0
   degeneracy applies to diffusion features, not to a from-scratch pixel
   critic), **no equalisation** (per §2 both sides share the decoder, so
   absolute level is a legitimate cue), **no action/prompt conditioning**.
   All three are also **frozen out** by the B2 scope freeze `[A19]`, so they
   are not re-openable as tuning during the arm.
6. **Vertical-band-matched real/fake crop pairing — COARSE, permanently**
   `[A3]` `[A24]`. In dashcam footage
   **vertical position is a strong proxy for content class** — sky at top,
   buildings mid, road surface at bottom — and those have genuinely different
   texture statistics. Random crops let a fake *road* patch be scored in the
   same batch as a real *sky* patch, which for a marginal critic averages out
   but adds variance and can drag road texture toward sky texture.
   **Rule**: the admissible top-row offset range for a 24-row crop out of the
   60-row latent is `[0, 36]`; quantise it into **3 equal bands** — top /
   middle / bottom thirds — (`pix_band_count=3`, offsets `0–12 / 12–24 /
   24–36`), draw the fake crop's band uniformly, and draw **every real crop
   paired with it from the same band**. Log the per-step band histogram (§7).

   **Coarse is a hard constraint, not a starting default** `[A24]`.
   `pix_band_count` is restricted to *thirds, or a handful of coarse bins* — 3
   as specced, 2 or 4 if an arm deliberately probes the sensitivity. **Exact
   y-coordinate matching is forbidden**, and so is every refinement that
   approaches it: per-row bands, `pix_band_count` scaled with the latent row
   count or the patch grid, or matching the real's top-row offset to the fake's
   ± a small tolerance.
   The principle is Point 7's, and it is the reason this constraint is
   permanent: **match the nuisance, never the target.** Vertical position is a
   nuisance covariate; texture is the judged property. Tightening the band
   until the real is effectively "the GT crop at the fake's y" reintroduces a
   *matched* real supply by degrees — the exact pathology the parked transition
   critic died of, arrived at through a knob instead of a design decision, and
   slowly enough that no single change looks wrong. If band-matching ever
   appears to need tightening in order to work, that is evidence **about the
   critic**, not a licence to tighten it.
   *Honest limit*: a 24-row crop spans 40 % of frame height, so the three
   bands overlap heavily — band-matching constrains the crop **centroid**, it
   does not isolate sky from road. It is a variance reduction, not a
   partition. This is nuisance-matching in the Point-7 sense (match the
   nuisance covariate, never the judged property) and does **not**
   reintroduce nearest-L1.
7. **Sample count is bought with crops and frames, not with patch positions**
   `[A3]`. With `pix_crop_lat=(24,32)`, a stride-8 patch grid and a ~70 px
   receptive field, adjacent patch logits share almost all of their receptive
   field (**~128:1 overlap**), so the patch grid is *not* an independent
   sample of the texture distribution:

   | quantity | value |
   |---|---|
   | patch logits per image | 768 |
   | non-overlapping receptive-field tiles per image | 6 |
   | overlap factor | ~128 : 1 |

   | config | fake images/step | patch logits | **effective** |
   |---|---|---|---|
   | original draft (`crops=2, frames=2`) | 4 | 3 072 | **~24** |
   | **`crops=4, frames=3`** | 12 | 9 216 | **~72** |
   | **`crops=8, frames=3`** | 24 | 18 432 | **~144** |

   **Spec: `pix_crops_per_step=4–8`, `pix_frames_per_crop=3`** — start at 4,
   raise to 8 if the §7 patch-logit variance or the §8.1 control says the
   critic is sample-starved. Crops are cheap (that is the point of
   latent-crop-then-decode); note the G-side grad-decode cost scales with
   `pix_crops_per_step × pix_frames_per_crop`, while the D-loop decodes stay
   `no_grad`.

   **This table is the FAKE side only.** Under the A20 real-batch invariant
   (§3.4a) the real side scales **1:1 with it**: 12 fakes means 12
   independently drawn reals, 24 fakes means 24. Raising `pix_crops_per_step`
   therefore raises the `no_grad` real-decode count in the same proportion —
   see §6 — and the extra reals must be *independent source frames*, not extra
   frames from the crops already drawn.

## 4. Critic architecture (`model/pixel_texture_disc.py`)

PatchGAN, **661,953 params** (measured on the built module; the layer table
below is exact — the earlier "~2-3 M" figure was arithmetic error, corrected
2026-08-23 by WP-PIXGAN and an independent reviewer). From scratch (no frozen
backbone — the teacher
feature basis is the thing we are moving away from for texture):

```
input  [N, 3, h, w] in [-1, 1]
conv 3→64   k4 s2  spectral-norm, LeakyReLU(0.2)
conv 64→128 k4 s2  spectral-norm, GroupNorm(8), LeakyReLU
conv 128→256 k4 s2 spectral-norm, GroupNorm(8), LeakyReLU
conv 256→1  k3 s1  spectral-norm            → per-patch logits [N, 1, h/8, w/8]
```

Receptive field ≈ 70 px — a *local* texture question. **Per-patch logits are
kept** (no global scalar): the D/G losses consume the patch grid directly,
fixing the 46,800-tokens-to-one-scalar collapse of the LADD critic
(GAN_ARCHITECTURE_BRIEF §2.3). For `pix_crop_lat=(24,32)` the map is
`[N, 1, 24, 32]` = 768 logits per image BEFORE the border trim; **after §3.2's
8-px trim the real count is 660** (192x256 -> 176x240 -> /8 -> 22x30). Harmless
under the mean reduction, but anything dividing by a literal 768 is wrong.
This is the `P` that §5.3's
mean reduction divides by `[A19]`.

### 4.1 From-scratch is a RECORDED EXCEPTION, not an oversight

Standing decision 4 in `GAN_REDESIGN.md` reads absolutely ("we never train a
critic from scratch; every discriminator projects onto a pretrained network").
The from-scratch PatchGAN here is a **deliberate, recorded exception**,
resolved 2026-08-23 (GAN_REDESIGN "[CC] (a) … RESOLVED"), on three grounds:

- **The VQGAN/SD precedent actively supports from-scratch here.** The latent
  space we operate in was *itself* produced by a from-scratch pixel PatchGAN,
  adopted for exactly our symptom (reconstruction + perceptual losses give
  insufficient fine image statistics). This is the reference implementation of
  the job, not an exotic choice.
- **The hypothesis class is small and the target is low-order.** A 2–3 M-param
  critic with a ~70 px receptive field learning a *marginal texture* statistic
  is a far easier learning problem than the semantic features a pretrained
  backbone supplies.
- **A first signal quickly beats optimality.** If a from-scratch critic
  separates, the design is validated and a pretrained backbone becomes an
  upgrade; if it does not, we escalate with evidence.

**The budget risk transfers intact** (GAN_REDESIGN Point 6): ~180 D-updates to
learn texture statistics from nothing, and `d_loss ≈ ln 2` reads identically
for *undertrained* and *wrong design*. Two mitigations are therefore
**mandatory**, not optional: the extended 400–600-step budget read as a
trajectory (§5.5) and the **positive control** (§8.1).

**Escalation** (GAN_REDESIGN B5): if the positive control still reports
UNDERTRAINED after an extended budget, replace the from-scratch stack with a
**pretrained pixel backbone** — DINOv2 (what ADD used for this exact job) or
the SD/VQGAN discriminator weights (trained for this objective on decoded
latents). That is the escalation trigger and it is the *only* condition under
which the exception is withdrawn.

## 5. Losses, penalties and schedule

### 5.1 Adversarial loss — patchwise non-saturating logistic or hinge `[A3]`

**NOT position-matched RpGAN.** The original draft specified RpGAN (R3GAN) at
patch level, position-matched per crop-pair. That asserts a correspondence
that does not exist: reals and fakes here are **unrelated scenes**, so patch
`(x,y)` of the fake has no counterpart at patch `(x,y)` of the real, and
per-position relativistic comparison is meaningless noise (GAN_REDESIGN
Point 1, "Why relativistic pairing must also change"; resolved in [CC] (b)).

```
D_loss = mean(softplus(-D(real_patches))) + mean(softplus(D(fake_patches)))
G_loss = mean(softplus(-D(fake_patches)))
```

`pix_loss_form ∈ {nsgan, hinge}`, default `nsgan`; `hinge` is the same
reduction with the hinge nonlinearity. The reduction is `mean_i[f(D_i)]` —
the nonlinearity is applied **per patch, before** any averaging, over both the
patch grid and the frame/crop batch rows.

Deliberate trade to record: dropping the relativistic form loses R3GAN's
convergence argument — **and, with it, the R1+R2 recipe** `[A19]`. R3GAN's
stability result is stated for the *relativistic* pairing regularised on both
branches; a non-relativistic patchwise critic does not inherit it, so B2 does
not inherit R2 either. What carries over is the zero-centred **R1** penalty on
the real branch alone — the standard regulariser for exactly this kind of
critic. **R1 = ON, R2 = OFF**; see §5.3.

### 5.2 Optimizer and cadence

- Separate Adam, betas (0.0, 0.9), `pix_gan_lr=1e-5`.
- `pix_gan_updates_per_step` follows `gan_updates_per_step` (**5** in the
  ganfix arms — `sbatch/run_ganfix_strict_rerun.sh:19`,
  `sbatch/run_ganfix_marginal.sh:33` — both verified 2026-08-23; the config
  default is 1, `configs/action_forcing_phase3_dmd.yaml:58`, key
  `gan_updates_per_step`).
- Warmups / start-step reuse the `gan_disc_start_step` / `gan_warmup_steps`
  pattern.
- D-updates decode with `no_grad` (inputs detached) — only the G-side pass
  backprops through the decoder, so D-loop cost stays flat.

### 5.3 Regularisation — **R1 ONLY**, and R1 differentiates the **MEAN** `[A19]`

**Directive A19 governs this section. It supersedes A12 (the mean-vs-sum
question is answered: MEAN) and the B2 half of A13 (there is no R1/R2 balance
question, because there is no R2).** Two decisions, both closed:

#### (1) R1 = ON, R2 = OFF

The pixel critic is **non-relativistic** (patchwise NS-logistic/hinge, §5.1),
so it does **not** inherit the R3GAN R1+R2 recipe. Real-branch R1 is the whole
of B2's regularisation.

**R2 has been deleted from the codebase — do not resurrect it for B2.**
Verified 2026-08-23: `ladd_r2*`, `r3gan_r2*` and `_ladd_r2_fires` have **zero **[CORRECTED 2026-08-23 — this is FALSE.** `model/dmd_action_forcing.py:1948-1970` still assigns FOUR `ladd_r2_*` attributes as dead stores (nothing reads them since the trainer's R2 was deleted), `configs/action_forcing_phase3_dmd.yaml` still sets the keys, and `sbatch/_fgan_holder*.sh` still pass them. A sub-agent deleted the dead stores and WP-PIXGAN reverted it as out-of-scope for B1. The R2 LOSS PATH is genuinely gone — no R2 penalty is computed anywhere — but the ATTRIBUTE SURFACE remains. It is LADD-side cleanup for whoever unparks that path.]**
references repo-wide**, and `_ladd_count_penalty_fires` now takes a single
`do_r1` argument (it emits `train/r3gan_r1_fired_total` and nothing for R2).
Concretely, for B2 there is **no** `pix_r2_gamma`, **no** `pix_r2_sigma`,
**no** `pix_r2_every_n`, **no** `pix_r2_phase_offset` and **no** R2 fire rate.
A build that introduces any of them is off-spec, and a run that logs a
`pix_r2_*` key has drifted.

The **only** future condition under which R2 is revisited: fake-side
discriminator gradients *demonstrably* need it — and then only for the
**parked RpGAN transition critic**, which is relativistic and does inherit the
R3GAN argument. Never for B2.

#### (2) R1 differentiates the MEAN patch score per image, never the sum

The critic emits `P = (h/8)·(w/8)` patch logits per image (768 at
`pix_crop_lat=(24,32)`, §4). The penalty is taken on the per-image **mean**
patch score:

```
s_n(x)  = (1/P) · Σ_i D_i(x_n)                          # per-image MEAN patch score
gsq_R1  = mean_n( ( ( s_n(real_n + εσ) − s_n(real_n) ) / σ )² )
R1      = 0.5 · γ_R1 · gsq_R1
```

- Finite difference on **pixels**: `pix_r1_sigma = 0.01` in the `[-1, 1]` input
  scale.
- **`γ_R1 = 1.0`** as the starting value, defined **against the mean**. That is
  what makes 1.0 portable: a sum-based `gsq` scales with `P²`, so a γ tuned at
  one crop size or patch-grid stride is meaningless at another. The mean is
  scale-free in `P` by construction.
- The mean reduction is **not a flag.** There is no `pix_r1_normalize` knob to
  leave off, and no un-normalised branch to fall back to; `s_n` *is* the
  quantity R1 regularises.
- Log the **raw** `gsq` **pre-γ** (§7, `pix_r1_grad_sq`) so γ is recalibrated
  from the first run's measured magnitude instead of assumed. The LADD path
  logs `train/r3gan_r1_grad_sq` for exactly this reason — grep the log key, not
  a line number.

#### Why MEAN is now a design choice, not a bug fix

A12 originally justified the mean reduction as the **root cause** of the
campaign's replicated *"R1 γ=1e6 pins every discriminator at ln 2"* finding —
the T²-artefact reframe. **That justification has been FALSIFIED** and is
withdrawn (`GAN_REDESIGN.md` Point 1, "[WITHDRAWN 2026-08-23 — FALSIFIED]").

The falsifying evidence: `sbatch/_roll_holder.sh` ran `ladd_r1_gamma=1e6`
**together with** `ladd_r1_normalize_tokens=true` **and token logits** — it
never sets `ladd_scalar_output`, which defaults to `False` in the trainer's
config read — and **still pinned at ln 2**. The T² artefact therefore cannot
explain the pinning; γ was simply ~1e6× too large.

Consequences to carry forward, precisely:

- Mean-vs-sum is settled on **scale-portability** grounds — γ=1.0 means the
  same thing at any crop size and any patch count — and **not** as a remedy for
  ln-2 pinning.
- Do **not** expect the mean reduction to buy a stability result it was never
  shown to buy. If B2 pins at ln 2, the mean reduction is not the explanation
  and γ is the first thing to re-read off `pix_r1_grad_sq`, not the last.
- Do not propagate the falsified justification into any downstream document.

#### Do not reuse the LADD penalty code paths verbatim

Audited 2026-08-23. Four separately-written R1 estimators survive on the LADD
side (matched micro-batched, matched inline, positional autograd, positional
FD). All of them estimate `‖∇_x Σ_i D_i‖²` unless `ladd_r1_normalize_tokens` is
set, they were normalised at different times, and they carry **different
cadence state** (a debt latch on the matched path, plain `step % n` on the
positional one) — see `GAN_REDESIGN.md` A16/A17 for the full audit and the
`ladd_r1_unified_cadence` repair.

The pixel critic implements **one** estimator, with the mean reduction baked in
and no normalisation flag to get wrong. Only the *cadence and counter*
machinery is reused (§5.4).

### 5.4 Cadence — one penalty, one fire rate, logged from day one `[A19]`

With R2 gone there is no R1/R2 balance question for B2: **A13's balance half is
superseded by A19; its fire-rate logging requirement stands, for R1.** The
history is worth stating anyway, because it is *why* the logging is mandatory.

The monotone counters added 2026-08-23 (`_ladd_count_penalty_fires`, surfaced
as `train/r3gan_r1_fired_total`) exposed an unintended asymmetry in the LADD
path: in the `_gtxn` arms **R1 fired at 0.20 while R2 fired at ~0.49** — the
fake-side penalty applied ~2.4× more often than the real-side one, for the
whole campaign, invisibly.

**The cause was not what it first looked like.** `ladd_r1_once_per_step` was
blamed and was **provably inert**: it has *zero* code references, and the
trainer's own override-guard comment names it as one of three flags that
silently did nothing (grep `ladd_r1_once_per_step` — the only hit is that
comment). The debt latch already consumed the debt on the first D-update, so R1
could not fire twice per step regardless. The real cause was **the same penalty
implemented twice**: the matched path is debt-latched (`_ladd_last_r1_step`,
one fire per step over `gan_updates_per_step=5` → 0.20) while the positional
path used plain `current_step % n` with no latch (→ ~0.53).

The lesson B2 inherits is structural, and it survives the deletion of R2: *a
penalty implemented twice will fire at two rates, and nobody notices without a
monotone counter.*

**B2 has one estimator and one rate.**

| knob | value | consequence |
|---|---|---|
| `pix_r1_every_n` | **1** | R1 fires on **every** D-update |
| `pix_r1_once_per_step` | **absent — not implemented** | no 1-of-N cap (and the LADD flag of that name was inert anyway) |
| `pix_r2_every_n`, `pix_r2_phase_offset`, any `pix_r2_*` | **do not exist** `[A19]` | R2 is off and deleted; there is no second rate to balance against |

**Target fire rate: R1 = 1.00.** The LADD phase-offset guard
(`ladd_r2_phase_offset`, itself now deleted with R2) existed to stop R1's and
R2's perturbed forwards stacking inside a single D-update — an OOM guard for a
full-depth Wan forward. It is doubly inapplicable here: there is no R2 to stack
with, and the pixel D-loop decodes are `no_grad` (§5.2) feeding a ~2–3 M-param
conv on 192×256 crops, so the one perturbed forward is trivial.

**If R1's cost ever needs bounding, subsample — do not skip.** An optional
`pix_r1_num_samples` may cap how many reals are perturbed per update; the
penalty is already a mean over per-real squared finite differences, so a
subsample is statistically free (this is the A17 resolution on the LADD side).
Subsampling keeps `pix_r1_rate = 1.00` by construction, whereas raising
`pix_r1_every_n` reintroduces exactly the silent-cadence class of bug described
above.

**Mandatory day-one telemetry** (§7): `pix_r1_fired_total`,
`pix_dupdate_total`, and the derived `pix_r1_rate` — as **monotone counters,
not per-step gauges.** The per-step gauge is what aliased to a permanent 0 and
produced the "R2 never fired" false alarm that survived months of runs (A10).
Read them on the **first** logged row; treat `pix_r1_rate < 0.99` as a build
bug, not a tuning outcome.

### 5.5 Generator weight — re-bracket from scratch `[A3]`

**`pix_gan_weight=0.03` is withdrawn as a default.** The measured
0.01-inert / 0.03-learns / 1.0-destroys bracket was established under a
**scalar latent** critic and does **not** transfer: new domain (pixels), new
reduction (768 patch logits, nonlinearity before the mean), new loss form.

Calibrate against telemetry rather than inheriting a constant:

1. Run a short weight-free probe and read `gan_dmd_grad_ratio` (A7 telemetry,
   already live at `gan_grad_telemetry_every=25`). The historical LADD arms
   measured **0.0005–0.012** (0.05–1.2 %) — far under the intended 5–20 %
   band, i.e. those critics were *small and near-irrelevant*.
2. Pick `pix_gan_weight` so the ratio lands in the **5–20 %** band, then run a
   3-point bracket at ×1/3, ×1, ×3 around it.
3. **There is no GAN-side gradient cap to fall back on.**
   `gan_grad_target_norm` (A4) was **REMOVED 2026-08-23** — `_apply_gan_grad_cap`,
   its flag parse and both call sites are deleted (the trainer carries the
   removal note at three sites; grep `gan_grad_target_norm cap REMOVED`).
   `train/gan_grad_norm` consequently reports the **uncapped** norm, which is
   what makes the step-1 ratio read meaningful in the first place. If an
   anomalous-excursion guard is wanted later it is a new build and a new
   decision, not an existing safety net. The surviving analogue is the DMD-side
   `dmd_grad_target_norm` (`model/dmd_action_forcing.py`, grep the symbol) — a
   cap-only rescale, never a per-rung downweight.

**Run length: 400–600 steps, and the readout is the `pix_d_loss`
TRAJECTORY, not its endpoint.** A from-scratch critic is *expected* to sit at
chance early; that is not evidence of anything. 200-step reads on this arm are
uninterpretable by construction (§4.1).

The pixel G-term is **added into `gen_gan_loss`** so the existing A3/A7 grad
telemetry (`train/gan_grad_norm`, `gan_dmd_grad_ratio`, `gan_dmd_grad_cos`)
measures it with no extra wiring.

## 6. Memory budget

Per gen update: 1 grad-decode of `pix_crops_per_step` crops
(13-frame 24×32 latent → 12×192×256 px each) — at `crops=4` that is decoder
activations for ~1.2 megapixel-frames, at `crops=8` ~2.4, small next to the
92 GB disc-transient lessons; D-loop decodes are `no_grad`. Pixel critic
forward is trivial (661,953 params). Target: fits inside the strict arm's
existing 45–48 GB envelope with ≥ 30 GB headroom. **Re-measure at
`crops=8, frames=3` before committing to the upper end of the §3.7 bracket** —
**the grad-decode cost scales with `pix_crops_per_step` ONLY — NOT with
`crops x frames`.** Verified by WP-PIXGAN against
`model/disc_holdout_probe.py::_decode_crops`: it calls `_vae_decode_nograd(sub)`
on the WHOLE latent crop and only THEN draws `sel = randperm(f_pix)[:k]`. Frame
selection happens strictly AFTER decode, so `pix_frames_per_crop` costs nothing
at decode and scales only the critic forward (~0.66 M params, negligible).
**Raising `frames_per_crop` is nearly free — the opposite of what this section
previously advised.**

**Where the memory actually goes (measured by WP-14B, `docs/WP_14B.md` §6).** The
expensive side is the GENERATOR-guidance forward, not the D-update:
  * G-guidance (eval-mode disc forward with the gradient going back to the input
    latent) costs **~8.5 GiB/row un-micro-batched** at F=3 / 60x104, and OOMs at
    12 rows on a 95 GiB GPU. This is true on the **1.3B** path too, so
    `ladd_gen_guidance_micro_batch_groups=4` is **load-bearing for ANY LADD arm**,
    not a 14B quirk.
  * The D-update is cheap by comparison: in train mode the disc input carries no
    grad, so the projector never checkpoints and no graph reaches the frozen
    backbone.
If a critic's memory profile ever surprises you, check that asymmetry first.

**The A20 invariant is a throughput cost, not a memory cost — budget it
anyway.** With one independently drawn real per fake, each D update decodes a
full fresh real batch, and `pix_gan_updates_per_step` D updates run per training
step, so the number of `no_grad` real-crop decodes per step scales as
`pix_gan_updates_per_step × pix_crops_per_step × pix_frames_per_crop`. Peak
memory is unaffected (the decodes are `no_grad` and can be serialised), but the
step time is not, and the A21 preferred path (fresh frames from the full
training dataset) also puts the *loader* on the critical path. Measure the
step-time delta in the smoke rather than assuming it, and if it bites, cut
`pix_crops_per_step` — **never** the 1:1 real ratio and **never** the A21
support floor. Both are correctness requirements (§3.4); crop count is a
variance/throughput trade (§3.7).

## 7. Telemetry (must-have, day one)

- `train/pix_d_loss`, `pix_g_loss`, `pix_d_real_mean`, `pix_d_fake_mean`
- **penalty fire rate** `[A19]`: `pix_r1_fired_total`, `pix_dupdate_total`,
  `pix_r1_rate` — monotone counters, **not** per-step gauges (the per-step
  gauge is what aliased to a permanent 0 and produced the "R2 never fired"
  false alarm; grep `_ladd_count_penalty_fires`). **No `pix_r2_*` key exists**
  — if one appears in a run, the build has drifted off the §5.3 spec.
- **raw penalty magnitude** `[A19]`: `pix_r1_grad_sq`, logged **pre-γ** and
  **mean-reduced** (§5.3), so γ is recalibrated from measurement rather than
  assumed
- patch-logit spatial variance (is the critic using locality?)
- **crop band histogram** `[A3]` `[A24]`: fraction of crops per
  `pix_band_count` band; a real/fake band-mismatch counter (must be 0); and
  `pix_band_count` itself echoed into the run log every run, so a silently
  tightened banding is visible in the trace rather than only in a diff
- **real-supply telemetry — MANDATORY and UNCONDITIONAL** `[A20]` `[A21]`
  (extends A6; no longer contingent on any real-source flag). Two groups,
  matching the two invariants of §3.4:
  - **per D update** (A20): `pix_real_images`, `pix_real_unique_frames`,
    `pix_real_unique_windows`, `pix_real_unique_rides`, `pix_real_repeat_frac`.
    Assert on the first logged row: `pix_real_images == pix_fake_images`, and
    `pix_real_unique_frames == pix_real_images` (i.e.
    `pix_real_repeat_frac == 0`). A violation is a build bug.
  - **across training** (A21): `pix_real_support_frames` — the cumulative count
    of **distinct** source frames ever shown to the critic as real — plus
    `pix_real_support_rides`, and, if a cache is used at all,
    `pix_real_cache_size` and `pix_real_cache_refresh_total`. The A21 floor
    (≥4,096 distinct frames minimum, 8,192–16,384 preferred) is a **measured**
    claim read off `pix_real_support_frames`, never an assumed property of the
    loader.
  - A22 note: whichever real-supply path is built, the held-out rides must be
    provably absent from it — the reserved-ride exclusion is verified against
    *this* telemetry (a prior holdout list leaked into training once before).
- decode count / step, crop coords sample
- the shared A3/A7 cosine/ratio (via `gen_gan_loss`)
- `pix_poscontrol_*` (§8.1)

## 8. What would falsify this design

- `pix_d_loss` → ln2 flatline with G-share < 2 % **and the §8.1 positive
  control reporting INFORMATIVE** → critic can't separate even in pixel space
  → texture problem is not marginal-per-frame (escalate to temporal /
  multi-depth critic). Without the positive control this reading is not
  available — `ln 2` alone is undecidable (§4.1).
- cos(g_pix, g_DMD) strongly negative with visual degradation → adversarial
  signal itself is the problem → switch to non-adversarial pixel objectives
  (arm J / B1: the diagnostic's measured targets — HF-fraction / anisotropy /
  kurtosis matching) — they are now *measured, directional* quantities.

### 8.1 The POSITIVE CONTROL — makes `d_loss ≈ ln 2` decidable `[A14]`

`d_loss ≈ ln 2` reads **identically** for "critic is undertrained" and "design
is wrong". That ambiguity has cost this campaign repeatedly. The positive
control removes it, and costs almost nothing (`no_grad`, off the training
graph, no gradient to anything).

**Mechanism.** Every `pix_poscontrol_every` steps, score, under `no_grad`, a
fresh batch of crops from **`decode(GT)` against a deliberately corrupted
real**, using the same crop/band policy as training. Two corruption sources:

- **C1 — structured-HF perturbation (primary).** Add an anisotropic
  high-frequency perturbation to the GT latent *before* decode, oriented to
  reproduce the measured failure direction: row-periodic energy that drives
  `fft_aniso_fy_over_fx` down from the A/B value (~1.07–1.15) toward the
  measured `C_late` **0.36**, with `angular_entropy` falling from ~0.98 toward
  **0.84**. Calibrate the amplitude **once, offline**, with
  `analysis/texture_stats.py:texture_battery` so the corruption's battery sits
  **between B and C_late** — i.e. it is a *milder* defect than the student's,
  so "separates the corruption but not the student" is a meaningful ordering.
  Self-generated, so it needs nothing on disk and works at any step.
- **C2 — the C_late student output already on disk (secondary).**
  `eval/texture_abc_strict03/rank0_texabc_strict03_texabc/*_rollout_raw.mp4`.
  *Caveat*: the A/B/C diagnostic was run with **no mp4 round-trip anywhere**
  (§0) — an mp4-sourced control is contaminated with codec HF, exactly the
  quantity being measured. Use C2 only as a weak cross-check, or regenerate
  C_late latents so the control is round-trip-free.

**Readout.** Report ROC-AUC of the patch-logit distributions and the mean
patch-logit gap, both with a bootstrap CI over crops, for **two** pairs on the
same step: `GT vs corrupted` (the control) and `GT vs student` (the arm).
Chance is AUC 0.5. Pre-register the "separates" threshold **before the arm
launches** and record it in the run log.

**Decision table:**

| control (`GT vs corrupted`) | arm (`GT vs student`) | verdict | action |
|---|---|---|---|
| cannot separate | — | **UNDERTRAINED** | extend the budget; if it persists, escalate to a pretrained backbone (§4.1, GAN_REDESIGN B5) |
| separates | cannot separate | **INFORMATIVE** | the student's texture is genuinely close *at this depth*; the failure is elsewhere, or deeper in the rollout than the arm reaches — push the arm deeper before redesigning |
| separates | separates | **HEALTHY** | read the arm normally (§9) |

The control is a **rank** comparison first: `AUC(GT vs corrupted)` should
exceed `AUC(GT vs student)` throughout, because C1 is calibrated milder than
the student defect. If that ordering ever inverts, the calibration is wrong,
not the critic.

## 9. Rollout plan

1. Implement flag-gated (this doc) → adversarial review sub-agent →
   fix findings. Launch gates **A23** (inference-parity fake, or the
   two-tensor battery equivalence) and **A22** (held-out generalisation test
   built and its reserved rides verified excluded) both cleared — see the
   banner. Scope freeze re-checked against the build: none of the frozen-out
   items has crept in.
2. Arm `ganfix_pixtex`: strict-rerun DMD base, transition GAN **off**,
   pixel texture GAN on at the §5.5-calibrated weight, **400–600 steps**
   `[A3]` (not 200) + canonical 60 s eval, with the §8.1 positive control
   running throughout.
3. Read, in this order:
   a. **§8.1 positive control** — decides whether steps b–d are interpretable
      at all;
   b. `pix_d_loss` **trajectory** (not endpoint); `pix_r1_rate` (**must be
      1.00**, §5.4 — there is no R2 rate to compare it against); the
      **real-supply invariants** (`pix_real_repeat_frac == 0` per update,
      `pix_real_support_frames` above the A21 floor — §3.4, §7); the A22
      held-out margins; A7 cosine and `gan_dmd_grad_ratio`;
   c. eval video texture at 15–30 s (the researcher's judgement remains the
      primary readout);
   d. the **JOINT battery** below.
4. **Success metric — JOINT, against the B distribution, or it is gameable**
   `[A3]`. Anisotropy → 1.0 alone can be satisfied by replacing vertical
   stripes with isotropic high-frequency *snow*, so all four terms are read
   together (`analysis/texture_stats.py:texture_battery` enforces this by
   returning them as one dict; `battery_delta` gives the ratio to B):

   | term | C_late now | target (the B distribution) | direction |
   |---|---|---|---|
   | `hf_power` | 0.0407 (2.21× A) | 0.0159–0.0165 (0.86–0.89× A) | **DOWN** |
   | `hv_anisotropy` (fft fy/fx) | 0.363 | 1.07–1.15 | → ~1.0 |
   | `angular_entropy` | 0.843 | ~0.98 | → 0.98 |
   | `luma kurtosis` | −1.21 | ~−0.16 to +0.18 | → A/B |

   (numbers from `eval/texture_abc_strict03/REPORT.md`.)
5. **Two separate criteria, both reported** (GAN_REDESIGN Phase-2 framing):
   - **Local correction** — does the battery move toward B *at a fixed depth*?
   - **Dynamical correction** — does that improvement **persist at 10 s, 25 s
     and 60 s**, or does it merely postpone divergence? Re-run the A/B/C
     diagnostic (`analysis/texture_abc/texture_abc_diag.py`) on the new
     checkpoint at **all three depths** `[A3]`. The 1.49× → 2.21× → 3.6×
     progression with depth is an autoregressive amplification mechanism that
     a pixel GAN may or may not reach; this is where the residual shows up.
