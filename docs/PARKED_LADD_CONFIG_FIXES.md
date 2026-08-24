# PARKED — LADD / Wan-projected transition-critic config corrections

> **STATUS: PARKED BACKLOG. NOT PART OF PHASE 1. DO NOT ACTION.**
>
> Every fix on this page applies to the **LADD-path (Wan-projected) transition
> critic**, i.e. the `gt_transition` discriminator built through
> `model/ladd_disc.py` and configured by the `ladd_*` flags. That critic is
> **PARKED in its entirety** — `GAN_REDESIGN.md` standing decision 2 ("the
> transition / temporal critic is PARKED — entirely, for the whole effort, not
> merely deprioritised") and execution-plan item **B6**. Phase 1 is the *pixel*
> texture critic (`TEXTURE_GAN_DESIGN.md`) plus the non-adversarial anisotropic
> anchor; nothing here belongs to it.
>
> This file exists so the corrections are **recorded once, precisely, while the
> evidence is fresh**, and can be applied verbatim on the day the transition
> critic is unparked. Applying any of it before then re-opens a path we have
> deliberately switched off.
>
> This is `GAN_REDESIGN.md` execution-plan item **A8**.

**One live exception — do not park it with the rest.** The *root cause* behind
fix 2 (summed-vs-mean logit scaling in R1/R2) transfers directly to the Phase-1
**pixel** critic, which emits a 768-patch logit map per image. That is execution
item **A12**, it is a live Phase-1 correctness question for the pixel critic's
spec, and it must be resolved there regardless of this file's parked status. Only
the *`ladd_*` flag settings* below are parked.

---

## The corrections

| # | flag | current value | correct value | one-line reason |
|---|---|---|---|---|
| 1 | `ladd_scalar_output` | `true` (forced by every arm) | **`false`** (= the code default) | restores per-token patch logits; Point 1 |
| 2 | `ladd_r1_normalize_tokens` | `false` | **`true`** | un-normalised R1 on token logits scales ~T² over ~47k tokens |
| 3 | `ladd_diff_aug_policy` | `flip` | **`flip,translation`** | translation buys shift-invariance; cutout deletes the evidence |
| 4 | mean / cross equalisation | `mean_equalize=true` in the probe arms | **all off** | equalisation *couples* the two distributions; augmentation only broadens each |
| 5 | `ladd_disc_timestep_shift` | `5.0` | **`<1` (≈0.25–0.5) or a fixed low `t`** | shift 5.0 puts 76 % of disc samples above t=625, 3 % below t=208 |
| 6 | `ladd_feature_blocks` | `[]` → auto `[6,12,18,24,29]` | **`[0,2,4,8,29]`** | nothing before block 6; texture lives in the early taps; Point 2 |
| 7 | R1/R2 fire-rate balance | R1 **0.20** vs R2 **0.49** | an explicit, logged, *intended* rate | `ladd_r1_once_per_step=true` × `gan_updates_per_step=5`; R3GAN assumes balance |

---

### 1. `ladd_scalar_output=false` — restore patch logits

* **Current:** every arm we have run sets it explicitly true —
  `sbatch/run_ganfix_strict_rerun.sh:19`, `sbatch/run_full_carn_probe.sh:55/58/64/75`.
* **Correct:** `false`. This is **already the code default**
  (`trainer/causal_action_forcing_train.py:883`,
  `scalar_output=bool(getattr(self.config, "ladd_scalar_output", False))`), so
  the fix is a *flag deletion*, not new code.
* **Reason** (`GAN_REDESIGN.md` Point 1): with `gt_transition` inputs the head
  produces 5 taps × 6 frames × 1560 tokens = **46,800 local realism votes**, and
  `model/ladd_disc.py:911-915` averages them *before* the adversarial
  nonlinearity. The decisive harm is not the (small) Jensen gap at our
  `d_loss ≈ 0.59–0.69` operating point — it is **gradient uniformity**: with
  `D = mean_i d_i`, `∂L/∂d_i = softplus'(Δ)/N` is *identical for every token*, so
  the critic is structurally incapable of telling the generator *which region* is
  wrong. With the flag false, `rpgan_d_loss` already `.mean()`s over all dims,
  i.e. it already computes `mean_i[softplus(·)]`.
* **Rider:** restoring patch logits while keeping *position-matched* RpGAN would
  assert that fake patch `(x,y)` corresponds to real patch `(x,y)` — false for
  our unpaired cross-ride reals. Pair with either the shuffled-pair RpGAN variant
  or the plain patchwise non-saturating/hinge loss, and expect to **re-bracket
  `gan_loss_weight` from scratch** (the 0.01-inert / 0.03-learns / 1.0-destroys
  bracket was established under scalar output and does not transfer).

### 2. `ladd_r1_normalize_tokens=true` — un-normalised R1 scales as T²

* **Current:** `false` in every arm (`run_ganfix_strict_rerun.sh:19`,
  `run_full_carn_probe.sh:55/58/64/75`); also the code default.
* **Correct:** `true` — mandatory the moment fix 1 lands.
* **Reason:** the in-code comment at
  `trainer/causal_action_forcing_train.py:7071-7076` (mirrored at `:5090-5096`)
  states it outright: with token logits, R1's `grad_sq` estimates
  `‖∇ Σ_i D_i‖²`, which **"scales with ~T² over ~47k tokens, forcing γ to a tiny
  un-portable value"**. Normalising by `d_r.shape[1]` makes R1 estimate
  `‖∇ mean_i D_i‖²`, which is token-count-independent and therefore portable.
* **Why it matters beyond the config:** this is the **likely mechanism behind the
  campaign's replicated "R1 γ=1e6 pins every discriminator at ln 2" finding**. If
  so, that headline result is a *scaling artefact*, not a law about our critic —
  and it should be struck from the record as evidence about discriminator
  capacity. Do not carry the γ=1e6 conclusion forward into any new critic without
  re-measuring it under token normalisation.
* **Transfers to Phase 1 (A12):** the pixel PatchGAN emits a 768-patch logit map,
  so `TEXTURE_GAN_DESIGN.md` §5's "R1/R2 at γ=1.0 on pixels" hits the *same*
  summed-vs-mean scaling. B2's spec must say explicitly whether R1/R2
  differentiate the **mean** or the **sum** of the patch map, and set γ to match.

### 3. `ladd_diff_aug_policy=flip,translation` — restore translation, keep cutout off

* **Current:** `flip` in every arm (`run_full_carn_probe.sh`, all modes;
  `run_ganfix_strict_rerun.sh:19`).
* **Config default:** `flip,cutout,translation`
  (`configs/action_forcing_phase3_dmd.yaml:98`) — so we actively *narrowed* it.
* **Correct:** `flip,translation`.
* **Reason** (Point 8): **translation/crop is exactly the augmentation that
  delivers shift-invariance for a stationary property like texture**, and we
  turned it off, then added mean/cross equalisation to suppress the resulting
  brightness shortcut instead — the wrong branch at that fork. **`cutout` should
  stay OFF**: it deletes texture regions, which is precisely the evidence a
  texture critic needs; it is an object-level-GAN regulariser, not a
  stationary-statistics one.
* **Explicitly not recommended:** "exposure jitter". In latent space its nearest
  analogue is shifting per-channel means — *exactly* the quantity CARN corrects
  and the drift probe measures as the dominant AR drift. Jittering it would make
  the critic blind to genuine DC drift. Treat exposure jitter as an RGB-domain
  (pixel-critic) tool only, where legitimate exposure variation and drift are
  separable.

### 4. No mean / cross equalisation

* **Current:** `ladd_gt_transition_mean_equalize=true` (+ `xeq_preserve_delta`)
  in the `run_full_carn_probe.sh` wave/strict arms; already `false` in
  `run_ganfix_strict_rerun.sh:19`.
* **Correct:** **off** — mean, cross and std equalisation all off for the texture
  objective.
* **Reason** (Point 8, and the mechanism is the whole argument):
  * **Equalisation COUPLES the two distributions** — it forces real and fake to
    share moments, and the information is **destroyed before the critic ever sees
    it**. Neither side can be judged on tone or local contrast again, *ever*.
  * **Augmentation BROADENS each distribution independently** — the critic
    becomes *invariant* to the nuisance without the nuisance being removed, and
    anything the augmentation does not span stays discriminable.
  For a texture critic the second is strictly better, because local contrast,
  tone relationships, colour-channel noise and road-surface luminance
  distributions are **part of the target, not nuisance**. (Equalisation may still
  be worth keeping in a *latent dynamics* critic, where they genuinely are
  nuisance.)

### 5. Disc timestep: shift < 1, or a fixed low `t`

* **Current:** `ladd_disc_timestep_shift=5.0` with `ladd_disc_sample_t=true`,
  `t∈[20,980]` (`run_ganfix_strict_rerun.sh:19`,
  `run_full_carn_probe.sh:42`); code default is also `5.0`
  (`trainer/causal_action_forcing_train.py:5767`).
* **Correct:** a **fixed low `t`**, or `ladd_disc_timestep_shift ≈ 0.25–0.5` to
  move the mass under `t ≈ 357`.
* **Reason** (Point 2c, **measured**, `t = 1000·s·u/(1+(s−1)u)`, `u ~ U[0.02,0.98]`,
  sampled at `:5776-5782`):

  | shift | p25 | median | p75 | frac `t > 625` | frac `t < 208` |
  |---|---|---|---|---|---|
  | 1.0 | 260 | 502 | 741 | 0.37 | 0.20 |
  | **5.0 (ours)** | **637** | **834** | **935** | **0.76** | **0.03** |

  **76 % of discriminator samples land above t=625 and only 3 % below t=208** —
  i.e. overwhelmingly where texture has already been destroyed by noise. LADD's
  own claim is the opposite: *low*-noise generative features are the ones that
  carry texture / local-detail feedback.
* **Note the symmetry of the error:** the wavelet branch forced `t=0`; the
  `strict03` / projected arms were introduced to escape that and replaced it with
  a distribution in which texture is largely gone. **Both settings are wrong for
  texture, in opposite directions.**

### 6. Early taps `[0,2,4,8,29]`

* **Current:** `ladd_feature_blocks: []`
  (`configs/action_forcing_phase3_dmd.yaml:74`) → the auto-default for a 30-block
  1.3B teacher resolves to **`[6,12,18,24,29]`**
  (`trainer/causal_action_forcing_train.py:737-747`), i.e. **nothing before block
  6**.
* **Correct:** `[0, 2, 4, 8, 29]` — shift the mass early, keep one late tap for
  structure.
* **Reason** (Point 2a): LADD extracts the full token sequence after *each*
  attention block and puts independent heads on them, emphasising that low-noise,
  *early* generative features carry the texture signal. Our five sparse, mature
  taps are not "LADD on Wan". This is a **config-only change** —
  `_validate_block_indices` (`model/ladd_disc.py:138-147`) only requires
  `0 ≤ idx < n_blocks`.
* **Adjacent, NOT config-only** (record, do not conflate):
  * the **patch-embedding output is unreachable** — `WanFeatureProjector._find_blocks`
    locates only the transformer-block `ModuleList`; hooking the patch embed is a
    small code addition.
  * `dim_teacher=1536` → `ladd_proj_dim=256` is a **6× channel bottleneck in
    front of every head** (`model/ladd_disc.py:294`), applied *before* the head
    sees anything. Widen `proj_dim` (or use per-tap dims) on the texture-bearing
    early taps, or the re-tap buys less than it looks.
  * our hook fires on the **whole block output** (attn + FFN + residual,
    `model/ladd_disc.py:231-241`); LADD taps after the attention block. Similar
    granularity, not identical.
  * changing the tap set changes `dim_teacher` bookkeeping and the CSM fusion
    ordering — verify the FPN top-down path still makes sense with unevenly
    spaced taps.

### 7. R1/R2 fire-rate imbalance — needs an intended-rate decision

* **Measured (2026-08-23, monotone counters):** in the `_gtxn` arms
  (`strict_rerun`, `poolrich`) **R1 fires at 0.20 while R2 fires at 0.49** — the
  fake-side penalty applied **2.4× more often** than the real-side one. The `_gt`
  `marginal` arm is balanced at 0.53/0.47, so the imbalance is specific to the
  flag combination, not universal.
* **Cause:** `ladd_r1_once_per_step=true` caps R1 at the *first* of
  `gan_updates_per_step=5` disc updates, while R2 has no such cap
  (`run_ganfix_strict_rerun.sh:19`, `run_full_carn_probe.sh:75`; gating logic at
  `trainer/causal_action_forcing_train.py:6998-7014`). Note
  `configs/action_forcing_phase3_dmd.yaml:59` ships `gan_updates_per_step: 1`,
  at which the flag is harmless — the imbalance is created by the *arms*.
* **Correct:** no single value; this needs an **explicit intended-rate decision**
  that is then **logged from day one**. Either drop `ladd_r1_once_per_step`, or
  cap R2 symmetrically, or state the asymmetry as deliberate with its
  justification. **R3GAN's stability argument assumes balanced R1+R2**, so
  inheriting a silent 2.4× asymmetry silently voids it.
* **Related, and settled:** the doubt over whether R2 ever fired is **resolved —
  it did**: 85 fires over 175 disc updates (48.6 %). The per-step `r2_fired`
  gauge reading 0 on every logged historical row was a **sampling alias**. "We
  ran R1+R2" is TRUE; strike that doubt from `GAN_REDESIGN.md` Point 5 [CC],
  `GAN_ARCHITECTURE_BRIEF.md` §7 and `ROLLING_CAMPAIGN.md` (execution item A10).
* **Transfers to Phase 1 (A13):** `TEXTURE_GAN_DESIGN.md` §5 explicitly reuses
  this cadence machinery and its monotone counters, so **B2 must state its
  intended fire rates and log them from day one** rather than inherit the
  asymmetry.

---


> **[CC] VERIFIED 2026-08-23 — the flag flip ALONE is not safe.** R1 and R2 are
> normalised *asymmetrically* in the code. In the micro-batched path (the one our
> arms use, `ladd_disc_micro_batch_groups=2`):
>
> ```python
> gsq_terms  = ((d_r_pert_g.sum(1) - d_r_owned.sum(1)) / (_r1_sigma * _r1_tok)).pow(2)  # R1: /(sigma * TOKENS)
> gsq2_terms = ((d_f_pert_g.sum(1) - d_f_g.sum(1))     /  _r2_sigma          ).pow(2)  # R2: /sigma ONLY
> ```
>
> `_r1_tok` is the token count when `ladd_r1_normalize_tokens=true`; **R2 has no
> token normaliser anywhere**. Both sum over the token axis, so R1 would estimate
> `||grad MEAN_i D_i||^2` while R2 still estimates `||grad SUM_i D_i||^2`.
>
> **Why no arm has been bitten yet:** with `ladd_scalar_output=true` the logit
> tensor is `[B, 1]`, so `T = 1` and the two are consistently scaled.
> **The bug is latent and this backlog would trigger it**: setting
> `ladd_scalar_output=false` *and* `ladd_r1_normalize_tokens=true` together — the
> exact pair recommended here — leaves R2 larger than R1 by ~T^2 (T ~ 46,800, so
> ~1e9x) at equal gammas.
>
> **Therefore this is a CODE change, not a config flip:** add the matching
> normaliser to R2 (all three branches: micro-batched, FD, autograd) before
> flipping either flag. Stacks with the A10 *cadence* imbalance — that one is
> about how OFTEN each fires, this one about how BIG each is.

## Also parked with the critic (not config, recorded for the same day)

These come from the same standing decision 2 / B6 and are listed so the unparking
review has one page to work from:

* **nearest-L1 real matching must go** (Point 7). The nuisance-matching rule:
  match exposure / time-of-day / weather — **never the judged property**.
  Nearest-L1 matches on the very thing being judged.
* the **cross-ride ring** (`ladd_real_pool_cross_ride=4096`) and its telemetry —
  live in the trainer, parked with the ring itself.
* Point 1's **test arms C/D** target this critic; they are **not** Phase-1 arms.
* **Real-sample diversity telemetry** (A6: unique rides / unique source windows /
  repeat rate per D-batch) should land *before* the unparked critic is judged.

## Order of application, when unparked

1. Fixes **1 + 2 together** — they are one change (patch logits are unusable
   without token-normalised R1), and their arm pair is Point 1's C/D table.
2. Fix **5** then **6** — the timestep and tap fixes are both "put the critic
   where texture actually is", and 6 is confounded by 5 if applied alone.
3. Fixes **3 + 4** together — one is the replacement for the other.
4. Fix **7** — decide and log the rate before any new critic inherits the
   machinery.

Every one of these lands **flag-gated and default-off, byte-identical when
disabled, adversarially reviewed before launch** — the standing constraint,
unchanged.

---

## CORRECTION 2026-08-23 — fix 2's justification is WITHDRAWN (falsified)

Fix 2 (`ladd_r1_normalize_tokens=true`) was justified here and in
`GAN_REDESIGN.md` on the grounds that un-normalised R1 over ~47k token logits
scales ~T^2 and is "the likely mechanism" behind the campaign's replicated
"R1 gamma=1e6 pins every discriminator at ln 2" finding — i.e. that the headline
result was a scaling artefact.

**That is false.** `sbatch/_roll_holder.sh` sets `ladd_r1_gamma=1e6` **together
with** `ladd_r1_normalize_tokens=true`, and never sets `ladd_scalar_output`
(code default `False` = token logits). So the gamma=1e6 arms already ran token
logits **with** the mean normalisation and still pinned at ln2. gamma was simply
~1e6x too large under the mean reduction; the ln2 finding STANDS as measured.

Fix 2 may still be desirable for portability, but it must not be sold as
explaining the pinning, and it is no longer a prerequisite for anything.

## CORRECTION 2026-08-23 — fix 7 (R1/R2 imbalance) is OBSOLETE but NOT resolved
R2 has been deleted, so the R1-vs-R2 imbalance is gone. **However** the R1
estimators remain mutually inconsistent: four live sites, of which the two
POSITIONAL ones ignore `ladd_r1_normalize_tokens` entirely and use a plain
`step % n` cadence with no debt latch. Applying fix 2 therefore does nothing on
the positional path, and a match-fallback step silently routes there — where
`normalize_tokens=true` is ignored and grad_sq jumps by ~T^2 at gamma=1.0.
