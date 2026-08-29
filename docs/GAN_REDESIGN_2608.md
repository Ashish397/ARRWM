# HANDOVER — the texture/GAN day, 2026-08-26

Written for the next agent. Everything below was re-verified against the repo
and the run logs before it was written; several claims carried in the session
transcript did **not** survive that check and are marked so. Where something is
not established it says so and is not counted as a result.

This file is the **arc and the operational record**. It deliberately does not
restate the measurement tables that already exist:

| for | read |
|---|---|
| the campaign brief, hazards, statistical rules | `docs/ONBOARDING_PIXDIRECT.md` |
| the 200-step pixdirect results (lattice, dose, falsified hypotheses) | `analysis/gan_tuning/PIXDIRECT_200_RESULTS.md` |
| the frozen-feature basis ranking and the VGG GO decision | `analysis/gan_tuning/TEXTURE_BASIS_BENCHMARK.md` |
| **whether the U-test's AUC 0.927 is rollout drift, and the time-window confound it exposes** | `analysis/gan_tuning/DRIFT_SEPARABILITY.md` |
| **the exact deployed VGG/RN50 AUC-vs-sample curves and surrogate verdict** | `analysis/gan_tuning/AUC_SAMPLE_BUDGET_VGG_RN50.md` |
| the surrogate architecture and its root cause | `analysis/gan_tuning/PIXEL_FEATURE_SOURCE.md` §7c |
| live questions awaiting the researcher | `COMMENTS_FOR_USER.md` |

---

## 0. STATUS BOARD (as of 18:18 UTC, 2026-08-26)

| what | state |
|---|---|
| `pixvgg_online` | **RUNNING**, holder 6143212, step 51/200, `run-20260826_175306-h7umfft2` |
| `pixrn50_online` | **RUNNING**, holder 6143213, step 51/200, `run-20260826_175425-332en4pd` |
| `pixdirect_{frozen,online,gtclean,onlinelr,strong}` | COMPLETE, 191/200 each |
| scalar-potential surrogate | **FALSIFIED at substeps=24.** Direct-vector calibration is tracked in §4.3 |
| D-wins tripwire | shipped, `model/gan_balance.py`, **defaults ON**, observability only |
| capacity knobs (`crop_y_*_frac`, `decode_split`, clamp counters) | built, **all default-OFF** |
| holders 6144621 / 6144623 | another session's `carncommit` pair, relaunched 18:0x |

Holders 6143212/13 have ~2h25 left; a 200-step arm has been taking ~1h25, so
both should finish inside the allocation.

---

## 1. THE ARC, IN THE ORDER IT HAPPENED

1. **Start:** a DINOv2-ViT-S/14-on-decoded-pixels discriminator (`pixdirect`)
   replacing the Wan patch-feature taps, to kill a 16 px lattice being printed
   into decoded video. Stride 14 is coprime with the 16 px Wan patch.
2. **It worked, on the lattice.** 15-42 % amplitude reduction on six
   `radial_spec` fold channels, against an A/A replicate floor 6-34x smaller,
   and it is **not** the pre-registered degenerate smoothing win (broadband HF
   sits *above* the dataset, not below). First 200-step texture effect in this
   campaign to clear its own noise floor. Tables in `PIXDIRECT_200_RESULTS.md`
   §1 / §1.1 — not repeated here.
3. **It did nothing to style.** Every radial band within ±0.18 log2 of the
   parent arm. The researcher's stated goal is style shift, so this is a
   mechanism win, not a goal win (§1.3 there).
4. **Why: the DINO disc ran near chance.** §2 below.
5. **The U-test** — a frozen-feature texture-sensitivity benchmark on real
   footage — explained *why the basis*, not the head, was the problem, and
   produced the VGG/ResNet50 GO. §3.
6. **The scalar-potential surrogate was falsified**, not merely broken. §4.
7. **The D-wins collapse** was found hiding inside a healthy-looking median.
   This is the most operationally important finding of the day. §5.
8. **Memory turned out never to have been the binding constraint.** §6.

---

## 2. THE ARM TABLE — the DINO disc runs near chance, and encoder lr is the lever

`d_loss` from the `[ActionForcing] step=` lines (printed every 10 steps),
**medians over steps ≥ 50, n = 15 per arm**, `[min, max]` in brackets.
`ln2 = 0.6931` is chance; the documented healthy band is 0.25-0.55.
Recomputed by this handover directly from the logs, not copied.

| arm | encoder | enc lr | `d_loss` median [min,max] | gap `d_real−d_fake` |
|---|---|---|---|---|
| `pixdirect_frozen` (PIXA) | frozen | — | 0.6490 [0.594, 0.679] | 0.111 |
| `pixdirect_frozen` (PIXB, A/A) | frozen | — | 0.6494 [0.589, 0.678] | 0.111 |
| `pixdirect_online` | trainable | 2e-6 | 0.5476 [0.248, 0.679] | 0.344 |
| `pixdirect_gtclean` | trainable | 2e-6 | 0.5873 [0.254, 0.678] | 0.253 |
| **`pixdirect_onlinelr`** | trainable | **8e-6** | **0.4560 [0.014, 0.684]** | 0.571 |
| `pixdirect_strong` | trainable | 8e-6 +K2+L3 | 0.3632 [0.002, 0.670] | 0.935 — **VOID, §5** |
| `carnpure` (Wan-feature parent) | trainable | — | 0.4343 [0.245, 0.667] | 0.636 |

**`pixdirect_onlinelr` is the best healthy arm** and is the one to compare
against. It reaches the parent's band with a pixel basis.

Two corrections to the record while you are here:

* `PIXDIRECT_200_RESULTS.md` §2 reports ONLINE at **0.5640 (n = 8)**. That was
  written mid-run. The completed run gives **0.5476 (n = 15)**. Same
  conclusion, different third digit; prefer the n=15 figure.
* `pixdirect_gtclean` has `ladd_pix_encoder_trainable_params = 22,056,192`,
  i.e. a **trainable** encoder at `lr_scale=0.1`. It is an A/A-style replicate
  of `pixdirect_online`, **not** of `pixdirect_frozen`. Its `ARM_VARIABLE` is
  `forward_noiser_apply_gt_former_FALSE`, and its completed pre-ruling run
  remains a valid single-variable comparison against `pixdirect_online`
  (0.5873 vs 0.5476, n=15 each). As a *treatment* it is retired — its flag is
  now subsumed by the trainer default (§7).

---

## 3. THE U-TEST — the pivotal experiment, and the transferable lesson

Full method, controls and tables: `analysis/gan_tuning/TEXTURE_BASIS_BENCHMARK.md`.
The three things a successor must carry forward:

**(a) DINOv2 did NOT discard the texture cue.** Ride-held-out linear probe,
`GroupKFold(5)`, 840 crops per class: **AUC 0.927 [0.863, 0.948]**. Positive
control (GT vs blur σ2.5) **1.000**; negative control (GT ride A vs ride B)
**0.01-0.53**; A/A null 0.499. So "the features cannot see it" is falsified,
and so is "the head was simply badly optimised" — both structural obstacles
are properties of the *basis*.

**(b) The ranking, by `TS_min`** = min two-sided texture sensitivity (blur↑↓,
grain↑↓) divided by the crop-translation nuisance. Pass mark 1.0:

```
rn50_layer1 5.40 | vgg_relu2_2 2.96 | dino_blk2 1.33 | rn50_layer2 1.19
vgg_relu1_2 0.82 | vgg_all 0.63 | dino_all 0.39 (WHAT WE SHIPPED) |
vgg_relu3_3 0.37 | handstat 0.17 | dino_blk11 0.13 | dino_blk8 0.12
```

An 8 px crop shift moves LADD's fused 4-tap DINO representation **more** than
any texture perturbation does — and the disc redraws its crop origin every
step and jitters phase by up to 8 px. The pixel gradient is phase-locked to
match: shift-consistency 0.35-0.42 for every DINO block against **0.884** for
`vgg_relu1_2` and 0.652 for `vgg_relu2_2`.

**(c) FUSION DILUTION is the lesson that generalises.** Adding a deeper,
phase-sensitive tap **destroys** the statistic: `vgg_relu2_2` alone 2.96 →
`vgg_all` 0.63; `dino_blk2` alone 1.33 → `dino_all` 0.39. Whatever basis comes
next, do not concatenate a deep tap onto a shallow one and assume monotone
improvement. This is now the design rule the VGG arm is built on.

---

## 4. THE SCALAR-POTENTIAL SURROGATE IS FALSIFIED

The earlier diagnosis (`PIXEL_FEATURE_SOURCE.md` §7c "RESOLVED") said the
distilled latent critic simply got **14** gradient steps — the generator
cadence `dfake_gen_update_ratio=5` above `gan_disc_start_step=20` — which
exactly predicted the observed `surrogate_n_teacher_refresh = 7`. That
arithmetic was right, the fix was built (`surrogate_distill_substeps`), and it
**ran**. It did not work.

Measured on `run-20260826_153844-vxpvdj0n` (`pixdino_frozen`, holder 6140644),
read from `wandb-summary.json`:

| key | value | reading |
|---|---|---|
| `surrogate_distill_substeps` | 24 | configured |
| `surrogate_substeps_achieved` | **24** | the fix fired |
| `surrogate_n_distill_substeps` | **840** | vs 14 before = **60x** |
| `surrogate_n_teacher_refresh` | 18 | teacher cadence unchanged |
| `critic_disc_corr` | **0.803** | the **value** channel learned |
| `critic_value_loss` | 0.008 | ditto |
| `critic_grad_loss` | **0.984** | the **gradient** channel did not |
| `surrogate_check_cos_sim` | **0.0075** | was 0.0097. Held-out; target ~0.8 |
| `surrogate_check_rel_err` | **1.0001** | was 0.99997. Target → 0 |
| `surrogate_param_grad_norm_unweighted` | **0** | over 825/825 params |
| `surrogate_param_n_reached_base` | 825 | DMD reaches all of them |
| `surrogate_param_n_reached_term` | 825 | non-`None`, total norm exactly 0 |

A 60x increase in critic gradient steps moved the held-out gradient cosine by
**−0.0022**. The CPU trajectory table in §7c predicted 0.80 at 120 steps on a
realistic teacher and 0.71 at 500 on a white-noise one; production took 840 and
got 0.0075. **The step-budget hypothesis is counter-proven.** No `pix_gan_weight`
is computable — any finite weight multiplies a zero field.

This ruling is about the learned scalar potential and its Sobolev
double-backward, not every possible synthetic-gradient construction. Section
4.3 records the matched VGG confirmation and the explicit first-order vector
alternative; neither may guide the generator before passing the same held-out
cosine gate.

### 4.1 The one sub-question that was OPEN and is now CLOSED

The transcript worried that `surrogate_param_grad_norm_unweighted` might be
**tautologically** zero at `PIXW=0.0` — i.e. that the probe was being fed the
*weighted* tensor despite its name. **It is not.** Traced through code:

* `trainer/causal_action_forcing_train.py:10136` computes
  `raw_sur, sur_logs = generator_surrogate_loss(critic, fake_lat, weight=1.0)`
  and separately forms `weighted_sur = raw_sur * weight if weight != 0 else None`.
* `_pix_g_raw` is bound to `raw_sur` (assigned at :19817), and it is `_pix_g_raw`
  that is handed to `_pix_surrogate_grad_telemetry` (:20120), which forwards the
  same tensor to `_param_grad_ratio` (:9317, :9345).

So the probe differentiates the **pre-weight** loss and the key name is honest.
The tautology hypothesis is **falsified by code inspection**, not by a run.

### 4.2 What is still genuinely unexplained

The **two probe sites disagree on the same tensor**. On the run above:

* chunk site: `surrogate_grad_norm_unweighted = 0.00113` (**non-zero**),
  `surrogate_grad_probe_site_is_chunk = 1`;
* parameter site: `surrogate_param_grad_norm_unweighted = 0` **exactly**, with
  825/825 gradients non-`None`.

Both are `d(raw_sur)/d(·)`. Non-`None`-but-all-zero is the documented
"graph-reachable, structurally receives nothing" signature. The leading
untested candidate is numeric flush-to-zero: 1.1e-3 at the chunk, propagated
back through the whole DiT in bf16, may underflow before the float32
sum-of-squares is taken. That is a hypothesis, **not a finding** — nobody has
instrumented it. It does not change the verdict in §4, because the held-out
`check_cos_sim` is an independent measurement of the same dead field.

### 4.3 VGG confirmation and direct-gradient replacement (2026-08-27)

Weight-zero VGG-teacher calibration `9u5rb6hr` completed 90/90 and closes the
remaining teacher-basis loophole. Its connection was real: ten teacher
logits/sample, four fresh targets per class, 24/24 fitting substeps, 312
student optimizer steps, seven teacher refreshes, a nonempty real pool, no
direct generator decoder graph, and zero applied generator weight. The field
still failed on every audit:

| key | final reading |
|---|---:|
| training-target gradient cosine | 0.1436 |
| normalized training gradient loss | 0.9805 |
| held-out gradient cosine | **0.0643** |
| held-out relative error | **0.9993** |
| held-out magnitude ratio | 0.1166 |
| held-out audits | 4 |

This is not merely a held-out generalization failure: the scalar student did
not fit its cached training field either. The earlier CPU success used a much
simpler local-convolution teacher and does not transfer to the trained
VAE+VGG discriminator. The scalar arm remains permanently blocked.

The replacement is named
`vgg_surrogate_directgrad_teacher_d5x_targets4perclass_fit24`. It predicts the
teacher's latent gradient vector directly, fits a unit-RMS direction field
with a first-order loss, tracks a synchronized EMA of teacher gradient RMS,
and gives the generator a detached linear synthetic-gradient loss. This
removes both restrictions above: no learned conservative scalar potential and
no double backward through the student. It is default-off under
`surrogate_gradient_mode=direct`, retains the same expensive teacher evidence
and refresh cadence, and stays at weight zero until held-out cosine >= 0.50.

The memory-safe serialized run `zffrfznl` completed 90/90. Target
microbatching fixed the prior step-40 OOM without changing the eight logical
targets per refresh (four real plus four fake). The four held-out audits were
0.1606, 0.2081, 0.1490 and **0.3242** cosine; final magnitude ratio was 1.2068
and relative error 1.2938. This is materially better than the scalar student,
but still below the 0.50 activation gate. The checker reports 36 passes and
two scientific failures: cosine below gate and exactly zero unweighted
generator-parameter gradient, despite a nonzero 0.2091 field at the flash
latent and 825/825 parameters being graph-reachable. Weight remains zero.

The controlled retry is explicitly named
`vgg_surrogate_directgrad_currentteacher_8targets_refresh_fit24`. The live
discriminator changes while the old capacity-eight cache uniformly replays
fields labelled by older discriminator versions. A holder CPU moving-teacher
control measured held-out cosine 0.5485 / 0.5174 / 0.4565 / 0.4396 for cache
capacities 1 / 2 / 4 / 8 respectively. The retry therefore keeps only the
latest real and fake target batches and changes nothing else about teacher
queries, target count, fitting steps, or generator weight. A matched-norm
alternating-sign tangent control is measured at the same flash tensor: if it
reaches generator parameters while the learned field does not, the failure is
Jacobian-nullspace cancellation; if both are zero, the flash graph is broken.

The 90-step results settle both questions. Current-teacher/DMD-source W&B
`ttxdfxu8` produced held-out cosines **0.154, 0.268, 0.342, 0.484**; the
otherwise identical aligned-flash teacher-source control `wucn3u1w` produced
**0.147, 0.313, 0.318, 0.440**. Latest-only targets materially improve the old
0.324 final result, but neither arm passes 0.50 and aligning the teacher's
trained/query fake distribution is not the missing factor. Each post-run
checker records 36 passes and three scientific failures (cosine, calibration
ratio, and tangent liveness). At the parameter
probe, both learned and same-norm tangent fields reached 825/825 parameters
yet both norms were exactly zero while the base gradient norm was 6.04.

That signature is now explained. The measured step rolled nine new frames in
three three-frame groups. The pipeline deliberately runs the trailing group
under `no_grad`, while `_pix_select_fake_latents` historically selected the
final two frames. Earlier live groups make the assembled CopySlices buffer
`requires_grad=True`, so parameter traversal reports every tensor reachable,
but an upstream gradient supported only on the detached tail is identically
zero. The issue was frame selection, not the learned vector field.

The corrected arm
`vgg_surrogate_directgrad_currentteacher_8targets_refresh_fit24_liveflash`
publishes a producer-authored `flash_dmd_gan_grad_mask` and selects the latest
contiguous live run. Historical behaviour remains default-compatible behind
`pix_flash_grad_select_enabled=false`; only the corrected calibration sets it
true. CPU controls prove mask `[live x6, detached x3]`, nonzero gradient from
the live prefix, zero from the tail, and loud failures for missing/dead masks.
Weight-zero W&B `79re58zl` completed 90/90 cleanly. The fix is decisive at the
parameter seam: the learned term reaches 825/825 parameters with unweighted
norm **10.1267** and ratio **0.6890** against the base gradient; the matched
tangent control is also live at norm **34.4474**. Its final held-out cosine is
**0.4647**, however, narrowly below the 0.50 activation gate (magnitude ratio
1.1988, relative error 1.1502). The post-run checker is **42 PASS / 1 FAIL**,
with cosine as the only failure. Routing is fixed; activation remains blocked.

The next controlled calibration is
`vgg_surrogate_directgrad_currentteacher_16targets_refresh_fit24_liveflash`,
W&B `u56r8lps` on holder 6150866. It changes only
`pix_crops_per_step=4 -> 8`, hence eight real plus eight fake current teacher
targets per refresh instead of four plus four. Cache capacity one, 24 fitting
substeps, target microbatch one, DMD teacher-fake source, live-frame selector,
and `pix_gan_weight=0` are unchanged. This tests independent field evidence,
not additional replay or an activated generator loss.

The 16-target run completed 90/90 without an OOM, but it did not improve the
field. Its final same-batch audit cosine was **0.4114**, below the matched
eight-target live-Flash arm's 0.4647; its parameter route remained healthy
with unweighted ratio **0.4804**. Review then found that the historical
"held-out" audit is resubstitution: the trainer fits `crops_f` and immediately
audits those same fake crops. The eight fake targets per rank are also
overlapping spatial crops from one two-frame slab, not independent rides.
Consequently neither a crossing nor a miss of the old 0.50 gate is a valid
generalisation ruling, and doubling this correlated target batch is retired.

Researcher direction then opened an explicitly active dynamics experiment,
`vgg_surrogate_directgrad_currentteacher_8targets_t0rungs_active`. It removes
the extra Flash-DMD forward (`flash_dmd_enabled=false`,
`flash_dmd_gan_t=0`), publishes/consumes the graph-live final ladder x0 via
`pix_finish_grad_enabled=true`, reserves that final rung with
`exit_exclude_last_rung=true`, and pins all former Flash-facing sources to
`ladder_endpoint` (`gen_aux_losses_x0_source`,
`forward_noiser_train_source`, and `forward_noiser_rollout2_source`). The LADD
teacher retains its DMD-scored clean-x0 band. It returns to the simpler four
real plus four fake targets, uses `pix_gan_weight=0.15`, and records its
below-historical-gate activation as `active_researchoverride` rather than
claiming a calibration pass. The authoritative restart is W&B `v53mvn1t`,
Slurm child `6150866.5` on preserved holder 6150866, for 200 steps with videos
every 15 steps. Incomplete attempts `7byixmlm` and `yt951mt0` are void: review
first found the FN source enums on legacy fallback, then observed that random
last-rung exits could leave no later rung for a graph-live x0. Both child tasks
were replaced before GAN engagement without cancelling the holder.

### 4.4 Detached rendered-state conditioning (2026-08-27)

The next arm tests whether the direct vector-field student is missing the
rendered local evidence needed to associate a teacher gradient with a spatial
latent location. The premise is sound but not literally one-to-one: Wan's VAE
maps one latent frame to four decoded frames and one spatial latent cell to an
approximately 8x8 pixel footprint, while convolutional receptive fields smear
influence across adjacent cells. The implementation therefore aligns groups
using the decoder's measured geometry rather than treating pixels and latent
coordinates as identical tokens.

The arm is named
`vgg_surrogate_directgrad_currentteacher_8targets_t0rungs_rgbmaxmin_active`.
For each real/fake target crop and each generator-consumption slab it:

1. detaches the latent and decodes it with `seed_first=true` under
   `torch.no_grad()`;
2. groups the exact four decoded frames belonging to each latent frame;
3. adaptively max-pools each exact spatial footprint back to the latent
   height/width;
4. retains RGB maxima **and RGB minima** (implemented as `-max(-rgb)`), six
   channels total, because plain max pooling systematically hides dark
   texture;
5. adds those channels through a separate zero-initialized convolutional side
   projection before the existing residual trunk.

The side projection is constructed after every baseline random parameter.
Consequently a fixed seed gives the same initial latent/coordinate trunk with
the gate on or off, and its zero initialization makes the first conditioned
forward exactly equal to the current unconditioned arm. It can then learn a
dependency on rendered extrema through ordinary first-order distillation.
The generator loss remains the same linear synthetic-gradient loss in the
original graph-bearing latent. The VAE output guide is detached twice (at the
decode helper and pooling helper), and the predictor fails loudly if handed a
condition carrying autograd history; no generator gradient can pass through
the pixels or decoder.

Teacher-target replay stores the pixel condition with the exact latent/value/
gradient tuple it labels. No-grad VAE conditioning decodes occur only on
teacher refreshes, not on cache replay; generator consumption decodes one
batch slice at a time (`surrogate_pixel_condition_decode_batch=1`). Runtime
proof keys include condition active/detached flags, six-channel count, temporal
and spatial scale (expected 4/8/8), decode-call count, sample count and RMS.
The projection's weight/gradient norms and the fitted-field RMS change under a
zero-condition ablation prove the branch learned and functionally influences
the vector field; the ablation loss delta says whether that influence helps on
the labelled batch. Non-integral temporal geometry is a hard error rather than
silently resized.

This is a controlled extension of `v53mvn1t`: the no-Flash t=0 rung sources,
eight logical current-teacher targets, 24 fits, refresh cadence two, cache one,
weight 0.15 and all other dynamics are unchanged. The 75-step GPU smoke is
durably queued behind the existing 200-step task on preserved holder 6150866;
it cannot start until child `6150866.5` exits, and neither the child nor holder
was stopped. It will produce native rollout videos every 15 steps. CPU review:
102/102 surrogate/model/trainer tests, 109/109 consumption/finish/CARN/launcher
tests, and 18/18 focused launcher tests pass (the launcher suite is included in
the 109 count; counts are reported separately to identify the arm review).

---

## 5. THE D-WINS COLLAPSE — the most important operational finding of the day

`pixdirect_strong` raised the disc's crop coverage on top of encoder lr 0.4:
`ladd_pixel_crops_per_row` 1→2 and `ladd_pixel_lat_frames` 2→3.

**The coverage change is counter-proven to have fired, exactly as predicted**
(wandb summaries, both runs complete at `_step=196`):

| counter | `pixdirect_onlinelr` | `pixdirect_strong` |
|---|---|---|
| `ladd_pix_images` | 3558 | **7116** (exactly 2x) |
| `ladd_pix_disc_forwards` | 729 | 729 (unchanged) |
| images / disc forward | **4.88** | **9.76** |
| `ladd_pix_grid_h` × `_w` | 13 × 17 | 13 × 17 (unchanged) |
| `ladd_pix_wan_projector_calls` | 0 | 0 |

**And the arm collapsed.** Its full logged `d_loss` trajectory:

```
strong    31 .668  41 .358  51 .631  61 .414  71 .432  81 .670  91 .664
         101 .363 111 .043 121 .048 131 .413 141 .491 151 .327
         161 .0104 171 .0039 181 .0024 191 .0102   <- 4 consecutive, still falling
onlinelr  31 .665  41 .574  51 .630  61 .355  71 .468  81 .159  91 .597
         101 .684 111 .135 121 .014 131 .624 141 .270 151 .469
         161 .530 171 .394 181 .278 191 .456     <- dips twice, RECOVERS
```

`d_fake` on `strong` is monotone over the last four logged steps:
−2.894 → −3.454 → −3.753 → −4.056, with a final `d_real − d_fake` gap of
**5.17**. The discriminator has separated real from fake essentially perfectly;
the generator receives nothing.

**Its median over steps ≥ 50 reads 0.3632, inside the healthy band, and would
have been reported as the best arm of the day.** Medians past step 50 are a
standing rule in `ONBOARDING_PIXDIRECT.md` §7, and here the rule actively
concealed the failure. What separates the two arms is a **late-window**
statistic: medians over steps ≥ 111 (n = 9 each) are **0.0431** (strong) vs
**0.3937** (onlinelr).

### 5.1 The rule to carry

**Consecutive `d_loss ≤ 0.135` beats any summary statistic.** k=3, not 5:
`onlinelr`'s recovered transient is exactly 2 consecutive observations
(0.1346 at 111, 0.0139 at 121, then 0.6243 at 131), so 3 is the smallest k
that separates the measured recovery from the measured collapse. This belongs
in the silent-failure taxonomy: an averaged statistic that reads healthy over
a run that ended dead.

Shipped as `model/gan_balance.py::DWinsTripwire`, wired at
`trainer/causal_action_forcing_train.py:4114-4142`, config
`gan_dwins_tripwire_enabled: true` / `gan_dwins_floor: 0.135` /
`gan_dwins_k: 3` (`configs/action_forcing_phase3_dmd.yaml:80-82`). It is
observability only — it touches no tensor, optimiser or RNG — it latches, and
it **rejects exactly-0.0** rows because that is the pre-`gan_disc_start_step`
not-run sentinel (steps 11 and 21 on every arm) and counting it would fire on
every run at step 21. Regression tests replay both arms verbatim:
`testing/test_gan_dwins_tripwire.py`.

### 5.2 What this does NOT establish

* **Attribution to coverage alone is not established.** `strong` moved *two*
  variables against `onlinelr` (crops_per_row AND lat_frames) — the sbatch
  labels itself `ARM_VARIABLE=MULTI(enc_lr0.4+crops2+latf3)_NOT_AN_ABLATION`.
  A single-variable coverage arm has not been run.
* **The exchange rate is INCONCLUSIVE.** The D's information per step goes as
  (images per disc forward) × `gan_updates_per_step` (=5 on these arms,
  last-wins over the config's 1): 4.88 × 5 = 24.4 healthy, 9.76 × 5 = 48.8
  collapsed. **n = 2 arms, one of each.** What is measured is that 2x at
  `gan_updates_per_step=5` collapsed and 1x did not. Nothing about where the
  boundary sits.

---

## 6. MEMORY WAS NEVER THE BINDING CONSTRAINT

Re-derived from live rank-0 `[ROLL] step_peak` over the completed 200-step
arms (n = 40 readings per arm), against 95 GB per card:

| geometry | arms | max `step_peak` | median |
|---|---|---|---|
| K=1, L=2 | frozen / online / gtclean / onlinelr | **56.90 - 57.53 GB** | 50.8 - 54.0 |
| K=2, L=3 | strong | **63.25 GB** | 59.7 |
| K=1, L=2 (VGG) | pixvgg_online (partial) | 56.71 | 50.0 |
| K=1, L=2 (RN50) | pixrn50_online (partial) | 53.06 | 49.9 |

`COMMENTS_FOR_USER.md` projects **~76 GB of 95 on the worst rank** even at
K=4/L=3/kf=8. **The constraint is the G/D balance, not VRAM.**

### 6.1 Correction to `ONBOARDING_PIXDIRECT.md` §2 — the "42.38 GB" figure

It is **not** a rank artefact and it is **not** the step peak. It is what
rank 0's `torch.cuda.max_memory_allocated` reads *after the `[ROLL]` block has
reset the peak stat*, so on rank 0 `peak_gb == alloc_gb`. From the `[MEMRANK]`
audit (`logs/pixdino_smoke_pixdirect_frozen_h6135660_20260826_134231.log`),
step 25: rank 0 `alloc_gb=42.38 peak_gb=42.38`, ranks 1-7 `alloc_gb=42.33
peak_gb=53.35`. Read `[ROLL] step_peak`, and read it knowing the reset.

### 6.2 Two numbers from the transcript I could NOT reproduce

* **"Rank correction is +3.56 GB."** Not found anywhere in the repo, and I
  cannot derive it. What the `[MEMRANK]` audit actually shows, on the one step
  where rank 0's peak had *not* been reset (step 15, `reset=1`, rank 0
  `peak_gb=49.79` vs max-rank `50.09`), is a rank spread of **+0.30 GB**. On
  the six steps where rank 0's peak *had* been reset the raw rank-0-to-max
  delta is +11.0 to +15.0 GB, and that number measures the reset, not the
  ranks. **Treat the true rank correction as ~+0.3 GB (n=1 clean step, 8
  ranks) and the "+3.56" and "+11" figures as both unsupported.**
* **"~78 GB worst-rank at K=4/L=3/kf=8."** `COMMENTS_FOR_USER.md` says **~76**.
  Neither is measured — both are projections. Use the written 76 and re-derive
  before betting on it.

---

## 7. RULINGS APPLIED TODAY (researcher's, binding)

1. **`gt_vs_fake` positives are NEVER noised.** This is now the **trainer
   default**, not a flag you turn on. Escape hatch
   `forward_noiser_allow_gt_vs_fake` (default `False`,
   `model/dmd_action_forcing.py:2385`, read at
   `trainer/causal_action_forcing_train.py:13915-13916`) exists only for a
   deliberate ablation. `gt_transition` keeps the CARN noiser, byte-identical,
   verified by an independent reviewer with a mutation control.
   **TWO violation sites** were found and fixed — the *decoupled* apply
   (:13932) and the *matched-pool* apply (:15063). The second was missed on
   the first pass; if you touch this path, check both.
   Proof of fire, live on the VGG arm right now:
   `[FN-GTVF-CLEAN] BYPASS site=decoupled pair_mode=gt_vs_fake cpp=1 n=3`.
2. **`_add_disc_noise` is OUT OF SCOPE** — symmetric diffusion noise, applied
   to real and fake alike, and already inert on these arms via
   `ladd_disc_force_clean=true`.
3. **`pixdirect_gtclean` is RETIRED as a treatment** (subsumed by the default),
   kept as an A/A replicate of `pixdirect_online` — see §2.
4. **DINO / ViT is retired in favour of VGG** (and ResNet50 as the second arm).
5. **`gan_loss_weight` stays 1.0, always.** It scales only the G side; cutting
   it converts an over-driven-G problem into exactly the D-wins failure in §5.
   Move `gan_lr` or `gan_updates_per_step`, which scale both sides.

---

## 8. WHAT WAS BUILT, AND WHAT IS RUNNING

### 8.1 The two replacement arms — LAUNCHED (verify before assuming)

Both were staged as "ready but unlaunched"; **both are in fact RUNNING** as of
17:53/17:54 on holders 6143212 / 6143213. Boot lines confirm the specification:

**`sbatch/pixvgg_online.sbatch`** — `vgg16:relu1_2,relu2_2`, `taps=[0,1]`,
`channels=[64,128]`, orderless `[mu, sigma, triu(Cov)]` normalised `1/HW`,
`proj_dim=32`, `hidden=128`, `stat_dims=[656,784]` (`64+64+528` and
`128+128+528`, total 1440), `encoder_params=0.26M` all trainable at
`lr_scale=0.1` → `encoder_lr=2e-06`, `pretrained=True`, `gan_loss_weight=1.0`,
`telemetry_every=1`.

**`sbatch/pixrn50_online.sbatch`** — `resnet50:layer1`, `taps=[0]`,
`channels=[256]`, same orderless readout, `stat_dim=1040` (`256+256+528`),
`encoder_params=0.23M`, same lr and weights.

Note `relu3_3` is deliberately absent (§3c) and both use `crops_per_row=1
lat_frames=2 frames_per_crop=2` — **no coverage change**, so they are clean
single-variable basis swaps against `pixdirect_online`, not against
`pixdirect_onlinelr` (which differs in encoder lr).

### 8.2 Also shipped

* `model/gan_balance.py` — the D-wins tripwire (§5.1). Defaults ON.
* Sample-budget observability, all in `model/ladd_disc.py` /
  `_ladd_pixel_logs`: `ladd_pix_lat_frames_cfg` / `_used` / `_avail` /
  `_clamped` (`lat_frames` was previously clamped by `L = min(lat_frames,
  F_lat)` with **no counter at all**), the same for `frames_per_crop` /
  `crops_per_row` / `crop_rows` / `crop_cols`, plus
  `r3gan_disc_inner_updates_total` (`r3gan_disc_updates_total` counts D
  *phases*, not the five inner iterations, so the one sanctioned counterweight
  could not previously be proven to have taken).
* Capacity knobs, **all default-OFF and byte-identical when off**:
  `ladd_pixel_crop_y_lo_frac` / `_hi_frac` (ships `0.0`/`1.0` = historical
  full-height draw), `ladd_pixel_decode_split` (ships `0`).
  `ladd_pixel_decode_batch` was found **INERT** — written into
  `disc.pixel_cfg` and echoed at boot but read by nothing; the echo now says
  `(INERT)`.

---

## 9. PROOF-OF-FIRE DISCIPLINE — the counters and their predicted values

Prove a flag fired from a counter, never from the patch. All `train/ladd_pix_*`
keys are **wandb-only** — read `wandb/wandb/<run>/files/wandb-summary.json`,
where they appear under the `gen/train/` prefix, not the console log. On the
new pooled arms:

| counter | must read | why |
|---|---|---|
| `ladd_pix_logits_per_sample` | **2** | = `crops_per_row × frames_per_crop`. **1768** is the DINOv2 token count (13×17 grid × 4 taps × 2 frames) and **5280** is the rn50 dense map ((192−16)/4 × (256−16)/4 × 2). Either means a **dense** head is back and **the arm is VOID** |
| `ladd_pix_dense_head_calls` | **0** | negative control for the same failure |
| `ladd_pix_pooled_readout` / `_pooled_calls` | 1.0 / >0 | built **and** ran; 1.0 with 0 calls = built-but-inert, the failure this campaign keeps hitting |
| `ladd_pix_vgg_pretrained` | **1.0** | 0.0 = random init = **null experiment** |
| `ladd_pix_n_taps` | **2** (VGG) / **1** (rn50) | fusion dilution guard (§3c) |
| `ladd_pix_wan_projector_calls` | **0** | never falls back to the Wan taps — the central claim |
| `ladd_pix_decode_grad` | **NON-ZERO** | see below |
| `ladd_pix_lat_frames_clamped` | 0 | a silent clamp would fake a coverage change |
| `gan_dwins_tripped` / `_max_streak` | 0 / small | §5. On a run that did not trip, read `_max_streak` — it says how close it came |

**`ladd_pix_decode_grad` polarity is INVERTED between the two routes.** On
`pixdirect` a **non-zero** value proves the direct decode-gradient route is
live (the completed arms read **204**). The "must be 0" expectation belongs to
the **SURROGATE** arm, where the generator route is the latent critic. Reading
204 as a failure is a false alarm. Already noted in
`ONBOARDING_PIXDIRECT.md` §2 and `PIXDIRECT_200_RESULTS.md` §6.1.

**Key-name trap.** `fn_gtvf_noise_skipped` / `_applied` carry a `_gt`
**suffix** in wandb. Every enabled LADD pair mode appends a non-empty suffix to
every key it publishes — `_gt` / `_adj` / `_gtxn`, defined in `enabled_modes`
at `trainer/causal_action_forcing_train.py:11982-11987` and applied at
**:12066** (`logs[k + suffix] = v`). The bare name returns **ABSENT**. There is
a resolver, `gan_log_lookup` (:399), which exists because the console GAN
health line silently omitted every field for a long time for exactly this
reason — a run once held `r3gan_d_real_gtxn = 4.529` (a runaway disc) while the
console showed nothing.

**A known false positive, do not chase it.** The `[override-guard]` ERROR names
`ladd_pixel_*` keys (8 of them on the VGG/rn50 arms) as unread. They **are**
read, via the `g("key", default)` closure at
`model/ladd_pixel_features.py:749-755`, which the guard's `getattr(config, …)`
regex does not match. Proven fired from the boot echoes, which print the
configured values back (`variant=vgg16:relu1_2,relu2_2`, `pretrained=True`,
`proj_dim=32` recoverable from `stat_dims`). `PIXDIRECT_200_RESULTS.md` §7c.

**A second alarm that cries wolf.** `[LADD-PIXFEAT] encoder param group: …
(0 encoder params => the encoder is FROZEN, which contradicts
ladd_pixel_encoder_trainable)` — the parenthetical is part of the **format
string**, not a branch, so it prints on healthy trainable arms
(`n_encoder=174`, 22.06M trainable). Deliberately not fixed yet: the arms hash
their trainer sources into `[ARRWM-PROVENANCE] srcmd5` and editing it mid-
campaign would break hash continuity across a running comparison. Full write-up
at the top of `COMMENTS_FOR_USER.md`.

---

## 10. OPERATIONAL LESSONS — these cost real work today

1. **The holder loop runs ONE command at a time.** Staging
   `logs/.holder_cmd_<jobid>.sh` onto a holder that already has a live job
   **destroys BOTH jobs** via `RendezvousConnectionError`. Measured:
   `logs/holdersmoke_pixrn50_online_h6144623.log:916` carries the traceback and
   the run logged **0** training steps; the arm it landed on (another session's
   matched baseline, `carncommit_off`) went down with it while its treatment
   partner on the other holder survived — leaving a paired smoke that was no
   longer an experiment. The surviving half looked perfectly healthy.
   A guard now lives in `sbatch/run_smoke_on_holder.sh`: it asks slurm what
   *steps* are live (it cannot test for `.holder_running_<jobid>.sh` — by the
   time the script runs, the loop has already created that file for **us**) and
   exits **3**. Override is `HOLDER_FORCE=1`; do not set it habitually.
2. **That guard shipped BROKEN and blocked every launch on every clean
   holder.** Under `set -euo pipefail`, `grep -v` matching nothing exits 1,
   `pipefail` propagates it as the pipeline status even though `wc -l`
   succeeded, and `set -e` aborts the script — silently, with no output.
   Verified in isolation: without `|| true` the pipeline exits 1; with it,
   exit 0 and `_LIVE=0`. **The `|| true` is load-bearing.**
   *The lesson is bigger than the bug:* **a guard nobody has seen PASS on the
   clean case is not evidence that it passes.** Exercise the permit path, not
   just the refuse path.
3. **A fresh log mtime is NOT evidence of progress** — it can be a traceback
   being written. Use `grep -c "ActionForcing] step="`. A "no harm done" report
   on the killed baseline was made purely from an mtime, and was wrong.
4. **`run_smoke_on_holder.sh` TRUNCATES `logs/holdersmoke_<SMOKE>_h<HOLDER>.log`.**
   Re-running the same SMOKE tag on the same HOLDER destroys the previous run's
   console log. wandb runs and `logs/dmd10k_<arm>/` survive; the console log
   does not. (Someone preserved one by hand today:
   `logs/carncommit_TREATMENT_h6144621.log.keep`.)
5. **Reusing a `PORTOFF` reuses the rendezvous id** (`rdzv_id = HOLDER+PORTOFF`),
   and a stale rendezvous from a dead run poisons the retry. Always use a fresh
   `PORTOFF`.
6. **Orphaned processes from dead launches hold GPU VRAM.** A later launch on
   those nodes died CUDA OOM with 6.39 GiB free of 95 while itself holding only
   33.56 GiB.
7. **Sockets churn on restart** — address peer sessions by ListAgents **name**,
   not socket path.
8. **FOUR OR MORE SESSIONS SHARE ONE UNCOMMITTED WORKING TREE.** Nothing here
   is in HEAD (`model/ladd_pixel_features.py`, `model/gan_balance.py` and every
   new sbatch are untracked). **No stash, no checkout, no reset.** Use worktree
   isolation if you need a clean base. Corollary: a wandb summary can carry keys
   that no longer exist in the tree — `surrogate_param_*` is emitted by a helper
   whose keys are assembled from an f-string prefix, so grepping the literal key
   name finds nothing (`_param_grad_ratio`, :9112, `prefix="surrogate_param"`).

---

## 11. WHAT IS **NOT** ESTABLISHED

Listed so nobody promotes them by accident.

* **That `strong`'s collapse is attributable to crop coverage alone.** Two
  variables moved (§5.2).
* **The images × updates exchange rate.** n = 2 arms (§5.2).
* **That the VGG translation advantage matters in training.** The ~2x
  `TS_min` margin and the 0.884-vs-0.394 gradient shift-consistency are
  measured on *frozen* features with a *linear* probe and a hand-built
  objective. The argument that the advantage survives into an adversarially
  trained head is that the nuisance is common-mode in RpGAN — that is
  **argued, not measured**. The running arms are the test.
* **Why the two surrogate probe sites disagree** (§4.2). A hypothesis exists;
  no instrumentation.
* **The true per-rank memory correction** (§6.2). One clean step, one audit.
* **Any style-shift claim from pixdirect.** All radial bands within ±0.18 log2
  of the parent (`PIXDIRECT_200_RESULTS.md` §1.3, §8).
* **The realised-GAN-share comparison, the shimmer direction, the stat-anchor
  gradient share, and the original DDP reducer-index-2 crash mechanism** — all
  carried over unresolved from `PIXDIRECT_200_RESULTS.md` §8.
* **VAE round-trip, x264/HEVC, wavelet scattering and Portilla-Simoncelli** were
  never tested as bases (`TEXTURE_BASIS_BENCHMARK.md` §9). The hand-rolled
  Morlet substitute is a **failed instrument**, not evidence about scattering.

---

## 12. WHAT TO DO NEXT

**First, read the two live arms.** `pixvgg_online` (6143212,
`run-20260826_175306-h7umfft2`) and `pixrn50_online` (6143213,
`run-20260826_175425-332en4pd`). In order:

1. **Validity before results.** `ladd_pix_logits_per_sample == 2`,
   `ladd_pix_dense_head_calls == 0`, `ladd_pix_vgg_pretrained == 1.0`,
   `ladd_pix_n_taps == 2` / `1`, `ladd_pix_wan_projector_calls == 0`,
   `ladd_pix_decode_grad != 0`. Any miss and the arm is void (§9).
2. **Balance before texture.** `gan_dwins_tripped == 0`; if it tripped, the
   texture numbers past `gan_dwins_trip_step` are void regardless of the median.
3. **Then** `d_loss` median over steps ≥ 50 **and** over steps ≥ 111, both with
   n and [min, max], against `pixdirect_online` 0.5476 (the matched-lr control)
   and `carnpure` 0.4343.
4. **Then** texture, two-sided, on textured crops, via the unmodified
   `analysis/sharpness/radial_spec.py`, against GT `fold2d` P=16 dotfrac 0.13
   and mod-8 row fold A8y 1.42 — and remember an A/A floor is required before
   any of it counts (`ONBOARDING_PIXDIRECT.md` §7).

**Then, the budget agent's recommendation** (full reasoning and the table in
`COMMENTS_FOR_USER.md`, "GAN SAMPLE BUDGET"), in this order and no other:

1. **Ship `ladd_pixel_lat_frames=3` alone.** It is the only coverage increase
   that adds **ZERO samples to D** — it changes *which* latent cells receive
   gradient (8.2 % → 12.3 % of the chunk), not how many pictures D scores, so
   `ladd_pix_images / ladd_pix_disc_forwards` must **still read 4.88**. It
   closes the measured hole where the oldest latent frame of every chunk is
   never adversarially supervised in any step, ever. Projected peak
   57.5 → ~58.9 GB. No counterweight needed.
2. **Run a balance arm at `gan_updates_per_step` 5 → 2** before any sample
   increase — `pixdirect_onlinelr` exactly, nothing else changed. It calibrates
   the counterweight before capacity is spent on it. Expect `d_loss` median
   0.45-0.60, `gan_dwins_tripped=0`, and
   `r3gan_disc_inner_updates_total / r3gan_disc_updates_total == 2.0`.
3. **Only then `crops_per_row=2`, paired with `updates=2`.** Never unpaired —
   that is §5.

**Do not**: launch surrogate arms (§4); touch `gan_loss_weight` (§7.5); noise
`gt_vs_fake` positives (§7.1); raise capacity before step 2 (§5.2).

**Still awaiting the researcher's sign-off** (in `COMMENTS_FOR_USER.md`, not
implemented): whether to bias the GAN crop sampler toward textured rows —
~39 % of G-update crops contain no road/verge at all, and it is *not* the sky
problem (median sky coverage is exactly 0.000). It is a training-recipe change,
so it does not happen without a word.

---

## 13. FILES

**New today**
- `model/gan_balance.py`, `testing/test_gan_dwins_tripwire.py`
- `sbatch/pixvgg_online.sbatch`, `sbatch/pixrn50_online.sbatch`
- `sbatch/pixdirect_onlinelr.sbatch`, `sbatch/pixdirect_strong.sbatch`,
  `sbatch/pixdirect_gtclean.sbatch`
- `testing/test_fn_gtvf_clean.py`, `testing/test_surrogate_gradient_field.py`,
  `testing/test_ladd_pixel_crop_budget.py`
- `analysis/gan_tuning/TEXTURE_BASIS_BENCHMARK.md`,
  `analysis/gan_tuning/PIXDIRECT_200_RESULTS.md`,
  `analysis/gan_tuning/DRIFT_SEPARABILITY.md`

**Logs (console; remember §10.4 — these truncate)**
- `logs/holdersmoke_pixdirect_{frozen_h6140644,frozen_h6136514,online_h6140643,gtclean_h6140643,onlinelr_h6143212,strong_h6143213}.log`
- `logs/holdersmoke_pix{vgg,rn50}_online_h614321{2,3}.log`
- `logs/pixdino_smoke_pixdirect_frozen_h6135660_20260826_134231.log` — the only
  log carrying the `[MEMRANK]` audit (§6)

**wandb (the only home of `train/ladd_pix_*` and `train/surrogate_*`)**
- pixdirect: `j3fillo5`, `wpost306`, `1r4u5ohi`, `wi5e2w16`, `zo06e980`, `ooql044t`
- surrogate substeps=24: `vxpvdj0n`
- live: `h7umfft2` (vgg), `332en4pd` (rn50)


---

# ADDENDUM (2026-08-26, evening): FIVE FALSIFICATIONS AND WHAT SURVIVES

Everything in this addendum post-dates the body above. Each entry says how the
hypothesis DIED, not merely that it did.

## A1. The discriminator has never separated real from fake on ANY pixel basis

Six configurations, d_loss median over steps>=50, n=15 each. ln2 = 0.6931 is
chance; the documented healthy band is 0.25-0.55.

| arm | basis | enc lr | match | d_loss | gap | dwins |
|---|---|---|---|---|---|---|
| pixdirect_frozen | DINOv2 | 0 | off | 0.6490 | +0.111 | - |
| pixdirect_online | DINOv2 | 0.1 | off | 0.5476 | +0.344 | - |
| pixdirect_onlinelr | DINOv2 | 0.4 | off | 0.4560 | +0.571 | - |
| pixvgg_online | VGG relu1_2+2_2 | 0.1 | off | 0.6440 | +0.103 | - |
| pixvgg_lr1 | VGG | 1.0 | off | 0.6241 | +0.144 | 0 |
| pixvgg_match | VGG | 0.1 | ON | 0.6697 | +0.047 | 0 |
| pixrn50_online | RN50 layer1 | 0.1 | off | 0.6664 | +0.054 | - |
| pixrn50_match | RN50 layer1 | 0.1 | ON | 0.6867 | +0.013 | 0 |

The five PIXEL-basis arms span 0.6241-0.6867 -- a spread of 0.063, INSIDE the
14-17% d_loss CV. Two feature bases, a 10x encoder-lr range, and matching on/off
produce no separable difference. All hard counters valid on every arm
(logits_per_sample=2, dense_head_calls=0, vgg_pretrained=1, wan_projector_calls=0),
so these are real results, not broken arms. gan_dwins_tripped=0 throughout, so
none of it is a D-wins artefact.

**When everything you vary makes no difference, the constraint is in what you
held constant.**

## A2. FALSIFIED: the feature basis
VGG relu2_2 (TS_min 2.96) and ResNet50 layer1 (TS_min 5.40) both scored far
better than DINOv2's fused 0.39 on the frozen-feature U-test. Both land in the
same dead band in training. A better basis on this setup is worth nothing.

## A3. FALSIFIED: the encoder learning rate
0.1 -> 1.0 moved d_loss 0.6440 -> 0.6241. Inside the CV.

## A4. FALSIFIED, and mildly HARMFUL: ladd_gt_vs_fake_match
The trainer's own comment (~:14253) says the matched path exists because the
positional GT chunk "produced the dead disc (d_real ~= d_fake, d_loss == log2)".
It had never been enabled on any arm in this campaign. Enabling it made things
slightly WORSE in BOTH bases -- VGG 0.6440->0.6697, RN50 0.6664->0.6867 -- with
the separation gap shrinking in both (+0.103->+0.047, +0.054->+0.013). A
consistent direction across two independent bases is unlikely to be chance.
Matching PROVEN FIRING from a site-tagged counter: `[FN-GTVF-CLEAN] BYPASS
site=matched` appears 3x on both matched arms and 0x on the control.

## A5. FALSIFIED: "the disc scores a single chunk with no accumulated drift"
Proposed to explain why the U-test (AUC 0.927 on a 7-chunk rollout) disagrees
with the disc. Killed by the trainer's own telemetry: `[42F-ALLROLL]` reads
`rolling=True student_chunks=7` on 36 of 48 observations (student_chunks=0 on
12). The rollout the disc's fake comes from carries seven student chunks of
accumulated context -- the same depth the U-test measured.

## A6. FALSIFIED: "the fake is too close to GT to separate"
Prediction: if so, steps where the student drifted FURTHER from GT should be
EASIER to separate => NEGATIVE corr(d_loss, gt_dist). Measured, n=15 each:
pixvgg_lr1 **+0.454**, pixvgg_match **+0.386**, pixrn50_match **+0.355**.
The sign is POSITIVE in all three arms -- more drift goes with WORSE separation.
At n=15 an r~0.4 is individually weak (p~0.1); the consistent sign across three
independent arms is what makes it actionable.

## A7. WHAT SURVIVES -- the sample budget

The one difference between the U-test and training that has NOT been eliminated:

* U-test probe: **840 crops per class**, fitted jointly, fixed crop phase, a
  linear decision boundary over the whole set.
* Live discriminator: **~4.9 images per forward, ~2.4 per class**, an ONLINE head,
  a moving generator, and +/-8 px re-phasing every step (ladd_pixel_grid_jitter=8)
  plus +/-16 px diff-aug translation.

That is a factor of ~350 in evidence per decision. The U-test deliberately matched
the disc's geometry, pairing, decode path and crop screen -- so sample count and
phase are what is left.

**CLOSED 2026-08-27:** the decisive AUC-vs-n measurement is now complete for
the exact deployed VGG and RN50 statistics; full method and tables are in
`analysis/gan_tuning/AUC_SAMPLE_BUDGET_VGG_RN50.md`. At n=2 per class VGG/RN50
score 0.609/0.627; at n=10, 0.807/0.814; at n=25, 0.882/0.862. The present live
budget is ~2.4 per class, a 5x increase is ~12 and a 10x increase is ~24.
Therefore the SAMPLE BUDGET remains the redesign target. The same measurement
does **not** validate the existing surrogate: its D-side decode work is already
no-grad, its Sobolev teacher refresh still uses `_vae_decode_grad`, and the
measured surrogate gradient cosine remains 0.0075 despite 840 distillation
substeps.

## A8. PROCESS: a void arm caught by proof-of-fire, not by care

pixvgg_nojit attempt 1 was VOID. The jitter override was passed via `$DEXTRA`
(line 628) but the arm re-sets `ladd_pixel_grid_jitter=8` at line 667, and
overrides are LAST-WINS. It ran as an unlabelled exact replicate of
pixvgg_online. Because a replicate lands at ~0.64 like everything else, it would
have READ AS CONFIRMATION THAT JITTER DOES NOT MATTER -- the `slide12` failure
mode exactly.
Caught by reading the boot line expecting `grid_jitter=0` and seeing `8`.
FIX: the arm now exposes `ladd_pixel_grid_jitter=${JITTER:-8}` -- env-overridable,
default 8 (byte-identical), and unclobberable by a later line.
LESSON: checking last-wins for ONE key does not license assuming it for the next.
Verify the resolved value of every override from the run's own boot output.

## A9. READY 2026-08-27: balanced 5x arms and a gated VGG surrogate

The exact AUC-vs-n result in A7 changes the next comparison from the older
sequential `L=3`, then updates=2, then K=2 plan in §12. The launch-ready direct
comparison now raises evidence **inside one discriminator forward** while
preserving total per-step scored-image exposure exactly:

```
parent: K=1 * frames=2 * D-updates=5 = 10 logits/row/step
new:    K=2 * frames=5 * D-updates=1 = 10 logits/row/step
```

`vgg_5x` and `rn50_5x` use D-side `L=3`, working `decode_split=1`, horizontally
stratified crop origins, and no vertical crop bias. They inherit the working
VGG/RN parents' clean GT-vs-DMD-fake pairing and frozen enabled action critic.
The corrected VGG arm completed its connection proof but OOMed on holder
6148537. The scalar surrogate completed on freed holder 6148536. The lighter
RN50 direct smoke then reused holder 6148537 and independently OOMed
immediately after step 35, so the holder-safety failure is not VGG-only. W&B
IDs are `z9qoa7xh` (VGG proof then OOM), `pkhhezr2` (RN50 proof then OOM),
and `9u5rb6hr` (completed scalar-surrogate calibration).

This is not numerical optimizer identity: the parent accumulates five
mean-loss backwards into its deferred D step, while the new recipe performs
one mean-loss backward over the larger set. The reduced repeated-head gradient
is the intended balance counterweight; scored-image count is what is exactly
conserved.

The surrogate selector/run name is now explicit:
`vgg_surrogate_teacher_d5x_targets4perclass_fit24` /
`pixvgg_surrogate_teacherD5x_targets4perclass_fit24_cal_w0_*`. The name states all
three budgets: fivefold teacher-discriminator evidence per D forward, four
fresh target windows per class per refresh, and 24 cached/replayed fitting
substeps. `cal_w0` says that the student gradient is measured but not applied.
The real-pool capacity is 2560 two-frame windows (5120 nominal source frames,
1.25x the audited 4096-frame steady-state support floor); the first
2048-window startup was stopped at step 0 after the A21 overlap-margin warning.
The 90-step smoke does not pretend this capacity is already filled: it starts
at 8 windows and admits one fresh window per eligible step, with realized size
and fill calls logged explicitly.

The surrogate ruling in §4 remains binding for generator training: the failed
DINO field may not be assigned a nonzero weight. A new VGG-teacher surrogate
is therefore prepared only as a **weight-zero calibration stage** first. It
uses the trained LADD VGG discriminator as teacher, normalized Sobolev loss,
nonzero head init, 24 fitting substeps per eligible call, teacher refresh every
two eligible steps, and a held-out gradient audit every 20 global steps. It
must reach held-out gradient cosine >= 0.50 and produce a finite, nonzero
unweighted gradient ratio before activation. The launcher refuses an inherited
nonzero calibration weight and refuses active mode without an explicitly named
reviewed run, literal approval, a positive calibrated weight, and a cosine
that passes the gate.

The completed VGG scalar-surrogate run (`9u5rb6hr`) proves the mechanism but
falsifies the field. Four held-out audits finish at cosine 0.0643, relative
error 0.9993 and magnitude ratio 0.1166; training-target cosine is only 0.1436
after 312 fitting substeps. Activation is permanently blocked. The explicit
first-order direct-gradient replacement first ran on holder 6148537 as W&B
`aauzcknz`. Its first held-out audit reads cosine 0.1273, magnitude ratio 1.247 and
relative error 1.496. That is better than the scalar arm's first cosine
0.0492 but still below the 0.50 activation gate; training cosine rises from
0.145 at step 21 to 0.254 at step 31. It remains weight-zero and has survived
the step-35 rolling-state boundary that killed both direct feature arms. Its
all-at-once target build then OOMed at step 40 inside the unchanged
checkpointed VAE teacher backward, before the second audit. The current retry
uses `surrogate_teacher_target_microbatch=1`: it builds one crop graph at a
time and concatenates the identical four targets/class afterward, changing
peak residency but not evidence, teacher cadence, labels or loss. The
serialized retry is W&B `zffrfznl`; it completed 90/90 with no OOM or NaNs.
Its audits were 0.1606, 0.2081, 0.1490 and 0.3242 cosine. Final magnitude ratio
was 1.2068 and relative error 1.2938. The post-run checker passed 36/38 gates;
the two failures are the binding scientific gates (cosine < 0.50 and zero
parameter-gradient ratio). The direct field at the flash latent was nonzero
(norm 0.2091), but the full generator-parameter norm was exactly zero with
825/825 tensors reachable.

The current-teacher control ran on holder 6150252 as W&B `ttxdfxu8`, arm
`vgg_surrogate_directgrad_currentteacher_8targets_refresh_fit24`. It keeps four
real plus four fake logical targets per refresh, uses cache capacity one,
retains 24 fits and target microbatch one, and remains `cal_w0`. Its held-out
cosines were 0.154, 0.268, 0.342 and 0.484. The otherwise identical aligned-
source W&B `wucn3u1w` on holder 6150866 set only
`ladd_fake_sample_source=flash`; its cosines were 0.147, 0.313, 0.318 and 0.440.
Neither passes 0.50, and DMD-source is better at the final audit.

Both learned and matched tangent terms reached all 825 parameters but produced
exactly zero norm. The source was the historical newest-frame selector taking
the deliberately no-grad trailing block of a mixed CopySlices flash buffer.
The new `liveflash` arm publishes/consumes a frame-level graph-liveness mask
and takes the latest contiguous live frames. W&B `79re58zl` completed 90/90 at
weight zero: parameter delivery passed, while its final cosine 0.4647 missed
the 0.50 gate. The 16-target evidence escalation `u56r8lps` is the remaining
weight-zero surrogate calibration.

Media audit (2026-08-27): the two completed controls used
`sample_interval=90` with `max_steps=90`. The interval request is consumed on
the following generator iteration, so neither emitted the ordinary
`sample/pred_image_rollout` key. They did independently emit the same full
168-frame seven-chunk rollout as `sample/pred_image_7_chunk` at W&B step 11.
Those existing MP4 artifacts have now been aliased into
`sample/pred_image_rollout` at step 82 on both runs; no model output was
recomputed. The live-frame run uses `sample_interval=15` and emitted its native
full rollout at step 16. Future holder commands use the immutable-snapshot
entry point `sbatch/run_gan_crop_arm_snapshot.sh`; it rejects
`SAMPLE_EVERY >= MAXSTEPS`, with a CPU regression test covering the boundary
case.

Implementation and runbook:

- `model/latent_gradient_surrogate.py`
- `sbatch/run_gan_crop_arm_on_holder.sh`
- `sbatch/run_gan_crop_arm_snapshot.sh`
- `sbatch/check_gan_crop_arm.sh`
- `testing/test_latent_gradient_surrogate.py`
- `testing/test_gan_crop_arm_launch.py`
- `analysis/gan_tuning/GAN_CROP_ARMS_2708.md`

Review status: shell syntax and generated last-wins blocks checked; the prior
focused crop-route, VGG/RN50, memory-ordering, historical surrogate and
direct-gradient suite passed 174/174 tests. The expanded current-teacher,
aligned-source, pixel-feature, surrogate and launcher review passes **309/309**
on holder 6150867. Live status is recorded below rather than inferred from
static review.

**First live attempt update:** VGG 5x run `29hshvsh` proved K=2/L=3/F=5,
ten logits/sample, split decode and stratification with no clamp/fallback. It
then OOMed at step 25 when the inherited every-step parameter-gradient probe
attempted two extra full generator backwards while the direct decoder graph
was live; the subsequent real backward was short by 52 MiB. Direct recipes
now set `gan_grad_telemetry_every=0` and `texture_tripwire_every=0`. These are
optional diagnostics; the discriminator loss, D-update and crop-plan counters
needed for the experiment stay enabled. The no-diagnostic retry `vg5szfuq`
then established that `decode_split=2` itself remains too wide during ordinary
checkpointed VAE recomputation. A third full-direct attempt `aqwkce2f` used
`decode_split=1` and still OOMed at the same generator-backward boundary.

The live recipe is now explicitly `direct_D5x_G1x`. D updates and R1 use
K=2/L=3/F=5 (ten logits/sample); generator guidance uses the proven parent
K=1/L=2/F=2 graph (two logits/sample). The discriminator/head is shared, so
this enlarges evidence for learning D without forcing the larger graph through
the VAE backward. New `ladd_pix_d_*` and `ladd_pix_g_*` counters prove both
routes independently. Default-null G geometry preserves all older arms.

The first split-geometry run, `0pnk5jy8`, proved both routes exactly and made
normal progress through step 35, but the following outer action-critic
backward OOMed after the detached deferred-D pass had left its decoded-RGB
cache and allocator footprint resident. It did not fail in the direct GAN G
or D work. The direct recipe now enables a narrowly scoped, default-off
`ladd_pixel_post_d_memory_release`: after the deferred pixel-D closure's last
consumer it clears only the no-grad decode cache and calls
the local references to the already-executed closures before calling
`torch.cuda.empty_cache()`. It changes no objective, crop plan, RNG draw,
optimizer update, or cadence, and reports
`ladd_pix_post_d_release_events`. Startup-only `usnwjuna` was stopped before
training when the closure-reference lifetime was found. The complete retry
`z9qoa7xh` passed all 33 connection/geometry checks and fired three release
events, but still failed `no crash`: after clean progress through step 35, the
next core fake-score critic backward requested 1.10 GiB with only 1.19 GiB
device-free. This is outside the GAN G/D block, so direct D5x/G1x is not
holder-safe in the full rolling state even with the scoped release. RN50 run
`pkhhezr2` reproduced the same boundary: clean through step 35, then OOM on
the next core backward despite its lighter feature source. No further blind
direct retry is planned. D5x evidence continues only through weight-zero
teacher calibration. The expanded focused suite passes 174/174.

## A10. CLOSED 2026-08-27: live-boundary discrimination benchmark

The historical U-test is no longer the discriminator-selection gate. A new
capture arm saved the frame-matched GT and current DMD fake pixels at the
actual LADD pixel boundary: after shared latent augmentation/crop geometry,
VAE decode, border/phase jitter and frame selection, but before the feature
source. The bank contains 216 paired rows from 24 rides at `K=2/L=3/F=5`.
Generator feedback from the candidate pixel GAN was zero during collection.

Five-fold CV holds out whole rides. At full `K2/F5`, converged row-mean AUC is
0.999 DINOv2, 0.992 VGG16, and 0.993 RN50. Crop-to-crop and endpoint-to-middle
directions all transfer at AUC >=0.938; within-ride randomized-label controls
return approximately chance. The pretrained frozen bases therefore can do the
live discrimination task. This is not the old 84-frame temporal-offset cue.

The binding result is online learnability. With the GAN head LR `2e-5` and the
nine D updates present in the 45-step capture, row AUC is 0.898 DINO, 0.684
VGG, and 0.758 RN50. At `2e-4` with the same evidence/update count, VGG and
RN50 both rise to 0.899; repeated cached head-only optimization at `2e-5`
reaches 0.867/0.886. The VGG/RN50 failure is therefore primarily insufficient
head optimization, not absence of texture/style evidence. Adding crops alone
at the old LR/cadence is not the next experiment.

Scope: VGG/RN50 use their exact deployed statistic and pooled head. DINO uses
the exact frozen taps with a controlled pooled head, not the historical dense
CCM/CSM readout. Frozen random PixGAN is only a random-feature control and is
not a verdict on a fully trainable from-scratch conv critic.

Full method, all evidence-budget tables, controls, limitations, result paths
and decision rule:
`analysis/gan_tuning/GAN_ALIGNED_DISCRIMINATION_2708.md`.

## A11. CLOSED 2026-08-27: ride-disjoint surrogate field selection

The surrogate is now evaluated on the same exact `K2/L3/F5` live boundary as
A10, including stored fake latent crops, exact crop origins and the realised
RGB views. The 72-record/24-ride bank re-decodes with zero 8-bit pixel error.
Three rotating splits keep scalar-head calibration, surrogate fitting and
surrogate audit rides disjoint. Both frozen-field convergence and
chronological next-generator-step tracking use the production 96-wide,
six-block direct predictor with three seeds.

The quickest scalar discriminator is VGG pooled-stat/head LR `1e-3`: median
held-out row AUC is .802 after one update and .920 after three. DINO is
stronger from six updates onward (.939 at six, .970 at 300) but its latent
gradient field is effectively unlearnable by the current surrogate: best
frozen held-out cosine .034 and online next-step cosine .008. RN50 reaches
.133/.051. VGG reaches .178 frozen-field cosine and .130 next-step cosine
with the predeclared `2e-4` head.

A decisive repeat using the faster VGG/head `1e-3` improves next-step
surrogate tracking to .197 (student LR `1e-3`), versus a held-out mean-field
baseline of .058. Its frozen-field cosine peaks at .173 after 192 updates and
then falls, so final-step convergence is not the selection metric. Detached
RGB max/min conditioning changes next-step cosine only .197 -> .198 and
slightly lowers the frozen peak; it is not a material spatial-association
fix.

Ruling: VGG/head `1e-3` is the best teacher and student LR `1e-3` is the best
moving-field tracker, but **generator activation remains NO-GO** because the
genuinely unseen-ride next-step cosine is far below the existing .50 gate.
The old live “held-out” metric reused fitted crop batches and cannot supersede
this result. The next offline student must add temporal mixing and explicit
global context before another nonzero-weight GAN arm. Full curves, source/LR
tables, controls and artifacts are in
`analysis/gan_tuning/SURROGATE_FEATURE_FIELD_2708.md`. Capture W&B is
`4ucxnoih`.

## A12. CLOSED 2026-08-27: honest surrogate gate reflow

The holder-only exact-bank screen completed. It tested production-matched
Adam, direction-only fitting, temporal decoder context, VGG-style clip-global
context and the corrected per-crop origins. None crossed the honest gate.
Across fourteen candidates the best rotation q1 values were
`.277/.217/.227`; the next-teacher values were `.193/.156/.173`. Two-crop
evidence was only modestly helpful and detached feature/RGB conditioning did
not solve the field. The generator remained disabled.

No method is authorised merely for crossing the old in-sample .50 metric. The
candidate must cross .50 on disjoint surrogate-test rides, track the next
teacher version, and beat the production-matched spatial control. Full review,
rejected alternatives, implementation map and holder protocol:
`analysis/gan_tuning/SURROGATE_GATE_REFLOW_2708.md`.

Decision-gate correction: the concatenated global cosine can be inflated by
high-norm/easy samples, whereas generator consumption normalises every sample.
It is now diagnostic only. Qualification requires the worst seed's lower-
quartile per-sample cosine to exceed .50 for current and final next-teacher
fields on every tested rotation, with time measured in cumulative student
updates. Even a qualifying offline winner permits only a weight-zero live
production-geometry transfer check. The exact bank's fake-only targets do not
prove that fake-only training beats a mixed real+fake student, so that option
remains unselected pending a direct comparison.

The live direct-field audit now exposes per-sample median, q1, minimum and
sample count, aggregated across all eight distributed ranks, plus the rolling
minimum q1 over its final two audits. Gate-v2 activation cannot use the
historical flattened cosine: it requires that final-two minimum to be at least
.50, in addition to the offline verdict and calibrated gradient ratio.

The fake-only uncertainty now has a separate paired-domain sidecar. It uses
the captures' already aligned real/fake latent pairs, evaluates decisions only
on fake test rides, and compares matched fake24/realfake24/fake48 arms. Global
cosine is diagnostic; selection uses per-sample q1 deltas. Real+fake must beat
both fake-only controls in every seed and rotation or fake-only remains the
selected production route.

The review then moved one level deeper: instead of asking `S(z)` to infer both
a pixel-side preference and its transport through the decoder, A13 gives the
current pixel cotangent explicitly and isolates the stationary decoder
pullback.

## A13. CLOSED 2026-08-27: stationary decoder-pullback surrogate

The exact fixed-decoder target is `J_decode(z)^T v`, where `v` is the current
detached signed pixel cotangent. A new benchmark-only student therefore tests
`S(z,v)` against a same-parameter `S(z)` control and a wrong-vector control on
Cartesian held-out z/vector splits. It uses the live self-seeded fp32 WAN
decode, all 12 frames, the eight-pixel trim, and the converged VGG rotation-0
head. An independent factorisation identity passes at cosine 0.9999999.

The conditional operator is strongly learnable for smooth structured
cotangents: unseen-z/unseen-v worst-seed q1 reaches .745 initially and .831
with all captured z rows, while z-only remains approximately zero and the
wrong-vector control becomes negative. This validates both the pixel-to-latent
pullback framing and the need to provide the current pixel signal.

It does **not** solve the deployment-shaped VGG target. Four times as many VGG
vector identities, three times as many latents, early multiscale z gates and
3.6x capacity produce a best fixed-VGG q1/median of only `.159/.190`, with
relative MSE about .97 and predicted RMS about .22 of target. The VGG
cotangents are far more high-frequency than the structured controls, and the
same VGG vector's exact pullback changes almost orthogonally across z (median
cross-z cosine .040 versus .656 for the structured family).

Ruling: rotations 1/2, live calibration and GAN integration are not authorised
for this architecture. The next candidate should distil a differentiable WAN
decoder and use its autograd VJP, or explicitly mirror the WAN decoder's
multiscale reverse hierarchy. Full protocol, tables, artifacts and decision:
`analysis/gan_tuning/DECODER_PULLBACK_2708.md`.

## A14. CLOSED 2026-08-28: compression oracle and naturally paired pullback

An exact resolution ladder now applies block-mean/repeat projections to real
VGG pixel cotangents and recomputes the WAN VJP. The current 4×8 temporal/
spatial latent-grid projection retains only `.129/.156` q1/median pullback
cosine. Temporal-only 2× reduction retains `.788/.817`; spatial-only 2× gives
`.586/.651`, and spatial-only 4× falls to `.268/.315`. The dominant
representation loss is spatial phase, not temporal evidence.

A separate deployment-aligned bank pairs every latent with the frozen-VGG
gradient generated by its own decode: 81 train and 81 ride-disjoint test
examples. Four direct identities pass above .9999997. The standard conditioned
student peaks at only .186 q1 and then overfits to .100; z-only reaches .156,
and wrong-vector control peaks at .036. A 39.3M high-bandwidth pyramid retains
four times the standard final element capacity but reaches only .098 robust q1
while exceeding .90 on train rides.

Ruling: pooling is conclusively destructive, but natural pairing and raw
bandwidth do not solve the transport. The next candidate must preserve spatial
phase and mirror/distil the actual WAN decoder backward hierarchy; no live GAN
arm is authorised. Full results:
`analysis/gan_tuning/COMPRESSION_PULLBACK_ORACLE_2808.md`.

## A15. CLOSED 2026-08-28: multiscale SFT and detached decoder state

Three naturally paired arms were tested at an equal 3,600 target
presentations: multiscale spatial SFT from the unpooled VGG cotangent, SFT plus
matching-resolution detached WAN decoder activations, and shuffled-gradient
SFT. No baseline or global-FiLM arm was run. The reverse grids exactly mirror
WAN's configured order: `12x192x256`, `12x96x128`, `6x48x64`, and
`3x24x32`. State summaries compress channels only and preserve every spatial/
temporal coordinate.

SFT reaches worst-seed unseen-ride q1/median `.203/.223`; detached state raises
this to `.241/.271`; shuffled SFT reaches `.150/.174`. At those selected
checkpoints train/test q1 is `.226/.203`, `.340/.241`, and `.192/.150`, so the
gains are not train-only artifacts. Continued fitting raises train q1 to about
.59 while test q1 falls to about .16.

Ruling: spatial injection is useful and WAN forward state contains additional
backward-routing information, but neither is sufficient. The .241 result does
not authorise a live arm. Further generic conditional-network scaling stops
here; the next candidate should mirror the WAN decoder backward hierarchy or
distil a differentiable decoder with JVP/VJP matching. Full report:
`analysis/gan_tuning/SFT_PULLBACK_2808.md`.

## A16. CLOSED 2026-08-28: blockwise linear local-VJP and composed chain

The decoder was decomposed first into six macro-stages, then nine operator
groups and finally 17 actual reverse boundaries. All 162 naturally paired
examples close the exact local chain with minimum cosine .99999976. Pixel and
boundary cotangents are never averaged: temporal/spatial upsample phases are
losslessly rearranged into channels.

Each student is exactly linear in its incoming cotangent; detached decoder
state controls only multiplicative spatial/temporal gates. The low core reaches
.990 q1. Pure temporal/spatial resamplers reach `.821-.953`, while residual
groups reach only `.563-.804`. Individual-block audit shows both a specific
hard region and composition error: mid-low residual blocks score .614/.653 in
the generic model, whereas the three mid-high blocks score `.816-.861`
individually but only .605 as one group.

Preserving the known identity shortcut exactly and learning only the residual
VJP correction is a material win. It raises eligible individual blocks by
`.019-.221`; mid-low block 1 improves `.614 -> .836`, and most blocks reach
`.886-.984`. Train/test q1 remains close. Mid-low block 2 remains below .8 at
.766.

The selected 17 operators were then composed on all 81 unseen-ride full
fields. The real frozen WAN RGB-head and three resampler VJPs were retained
exactly, each independently verified at q1 at least .99999994. Even so, final
latent q1 is `.432/.438/.429` across seeds (medians `.483/.489/.470`). Error
accumulates mainly through the mid-high and mid-low residual blocks. The
worst-seed .429 is below the .50 gate, so no live GAN arm is authorised.

Next: mirror the inside of each residual block, tying shortcut and
transpose-convolution weights and conditioning its two reverse
norm/nonlinearity gates on matching detached internal activations. Judge only
the full 81-example composed q1. If that fails, proceed to a lightweight WAN
decoder distilled with output/intermediate plus JVP/VJP supervision. Full
tables, boundary traces and artifacts:
`analysis/gan_tuning/LOCAL_VJP_AUDIT_2808.md`.

## A17. PASSED 2026-08-28: exact-block ladder and graph-free hybrid

The exact-block substitution ladder resolves A16's remaining question. The
original `.432/.438/.429` numbers were measured at WAN's first internal
decoder boundary; after adding the cheap exact scale/1x1/self-seed prefix, the
true public-latent baseline is `.472/.482/.476`. The prefix oracle is at least
`.99999994`, so this correction is exact.

Replacing only mid-low residual block 1 with its exact tied VJP yields
`.545/.550/.538` true-latent q1. Block 2 alone gives `.534/.543/.537`; both
give **`.602/.609/.605`** on all 81 ride-disjoint naturally paired examples.
All mid-low gives `.645/.657/.645`, all mid-high `.640/.644/.638`, and both
regions `.851/.856/.846`. Therefore a small number of local rotations, not a
globally unlearnable decoder backward, caused the failed chain.

The selected exact pair also passes the deliberately stricter transfer test:
108 unseen-latent x unrelated unseen-positive-VGG-cotangent combinations give
q1 **`.528/.522/.508`** across seeds. Stored and freshly recomputed exact
targets agree at q1 `.99999988`. Structured held-out cotangents reach at least
`.907` q1. Natural `v(z)` correlations do not explain the pass.

The graph-free runtime captures detached WAN state, obtains the current pixel
teacher cotangent, composes the frozen hybrid, and serves the resulting field
through a linear generator loss. Runtime capture, clamp/trim suffix and bundle
loading reproduce the audit path. End-to-end decoder transport is `125.95 ms`
and `4.13 GiB` median/peak versus `214.38 ms` and `11.13 GiB` for the exact
decoder VJP. This supports more evidence but does not support the previously
hypothesised 10x crop claim by itself.

Ruling: the exact mid-low 1+2 hybrid passes the offline `.50` gate and is
authorised for live use. A full exact WAN audit cannot coexist with the DMD
generator graph: after the graph-free hybrid completed and released its state,
the telemetry-only exact decode attempted to add the exact VJP's measured
`11.13 GiB` peak to a trainer with less than `1 GiB` headroom and OOMed. This
does not implicate the hybrid. Production therefore keeps
`surrogate_decoder_shaped_audit_every=0` and uses the three-seed ride-disjoint
plus strict Cartesian q1 (`.528/.522/.508`) as the activation gate.

The corrected weight-zero run `koq188e9` completed 20 steps without OOM and
passed 35/36 initial connection checks; its only failure was an observability
bug that discarded the common fake-selector log when the decoder branch
returned. The branch now merges those logs and has a regression test. The
calibration proved current LADD-pixel cotangent RMS `3.65e-6`, hybrid field RMS
`2.75e-5`, all `825/825` generator parameter tensors reached, and unweighted
hybrid/base parameter-gradient ratio `.0840847`. The calibrated 10% full
weight is therefore `0.10/.0840847 = 1.18928`.

The first nonzero run `pogftktc` completed 15 steps and passed all 35 revised
connection/health checks. It confirmed the ladder-endpoint fake source, the
current teacher, detached decoder state, exact stages 6+7, nonzero cotangent,
generator consumption and application. Prediction, rollout, real/fake and
conditioning videos were logged at steps 1, 6 and 11. The inherited 25-step
GAN warm-up was intentionally retained: at step 10 the applied weight was
`.475712`.

Run `mjtfwipq` then completed 30 steps and passed all 35 checks again. The
full `1.18928` weight was applied on generator step 25; step 26 completed with
peak allocation `66.16 GiB` and logged prediction plus rollout videos. The
latest logged discriminator loss was finite at `.6782`; there were no NaNs or
crashes. The first active probes' gradient-share values are non-binding: an
audit found that on GAN-active steps their denominator was taken after the
pixel term had already been folded into `gen_gan_loss`, partly comparing the
hybrid to itself. The weight-zero `.0840847` calibration is unaffected. The
probe now snapshots the complete non-pixel objective before folding, is pinned
by a regression test, and decoder arms use `log_interval=5` so every generator
turn exposes actual weight/share. The distilled lightweight decoder branch
remains deferred while the smaller tied/exact hybrid works.

## A18. INVALIDATED 2026-08-28: complete CARN-base live weight sweep

**Invalidated history.** Every A18 VGG arm used
`ladd_pixel_decode_cache=true` with the former unsafe temporary-`data_ptr`
key. Its live discriminator, R1, weight and treatment results cannot select a
GAN recipe. Only the offline Q1~0.60 measurement and mechanism/route-fire
observations remain interpretable. A19 documents the cache-off replacement.

The A17 exact-mid-low-1+2 hybrid is now running inside the exact source recipe
of W&B `g5ndc0fz`, rather than the earlier reduced surrogate smoke recipe.
Four isolated one-node arms initially bracketed reduced-smoke nominal shares
`.05/.10/.20/.30` at weights `.59464/1.18928/2.37856/3.56784`. They preserve
the frozen `.3` action critic, Flash-60 auxiliary path, CARN cycle, clean DMD-
band GT-vs-fake pairing and commit-off treatment. Initial videos and runtime
connection proofs pass on all arms. The full-base live calibration is
`.231-.294`, not the reduced smoke's `.0841`, so `.2/.4` replacements took
over the two overlarge nodes after their step-36 long videos. The inherited
VGG-head LR also remained near chance (`d_loss` about `.67-.72`), matching
the A10 under-optimization diagnosis; after step-56 long videos, the two
upper-weight nodes were reused for matched `.2/.4`, `gan_lr=1e-3` arms. No
holder was cancelled.
The follow-up `U=3` pair is now the cleanest live result: at steps 31/36 the
`.2` arm reaches D losses `.490/.373`, real/fake logits `+.280/-.180` then
`+.591/-.256`, with weighted parameter-gradient shares `.051/.121`. The `.4`
arm separates too, but its share reaches `.390` before the weight ramp has
finished and it opposes DMD more strongly. Thus the Q1 `.60` hybrid works in
the actual model and `.2` is the current parameter leader; later fully-ramped
rides are still running to distinguish stable tracking from a transient peak.
That transient did not remain safe: the matched step-76 long rollouts of both
`1e-3/U3` replacement arms develop a block/lattice, overexposed tail absent
from the base. They prove that the Q1 `.60` pullback is active, but reject
`.2/.4` at that optimization rate as production settings. Lower-weight and
lower-LR controls are running. The researcher clarified that the old base GAN
is not part of the requested treatment: `g5ndc0fz` supplies the CARN/DMD/
action/Flash reference, and the new GAN replaces its old GAN. The briefly
prepared additive PatchGAN queue was removed before it started. The requested
replacement arms are live: new-GAN+CARN-commit is W&B `vy8qpq4x` and
new-GAN+aux-minus is `xzn3igrt`. The aux-minus arm has already logged a
nonzero internalisation loss and forward-noiser gradient; the commit arm has
fired at the committed-memory site and reaches nonzero relative displacement
`.008157` at call 50. Both requested CARN consumers are therefore genuinely
active with the replacement GAN.

The lower-weight refinement is provisionally safer than lowering the D LR.
At the first fully-ramped step 51, `.1/LR=1e-3/U3` uses `.0422` of the full
parameter gradient, versus `.1026` for `.2/LR=5e-4/U3`; on the preceding
fresh phase their shares were `.055/.203`. Both step-56 168-frame rollouts
are visually clean. The `.1` arm remains close to the base (RGB RMS
deviation `.01561`, temporal RMS `.00604` versus `.00589` base) while adding
about 8% local Laplacian energy. The matched step-76 ride remains the decisive
test because that is where the rejected stronger settings failed. It rejects
both refinements: `.1/LR=1e-3/U3` and `.2/LR=5e-4/U3` reproduce the same late
grid/lattice and overexposed tail. Their RGB RMS deviations from the base are
`.23277/.23236`, with temporal RMS `.11891/.12627` versus `.06402` base.
The `.1` arm's HF tripwire rises `3.45x` while action loss stays normal.

This isolates U3 cadence as the common failure driver. By contrast, the
completed U1 `.2/.4` arms are visually clean on the same step-76 ride and at
step 96; U1 `.2` at step 96 has RGB RMS deviation only `.00843` and temporal
RMS `.00517` versus `.00518` base. A queued `.05/.1,LR=1e-3,U1` bracket will
identify the smallest useful safe generator weight. The live commit/aux-minus
U3 arms remain useful controls: their late videos will test whether either
CARN route suppresses the otherwise reproducible U3 failure.
Full contract, run IDs and results:
`analysis/gan_tuning/CARN_BASE_DECODER_SURROGATE_2808.md`.

## A19. ACTIVE CACHE-OFF 2026-08-28: Q1~0.99 all-residual replacement

**Critical live-GAN correction (28 August).** Offline pullback accuracy
(`.990999` paired / `.992351` Cartesian), the exact residual ladder and its
production loader remain valid. The live VGG discriminator experiments in
this section do not: their no-grad decode cache used temporary tensor
`data_ptr` as identity, and allocator reuse served stale pixels between
distinct real/fake/R1 micro-groups. Therefore `3rlvysdv` cannot authorize
`PIXW=.10`; `4vo4kwys/ududui2a/1wyle11y/u3i9p6ya`, the stopped transition/R1
wave, and the `fzuybrrm` screen cannot rank GAN or CARN settings. Preserve
them as bug-history artifacts, not controls.

The corrected launcher disables decode caching and reports an explicit
enabled flag. The first four cache-off calibrations
(`br6ivuzw/zoud58dc/ob5gqkvo/xlf99yu7`) finished with rising decode calls,
current-teacher/detached/stale-0, all 12 exact residual stages and complete
parameter reach. They exposed a second independent issue: the pullback field
was constructed before `_compute_r3gan_losses`, while D itself was deferred
until after generator backward. Thus these are clean transport controls but
not fresh-head weight authorities.

The definitive mode now resolves
`surrogate_decoder_fresh_disc_order=true` and
`ladd_defer_disc_update=false`. It skips the early field, performs the inline
current-batch D update, requires its monotone counter to increase, constructs
the Q1~0.99 pullback from that updated head, and folds the result exactly once.
Telemetry must show `disc_updates_before_field>0` and
`disc_updated_before_field=1` on every active field. Four matched repeats
completed as DC/R1=10 `83wlndxr`, DC/R1=1 `y9ry8jhr`, raw/R1=10 `bbj4flwh`
and DC+aux-minus/R1=10 `bfvw2wik`. Every field passed the fresh-order,
cache-off, exact-stage, current/detached/stale-0, `825/825` and Flash proofs.
Median unweighted ratios are `.882/2.404/.946/.777`.

The first valid active wave normalized to about 3% median share: DC/R1=10
`.035` (`qa0uuodt`), DC/R1=1 `.0125` (`cupkuznr`), raw/R1=10 `.032`
(`cedh3jod`), and DC+aux/R1=10 `.040` (`ybfvxoeh`). It remained finite through
about step 66 and its child steps alone were stopped to promote the comparison
to 300 steps. The 300-step wave is live as calibrated DC/R1=10 `.035`
`wkq1l6p2`, calibrated DC/R1=1 `.0125` `7o96m8uh`, and a matched strong-stable
DC/R1=1 `.0175` reference/aux-minus pair `htfhu37j/acmv0qur`. The matched pair
keeps the GAN fixed and changes only the intended CARN policy from reference
cycle to the tested R2-to-R1 aux-minus `.25` contract; both holders remain
alive.

A18's Q1~0.60 exact-6+7 transport is retired as the active experimental base;
its outputs remain historical controls.  New work is pinned by artifact hash
to the all-residual graph-free bundle measured at paired worst-seed Q1
`.990999` and strict Cartesian worst-seed Q1 `.992351`.  All 12 WAN residual
stages are analytic/tied, all four fixed stages and the public latent prefix
are exact, and only the frozen stage-0 low core remains learned.  This gives
the current discriminator cotangent first-field quality without surrogate
optimizer updates, warm-up, replay or reconvergence.

The production calibration also restores the intended Flash GAN surface:
the VGG discriminator and generator pullback both consume the t=60 Flash
fake, with the latter restricted to graph-live contiguous frames.  DMD, stat
anchor, Flash-DMD and recent CARN consumers are retained.  The old direct
decoder GAN term, latent-gradient surrogate and pixel PatchGAN are not built
or applied.  In-trainer exact VJP audit remains disabled.

The following calibration account is retained as invalidated history. The two
former `PIXW=0` calibrations completed, but neither is a weight authority.
Slow-head control W&B
`nze64p52` passed the mechanism but remained wrong-order at all four active D
probes.  Fast-head W&B `3rlvysdv` passed the 12-stage/current/detached/stale-0,
Flash-live and `825/825` reach checks on all five fields.  Its unweighted
full-parameter ratios were `.30381/.34103/.27178/1.05273/.26399` (median
`.30381`), while D margin improved to `+.9077` and D loss to `.3431`.  Peak
working allocation was about `69 GiB` on a 95-GiB device.

This yields a fresh first weight of `.10`: median expected generator share is
3.04% and the observed high-field event is capped at 10.53%.  It does not
reuse the obsolete `1.18928` calibration.  The nonzero smoke is W&B
`mpt1nx4y`, child step `6161257.16` on `nid010293`, with `LR=1e-3,U1`, Flash
t=60 and videos every 15 steps.  That child finished before the old holder was
retired under explicit researcher instruction.  The definitive
launcher explicitly disables the vestigial pixel-texture critic, OF GAN,
alternate fake head, online real teacher, frozen-teacher pass, VAE roundtrip,
state-probe auxiliary and old latent surrogate.  The small tested frozen
action critic remains alongside DMD, stat anchor, Flash GAN and the retained
recent-CARN modes.  Detailed contract and live readings are in
`analysis/gan_tuning/CARN_BASE_DECODER_SURROGATE_2808.md`.

The 35-step smoke completed with `.02/.04` ramped weights and
`1.916%/1.485%` applied full-parameter shares.  Its step-31 discriminator was
correct-order, transport remained current and complete, and the 144-frame
rollout has no lattice or exposure tail.  It therefore passes the immediate
closed-loop gate but, correctly, is not treated as a full `.10` late-rollout
test.  Old holders `6161257/6161260` were then cancelled under explicit
researcher instruction after their children had finished.  Four matched
100-step validations completed on the replacement holders: reference
`4vo4kwys`, commit `ududui2a`, aux-minus `1wyle11y`, and commit+aux
`u3i9p6ya`.  Every field proved all 12 exact stages, current teacher, detached
state, zero staleness, Flash selection and `825/825` parameter reach.  All
step-96 seven-chunk videos are free of the old lattice/exposure failure.

Across full-weight steps 46--96, reference is the most stable head: correct D
ordering 9/11, median margin `+.04797`, median loss `.66976` and applied-share
median/max `2.73%/6.92%`.  Commit is 8/11 and reaches 18.37% share plus a late
action-loss warning; aux-minus is 6/11 with a 34.93% transient share; combined
is 9/11 but ends wrong-order.  None cleanly improves the reference GAN in this
100-step window.  This is an interaction result, not a reversal of the longer
standalone CARN visual ruling.

The matched CARN wave uses raw VGG inputs.  The separate completed brightness
screen selects the guarded production front end: Flash/U1, per-frame/channel
spatial-DC rejection, and stat anchor (`fzuybrrm`).  It cuts the hazardous
step-76 tail-luma error from raw `+21.45` to `+13.49`; removing the anchor
collapses to `-40.26`.  Thus raw arms remain controls and any head-parameter
winner must be transferred once to DC before promotion.

The inherited R1 gamma 10 produces mode-agnostic finite spikes up to 2622, so
the launcher now exposes a validated `LADD_R1_GAMMA` knob.  Wave two is live
without cancelling either replacement holder: former-plus `ir2yhn2f`,
latter-minus `jrcll5m8`, R1-gamma-1 reference `sf4skqwf`, and U2 reference
`14anf43a`.  The latter pair isolates R1 strength and update cadence while
holding Flash, CARN reference, `.10` weight and Q1~0.99 transport fixed.

## A20. ACTION-CONDITIONAL CORRECTION (2026-08-28)

The 300-step visual comparison exposed a control failure before completion:
the strong R1=1 `.0175` reference (`htfhu37j`) and aux-minus (`acmv0qur`)
arms leave the commanded trajectory, while the calibrated R1=1 `.0125`
(`7o96m8uh`) and R1=10 `.035` (`wkq1l6p2`) arms remain on it. This is
consistent with increased GAN leverage but is not an action-critic
configuration difference. All four runs have the same frozen Flash action
critic and guidance weight `.3`, while all four VGG discriminators resolve
`ladd_use_prompt_cond=false` and `ladd_cmap_dim=0`. The aligned action token
and modulation tensors are built by the trainer, but the VGG pixel branch
does not consume the WAN `conditional_extra`; its score is therefore the
marginal real-image texture/style score, independent of throttle and steer.

Through W&B step 161 the applied-share medians are `.0177/.0139` for the
aux/reference strong pair versus `.0136/.0149` for the R1=10/R1=1 calibrated
pair, so typical realized share does not cleanly separate good from bad. The
peak excursions do: `.197/.228` versus `.126/.114`. The separate action loss
rises late on both visually good and bad arms, so increasing its scalar alone
is not a clean causal repair. The mechanistic failure is that a stronger
action-blind image prior can prefer a plausible-looking trajectory that is
incompatible with the supplied control while the independent action critic
only competes with it after gradients are combined.

A default-off action-aware path is now implemented. The first attempt targeted
the dense LADD projection heads and failed before W&B registration because the
winning VGG GAN deliberately does not build those heads: its discriminator is
the orderless `[mu,sigma,Cov] -> MLP -> scalar` pooled readout. The corrected
path retains that readout and adds a standard projection-discriminator term
between its final pooled statistic embedding and a learned action cmap. The
temporal conditioner retains mean, scale, first, last and signed endpoint delta
rather than globally averaging away action order.

Each D update also scores the *same pooled VGG evidence* under a different
valid action sequence from another row and adds a wrong-action negative loss.
This prevents the conditional-GAN shortcut in which D simply ignores its
condition. The alternate score reuses the exact decoded pixels, frozen VGG
maps and pooled hidden vector, so it adds no VAE decode or VGG forward and
does not recreate a dense spatial lattice. The Q1~0.99 generator query now
receives the action window aligned to the selected contiguous Flash latent
slab; the post-current-D pixel cotangent is therefore conditioned on the same
actions instead of silently querying an unconditional teacher.

Initial scope is deliberately fail-closed: pooled VGG `ladd` readout,
positional GT-vs-fake, FD R1 and the existing micro-batched D path. New knobs are
`ladd_use_action_cond`, `ladd_action_cmap_dim` and
`ladd_action_mismatch_weight`; all default off/zero. Runtime telemetry is
`ladd_action_cond_active`, `ladd_action_mismatch_weight`,
`ladd_action_mismatch_token_rms`, `r3gan_d_wrong_action` and
`r3gan_d_loss_action_mismatch`. CPU contracts prove temporal sensitivity,
single-feature-pass reuse, fail-loud missing conditions, a valid row
derangement, pooled-readout projection, conditioned `score_pixels`, and
inclusion of the mismatch loss in D backward (eight focused contracts pass).

Do not retrofit or relabel the four running 300-step controls. The next
holder experiment is a weight-zero, DC/R1=1, fresh-D, cache-off Q1~0.99
calibration with `GAN_ACTION_COND=true` and action cmap 64. Two corrected
pooled-VGG calibrations are live on holder `6170182`: mismatch weight 1.0 is
W&B `lmeubie7`, and mismatch weight .25 is `uhb3i1lc`. They must prove nonzero
condition RMS, aligned generator-side conditioning and a learned matched-real
versus wrong-action margin before deriving a nonzero `PIXW`; `.0125` or
`.0175` must not be inherited blindly because the discriminator head changed.

First active-field proof (W&B step 21) is clean on both arms. The mismatch
token RMS is nonzero (~`.01024`); the action-aware generator query selects
three action rows at absolute ride frames `[24,27)` for its three selected
Flash latents; D-updates-before-field and updated-before-field are both one;
Flash selection is `15/18`; decode cache is zero while decode calls are 30;
current-teacher/state-detached/staleness-zero/all-exact-residual flags all pass.
The wrong-action loss starts at `log(2)=.69315` and wrong/real logits are still
equal on the first optimizer update, which is the correct untrained baseline,
not yet evidence of a learned margin. Initial unweighted parameter-gradient
ratios are `.927` (mismatch 1) and `.842` (mismatch .25), while DC ratios remain
`5.36e-9/4.77e-9`. Selection must use the 60-step median and margin trajectory,
not this single first field.

Both 60-step action-conditioned calibrations subsequently finished and reject
this pooled-head formulation. For mismatch weights `1/.25`, median
correct-real minus wrong-action margins are `-1.80e-4/-1.48e-4`, while the
mismatch losses remain `.69324/.69324`. The shuffled action tensors are
nonidentical, all generator/Flash/Q1 wiring proofs pass, and final real-minus-
fake margins reach `2.48/2.89`. Thus this is not a dead input or stale-field
bug: the orderless pooled VGG statistic has enough information for texture
real/fake discrimination but not action-to-motion compatibility. Do not run a
nonzero pooled action-conditional arm.

The fallback comparison reuses `htfhu37j` unchanged (`PIXW=.0175`, DC,
R1=1, stat anchor 1, Flash t=60, U1, fresh D-before-field, Q1~.99 transport)
and strengthens the independent action apparatus instead. Holder `6170182`
runs frozen guidance `.6` as W&B `ea2g8x49` and an online action critic as
`pstbyie5`. The online treatment retains generator guidance `.3`, trains the
59.3M-parameter critic at LR `3e-4` for two updates/step against
`action_teacher_mode=all`/`teacher_action_encoder=pca_raw`, and weights its
teacher regression `.5`. Its PCA-space self-check is `5.96e-8`; the action GAN
projection and mismatch loss are off in both arms. Judge action retention from
the matched 15-step videos and action/critic telemetry through step 300.
