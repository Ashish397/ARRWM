# Deep-research prompt: audit our GAN / adversarial-distillation stack against the state of the art

You are a research agent with web access. Your task is to audit the
adversarial (GAN) component of a DMD-distilled driving world model against
how the field actually builds discriminators for diffusion distillation, and
to tell us **where our implementation makes silly assumptions, skips standard
steps, or deviates from what works** — ranked by how suspicious each
deviation is.

Everything below is a faithful transcription of what our code ACTUALLY does
(file:line citations refer to our private repo, branch `dmd_one_step`; you
cannot see the code, so treat the transcription as ground truth). Our GAN has
a documented, reproducible failure mode — **every learning discriminator
slides monotonically to a win and then breaks the generator, and our
best-rated runs turned out to have a functionally dead disc** — so read the
empirical section carefully before judging.

---

## 1. The system, as if you were joining the team

### 1.1 Host model and training loop (context)

- **Generator**: Wan2.1-family causal video DiT (~1.3B class),
  action-conditioned (per-frame action tokens + AdaLN modulation), KV-cache
  streaming autoregressive rollout, 16-channel Wan-VAE latents at 60x104,
  3 latent frames per chunk (`npb=3`). Distilled to a 4-rung sampler
  `[1000, 625, 357.14, 208.33]` by reverse-KL DMD from a frozen
  teacher-forced bidirectional teacher ("14e"); teacher CFG scale 0. The
  DMD side is audited by a companion prompt; here you only need: the DMD
  gradient is `pred_fake_x0 - pred_real_x0` (eq.8-normalized) on 3-9
  supervised frames of a 21-frame window, the critic (fake_score) trains
  every iteration, and the generator trains every 5th iteration
  (`dfake_gen_update_ratio=5`).
- **Batch = 1 ride per rank** (one long driving video per GPU), 8-32 ranks,
  DDP, bf16, gradient checkpointing everywhere. Under "rolling" training the
  generator autoregressively rolls 2-6 chunk-steps deep through the ride,
  committing its own (detached) predictions into its KV cache; under
  "stationary" training only the first window is trained.
- **The GAN fires only on generator iterations** (every 5th trainer step).

### 1.2 What the discriminator sees (fake side)

The gradient-bearing fake is **`flash_dmd_gan_x0`**
(`model/dmd_action_forcing.py:10312`, `_surface_flash_gan_slab`): an extra
near-clean generator forward at `t = flash_dmd_gan_t = 60` over the current
rolling slab (`streaming_chunk_size = 18` latent frames = 6 chunks in the
rolling runs). The G-side adversarial gradient flows ONLY through this t=60
refinement forward — never through the high-noise sampling rungs (those are
no_grad in the rollout). Under rolling, these chunks are student-rollout
chunks whose content has drifted from the GT world position.

### 1.3 Discriminator architecture (`model/ladd_disc.py`)

LADD-style frozen-teacher-feature discriminator
(`gan_backbone = ladd_teacher_feat`):

1. **Frozen Wan teacher backbone** (`real_score` itself, ~1.3B, frozen):
   input latents forwarded once with local hooks capturing 5 evenly-spaced
   transformer-block outputs (`WanFeatureProjector`, `ladd_disc.py:75`;
   hooks are installed per-call, not persistent). The disc input carries
   `requires_grad=True` on the gen side, so gradients flow back through the
   frozen teacher to the fake latents.
2. **CCM**: per-tap 1x1 linear channel mix to a common dim
   (`ladd_proj_dim`; 256 in the committed config, 512 in the rolling runs).
3. **CSM**: FPN-style top-down cross-scale fusion (`ladd_use_csm=true`).
4. **Per-tap heads** (`LADDDiscHead`, `:411`): spectral-norm 2D convs
   (kernel 3, `n_power_iterations=1`) + GroupNorm + LeakyReLU residual
   block -> **per-token logits** (one logit per spatial token per frame,
   concatenated across taps; a 6-latent-frame pair yields ~4.7e4 logits).
5. **Prompt conditioning**: StyleGAN-T-style cmap inner product,
   `ladd_cmap_dim = 64`, keyed on the mean pooled T5 prompt embedding.
6. **Action conditioning "for free"**: the frozen teacher forward consumes
   the per-frame action tokens + action modulation for both real and fake
   inputs (built from the ride's action window, detached).
7. Optional **stat head** (`LADDStatHead`, `:482`, spectral-norm MLP over
   channel statistics) — off in the runs discussed here.
8. Trainable disc params = CCM + CSM + heads only, ~10-15M.

An alternative plain 3D-ConvNeXt R3GAN discriminator exists
(`model/r3gan.py:R3GANDiscriminator3D`, GroupNorm+GELU, no spectral norm, as
the R3GAN paper prescribes) but the LADD teacher-feature disc is what the
current campaign uses. NOTE the committed default config
(`configs/action_forcing_phase3_dmd.yaml`) does not even construct the
current stack (gan_backbone default mismatch) — the real settings live in
run overrides.

### 1.4 Losses (`model/r3gan.py:231-355`)

Relativistic pairwise (RpGAN, R3GAN-style), elementwise over tokens:

```
D-loss: softplus(D(fake) - D(real)).mean()          # rpgan_d_loss, :231
G-loss: softplus(D(real) - D(fake)).mean()          # rpgan_g_loss, :240   (real detached)
```

Variants implemented: **all-pairs** (every real vs every fake across the
batch dim, `:251/:276`) and the default **block-diagonal matched** mode
(each fake scored only against its own K matched reals — see 1.5).

**R1/R2**: zero-centered gradient penalties. The R3GAN module has exact
autograd versions (`r1_penalty/:288`, `r2_penalty/:326`, gamma/2 * ||grad||^2
with create_graph), but the LADD path uses a **finite-difference
approximation**: perturb the input by `sigma = 0.01` Gaussian, R1_grad_sq ~=
`((D(x+eps).sum_tokens - D(x).sum_tokens)/sigma)^2`, penalty `0.5 * gamma *
mean`, applied to at most `ladd_r1_num_samples = 6` subsampled unique reals,
on a debt-based cadence (`ladd_r1_every_n_steps`; a modulo-cadence bug that
could skip R1 forever was fixed). A `ladd_r1_normalize_tokens` flag divides
the FD by the token count T~4.7e4 — with it ON, an intended gamma=10 was an
EFFECTIVE 4.6e-9 (gamma carries 1/T^2), which is how "gamma had to be ~1e5x
bigger than anyone tried" happened. R2 defaults off
(`ladd_r2_gamma = 0`).

### 1.5 Pair construction (`trainer/causal_action_forcing_train.py:5059`,
`_ladd_run_pair_mode`)

Three modes (composable; the current campaign uses **gt_transition**):

- **gt_vs_fake**: real = GT chunk, fake = student chunk, position-matched
  (or all-pairs / wide-real / nearest-match variants).
- **adjacent_chunks** (ASD-style): real = student chunk_i, fake = student
  chunk_{i+1}.
- **gt_transition** (the winner): each sample is a **transition pair**
  `cat(chunk_i, chunk_{i+1})` along frames = 6 latent frames; real = GT
  pair, fake = student pair; all 5 consecutive pairs of the 6-chunk slab
  per gen-iter. Judges DYNAMICS rather than single-chunk appearance —
  gt_vs_fake lost in all 4 tried variants.

**Nearest-match real retrieval** (`ladd_gt_transition_match=true`, trainer
`:5900+`): for each fake transition, retrieve its top-M=8 nearest GT
transitions by mean-L1 on the (optionally mean-equalized) latents from a
ride-local candidate pool (`streaming_state["gt_match_latents"]`; measured
**n_cand = 22** candidates under rolling), then sample K=4 fresh per
D-update (step-seeded), dedupe → 7-11 unique real windows per update, hard
cap 12. RpGAN is block-diagonal: each fake vs its own K reals.

Optional input normalizations (mostly off in current runs): magnitude
equalization of pair members (m1/m1m2 per-channel RMS/standardize),
cross-pool mean/std equalization (kills absolute-brightness cues,
`xeq_preserve_delta` keeps the within-pair brightness transition), action
blinding, **CARN-former** (degrade the real pair's former chunk with a
learned forward-noiser so real = "self-correcting transition"; with moment
restoration to avoid a measured white-collapse feedback loop).

### 1.6 Wavelet-HF disc input (`model/wavelet_hf.py`)

Single-level Haar **SWT** (stationary, resolution-preserving) per channel →
LL/LH/HL/HH same-size sub-bands; `drop_ll=true` discards LL; a 1x1 conv
adapter (Xavier gain 0.1, learned, disc-trained) maps the HF channels back
to 16. Intended as WGSR-style "disc sees only HF". **Critical config fact:**
the rolling runs used `ladd_wavelet_hf_augment=true`, meaning the disc input
is `raw_latent + adapter(HF)` — NOT HF-only — so the adapter is a *learned,
adversarially-optimized high-pass pre-emphasis on top of a full-spectrum
passthrough*. Also: wavelet ON forces disc inputs clean (`disc_t=0`);
wavelet OFF noises disc inputs at t=60 (trainer `:5614-5640`) — the two
knobs were never varied independently (hidden confound).

**DiffAugment** on latents: `ladd_diff_aug_policy = "flip,cutout,
translation"`, same per-sample randomness on real and fake, differentiable,
applied before the projector (skipped in the matched branch's own D-loop
except flips). Not adaptive (no ADA-style p控制): fixed policy, always on.

### 1.7 Schedule, optimizer, warmups

- Disc updates: `gan_updates_per_step = 5` D-updates per generator iter
  (each with a fresh K-match resample + fresh noise/aug), i.e. 5 D-updates
  per 5 trainer steps, GAN active only on gen iters. In the rolling runs:
  `gan_lr = 2e-5`, AdamW betas (0.0, 0.9), wd 0, `gan_loss_weight = 1.0`
  (the committed default is 0.05 but the campaign ran 1.0),
  `ladd_disc_loss_weight = 1.0`, grad clip 10.0.
- G side: after the D-updates, one disc forward in eval mode with disc
  params frozen; `g_loss = softplus(D(real_matched).detach() - D(fake))`
  added to the generator total at weight `gan_loss_weight x
  ladd_disc_loss_weight x warmup_ramp` (linear/quadratic/cosine ramp over
  `gan_warmup_steps = 500` after `gan_critic_warmup_steps`; ramp done long
  before the observed failures). The gen-side weight can optionally be
  coupled to the DMD MAE gate (`_couple_gan_weight_to_gate`).
- `gan_disc_start_step` delays D-updates; `ladd_defer_disc_update` defers
  the D backward out of the peak-memory window; the D-update can be
  **micro-batched** (`_ladd_disc_update_microbatched`, trainer `:4798`,
  bit-faithful gradient accumulation with the last group carrying the DDP
  allreduce) — this was the fix for a 92.9GB disc-backward transient.
- No EMA on the discriminator; no EMA generator is used for the GAN; no
  adaptive weighting (no LDM/VQGAN-style adaptive lambda from gradient
  norms), no TTUR asymmetry beyond the 5:1 update count, no disc lr decay,
  no top-k/instance selection, no multi-scale image pyramids (multi-tap
  teacher features stand in for multi-scale).

---

## 2. Empirical record (treat as established; this is the heart of the audit)

From `docs/GAN_FORENSICS.md` (wandb datastore forensics over 6 rolling runs)
and the stationary campaign:

1. **Two attractors, no equilibrium.** Every learning-disc arm slides
   monotonically toward a disc win; no run found a stable adversarial
   equilibrium at this operating point (weight 1.0, lr 2e-5, 5 updates/gen,
   ride-local pool). The two observed end states: **dead disc** (d_loss
   pinned at ln 2, zero gradient both ways — and those are the runs rated
   BEST visually) and **winning disc** (d_loss < 0.10 sustained → generator
   breakdown).
2. **Breakdown threshold is sharp and reproducible**: strain (DMD-loss
   spikes, disc-flip oscillations where d_fake goes positive) once EMA
   d_loss < ~0.3; irreversible breakdown once d_loss sustained < ~0.10
   (2.25-logit gap): rollgansure broke at ~step 321, rolllong at ~466-471
   (corr(log d_loss, log dmd_loss) = -0.73 at lag 0). Post-breakdown flow
   analysis: forward divergence dies first (egomotion loses its radial
   depth component), then uniform pan/thrash, then a frozen world.
3. **The disc always wins because it can memorize**: the real pool is
   ride-local, measured 22 candidate transitions per ride, 7-11 unique
   reals per D-update, one ride per rank — a 15M head stack over frozen
   5-tap teacher features vs ~22 real windows. Nothing about lr/weight
   changes the endgame, only its date.
4. **R1 findings**: gamma=1e6 on all 5 updates = lobotomy (d_loss pinned at
   ln2, |logits| < 1e-3 — the rated-GOOD "GAN" run was functionally
   GAN-off). gamma=0 = disc wins in ~100 steps. gamma=1e3 @ 1-of-5 =
   plateau at d_loss 0.2-0.4 with intermittent DMD spikes and
   cartoon/texture artifacts — slower slide, same direction. A code-trace
   prediction that spectral-norm heads structurally annihilate R1 was
   **falsified by measurement** (R1 at 1e6 demonstrably reshaped the disc).
5. **Dead disc = best visual quality** so far; a WIDER disc gap correlates
   with WORSE samples (gt_vs_fake v2 had the widest gap 2.9-4.2 and greyed
   out). "Healthy adversarial learning" (d_loss 0.10-0.17 with drift flat)
   was observed once (rollcombo2, stopped at ~375 — on the same d_loss
   trajectory rolllong followed into breakdown ~90 steps later).
6. **Wavelet paradox**: on the STATIONARY gt_transition GAN a 2x2 smoke
   proved wavelet-ON = dead disc / wavelet-OFF = healthy, regardless of
   equalization → project rule "ladd_wavelet_hf_enabled=false". Under
   ROLLING the wavelet-ON arms were NOT dead (they learned and slid like
   the rest). The wavelet knob silently flips disc input noise (ON→t=0,
   OFF→t=60), so band restriction and input noise have never been
   deconfounded.
7. **Texture-drift target**: the failure the GAN is *supposed* to fix is
   measured precisely (frozen-model probe): radial spectrum tilt (high-band
   loss front-loaded, low-band/DC inflation walk), channel-covariance
   eigenspectrum collapse (-0.24 effective rank/roll), kurtosis decay
   (Gaussianization). The DMD+stat-anchor+affine recipe fixes DC/mean drift
   but not these; the factorial result is "affine necessary for drift, GAN
   necessary for texture/life, both jointly sufficient" — yet the GAN in
   that winning arm was later shown to be the near-dead-R1 configuration.
8. **Misc bugs found during forensics** (already known): dead knob
   `ladd_pair_start_seed_boundary` (read nowhere); `last_r1_fired` logging
   overwritten by updates 1-4; d_* logging bug in multi-mode runs; disc
   checkpoint head-shape mismatches silently dropping cls layers on resume.

---

## 3. What we want from you

Compare our adversarial stack, step by step, against the actual published +
released implementations of:

- **LADD** (Sauer et al., latent adversarial diffusion distillation) and
  **ADD/SDXL-Turbo** — teacher-feature discriminators: head architecture,
  loss (they use hinge, not RpGAN), R1 usage, conditioning, update ratios,
  learning rates, how they avoid disc domination.
- **DMD2's GAN term** (Yin et al.) — disc on backbone features of the FAKE
  score network, its weighting relative to the DMD loss, its update
  schedule, batch sizes.
- **R3GAN** ("The GAN is dead; long live the GAN") — their exact RpGAN +
  R1 + R2 recipe, gamma ranges, why they reject spectral norm, whether
  R1-only (no R2) is known to be insufficient (we run R2=0).
- **StyleGAN-T / StyleGAN-XL / Projected GAN** — frozen-feature-backbone
  disc practice: multiple backbones, random projections, per-scale heads,
  cmap conditioning, DiffAugment/ADA usage, known pathologies of
  frozen-backbone discs (e.g., the projected-GAN "cheating" literature).
- **APT / adversarial post-training for video** (Seaweed-APT etc.) and
  **Self-Forcing / CausVid**-adjacent adversarial video terms — disc design
  for autoregressive/rolling video specifically, real-pool construction,
  trajectory vs frame discrimination.
- **GAN stability literature** as it applies here: TTUR, ADA (adaptive
  augment probability), disc EMA / historical averaging, replay buffers of
  fakes, top-k training, relativistic vs saturating losses at tiny batch,
  R1 gamma scaling laws (gamma ~ resolution/batch heuristics), lazy
  regularization correctness, instance noise schedules, disc capacity vs
  data-size rules of thumb.

Then answer, concretely and with citations:

1. **Per-step diff**: for each stage — (a) fake construction (our t=60
   flash slab; is restricting adversarial gradient to a near-clean
   refinement pass standard? LADD noises BOTH sides at sampled student
   timesteps — we noise neither, or only at fixed t=60), (b) real-pool
   construction (ride-local 22-candidate pool, nearest-match retrieval,
   B=1/rank), (c) disc architecture (frozen in-domain teacher taps +
   spectral-norm conv heads + per-token logits + GroupNorm), (d) loss
   choice (block-diagonal matched RpGAN; softplus pairwise), (e)
   regularization (FD-R1 sigma=0.01 on 6 reals, R2 off, token-norm
   footgun, no ADA), (f) schedule/optimizer (5 D-updates per gen iter, lr
   2e-5 vs gen 1e-5, betas (0,0.9), weight 1.0 vs DMD ~0.5 magnitude), (g)
   conditioning (prompt cmap + action modulation; matched reals carry
   different actions than fakes) — what does each reference system do
   differently, and which deviations plausibly explain the two-attractor
   behavior?
2. **What are we missing entirely?** Candidates we suspect: adaptive
   discriminator augmentation (ADA) tuned by an overfitting heuristic
   (r_t); a d_loss-based freeze/thaw gate or adaptive G-weight (VQGAN-style
   lambda from grad-norm ratios); cross-ride / replay-buffer real pools;
   EMA generator as the G the disc sees; disc EMA; multi-crop or
   multi-scale token pooling instead of per-token logits; hinge loss;
   R2 alongside R1 (R3GAN says both are required for convergence);
   instance-noise annealing; batch-statistics tricks (minibatch std) that
   are impossible at B=1 and what the field does instead; timestep-sampled
   adversarial pairs (LADD-style) instead of fixed t=60. Which of these are
   load-bearing in the reference systems?
3. **Judge the diagnosis**: our forensics concluded the root cause is
   "tiny memorizable real set + per-token logit capacity + 5:1 updates at
   weight 1.0 → disc always wins; R1 only delays it," and proposed (ranked)
   (v) replacing the disc with frozen-teacher feature matching
   (min-over-K, non-adversarial), (gate) d_loss freeze/thaw hysteresis at
   0.45/0.60, (iii) cross-ride balanced replay pools, (iv) fixed band
   weighting instead of the learned HF adapter, (i) seam-tight 2-frame
   windows, (ii) trajectory-level disc. Critique this ranking against the
   literature: is a non-adversarial perceptual/feature-matching term a
   known adequate substitute for the GAN in distillation (cf. LPIPS terms
   in consistency/distill work)? Is the freeze/thaw gate a known pattern
   (cf. "GAN balancing" literature) or a hack with known pathologies?
4. **Explain the wavelet paradox if you can**: is there precedent for
   HF-only or HF-emphasized discriminator inputs (WGSR etc.) in
   distillation, and for the observed stationary-dead vs rolling-alive
   flip? Is `raw + learned_HF_adapter` (our augment=true) a known
   anti-pattern (adversarially learned pre-emphasis)?
5. **Rank the top 5-10 most suspicious deviations** from field practice by
   (expected impact on: disc-domination breakdown at ~466, cartoon/texture
   shaping, dead-disc uselessness) x (confidence the field's way is
   better). For each, name the exact change and the source justifying it.
   Include concrete numeric suggestions (R1 gamma given ~4.7e4-token
   logits and 60x104 latents, update ratio, lr pair, gan weight relative
   to a DMD loss of ~0.5, ADA target r_t, replay-buffer size).

Format: a structured report with sections matching questions 1-5, explicit
paper/repo citations (arXiv IDs, GitHub paths where possible), and a final
one-page executive summary. Where our transcription is ambiguous, state your
assumption instead of asking.
