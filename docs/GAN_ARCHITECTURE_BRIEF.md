# The transition GAN: complete design brief

**Purpose of this document.** We are distilling a bidirectional video-diffusion
teacher into a *causal, few-step, autoregressive* driving world model. A GAN was
added to supply **texture/style realism** that the distillation loss does not.
It is not doing so. This document states every design choice in the current GAN
— architecture, inputs, targets, conditioning, losses, schedules — together with
the measured behaviour, so that an outside reader can answer two questions:

1. **Why is this critic failing to transfer real-world texture to the student?**
2. **What fundamental architectural change would make it work?**

Everything below is read off the implementation, with `file:line` anchors. Where
something is *unverified* it is marked **[UNVERIFIED]**.

---

## 1. System context

| | |
|---|---|
| Student | Wan-1.3B DiT, **causal**, KV-cache streaming, 4 denoising rungs `[1000, 625, 357.14, 208.33]` |
| Teacher | `v14e` — same Wan backbone + LoRA, **bidirectional**, frozen |
| Distillation | DMD/DMD3 (reverse-KL), teacher = real score, a trainable copy = fake score |
| Latent space | Wan VAE, **8× spatial / 4× temporal**, 16 channels |
| Tensor shape | `[B, F, 16, 60, 104]` ← RGB `480×832 @ 16 fps` |
| Chunk | `num_frame_per_block = npb = 3` latent frames = 12 RGB frames = 0.75 s |
| Attention | sliding local window, **21 latent frames** (5.25 s) |
| Extra corrector | **CARN** — a per-channel affine (mean/std) applied to each generated chunk at the KV-commit site, `no_grad`, λ=0.5 |

The GAN's assigned job in this division of labour: **CARN corrects the
statistical drift of self-generated context; the GAN is supposed to be the
detector/teacher for texture and style.** CARN is validated. The GAN is not.

---

## 2. Discriminator architecture

`model/ladd_disc.py`. LADD-style: a **frozen pretrained DiT is the backbone**,
small trainable heads read its intermediate features.

### 2.1 Backbone (frozen, zero trainable params)

`WanFeatureProjector` (`ladd_disc.py:75-270`) holds a weak reference to the
**teacher itself** (`real_score`, the frozen bidirectional v14e Wan model) and
registers *temporary* forward hooks on selected transformer blocks for the
duration of one disc forward, removing them in `finally` (so the hundreds of
other teacher forwards per step — DMD scoring, aux passes — don't fire them).

- **Tap indices**: config `ladd_feature_blocks` is empty, so they are auto-chosen
  as `int(n_blocks * f / 5)` for `f ∈ (1,2,3,4,4.95)`
  (`causal_action_forcing_train.py:737-746`). For our 30-block teacher this is
  **blocks [6, 12, 18, 24, 29]** — i.e. 20%, 40%, 60%, 80%, 97% depth. **There is
  no tap earlier than block 6.**
- The teacher's params are frozen; features still carry gradient w.r.t. the
  *input* latent, which is how the generator gets a signal.

### 2.2 Trainable stack (~10–15 M params)

Per tap, in order:

1. **CCM** `LADDChannelMixer` (`:271`) — 1×1 linear, `dim_teacher → ladd_proj_dim = 256`.
2. **CSM** `LADDFeatureFusion` (`:316`) — optional FPN-style top-down fusion
   across taps (`ladd_use_csm=true` in all our arms).
3. **Head** `LADDDiscHead` (`:411`) — `GroupNorm(8) → LeakyReLU(0.2) →
   spectral 1×1 conv → _ResBlock2D(k=3) → spectral conv → 1 channel`.
   **This is the entire spatial receptive field of the trainable part: one 1×1
   and one 3×3 residual block on a 30×52 grid of teacher features.**

### 2.3 Token layout and the reduction to a scalar

`LADDDiscriminator.forward` (`:787-928`). Wan patch size is `(1,2,2)`, so per
latent frame the token grid is `H'×W' = 30×52 = 1560` spatial tokens, plus
per-frame action tokens which are **sliced off** before the 2-D reshape
(`:880-900`).

Head output per tap is `[B·T', 1, 30, 52]`, flattened to `[B, T'·1560]` and
concatenated across 5 taps. Then, with `ladd_scalar_output=true` (all current
arms):

```python
visual_logits = visual_logits.mean(dim=1, keepdim=True)   # ladd_disc.py:911-915
```

For a `gt_transition` input (`T' = 2·npb = 6` frames) this averages
**5 taps × 6 frames × 1560 tokens = 46,800 token logits into ONE scalar per
sample.** That single scalar is what the D loss, the G loss, R1 and R2 all
differentiate.

The stated rationale (`:911-914`) is that D/G/R1/R2 then optimise the same
quantity instead of a resolution-dependent token sum. The consequence for our
problem is that **the critic has no way to say "this patch is fake and that one
is real" — it can only emit a single global realism score per 6-frame clip.**

---

## 3. What is actually discriminated

The active mode in every arm is **`gt_transition`** (`ladd_gt_transition_enabled=true`).

- `npb = 3`, and a "pair" is the chunk-pair `(i, i+1)`, so **each disc sample is
  6 latent frames = 24 RGB frames = 1.5 s** (`causal_action_forcing_train.py:5191-5222`).
- **Real** = ground-truth latents for a chunk-pair. **Fake** = the student's
  chunk-pair, concatenated along `F` (`:5508-5537`).
- So the critic is asked: *"is this 1.5-second latent transition a real one or a
  generated one?"* — a clip-level, whole-frame question, never a patch-level one.

### 3.1 Which student tensor is the fake

The fake side is `flash_dmd_gan_x0` — the student's `x0` prediction taken from
the **flash-DMD pass at `flash_dmd_gan_t = 60`**, i.e. its lowest-noise rung
output (`model/dmd_action_forcing.py:10606-10637`). Not the final committed
rollout frame, and not a multi-rung sample.

### 3.2 Which half receives generator gradient

`ladd_gt_transition_gen_detach_former = true` in all GAN arms: the *former*
chunk of the fake pair is detached, so **the adversarial gradient reaches only
the latter chunk** (`:5523-5536`). Intent: teach "given my drifted context, make
the next chunk clean," without pushing the student to degrade its own context.

### 3.3 Equalisation applied before the critic sees anything

`ladd_gt_transition_mean_equalize` (true in the wavelet/raw arms, false in
strict) rescales real and fake to a **common per-row magnitude level**, so the
critic gets *no absolute-brightness cue* (`:5540-5560`). With
`xeq_preserve_delta=true` the within-pair brightness *ratio* survives but the
absolute level does not. CARN-former variants additionally restore the former's
per-channel DC and mean-magnitude so only HF structure changes (`:5490-5502`).

**Net: several of the cues that most directly encode "looks like real film" —
absolute level, per-channel DC — are deliberately removed before the critic.**

---

## 4. Noise level of the discriminator input

`causal_action_forcing_train.py:5700-5790`. Both real and fake always get the
*same* timestep, but which timestep depends on the arm:

| condition | `disc_t_int` |
|---|---|
| `ladd_wavelet_hf_enabled=true` (wavelet arms) | **forced 0** (clean latents) |
| `ladd_disc_force_clean=true` | forced 0 |
| otherwise, flash-DMD on | `flash_dmd_gan_t = 60` |
| `ladd_disc_sample_t=true` | one scalar `t ~ U[20, 980]` re-mapped by shift 5.0, shared by real+fake |

The wavelet branch *forces* `t=0` because noising would leak broadband energy
into the HF sub-bands. That coupling means **"wavelet on/off" and "disc timestep"
were never varied independently** until the 2×2 we are running now. APT reports
`t=0` diffusion features are a poor discrimination basis, so part of what we have
attributed to the wavelet may be the timestep.

---

## 5. Where the real samples come from (likely central to the failure)

Per D-update, each fake needs `Kk = ladd_gt_transition_match_k = 4` real
partners. They are supplied by two mechanisms:

1. **Nearest-neighbour matching within the current ride.** Candidates are scored
   by **mean L1 distance in latent space** (`torch.cdist(p=1)/D`,
   `:6217-6244`), the `M = ladd_gt_transition_match_pool = 8` nearest are kept,
   and `Kk` are sampled from those 8 per update.
2. **Cross-ride replay ring.** `ladd_real_pool_cross_ride = 4096`,
   `push_per_ride = 8`. When active, **half** the slots (`Kh = Kk//2 = 2`) are
   drawn **uniformly** from the ring and the other 2 from the top-8 local pool
   (`:6264-6280`). The matched-unique cap is halved to keep memory constant.

Consequences to weigh:

- The local half is chosen to be the **L1-nearest real latents to the student's
  own output**. That is by construction the subset of reality that *most
  resembles what the student already produces* — the least informative reals for
  a texture critic, and a direct route to a degenerate support.
- `ladd_gt_transition_match_max_real = 12` caps distinct reals forwarded per
  update; on collision a fake's pick is remapped onto one of its own nearer
  picks, so effective diversity can fall below 12.
- **[UNVERIFIED]** There is **no telemetry on the ring** — no logged ring length,
  no unique-real count. The only pool quantity logged is `match_pool_m = 8`. We
  therefore *cannot currently confirm the 4096 cross-ride ring ever populates or
  is sampled from.* This should be treated as an open question, not an assumption.

---

## 6. Conditioning

- **Action conditioning. CORRECTION 2026-08-23 — this knob has NEVER had any
  effect. Every GAN arm ever run was ACTION-CONDITIONED.**
  `ladd_gt_transition_action_blind` is read only as
  `getattr(self.model, "ladd_gt_transition_action_blind", False)`
  (`trainer/causal_action_forcing_train.py`, 2 sites), but
  `ActionForcingDMD.__init__` **never parses it from `args`** — unlike every
  sibling knob (e.g. `ladd_gt_transition_match` at
  `model/dmd_action_forcing.py:2022`). There is no `__getattr__` passthrough and
  no `setattr` loop, so the attribute does not exist and the `getattr` default
  `False` is returned on every call.
  Consequence: the wavelet/raw arms set `...action_blind=true` in their DEXTRA
  and ran **action-conditioned anyway**. The "action-blind vs strict" contrast
  reported for `wave01` / `horizon_wave90` vs `strict03` **was never realised** —
  those arms differed in the wavelet stage, weight, D-steps and disc timestep,
  but NOT in action conditioning. Any conclusion resting on that contrast is
  void. (Root cause is generic: the trainer parses `--override` with
  `OmegaConf.from_dotlist` + `merge`, which silently accepts unknown keys — no
  allowlist, no typo check.)
- **Prompt conditioning.** `ladd_use_prompt_cond=false`, `ladd_cmap_dim=0` in all
  current arms, so the StyleGAN-T-style projection-discriminator inner product is
  disabled and the head's `cls` conv emits a plain 1-channel logit
  (`ladd_disc.py:437-441`).
- **Projector mixing frozen.** `ladd_freeze_projector_mixing=true` — the CCM
  mixing layers are frozen while the heads train, so the critic cannot cheaply
  rewrite the teacher feature basis that supplies its perceptual signal.

---

## 7. Losses

**RpGAN (relativistic, logistic)** — `model/r3gan.py:231-248`:

```
D: E[softplus(D(fake) - D(real))]
G: E[softplus(D(real) - D(fake))]     # d_real detached at the call site
```

Position-matched by default; `*_allpairs` variants (every real vs every fake,
`r3gan.py:251-283`) exist but are **off** in our arms.

**Gradient penalties.** Zero-centred, on perturbed inputs (finite-difference
style), `γ_R1 = γ_R2 = 1.0`, `ladd_r1_num_samples=6`,
`ladd_r1_normalize_tokens=false`, `σ=0.01`. R1 fires on a **debt-based** cadence
(fires once ≥ `every_n` steps have elapsed since the last actual firing); R2
fires on an **exact modulo** cadence `(step - offset) % every_n == 0` with
`every_n=2, offset=1`, intended to be disjoint from R1 as an OOM guard.

> **RESOLVED 2026-08-23 — R2 does fire; the gauge was a sampling alias.**
> `r3gan_r2_fired` read **0 on every logged row of every historical run**, because
> wandb samples every ~10 steps and that cadence aliases with both the R2 parity
> cadence and the generator-update cadence (`dfake_gen_update_ratio=5`). Monotone
> counters added 2026-08-23 now report `r3gan_r2_fired_total = 85` over `175`
> disc updates — a **48.6 %** fire rate. **"We ran R1+R2" is TRUE.**
>
> **However the counters expose an unintended IMBALANCE.** In the `gt_transition`
> arms **R1 fires at 0.20 while R2 fires at 0.49** — the fake-side penalty applied
> **2.4x more often** than the real-side one, because `ladd_r1_once_per_step=true`
> caps R1 at the first of `gan_updates_per_step=5` updates while R2 has no cap.
> A `gt_vs_fake` arm is balanced (0.53/0.47), so this is specific to that flag
> combination. **R3GAN's stability argument assumes balanced R1+R2**, so the
> effective regularisation in every 5-D-step arm was asymmetric in a way nobody
> intended.

> **DEFECT 2026-08-23 — diff-aug asymmetry in the matched path.** The D-update
> augments the REAL side only (`latent_diff_augment(_rn, _rn, policy=flip)`) and
> passes the fake through un-augmented; the G-side augments both but with
> DIFFERENT seeds. Affects every `ladd_gt_transition_match=true` arm.
> **Adversarially reviewed — the consequence that matters is NOT mirror parity**
> (which is faint after scalar mean-pooling and largely self-cancels, since the
> G-side fake IS flipped 50% of the time). It is: **(1)** the real latent is
> mirrored while its ACTION TOKENS are not, so ~50% of reals are
> action-image-inconsistent while 100% of fakes are consistent — the critic can
> learn "image contradicts the steering => real", which is directly adversarial
> to an action-conditioned critic; and **(2)** D is trained on never-flipped
> fakes but queried on 50%-flipped fakes on the G side. See `ROLLING_CAMPAIGN.md`.

**Optimiser / schedule.** `gan_lr` 5e-6 (wavelet/raw arms) or 1e-5 (strict),
betas `[0.0, 0.9]`, `gan_max_grad_norm=10`, `gan_disc_start_step=20`,
`gan_critic_warmup_steps` 20–40, `gan_warmup_steps=25` linear ramp on the G-side
weight, `gan_updates_per_step` 1 (wavelet/raw) or 5 (strict),
`ladd_disc_micro_batch_groups=2`, DiffAugment policy `flip` only.

---

## 8. Variants that exist

- **Wavelet-HF** (`model/wavelet_hf.py`). Single-level **Haar SWT** (undecimated,
  so resolution is preserved) on the 16-channel latent → LL/LH/HL/HH = 64
  channels → a learned **1×1 conv adapter back to 16 channels** so the teacher
  projector sees an in-distribution shape. `drop_ll=true` in our arms (HF only).
  Init gain 0.1. Rationale is WGSR's "disc sees only HF" — `L1-on-LL ~ DMD`,
  `GAN-on-HF ~ detail`. Forces `disc_t = 0`.
- **Projected-GAN** variant — sampled disc-t, no wavelet. Queued, never scored.
- **Stat head** `LADDStatHead` (`ladd_disc.py:482-607`) — a parallel critic on
  per-frame / per-channel **stds** of the raw latent, avg-pooled to 4×4, emitting
  one extra scalar. **Disabled** (`ladd_stat_head_enabled=false`) in all arms.
- **MomentDiscriminator** (`:1186`) — exists, unused.

---

## 9. Exactly what was run, and what happened

All arms: fresh from the same KL-ODE init, 200 steps, 2 nodes × 4 GPUs, then an
identical fixed-route 60-second evaluation (same Madrid ride, seed 42, 4
denoising steps, 3 real seed chunks, 100 generated chunks, inference-CARN λ0.5).

| arm | GAN | weight | D-steps | disc t | action | outcome |
|---|---|---|---|---|---|---|
| `fullcarn_bidir_kl_nogan200` | off | — | — | — | — | melts ≈8 s |
| `fullcarn_bidir_kl_wave01` | wavelet | 0.01 | 1 | 0 | ~~blind~~ **cond.** | holds ≈12 s, then melts |
| `horizon_nogan90` | off | — | — | — | — | melts ≈12–16 s |
| `horizon_wave90` | wavelet | 0.01 | 1 | 0 | ~~blind~~ **cond.** | holds ≈16 s, then **scanline banding / total texture death** |
| `fullcarn_bidir_kl_strict03` | raw | 0.03 | 5 | sampled | conditioned | eval pending |

**Critic health (from W&B history, not summaries):**

| | `wave01` | `horizon_wave90` | `strict03` (@step 55) |
|---|---|---|---|
| `d_loss` start → end | 0.659 → 0.590 | 0.657 → 0.584 | 0.692 → 0.589 |
| weighted G-term vs median DMD | 0.0072 / 0.61 ≈ **1.2 %** | 0.0073 / 1.20 ≈ **0.6 %** | ≈ **10–20 %** |

`ln 2 = 0.693` is chance. Sustained `d_loss < 0.10` is our measured
generator-breakdown threshold. **No arm ever came close to breakdown; every
wavelet critic stayed near chance and applied ~1 % of the DMD gradient.**

**The qualitative result that matters most.** Judged on the videos by the
researcher (whose visual judgement has repeatedly out-diagnosed our metrics):
the no-GAN arm is *better* than the GAN arm, because although the GAN arm holds
scene structure a few seconds longer, **it ends in horizontal scanline banding —
a complete texture collapse — rather than the no-GAN arm's ridge-melt.** A critic
applying 0.6 % of the gradient producing a failure mode that specific implies the
gradient is *pointed the wrong way*, not that it is too strong.

**Not yet measured (this is the instrument gap):** `‖g_GAN‖ / ‖g_DMD‖` and
`cos(g_GAN, g_DMD)` at a shared late generator block. Loss ratios are not update
shares, and only the cosine separates "small and irrelevant" from "small and
destructive". This is being built now.

---

## 10. Prior findings that constrain any proposal

- **R1 γ = 1e6 pinned every discriminator at `ln 2`.** Replicated across many
  runs; γ=0 overshoots (disc dominates). Current γ=1.0 is the resolved value.
- **Real-pool memorisation** (a small ride-local pool) was previously identified
  as the root cause of disc domination — the cross-ride ring was the response.
- **GAN weight bracket, measured:** 0.01 inert / 0.03 learns / 1.0 destroys.
- An earlier "wavelet kills the GAN" verdict was later attributed to a `t=0`
  feature artefact, which is exactly the confound §4 describes.
- The student's own failure signature is **high-frequency *gain*** — end-of-rollout
  Laplacian variance is **3.6× the real seed's** — i.e. it is not hedging/blurring,
  it is over-committing to fabricated texture. The 14e paper's quality battery,
  built to detect blur/haze, scores this "clean".

---

## 11. Questions for the research agent

1. **Is a single global scalar per 1.5-second clip a viable objective for a
   texture critic at all?** Texture realism is a local, high-spatial-frequency,
   stationary property. Our critic averages 46,800 token logits into one number
   before any loss touches it. What is the right granularity — per-token/patch
   logits (PatchGAN), multi-scale patch critics, or a patch-statistics objective?
2. **Is a frozen diffusion-DiT's block-6-to-29 features the right basis for
   texture?** Those taps are semantic/structural. Should the critic instead read
   very early blocks, the patch-embedding output, or bypass the DiT and operate
   directly on latents or decoded RGB?
3. **Does discriminating in an 8×-downsampled 16-channel VAE latent make
   fine-grained texture discrimination possible in principle?** The scanline
   banding suggests the critic is shaping a latent-space frequency artefact that
   the decoder turns into structured garbage. Should the critic (or an auxiliary
   one) see decoded pixels?
4. **Is the nearest-L1 real matching actively harmful?** Selecting the reals most
   similar to the student's current output is the minimum-information choice for a
   texture critic. What should the retrieval objective be — action-matched,
   scene-matched, or deliberately diverse?
5. **Have we destroyed the signal by equalisation?** Mean/cross equalisation
   removes absolute level and per-channel DC before the critic. Is that
   over-sanitised?
6. **Non-adversarial alternatives.** Given the failure is a *measured, directional*
   spectral drift (high-frequency energy rising to 3.6× the seed), would direct
   feature-statistic matching — Gram/covariance, radial power-spectrum, or
   kurtosis matching against real latents — dominate an adversarial critic here?
7. **Does the transition framing help or hurt?** We discriminate 2-chunk
   transitions to target temporal coherence. Does that dilute the texture signal
   relative to a per-frame texture critic plus a separate temporal term?
