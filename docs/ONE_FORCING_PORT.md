# ONE-FORCING PORT — GAN on the fake score (Option D)

**Status:** spec, 2026-08-24. Branch `gan-work-p3`.
**Source:** One-Forcing (arXiv 2605.23458), code cloned at
`ARRWM_data/one_forcing`. Paper text extracted; every claim below was read
off their shipped code, not the prose.

## Why this arm exists

`GAN_REDESIGN_TWO.md` (binding) says every latent-GAN failure traces to
projecting the disc onto **`real_score` — the DMD teacher's own weights**,
and that the fix is decoupling. One-Forcing decouples a different way than
options A/B/C: it puts the disc on the **trainable `fake_score` critic**.
This obeys the binding rule (the disc is NOT built from `real_score`) and
adds a property none of A/B/C have: the disc backbone is *already being
trained every step* to denoise the current student's samples, so its
feature basis tracks the student's distribution instead of being frozen.

Call it **Option D**. It does not replace A/B/C; it is the faithful port
the researcher asked for, to be measured against them.

## What we got wrong vs. what they do (the five divergences)

| | ours @ a8639f8 | One-Forcing |
|---|---|---|
| disc host | frozen `real_score`, heads-only (CCM/CSM frozen too → ~5 SpectralConv heads) | **trainable `fake_score`**, whole 1.3B backbone gets adversarial grad |
| readout | `ladd_scalar_output` mean-pools 46,800 token logits → 1 scalar pre-softplus | learned register-token cross-attn pooling per tap |
| fake sample | `flash_dmd_gan_t=60` slab — a **different sub-graph** from DMD | the **same `pred_image`** DMD scores |
| pairing | nearest-GT L1 match, block-diagonal RpGAN | plain unmatched real vs fake, non-saturating softplus |
| aux task | none — disc features never refreshed | disc shares the backbone with the denoising critic loss |

Their Fig. 4 is the warning that binds our pairing choice: a disc whose
real/fake sides are near-identical collapses to logit gap ≈ 0. Nearest-GT
matching *engineers* that condition; unmatched real data holds μ≈1.5.

## What is already in our tree

The register-token head is ALREADY shipped (Self-Forcing ancestry):
- `utils/wan_wrapper.py:327` `adding_cls_branch(...)` builds
  `_cls_pred_branch` / `_register_tokens` / `_gan_ca_blocks`.
- `wan/modules/model.py:789-848` `classify_mode` runs the taps and head;
  `RegisterTokens:489`, `GanAttentionBlock:362`, `WanGanCrossAttention:198`.
- `utils/wan_wrapper.py:692-700` forwards `classify_mode` end-to-end.
- `fake_score` is the **bidirectional** `WanModel` (`model/base.py:229`), so
  this path is directly usable on it.

Two mismatches vs the paper that the port must fix:
1. tap layers are **hardcoded** `[7,13,21,29]` at `wan/modules/model.py:825`
   → must become configurable (paper framewise uses `[21,29]`).
2. our head is heavier (2 blocks/token, 4-layer residual MLP, dropout 0.2)
   → must be configurable, defaulting to the paper's shape.

## Flags (all `gan_of_` — inside `_OVERRIDE_GUARD_PREFIXES` via `gan_`)

Every flag read as `getattr(args, ...)` with the OFF value as default, so a
config predating this arm is byte-identical.

| flag | default | meaning |
|---|---|---|
| `gan_of_enabled` | `false` | master gate. **Hard-raise if `gan_enabled` (LADD) is also true** — one GAN at a time |
| `gan_of_g_weight` | `0.03` | λ_G |
| `gan_of_d_weight` | `0.03` | λ_D |
| `gan_of_feature_layers` | `[21,29]` | tap blocks |
| `gan_of_blocks_per_token` | `1` | GanAttentionBlocks per register token |
| `gan_of_block_ffn_dim` | `2048` | paper framewise |
| `gan_of_block_num_heads` | `12` | |
| `gan_of_head_hidden_dim` | `1536` | |
| `gan_of_head_num_layers` | `1` | **1** = paper's LN/Linear/SiLU/Linear; `>=2` adds ResidualMLPBlocks + a 2nd LayerNorm |
| `gan_of_head_dropout` | `0.0` | paper has none |
| `gan_of_t_min` / `gan_of_t_max` | `20` / `980` | critic timestep range |
| `gan_of_timestep_shift` | `5.0` | SD3 shift on the sampled t |
| `gan_of_relativistic` | `false` | paper framewise = plain non-saturating |
| `gan_of_shared_noise` | `true` | real and fake share ε in the D loss |
| `gan_of_r1_weight` / `gan_of_r2_weight` | `0.0` / `0.0` | paper framewise has none |
| `gan_of_r1_sigma` / `gan_of_r2_sigma` | `0.01` / `0.01` | FD scale |
| `gan_of_disc_start_step` | `0` | |
| `gan_of_warmup_steps` | `0` | |
| `gan_of_fake_source` | `"pred_image"` | **layer-on knob**: `"flash"` uses the `flash_dmd_gan_t` slab |
| `gan_of_real_source` | `"aligned_gt"` | **layer-on knob**: `"nearest_match"` reuses LADD matching |
| `gan_of_telemetry_every` | `25` | logit gap / grad-ratio cadence |

One non-`gan_of_` flag belongs to this arm's blast radius:

| flag | default | meaning |
|---|---|---|
| `boundary_vae_roundtrip_keep_graph` | `false` | keeps the VAE decode/encode under `no_grad` but moves the replacement `cat` OUT of that block, so the rolled chunk keeps its `grad_fn`. Default `false` reproduces the measured (severed) behaviour bit for bit. See §6. |

`fake_source` and `real_source` exist so our two ideas can be tested
**separately, later** — the initial arm runs both at the faithful default.

## Losses (exact, from their `model/one_forcing.py`)

D (in the critic step, `:275-376`), shared noise, both members one batch:
```
gan_d = softplus(-d_real).mean() + softplus(d_fake).mean()      # :333
```
G (in the generator step, `:208-273`):
```
gan_g = softplus(-d_fake).mean() * gan_of_g_weight              # :249
```
Relativistic variants (`:266`, `:336`) exist behind `gan_of_relativistic`.
R1/R2 are finite-difference on the **real**/**fake** noised latents with
`0.5*` multipliers (`:339-368`) — off by default.

## Wiring contract

1. **Construction.** When `gan_of_enabled`, call the parameterized
   `adding_cls_branch` on `fake_score` **before** DDP wrap and before the
   `fake_optimizer` is built, so head params land in both.
2. **Critic step (every iter).** Compute `gan_of_d_loss` on
   `(noised real GT window, noised detached rollout)` at a freshly sampled
   shared `t`; add to the existing denoising `critic_loss`; **one backward,
   one `fake_optimizer.step()`** — the backbone therefore receives denoising
   AND adversarial gradient together. This is the whole point of the arm.
3. **Generator step (every `dfake_gen_update_ratio`=5).** Compute
   `gan_of_g_loss` on `pred_image` — the *same tensor* DMD scores — and add
   into `generator_loss` before the single existing `.backward()`.
4. **No critic grads on the generator step.** The G-adv disc forward must
   not train the disc. Set `fake_score.requires_grad_(False)` around it and
   restore after (their `set_discriminator_requires_grad`, `:92`), AND
   **call the unwrapped module, not the DDP wrapper** — a DDP forward whose
   backward produces no param grads is a multi-node hang/error risk. Use
   `fake_score._unwrapped_model()`-style access for this forward only.
5. **Checkpoints.** `_cls_pred_branch` / `_register_tokens` /
   `_gan_ca_blocks` params must save AND restore; a resumed arm must not
   silently start from a fresh disc.
6. **Streaming path** (`streaming_mode: true` — the base default, and
   what every real phase-3 DMD arm runs). **CORRECTED 2026-08-24 after
   the first GPU smoke (`logs/of_smoke_r2.log`) falsified the original
   text.** What follows is what the code does, traced on the GPU.

   *What the original spec got wrong.* It said the G term attaches to
   `train_chunk`, "the object passed to `compute_generator_loss_streaming`,
   i.e. the root of the DMD scoring graph (`score_image = chunk` …)".
   Two errors, one fatal:

   * `train_chunk` is the **root** of the scoring graph, not the scored
     tensor. `compute_generator_loss_streaming` converts it into
     `score_image` — `f42["noisy_x"]` on the 42f path, the asymmetric
     builder's `noisy_x` on the asym path, `chunk` itself only on the
     legacy path — and hands *that* to
     `compute_distribution_matching_loss`. In the 42f geometry
     `score_image` is 21 frames of which only the supervised band carries
     a graph; the rest is GT scaffold and detached student context.
   * Under the smoke's own recipe (`boundary_vae_roundtrip=true`)
     `train_chunk` carries **no graph at all** on every roll with overlap.
     `generate_next_chunk` rebuilt the chunk with a `torch.cat` executed
     *inside* the `no_grad` block wrapping the boundary VAE round-trip, so
     the whole rolled chunk came back with no `grad_fn`. All 8 ranks
     raised out of `_of_g_grad`'s fail-loud guard on roll 2 — which is the
     only reason this was ever noticed.

   *Collateral finding, bigger than the arm.* That same severing made the
   **streaming DMD generator loss a constant on every roll after the
   first** in any run with `boundary_vae_roundtrip: true`. With
   `dmd_supervise_roll_mode=random` (target drawn uniformly in
   `[1, max_rolls]`) most rides then received **zero** generator gradient.
   Nothing reported it: `_phase_lora_ghost_anchor` folds a live
   `0.0 * ghost` into `generator_loss`, so `generator_loss.requires_grad`
   stayed `True` and the DDP-lockstep `gen_backward_skipped` gauge read
   healthy. The only fingerprint in the logs was the debug line
   `[42F-ROLLING] … (graph-on=False)`; runs with the flag off print
   `graph-on=True` at identical geometry. Fixed behind
   `boundary_vae_roundtrip_keep_graph` (**default `false` = byte-identical
   to the measured behaviour**, because flipping it is a training-recipe
   change and needs sign-off). `sbatch/run_of_smoke.sh` sets it `true`.
   New telemetry `dmd_sup_band_graph_on` makes the condition a logged
   gauge instead of a debug print.

   *The corrected attach point.* Both terms are built from ONE
   (fake, real, cond) triple, resolved once per roll by
   `ActionForcingDMD._of_publish_streaming_band`, called from **inside**
   `compute_generator_loss_streaming` immediately after the DMD scorer —
   the only scope in the process that holds `score_image`.

   * **Fake**: `score_image[:, lo:hi]`, where `[lo, hi)` is the contiguous
     True span of `score_grad_mask` — the same mask
     `compute_distribution_matching_loss` multiplies its gradient by.
     Read off the mask rather than re-derived from
     `n_ctx`/`sup_offset`/`sup_span`, because that arithmetic already
     exists in three builders with three geometries. Empty or
     non-contiguous ⇒ raise.
   * **Real**: `score_gt_target[:, lo:hi]`. Frame-locked **by
     construction**: every builder makes its `gt_target` the GT
     counterpart of its own `noisy_x`, position for position (42f:
     `ride_lat[noisy_lo+m : noisy_hi+m]`), so the band slice of one pairs
     with the band slice of the other. Length mismatch ⇒ raise. The old
     `chunk_lo`/`chunk_hi` ride-window slice is **no longer used on this
     path** — it was the 18-frame rollout window, a different geometry
     from the 21-frame scoring window.
   * **Cond**: the scorer's own `score_cond`, stripped of the
     teacher-forcing `_clean` streams and sliced to `[lo, hi)` with the
     same `_slice_per_frame_streams` the scorers use.
   * **Why not the whole `score_image`.** It is literally the tensor DMD
     scores, but in the 42f geometry 12 of its 21 frames are GT — real and
     fake would be identical over most of the window, which is exactly the
     Fig-4 condition that collapses the logit gap to 0. The band is the
     largest sub-window on which the two objectives share a graph *and*
     the pair stays informative.
   * **G**: `compute_of_g_loss` runs inside the publisher, so the disc
     forward, the `_of_disc_frozen` window and the
     `torch.autograd.grad(g_loss, fake)` all sit beside the DMD loss. The
     trainer's `_of_streaming_g_term` now only reads the stash, publishes
     the logs and the grad telemetry, and adds the term into
     `generator_loss` before the single
     `generator_loss.backward(retain_graph=True)`.
   * **D**: `_of_streaming_d_term` reads the **same** stash and detaches
     the **same** fake, then folds into the tensor returned by
     `compute_critic_loss_streaming` before its single `.backward()`, and
     into each extra inner critic loss when
     `streaming_fake_updates_per_gen > 0`. Folded OUTSIDE the model method
     on purpose: that method has a PER-RANK `gradient_mask.any()` early
     return, and a disc forward behind it would desynchronise the
     fake_score reducer. D and G can no longer disagree about what "the
     fake" is, because there is one resolution and two consumers.
   * **`gan_of_fake_source="flash"` is reachable here and only here** —
     the `flash_dmd_gan_t` slab is published only by the streaming
     rollout, is a deliberately DIFFERENT sub-graph, and carries the
     rollout geometry rather than the scoring window's, so on that branch
     the real is `ride_latents_window[chunk_lo : chunk_lo + slab_frames]`
     via `of_streaming_real` (length-asserted) and the cond is
     `of_streaming_cond`. The trainer refuses `flash` at construction
     under `streaming_mode=false` or `flash_dmd_enabled=false`.
   * **`dmd_only_first_chunk_per_ride` is refused at construction.** It
     early-returns out of `compute_generator_loss_streaming` on rolls
     2..N, which is the one path that would leave the band unpublished.
     `dmd_supervise_roll_mode` is the supported way to supervise a subset
     of rolls: it skips the SCORER but still builds the band.
   * **Firing rule — G fires exactly when the DMD scorer fires.**
     `dmd_fired = not _skip_scorer`, whose operands are the rank-0
     broadcast supervise target, the MIN-reduced roll cap and the lockstep
     ride-depth counter — so the decision is identical on every rank. On a
     roll the DMD scorer skips, the generator gets no
     distribution-matching gradient at all, and an adversarial push there
     would make the two objectives shape the student on *different* rolls,
     which is the opposite of the co-occurrence this arm exists to test.
     (The G weight's own ramp is still `gan_of_disc_start_step` /
     `gan_of_warmup_steps`; it is deliberately NOT coupled to
     `dmd_loss_start_step`.) **The D term is not gated this way**: the
     critic trains on every roll as it always has, on the same band —
     and it must, because the disc head is reached by exactly one forward
     and skipping it would leave head parameters ungradiented in the
     `fake_score` reducer (see `of_head_touch`).
   * **Rank uniformity**: every gate is a pure function of (config, global
     step, rank-uniform broadcast) — `gan_of_enabled`,
     `of_weight_at_step(..., self.step, ...)`, and `dmd_fired`. Nothing
     reads a per-rank quantity: not the ride length, not
     `gradient_mask.any()`, not `avg_mae`. The only thing that can skip
     the whole block is the MAX-reduced toothpaste GONE flag, which
     already skips Stage 4 on every rank together.
   * **Proof, not assertion.** `testing/test_one_forcing_gan.py`
     ::`TestAdversarialGradientReachesTheGeneratorThroughTheDMDGraph`
     builds a real `nn.Module` generator, assembles the 42f rolling
     geometry around its output, publishes through the real publisher,
     runs the real `compute_of_g_loss` against a checkpointed stub disc,
     and then asks autograd: the G term's gradient at the generator
     parameters is non-zero; the fake differentiates back into the very
     `score_image` object the scorer was handed, on the band frames and no
     others; a detached band raises instead of adding a constant. The
     previous, confidently wrong claim was source-level only, which is
     precisely why no test caught it.

7. **Telemetry.** `of_d_loss`, `of_g_loss`, `of_d_real`, `of_d_fake`,
   `of_logit_gap` (= |d_real − d_fake|, the paper's Fig-4 health metric),
   `of_gan_grad_norm`, and the GAN:DMD grad ratio/cos on `pred_image`.
   **The logit gap is the primary go/no-go instrument for this arm.**

## Standing rules this arm must obey

- `real_guidance_scale=0.0` (rg3 collapses the student).
- TF head only: `dmd_tf_head_weight=1.0`, `dmd_ar_head_weight=0.0`.
- Stat anchor: **the arm runs it at 1.0, deviating from the "retired,
  default off" rule, deliberately and on the record.** One-Forcing has no
  such term, so a faithful port would set it to 0. But `_roll_holder.sh`
  records that removing it collapses the run to black by ~step 61, and
  every recent working arm ran 1.0. At 0 we could not tell "the GAN did
  nothing" apart from "the recipe collapsed", and the GAN would be
  measured against a different baseline than every arm we want to compare
  it to. Revisit only once the arm is known to train.
- Never noise the clean context; no `context_aug`.
- `fake_alt_head_enabled=false` — it flips the fake_score reducer to
  `find_unused_parameters=True`, which DDP does not support with the two
  grad-on fake_score forwards (denoise + disc) the OF critic step runs
  before its single backward. Refused at construction.
- Flag-gated, default-off, byte-identical when off; adversarial review
  before launch; smokes on 2 nodes.
- `streaming_mode=true` is REQUIRED and refused at construction otherwise.
  The D term is folded into the streaming critic loss and has no
  non-streaming call site; the G half exists on `_fwdbwd_one_step` but is
  unreachable by construction, and completing that wiring means adding a
  D twin there BEFORE dropping the refusal.
- The disc head is reached by exactly one forward (`classify_mode`), so
  whenever the D loss is inactive — `gan_of_disc_start_step > 0`,
  `gan_of_d_weight = 0` — it must still be tied into the critic backward
  or the `fake_score` DDP reducer (`find_unused_parameters=False`)
  raises "Expected to have finished reduction" on the next step.
  `ActionForcingDMD.of_head_touch()` is that tie: an exactly-zero scalar
  summing every head parameter, folded into the D term unconditionally.
  Conversely a classify-ONLY backward leaves exactly three params
  ungradiented (`head.modulation`, `head.head.weight`, `head.head.bias`),
  which is safe only because the D term always rides the denoising
  `critic_loss` — a third reason the term must never be decoupled.
