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

## The confound, and the A/B that resolves it

**Stated plainly because as first shipped this arm moved two variables at
once.** Putting the disc on `fake_score` changes:

* **(a) the feature source** — the disc reads features from the
  *trainable* `fake_score` critic rather than the frozen `real_score`
  teacher. Even with no adversarial gradient reaching it, that backbone
  is retrained every step by its **denoising** loss, so the disc's
  feature basis tracks the current student's distribution instead of
  being frozen. This alone is a real change from options A/B/C.
* **(b) adversarial co-training** — the same backbone additionally
  receives the adversarial D gradient, in the same backward, from the
  same optimizer step.

The port's thesis is that **(b)** is what makes One-Forcing work. With
both moving together, a *positive* result cannot say whether the win came
from (a) or (b), and a *negative* result cannot say whether (b) is
useless or whether (a) poisoned it. That is a confounded experiment, and
the flag below exists to un-confound it.

| arm | `gan_of_backbone_trainable` | (a) disc taps `fake_score` | (b) backbone co-trains on the adversarial loss |
|---|---|---|---|
| **D** (faithful paper setting, the launch arm) | `true` (default) | yes | **yes** |
| **D-heads** (control) | `false` | yes | **no** |

`false` is **"adversarial gradient reaches the head modules only"**, NOT
"backbone frozen". The `fake_score` backbone is still trained every step
by its own denoising loss — freezing it globally would destroy the DMD
critic and change a different variable again. Read the difference between
the two arms as *the effect of adversarial co-training, holding the
feature source fixed*.

What the control does **not** isolate, recorded so the result is not
over-read: it does not make the disc's features static. Under `false` the
tap features still drift, because the backbone still moves under the
denoising loss. Isolating *that* would be a third arm (a frozen feature
host), which is what options A/B/C already are.

**Implementation** — `ActionForcingDMD._of_head_only_surrogate`, applied
at the very end of `compute_of_d_loss`, after telemetry. The D term keeps
its exact VALUE and its exact logged numbers; only its autograd edges
change. It is built as
`torch.autograd.grad(d_total, head_params)` folded into the
`_of_g_surrogate` construction `(p*g).sum() - .detach() + value`, i.e.
the same "consume the gradient here, hand the caller a surrogate" pattern
the G side already uses. Deliberately **not** `torch.no_grad()` (which
would sever the input→feature path the G side needs), **not**
`requires_grad_(False)` around a forward whose backward happens later
(checkpoint replay reads those flags — that is the `CheckpointError` /
silently-wrong-gradient bug in `_of_disc_frozen`'s docstring), and **not**
a detach at the feature tap (incomplete: with `concat_time_embeddings`
the head also consumes the backbone's `time_embedding`). Restricting by
*parameter* covers every path into every non-head parameter.

Guarantees: after the caller's backward the adversarial contribution to
`.grad` is exactly zero on every `fake_score` parameter outside
`_cls_pred_branch` / `_register_tokens` / `_gan_ca_blocks`, and
bit-identical to the `true` arm on every head parameter. The G side is
unchanged in both arms (it already ran inside `_of_disc_frozen` and
bounded its gradient at the fake tensor, so it never moved a `fake_score`
weight). Every trainable head parameter still gets an autograd edge, so
the `find_unused_parameters=False` reducer on `fake_score` is satisfied
exactly as `of_head_touch` promises, and the backbone is still gradiented
by the denoising loss that shares the backward.

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
| `gan_of_backbone_trainable` | `true` | **experimental-design control, not a paper knob.** `true` = faithful One-Forcing = the arm as launched (one backward carries denoising AND adversarial gradient into the `fake_score` backbone). `false` = the disc still TAPS `fake_score` but the adversarial gradient reaches the head modules only. See §"The confound" |
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
     teacher-forcing `_clean` streams, **detached**, and sliced to
     `[lo, hi)` with the same `_slice_per_frame_streams` the scorers use.
     **CORRECTED 2026-08-24 after the third GPU smoke
     (`logs/of_smoke_r3.log`).** The detach was missing and it killed the
     run: `score_cond` is `build_action_conditional`'s output, i.e. the
     live `action_projection` / `action_token_projection` subgraph
     (`model/base.py` does `action_projection.requires_grad_(True)`
     unconditionally), so every disc forward hung off the GENERATOR's
     graph through its conditioning input. `critic_loss.backward()` (no
     `retain_graph`) freed that shared subgraph, and the first
     `streaming_fake_updates_per_gen` inner `_extra_loss.backward()`
     re-traversed it — "Trying to backward through the graph a second
     time", all 8 ranks. Note what was NOT the cause: the D term was
     already rebuilt per inner iteration with a fresh disc forward and a
     fresh (t, eps), and its fake and real were already detached. The
     poison was an INPUT, so recomputing the term could not have helped.
     Second, quieter consequence: it pushed DISCRIMINATOR gradient into
     `action_projection` — a generator-owned parameter applied by the
     GENERATOR's optimizer, wrong-signed, with nothing in the trace to
     show it. Fixed in `_of_disc_cond`, the single choke point all four
     disc call sites funnel through (both publish branches,
     `compute_of_d_loss`, `compute_of_g_loss`); the `flash` branch had
     always detached via `of_streaming_cond` and `pred_image` had not,
     and that asymmetry *is* the bug. `_of_assert_cond_detached` now
     raises at publish time if the detach is ever removed. The G side is
     unchanged in value — `_of_g_grad` already bounded the adversarial
     gradient at the fake tensor, so no cond gradient was materialised
     there; the detach only makes that boundary structural.
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
     `streaming_fake_updates_per_gen > 0`. **Every one of the
     `1 + streaming_fake_updates_per_gen` critic updates therefore
     carries a live D term** — a fresh disc forward at a fresh (t, eps)
     per update, which is what "the disc co-trains on every
     `fake_optimizer.step()`" means and is the arm's thesis. The N+1
     sequential backwards are legal only because every input to that
     forward is graph-free (fake, real AND cond); `retain_graph=True` is
     the wrong repair for any recurrence — it would pin the whole
     generator graph alive across five backwards. Proven executably by
     `testing/test_one_forcing_gan.py`
     ::`TestEveryCriticBackwardCarriesALiveDTerm`, which runs the real
     publisher / real `_of_streaming_d_term` / real `compute_of_d_loss`
     against an `nn.Module` disc for 1 + 4 sequential backwards, asserts
     a non-zero disc-head gradient and a distinct logit gap on each, and
     carries a mutation control that reverts ONLY the cond detach and
     reproduces the exact GPU `RuntimeError`. Folded OUTSIDE the model method
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

## S6 — disc-forward memory

**Measured 2026-08-25, CPU instrumentation on the real modules
(`torch.autograd.graph.saved_tensors_hooks`) plus exact arithmetic.**
Recorded because the review's sizing was wrong in BOTH directions and
the corrected numbers change what is worth doing.

### The geometry the disc actually sees

The review sized the disc batch off `streaming_chunk_size=18`. That is
the ROLLOUT chunk, not the scoring band. The OF fake is
`score_image[:, lo:hi]` where `[lo, hi)` is the True span of
`score_grad_mask`, and on both the rolling and the iter-1 path that span
is **9 frames**, not 18:

| | frames | |
|---|---|---|
| `n_ctx` | 9 | detached student overlap (GT on iter 1) |
| **`sup_span`** | **9** | **the band — this is what the disc scores** |
| `gt_after_frames` | 3 | GT scaffold at the OOD slot |
| `score_image` | 21 | total |

`sup_frames = new_frames = streaming_force_new_frame_chunks * npb =
3 * 3 = 9` (`model/dmd_action_forcing.py:12191`, `:13301`, `:13480`);
confirmed empirically by the smoke's own log line
`[42F-ROLLING] ... sup=[30,39)` (`logs/of_smoke_r2.log:558`). So the
disc forward is **2 rows x 9 frames x 1561 tokens = 28,098 tokens**,
exactly HALF the review's 56,196.

### The footprint, per disc forward

At dim 1536, 30 blocks, taps `[21,29]`, bf16,
`fake_score_gradient_checkpointing=true`:

| term | measured | note |
|---|---|---|
| DiT checkpoint boundaries | **2.41 GiB** | 30 x [2, 14049, 1536] bf16 |
| tap blocks (2 taps) | **1.18 GiB** | 21.5 KiB/token/row/tap — **outside** the checkpoint |
| head + unpatchify | **0.24 GiB** | 13.7 MiB/frame/row — **result discarded** |
| **total** | **~3.83 GiB** | |

vs the review's ~6.5 GB estimate. The review was right that the taps sit
outside the checkpoint and right that the head is computed-and-discarded;
it was wrong about the band width, which halves everything.

### The 92.7 GB scare does NOT apply to this path

A peer session measured a trainable-backbone disc forward at 92.7 GB vs
62-73 GB frozen, and attributed it to `F.linear` saving its INPUT when
the weight requires grad — `real_score` is frozen, `fake_score` is not.
The mechanism is real, but **`use_reentrant=False` checkpointing erases
it**, because a checkpointed region saves nothing but its boundary input
and the internals are recomputed one block at a time. Measured directly
(`TestCheckpointHidesSavedInputTerm`, and reproduced on an 8-block
stack):

```
ckpt=False: frozen=145.12 MiB  trainable=291.38 MiB  DELTA=146.25 MiB
ckpt=True : frozen= 73.12 MiB  trainable= 73.12 MiB  DELTA=  0.00 MiB
```

Our disc forward runs with `fake_score_gradient_checkpointing=true`
(`sbatch/_fgan_holder.sh`, applied at
`trainer/causal_action_forcing_train.py:419`), so the backbone carries
**no** requires_grad-dependent term. Sizing it off `requires_grad` would
overstate the cost by ~25x.

**Falsifiable prediction for the sibling `gan_of_backbone_trainable`
flag.** Because checkpointing already hides the saved-input term,
flipping that flag to `false` should change peak memory by only the
**tap** delta — measured 30,736 vs 43,024 B/token at 2 rows, i.e.
~0.33 GiB of the 3.83 GiB — **not** the 20-30 GB the frozen/trainable
comparison suggests. If a GPU run shows a multi-GB delta, then something
on that path is NOT checkpointed and this whole section is mis-sized.

### The five-forwards-per-step worry

There are `1 + streaming_fake_updates_per_gen` = 5 D forwards per active
step, but each has its OWN `critic_loss.backward()`, so they are
sequential: forward, backward, free. They do **not** accumulate. Peak is
one disc forward (~3.83 GiB) resident alongside the critic's denoising
forward — plus, on generator steps, the G-side disc forward landing while
`generator_loss.backward(retain_graph=True)`'s graph is alive.

### What shipped, and what it actually buys

| flag | default | mechanism | expected saving |
|---|---|---|---|
| `gan_of_checkpoint_taps` | `false` | checkpoint the tap stacks | **~1.1 GiB** (the largest addressable term) |
| `gan_of_classify_skip_head` | `false` | skip head+unpatchify on classify | **~0.24 GiB** |
| `gan_of_disc_micro_batch_groups` | `1` | split the `[fake;real]` rows into N forwards | see caveat |
| `gan_of_r1_num_samples` | `0` | R1/R2 real subsample | inert (weights are 0.0, B=1) |

**Honest caveat on the micro-batch knob.** It is NOT the memory fix the
LADD analogue is, and the reason is architectural. `ladd_disc_micro_batch_groups`
saves memory because that path **owns its optimizer** and calls
`.backward()` per group, so group *i*'s graph is freed before group
*i+1* runs (which is also why it must rescale each partial loss by
`/N_total`). The OF D term is *returned* and folded into the streaming
`critic_loss` for a **single** backward and a single
`fake_optimizer.step()` — that shared backward IS the arm's thesis, and
it is also what keeps `head.modulation` / `head.head.*` gradiented in a
`find_unused_parameters=False` reducer. Splitting only the forward
therefore leaves every group's graph resident until that one backward,
so the retained footprint is **unchanged**; only the forward's transient
working set shrinks. Giving the D term its own backward would need
`no_sync()` orchestration threaded from the trainer and would break the
one-backward contract — **not done, deliberately, and it should not be
done without sign-off.** The knob is shipped because it is free at
`groups=1`, because it makes the split a config knob if a future variant
does own its backward, and because at B=1 the batch is only 2 rows, so
`2` is the largest value that does anything at all.

**Recommended if the 8-node run OOMs:** `gan_of_checkpoint_taps=true`
first (biggest, ~1.1 GiB, value- and gradient-neutral), then
`gan_of_classify_skip_head=true` (~0.24 GiB). Together ~1.35 GiB of the
~3.83 GiB per-forward footprint, i.e. **~35%**.

### Rank uniformity

The group count is `disc_micro_batch_bounds(n_rows, groups)` — a pure
function of a config int and `latent.shape[0]`, clamped to `[1, n_rows]`
so no group is ever empty. No per-rank quantity is read, so the NUMBER
of disc forwards issued before the shared backward is identical on every
rank; a disagreement there would hang the `fake_score` reducer rather
than merely slow it. The R1 subsample draws from a CPU generator seeded
only by `(current_step, B)` for the same reason.

### Byte-identity

`groups=1` takes the verbatim historical single-call branch (asserted by
a forward-COUNT test, not just a numeric one); both model-side flags
default off and leave the tap loop and head path textually unchanged.
Value-equivalence across `groups in {1,2,3,4,7}` is asserted at
`rtol=0, atol=0`, with two mutation controls (disabling the cond
row-slicer; an off-by-one in the bounds) confirmed to fail the suite.

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
