# CARN final plan — in-place de-drift at commit time

Written 2026-08-26. Status: **authorized to implement + smoke-test at short
scale**; not authorized for a long/full training run until the smoke is
reviewed.

## Goal (end state, per researcher instruction)

Today CARN (forward-noiser F, reverse-noiser R) is scattered across five-plus
application sites: the GAN discriminator's real-input augmentation (sites
A-E in `analysis/gan_tuning/CARN_TRANSITION_REVIEW.md`), a flash-slab
de-drift (`reverse_noiser_dedrift_apply_to_flash`), a pre-DMD-scoring
de-drift of `train_chunk` (`reverse_noiser_dedrift_enabled`), and a
just-added, not-yet-enabled de-drift of the teacher's DMD target
(`reverse_noiser_dedrift_apply_to_real_target`). None of these touch what the
model actually **remembers** — the committed KV-cache content that becomes
every future chunk's context.

**The end goal is for this document's mechanism to be the *only* CARN.**
Once it is built and validated, the plan is to retire the discriminator
application sites, the flash de-drift, the pre-scoring de-drift, and the
DMD-target de-drift, in favour of one place where correction happens: the
moment the generator's chunk is committed to context. This document does
**not** implement that retirement yet — it scopes the new mechanism and
leaves the old sites alone so the in-flight `carn_ctrl`/`carn_fwd` sweep and
the parallel `gt_vs_fake` decoupling work (another session, see
`docs/ONBOARDING_PIXDIRECT.md`) are not disturbed mid-flight. Retirement is a
deliberate follow-up once this is proven.

## The mechanism

Apply the reverse-noiser R **in place**, at the instant a chunk is finalized
and about to be committed to the KV cache — so the *corrected* chunk, not the
raw one, is what the model conditions on for every subsequent chunk. This is
the one site nothing has touched: every existing de-drift call only affects a
loss-computation copy (`train_chunk`) or a side-channel GAN forward
(`flash_dmd_gan_x0`, which per `docs/THINGS_STILL_TO_DO.md` item A is not
even the tensor that gets committed when flash is on). The actual commit
path has only ever had the crude, non-learned mean/std affine
(`_carn_seam_correct`, `pipeline/action_forcing_training.py:1148`).

### Where it plugs in

`_carn_seam_correct` is already called from **two** methods in
`pipeline/action_forcing_training.py` — the "both rollout twins" this
project's docs refer to (see the FIX 3 discussion in
`docs/THINGS_STILL_TO_DO.md`):

- `inference_with_trajectory` (`:1348-2229`) — the gradient-tracked,
  main per-step training rollout (called from
  `model/dmd_action_forcing.py:6306`, inside the primary streaming-DMD
  per-chunk training path).
- `generate_chunk_with_cache` (`:2369-...`) — the no-grad utility used by
  `generate_next_chunk` (`model/dmd_action_forcing.py:13975`, the general
  per-chunk driver), by anchor-chunk generation
  (`model/dmd_action_forcing.py:12524`, `:13267`), and — **this is the
  load-bearing fact for the second half of this plan** — by
  `_prebuild_rollout2_for_v24` (`model/dmd_action_forcing.py:13155`,
  call at `:13267`), which builds the rollout2 half of every
  rollout1→rollout2 pair the forward/reverse noiser trains on.

Add the in-place correction alongside the existing `_carn_seam_correct` call
in **both** methods, applied to the same tensor
(`commit_input_clean`/`_commit_src`) right before the KV-cache commit
forward. New flag: `reverse_noiser_dedrift_apply_to_commit` (bool, default
**False** — byte-identical off, same convention as every other
`reverse_noiser_dedrift_*` flag). Reuse `_dedrift_with_reverse_noiser`'s
existing stepping algorithm and its `reverse_noiser_dedrift_level` /
`_min_level` / `_steps` / `_alpha0` / `_alpha_decay` knobs rather than adding
parallel ones — this is the same relaxed multi-step Euler correction already
used at the flash and DMD-target sites, just moved to a new tensor.

### Why the reverse-noiser must keep training concurrently

This was the researcher's explicit condition, and it is correct: once the
correction is live, the model's *actual observed* step-to-step drift changes
(that is the whole point), so a frozen, pre-trained R is chasing a target
that stops matching reality the moment it starts acting on it. R has to keep
learning the **residual** drift that survives its own correction — a
self-referential, bootstrapping objective: R gets better → observed drift
shrinks → R's training pairs reflect a smaller residual → R keeps refining
against that smaller residual. Starting from R's near-identity init
(`out_proj` zero-initialized, so `R(x, 0) ≈ x`) this should be a gentle
on-ramp, not a discontinuity, but it is a moving-target training setup and
needs to be watched, not assumed safe.

**This falls out for free, structurally**, because of the call graph traced
above: `_prebuild_rollout2_for_v24` builds rollout2 via
`generate_chunk_with_cache`. Once that method applies the in-place
correction, rollout2 chunks are *already* corrected, and the forward/reverse
noiser training loop (`_train_forward_noiser_tf`, or
`_compute_fn_loss_rollout_to_gt` in `fn_pair_mode='rollout_to_gt'`) reads
rollout1/rollout2 as it always has — no separate plumbing change is needed
for the training pairs to reflect the corrected dynamics. This must be
**verified, not assumed** (see Testing below) — per this project's own
established rule, "prove a flag fired from a counter, never from the patch."

### A deliberate reversal of an existing safeguard

The `train_chunk` pre-scoring de-drift site
(`trainer/causal_action_forcing_train.py:19321-19337`) has an explicit
comment: *"Preserve the RAW student rollout1 for FN/cycle training below —
the cycle must learn the raw rollout1→rollout2 map, NOT G's de-drifted
output (that would corrupt the training pairs / create a feedback loop)."*
That safeguard is correct **for that site's purpose** — a scoring-only
correction should not contaminate what the corrector itself learns from,
because the correction there is not structurally part of generation.

This plan is different in kind, not degree: once correction is applied
in-place at commit time, it *is* structurally part of generation, and R
learning from the corrected trajectory is the entire point, not a
contamination risk. Anyone reading both sites side by side should not
"fix" this plan's site to match the other's safeguard — they are answering
different questions. This paragraph exists so that reversal is legible as a
decision, not an inconsistency.

## Risks

1. **Bootstrapping stability.** The self-referential fixed point (R corrects
   → drift shrinks → R's target shrinks) could converge cleanly, or could
   oscillate/collapse if R's early corrections are wrong-signed or
   overshoot. Mitigated by the existing relaxed-step (`alpha0`, decaying
   `alpha_decay`) design, but this is the first time that design is used
   somewhere its own output becomes the next input, repeatedly, across an
   entire rollout — a qualitatively different regime from a single
   corrective step at one scoring boundary.
2. **Exposure-bias / distributional shift.** The rest of the model (the
   generator, the DMD scorers, the GAN) has never been trained conditioning
   on R-corrected context. Even with R near-identity at init, as R's
   correction strength grows the context distribution the generator sees
   during training drifts away from anything the frozen teacher / real_score
   was validated against.
3. **Diminishing training signal.** As R's correction converges, the
   residual drift it is training on shrinks toward zero — a legitimate
   sign of success, but also a classic vanishing-gradient-style risk for R's
   own optimizer if not monitored (is R still receiving a usable signal at
   step 500, or has it gone quiet?).
4. **Cost.** This runs R at every chunk commit, in every rollout twin,
   including the rollout2 prebuild — not just once per DMD-scoring step like
   the existing sites. Meaningfully more forward passes through a ~30M-param
   network per training step than any current CARN application.
5. **Interaction with existing flags.** `carn_seam_affine_lambda` (the crude
   affine, already live at both twin sites) and the new mechanism both
   target the same tensor. Decide explicitly whether they compose (affine
   after learned correction), are mutually exclusive, or the affine gets
   retired here too — do not let both silently stack without a documented
   order, the exact class of bug `docs/THINGS_STILL_TO_DO.md` item B already
   found once in this file family (duplicate override keys, last-wins,
   undocumented).
6. **Gradient-flow correctness.** Unlike the DMD-target change (which lives
   entirely inside a `torch.no_grad()` block and was proven inert to the
   generator's gradient), `inference_with_trajectory`'s commit path is
   gradient-tracked — `cache_pred`/`commit_input_clean` there very likely
   carries the generator's live graph. Inserting R here needs the same
   "freeze R's params, keep the input differentiable" pattern
   `_dedrift_with_reverse_noiser` already uses, verified explicitly (not
   assumed) to confirm gradient still reaches the generator's own parameters
   through R rather than being severed — this project has one prior incident
   (`boundary_vae_roundtrip`) of exactly this kind of plumbing silently
   killing the DMD gradient past roll 1.

## Scope of what's authorized right now

Implement the flag-gated mechanism (default off, byte-identical when off),
add tests, and run **one short smoke** (a handful of nodes, a short step
count) to check basic mechanical health: no NaN/collapse, gradient still
reaches the generator, R's training loss is nonzero and trending down, and a
rollout-drift proxy (e.g. `roll_mae` or per-chunk latent std trend) moves in
a plausible direction. This is a mechanism check, not an effect-size study —
`analysis/gan_tuning/CARN_TRANSITION_REVIEW.md` §5 already established that
most GAN-adjacent telemetry needs ≥3 seeds past step ~130 before any
conclusion is trustworthy; this smoke is not that study. A longer/full run,
and the retirement of the older CARN sites described in "Goal" above, both
need a separate go-ahead once the smoke is reviewed.

## Testing plan

1. Byte-identical-off: with the new flag False, both twin methods' commit
   path is unchanged (existing test-writing convention:
   `testing/test_fn_apply_decoupled.py`,
   `testing/test_reverse_noiser_dedrift_real_target.py`).
2. Gradient-flow guard: with the flag True inside
   `inference_with_trajectory`'s path, confirm the generator's parameters
   still receive a nonzero, finite gradient after a backward pass, and that
   R's own parameters do too (both should train).
3. Rollout2-propagation proof: with the flag True, assert (via a direct
   check, not inference from other behaviour) that `_prebuild_rollout2_for_v24`'s
   output actually differs from what it would be with the flag False on the
   same seed/input — proving the "falls out for free" claim in this doc
   rather than trusting it.
4. Smoke launch per "Scope" above, on a free holder, short step count,
   watching for the specific failure modes in Risks 1-3.
