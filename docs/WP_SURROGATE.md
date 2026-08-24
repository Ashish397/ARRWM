# WP-SURROGATE (B3) — latent critic serving pixel-disc gradients

Work-package doc for B3 in `docs/GAN_REDESIGN.md` §B. Only the main agent
edits `GAN_REDESIGN.md`; this file is the B3 report.

**Owned files (exclusive):** `model/latent_texture_critic.py`,
`testing/test_latent_texture_critic.py`, `docs/WP_SURROGATE.md`. As of
2026-08-24, also `testing/test_surrogate_trainer_wiring.py` (new,
standalone) and a bounded, anchor-identified block inside
`trainer/causal_action_forcing_train.py` / `configs/action_forcing_phase3_dmd.yaml`
(shared files — edited only once the trainer-file lock was explicitly
granted, per the landing order; see §4 for exactly what changed and what
did not).

**Status 2026-08-24 (end of day):** wiring is FULLY LANDED — build step
(§4.1/§4.5, lock granted 03:16) and per-step consumption + save/resume
(§4.2–§4.4, implemented by MAIN under researcher order mid-day, reviewed
and finalized by WP-SURROGATE as spec owner). All tests green:
`testing/test_latent_texture_critic.py` 55/55,
`testing/test_surrogate_trainer_wiring.py` 34/34 (15 build-step + 19
consumption/resume/warning, added on this review). Both original
contract deviations (§2a batch reduction, §2e normalized Sobolev) have
**main-agent sign-off**; see §6.5 and §6.6. §6.8's ratio-measurement item
is unchanged and still open. **Three things found and fixed on this
review, none of which were caught before real tests were written against
the real landed code:** the resume-gate OR-chain missing
`surrogate_critic_enabled` (§4.4, dead-safe but now closed for defense
in depth); a config that silently builds a fully inert surrogate with no
signal that it is inert (§4.4, new WARNING); and a genuine crop-origin
correctness bug where a per-crop origin list was passed into an API that
expects one batch-shared origin, silently reading one crop's y-value as
another's x-coordinate (§4.3, fixed by aggregating to a mean origin at
the trainer call site). Still unwired/unaddressed: the KV recompute-order
hazard (a separate, campaign-wide risk this doc's §5b already flagged
for exactly this arm) and the §6.8 real-teacher scale-ratio measurement —
neither blocks the code that is now landed, both gate whether an actual
training arm can be trusted to say anything about texture quality.

---

## 1. What this package is for

WP-PIXGAN's generator gradient crosses `VAE.decode` in graph-on mode
once per generator step. At crop scale (a 24×32 latent crop) that is
affordable — which is exactly why B1 runs standalone first. It does not
scale toward R10's ≥60% frame coverage: the decoder backward is the
dominant transient, and it grows with coverage.

The fix is the distilled-critic flow. An expensive frozen teacher
supervises a cheap latent-space student, and the **student** is what the
generator differentiates through. The generator's autograd graph then
never crosses the VAE or the pixel disc — only a ~24 M-param latent
module. Per-step decode cost drops out; the teacher's cost is paid once
every `pix_teacher_refresh_every` steps instead of every step.

This is not a new idea in this codebase. It is the same pattern as the
live action critic (`trainer/causal_action_forcing_train.py`:3038-3211),
where an expensive teacher (CoTracker → ss_vae) supervises a cheap
latent critic that delivers z-guidance to the generator. That block is
the design template for the frozen-critic idiom here and is **not
modified** by this package.

### Restore, don't reinvent

The full-fidelity implementation existed and was deleted at `01ea13d`:

| Piece | Ancestor location |
|---|---|
| Critic module | `835b1df:model/latent_sam2_critic.py` |
| Two-target distillation (value MSE + Sobolev ∇_z MSE) | `835b1df:trainer/causal_action_forcing_train.py`:2783-2965 |
| Generator consumption + two-stage warmup | `01ea13d^:trainer/causal_action_forcing_train.py`:5109-5155 |
| SAM2 teacher (replaced) | `835b1df:model/r3gan_sam2.py` |

`model/latent_texture_critic.py` is that critic, renamed, SAM2 stripped,
with the three deltas §2 records. The distillation and generator blocks
have been lifted out of the trainer into `LatentSurrogateDistiller` and
`generator_surrogate_loss` so the mechanism is testable without a
trainer, a VAE, a GPU or a process group — and so the eventual trainer
wiring is a handful of calls rather than 180 lines of re-typed logic.

### Why both distillation targets

* **Value distillation**: `MSE(g_φ(z), f(z).detach())`.
* **Gradient distillation (Sobolev)**: `MSE(∇_z g_φ(z), ∇_z f(z).detach())`.

`g ≈ f` does **not** imply `∇g ≈ ∇f`, and the gradient field is the only
thing the generator ever consumes. Without the Sobolev term the
generator is free to move the surrogate's *values* without improving
pixel texture at all.

This is demonstrated, not asserted:
`test_value_only_distillation_does_not_get_the_gradient_field` fits both
variants against a teacher whose value is ≈0 everywhere (zero-mean
field) while its gradient field is rich, then measures held-out
gradient cosine. Value-only scores a near-perfect value loss and a
gradient direction largely uncorrelated with the teacher's; Sobolev
recovers the field.

---

## 2. Deltas vs the ancestor

### (a) Teacher = `pixel_texture_disc ∘ decode(latent crop)`

Per frozen contract 1. Teacher value is the **mean patch logit**;
teacher gradient is `autograd.grad(disc(decode_grad(z_crop)).mean(), z_crop)`.

The teacher is **injected as a callable**, not imported:

```python
teacher_value_fn(z_crop [N, F, C, h, w]) -> patch logits
```

Three reasons, all load-bearing:

1. This module had to build and test before WP-PIXGAN landed. It did.
2. It keeps file ownership clean — B3 never imports or edits a B1 file.
3. It makes the whole mechanism testable against *analytic* teachers
   with closed-form gradients, so the Sobolev claims are checked exactly
   rather than by eyeball.

`reduce_patch_logits()` accepts `[N]`, `[N, 1]`, `[N, 1, ph, pw]` and
`[N, Fpix, 1, ph, pw]`, reducing by mean over every non-batch dim. This
is the contract-1 reduction and matches
`model/disc_holdout_probe.py::_reduce_scores` (which detaches; this one
must not, since the teacher-gradient path differentiates through it).

**`P` is never hardcoded.** B1 confirms the real patch count is **660**
(24×32 latent → 192×256 px → 8-px border trim → 176×240 → /8 → 22×30),
not the 768 in `TEXTURE_GAN_DESIGN` §4. Because both sides reduce by
mean, `P` cancels out of the objective entirely; it would only matter if
someone divided by a literal. Nothing here does — the reduction is
computed from tensor shape.

**Batch reduction — a deliberate, documented deviation from the literal
contract expression.** The teacher gradient is taken as
`autograd.grad(value.sum(), z_crop)` where `value` is the per-sample
mean patch logit, rather than `autograd.grad(value.mean(), z_crop)`.
Crops are independent, so row *i* of the result is **exactly**
`autograd.grad(disc(decode_grad(z_i)).mean(), z_i)` — the contract
expression evaluated per crop, with no approximation. The batch `.mean()`
form would additionally divide every row by `N`, making the Sobolev loss
scale as `1/N²` and forcing `grad_loss_weight` to be re-tuned whenever
`pix_crops_per_step` changes. Verified against a closed form *and*
against a literal per-crop recomputation in
`test_teacher_targets_match_closed_form_gradient`. Flagging it here
because contract changes need main-agent sign-off; I read this as the
same objective, not a different one, but the call is yours.

### (b) `pix_teacher_refresh_every=N` — the new cadence knob

The historical code ran the teacher **every step**;
`gan_critic_grad_full_every` (`835b1df`:2808) only widened *frame*
coverage within a step. The `% N` gate shape is reused from there.

Here the teacher forward + gradient runs on steps where
`current_step % N == 0`; between refreshes the distillation replays
cached targets. **The critic serves the generator on every step
regardless** — that is the entire point of the surrogate.

The non-obvious part is that the cache must store **`z` alongside the
targets**. A Sobolev target is only meaningful at the latent it was
taken at; between refreshes the generator's fakes have moved, so
re-using a target against a *new* latent would be silently wrong.
Replaying the frozen `(z, value, grad)` triple is correct-but-stale, and
the staleness is observable rather than hidden:
`train/surrogate_target_age` logs it and `critic_disc_corr` /
`critic_grad_cos_sim` measure what it costs.
`test_replay_uses_the_cached_latent_not_the_live_one` pins this.

Two smaller consequences:

* `TeacherTargetCache` samples uniformly from the buffer rather than
  always replaying the newest entry. With `N` large the critic would
  otherwise take `N-1` consecutive gradient steps against one batch and
  overfit it.
* Step 0 always refreshes regardless of `N` — the cache is cold, and
  skipping would leave the critic untrained until step `N`. A replay
  step with a cold cache sets `train/surrogate_skipped=1.0` and emits no
  loss keys, rather than logging a fake zero.

Memory is not a concern at crop scale: `pix_crops_per_step=4` at
`(F=3, C=16, 24, 32)` fp32 is ~590 KB for `z` plus the same for `grad`,
so capacity 8 per tag ≈ 9 MB. Full-frame `(3, 16, 60, 104)` entries are
~4.8 MB per pair — use a small capacity, or `cache_on_cpu=True`.

### (c) Health diagnostics kept verbatim, plus a new audit

`critic_grad_cos_sim` and `critic_disc_corr` are carried over verbatim
from `835b1df`:2925-2949 — they are the only honest readout for a
Sobolev surrogate. The *computation* is unchanged; what changed is when
the keys are emitted at all, and that is worth recording because this
package argued the general case elsewhere and then had to apply it to
itself.

**Forgeable zeros — corrected.** The ancestor emitted
`critic_grad_cos_sim = 0.0` in value-only mode, commented "n/a in
value-only mode", and this module reproduced it, with a test pinning the
0.0. That is wrong, and it is the same defect class as §2(e): telemetry
that cannot be read back. Each of the three gradient keys has a
legitimate 0.0 that means the **opposite** of "term disabled":

| key | what 0.0 actually means |
|---|---|
| `critic_grad_loss` | the surrogate matches the teacher's field EXACTLY (under normalization, the best attainable score) |
| `critic_grad_cos_sim` | the surrogate's field is ORTHOGONAL to the teacher's — a catastrophic surrogate |
| `surrogate_grad_mag_ratio` | the surrogate hands the generator a zero-magnitude gradient — an inert critic |

Zero-filling a disabled term therefore does not report a missing reading,
it reports a **wrong** one — and in the `critic_grad_loss` case it reports
a *perfect* result for a computation that never ran. So the three keys are
now **omitted** when `grad_loss_weight == 0`, and
`train/surrogate_grad_distill` (1.0/0.0) records the regime so an absent
key is distinguishable from a plumbing bug. Tests assert absence rather
than zero.

The rule generalizes, and WP-PIXGAN reached the same place independently
on `patch_logit_telemetry`'s spatial-variance key (a spurious 0.0 there
would read as "critic is not using locality", a real §7 diagnostic):
**a diagnostic must never be given a placeholder value inside its own
meaningful range** — omit it, or use a sentinel outside the range.

**New — `surrogate_grad_check()`**, the periodic direct-vs-surrogate
audit required by the brief. The distillation's own `critic_grad_cos_sim`
is measured on the samples the critic was *just fit to*, so it is a
training metric and will look good even if the surrogate has overfit the
cached refresh batches. `surrogate_grad_check` runs the true teacher on
an arbitrary `z` — point it at the live fake latent — and compares the
gradient the generator *would* have received from the teacher against
the one it *actually* receives from the surrogate:

| key | meaning |
|---|---|
| `train/surrogate_check_cos_sim` | direction agreement — the headline number |
| `train/surrogate_check_mag_ratio` | `‖g_surrogate‖ / ‖g_teacher‖` |
| `train/surrogate_check_rel_err` | `‖g_surrogate − g_teacher‖ / ‖g_teacher‖` |

**Read it like this:** `surrogate_check_cos_sim` drifting toward 0 while
`critic_grad_cos_sim` stays high is the overfit-to-cache signature →
lower `pix_teacher_refresh_every`. Both falling together means the
surrogate cannot represent the teacher → raise `d_model`/`num_blocks`,
or fall back to B1's direct path. `mag_ratio` far from 1 with a healthy
cos means the *direction* transfers but the effective GAN weight has
silently changed → re-bracket `pix_gan_loss_weight`.

It costs one extra teacher forward+grad, so keep `grad_check_every` well
above `pix_teacher_refresh_every`.

### (d) Two further deltas the new teacher forced

Not in the brief, but required for correctness; recording them so they
are reviewable.

**Dense output.** The ancestor pooled to `[B]` because the SAM2 disc
emitted one scalar per clip. The pixel texture disc emits a patch-logit
grid (`TEXTURE_GAN_DESIGN` §4: "per-patch logits are kept (no global
scalar)"), and researcher directive **R2** requires spatial/token-level
gradients — so the surrogate is dense too. `forward` returns
`[B, F, Hs, Ws]`, matching frozen contract 3 ("dense per-token value
map"), and the generator consumes `-critic(z).mean()` as specified.
`pool()` reproduces the ancestor's `[B]` frame_pool reduction for
symmetry with the disc contract; it is not on the generator path.

**2-D absolute positional embedding with a crop origin.** The ancestor
used a flat `[1, 1, max_spatial, d]` table indexed in raster order,
which is only well-defined at one fixed `Ws`. This module is called at
**two resolutions** — `(24, 32)` crops during distillation and the full
`(60, 104)` latent when the generator consumes it — so a raster-indexed
table would assign the same embedding to different absolute positions.
The table here is 2-D `[max_token_h, max_token_w]` in stem-token units,
and `forward(z, latent_origin=(y0, x0))` embeds a crop at its **true**
position in the frame. Vertical position is a genuine texture covariate
(sky / buildings / road — the same fact that motivates A24's coarse band
matching in `disc_holdout_probe.band_plan`), so this is not cosmetic.

*Which origin?* The **latent** crop origin, in latent rows/cols. The
critic never sees pixels: the decode, B1's 8-px border trim and the
patch grid all live inside the teacher callable, on the far side of the
`autograd.grad`. So the crop call and the full-frame call are indexed in
one coordinate system by construction, and the 1-latent-row ambiguity
B1 flagged never enters. The origin is also quantized by the stem's 8×
spatial stride, which rounds a 1-row difference away regardless; it is a
positional prior, not an index.

### (e) The Sobolev term in the ancestor was decorative — normalized here

**This is the one finding of the package that changes a number rather
than a shape, so it is flagged for sign-off.**

The two distillation targets do not live on the same scale, and the gap
is large and systematic. The teacher's value is a *mean patch logit* —
O(1), because a logit is O(1). Its derivative w.r.t. a single latent
element is that same O(1) response spread across the crop: roughly
`1/P` times a local sensitivity. So `L_value ~ v² = O(1)` while the raw
`L_grad ~ (∂v/∂z)² = O(1/P²)` elementwise.

Measured on this module's analytic teachers at the small test crop
(F=2, C=16, 8×8 = 2,048 latent elements):

| teacher | `L_value / L_grad` |
|---|---|
| local conv+tanh patch teacher | ≈ **683** |
| linear field teacher | ≈ **2957** |

and the ratio grows with crop size — the production crop
(F=3, C=16, 24×32 = 36,864 elements) is an order of magnitude larger again.

The ancestor shipped `gan_critic_grad_loss_weight: 1.0`
(`835b1df:configs/action_forcing_phase1.yaml`:704, unchanged through
`01ea13d^`:696). At weight 1.0 the Sobolev term therefore contributed
well under 1% of the critic's loss. **The gradient-distillation term in
the deleted implementation was, in effect, decorative — the critic was
trained by value distillation alone.**

Reproduced end-to-end (120 fit steps, held-out gradient cosine against
the true teacher, `test_unnormalized_sobolev_at_the_ancestor_weight_is_decorative`):

| configuration | held-out grad cos | ‖g_sur‖/‖g_true‖ |
|---|---|---|
| value-only (`grad_loss_weight=0`) | **−0.003** | 0.00 |
| ancestor: raw MSE, weight 1.0 | **−0.002** | 0.00 |
| normalized, weight 1.0 | **+0.986** | 0.89 |

The ancestor's exact configuration is statistically indistinguishable
from switching the Sobolev term off. That matters because value
distillation provably does not pin the gradient field (§1), and the
gradient field is the only thing the generator consumes.

So restoring the mechanism faithfully could **not** mean restoring the
weight verbatim. Two fixes:

* **(a) raise `grad_loss_weight` to ~1e3–1e5.** Rejected: the right
  value depends on crop size, patch count and the disc head's
  calibration, so it silently changes meaning whenever `pix_crop_lat` or
  the disc changes, and has to be re-bracketed every time.
* **(b) normalize by the teacher gradient's own mean power**, making the
  term a dimensionless *relative gradient error* in [0, ~1]: 1.0 when
  the critic's field is zero, 0.0 on an exact match.

**(b) is the default** (`grad_loss_normalize=True`).
`grad_loss_weight=1.0` then honestly means "weigh value error and
relative gradient error equally", and keeps that meaning across crop
sizes and teacher rescalings (test-pinned).
`grad_loss_normalize=False` reproduces the ancestor's raw MSE exactly
for anyone who wants the historical objective.

Confirmed on the realistic teacher too (local conv → tanh → patch grid,
the shape of `disc ∘ decode`): normalized Sobolev reaches held-out cos
**0.877** vs **0.012** for value-only.

---

## 3. Module API

`model/latent_texture_critic.py`, ~950 lines including the rationale
comments.

### `LatentTextureCritic(nn.Module)` — frozen contract 3

```python
forward(z: [B, F, 16, H, W], latent_origin: (y0, x0) | None) -> [B, F, Hs, Ws]
value(z, latent_origin) -> [B]        # dense.flatten(1).mean(1) — the distilled quantity
pool(dense)             -> [B]        # ancestor's frame_pool reduction
token_grid(h, w)        -> (Hs, Ws)   # stem is 8× spatial, temporal-preserving
num_params
```

Stem: 3× `Conv3d(k3, s=(1,2,2))` → GroupNorm → SiLU, 8× spatial
reduction, temporal preserved (60×104 → 8×13). Body: `num_blocks` pre-LN
transformer blocks, full space-time self-attention over `F·Hs·Ws`
tokens. Head: LayerNorm → `Linear(d, 1)` **per token**, zero-init.

Zero-init means a fresh critic is the constant-zero value field, so a
generator term that mis-fires before warmup pushes with exactly zero
magnitude. Test-pinned.

**The hand-rolled attention is load-bearing — do not replace it with
SDPA.** `L_grad.backward()` is a second-order backward through the
critic. `F.scaled_dot_product_attention` selects the FlashAttention
backend on this hardware and Flash has **no double-backward kernel**.
The explicit `softmax(QK^T/√d)V` path has working double-backward
everywhere. Two tests guard this:
`test_attention_double_backward_produces_param_grads` (attention params
receive second-order gradient) and
`test_attention_double_backward_matches_finite_difference` (the
second-order gradient is *correct*, checked in float64 against central
differences — non-crashing is not the same as right).

### `LatentSurrogateDistiller`

Owns the cadence, both targets, the optimizer step and the telemetry.

```python
d = LatentSurrogateDistiller(critic, value_loss_weight=1.0, grad_loss_weight=1.0,
                             pix_teacher_refresh_every=N, cache_capacity=8,
                             grad_check_every=K, max_grad_norm=..., sync_grads=True)
logs = d.step(z_real=..., z_fake=..., teacher_value_fn=..., current_step=s,
              optimizer=opt, origin_real=(y,x), origin_fake=(y,x))
chk  = d.surrogate_grad_check(z, teacher_value_fn, origin=(y,x))
```

**DDP: pass the UNWRAPPED critic.** PyTorch's `DistributedDataParallel`
does not support double backward — the reducer's autograd hooks fire on
the first backward, so `create_graph=True` followed by `L.backward()`
(exactly the Sobolev path) either errors or silently produces wrong
bucket views. This is the same restriction that forces gradient-penalty
GANs around DDP. **The ancestor at `835b1df` ran the distillation
through `self.latent_critic_ddp` and was exposed to this.** Here the
forward is unwrapped and `step()` all-reduces the critic's parameter
gradients by hand (`sync_grads=True`), giving the same averaged update.
Single-rank behaviour is identical either way.

Each rank draws replay samples independently from its own cache, built
from its own data shard. That is ordinary data parallelism — the
all-reduce averages over a *wider* sample, not a narrower one. Do not
"fix" it by broadcasting an index; the caches hold different tensors on
different ranks. (Contrast `_sample_critic_grad_frame_indices`
(trainer:3707-3731), which *must* broadcast because there the frame
subset selects which elements of a *shared* target get compared.)

### `generator_surrogate_loss(critic, z_fake_grad, latent_origin, weight)`

Returns `weight * (-critic(z).mean())` with the critic's parameters
frozen for the duration — the idiom from the live action critic
(trainer:3186-3199) and from `01ea13d^`:5133-5145. The generator's
backward must not write into the critic's parameters, because the critic
was already updated this iteration and its optimizer must see only the
distillation gradient. `try/finally`, because an exception in the
forward would otherwise leave the critic permanently frozen and silently
stop distillation — test-pinned, in both directions.

### `two_stage_gen_weight(step, critic_warmup_steps, gen_warmup_steps, gan_loss_weight, shape_fn)`

The two-stage warmup from `01ea13d^`:5117-5133, extracted as a pure
function so the trainer wiring is one call and the schedule is
unit-testable. Stage 1: weight 0 while the surrogate is being fit.
Stage 2: ramp 0 → `gan_loss_weight` over `gen_warmup_steps`.

---

## 4. Wiring plan — FULLY LANDED 2026-08-24

Landing order per §B held exactly through the build step: WP-14B →
WP-PIXGAN → WP-SURROGATE, lock granted 03:16. §4.1/§4.5 (build step +
config) landed then, scoped narrowly to the build call + echo per MAIN's
instruction.

**§4.2–§4.4 (teacher closure, per-step consumption, save/resume) landed
later the same day** (2026-08-24, ~10:02–~13:00), under a different
process worth recording accurately: no WP-SURROGATE session was live
when the researcher asked for the arm to be switch-ready, so MAIN
implemented the mechanism directly against this doc's own spec
(`docs/TASK_SURROGATE_CONSUMPTION.md`), then **explicitly retracted its
own lock claim** rather than release it as a unilateral decision —
handing the file, and the one genuine judgment call in it (the ordering
deviation, §4.3), back to this package's spec owner. On resuming, that
implementation was reviewed in full against this spec (every site read,
not sampled), one design decision was made (§4.3), two gaps were found
and fixed (the resume-gate OR-chain and the inert-config warning, both
§4.4), and a corresponding test section was added
(`testing/test_surrogate_trainer_wiring.py` §6–8). This is the intended
shape for cross-package work on an owned module: implemented by whoever
has the time when it is genuinely needed, decided and verified by the
owner before it is considered done — neither "coordination overrides
ownership" nor "ownership means only the owner may ever touch the file."

### 4.1 Build — LANDED

In `trainer/causal_action_forcing_train.py`, immediately after the
`pixel_texture_disc` build block (anchor comment: `# WP-SURROGATE (B3) —
latent surrogate critic. BUILD STEP ONLY this`). As landed:

```python
from model.latent_texture_critic import build_from_config
(
    self.latent_texture_critic,
    self.latent_critic_optimizer,
    self.latent_texture_distiller,
) = build_from_config(self.config, device=self.device)
self.surrogate_critic_enabled = self.latent_texture_critic is not None
if self.surrogate_critic_enabled and self.is_main_process:
    logging.info(
        "[ActionForcing] Latent surrogate critic built: "
        "params=%.2fM d_model=%d num_blocks=%d "
        "pix_teacher_refresh_every=%d grad_loss_normalize=%s "
        "grad_loss_weight=%.4g value_loss_weight=%.4g "
        "cache_capacity=%d teacher_use_checkpoint=%s",
        self.latent_texture_critic.num_params / 1e6,
        self.latent_texture_critic.d_model,
        self.latent_texture_critic.num_blocks,
        self.latent_texture_distiller.pix_teacher_refresh_every,
        self.latent_texture_distiller.grad_loss_normalize,
        self.latent_texture_distiller.grad_loss_weight,
        self.latent_texture_distiller.value_loss_weight,
        self.latent_texture_distiller.cache.capacity,
        self.latent_texture_distiller.teacher_use_checkpoint,
    )
```

No DDP wrapper (matches the ancestor and `LatentSurrogateDistiller`'s own
requirement — DDP does not support the double backward the Sobolev term
needs). fp32 throughout.

**Two deliberate deviations from the ORIGINAL plan (the code block that
used to be here), recorded rather than silently dropped:**

1. **No cross-gate `raise` requiring `gan_pixel_texture_enabled`.** The
   original plan raised `ValueError` if the surrogate was enabled
   without the pixel critic, on the grounds that "the surrogate's
   teacher IS the pixel texture disc." That is still true, but nothing
   in THIS window's code ever calls the teacher — the critic can be
   built standalone, and per-step consumption (where the coupling
   actually matters) is a separate future window. Adding validation
   logic for a dependency this window doesn't exercise is scope beyond
   what was granted, and it is its own thing that could be wrong and
   would need testing. The coupling is documented in three places
   instead — the trainer block's own comment, the yaml block's comment,
   and here — read at exactly the point someone would flip the flag.
   **This check belongs in the window that wires per-step consumption,
   not this one; do not forget to add it there.**
2. **`surrogate_critic_enabled` is DERIVED, never independently read
   off config a second time**: `self.surrogate_critic_enabled =
   self.latent_texture_critic is not None`, rather than a parallel
   `getattr(self.config, "surrogate_critic_enabled", False)` the way
   `self.gan_pixel_texture_enabled` is set in the sibling block. This is
   the direct, deliberate fix for the seam class this whole exchange
   kept finding (two readers of one config key that can drift) — see
   `test_surrogate_critic_enabled_is_derived_not_independently_settable`
   in the new wiring-test file. `max_grad_norm` was also dropped from
   the constructor call (the original plan threaded
   `self.gan_max_grad_norm` through) — `LatentSurrogateDistiller`
   already defaults `max_grad_norm=None`, and threading a second
   trainer attribute through a build call this window never exercises
   is the same unnecessary-coupling argument as (1). Add it back when
   per-step wiring lands, if grad clipping is wanted.

**Config-resolution-time validation** (the "check everything at
construction, not inside whichever consumer reaches it first" pattern
`_pix_resolve_cfg` established) is likewise deferred — there is nothing
to validate yet beyond what `build_from_config`'s own `getattr` defaults
already handle safely.

### 4.2 Teacher closure — LANDED

Landed in `_maybe_run_surrogate_distillation` (trainer, anchor
`def _maybe_run_surrogate_distillation`), matching the planned shape
closely. As landed:

```python
def _teacher(z_crop: torch.Tensor) -> torch.Tensor:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    px = self._vae_decode_grad(z_crop)
    if border > 0:
        px = px[..., border:-border, border:-border]
    n, f = int(px.shape[0]), int(px.shape[1])
    logits = self.pixel_texture_disc(px.flatten(0, 1))
    return logits.reshape(n, f, *logits.shape[1:])
```

Graph-on via `_vae_decode_grad` as required (never the no-grad twin);
border trim and crop geometry come from `self._pix_resolve_cfg()`, B1's
single config-resolution point, rather than a hardcoded `8`; the reshape
generalizes to whatever shape `pixel_texture_disc` actually returns
instead of assuming `[n, f, 1, *_hw]`. `empty_cache()` before the decode,
exactly as planned. Checkpointed recompute doubling (§(d) of this doc)
applies here as documented — expected, not a regression.

### 4.3 Per-step order — DECIDED: one-step staleness accepted

**The plan above said "not negotiable." That framing is superseded, and
recorded here rather than silently dropped — the same standard this
whole package has held everything else to.**

The "not negotiable" ordering was ported directly from the `01ea13d^`
ancestor's structure, in which D-update → distill → generator-consumption
were literally sequential steps inside one function — "just-updated"
was free. That is not the current trainer's structure: B1's own G-term
(`_compute_pixel_texture_g_loss`) runs **earlier in the step** than the
D-loop, a pre-existing architectural fact this plan did not know about
when written (`_pix_g_snapshot_disc` is B1's own answer to the same
problem — the direct path consumes a **pre-update disc snapshot**, one
step stale, by design).

Landed shape, as implemented and reviewed:

```
1. pixel disc D-update                         (WP-PIXGAN, unchanged)
2. distiller.step(...)                          ← fits the JUST-updated disc
   [ generator's G-term for THIS step already ran, EARLIER in the step,
     against the critic as of the END of the PREVIOUS step ]
3. generator consumes the critic on the NEXT step
```

**Decision (WP-SURROGATE, spec owner, 2026-08-24): ACCEPT the one-step
staleness.** Two reasons, not just "it was already built this way":

1. **Restructuring the step order is the wrong fix.** It would move
   B1's G-term to after the D-loop for *every* arm, not just the
   surrogate one — a much larger, more invasive change to a component
   this package does not own, to save one step of staleness on a
   mechanism that is *already* staleness-tolerant by design (that is
   what `pix_teacher_refresh_every` **is** — the whole affordability
   trade is "serve the generator every step regardless of how stale the
   critic's last refresh was").
2. **Symmetry has real experimental value.** Keeping the surrogate path
   at the same one-step staleness as B1's own snapshot semantics means
   an eventual surrogate-vs-direct comparison isolates the actual
   research question (does the surrogate approximate the direct
   gradient well) from a confound (one path being structurally fresher
   than the other for reasons unrelated to the surrogate itself).

The cost is honestly logged (`train/surrogate_staleness_steps = 1.0`,
landed) rather than hidden, which is the actual property this section's
"not negotiable" language was trying to protect in the first place.

**Alternatives-never-summed — landed as a branch, not an addition.**
Inside `_compute_pixel_texture_g_loss`: `if surrogate_critic_enabled: ...
return` sits before B1's direct decode/disc code, so the direct path
never executes when the surrogate is active — verified by telemetry, not
just control flow: the direct path's keys (`pix_g_fake_logit_mean` etc.)
are **absent**, not zero, whenever the surrogate served the term
(`test_gterm_surrogate_happy_path_never_reaches_direct_path_keys`).

**Crop origins — a real bug found on review, fixed.** `_pix_take_crops_with_origins`
correctly returns `(crops, ys, xs, bands)` — one `(y, x)` pair per crop,
as designed on B1's side. What the *first draft* of the consumption
wiring did next was not correct: it passed the raw per-crop list
straight through as `origin_fake=[int(y) for y in ys_f]` (and the
equivalent for `origin_real`). But `LatentTextureCritic.forward`'s
`latent_origin` — and therefore `LatentSurrogateDistiller.step`'s
`origin_real`/`origin_fake` — is **one `(y0, x0)` pair applied to the
whole batch in a single call** (§3, `model/latent_texture_critic.py`:403-410),
not a per-crop list. `compute_teacher_targets` reads it as `(origin[0],
origin[1])` — so for a 2-crop batch, crop 0's y silently became `y0` and
crop 1's y was silently *misread as `x0`*, with every other crop's real
origin dropped entirely.

Caught by this package's own test-writing, not inspection: the happy-path
functional test (`test_distillation_happy_path_threads_origins_and_calls_distiller_step`)
asserted what origin actually reached the distiller's cache and got the
wrong value back — the kind of thing a source-reading review does not
reliably catch, since the code reads as plausible line by line.

**Fixed** (module owner, this review) by aggregating to a single
representative origin at the trainer call site: mean `y` across the
batch's crops, `x` fixed at `0`. `x=0` is not a compromise — A24 does not
band-match horizontal position at all (`TEXTURE_GAN_DESIGN.md` §3.6:
"dashcam content is not horizontally stratified"), so there was never a
meaningful per-crop `x` to preserve. Mean `y` **is** a compromise: it is
the same batch-shared-origin approximation the module's own positional
embedding already accepts elsewhere ("a positional prior, not an index"),
now applied across crops instead of just across quantization — coarser,
but not a new *kind* of imprecision, just a larger dose of the existing
one.

**Open item, not blocking anything currently gated:** doing this
precisely — a genuinely per-crop origin — would mean `LatentTextureCritic.forward`
indexing `self.spatial_pos` per-batch-element instead of once for the
whole call (feasible with gather/advanced indexing, not attempted here).
Worth it if `surrogate_grad_cos_sim`/`surrogate_check_cos_sim` on a real
arm ever look worse than the crop-level fits in
`testing/test_latent_texture_critic.py` predict — that gap is exactly
what mean-origin batching would cost. Until there is a real number
suggesting it matters, the coarser fix is the right amount of
engineering.

**Mask the fake latent before it reaches either path — landed.**
Confirmed exactly as this doc specified: `_pix_select_fake_latents` (B1's
mask-aware helper, fail-loud on an absent/all-false mask) is called
**before** `_pix_take_crops_with_origins` in both the distillation
function and the G-term branch — pinned structurally by
`test_distillation_masks_before_cropping_not_after`, since a source-order
regression here reintroduces the zero-gradient trap silently.

### 4.4 Save / resume — FAIL-LOUD, LANDED

Landed exactly as specified. Save (parent block,
`causal_rolling_staircase_train.py`): gated on `latent_texture_critic is
not None` — **the attribute existing**, never `gan_enabled` (the B1(d)
trap) nor any other package's gate. Resume
(`ActionForcingDMDTrainer._maybe_resume`): `RuntimeError`, not a warning,
on a missing or unclean `latent_texture_critic` / `latent_critic_optimizer`
key while `surrogate_critic_enabled` is true.

**One gap found and fixed on review.** `_maybe_resume`'s own top-level
early-return OR-chain — the same one WP-PIXGAN's comment warns about one
clause up ("without this clause the pixel restore below would never
run") — did **not** list `surrogate_critic_enabled`. Added. Currently
**dead-safe** even before the fix: both consumption sites are
structurally gated on the pixel critic's presence (the G-term branch is
unreachable when `gan_pixel_texture_enabled` is False — that gate is
already in the OR-chain — and `_maybe_run_surrogate_distillation`
early-returns via its own no-teacher check), so a surrogate enabled
without the pixel critic never trains and has nothing to lose on resume.
Fixed anyway: that safety depended on an invariant enforced nowhere
else, and a future change to the no-teacher branch (a fallback path, a
value-only mode against a different source) would turn a latent gap into
a live one with no warning. One line, matches WP-PIXGAN's own precedent
exactly.

**A second gap found on the same review, not in the original plan at
all: a config that builds a fully inert surrogate with no signal that
it is inert.** `surrogate_critic_enabled=true` with
`gan_pixel_texture_enabled=false` builds the critic (the INFO echo
fires, §4.1) but — per the structural gating just described — the
critic then **never trains and is never consumed, for the entire run**.
Nothing before this review would have said so; the build-time INFO line
looks identical to a working configuration. Fixed with a `logging.warning`
at build time, fired exactly in this case (never on a valid
configuration, never off the main process) — a WARNING rather than a
`raise` because validating gate combinations is explicitly out of this
window's authorized scope (`docs/TASK_SURROGATE_CONSUMPTION.md` §5); the
warning closes the "looks active, is inert" gap without changing what
any existing config does.

### 4.5 Config block — LANDED (appended at END of `configs/action_forcing_phase3_dmd.yaml`)

```yaml
surrogate_critic_enabled: false
surrogate_critic_d_model: 512
surrogate_critic_num_blocks: 4
surrogate_critic_num_heads: 8
surrogate_critic_max_frames: 64
surrogate_critic_lr: 2.0e-4
surrogate_value_loss_weight: 1.0
surrogate_grad_loss_weight: 1.0
surrogate_grad_loss_normalize: true  # relative grad error; false = ancestor's raw MSE (see §2e)
pix_teacher_refresh_every: 4         # teacher every N steps; critic serves every step
surrogate_cache_capacity: 8
surrogate_grad_check_every: 100      # keep >> pix_teacher_refresh_every
surrogate_teacher_use_checkpoint: true
```

Landed exactly as planned, no key changes. Default-off and
byte-identical when off — verified, not just claimed, by
`testing/test_surrogate_trainer_wiring.py`'s
`test_gate_off_emits_no_log_line` (no new log key) and
`test_gate_off_consumes_no_rng` (the global torch RNG stream is
untouched by an off build — constructing the critic draws from it, so an
accidentally-ungated build would shift every downstream draw in the
run). A companion mutation-control test,
`test_gate_on_does_consume_rng_so_the_off_test_has_teeth`, confirms the
RNG check is not vacuously passing.

### 4.6 Bring-up sequence

1. `pix_teacher_refresh_every: 1` first — the ancestor's cadence, one
   variable changed at a time. Confirm `critic_grad_cos_sim` climbs and
   `surrogate_check_cos_sim` tracks it.
2. Only then raise `N` to 4, 8, 16. The knob is bought with staleness;
   `surrogate_check_cos_sim` is the price tag. Stop where it starts to
   fall.
3. Compare against B1's direct path at the same coverage before scaling
   coverage up. If the surrogate cannot match the direct path at crop
   scale, it will not rescue full coverage.

---

## 4b. TEACHER BACKBONE PIVOT — researcher directive, 2026-08-24 afternoon

**Directive (verbatim intent):** the surrogate branch must use a
PRETRAINED teacher — SAM2 — not the from-scratch PatchGAN. "We do not
want from scratch in this branch." The from-scratch pixel critic remains
B1's arm, untouched; the surrogate-gradient mechanism remains a pathway
on the pixel-GAN G-term exactly as wired in §4.3.

Clarification recorded while executing this: the "old surrogate" was
always SAM2-taught — `835b1df`'s teacher was a frozen SAM2 Hiera-B+
encoder with trainable ADM 2D heads (`model/r3gan_sam2.py`), deleted at
`01ea13d` along with the critic. There was never a DINO surrogate; DINOv2
exists in this campaign only as the B5 escalation option (never built).
This pivot therefore RESTORES the ancestor's full teacher identity, where
the earlier §2(a) delta had swapped in B1's PatchGAN per the frozen
contract of the time.

What landed:

- **`model/r3gan_sam2.py` restored verbatim** from `835b1df` (626 lines,
  lazy `sam2` import, frozen encoder + ADM heads, zero-init linear head).
  Package + `sam2_checkpoints/sam2.1_hiera_base_plus.pt` were already on
  disk. GPU preflight on holder 6109490: build 1.6 s, heads 11.22 M
  trainable, forward + input-grad + D-step all clean, `d_loss = 2 ln 2`
  exactly at init (zero-init head), peak 5.5 GiB.
  **`pad_to_square=true` is REQUIRED for crop inputs** — Hiera's
  pos-embed tiling rejects non-square inputs (the preflight caught the
  exact failure the ancestor grew the knob for; yaml default flipped).
- **`surrogate_teacher_backbone: pixel | sam2`** (default `pixel` — every
  existing config byte-identical). With `sam2`: the trainer builds
  `self.sam2_teacher_disc` + Adam over `heads_module` (lr
  `surrogate_sam2_lr`, betas (0,0.9)); `_maybe_run_surrogate_distillation`
  runs an **NS-logistic heads D-update** on this step's real/fake crops
  (decoded no-grad) BEFORE distillation, so the critic distils from
  just-updated heads; the teacher closure feeds the SAM2 disc whole
  `[N,F,3,H,W]` crops (it frame-pools internally to `[N]`;
  `reduce_patch_logits` accepts `[N]` unchanged). Deliberately simpler
  than the ancestor's RpGAN+R1+R2 — mechanics first, penalty stack is a
  follow-up.
- **Surrogate-only mode admitted at the three gates** (G-term call site,
  distillation call site, G-helper top): with the pixel gate off and the
  surrogate on, the G-term is served by the critic and the direct
  decode/disc path never runs. Both-off remains byte-identical.
- **Real-pool self-supply**: the A21 pool's only producer was B1's D-loop,
  which never runs in surrogate-only mode — the SAM2 heads would have
  trained against no reals while every metric looked healthy (the
  looks-active-is-inert shape again). The distillation step now admits
  its own windows via the same `_pix_pool_fill` primitive (A21 semantics
  preserved: cross-ride, band-stratified, FIFO refresh).
- **The `surrogate_grad_check` audit had NO trainer call site** — the
  knob was threaded by `build_from_config` and never consumed (this
  package's own share-not-value rule, instance N+1, in this package
  again). Call site added after `distiller.step` under
  `should_grad_check`, exception-guarded (telemetry must not kill a
  step, but errors flag `surrogate_check_err=1.0`, never silence).
  NOTE: landed mid-smoke via atomic replace — the two runs below imported
  the pre-fix module and do NOT carry the audit; any rerun will.

**In flight (researcher-assigned holders):** `smoke_sam2sur` on both —
REFRESH_N=4 on 6109490 vs REFRESH_N=1 (ancestor cadence control) on
6109486, 60 steps, MECHANICS ONLY per SMOKE SEMANTICS and the KV
readout-ordering rule. Verdict keys: `surrogate_n_teacher_refresh ==
ceil(60/N)`, `surrogate_sam2_d_loss` falling from 2 ln 2,
`critic_value_loss`/`critic_grad_loss` moving, `surrogate_consumed=1`,
and the N=4 vs N=1 step-time delta (the affordability number).

---

## 4c. PRETRAINED TEACHER ALTERNATIVES — DINOv2 + ConvNeXt (2026-08-24)

Added on request so the teacher basis is **measured, not argued**.

### The three backbones and why these three

| backbone | family | pretraining | scales | known bias |
|---|---|---|---|---|
| SAM2 Hiera-B+ | hierarchical ViT | segmentation (SA-1B) | multi | region / boundary |
| DINOv2 ViT-S/14 | plain ViT | self-supervised | single (block taps) | semantic / shape |
| ConvNeXt-T | conv | supervised ImageNet | multi (4/8/16/32) | **texture** |

* **DINOv2** is the doc-sanctioned option: `TEXTURE_GAN_DESIGN.md` §4.1
  names it as the B5 escalation and records that **ADD used it for this
  exact job**. Layer taps mirror ADD's evenly-spaced ViT hooks
  (`[2,5,8,11]` for the 12-block ViT-S) — shallow blocks carry the local
  signal a texture critic needs, which the final semantic block has
  largely abstracted away.
* **ConvNeXt** is the chosen second alternative because ImageNet-supervised
  CNNs are *texture-biased* (Geirhos et al., ICLR 2019). For a critic whose
  entire job is texture that bias is a feature, and it is the sharpest
  available contrast to DINOv2: conv-vs-transformer **and**
  supervised-vs-self-supervised in a single swap.

### The property that makes this a comparison rather than three builds

`model/pretrained_pixel_disc.py` reuses `r3gan_sam2._R3GANDiscHeads`
**verbatim** — same ADM 2D heads, same per-scale mean, same frame_pool,
same zero-init final linear. So an arm-to-arm difference is attributable
to the **feature basis and nothing else**. Writing a second head stack per
backbone would have quietly confounded exactly the comparison this exists
to make.

Both are **offline by construction** (weights pre-cached; compute nodes
have no reliable network — anything needing a live download would fail at
step 0 on a holder rather than here). Both GPU-selftested green.

### Two bugs this work surfaced

1. **The shared head imposes a minimum feature resolution.** The ADM head
   applies three `Conv2d(k=4,s=2,p=1)` in series (`n -> floor((n-2)/2)+1`),
   so 8→4→2→1 survives but **7→3→1→crash** (a 1×1 padded to 3×3 is
   smaller than the 4×4 kernel). ConvNeXt's stride-32 stage at 224 px is
   exactly 7×7. Fixed with per-backbone resolution defaults (dinov2 518,
   convnext 512) **plus a build-time guard** that names the offending tap
   and the minimum resolution — a resolution mistake now fails in the
   constructor, not six frames deep in a conv.
2. **Zero-init heads make the teacher gradient exactly zero at init.**
   Every pretrained teacher here ships a zero-init final linear, so
   `d(logit)/d(input) == 0` until the head takes its first D-step
   (measured: `input-grad norm=0` in all three preflights). Under the
   normalized Sobolev loss that divides by the `1e-12` floor → a ~1e12
   loss and an instant NaN, **not** the "numerically loud but survivable"
   behaviour §4.3 previously claimed. The live ordering (teacher D-update
   *before* distillation) prevents it today; a guard now degrades a
   degenerate pair to "term skipped, counted"
   (`surrogate_grad_degenerate_pairs`) instead of killing the run.

---

## 4d. THE NULL SMOKES — my own instance of the failure this campaign keeps finding

**Two complete 60-step "SAM2 surrogate" smokes (14:15, N=4 and N=1) were
NULL: the distillation never executed once.** Recorded in full because the
mechanism is the one this package has spent all day writing rules about,
and it still happened here.

*What happened.* The teacher-selection block — the lines defining
`backbone` and `teacher_disc` — was written by an edit script that hit an
assertion **before** its write step, so nothing landed. I then re-ran only
the *second half* of that script (the closure and D-update), which
succeeded. Net result: the function referenced `backbone` but never
defined it, and reached that code only after an earlier guard
(`if pixel_texture_disc is None: return`) had already returned — which in
surrogate-only mode is *always* true, since the pixel critic is off. So
the early return fired every step, the `NameError` was never reached, and
the runs completed 60 clean steps at a healthy-looking loss.

*Why nothing caught it.* Every individual safeguard worked and it still
got through:
- the regime flag `surrogate_distill_no_teacher=1.0` **was** emitted every
  step — exactly as designed — and nobody read it;
- both runs' `gen_loss`/`critic_loss` were near-identical between N=4 and
  N=1, which I noted at the time and rationalised as "the cadence only
  changes refresh frequency". It was in fact the signature of the
  surrogate contributing *nothing at all* in both;
- the launcher's own verification only checked the process exited 0.

*The general lesson, which is not new and that is the point:* a green run
is not evidence that the thing under test ran. This is the same shape as
`pix_finish_grad_enabled` (flag on, path inert, all suites green) and the
same shape as the `surrogate_grad_check` knob that was threaded but never
called. **Emitting a regime flag is only half the rule — something has to
assert on it.**

*What changed as a result:*
- `surrogate_distill_ran` (0/1) plus `surrogate_teacher_is_pretrained`
  now make "the distillation executed" a *positive, assertable* fact
  rather than something inferred from the absence of a warning;
- both relaunch command files **grep their own run's log** for the
  teacher-built line and print `*** TEACHER NEVER BUILT -- run is NULL ***`
  when it is missing, so a null run announces itself in the holder output
  instead of being mistaken for a result;
- the edit-then-verify discipline is now explicit: after every patch to a
  shared file, grep the file for the landed text before moving on. The
  lost edit was invisible precisely because I checked `parses` (which
  passed) rather than `is the change actually there`.

**Consequence for the record: no SAM2 surrogate result exists yet.** The
14:15 pair must not be cited for anything.

---

## 4e. THE INPLACE BUG — where §4.3's staleness analysis was wrong

The first real distillation runs (dinov2 and sam2, both N=4) died on **all
8 ranks at step 21 — the first distillation step** with

```
RuntimeError: one of the variables needed for gradient computation has been
modified by an inplace operation: [torch.cuda.FloatTensor [512, 1]],
... is at version 4; expected version 3
```

**Mechanism.** Within one iteration: (1) the surrogate G-term forward
builds a graph through the *live* latent critic; (2)
`_maybe_run_surrogate_distillation` later calls
`latent_critic_optimizer.step()`, which is **in-place** on those weights;
(3) `generator_loss.backward()` runs later still and finds the saved
tensors at a bumped version.

**Why §4.3 missed it.** That section reasoned carefully about *staleness* —
which critic version the generator is trained against — and concluded the
one-step offset was acceptable and symmetric with B1's snapshot semantics.
That reasoning is still right, and it is also not the whole problem: the
consumer-before-updater ordering is additionally an autograd **correctness**
violation, independent of whether stale gradients are scientifically
acceptable. Ordering was analysed as a *research* question when it was also
a *mechanical* one.

**Two things make this worse than an ordinary bug, and both are on me.**
`generator_surrogate_loss` already sets `requires_grad_(False)` on the
critic, which *looks* like it should prevent exactly this — it does not,
because the version counter is checked on the **saved tensor**, not on its
grad-ness. And B1's `_pix_g_snapshot_disc` docstring documents this precise
trap, in this same file, including the identical error text and the note
that `requires_grad=False` does not save you. I had read that docstring —
I quoted it in §4.3 when justifying the staleness decision — and still
rediscovered the bug from a stack trace rather than recognising that the
snapshot existed *because of* the hazard, not merely to define semantics.

**Fix:** `_surrogate_g_snapshot_critic()`, mirroring B1 exactly — a
deepcopy refreshed from the live critic at the top of every G-term call,
`requires_grad_(False)`, `eval()`. Three consequences, all wanted and all
matching B1: the later optimizer step cannot corrupt the graph; no
gradient lands on the real critic's `.grad` (so the G backward cannot
pollute the next distillation); and G is trained against the critic as of
the start of the step — the same one-step offset §4.3 already accepted and
logs as `surrogate_staleness_steps`. The snapshot is re-frozen on **every**
call, because `generator_surrogate_loss`'s `finally` block restores
`requires_grad_(True)` (correct for the live critic it was written for),
which would otherwise leave the snapshot accumulating a `.grad` no
optimizer ever consumes — a slow leak.

### And a second, unrelated failure in the same batch

ConvNeXt died at startup: with `HF_HUB_OFFLINE=1`, timm requests
`pytorch_model.bin` and **cannot discover** that the cache holds
`model.safetensors`, because finding the safetensors alternative needs a
HEAD request that offline mode blocks. This is precisely the risk
`pretrained_pixel_disc.py`'s own docstring warns about — "anything
requiring a live download would fail at step 0 on a holder rather than
here" — and it then did exactly that, having passed every login-node test
because the login node has network. **A backbone that loads offline on the
login node is not evidence it loads offline on a compute node.** Fixed by
exporting the weights once to `pretrained_backbones/` and loading them
directly; hub resolution remains as a clearly-logged degraded fallback so
a silent fall-through cannot be mistaken for the offline-safe path.

### Resolution — both fixes confirmed in vivo

`dinov2 @ N=4` **cleared step 21**, the exact point both prior attempts
died on all 8 ranks: zero `inplace operation` hits, and the distillation
demonstrably executing rather than assumed to be —

| step | evidence |
|---|---|
| 11 | `peak_gb=35.49`, no `c_corr` key — distillation not yet started (gated to 20) |
| 21 | `peak_gb=38.27`, `c_corr=+0.000` appears — first distillation step, teacher graph-on decode engaged |
| 31 | `peak_gb=38.66`, `c_corr=-0.461` — the surrogate is *moving* toward the teacher, not idling |

The `c_corr` key's **appearance** between step 11 and 21 is the positive
proof the earlier null runs lacked: it exists only if `distiller.step()`
returned logs. Memory rising ~2.8 GB at the same boundary is the
independent physical corroboration.

`convnext`'s NULL verdict in the same batch was the *pre-fix* run and is
expected; it re-runs behind `dinov2` on holder A.

Test state after both fixes: **106 green** — 55 module, 35 trainer-wiring
(+1: the old "pixel gate off ⇒ no G-term" assertion was correct only while
the pixel critic was the surrogate's sole teacher, and is now split into
*both*-gates-off vs surrogate-only-serves-the-term), 16 pretrained-backbone.
The 8 wiring failures the trainer changes caused were all genuine detections
of a real structural change, not flakiness — the "exactly one function in
the slice" assertion in particular caught the new helper being silently
swallowed into the distillation slice.

### THREE-BACKBONE RESULT — 2 of 3 positive, DINOv2 alone inverted

All three `N=4` runs completed cleanly (`exit=0`, all past step 21) under
identical code, cadence, seed and crop plan, with **identical ADM heads**.
`c_corr` (critic-vs-teacher value correlation, n=16):

| step | sam2 | convnext | **dinov2** |
|---|---|---|---|
| 31 | +0.031 | +0.542 | **−0.461** |
| 41 | +0.534 | +0.125 | **−0.501** |
| 51 | **+0.444** | **+0.586** | **−0.515** |

**ConvNeXt independently confirms the mechanism.** Two unrelated
pretrained bases both produce a positively-correlated, climbing critic —
which is what value distillation is supposed to do — while DINOv2 alone
sits at ≈ −0.5 and is *monotone* there rather than noisy. With n=16 and a
single seed no individual number is resolvable, but "2 of 3 positive, the
third consistently opposite-signed" is a qualitatively different claim
from any single pairwise gap, and it is the claim the shared-heads design
was built to license.

Consequence: the distillation machinery (cadence, cache, snapshot,
Sobolev guard, origin threading) is **not** the suspect. Whatever is
happening is specific to DINOv2's feature basis.

*(Bookkeeping: the dinov2 `exp_*.err` was subsequently overwritten by the
diagnostic rerun, which reuses the arm name. The values above are the
completed first run's, recorded here at the time; the rerun carries extra
telemetry and is not a replacement.)*

### CORRECTION — the cadence control breaks the backbone story

`sam2 @ N=1` (same backbone, same seed, same code; only the teacher
refresh cadence differs) came back **negative**:

| run | 31 | 41 | 51 |
|---|---|---|---|
| sam2 **N=4** | +0.031 | +0.534 | **+0.444** |
| sam2 **N=1** | **−0.327** | **−0.322** | — |
| convnext N=4 | +0.542 | +0.125 | +0.586 |
| dinov2 N=4 | −0.461 | −0.501 | −0.515 |

**This retracts the previous section's inference.** I wrote that "2 of 3
positive isolates DINOv2's inversion as basis-specific, not plumbing".
That is no longer supported: sam2 flips sign under a *cadence* change
alone, so the sign is not a property of the feature basis. The honest
statement is that **`c_corr` sign is unstable across both backbone and
cadence at n=16 on a single seed over 60 steps** — i.e. it is not a
reliable readout at this scale, which is precisely what the campaign's
seed floor says and what I partially talked myself out of by building a
narrative on three same-cadence points.

Two readings remain open and they are NOT distinguishable from this data:

1. **Noise.** n=16 per reading, one seed, ~10 distillation steps at N=4.
   Nothing here is resolvable and the pattern is coincidence.
2. **Cadence stabilises the target — a real and useful effect.** At N=1
   the teacher's heads are updated *every* step, so the distillation
   target moves under the critic every step and a lagging chase reads as
   anti-correlation. At N=4 the cached targets are held still for four
   steps, giving the critic time to actually converge onto them. If this
   is real it reframes `pix_teacher_refresh_every`: not merely a
   cost/staleness trade, but a **target-stability** knob where *more
   frequent refresh is worse* — the opposite of the intuition the knob
   was designed under.

Reading 2 is testable and cheap: `convnext @ N=1` and `dinov2 @ N=1`.
If both flip negative like sam2 did, cadence dominates and the backbone
comparison must be re-run at fixed, larger N with ≥3 seeds before any
basis claim is made at all.

**Standing consequence: no backbone recommendation can be made from these
runs.** They establish that the mechanism runs on three pretrained bases
and nothing more.

### THE OPEN QUESTION — `critic_disc_corr` is drifting the WRONG WAY

The mechanism now runs. Whether it *works* is a separate question, and the
one health metric that answers it is currently pointing the wrong way.

`dinov2 @ N=4`, correlation between the critic's per-sample values and its
teacher's, over 16 samples (8 real + 8 fake):

| step | `c_corr` | reading |
|---|---|---|
| 21 | `+0.000` | the zero-init critic — constant output, zero variance, so 0.0 is *definitionally* right here, not a measurement |
| 31 | `-0.461` | |
| 41 | `-0.501` | |

Value distillation minimises `MSE(critic_val, teacher_val)`, so this should
climb toward **+1**. It is going the other way, and consistently across two
consecutive reads rather than oscillating around zero.

**This is NOT reported as a finding.** Per SMOKE SEMANTICS these are
mechanics runs, and per the campaign's seed floor a single-run reading is
not evidence — at n=16 an |r| of 0.5 is around the edge of resolvable, and
the teacher is itself training hard (its heads go from zero-init upward via
an NS-logistic update every distillation step), so the target is moving
under the critic. A lagging chase against a fast-moving target can look
anti-correlated transiently.

**But it is the thing to check first on any real arm**, because if it
persists it means the surrogate is anti-fitting its teacher and the whole
mechanism is inverted — which no amount of "the run was green" would
reveal. Concretely, the first arm should:
1. read `critic_value_loss` (wandb; it does not reach stderr) — if it is
   *falling* while `c_corr` is negative, the two are inconsistent and
   something is genuinely wrong;
2. compare `N=1` against `N=4` — if the drift is a staleness artefact of
   replaying cached targets it should be markedly weaker at `N=1`;
3. read `surrogate_check_cos_sim`, the direct-vs-surrogate audit, which is
   the honest gradient-space readout and does not depend on value
   correlation at all.

Cheap and worth doing regardless: the step line carries `c_corr` but not
`critic_value_loss`, which is why this could only be half-diagnosed live.
Adding the value loss to that line would have made the above decidable
from the log alone.

### MATCHED-DEPTH RESULT — the sign flips between backbones

Both `N=4` runs reached step 51 under identical code, cadence, seed and
crop plan. The **only** difference is the feature basis, which is exactly
what sharing `_R3GANDiscHeads` verbatim was for.

| step | **sam2** | **dinov2** |
|---|---|---|
| 21 | +0.000 | +0.000 |
| 31 | +0.031 | −0.461 |
| 41 | **+0.534** | −0.501 |
| 51 | **+0.444** | **−0.515** |

(Step 21's `+0.000` is definitional on both — the zero-init critic has no
variance — not a measurement.)

**This resolves the §4e worry in the direction that matters: the mechanism
is NOT inverted.** SAM2's critic correlates *positively* with its teacher
and climbs there from zero, which is precisely what value distillation is
supposed to do. So the distillation machinery — cadence, cache, snapshot,
Sobolev guard, origin threading — is sound. Had both backbones drifted
negative, the plumbing would have been the prime suspect; it now cannot be.

**DINOv2's negative drift is therefore specific to that feature basis**,
and the diagnosis moves from "is the surrogate broken?" to "does this
basis suit the target?" — different question, different fix.

**Not a verdict, and the reason is not politeness.** Single seed, n=16 per
reading, mechanics-only per SMOKE SEMANTICS, and the campaign's own seed
floor (min resolvable `seed_log_dist` gap 0.489) exists precisely because
single-seed differences of this size have repeatedly failed to replicate.
What makes this *worth acting on* rather than dismissing is the **sign
flip** — a magnitude difference at n=16 would be unremarkable, an opposite
sign under matched everything is a qualitative split that is cheap to
re-test.

**The next diagnostic is already identified and is NOT more runs.** The
first thing to read is the DINOv2 **teacher's own** `d_loss`: if its ADM
heads are not separating real from fake, its values are near-noise and the
surrogate's correlation with them is arbitrary — a *teacher* problem, not
a surrogate one, and it would make the negative sign meaningless rather
than meaningful. That key does not reach stderr, which is the concrete
telemetry gap this session hit twice (see the note above on
`critic_value_loss`). **Both belong on the step line before the next
arm** — the diagnosis is currently blocked on logging, not on compute.

ConvNeXt built cleanly from local weights on the compute node
(`weights loaded from pretrained_backbones/... (offline, no hub
resolution)`), confirming the offline fix, and is the third data point.

---

## 5. Tests

`testing/test_latent_texture_critic.py` — CPU-only, no VAE, no pixel
disc, no GPU, no process group. Runs as a script or under pytest:

```bash
# Thread limits are NOT optional on this node (WP-PIXGAN §6.5, confirmed
# here): nproc=144, torch grabs all of them and thrashes on these small
# CPU convs. Without the limits the suite looks hung, and under load from
# several concurrent agent sessions it dies on RLIMIT_NPROC with
# "OpenBLAS blas_thread_init" / "CPU dispatcher tracer already initlized"
# — an environment failure that reads exactly like a code failure.
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
python testing/test_latent_texture_critic.py          # all
python testing/test_latent_texture_critic.py -k sobolev -v
pytest -q testing/test_latent_texture_critic.py

# NEW 2026-08-24: the trainer wiring test. Needs the CUDA-stub import
# idiom (Wan's T5 wrapper touches torch.cuda at import time), so it must
# run under pytest, not the plain-script runner above.
PYTHONPATH=. pytest -q testing/test_surrogate_trainer_wiring.py
```

The five distillation fits are the expensive part (~30-50 s each) and are
memoized by config, so several tests share one fit. Everything else is
seconds.

Teachers are analytic with closed-form gradients, so the Sobolev claims
are checked exactly. `LinearFieldTeacher` has value ≈0 with a rich
gradient field — that gap is what separates the two distillation modes.

| Group | What it protects |
|---|---|
| shapes / contract 3 | dense map shape at crop **and** production 60×104 width; `pool`/`value`; zero-init ⇒ zero field ⇒ zero generator gradient |
| positional origin | crop origin changes the field; `None` ≡ `(0,0)`; stride quantization documented; out-of-table raises |
| **double backward** | attention params receive second-order gradient; float64 finite-difference check that it is *correct* |
| teacher targets | contract-1 reduction over all four shapes; gradient matches closed form **and** the literal per-crop contract expression; checkpoint path agrees; batch mismatch is loud |
| **cadence (b)** | teacher fires exactly `ceil(steps/N)` times for N ∈ {1,2,5} while the critic trains every step; `target_age` cycles `0..N-1`; replay uses the **cached** latent; cold cache flagged not faked; FIFO + CPU offload |
| **Sobolev (why it exists)** | Sobolev reaches held-out grad cos > 0.9 (linear) / > 0.8 (patch); value-only lands at ~0 on both; value-only reports the diagnostic as n/a rather than faking it |
| **Sobolev scaling (§2e)** | the value/grad scale gap is measured and pinned; the ancestor's unnormalized weight-1.0 config is shown to match value-only; the normalized form is verified to be a relative error, scale-invariant under a 100× teacher rescaling, while the raw form moves by 10⁴ |
| grad-check audit | a *perfect* surrogate scores cos ≈ 1 (so the metric can be trusted when it says "drifted"); cadence gate; training mode untouched |
| **seams (origin plumbing)** | the `latent_origin` actually travels caller -> teacher target -> critic forward, for `step()`, the replay path, `surrogate_grad_check` and `generator_surrogate_loss` — each with a mutation control asserting the check FAILS on a cut seam |
| frozen-critic idiom | generator backward writes **no** critic param grads but does reach `z`; `requires_grad` restored, including on exception; sign pushes the value **up** |
| warmup | full schedule incl. monotonicity, no-ramp case, custom shape |
| telemetry | every promised key present, float, finite; the three gradient keys ABSENT (not zero-filled) in value-only mode, with `surrogate_grad_distill` recording the regime |

`testing/test_surrogate_trainer_wiring.py` (NEW, 2026-08-24, 15 tests) —
covers exactly the landed §4.1 block via the same anchor-extraction +
CUDA-stub idiom `testing/test_pixgan_trainer_wiring.py` established
(read that file's own header for the rationale). Gate defaults false;
gate-off builds nothing, logs nothing, and — the mutation-controlled
check — consumes no RNG; gate-on builds all three objects and the
`surrogate_critic_enabled` flag agrees; the block does not couple to
`gan_pixel_texture_enabled` in code (only in an explanatory comment,
which a naive substring check on the first draft of this test flagged
as a false positive — fixed to check code lines only); the resolved-value
echo is checked TWICE — once end-to-end (render the actual log line and
look for the requested cadence in the rendered string) and once
statically (every argument expression in the log call reads a
`self.latent_texture_critic.*` / `self.latent_texture_distiller.*`
attribute, never `self.config` a second time) — the second is the
property the echo rule actually requires; the first only shows the
numbers happen to agree, which a parallel-but-disconnected re-read could
also produce by coincidence.

---

## 5b. Seam coverage — a defect this package had, and the rule that found it

WP-PIXGAN found `pix_finish_grad_enabled` read off the *pipeline object* by
the pipeline and off *`self.config`* by the trainer, with nothing assigning
it to the pipeline: T1's tests set the attribute on a stub, T3-B's tests
read the config, **both halves green, the seam between them untested**, and
a 60-step smoke ran with the flag "on" while the grad path never engaged.

This package had the identical defect in its `latent_origin` plumbing.
Every positional test called `critic.forward(latent_origin=...)` directly;
the one end-to-end test passed `origin_real`/`origin_fake` into `step()`
and asserted only that the numbers came out finite. **Deleting
`latent_origin=tgt.origin` from `step()` left all 44 tests passing.**

That was verified, not assumed. A mutated copy of the module with the seam
cut was built in a scratch directory (never the shared file — a peer runs
pytest against it) and exercised:

```
NEW seam test detects the break : True   (critic saw [None, None])
OLD origin tests still PASS     : True   (they were blind to it)
telemetry still finite+healthy  : True   (nothing else would have caught it)
```

Four seam tests now cover `step()`, the replay path (the cached triple must
carry its own origin — a replayed crop scored at the wrong frame position is
silently wrong), `surrogate_grad_check` and `generator_surrogate_loss`. Each
carries a **mutation control**: the same assertion is run against a critic
that drops the origin, and the test asserts the check fails there. A seam
test that cannot fail on the broken code is not evidence.

**The rule, WP-PIXGAN's formulation:** *a flag is not wired until one test
drives it end to end from config to consumer, and that test must be shown to
FAIL on the pre-fix code. Stub-to-stub agreement is not evidence.*

**Consequence for §4's wiring, which is not yet written and must not repeat
this.** `surrogate_critic_enabled` and `pix_teacher_refresh_every` will each
cross exactly the seam that bit B1 — YAML -> `cfg` -> trainer attribute ->
`LatentSurrogateDistiller` -> behaviour. Neither is wired until a test
drives it from the config object through to an observable consequence
(distiller constructed / teacher called exactly `ceil(steps/N)` times), and
that test is demonstrated to fail with the wiring removed. The cadence knob
is the more dangerous of the two: if `pix_teacher_refresh_every` fails to
reach the distiller, it silently defaults to 1 — the teacher fires every
step, the run is simply slower, and **every metric stays healthy**. That is
the affordability mechanism quietly not running, which is the entire point
of the package.

---

## 6. Open items / risks

1. **Cross-resolution generalization is the real risk.** Distillation
   happens on `(24, 32)` crops; the generator consumes the full
   `(60, 104)` latent. Attention is global over tokens, so a function
   fit on 3×4-token windows is not *guaranteed* to behave the same on an
   8×13-token field. The absolute positional embedding is the mitigation
   (a crop is embedded where it actually sits), but it is a mitigation,
   not a proof. Two ways out if `surrogate_check_cos_sim` is healthy on
   crops but the generator misbehaves: (i) run
   `surrogate_grad_check` on the **full** latent, which measures exactly
   this gap and costs one full-frame teacher pass; (ii) distil on full
   latents too — affordable *precisely because* of the refresh cadence,
   since the teacher now fires once every N steps. The module is
   resolution-agnostic, so this needs no code change.
2. **Sobolev on a crop constrains ∇ only on the crop's support.** Same
   root cause as (1); same two remedies.
3. **Staleness has no theory, only telemetry.** There is no principled
   `N`. §4.6 treats it as an empirical bracket read off
   `surrogate_check_cos_sim`.
4. **B3 without B1's disc has no teacher.** Stated in §B and unchanged.
   The module is built and green, but it cannot be exercised against the
   real teacher until WP-PIXGAN lands.
5. - [x] **The batch-reduction deviation in §2(a)** — **APPROVED**
   (main agent, 2026-08-23) on the scale-invariance rationale: `.sum()`
   over crops is the contract expression evaluated per crop, and keeps
   the Sobolev loss scale independent of `pix_crops_per_step`.
6. - [x] **`grad_loss_normalize=True` (§2e)** — **ADOPTED; verbatim
   restore REJECTED** (main agent, 2026-08-23). Ruling: −0.002 vs +0.986
   is not a style choice, and re-shipping a decorative term behind a
   plausible loss curve is the exact failure class the standing-rule
   block exists to prevent. The "if verbatim, then
   `surrogate_grad_loss_normalize: false` **and**
   `surrogate_grad_loss_weight: ~1e4`" escape hatch is recorded but **not
   taken** — do not re-open it by setting the weight to 1.0 with
   normalization off, which is the configuration that made the term inert
   in the first place.
7. **Grad-mask coverage caps what the surrogate can be trained on
   (§4.3).** With ~14% of frames detached, both the crop pool and the
   generator term shrink accordingly. Worth checking at bring-up whether
   the masked-in frame set is *biased* (it is structurally the trailing
   block, so it is not a random subset) — a surrogate fit only on
   early-block frames is being asked to generalize to late-block ones.
8. **The scale gap was measured on analytic teachers, not on the real
   `disc ∘ decode`.** The argument (`v` is O(1), `∂v/∂z` is O(1/P)) is
   teacher-independent, and two structurally different teachers agree,
   but the exact ratio for B1's disc is unmeasured until wiring.
   **Booked (main agent, 2026-08-23):** the check is on the first real
   pixel-critic arm's launch checklist — log `critic_grad_loss`
   *unnormalized* alongside `critic_value_loss` for ~50 steps and read
   the ratio. No dedicated allocation required; it rides an arm that is
   running anyway. This item stays open until that number exists.

## MAIN wiring completion note (2026-08-24 11:0x)

§4.2-§4.4 are LANDED (by MAIN, researcher-ordered; spec + deviations in
docs/TASK_SURROGATE_CONSUMPTION.md; tests in
testing/test_surrogate_consumption_wiring.py, 30/30 green). The arm is
switch-ready: `surrogate_critic_enabled=true` is the whole flick
(sbatch/run_smoke_surrogate.sh stages the mechanics smoke). §4.3 ordering
deviation: consumption is one step stale (symmetric with B1 disc
snapshot), logged as train/surrogate_staleness_steps. Distillation real
crops come from `_pix_real_pool` read-only (option (a)) — A20/A21
telemetry still describes exactly the D-loop.
