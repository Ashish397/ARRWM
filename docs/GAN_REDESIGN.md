# GAN redesign — proposed fixes

Companion to `GAN_ARCHITECTURE_BRIEF.md` (which states the current design and
the measured failure). This document collects proposed changes point by point,
each with the argument, the concrete implementation, the risks, and the test
that would confirm or refute it.

Standing constraint: every change lands **flag-gated and default-off**, is
byte-identical when disabled, and goes through adversarial review before launch.

---

## Point 1 — the global average over patch logits is the wrong reduction

### The claim

The discriminator has a *local* head — one 1×1 conv and one 3×3 residual block
on a 30×52 feature grid — which produces thousands of local logits, and then
collapses them:

```python
visual_logits = visual_logits.mean(dim=1, keepdim=True)   # model/ladd_disc.py:911-915
```

**before** the adversarial nonlinearity. With `gt_transition` inputs that is
5 taps × 6 frames × 1560 tokens = **46,800 local realism votes averaged into one
number.** It is no longer a PatchGAN.

The reduction order matters. Ours is

$$\text{softplus}\Big(\underset{i}{\text{mean}}\,[D_f(i) - D_r(i)]\Big)$$

whereas a patch discriminator gives

$$\underset{i}{\text{mean}}\,\Big[\text{softplus}\big(D_f(i) - D_r(i)\big)\Big].$$

Since softplus is convex, Jensen gives `softplus(E[x]) ≤ E[softplus(x)]`: a
disastrous set of fake local patches can be **cancelled by sufficiently normal
patches before the loss ever sees them**. For a failure that reads as "there is
horizontal garbage over 15% of the road", that is exactly the wrong reduction.

Canonical VQGAN / Stable-Diffusion autoencoder training does the opposite — the
discriminator emits a spatial patch map `[B,1,H,W]` and the adversarial loss
averages the patch logits *after* evaluating them.

### A second, sharper mechanism (stronger than the Jensen argument here)

At our operating point the convexity gap is small: `d_loss ≈ 0.59–0.69` means the
logit gap sits near 0, where `softplus(x) ≈ ln2 + x/2 + x²/8`, so the Jensen gap
is only `≈ Var(x)/8`.

The decisive effect is **gradient uniformity**. With `D = mean_i d_i`,

$$\frac{\partial L}{\partial d_i} = \frac{1}{N}\,\text{softplus}'(\Delta)$$

— *identical for every token i*. Every one of the 46,800 positions receives the
same gradient magnitude and the same sign. **The critic is structurally incapable
of telling the generator which region is wrong.** With patch logits the
derivative is `softplus'(Δ_i)/N`, token-specific, and the signal becomes
positional. This holds regardless of how convex softplus is near the operating
point, so it is the more robust half of the argument.

### The One-Forcing nuance

One-Forcing does use a scalar discriminator successfully — but it **pools with a
learned attention** (register queries attend over the latent tokens, an MLP emits
the scalar). That lets the network say *"that suspicious region matters more than
the 95% of this image that looks fine."* Arithmetic averaging of 46,800
pre-computed votes cannot express that. So "scalar output" is not the error;
**unweighted mean pooling** is.

### Why relativistic pairing must also change

Restoring patch logits while keeping spatially-matched RpGAN would assert that
patch `(x,y)` of the fake corresponds to patch `(x,y)` of the real. Our reals are
**nearest-L1 matches from the same ride and uniform draws from a 4096-entry
cross-ride ring** (`GAN_ARCHITECTURE_BRIEF.md` §5) — different scenes entirely.
There is no pixel correspondence to exploit, so per-position relativistic
comparison is meaningless noise.

Proposed loss — ordinary patchwise non-saturating logistic (or hinge):

```
D_loss = mean(softplus(-D(real_patches))) + mean(softplus(D(fake_patches)))
G_loss = mean(softplus(-D(fake_patches)))
```

### Implementation — cheaper than it looks

Both halves are largely already in the codebase.

1. **`ladd_scalar_output=false`.** This is the *default*
   (`causal_action_forcing_train.py:882`); our arms explicitly set it true. With
   it false, `rpgan_d_loss` already does `.mean()` over all dims, i.e. it already
   computes `mean_i[softplus(·)]`. **The convexity fix is a flag flip.**
2. **`ladd_r1_normalize_tokens=true`.** Currently false. The inline comment at
   `causal_action_forcing_train.py:7071-7076` states that with token logits R1's
   `grad_sq` estimates `‖∇ Σ_i D_i‖²`, which "scales with ~T² over ~47k tokens,
   forcing γ to a tiny un-portable value." **This is very likely the mechanism
   behind our replicated "R1 γ=1e6 pins every discriminator at ln 2" finding.**
   Normalising makes R1 estimate `‖∇ mean_i D_i‖²`, token-count-independent, so
   patch logits and a portable γ are compatible. This reframes a headline prior
   result as a scaling artefact rather than a law.
3. **New non-relativistic patchwise loss** — the only genuinely new code. Add
   `ladd_loss_form ∈ {rpgan, nsgan, hinge}` (default `rpgan` = byte-identical),
   with `nsgan`/`hinge` as above. R1/R2 are unchanged in form and still apply.

### Risks and open questions

- **Patch logits from a DiT are not spatially isolated evidence.** In a CNN
  PatchGAN a patch logit sees a bounded receptive field. Here every tapped token
  at block ≥6 has already attended globally, so patch logits restore *positional
  attribution of the gradient* but not *local evidence*. Real improvement, partial
  mechanism — and an independent argument for tapping much earlier blocks (see
  Point 2 when written).
- **Dropping relativistic loses R3GAN's convergence argument.** Most of R3GAN's
  stability comes from the zero-centred R1+R2 penalties, which carry over
  unchanged; but this should be stated as a deliberate trade, not overlooked.
- **Middle options worth measuring rather than assuming away:**
  - *Shuffled-pair RpGAN*: keep the relativistic form but pair fake patch `p`
    with a **randomly drawn** real patch `q`. Since no correspondence exists,
    random pairing is as valid as index pairing and preserves R3GAN's structure.
  - *Learned / soft-max pooling to a scalar*: replace `mean` with log-sum-exp or
    a top-k mean, so one bad region dominates. ~3 lines, keeps one scalar per
    sample and therefore leaves R1/R2 semantics untouched — the closest cheap
    approximation to One-Forcing's learned pooling.
- **The adversarial scale changes** when going from one logit to 46,800. Expect
  to re-bracket `gan_loss_weight`; the measured 0.01-inert / 0.03-learns /
  1.0-destroys bracket was established under scalar output and does not transfer.

### Test that would settle it

Fixed-route 60 s eval, matched arms at 200 steps against the `nogan200` control:

| arm | `scalar_output` | loss form | `r1_normalize_tokens` |
|---|---|---|---|
| A (control) | — | GAN off | — |
| B (current) | true | rpgan | false |
| C | **false** | rpgan | **true** |
| D | **false** | **nsgan** | **true** |

C isolates the reduction order; D adds the correspondence fix. Primary readout is
the researcher's judgement of the videos; supporting instruments are
`cos(g_GAN, g_DMD)` (does the GAN gradient stop fighting DMD?), `d_loss`
trajectory, and whether the scanline-banding failure mode disappears.
