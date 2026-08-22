# CARN v2 design — review of the seam affine + measured residual anatomy

*2026-08-21. Grounded in: `pipeline/action_forcing_training.py` (commit sites at
lines ~1669 and ~2334), `model/dmd_action_forcing.py` (target publisher, lines
~496 and ~9039), `model/anti_collapse.py` (stat definitions), and the
`analysis/drift_probe_v1/latdump_rank{0..7}.pt` frozen-model latent dumps
(40 rollouts, 120 roll-to-roll transitions, steps 200–295, lat
`[1,F,16,60,104]` fp16 finish-denoised chunks; roll 1 = 18-frame first slab,
rolls 2–4 = 9-frame AR chunks).*

---

## 1. What the current mechanism is (v0, as implemented)

At both KV-commit sites the committed chunk `commit_input_clean` passes, in
order, through three gated ops before the `t=context_noise` commit forward:

1. **Temperature** (`carn_seam_temp`, default 1.0=off): per-channel deviation
   scaling `x ← μ + T·(x−μ)`, μ over dims `(0,1,3,4)`. Counters the per-chunk
   variance contraction (k≈0.917 for the 4-rung sampler ⇒ T≈1/k).
2. **Drift counter-bias** (`carn_seam_drift_lambda` + `carn_seam_drift_file`):
   `x ← x − λ_d·d̂`, with `d̂ = global_drift_mu16.pt` (16-vector, ‖d̂‖=0.5708),
   the per-roll channel-mean drift fitted from this same probe. μ-only by
   design (grep-sign study: the σ-half of a Gaussian-referenced pull is
   wrong-signed when σ<1 everywhere).
3. **Seam affine** (`carn_seam_affine_lambda`, the favorite run uses 0.5):

   ```
   μ, σ  = per-channel mean/std of the chunk, dims (0,1,3,4)   # pooled over frames
   x ← (x−μ)/σ · (λ·σ* + (1−λ)·σ) + (λ·μ* + (1−λ)·μ)
   ```

   Target `(μ*, σ*)` is published per `generate_next_chunk` call by
   `model/dmd_action_forcing.py` (~line 9039) from
   `streaming_state["seed_latents"]` — i.e. the **ride seed's** per-channel
   stats, fixed for the whole ride, reset at ride reset. λ=0 is byte-identical
   off.

Key structural facts:

- The affine is **per-channel (16), frame-pooled** within the chunk. It
  constrains exactly 32 numbers per commit.
- It runs **inside** the AR loop, on the committed context only — the DMD/GAN
  losses never see a gradient from it (commit path is `no_grad`); it is pure
  inference-time/rollout-time state control. This is why it composes safely
  with everything and why v2 should keep the same placement.
- It is inference-parity by construction (seed is available at deploy time).

---

## 2. Measured drift anatomy (what the probe dumps actually say)

Per-roll deltas over 120 transitions, z = per-roll drift magnitude divided by
the **natural cross-scene variability** of the same statistic (std of the
roll-1 value across the 40 rollouts) — the honest "is this drift or just the
scene changing" scale:

| statistic (per roll) | mean ± std | natural scale | **z** |
|---|---|---|---|
| spatial power spectrum, **high band** (r∈[0.31,0.5]), |Δlog| | 0.265 ± — | 0.199 | **1.33** |
| channel-covariance **log-eigenspectrum**, ‖Δ‖₂ | 1.18 ± 0.97 | 1.00 | **1.18** |
| spatial power spectrum, mid band (r∈[0.13,0.31]) | 0.261 | 0.242 | 1.08 |
| per-channel **std**, ‖Δ‖₂ (16ch) | 0.475 ± 0.29 | 0.451 | 1.05 |
| spatial power spectrum, low band (r<0.13) | 0.336 | 0.448 | 0.75 |
| per-channel **kurtosis**, ‖Δ‖₂ | 2.30 ± 1.54 | 3.12 | 0.74 |
| per-channel **mean**, ‖Δ‖₂ | 0.818 ± 0.43 | 1.413 | **0.58** |

Signed structure (the direction of each drift):

- **Spectral tilt** — per radial bin, mean signed Δlog-power per roll:
  `+0.181, −0.071, −0.111, −0.116, −0.115, −0.127, −0.117, −0.057` (bins
  low→high). Every band above DC loses energy; DC/low gains. Per-transition:
  the high-band loss is front-loaded (roll1→2: −0.43; roll2→3: +0.06;
  roll3→4: +0.07) — a **one-shot contraction onto the sampler's operative
  spectrum**, followed by a slow **low-band inflation walk** (+0.11, +0.02 —
  the DC/haze walk continues every roll). Blur = high loss; haze/cartoon
  flat-fill = low gain. Both live in the radial profile.
- **Cross-channel collapse** — corr-matrix off-diag Frobenius **grows**
  +0.48/roll; effective rank ( (Σλ)²/Σλ² of the 16×16 channel covariance )
  **shrinks** −0.24/roll. Channels de-decorrelate → the latent palette
  collapses toward fewer joint directions. This is the "cartoon palette"
  signature and is invisible to any per-channel op.
- **Gaussianization** — mean kurtosis −0.18/roll, roll4−roll1 = −0.55 (from
  a near-Gaussian ~2.9 start, 13/16 channels drop). Heavy tails = sparse sharp
  detail; losing them = texture flattening. Also per-channel, but 4th-moment —
  untouched by the affine.
- **Mean drift decomposition** (16-d channel-mean space): projecting each
  per-roll Δμ onto d̂ gives **bias +0.571 ± 0.559 along d̂** (this *is* the
  memory's |d|=0.57/roll — it is the systematic component; ‖g‖=0.5708 by
  construction of the fit) plus **0.413 ± 0.21 orthogonal diffusion**. Net
  ride-drift directions agree across rollouts at mean pairwise cos **0.848**.
  So μ-drift is ~58% deterministic walk, ~42% diffusion.

**Answer to Q1.** Ranked by z, the statistics the affine does NOT constrain
that drift the most are: **(1) the radial power-spectrum profile (high-band
loss + low-band gain), (2) the channel-covariance eigenspectrum /
cross-channel correlations, (3) per-frame spatial variance structure**
(the affine matches frame-pooled σ only), **(4) kurtosis** (steady
Gaussianization). Per-channel mean — the thing v0 anchors hardest — is
actually the *least* anomalous relative to natural scene variability (z=0.58),
though its directional consistency makes it the most visible as long-horizon
color/exposure haze. Texture drift and the cartoon look map cleanly onto (1),
(2) and (4); v0 cannot see any of them.

**Staleness measurement (for candidate d).** Within roll 1 itself (first 9 vs
last 9 frames, the closest thing to GT in the dumps since it is generated
under near-full GT-seed conditioning): |Δμ| = **0.98**, |Δσ| = 0.53,
|Δlog-spec| = 0.27 — per 9 frames of *real scene evolution*. That is **larger
than the per-roll AR injection** (0.74 median |Δμ|) and ~70% of the full
scene-to-scene spread (1.41). Caveat: this is a proxy (generated frames, not
raw GT), but the conclusion is robust: **seed stats go stale at roughly the
same rate the model drifts.** A fixed seed target mis-anchors about as much
as it corrects once the ride has evolved a few chunks — pulling every new
scene's palette back toward the seed's is itself a plausible contributor to
the flattened/cartoon global look.

---

## 3. The λ question (Q3): exact AR(1) analysis

Let `μ*` be the target, `m_k` the raw per-channel mean of chunk k before
correction, `m'_k` after. The affine gives `m'_k = (1−λ)m_k + λμ*`. The AR
injection model (empirically justified by the 0.85 direction consistency:
the model reproduces its context's stats plus an injection)
is `m_k = m'_{k−1} + δ_k` with `δ_k = 0.57·d̂ + η_k`, `E‖η‖ ≈ 0.41`. The
residual `e_k = m'_k − μ*` then obeys **exactly an AR(1) with pole (1−λ)**:

```
e_k = (1−λ)(e_{k−1} + δ_k)
```

- **Steady-state bias**: `e_∞ = ((1−λ)/λ)·δ`. At λ=0.5, `e_∞ = δ` — the
  standing residual equals one full roll's injection (≈0.57 along d̂, i.e.
  the model permanently sits one uncorrected roll away from the anchor).
- **Fluctuation**: stationary std of the η-driven part is
  `(1−λ)/√(1−(1−λ)²) · std(η)` = 0.58·0.41 ≈ **0.24** at λ=0.5.
- **Memory**: time constant `τ = −1/ln(1−λ)` = **1.44 rolls** at λ=0.5.

| λ | e_∞ (×0.74 median/roll) | settle τ (rolls) |
|---|---|---|
| 0.3 | 1.73 | 2.80 |
| **0.5** | **0.74** | **1.44** |
| 0.7 | 0.32 | 0.83 |
| 0.8 | 0.19 | 0.62 |
| 0.9 | 0.08 | 0.43 |

So yes — **λ=0.5 per-commit is literally an AR(1) with decay 0.5**, and the
answer to "is λ too low?" is: *for the strain window, λ is not the variable.*
Because τ ≈ 1.4 rolls, the residual tracks the instantaneous injection δ(t)
essentially without lag on the scale of a 75-training-step window. A
**bounded, self-recovering strain window at steps ~466–541 therefore cannot
be a λ-accumulation effect — the mechanism has no memory beyond ~3 rolls. It
must be `e_∞ ∝ δ(t)`: the per-roll injection itself transiently grew** with
training time (GAN/critic pressure phase, gen-update cadence, or a batch of
hard rides — backward/degenerate directions have known larger injection), and
receded. The self-recovery is the affine's geometric forgetting working as
designed; had λ been statically too low, the strain would be constant in
time, not windowed.

Two caveats before turning λ up anyway: (i) raising λ shrinks e_∞ but pushes
harder toward a target that Section 2 shows is *stale* (staleness ≈ injection
in magnitude) — at λ→1 you hard-pin a moving scene to the seed's palette;
(ii) the fluctuating η part is only mildly damped by λ. The right split is:
**keep λ≈0.5 for the mean/σ anchor, fix the target (candidate d), and
diagnose the window by logging the injection directly.**

**Cheap confirmatory logging (do this first, ~4 scalars/commit, no behavior
change):** at each commit site log `‖m_k − m'_{k−1}‖` (= δ̂, the raw
injection), its projection onto d̂, the pre/post-affine offset `‖m_k − μ*‖`,
and the high-band/low-band power ratio of the chunk. If δ̂ rises inside
466–541 and falls after, the GAN-pressure hypothesis is confirmed; tag by
ride direction to separate data-mix effects.

---

## 4. CARN v2 candidates, ranked by cost

All candidates keep v0's placement (commit-site, `no_grad`, inference-parity,
composed after temp/drift/affine), the λ-blend convention, and default-off
gating. Publisher side: extend the block at `model/dmd_action_forcing.py`
~9039 to publish the extra seed targets next to `pipe._carn_seam_target`.

### (d) EMA / mixed target — cost: ~zero (bookkeeping only) — **do first**

Motivation: staleness ≈ injection (Section 2). Replace the fixed seed target
with a leaky integrator that admits legitimate scene evolution while still
suppressing fast drift:

```
μ*_k = ρ·μ*_{k−1} + (1−ρ)·[ α·μ_seed + (1−α)·m'_k ]      (same for σ*)
```

Knobs: `carn_seam_target_ema_rho` (per-roll EMA pole, suggest 0.9 ⇒ ~10-roll
horizon), `carn_seam_target_seed_mix` = α (suggest 0.3 — the permanent seed
tether that keeps the anchor from following an unbounded walk; α=1, ρ=0
recovers v0 exactly). Init `μ*_0` from the seed as today. Dynamics: the
anchor now tracks slow (scene-rate) stat changes with lag 1/(1−ρ) rolls while
the affine still kills per-roll injection; the un-killable component is only
the drift that masquerades as scene change *within* the EMA horizon —
bounded by `(1−α)·δ·ρ/(1−ρ)`-ish, tune α up if haze returns. During training,
where `ride_latents_window` GT is on hand (the `gt_chunk` built for
baseline MAE), optionally publish GT-chunk stats into the EMA instead of
`m'_k` (`carn_seam_target_gt_when_avail=true`) — exactly on-policy-corrected,
and inference falls back to the self-EMA path automatically.

### (a) M2/TV re-anchor (second-moment extension) — cost: negligible compute, no new state

Extends the affine to the two stats the anti-collapse family already treats
as canonical (`model/anti_collapse.py`): per-frame **M2** = Σ_c σ_c² over
(H,W) (`_per_frame_M2`) and per-frame **TV** = Σ_c mean|Δx| along H and W
(`_per_frame_TV`). Two deltas vs v0:

1. **Per-frame σ, not frame-pooled** — v0's σ match over dims (0,1,3,4) lets
   within-chunk frame-to-frame variance structure sag (the probe's per-frame
   spread drifts under the pooled constraint). Apply the σ half of the affine
   per (B,F,C) instead — this is exactly `impose_channel_stats(mode="m2")`
   semantics with the seed target from `seed_channel_stat_target` (reuse
   those functions; targets are already [B,C]). Knob:
   `carn_seam_affine_per_frame=true`.
2. **TV/detail gain** — a per-channel multiplicative detail re-inflation:

   ```
   g_c = λ_tv · TV*_c / TV_c(chunk) + (1−λ_tv)          # clamp to [1/g_max, g_max], g_max≈1.5
   x_c ← μ_c + g_c·(x_c − μ_c)      # after the affine, so μ/σ then get re-normalized by it if re-ordered
   ```

   with `TV*_c` the seed's per-channel TV published alongside μ*/σ*. Knob:
   `carn_seam_tv_lambda`. Note TV and σ are coupled (a pure TV gain also
   scales σ); run TV **before** the affine so the affine re-establishes σ —
   net effect is then a *spectral tilt* at fixed variance: more edge energy,
   same total power. That is precisely the anti-blur direction.

   Caveat: TV is a single number per channel — it fixes the first moment of
   the gradient distribution, i.e. roughly one degree of freedom of the
   spectrum. It bounds blur but cannot separately fix the low-band inflation.
   That is (b)'s job; (a) is the cheap 80% version.

### (b) Radial-spectrum re-anchor at commit — cost: one fft2/ifft2 per commit (9×16×60×104, trivial) — **the aimed shot**

Directly targets the z=1.33 finding, and bounds blur (high loss) *and*
cartoon/haze (low gain) simultaneously because it matches the whole profile:

```
X = fft2(x_c)  per frame, per channel
P_c(b) = mean |X|² in radial bin b        # 8 bins over r∈[0,0.5], the Section-2 binning
g_c(b) = sqrt( λ_sp·P*_c(b)/P_c(b) + (1−λ_sp) )   # clamp to [1/g_max, g_max], g_max≈2
x ← ifft2( X · G )     # G = radially-smooth interpolation of g_c(b) over the (fy,fx) plane
```

`P*_c` = seed per-channel radial profile, published once per ride next to
`(μ*, σ*)` (target under (d): same EMA treatment). Phase untouched — this is
a zero-phase radial filter, so it cannot move content, only re-tilt texture
energy; DC bin (b=0) should be **excluded** (`g(0)=1`) because DC is already
owned by the affine's μ — double-anchoring DC would fight the drift-vec term.
Interpolate g between bin centers (linear in r) to avoid ringing from
piecewise-constant gains. Knobs: `carn_seam_spec_lambda` (suggest 0.3–0.5,
weaker than the affine — spectra are noisier per-chunk), `carn_seam_spec_bins`
(8), `carn_seam_spec_gain_max` (2.0). Order: temp → drift-vec → affine →
spectrum (spectrum is variance-preserving up to the gain clamp; if drift in
total σ is observed, re-run the affine's σ half after, it is 10 lines).

What (b) does NOT fix: the cross-channel eigenspectrum collapse (z=1.18) —
that would need a 16×16 whitening/recoloring `x ← Σ*^{1/2} Σ^{-1/2} (x−μ) + μ*`
against the seed channel covariance (a "cov" mode of the same machinery;
eigvalsh on 16×16 is free — see `_per_frame_stable_rank` for the identical
gram-trick). Worth a flag (`carn_seam_cov_lambda`) but start λ very low
(0.1–0.2): a full recolor per commit is a much harder intervention on content
than a radial filter, and the eigenspectrum drift is partially a *consequence*
of the spectral tilt (blur correlates channels); measure again with (b) on
before tuning (cov) up. Kurtosis (z=0.74) is deliberately left to no candidate
here: per-channel quantile/histogram matching at commit would fix it but is
the most content-destructive op in this list — park it as (e) stretch, only
if (a)+(b) leave visible texture flattening.

### (c) Residual predictor h(z_ctx) → Δ — cost: new params + offline training + closed-loop risk — **last**

A small MLP consuming pooled context stats (per-channel μ, σ, TV, band
powers of the current committed context: ~16×(2+1+8)=176-d input) predicting
the *next-roll injection* `δ̂_{k+1}` (16-d μ-delta, optionally + per-band
log-gains), trained on the probe dumps (120 transitions now; regenerate the
probe at 10× — the dump path `rollout_latent_dump_dir` in
`trainer/causal_action_forcing_train.py` ~line 9280 makes this a config-only
rerun). Commit-site application: `x ← x − λ_r·h(stats(x))` before the affine.

This is the only candidate that can catch the **state-dependent** part of the
injection (the 0.41/roll orthogonal diffusion has structure the global d̂
misses — direction consistency 0.85 < 1). But: it is trained off-policy on a
frozen model and applied closed-loop — the classic compounding mismatch. Gate
it hard: `carn_seam_resid_gate_cos` — apply only when
`cos(h(stats), d̂) > 0.5` (i.e. only when the predictor agrees with the known
drift direction, degrading gracefully to v1's global counter-bias), plus
`‖h‖ ≤ carn_seam_resid_max` (clamp at ~1.5×‖d̂‖). Knobs:
`carn_seam_resid_lambda`, `carn_seam_resid_ckpt`. Expected marginal gain over
(a)+(b)+(d) is the smallest of the list per unit risk; build it only if the
strain-window logging (Section 3) shows large state-dependent injection that
survives (d).

### Recommended order

1. **Logging** (Section 3) — lands with any run, settles the 466-window
   question for free.
2. **(d)** EMA/mixed target — zero cost, directly attacks the measured
   staleness, and makes every other candidate's target better.
3. **(a)** per-frame σ + TV gain — one afternoon, reuses `anti_collapse.py`
   machinery, bounds blur.
4. **(b)** radial-spectrum re-anchor — the aimed shot at the top-z residual;
   validate with the same probe pipeline (rerun the dump with v2 on, recompute
   the Section-2 table; success = high-band z → ~parity with natural scale and
   the signed low-band walk → ~0).
5. **(cov)** low-λ channel recolor only if the eigenspectrum z stays >1 with
   (b) on.
6. **(c)** residual MLP only on evidence from step 1.

Everything above is flag-gated default-off, byte-identical at λ=0 — same
contract as v0 — and none of it touches the clean-context rule (all ops act on
the model's own committed prediction, never on GT context, and never add
noise).
