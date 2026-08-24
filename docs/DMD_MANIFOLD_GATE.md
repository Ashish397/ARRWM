# DMD MANIFOLD GATE — why DMD amplifies drift, and the gate that stops it

**Researcher thesis, 2026-08-24 (recorded verbatim in substance; the
analysis in §3 onward is WP-SURROGATE's and is marked where it differs).**

## 1. The thesis

1. The rollout **compounds drift**: the further into a rollout a chunk
   sits, the further the student's sample has wandered from the data
   manifold.
2. **DMD's direction is only trustworthy near the manifold.** DMD points
   *toward* the data manifold when the sample is sufficiently close to
   it, and *away* from it when sufficiently far. So training DMD on the
   drifted tail does not correct the drift — it **amplifies** it.
3. That is a positive feedback loop: drift → DMD points wrong → more
   drift → further off-manifold → DMD points more wrong.
4. **The GAN should have caught this.** When the sample expands off the
   manifold the discriminator ought to see it and pull it back. Ours did
   not, because it lived in **the same latent space, and looked through
   the same teacher model's eyes, as the thing producing the error.**
   A critic built from the erroring model's own features is structurally
   blind to that model's errors. (This is the decoupling thesis of
   `GAN_REDESIGN_TWO.md`, and it is *why* the campaign moved to a pixel
   GAN.)
5. But the pixel GAN alone does not fix it, because DMD keeps training on
   the pulled-off tail and keeps making it worse.
6. **The fix is a weighted gate on DMD keyed to distance from the
   manifold**, using DMD's own error as the signal: as the error grows,
   DMD is downweighted.
7. **Assumption made explicit:** the data manifold *as DMD's one-step
   teacher sees it* is the same as GT, because GT is drawn from the same
   distribution that trained that teacher in the first place. So
   distance-from-GT is a usable proxy for distance-from-manifold.

## 2. What this buys — the division of labour

With the gate in place the two objectives **automatically partition the
rollout by drift**, with no hand-tuned schedule:

| region | drift | DMD | GAN |
|---|---|---|---|
| early rollout / near manifold | low | **ON** — teacher is a trustworthy oracle here | secondary |
| late rollout / far off manifold | high | **gated OFF** — its direction is unreliable and amplifies | **ON** — fixes the style of the drifted samples |

The GAN works on exactly the samples that have drifted off-distribution
(getting their *style* right); DMD works on the ones still near it. As
the GAN pulls a drifted sample back, its error falls, the gate re-opens,
and DMD is *automatically* allowed to look at it again. The handoff is
continuous and self-scheduling — that is the elegant part, and it is why
this is a gate rather than a depth cutoff.

It also removes the reason the pixel GAN's effect has been hard to
observe: DMD has been actively fighting it on the drifted tail.

## 3. WHAT EXISTS — and the polarity problem (WP-SURROGATE analysis)

**A gate is already in the tree and wired**:
`model/dmd_action_forcing.py::_dmd_mae_gate_weight` (:5063), applied to
`dmd_loss` at :6700, knobs `dmd_mae_gate_{enabled,r_full,ema,min_weight,exponent}`.
It is **not** dead code — `sbatch/smoke_stat_wave.sbatch` runs it on.

It measures exactly the right quantities:

```
m_real = |pred_real   - GT|     # TEACHER's error on this chunk
m_fake = |pred_student - GT|    # STUDENT's error on this chunk
r      = EMA(m_fake / m_real)
w      = clamp( ((r - 1) / (r_full - 1)) ** exponent , min_weight, 1 )
```

**But its polarity is the OPPOSITE of what §1 needs**, and this matters
enough that resurrecting it as-is would make the problem worse rather
than better:

| student state | `r` | existing gate | §1 wants |
|---|---|---|---|
| student ≈ or better than teacher | `r → 1` | **w → 0** (DMD off) | DMD **on** — we are near the manifold, this is where DMD works |
| student far worse than teacher | `r ≫ 1` | **w → 1** (DMD FULL) | DMD **off** — we are off-manifold, DMD amplifies |

So the shipped gate delivers **full DMD weight precisely on the drifted
tail** — the exact regime §1 identifies as harmful. It is not a
mis-implementation: it was built for a *different, also-real* problem,
recorded in its own docstring — stopping DMD from dragging a student that
has **surpassed** the teacher back down to the teacher's level. That
concern lives at the `r ≤ 1` end. The thesis above lives at the `r ≫ 1`
end. **They are two different cutoffs on opposite ends of the same axis.**

### The synthesis: a two-sided gate

Both concerns are real, so the correct object is a band, not a ramp:

```
        w
        1 |        ______________
          |       /              \
          |      /                \
   min_w  |_____/                  \______
          +----------------------------------> r
             1   r_lo            r_hi
        student≥teacher      far off-manifold
        (existing gate)      (THIS thesis)
```

* `r → 1`: student has caught the teacher — DMD would drag it back. Off.
  *(the existing gate's job; keep it)*
* middle band: teacher is a better oracle **and** we are near enough to
  the manifold for its direction to be trustworthy. **DMD on.**
* `r ≫ r_hi`: off-manifold, DMD direction unreliable and amplifying.
  Off. **This is the new half.** *(GAN takes over here)*

## 4. WHICH SIGNAL — DMD's own error, **not** any MAE-vs-GT quantity

*Researcher correction, 2026-08-24: "that one is gated on MAE so you need
to change it to track DMD's error itself." Taken, and it is the stronger
choice — an earlier draft of this section proposed `m_real` (teacher-vs-GT
MAE), which is still a GT-keyed proxy and carries the defect below.*

The signal is

```
e = | x0 - pred_real |          # on the supervised slots
```

— how far the frozen one-step teacher wants to move the student's **own**
sample. Three reasons this beats every MAE-vs-GT variant:

1. **It needs no GT.** Every MAE gate compares against `gt_target` and is
   therefore confined to supervised slots. The drifted tail is exactly
   where GT alignment is weakest, so a GT-keyed gate is least trustworthy
   in the regime it exists to police.
2. **It measures TEACHER COMPETENCE, not student badness.** Large `e`
   means the teacher strongly disagrees with this sample — the signature
   of being outside its training region, which is the actual precondition
   for its score estimate (and hence the DMD direction) being junk.
   Student-vs-GT error can be large simply because the student is bad
   while the teacher is perfectly competent, and that is the regime where
   DMD is **maximally useful**. A gate keyed on student error would
   switch DMD off precisely when it works.
3. **It is already computed.** `p_real = x0 - pred_real` in
   `_compute_kl_grad`; the CausVid normaliser already divides by its
   mean. Nothing new is estimated, and the gate keys on the same quantity
   the gradient is normalised by.

**Landed** as `_dmd_error_gate_weight` (`model/dmd_action_forcing.py`),
composing **multiplicatively** with the existing MAE gate — they cut
opposite ends of the same axis (§3), so together they form the band.

## 5. THE MEASUREMENT — do not guess the threshold, measure it

The thesis makes a **falsifiable, directly measurable** claim: that the
DMD update direction stops pointing toward the manifold beyond some
distance. That is a cosine, and it needs no new machinery:

```
align = cos( -grad_DMD ,  GT - x_student )      on the supervised slots
```

`-grad_DMD` is the direction the update actually moves the sample;
`GT - x_student` is the direction of the manifold. So:

* `align > 0` → DMD is pulling toward GT (the regime where DMD works)
* `align < 0` → DMD is pushing away (the regime the thesis predicts, and
  the regime the gate must switch off)

**The gate threshold is the crossover**: the value of `m_real` (and/or
`r`, and/or rollout depth) at which `align` changes sign. Plot `align`
against each and read it off. If `align` never goes negative, the thesis
is falsified for this configuration and the gate should not be shipped —
which is exactly why this is measured before it is wired.

This is cheap: every quantity already exists inside
`_dmd_mae_gate_weight` (`pred_real`, student latent, `gt_target`,
`gradient_mask`) plus `grad`, which is in scope at the call site. It is
an instrumentation change plus one short run, not a standalone probe.

## 6. Status

- [x] Thesis recorded (§1), division of labour (§2)
- [x] Existing gate located, read, polarity problem identified (§3)
- [x] Signal choice argued (§4)
- [x] Signal corrected to DMD's own error per researcher (§4)
- [x] `_dmd_error_gate_weight` landed — measurement-only until armed;
      thresholds have NO defaults and **raise** if enabled unset (same
      discipline as `pix_gan_weight` / `pix_r1_gamma`: a threshold that
      was never measured must not be silently inherited)
- [x] `align` + `dmd_err` + `gt_dist` on the step line — crossover
      readable from the terminal trace alone
- [x] Gate + config keys landed, **inert** (`dmd_err_gate_enabled: false`,
      thresholds `null`). 13/13 tests
      (`testing/test_dmd_manifold_gate.py`), including an explicit
      POLARITY test that fails if the ramp is ever wired the MAE gate's
      way round, and fail-loud tests on unset thresholds
- [x] `analysis/dmd_gate_crossover.py` — reports the crossover, or prints
      **"THESIS NOT SUPPORTED"** on a null so it cannot be fitted around
- [x] Per-frame `align`/`e` breakdown, so a band-mean masked by the
      bidirectional teacher's clean-GT anchor still shows as a gradient
      ACROSS the band (and a flat profile is a real negative)
- [ ] Measurement run — IN FLIGHT (holder 6109490, trace
      `logs/dmd_gate_trace_h6109490_155407.jsonl`, wiring verified in the
      run's own resolved config)
- [ ] Gate armed against the measured threshold
- [x] Depth study run on the SUPERSEDING fingerprint probe. Verdict
      `PROBE CANNOT RANK OFF-MANIFOLD DISTANCE -- GATE NOT VIABLE`,
      0/8 arms — see §6b for the two-basin reading and why a gate that is
      a pure function of the current score is ill-posed
- [x] Stateful ratchet landed on the fingerprint gate (default-off,
      byte-identical). `docs/DMD_FINGERPRINT_PROBE.md` §3.3

## 6b. THE DEPTH STUDY VERDICT, AND THE TWO-BASIN READING

**2026-08-24.** `analysis/dmd_fp_depth_study.py` returned

```
PROBE CANNOT RANK OFF-MANIFOLD DISTANCE -- GATE NOT VIABLE
```

**0 of 8 arms**, consistent across `t = 250 / 500 / 750`. Recorded as
printed. Not overwritten, not spun.

The verdict indicts the **criterion**, not the probe. The criterion was
**monotonicity** — "does `s` fall as the sample gets further off-manifold"
— and the measured geometry is **U-shaped**, which is the expected shape
once §1's thesis is taken seriously in both directions:

* near depth 0 the sample is near the **DATA manifold**, the teacher's
  field is locally restoring, `s` is high;
* at large depth the sample has been captured by the **MODEL'S OWN
  ATTRACTOR** — the degenerate fixed point §1 predicts — where the field
  is **also** locally restoring, so `s` is high **again**;
* the trough between them is the transition. **Measured: trough at depth
  16 in 11 of 12 (arm, timestep) series.**

DMD points toward its nearest attractor. Near the manifold that
attractor *is* the manifold; far away it is the model's own fixed point,
and `s` alone cannot distinguish them. The numbers confirm it:

```
deepest s minus depth-0 s, over 12 (arm, t) series:
    mean +0.0244, sd 0.0709, 9/12 POSITIVE
    = 1.78x the MEAN seed-noise floor (0.0137)
      but only 0.72x the WORST floor  (0.0338)
```

Below the worst seed floor: depth 0 and depth 32 are **statistically
indistinguishable** by `s`, and where they *do* differ the **DEEP** end
scores **HIGHER**.

**`s` and `m` are non-injective in depth**, so **any gate that is a pure
function of the current `m` is ill-posed** — the same reading means
opposite things. This applies to the `e`-keyed gate in this document too,
for the same structural reason and independently of §8's separate
falsification of `e`: a scalar read of "how far is the teacher from this
sample" has the same two-basin ambiguity.

**What the probe measures is "near SOME attractor"** — strictly weaker
than "near the DATA manifold", which is what §1's thesis needs. The claim
is narrowed here rather than defended.

**The ratchet is what converts the weaker signal into a usable gate.** It
adds the one fact the probe lacks and the trainer has — TIME ORDER, since
a ride *starts* on the manifold:

```
    w_t = min( w_{t-1}, f(ema(m)_t) )      reset at RIDE start
```

Full treatment, telemetry, reset semantics and the latch-forever hazard:
`docs/DMD_FINGERPRINT_PROBE.md` §3.3. The short version: the `min`
(diode) means the far branch's rising `m` can never re-open the gate, so
no threshold has to know where the trough is; the EMA (capacitor) closes
it progressively; and the reset is hooked to the trainer's ride boundary,
counted on the step line, and backstopped by a loud in-model reset,
because a running minimum that never resets silently zeroes DMD for the
rest of training.

This does **not** rehabilitate the verdict. The gate remains not viable
as a pure function of `m`. The ratchet is a different object — a gate on
`m`'s history within a ride — and it is only as sound as the assumption
that the ride began on the manifold.

## 7. Telemetry bug found while building this (worth generalising)

The first version of this instrumentation wrote `align` / `e` into
`dmd_log_dict`. **That dict is wandb-only — it never reaches stderr.** A
step-line lookup against it would have found nothing, forever, silently,
and the measurement would have looked merely "slow" rather than broken.

That is the same shape as the two NULL surrogate smokes and the
`surrogate_grad_check_every` knob that was threaded but never called
(`docs/WP_SURROGATE.md` §4d). The working path in this file is an
attribute on the model read via `getattr` at the step line — which
`_last_dmd_mae_gate_weight` has been doing all along, so the pattern was
already there to copy.

Generalised rule, now three-for-three in this campaign: **adding a key is
not the same as the key arriving.** Trace a new metric to the surface it
is supposed to appear on, once, before relying on it — and prefer the
mechanism something comparable already uses over inventing a second one.
- [ ] Verified: pixel-GAN effect observable once DMD stops fighting it

---

## 8. FIRST MEASUREMENT — align is WORSE THAN RANDOM (n=32, GT context)

Gate disabled; measurement only. `align = cos(-grad, GT - x0)` on the
supervised slots, 32 DMD calls.

| | |
|---|---|
| align negative | **32 / 32** |
| thirds (call order) | −0.568 / −0.348 / −0.451 |
| sign flips | **0 / 31** |
| per-frame, band start → end | −0.403 → **−0.470** |
| crossover | **none** — never positive anywhere |

### Why "worse than random" is the load-bearing phrase

Numerically simulated, not asserted (independent draws, D=4096):

| if the DMD update moved x0 toward… | align vs a specific GT |
|---|---|
| the GT sample | +1.000 |
| the distribution MEAN (mode-seeking) | **+0.707** |
| a RANDOM direction | 0.000 |
| a teacher guess 2σ off-mean | **+0.316** |

Every benign hypothesis is **positive**. An earlier draft of this doc
blamed mode-seeking for the negative sign — **that was wrong and is
retracted**: mode-seeking predicts +0.707. Measuring −0.3…−0.8 is *worse
than a random direction*, which no benign account produces.

### Per-frame gradient

align degrades from the band's start to its end (−0.403 → −0.470). At
n=8 this looked flat (−0.008) and I reported it as flat; at n=32 it is
−0.066. **A second instance of over-reading a small sample in this
session** (the other being the surrogate backbone `c_corr` story, also
retracted). Both times the early trend inverted or vanished. The rule the
campaign already has — the seed floor — exists for exactly this, and
citing it at other people's results while not applying it to my own is
the failure mode to name.

## 9. WHAT THE TEACHER ACTUALLY SEES (answering "is that how we do it?")

Verified in code, not assumed:

* **Cleanest rung: ALREADY CORRECT.** `dmd_rolling_ctx_last_rung=true`
  overrides the overlap context from the prior iter's *exit rung* — "a
  random-rung estimate (t up to 1000 => near-noise x0 ~25% of the time)"
  — to the **finish-denoised** slab, which is what the KV cache commits
  and what inference uses. That is the diff_t≈0 end.
* **noisy_x: ALREADY ROLLOUT on k≥2.** With
  `dmd_42f_rolling_sup_new=true` and overlap>0 the whole 42f window rolls
  with the student: `[n_ctx STUDENT ctx | new frames (supervised) | npb
  GT future scaffold]`. GT context in the noisy half is **iter-1 only**.
  The only GT left there is one `npb` scaffold chunk at the
  structurally-OOD newest slot, which must be filled for v14's exact 21f
  geometry.
* **clean_x: STILL GT.** "clean_x stays positional GT and rolls forward
  with the window." **This is the anchor**, and the prime suspect for the
  sub-random align: the teacher is conditioned on the GT trajectory while
  denoising a band from the STUDENT's. Conditioned on one trajectory,
  scoring another, its prediction points where the *GT continuation*
  would have gone — not toward fixing the student's actual sample.

## 10. ARMS

* **ARM 1 — whole thing rollout** (`dmd_42f_clean_self_forward=true`):
  overwrite the entire student-covered span of `clean_x` with the
  student's own rolled content. Both halves then come from ONE
  trajectory. IN FLIGHT.
  * Note: `dmd_42f_clean_self` cannot be used — it raises against
    `dmd_42f_clean_drift_enabled`, which the base recipe sets (a real
    mutual-exclusion guard; it killed the first attempt in ~1 min).
    `clean_self_forward` is drift-compatible.
  * The run prints `*** G4 rewrite never fired ***` if the overwrite does
    not happen, so this cannot come back as a silent null.
* **ARM 2 — whole of noisy_x rollout**: mostly already true (above); the
  delta is only the `npb` GT scaffold at the OOD slot. Not yet run.
