# DMD SELF-FINGERPRINT — measuring off-manifold-ness with DMD alone

**Researcher directive, 2026-08-24.** Supersedes the realism-critic probe
that briefly existed in the trainer (removed).

## 1. Why no external critic can do this

The rejected design used a pretrained pixel disc (SAM2 / DINOv2 /
ConvNeXt) input-gradient as a "toward real" direction. Two fatal
objections, the second decisive:

1. **They disqualify everything.** Those backbones were trained on
   natural images. Our samples are VAE reconstructions of dashcam
   latents — GT *and* rollout alike sit outside their training
   distribution. A critic that calls both "not of this world" is
   measuring its own distribution mismatch, not ours.
2. **It answers the wrong question.** We are trying to establish **how
   wrong DMD is**. Substituting another model's opinion measures *that
   model*. The only instrument that can qualify DMD's competence is DMD.

## 2. The probe

The insight: **DMD's restoring response to a controlled perturbation is a
fingerprint of where the sample sits relative to the manifold DMD knows.**

For a sample `x`:

```
  delta   = PERTURB(x)            # NON-gaussian, structured
  x_p     = x + delta
  x_t     = noise(x_p, t)         # gaussian, the diffusion forward process
  x0_hat  = teacher(x_t, t)       # DMD's one-step denoise
  r       = x0_hat - x_p          # the RESTORING RESIDUAL
  s(x)    = cos(r, -delta)        # does DMD push back along the perturbation?
```

* **On-manifold `x`:** DMD recognises `x_p` as displaced and pulls it
  back. `r` acquires a component anti-parallel to `delta`, so `s` moves
  **up**.
* **Off-manifold `x`:** DMD no longer knows where the manifold is here.
  Its response carries no restoring component and `s` falls to its
  no-information floor.

`s` is therefore a direct, DMD-only *ordinal* estimate of how well DMD
can localise the manifold at this point — which is exactly the quantity
the gate needs, and exactly what "how off is the DMD" means.

### `s` has NO absolute meaning — the floor is not zero

**An earlier revision of this doc claimed the off-manifold limit is
`s → 0`. That is wrong, and it is wrong by a lot.** Do not read a raw
`s` against 0; read it only through the two-anchor calibration in the
next section.

The reason is the `−x_p` term in `r = x0_hat − x_p`. A teacher that has
*no idea* where the manifold is does not return zero response — it
returns some `x0_hat` uncorrelated with `x_p`, and the residual is then
`r ≈ −x_p`, which is a large, systematic direction, not noise. Whether
that direction happens to align with `−delta` is fixed by the geometry
of the perturbation, not by the teacher's competence.

And our perturbations are *replacements*, not additions: every mode in
the table below builds `delta` from a rearrangement of the sample's own
content (`d = P(x) − x`, then renormalised to `scale·‖x‖`). Write
`a = x`, `b = P(x)`; the shuffles preserve the norm (`‖a‖ ≈ ‖b‖ ≈ n`)
and destroy the alignment (`⟨a, b⟩ ≈ 0`). With `k = scale/√2`:

```
  delta   =  k (b − a)
  x_p     =  (1 − k) a  +  k b
  −delta  =  k (a − b)
```

For an oblivious teacher (`x0_hat ≈ 0`, the latent mean), `r ≈ −x_p` and

```
  s  =  cos(−x_p, −delta)  =  (2k − 1) / ( √2 · √((1−k)² + k²) )
```

which is **≈ −0.71** as `scale → 0` and **≈ −0.62** at `scale = 0.15`.
The `−x_p` term anti-correlates with `−delta` *by construction*, because
`x_p` is largely made of the replacement material `b` that `−delta`
points away from.

Verified against the shipped `_dmd_fp_perturb` with a literal
`x0_hat = 0` teacher (`s` over 4 draws, `[1,6,4,16,16]` gaussian `x`):

| mode | scale 0.05 | 0.15 | 0.30 | formula (0.05 / 0.15 / 0.30) |
|---|---|---|---|---|
| `channel_rot` | −0.673 | −0.610 | −0.488 | −0.681 / −0.619 / −0.499 |
| `patch_shuffle` | −0.657 | −0.591 | −0.460 | −0.681 / −0.619 / −0.499 |
| `hf_scramble` | −0.567 | −0.490 | −0.354 | −0.681 / −0.619 / −0.499 |

The formula is tight for `channel_rot` / `patch_shuffle`, where the
rearrangement really does decorrelate (`⟨a, b⟩ ≈ 0`). `hf_scramble` sits
consistently *above* it because it scrambles only the high-frequency
band and leaves the low frequencies intact, so `⟨a, b⟩ ≫ 0` and the
`−x_p` term is less hostile — i.e. **the floor is mode-dependent as well
as scale-dependent**, which is another reason no absolute reading of `s`
is safe. The measured in-training baseline (2026-08-24) is **≈ −0.4**:
the same strongly-negative regime, a little above the oblivious floor.

Consequences, all of which the code already relies on:

* **`s < 0` is not evidence of anything.** It is the default state. Only
  `s` *relative to the anchors* carries information.
* **`s ≈ 0` is not "no information" — it is well above the floor**, and
  on this scale would be a comparatively good score.
* **The span `s_gt − s_off` is the unit.** Everything downstream
  (`m`, the per-frame weights, the gate) is computed from
  `(s − s_off) / (s_gt − s_off)` for exactly this reason, and
  `dmd_fp_degenerate` fires when that span collapses.

### Why the perturbation must be NON-gaussian

Gaussian displacement is the forward process the teacher was **trained to
invert**. Perturbing that way and asking it to denoise measures nothing
but its ordinary competence — it would score high everywhere. A
*structured* perturbation moves the sample off the data manifold in a way
the noise schedule does not model, so undoing it requires the teacher to
actually know where the manifold is. That is the discriminating case.

Implemented perturbations (`dmd_fp_perturb`):
| mode | what it breaks | why |
|---|---|---|
| `patch_shuffle` | local spatial arrangement | destroys layout, preserves marginal statistics |
| `hf_scramble` | fine texture, keeps low frequencies | targets exactly the band our texture measurement found dead (target/GT ≈ 0.37) |
| `channel_rot` | cross-channel structure | latent channels are not interchangeable; breaks their joint statistics |

### Calibration — the reference fingerprints

`s` is only interpretable against known anchors, and we have two by
construction:

* `s_gt` — measured on **GT**, which is on-manifold *by definition*.
  This is the ceiling.
* `s_off` — measured on **GT + a large structured corruption**, which is
  off-manifold *by construction*. This is the floor.

The normalised score
```
  m(x) = (s(x) - s_off) / (s_gt - s_off)     clamped to [0, 1]
```
is then "fraction of the way from provably-off to provably-on", in DMD's
own terms, with no external model and no reference to any particular GT
frame. **This is the confound fix**: a rollout on a different-but-valid
branch of a multimodal manifold still scores high, because DMD still
restores it — whereas a GT-referenced cosine would have marked it wrong.

## 3. Relationship to the gate

The 2026-08-24 measurement (`docs/DMD_MANIFOLD_GATE.md` §8) killed the
original gate signal: `e = |x0 - pred_real|` was **flat** (~0.18) across
the band while `align` swung **+0.094 → −0.530**. `e` does not predict
alignment, so a gate keyed on it gates on noise.

`m(x)` is the replacement, and unlike `align` it needs no GT at all. The
weight is an **explicit** ramp, not `w = m`:

```
  u = clamp( (m - m_lo) / (m_hi - m_lo), 0, 1 )
  w = min_weight + (1 - min_weight) * u ** exponent
```

`w = m` (what the first implementation did) silently assumes the gate
response is *linear in `m` across the whole of [0, 1]* — an assumption,
not a measurement, and one that would never have been revisited once it
ran. `m_lo` / `m_hi` / `exponent` make the response curve a thing that
gets measured. Note the floor is applied **affinely**, not as a
`clamp(min=)`: with a clamp, `min_weight=0.3` and `exponent=2` flattens
every `u < 0.55` onto the floor, killing half the calibrated range
without saying so.

DMD is weighted by how well DMD itself can localise the manifold at that
frame. Where it cannot, the term is switched off and the (decoupled,
pixel) GAN owns the sample — which is the division of labour the whole
gate exists to create.

**Validation before arming:** `m` must correlate with the `align` cliff
already measured (positive at band frames 9–13, collapsing past 14). If
it does not, `m` is not measuring what it claims and must not be shipped.
Both are traced side by side for exactly this comparison.

### 3.1 Class A vs Class B — enforced, not advisory

| class | keys | default | why |
|---|---|---|---|
| **A — measurement** | `dmd_fp_every`, `dmd_fp_perturb`, `dmd_fp_seeds` | working defaults | the probe and the depth study must be able to run before any calibration exists |
| **B — calibration** | `dmd_fp_scale`, `dmd_fp_off_scale`, `dmd_fp_m_lo`, `dmd_fp_m_hi`, `dmd_fp_gate_exponent`, `dmd_fp_gate_min_weight` | **`null`** | these ARE the response curve; they are measured per teacher/data pair by `analysis/dmd_fp_depth_study.py` |

`dmd_fp_gate_enabled=true` with **any** Class-B value unset raises a
`ValueError` naming exactly which keys are missing. The old in-code
defaults (`scale=0.15`, `seeds=2`, `exponent=1.0`, `min_weight=0.0`, and
an off-anchor buried as the literal `max(0.6, scale*4)`) were guesses. A
guessed default that silently works is the failure mode this campaign
keeps rediscovering, so they are gone rather than kept "as a start".

**Probe-only operation** (`dmd_fp_every>0`, gate off) still requires
`dmd_fp_scale` and `dmd_fp_off_scale` explicitly — the probe does **not**
sweep internally and does **not** substitute a number. An internal sweep
would make the probe's own output depend on a grid nobody chose, and the
scale sweep is the study's job (it varies scale *across* runs and reads
off which one discriminates). An unset scale is therefore a loud error at
the moment the probe is switched on. A probe-only run produces the
measurement (`dmd_fp_m`, `dmd_fp_m_frame_*`) and **no weights** — weights
are only computed once the whole response curve is present.

### 3.2 Telemetry

On the **step line** (not just wandb — `dmd_log_dict` never reaches
stderr, a bug already caught twice here; the trainer reads model
attributes via `getattr`): `fp_m`, `fp_w`, `fp_wmin`, `fp_wmax`, and
`fp_share` = mean applied weight as a fraction of the unattenuated 1.0,
i.e. **how much of DMD survives the gate**. Every slot starts as `None`,
so an unavailable diagnostic is an *absent key*, never a forgeable `0.0`
(which would read as "DMD fully gated off" — a real regime).
`dmd_fp_gate_w_age` (wandb) counts DMD calls since the last probe, so a
gate attenuating on frozen weights is visible.

### 3.3 THE DEPTH STUDY RESULT — two basins, and why a pure function of `m` is ill-posed

**The verdict, unaltered.** `analysis/dmd_fp_depth_study.py` returned

```
PROBE CANNOT RANK OFF-MANIFOLD DISTANCE -- GATE NOT VIABLE
```

**0 of 8 arms**, consistent across `t = 250 / 500 / 750`. That verdict
stands and is not being reinterpreted, softened, or overwritten. It is
recorded here exactly as the study printed it.

What it is a verdict *about* matters, though. The study's criterion was
**monotonicity**: it asked whether `s` (and therefore `m`) *falls* as the
sample is pushed further off-manifold, i.e. whether `m` can RANK
off-manifold distance. The measured geometry does not satisfy that
criterion, and there is a reason it does not.

#### The measured curve is U-shaped

| | |
|---|---|
| shape of `s` vs depth | **U-shaped** |
| trough location | **depth 16**, in **11 of 12** (arm, timestep) series |

Two basins, and the probe's mechanism is locally identical in both:

* **Near depth 0** the sample is near the **DATA manifold**. The
  teacher's field is locally restoring there, so a structured
  displacement gets pushed back and `s` is high.
* **At large depth** the sample has been captured by the **MODEL'S OWN
  ATTRACTOR** — its degenerate fixed point. The field is *also* locally
  restoring there, so a structured displacement gets pushed back again
  and `s` is high **again**.
* The **trough between them is the transition**, where the sample is
  near neither.

DMD points toward its nearest attractor. Near the manifold that
attractor **is** the manifold; far away it is the model's own fixed
point. **The probe cannot tell those two apart from `s` alone.**

#### The two ends are not separable, and the deep end scores HIGHER

```
deepest s minus depth-0 s, over 12 (arm, t) series:
    mean  = +0.0244      sd 0.0709      9/12 POSITIVE
    = 1.78x the MEAN seed-noise floor  (0.0137)
      but only 0.72x the WORST floor   (0.0338)
```

Below the worst-case seed floor, so depth 0 and depth 32 are
**statistically indistinguishable** by `s` — and in the direction they do
differ, the **DEEP** end scores **higher**.

#### Consequence: `s` and `m` are NON-INJECTIVE in depth

The same reading means opposite things. Therefore **any gate that is a
pure function of the current `m` is ill-posed** — not badly tuned, not
under-calibrated, *ill-posed*. No choice of `m_lo` / `m_hi` / exponent
fixes a mapping whose input does not determine its answer.

#### What the probe actually measures

`m` measures **"near SOME attractor"**. That is strictly **weaker** than
**"near the DATA manifold"**, which is what the gate needs and what §2
implicitly claimed. The claim is hereby narrowed, not defended.

#### The ratchet is what converts the weaker signal into a usable gate

The one fact the probe does not have and the trainer does is **TIME
ORDER**: a ride *starts* on the manifold and drifts away from it. Adding
that turns a non-injective instantaneous reading into a usable one.

```
    w_t = min( w_{t-1}, f(ema(m)_t) )        reset at RIDE start
```

* **DIODE** (`min`). The weight never increases within a ride, so the far
  (attractor) branch's rising `m` can **never re-open** the gate. The far
  side becomes automatically "no" **without any threshold needing to know
  where the trough is** — there is deliberately no trough-location
  constant anywhere in the implementation.
* **CAPACITOR** (`dmd_fp_ratchet_ema`). An EMA on `m` **before** the
  `m_lo`/`m_hi`/exponent/`min_weight` ramp, so the gate closes
  progressively instead of snapping shut on one noisy probe. On `m` and
  not on `w` because the ramp is nonlinear and clamped at both ends: an
  EMA applied after the clamp cannot recover what the clamp destroyed.
* **ELEMENTWISE.** The running minimum is a `[F]` vector and the `min` is
  taken frame by frame against the same frame index, so the per-frame
  path is preserved exactly — the align cliff lives inside a single band
  and a scalar ratchet would average it away. Frame *index* is the
  carrier of identity across calls, not the underlying content (which
  shifts by the rollout stride each roll) — the same convention the
  per-frame weights already use.

**This does not rehabilitate the study's verdict.** The gate is still not
viable *as a pure function of `m`*. The ratchet is a different object: it
is a gate on `m`'s **history within a ride**, and it is viable only to
the extent that "the ride began on the manifold" is true.

#### RESET — the dangerous direction

A running minimum that never resets **latches shut and silently zeroes
DMD for the rest of training**. That is the failure mode of this design,
so the reset is:

* **explicit** — `reset_dmd_fp_ratchet()`, called by the trainer at the
  one line that means "new ride" (`self._chunks_in_current_ride = 0` in
  `_streaming_step`), **not** keyed on a step counter;
* **unconditional** — it runs whether or not the ratchet is armed, so
  default-off runs exercise the call site;
* **counted** — `fp_rt_rst` on the step line. If rides turn over and that
  number stops moving, the ratchet is latching;
* **backstopped** — `_dmd_fp_ratchet_observe_depth` resets **loudly** on
  stderr if ride depth ever goes *backwards* without the explicit reset.
  The test is a **strict** decrease: several DMD calls can share one
  depth, and a `<=` test would reset on every repeat and silently defeat
  the diode — the same class of bug in the opposite direction.

A mid-ride **shape change raises** rather than resetting: a reset there
would re-open the gate on the deep end, which is exactly what the diode
exists to prevent.

#### Ratchet telemetry (step line)

| key | meaning |
|---|---|
| `fp_rt_w` | mean running-minimum weight in force |
| `fp_rt_shr` | **SHARE** of the un-ratcheted gate weight retained — the ratchet's own marginal effect (`fp_share` remains the absolute share of DMD surviving everything, and now includes the ratchet) |
| `fp_rt_lat` | 1.0 once the diode has **BLOCKED a rise** this ride |
| `fp_rt_d` | ride depth at which the running minimum **last decreased** |
| `fp_rt_rst` | ride resets the hook has seen |

`fp_rt_d` is the **empirical cross-check on the trough**. Training-time
depth is known exactly (`_chunks_in_current_ride`), and on a U-shaped `m`
the running minimum stops decreasing *at the trough*. If `fp_rt_d`
settles near ~16 that independently corroborates the offline study's
depth-16 measurement from a completely different code path. **Caveat
worth stating up front:** the ratchet resets per ride, so it can only
observe depths the ride actually reaches — `max_rolls_per_ride` well
below 16 means `fp_rt_d` will sit at the ride's own cap and corroborates
nothing.

#### Standing hazards when arming the ratchet

Recorded rather than silently handled:

1. **The EMA's time base is the DMD call, not the probe.** With
   `dmd_fp_every > 1` the probe's `m` is stale between refreshes and the
   capacitor charges toward the same stale reading on every intervening
   call — effective smoothing is *weaker* than the knob reads, and the
   diode takes several redundant minima of one measurement. Deliberately
   not special-cased (that would make one knob's meaning depend on
   another's). Run the ratchet at `dmd_fp_every: 1`, and watch
   `dmd_fp_gate_w_age` if you do not.
2. **`dmd_fp_gate_min_weight = 0` + one bad probe = DMD off for the whole
   ride.** A single reading at or below `m_lo` pins the running minimum
   at zero and the diode never lets it back up. That is the design
   working as specified, but the probe's seed-noise floor is 0.0137 mean
   / 0.0338 worst, so *one noisy draw* can do it. The capacitor is the
   intended defence; a nonzero `min_weight` is the belt-and-braces one.
   Not enforced in code — `min_weight` is a measured Class-B value and
   the ratchet must not overrule a measurement — but it is the first
   thing to check if `fp_share` collapses.
3. **Per-rank state under DDP.** The ratchet is derived from the
   per-rank probe and holds per-rank state, so ranks can carry different
   weights. That was already true of the un-ratcheted per-frame gate and
   the ratchet adds no collectives, so there is no new hang surface — but
   the DMD term is then a cross-rank average over *differently* weighted
   local terms, which is worth knowing before reading `fp_share` as a
   global quantity.

Class-B discipline applies: `dmd_fp_ratchet_ema` ships `null` and arming
`dmd_fp_ratchet_enabled` without it raises, naming it. Arming the ratchet
with `dmd_fp_gate_enabled=false` also raises — it wraps that gate's
weight, so with the gate off it would be silently inert.

## 4. Cost

One extra teacher forward per probed sample per noise draw. Gated by
`dmd_fp_every` (0 = off) and `dmd_fp_seeds`, so it is a periodic
diagnostic rather than a per-step tax. The reference fingerprints
`s_gt` / `s_off` are computed on the same call from the GT already in
scope, so calibration costs no extra data plumbing.

## 5. Status

- [x] Design (this doc); realism-critic probe removed from the trainer
- [x] `_dmd_fingerprint_probe` implemented
- [x] Gate fully wired config → consumer, per-frame, with step-line
      telemetry; `dmd_err_gate_*` marked SUPERSEDED in the config
- [x] Armed but **DEPOPULATED**: every Class-B value ships `null` and
      enabling the gate without them raises. One config edit from live,
      structurally incapable of running on invented numbers.
      Tests: `testing/test_dmd_fp_gate.py`
- [x] In-training denoise hook (`_install_dmd_fp_denoise_fn`) rewritten
      after four simultaneous defects — see §6. Tests:
      `testing/test_dmd_fp_gate.py`, one per defect, each verified red
      against that defect restored.
- [x] **Depth study RUN. Verdict: `PROBE CANNOT RANK OFF-MANIFOLD
      DISTANCE -- GATE NOT VIABLE`, 0/8 arms, t=250/500/750.** Recorded
      as printed; see §3.3. The criterion was MONOTONICITY and the
      measured geometry is U-shaped (trough at depth 16, 11/12 series),
      with the two ends statistically indistinguishable (deepest minus
      depth-0: mean +0.0244, 9/12 positive, 0.72x the WORST seed floor)
      and the deep end scoring HIGHER where they differ. `m` is
      NON-INJECTIVE in depth, so a gate that is a pure function of `m`
      is ill-posed.
- [x] Ratchet ("capacitor-diode") landed — the stateful per-ride
      `w_t = min(w_{t-1}, f(ema(m)_t))` that converts the weaker
      "near SOME attractor" signal into a usable gate by adding TIME
      ORDER (§3.3). Default-off, byte-identical; `dmd_fp_ratchet_ema`
      ships `null` and arming without it raises. Tests:
      `testing/test_dmd_fp_gate.py`, each verified red under a targeted
      mutation.
- [ ] **Correlation against the measured `align` cliff — the arming
      gate.** Still the precondition. If `m` does not track the cliff,
      it is not measuring what it claims and must not be armed. NOTE
      this is now a weaker precondition than it reads: §3.3 shows `m`
      answers "near SOME attractor", so tracking the cliff is necessary
      and not sufficient.
- [ ] Fill the six Class-B values from `analysis/dmd_fp_depth_study.py`
- [ ] Measure `dmd_fp_ratchet_ema` against the probe's seed-noise floor
      (mean 0.0137 / worst 0.0338) — it is the only ratchet knob and it
      ships `null`
- [ ] **Confirm rides are deep enough to reach the trough.** The ratchet
      resets per ride; if `max_rolls_per_ride` << 16 it can never observe
      the far basin and the diode has nothing to block.

## 6. Known-defect history — the in-training denoise hook

**2026-08-24.** The hook `_compute_kl_grad` installs as
`self._dmd_fp_denoise_fn` — the teacher forward the probe calls — shipped
with **four** defects at once. Recorded here because the mechanism
(a convoluted body that hid its own faults) matters more than any one of
them:

1. **It never noised the sample.** The noising line was
   `_xt = self._add_noise_to_x0(_xp, _n, _t) if hasattr(self,
   "_add_noise_to_x0") else _xp`. **No method of that name exists
   anywhere in the repo**, so the `hasattr` guard was permanently false,
   `_xt = _xp`, and the gaussian forward process — the entire mechanism
   of the probe — never ran. `_n = torch.randn_like(_xp)` was drawn and
   discarded. The probe was measuring the teacher's response to an
   *unnoised, structurally-perturbed* sample, which is not `s(x)`.
2. **It returned the wrong tuple element.** It took `[0]`. In this
   codebase `real_score` returns `(flow, x0)` — see the canonical calls
   `flow_real_cond, pred_real_image_cond = self.real_score(...)` and
   `_flow, _x0 = self.real_score(...)` in the same method. `[0]` is the
   **flow** prediction, so `r = x0_hat − x_p` was actually `flow − x_p`.
3. **It omitted `**tf_kwargs_real`.** Every other `real_score` call in
   `_compute_kl_grad` passes it (that is the clean-half / aug-t
   contract). The probe therefore ran the teacher off-contract relative
   to how that same teacher is used everywhere else — so even a correct
   `s` would not have been about the deployed teacher.
4. **It called `real_score` up to three times per invocation** — once
   for an `isinstance` check, once for the value, once in a fallback
   branch. 2–3 full TF forwards for one result.

**No reported result came through this path.** Verified, not assumed:
the hook, the probe, the `dmd_fp_*` config block and this doc were all
added on 2026-08-24 and are *uncommitted working-tree changes*
(`git log -S _dmd_fp_denoise_fn -- model/dmd_action_forcing.py` and
`git log -S dmd_fp_every -- configs/action_forcing_phase3_dmd.yaml` are
both empty); the only config carrying the keys ships
`dmd_fp_every: 0` and every Class-B key `null` with
`dmd_fp_gate_enabled: false`; no sbatch or pipeline script sets
`dmd_fp_every`; and no training log contains any `dmd_fp_*` telemetry.
`analysis/dmd_fp_depth_study.py` is unaffected — it deliberately builds
its **own** denoise hook with `scheduler.add_noise` rather than using
`model._dmd_fp_denoise_fn`, for exactly this reason.

**The fix, and why it is shaped the way it is.** The hook body moved to
`_install_dmd_fp_denoise_fn` (CPU-testable against a stub `self`, as the
other `dmd_fp` helpers already are) and is now: noise with the same
`scheduler.add_noise` call every other noising site in the file makes;
**one** `real_score` call; explicit `_flow, x0_hat = ...` unpack
returning `x0_hat`; `**tf_kwargs_real` passed through. No capability
guards, no `isinstance` branch, no fallback — the convolutedness is what
hid four defects, so simplicity is a correctness requirement here and
not a style preference.

**A hook that cannot be built is `None`, never an un-noised
passthrough.** If `self.scheduler` cannot noise, the hook is not
installed, the probe does not run, and a message says so on **stderr**
(`dmd_log_dict` is wandb-only). A silently-`None` hook means the probe
silently does not run; a silently-un-noised hook is worse — it reports a
number for the wrong quantity. The old bare `except Exception:
self._dmd_fp_denoise_fn = None` did the first and its fallback did the
second.
