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
- [ ] **Correlation against the measured `align` cliff — the arming
      gate.** Still the precondition. If `m` does not track the cliff,
      it is not measuring what it claims and must not be armed.
- [ ] Fill the six Class-B values from `analysis/dmd_fp_depth_study.py`

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
