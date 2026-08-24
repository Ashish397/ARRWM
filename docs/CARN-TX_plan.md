# CARN-TX — a learned rollout→GT style operator, applied in the loop

**Date**: 2026-08-24. **Status**: researcher-approved design, pre-implementation.
**Decision**: researcher directive, this date — *"copy the CARN network and make
a version that, instead of looking between 1-chunk and 3-chunk rollouts, looks
at 3-chunk rollouts and GT to compare the stylistic difference."* Approved with
the three risk mitigations in §4.

**Gates**: `carn_tx_enabled` (training of the operator, default **false**) and
`carn_tx_apply_lambda` (application strength, default **0.0**). Both byte-identical
off. **Only the main agent edits this file**; the implementer reports into
`docs/WP_CARNTX.md`.

---

## 1. Why this exists, in one paragraph

The measured texture collapse (fabricated HF ×2.2, fy/fx anisotropy 1.10→0.36,
luma-kurtosis wash) is **directional and grows with rollout depth**
(1.49×→2.21×→3.6×) — an autoregressive amplification loop where corrupted
context begets worse texture. The restoring-force framing (researcher,
2026-08-24) says a corrector need not be perfect per step: if it knows the axes
along which in-distribution content drifts OOD and points back inward, the
correction **compounds**. CARN-TX implements that force as an **applied map at
the commit site** rather than a gradient hint through the generator's weights:
corrected context feeds the next chunk, so deep drift never develops and the
operator always runs near the regime it was trained on. Precedent at the same
hook: the CARN seam affine (per-channel mean/std re-anchor,
`carn_seam_affine_lambda=0.5` in live arms) already recovers **~80 % of the
drift walk**. CARN-TX generalises that rung from affine to a learned,
texture-capable operator. It is **supervised regression onto the stationary
class** (GT style): no adversarial game, no D/G balance, no weight/γ
calibration.

Relationship to the pixel GAN (B1) and the projected-backbone work (surrogate
agent): **complementary, not competing** — the operator edits the *state*, the
critic edits the *generator*. First CARN-TX arm runs **without** the pixel GAN
for attribution (single-variable arms).

## 2. What gets built

### 2.1 The operator
- New module `model/carn_tx.py`: **clone** `model/forward_noiser.py::ForwardNoiser`
  (the validated CARN net — cycle training converged, run 5285127, cycle loss
  0.009→0.001), rename, strip what the clone does not need. Do NOT modify
  `forward_noiser.py` itself.
- **Residual parameterisation**: `T(z) = z + f(z)`, modest capacity (same order
  as the noiser). Input/output: one latent chunk `[B, F=npb, 16, 60, 104]`
  (student latent domain — Case 1 established the artefact enters in the
  latent, so a latent operator can reach it; no decode in the loop).
- fp32 module; cast at call boundaries per the `_pix_vae_dtype` lesson (derive
  dtypes, never hardcode; bf16 latents meet fp32 weights here too).

### 2.2 Training pairs — free, and ONLINE by the stationarity argument
Pair = (rollout chunk at depth d, **GT chunk of the same ride at the same
timestamps**). Both already flow through the trainer: the streaming state holds
`ride_latents_window` (GT) and the detached rollout chunks; the exposure work
already tags chunk depth. **Train ONLINE inside the existing trainer**,
flag-gated: the input class (current rollout style) is the *moving* class, so
the corrector must track it — the same reasoning that withdrew the fake replay
ring ("cache the stationary class, never the moving one"; here: *regress onto*
the stationary class, *sample* the moving one live). No offline LMDB phase.

### 2.3 Losses — the three §4 mitigations are load-bearing here
- **Pointwise residual term** (L1/Huber) between `T(z_roll)` and `z_gt`,
  **weighted DOWN with depth / content divergence**: at depth the rollout's
  *content* has legitimately diverged from GT, and an unweighted pointwise loss
  would teach content repainting. Gate the pointwise term to shallow pairs
  (weight `w_pt(d)`, configurable, default decaying by depth bin).
- **Style term carries the deep pairs**: match per-channel spatial statistics
  of `T(z_roll)` to `z_gt` — channel mean/std plus a **spatial power-spectrum
  envelope loss in the latent** (the measured collapse axes: HF power and the
  fy/fx anisotropy). Differentiable, cheap, content-free.
  **Recorded supersession**: the standing rule "texture statistics are
  instruments, never loss terms" was scoped to the *adversarial* texture path;
  the researcher's corrector directive supersedes it **for CARN-TX only**. Say
  so in the code comment; the instruments in `analysis/texture_stats.py` remain
  eval-only and are NOT imported into the loss (write a small latent-domain
  spectral loss instead — the eval instrument must stay independent of the
  thing it judges).
- **Optional cycle term (Phase 2, off by default)**: also train g: GT→rollout
  style and add cycle consistency, per the original CARN. Only if the plain
  corrector under-delivers.

### 2.4 Depth weighting (mitigation for identity collapse)
Oversample the deepest chunks the training regime produces: sample training
pairs uniformly over depth **bins** with a guaranteed minimum share for the
deepest bin (`carn_tx_depth_min_share`, default 0.4). The exposure build's
depth histogram pattern is the template; emit the same histogram for pairs
actually trained on. If trained mostly at depth≈0 where rollout≈GT, T learns
identity and does nothing at depth — this is the failure the weighting exists
to prevent, and the histogram is what proves it prevented.

### 2.5 Application — the commit hook, train/eval BOTH
```
z_commit' = z_commit + λ · (T(z_commit) − z_commit),   λ = carn_tx_apply_lambda
```
- **Order**: T first, THEN the existing seam affine (affine re-anchors
  mean/std as the outer safety net; T handles structure inside it). Compose,
  do not replace.
- **Hook sites — BOTH, or neither** (the A2/A23 lesson: training and inference
  must commit the same thing):
  * training: the commit path in `pipeline/action_forcing_training.py` — grep
    `carn_seam_affine_lambda` (~:1650-1680 and the second path ~:2593); put
    T at the same site(s), same idiom.
  * inference/eval: `utils/eval_causal_AR.py` — grep `_apply_carn_seam_affine`
    (:169, consumed at :531). Same λ, operator loaded from the checkpoint.
  A flag that reaches one consumer and not the other is the **seam class** of
  defect; the config→pipeline propagation must follow the verified
  `pix_finish_grad_enabled` pattern (trainer sets the attribute on
  `self.pipeline` — the object whose class performs the read), and the seam
  test must drive config→BOTH consumers and be mutation-controlled.
- **Scope of application**: the committed **context** (and the rendered eval
  output) ONLY. The tensors the DMD losses consume stay raw — gradients are
  not laundered through T, and the training recipe for the student is
  untouched.
- **λ ramp**: `carn_tx_apply_start_step` + linear warmup; λ=0 exactly until
  the operator has trained (`carn_tx_min_pairs_before_apply`, fail-loud if
  applied earlier). Applying an untrained T is worse than nothing.

### 2.6 Content-preservation guard (mitigation for repainting)
- Telemetry: per-depth-bin **correction magnitude** `‖T(z)−z‖/‖z‖` — this is
  simultaneously the drift meter (it should GROW with depth pre-application)
  and the safety readout.
- Hard cap: `carn_tx_max_correction` (relative norm, default e.g. 0.35) — a
  correction exceeding it is **clamped and counted** (`carn_tx_clamped_total`,
  monotone). A corrector that wants to move a latent 50 % is repainting, not
  restoring.

### 2.7 Phase 0 — the SPECTRAL BASELINE (build FIRST, ~a day)
Non-learned version at the same hook: match each committed chunk's per-channel
latent **power-spectrum envelope** to the ride-seed reference
(AdaIN-in-frequency; magnitude only, phase untouched), `spectral_anchor_lambda`
gated, composing with the seam affine identically. Purpose: (a) it plumbs and
tests the exact hook CARN-TX will use, (b) it is the **baseline CARN-TX must
beat to justify its parameters**, and (c) it may claim a real slice of the fix
for free — the collapse is spectral, and the affine already took 80 % of the
walk. Same train/eval parity requirement.

## 3. Telemetry & standing rules (all mandatory, all learned the hard way)
- Every knob **resolved-value echoed** (derived, not requested), emitted at
  WARNING **+ stderr** (`_pix_emit_actionable` pattern — INFO is conditionally
  invisible; a proof-of-connection that is only conditionally connected proves
  nothing).
- Monotone counters, never per-step gauges: pairs trained (per depth bin),
  clamp events, apply events. **Omit-never-fake** for anything whose 0.0/1.0
  has meaning. Safety claims (e.g. "operator was actually applied this run")
  must be **fresh per emission**, never computed once and republished.
- **Fail-loud save/resume** for `carn_tx` + its optimizer (the silent re-init
  trap); resume with the gate on and a missing key RAISES.
- Loss-term **shares**, not just values: log the pointwise/style split of the
  gradient (the decorative-Sobolev lesson: a term below ~1 % of its intended
  share is reported *inert*, not present).
- Every guard asserting absence ships a **planted-violation companion proven
  to fire**, exercising product code; seam tests get the mutation control
  (physical copies, never symlinks; shadow the TEST file too so `_ROOT`
  resolves into the shadow; pre-import via a shadow conftest — full recipe in
  `docs/WP_PIXGAN.md` §30-31).

## 4. Evaluation — pre-registered, dual criteria (§9.5 discipline)
1. **Battery vs the B distribution** at fixed shallow depth (local correction):
   `hf_power` ↓ toward 0.86-0.89×A, `hv_anisotropy` → ~1.1,
   `angular_entropy` → ~0.98 — read JOINTLY, never anisotropy alone.
2. **Persistence at 10 s / 25 s / 60 s** (dynamical correction) — this is where
   an in-loop operator should *shine* relative to a per-frame critic; if the
   improvement does not persist, the restoring map is too weak or too shallow.
3. **Controllability unharmed**: the noop/held-still eval and a swap-judge
   spot-check — the operator must not damp legitimate motion (it edits
   texture-style, not dynamics; the correction-magnitude cap is the guard).
4. Multi-seed protocol for any close call (≥3 seeds; anisotropy/entropy are
   noise-dominated single-seed).
5. Arms: `carntx_only` (no pixel GAN) vs `noanchor` control, then optionally
   `carntx+pixgan` for the belt-and-braces composition.

## 5. Ownership, landing order, coordination
- NEW: `model/carn_tx.py`, `testing/test_carn_tx.py`, `docs/WP_CARNTX.md`.
- EDIT: `trainer/causal_action_forcing_train.py` (new carn_tx block — take the
  file via the grid INBOX lock queue; the SURROGATE agent and MAIN are active
  in it), `pipeline/action_forcing_training.py` (commit hook),
  `utils/eval_causal_AR.py` (eval hook), `configs/action_forcing_phase3_dmd.yaml`
  (carn_tx block appended at END).
- DO NOT TOUCH: `model/pixel_texture_disc.py` and all pix_* wiring (B1,
  frozen), the surrogate files, `model/forward_noiser.py` (clone, don't edit),
  `analysis/texture_stats.py` (eval instrument stays independent).
- Phase 0 (spectral baseline) lands first and smokes on a held holder; CARN-TX
  training lands second; application ramp last.
- Env: `/scratch/u6ex/as1748.u6ex/miniforge3/envs/arrwm/bin/python`;
  `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8`; one pytest
  process per suite; `inspect` source scans can flake right after an edit on
  Lustre (re-run before believing); never sbatch/scancel — holders via the
  grid.

## 6. Acceptance
- All existing suites stay green; new suite green; byte-identical with both
  gates off (RNG-fingerprint pattern).
- Phase-0 smoke: hook fires at commit in train AND eval, λ echo resolved, no
  step-time blowup.
- CARN-TX smoke: pairs flowing with the depth histogram showing the deep-bin
  share, correction magnitude growing with depth, style loss decreasing,
  clamp counter ≈ 0.
- First arm read per §4. The operator's parameters are justified only if it
  beats the Phase-0 spectral baseline on the battery at equal λ.
