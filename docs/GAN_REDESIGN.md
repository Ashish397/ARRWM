# GAN redesign — proposed fixes

> **2026-08-23 evening: SUPERSEDED IN DIRECTION by `docs/GAN_REDESIGN_TWO.md`**
> — researcher's diagnosis: the critic must be DECOUPLED from the DMD
> teacher's weights (14B prefix / stock 1.3B / pixel+surrogate). This file
> remains the ledger of everything built and learned; the new file is the
> plan.

Companion to `GAN_ARCHITECTURE_BRIEF.md` (which states the current design and
the measured failure). This document collects proposed changes point by point,
each with the argument, the concrete implementation, the risks, and the test
that would confirm or refute it.

Standing constraint: every change lands **flag-gated and default-off**, is
byte-identical when disabled, and goes through adversarial review before launch.

**Convention.** Under each point, *The claim* states the proposal as received.
Sections headed **[CC]** are my critique, verification against the
implementation, or extension — flagged so the two are never confused.

STANDING RULES THAT YOU NEVER BREAK:

1) NEVER USE THE AR HEAD FOR THE TEACHER OR THE FAKE SCORE
2) THE REAL AND THE FAKE HAVE TO BE ALIGNED - DO NOT MISMATCH THEM 
3) ONLY HAVE FOUR HOLDERS GOING AT ANY GIVEN TIME
4) STAT ANCHOR IS RETIRED — DEFAULT OFF (researcher 2026-08-23). Any arm
   that enables it must say why.

---

# EXECUTION PLAN (added 2026-08-23 — classification of everything below)

The point bodies below stay in their original numeric order because they
cross-reference each other by number; implementation works down THIS list.
Every item lands flag-gated default-off + adversarial review, per the standing
constraint.

## A. No-brainer wins and cheap fixes — LEDGER (compressed 2026-08-23; full histories in git)

**STILL OPEN (Opus code items, priority order — updated after the 14:26
holder validation run):**
1. **T3 — pixel-GAN trainer wiring** (WP-PIXGAN's last track; its WP-14B
   gate has landed): `gan_pixel_texture_enabled` gate, `pixel_texture_disc`
   attrs + `pix_optimizer`, D-update loop, G-term into `gen_gan_loss`,
   fail-loud save/resume, real/fake supply, §7 telemetry, config `pix_*`
   block. The pixel GAN CANNOT RUN until this lands.
2. ~~Fix the 6 test failures~~ **CLOSED 2026-08-23 ~18:00 (stale): the
   14:26 counts were against pre-fix files; WP-PIXGAN re-ran all three
   suites on a holder GPU — 91+20sub and 44 passed, all green. T3 merge
   precondition met.** Ops notes from that run, now standing: (i) suites on
   CPU need `OMP/MKL/OPENBLAS_NUM_THREADS=8` or torch thrashes 144 threads
   (30+ min "hang"); GPU: 5 s. (ii) spec-drift tripwires that grep module
   source for forbidden symbols must strip comments/docstrings first, with
   a planted-violation companion test.
3. **14B backbone: MEASURED OOM — the smoke dies at ~92 GB on all ranks**
   (60-step smoke, 2026-08-23 15:12, path-form flag). Two required fixes
   before re-smoke: (i) run the prefix backbone at the ACTUAL disc-chunk
   token count, not the wrapper's 18721-token padded seq_len (the known
   trap — at 5120-dim × 9 graph-on blocks the padding dominates);
   (ii) checkpoint/micro-batch the projector's 14B forward like the LADD
   disc path does. Also: `load_wan14b_prefix` should resolve model NAMES
   via `_default_wan_model_path` (name form FileNotFoundErrors).
4. **STAT ANCHOR DEFAULT → OFF (researcher decision 2026-08-23):** we no
   longer use the stat anchor. Flip `stat_anchor_loss_weight` to 0.0 in
   `configs/action_forcing_phase3_dmd.yaml`, make `STAT_ANCHOR` default 0.0
   in `sbatch/_fgan_holder.sh` / `_fgan_holder1n.sh` (currently required
   with the comment recommending 1.0), and sweep launch scripts. Enabling
   it is now the deliberate exception, not disabling it.
5. **A21 residual** — measure the loader's DISTINCT real-source-frame count
   (`pix_real_support_frames`); must clear the 4,096 floor before B1 runs.
6. **A1 residual** — make the missing-`rank0_ride.json` fallback loud.


- [x] **A1. Instruments fixed + validated.** `analysis/texture_stats.py`
  (directional spectrum, angular entropy, Haar LH/HL/HH, HF kurtosis) wired
  into `rollout_quality.py`; review found + fixed 5 math bugs (Hermitian
  fold, fx=0 drop, no window, zero-grad ratio, double /255) and 2 wiring
  bugs; seed length now in FRAMES from `rank0_ride.json` (evals are 20 fps,
  not 16 — the old 2.25 s window counted 9 generated frames as real).
- [x] **A4. `gan_grad_target_norm` cap DELETED** (botched port: NaN
  run-killer, wrong tau scale, per-chunk teacher backward). A7 telemetry
  kept; `train/gan_grad_norm` now reports the UNCAPPED gradient.
  `dmd_grad_target_norm` untouched.
- [x] **A5. No-grad decoder tripwire** — implemented, default-off, verified
  no-grad; live in the ganv2f wave (`texture_tripwire_every=25`). 2-frame
  isolated decode → trend-only, not on the A/B/C scale.
- [x] **A6. Real-diversity telemetry** — both paths covered, per-step reset
  fixed, positional path tagged `gan_real_div_positional=1.0`; never
  silently absent.
- [x] **A8. Parked-path config corrections recorded**
  (`docs/PARKED_LADD_CONFIG_FIXES.md`); apply when the LADD path unparks.
  The γ=1e6 R1-scaling explanation is falsified; the flags stand.
- [x] **A11. Battery VALIDATED against the known verdict.** All 10 arms
  scored (`sbatch/run_texture_score.sh`; seed_end=36 everywhere). Verdict
  rule in code = `seed_log_dist` voting on hv_anisotropy + haar_HL_LH_ratio
  ONLY — **hf_power anti-correlates with the eye late (0/29) and never
  votes**. Ranking (lower = closer to seed reality): marginal 0.149 <
  strict03 0.203 < poolrich 0.217 < horizon_nogan90 0.287 < nogan200 0.360 <
  wave01 0.590 < horizon_wave90 1.004 < wave_ts 1.015 < raw_t0 1.445 <
  strict_rerun 2.307. Reproduces nogan90 > wave90 (3.5×); failure is
  two-sided (collapse below seed vs blow-up above). Do not cite the old
  0.62 figure (correct value 1.56).
- [x] **A12/A13/A15 — superseded/cancelled** (A19 answers R1-on-MEAN; A4's
  deletion moots A15).
- [x] **A16. Positional R1 fixed** — both positional estimators honour
  `ladd_r1_normalize_tokens`; cadence unified behind
  `ladd_r1_unified_cadence` (default off, byte-identical when off).
  **Patch-logit arms MUST set `ladd_r1_normalize_tokens=true` AND
  `ladd_r1_unified_cadence=true`.**

  **A16-D1 — the unified cadence STARVED R1; fixed 2026-08-23.** The first
  implementation put the decision *inside* the `for _ in range(n_disc_updates)`
  loop behind a single GLOBAL `self._ladd_last_r1_step` latch. Three failures,
  all on the exact path A16 steers patch-logit arms onto:
  (a) the first iteration wrote the latch and iterations 2..S saw
  `delta == 0 < N`, so with `gan_updates_per_step=5` (every live GAN script)
  R1 dropped from **5 applications per due step to 1**;
  (b) `_compute_ladd_losses` calls `_ladd_run_pair_mode` once per enabled
  mode, so the first mode claimed the shared latch and every later mode's R1
  was permanently **zero** — with `gt_vs_fake` + `gt_transition` the
  `gt_transition` head trained with NO gradient penalty while
  `r3gan_r1_fired_gtxn` read 0 and the OR'd top-level gauge read 1;
  (c) with `ladd_defer_disc_update=true` the deferred MATCHED `_run_disc_updates`
  runs after the gen backward, so a positional loop claimed the latch first and
  the matched D lost its R1 — the inverse of the intended priority.

  **DECIDED SEMANTICS (now stated in code above `_ladd_r1_block_due`):**
  * `ladd_r1_every_n_steps` (N) is a cadence over **training steps**, not over
    D-update iterations and not over pair modes. Lazy-R1 amortises COST across
    steps (StyleGAN2 lazy regularisation); it was never meant to thin the
    penalty *within* a step. **On a due step R1 fires on EVERY one of the
    mode's S = `gan_updates_per_step` D-updates.** N=1 therefore means
    "R1 on every D-update", which is what the pre-unification positional
    modulo already did.
  * The debt latch is keyed by **`pair_mode`**. Each enabled mode trains the
    shared disc against its own real distribution, so each needs its own
    penalty; one mode firing must never spend another mode's debt.
  * The latch is **not** keyed by path (matched vs positional). Within a step a
    mode runs either the matched branch or the positional branch, never both
    (the matched branch returns first); across steps a `[LADD-MATCH-FALLBACK]`
    must not reset the cadence — it is the same D head either way. This is
    also what keeps the decision **rank-invariant** under DDP when one rank
    falls back to positional and another matches.
  * The latch is written **once per D-update block** — once per
    `(step, pair_mode)`, hoisted *above* the `range(n_disc_updates)` loop.
  * Debt (`step - last_fire >= N`) not `step % N`: the GAN only runs on
    generator iters, so an exact modulo can be missed forever when N and
    `dfake_gen_update_ratio` are not commensurate.

  **FIRE-RATE TABLE — R1 applications per DUE step.** S = `gan_updates_per_step`,
  M = number of enabled pair modes, N = `ladd_r1_every_n_steps`.

  | path | `unified=false` (legacy) | `unified=true` **before** D1 | `unified=true` **after** D1 |
  |---|---|---|---|
  | positional | `M*S` (`step % N`, re-evaluated per iter) | **1** (global latch, 1st iter of 1st mode) | `M*S` |
  | matched | **1** (global latch, 1st iter of 1st mode) | **1** | `M*S` |
  | mixed (fallback / defer) | `M*S + 1`, can double-fire | **1** | `M*S` |

  Worked example — the FT_v3 shape (`gan_updates_per_step=5`,
  `gt_vs_fake` + `gt_transition`, N=1): intended 10 R1 applications/step;
  legacy positional gave 10 but could double-fire against a matched mode;
  A16-as-shipped gave **1**; post-D1 gives **10**. With N=4 the same run gives
  10 on steps 0,4,8,… and 0 in between (per mode, independently).

  **TELEMETRY (new).** `train/r3gan_r1_block_fires{,_gt,_adj,_gtxn}` = R1
  applications made by that `(step, pair_mode)` block (NaN when the unified
  cadence is off, so the trace gaps instead of lying);
  `train/r3gan_r1_mode_updates_total` / `_mode_penalties_total` /
  `_mode_fire_rate` = **per-mode** monotone counters + lifetime rate;
  `train/r3gan_r1_fire_rate_min_mode` and `train/r3gan_r1_modes_with_penalty`
  aggregate them so failure (b) is visible at the top level — the OR'd
  `r3gan_r1_fired` structurally cannot show it. Also fixed: the top-level
  `train/r3gan_r1_fired` gauge was `max()`'d over a **substring** match that
  also caught `r3gan_r1_fired_total_<mode>`, a monotone counter, so the 0/1
  gauge reported hundreds. Tests: `testing/test_r1_cadence_and_override_guard.py`.
- [x] **A6-D2. Real-diversity telemetry emitted ZERO keys in two-phase runs**
  — fixed 2026-08-23. `_match_select` (the only writer of
  `_ladd_real_div_telemetry`) runs in the `d_only` call, and the two-phase
  overlay whitelist in `_compute_ladd_losses` enumerated only `r3gan_*` /
  `ladd_*_n_real`, so every `gan_real_*` key was dropped; the `g_only` call
  then reset the attribute and, during critic warmup or with zero gen GAN
  weight, drew nothing of its own. With `gan_real_diversity_log=true` and two
  modes the result was **no A6 keys at all** — indistinguishable from the flag
  being off. Two halves: the overlay is now **prefix-based**
  (`train/gan_real_*`, so future A6 keys are covered automatically, and the
  d_only D-update draw wins because that is what D actually trained on), and
  the draw is stashed **per step per pair_mode** (`_ladd_stash_real_div` /
  `_ladd_real_div_logs`) so a call that does not itself draw still reports this
  step's D-update draw. Cross-step staleness — the bug the original A6 reset
  removed — remains impossible: the store is discarded when `current_step`
  changes. Gated as before by `gan_real_diversity_log` (default off).
- [x] **A17. `ladd_r1_num_samples` honoured on the inline path** (same
  seeding as micro-batched). Changes raw_t0-style recipes — intended.
- [x] **A18. Hygiene** — dead `ladd_r1_once_per_step` stripped from scripts.
  Standing rules: unknown dotlist keys merge SILENTLY (grep every launch
  script on any flag rename); never edit a script a live run is executing.

  **A18-D3 — the guard would have FALSE-POSITIVED on `pix_*`; fixed
  2026-08-23.** The scan was purely textual: `re_getattr` excludes parentheses
  from the receiver and `_is_config_receiver` only accepts a dotted tail in
  `{args, config, cfg, conf, opts, hparams}`. `model/disc_holdout_probe.py`
  reads knobs as `getattr(getattr(trainer, "config", None), key, None)` with
  `key` a **variable** and the literals in a `DEFAULTS` dict — both idioms are
  structurally invisible to that scan. `pix_` is a guarded prefix reserved for
  B2's `model/pixel_texture_disc.py`, which `TEXTURE_GAN_DESIGN.md` specs with
  ~15 `pix_*` knobs; had B2 copied the idiom, **every** `pix_*` knob would have
  been reported unread and, under `strict_override_keys=true`, would have
  killed the job before the trainer was constructed.

  Fix: an **AST pass** (`_ast_config_sourced_keys`) unioned into the textual
  scan — strictly additive, so it can only widen the allowlist and any parse
  failure degrades to "found nothing". It recognises three things the textual
  scan cannot:
  1. **nested config receivers** — `getattr(getattr(t, "config", None), "k", d)`;
  2. **`DEFAULTS`-style tables** — every module-level all-string-keyed dict
     literal, but **only** in a module that demonstrably reads config with a
     non-literal key, so an unrelated lookup table is never mistaken for a
     consumer;
  3. **an explicit registration hook** — a module-level
     `CONFIG_KEYS` / `CONFIG_OVERRIDE_KEYS` / `OVERRIDE_GUARD_KEYS` /
     `_OVERRIDE_GUARD_KEYS` list/tuple/set/dict of string literals.

  Verified additive: over the whole repo the sourced set is unchanged
  (117 keys before and after, nothing lost). A read off some *other* object
  (`getattr(self.model, "k", ...)`) still does **not** count — the
  `ladd_gt_transition_action_blind` failure mode stays caught.

  **Cost:** the walk goes from ~50 s to ~64 s, once, at startup on this
  filesystem. A cheap substring pre-filter was written and then **deliberately
  removed**: a gate that misses a variant spelling (nested `getattr` split
  across lines, a differently-named table) silently restores the very false
  positive this fix exists to remove, and under `strict_override_keys` that is
  a dead job. 14 s is not worth that trade.

  **ACTION FOR B2:** `model/pixel_texture_disc.py` today reads no config at all
  ("Every `pix_*` value is a constructor argument"), so `pix_*` sources to the
  empty set and any `pix_*` override is *correctly* reported as having no
  effect. When B2 wires the knobs, either read them literally off the config in
  the trainer, or declare `CONFIG_KEYS = (...)` in the module. Do not rely on
  the `DEFAULTS` harvest alone unless the module also reads config by variable
  key.
- [x] **A18-D8. The guard could kill a run despite promising it cannot** —
  fixed 2026-08-23. `_warn_ignored_override_keys` ended with
  `except RuntimeError: raise`, which re-raises **any** RuntimeError from the
  scan itself — `RecursionError` is a RuntimeError subclass — contradicting the
  guard's own docstring ("every failure mode of the scan itself degrades to a
  silent no-op"). The strict verdict is now recorded in a local and raised
  **after** the `try` block as a dedicated `OverrideGuardError(RuntimeError)`,
  so nothing raised from inside the try can escape.

### A19–A24 — advisor directives — ALL LANDED 2026-08-23

- [x] **A19. B2 critic is R1-ONLY, R1 on the MEAN patch score, γ≈1.0** —
  specced in `TEXTURE_GAN_DESIGN.md` (R2 gone; scope freeze: no
  ADA / timestep-conditioning / DINO / multi-scale-temporal / wavelet /
  nearest retrieval / action conditioning / transition pairing until their
  recorded triggers).
- [x] **A20. Real-batch invariant** — one INDEPENDENT reconstructed-real
  per fake per D update (12-and-12); frame-within-crop expansion is a
  fake-side device only; §7 telemetry mandatory, unconditional.
- [x] **A21. Total real support pinned** — fresh GT preferred; cache floor
  ≥4,096 distinct source frames (8–16k preferred, continuously refreshed,
  cross-ride, uniform within nuisance band, never similarity-matched).
  RESIDUAL: open item 2 above.
- [ ] **A22. REOPENED — Held-out D generalisation probe built**
  (`model/disc_holdout_probe.py`, 36 tests, `disc_holdout_probe_every`,
  default off). The historical leak REPRODUCED with a null holdout list
  (n_leak=20) and 0 with it set; **BOTH `weu` and `weunz` contain the
  reserved rides** — either preset leaks with a null list. Read
  `dhp_leak_rides` / `dhp_leak_seen` / `dhp_ring_delta` before trusting any
  margin. Residual: name-disjoint ≠ content-disjoint.

  **REOPENED 2026-08-23 after adversarial review — verdict: "I would not trust
  this probe to gate a real experiment." The two halves fail differently.**
  **LEAK VERDICT — conditionally sound, but FAILS OPEN two ways.** Where the
  reserved root is configured and `dataset._rides` is intact the static check is
  correct, rank-symmetric, and does catch the historical weu/weunz superset case.
  BUT: (D2) `holdout_eval_root` **appears in no yaml** — it exists only on
  `--override` lines — so an unset or misspelled root yields `_holdout_paths=[]`,
  an empty reserved set, a permanently no-op runtime tripwire, and
  `dhp_leak_rides=0 / dhp_leak_seen=0 / dhp_ring_delta=0`: **the exact green
  signature, while checking nothing.** (D3) an unreachable `dataset._rides` gives
  `n_train=0, n_leak=0` the same way. A gate must fail CLOSED.
  **MEMORISATION VERDICT — not usable at all.** (D1) `train_from_observed=True`
  (the default) builds the train pool from rides observed at the FIRST fire —
  which at step 0 is normally ONE ride — and caches it for the whole run: 4
  windows from 1 ride against 32 windows from 8. Monte-Carlo with a critic that
  memorises NOTHING returns MEMORISING 8.5%, INVERTED 15%, and a stable spurious
  `real_auc > 0.7` in **30%** of runs — `real_auc` being the number the module
  calls its sharpest signal. (D4) the null SE is computed at frame-level n=24 but
  the 24 scores are 8 crops x 3 frames from the SAME latent crop: true SE 0.142
  vs advertised 0.084, so ~20% of fires on pure noise return a non-chance verdict
  and 8% return MEMORISATION. (D5) no orientation guard — a sign-flipped critic
  that memorises perfectly reports BOTH_CHANCE, or with a control, the confident
  wrong verdict D_UNDERPOWERED.
  MINIMUM FIXES: gate pool construction on `len(observed) >= train_rides` (or
  default `train_from_observed=False`); SE at CROP-level n; raise/flag on
  `n_holdout==0` or `n_train==0` instead of returning quietly; two-sided
  orientation check; strict mode must re-raise from cache (D6, currently one-shot);
  add `disc_holdout_probe_` to the A18 override-guard prefixes (D14 — a typo'd
  `_evrey` is silently accepted today, the exact class A18 exists for).
  VERIFIED CORRECT meanwhile: `roc_auc` exact to 1.1e-16 vs sklearn; the
  one-shared-fake identity `gen_gap == real_gap` holds algebraically; no dtype or
  normalisation asymmetry across the three sides; the single-funnel claim survived
  every attack; no collectives/grad/global-RNG; default-off is clean; 36+16 tests
  pass.
- [x] **A23. Commit-tensor equivalence gate RUN — FAILED late** (d 0.109 =
  6.6× the noise floor, monotone by decile, 34/34 late chunks one
  direction): the t=60 flash fake is systematically CLEANER than the
  rendered video exactly where the critic must deliver, and training's
  rollout depth sits entirely inside the equivalent region, so the mismatch
  is invisible to training telemetry. ~~**B1 must take path (a)**~~ **OVERTURNED 2026-08-23 — see the A23 GATE RUN entry above. NOT ESTABLISHED: 1-of-7 cells, seed outlier. Two independent sessions confirm.** WP-PIXGAN adds a sharper argument than the one I made: `compare_commit_tensors.py:118` sets `VOTERS = (hv_anisotropy, haar_HL_LH_ratio)`, and **`hv_anisotropy` is a statistic this campaign's own noise-floor protocol already flags as noise-dominated** (sd up to +-0.34; never rank close arms single-seed). So the original verdict was a single-seed read on a voter we had already documented as unsafe to read single-seed — the .1093/.0113/.0128 spread is exactly what that warning predicts. **Also: the recorded acceptance criterion for path (a) is a TAUTOLOGY** — under (a) the fake IS the ladder endpoint, so both arms are the same tensor and every distance collapses to 0.0 by construction. A23 never justified a launch block. Path (a) is now an OPTION, not a gate; WP-PIXGAN has it built, tested and flag-gated default-off (`pix_finish_grad_enabled`), so the fake source is an experimental variable.
  1 above.
- [x] **A24. Band matching pinned COARSE** — top/middle/bottom thirds;
  exact-y matching and every approach toward it forbidden.

## A-DONE — earlier completions (compressed; full arguments in git history)

- [x] **A2.** Training commits the t=60 flash tensor; inference commits the
  ladder endpoint (σ 0.062 vs 0.208). Training-time sample videos also
  render the t=60 tensor. Hardened into the A23 launch gate.
- [x] **A3.** Phase-1 critic parameters pinned in `TEXTURE_GAN_DESIGN.md`
  §5.1–5.5 (since narrowed by A19 to R1-only).
- [x] **A7.** `cos(g_GAN,g_DMD)` + norm-ratio telemetry live
  (`gan_grad_telemetry_every=25`). First readings: ratio 0.0005–0.012, cos
  −0.02..−0.10 — the current critics are small and near-irrelevant.
- [x] **A9.** CARN_V2 candidate (b) corrected to anisotropic (radial
  averaging cannot see the banding mode).
- [x] **A10.** R2 had been firing all along (48.6%); the 0.20/0.49 R1-rate
  imbalance traced to the two separately-written R1 estimators; R2 since
  deleted outright.
- [x] **A13/A14.** R1 fire-rate logging + the positive control specced into
  `TEXTURE_GAN_DESIGN.md` (§5.4, §8.1).

## B. Remaining builds — THREE PARALLEL WORK PACKAGES (plan 2026-08-23)

Three Opus agents run simultaneously. **File ownership is exclusive** — an
agent MUST NOT edit a file another package owns. Shared-file edits follow the
LANDING ORDER below. Each agent reports into its OWN doc
(`docs/WP_PIXGAN.md` / `docs/WP_14B.md` / `docs/WP_SURROGATE.md`); only the
main agent edits THIS file.

**Frozen interface contracts (all three code to these; changing one requires
main-agent sign-off):**
1. Pixel critic: `model/pixel_texture_disc.py::PixelTextureDisc`,
   `forward(px [N,3,H,W] in [-1,1]) -> [N,1,h,w]` patch-logit map; trainer
   attribute name `pixel_texture_disc` (the holdout probe then needs zero
   edits — `model/disc_holdout_probe.py:78-100,207`).
2. Inference-parity fake: `info["finish_denoised_chunk_grad"]` (graph-on,
   flash-off path); the existing detached `finish_denoised_chunk` key is
   unchanged.
3. Surrogate critic: `model/latent_texture_critic.py::LatentTextureCritic`,
   `forward(z [B,F,16,60,104]) -> dense per-token value map`; generator
   consumes `-critic(z).mean()` under the frozen-critic idiom; teacher
   refresh knob `pix_teacher_refresh_every` (teacher = pixel disc ∘ decode).
4. 14B projector: `build_ladd_disc(..., backbone=<raw WanModel>,
   dim_teacher=5120, a_per_f=0)`; knobs `ladd_disc_backbone_model_name`,
   explicit `ladd_feature_blocks=[0,2,4,8]`, hard assert
   `max(taps) < num_layers_loaded`.

**LANDING ORDER for the only shared file
(`trainer/causal_action_forcing_train.py`):** WP-14B first (small build
branch at :694-937 only) → WP-PIXGAN (everything else) → WP-SURROGATE
(trainer wiring last; until then it builds module + standalone tests only).
`configs/action_forcing_phase3_dmd.yaml`: each package appends its OWN new
block at the END of the file — no edits to existing keys.

---

1. - [ ] **B1 / WP-PIXGAN — pixel-space texture PatchGAN + the A23 grad
   path.** Spec = `docs/TEXTURE_GAN_DESIGN.md` (complete; §4 arch, §5
   R1-only on the MEAN patch score, pix_* names are mandatory) + A19–A24.
   OWNS: new `model/pixel_texture_disc.py`;
   `pipeline/action_forcing_training.py`; `model/dmd_action_forcing.py`
   (only :9513-9516); `trainer/causal_rolling_staircase_train.py`
   (save block :1623-1637); trainer (after WP-14B lands); config pix_* block.
   Build steps, with verified anchors:
   (a) `PixelTextureDisc` per §4 (3→64→128→256→1, spectral-norm,
       GroupNorm(8), LReLU 0.2) + patchwise NS-logistic/hinge + the single
       R1 FD estimator (mean-reduced, γ=1.0, `pix_r1_sigma=0.01`) — in the
       new file, NOT reusing LADD penalty code (§5.3).
   (b) **A23 grad path (launch gate):** grad-on FINAL finish rung in
       `pipeline/action_forcing_training.py:2137-2165` (`_ckpt`-wrapped,
       earlier rungs stay no_grad); un-detached second buffer at
       :2289-2294 (the detached `clean_chunk`→KV commit at :2302 must stay
       detached); publish `finish_denoised_chunk_grad` at
       `model/dmd_action_forcing.py:9513-9516`. Acceptance: re-run
       `analysis/compare_commit_tensors.py`; d_vote ≤ d_null every stratum.
   (c) Crops/reals: import `band_plan`/`take_crops`/`_decode_crops` from
       `model/disc_holdout_probe.py:595-704` (reals, no_grad); write the
       grad twin for the fake using `_vae_decode_grad`
       (trainer:3763-3811, `empty_cache()` first — the step-2 allocator
       failure is real, trainer:10833-10839); every random draw
       DDP-broadcast via `_sample_critic_grad_frame_indices`
       (trainer:3707-3731).
   (d) Trainer wiring: parallel attrs (`pixel_texture_disc`, `_ddp`,
       `pix_optimizer` Adam(0.0,0.9) lr=`pix_gan_lr`), gate
       `gan_pixel_texture_enabled` (NOT a new `gan_backbone` value); G-term
       added into `gen_gan_loss` BEFORE the telemetry at trainer:10792;
       `register_fake_source(self, fake_latents)` before :10877; save +
       resume keys FAIL-LOUD (the parent save block is gated on
       `gan_enabled` — silent critic re-init on resume is the known trap).
   (e) Telemetry per §7 (all `pix_*` log keys, monotone counters, A20/A21
       real-diversity + support keys incl. `pix_real_support_frames`).
   Traps: P=660 after the 8-px border trim, not the doc's 768 (mean
   reduction makes it harmless — log it); DDP
   `find_unused_parameters=False` vs the R1-subsample branch; 60 no_grad
   decodes/step at updates=5 puts the loader on the critical path.

2. - [ ] **B2 / WP-14B — Wan 2.1-T2V-14B as the disc projection backbone.**
   OWNS: `model/ladd_disc.py`; new `model/wan14b_prefix.py`;
   `wan/modules/model.py` (optional `max_block` early-exit); trainer
   :694-937 build branch (lands FIRST); config block.
   Build steps, with verified anchors:
   (a) Prefix loader: `WanModel(num_layers=max_tap+1, dim=5120,
       ffn_dim=13824, num_heads=40, in_dim=16)` + manual safetensors
       prefix load (shards 1-2 of 6 cover blocks 0-8 + embeddings; keys
       `blocks.{i}.*` + 4 embedding prefixes; fp32 on disk → cast bf16
       ≈6.8 GB), `requires_grad_(False).eval()`. Head either loaded from
       shard 6 or random — the projector discards the return value
       (`model/ladd_disc.py:243-246`).
   (b) Projector accepts a raw `WanModel` (bypass `WanDiffusionWrapper` —
       its :749-753 consumes the output): call
       `model(x.permute(...), t, context, seq_len=ACTUAL disc-chunk token
       count)` — do NOT inherit the wrapper's 18721-token padding
       (`wan/modules/model.py:735-741`), it dominates cost at 5120-dim.
   (c) Trainer branch on `ladd_disc_backbone_model_name`: pass the 14B's
       own `dim_teacher=5120`, `patch_size=(1,2,2)`, **`a_per_f=0`** (do
       not harvest `action_tokens_per_frame=1` from real_score — silent
       reshape mis-slice), device/dtype placement (trainer:213-215).
   (d) Config: `ladd_feature_blocks: [0,2,4,8]` EXPLICIT (the auto-default
       for 40 blocks is [8,16,24,32,39] → full 28 GB load) + the assert.
   Traps: do NOT touch `real_name`/`model/base.py:183-188` (DMD breaks);
   resume drops mismatched CCM shapes (trainer:2845-2860, expected);
   distribution shift — stock T2V 14B never saw driving data or action
   tokens; the A7 cos telemetry is the health readout (prediction:
   |cos(g_GAN, g_DMD)| falls vs the v14e-backbone critic).

3. - [ ] **B3 / WP-SURROGATE — latent critic serving pixel-disc gradients
   (the affordability mechanism).** RESTORE, do not reinvent: full-fidelity
   version at commit `835b1df` — `model/latent_sam2_critic.py` (hand-rolled
   attention IS load-bearing: double-backward has no flash kernel),
   teacher-side `model/r3gan_sam2.py`, trainer blocks :2783-2965 (both
   distillation targets: value MSE + Sobolev ∇_z MSE), gen consumption +
   two-stage warmup at `01ea13d^`:5109-5155. Deleted at `01ea13d`; the
   action-critic z-guidance (live, trainer:3038-3211) is the design
   template for the frozen-critic idiom and stays untouched.
   OWNS: new `model/latent_texture_critic.py` (the 835b1df critic, renamed,
   SAM2 references stripped); new `testing/test_latent_texture_critic.py`;
   `docs/WP_SURROGATE.md`. Trainer wiring ONLY after WP-PIXGAN lands.
   Deltas vs the ancestor:
   (a) Teacher = `pixel_texture_disc ∘ decode(latent crop)` (contract 1),
       not SAM2: teacher value = mean patch logit; teacher grad =
       `autograd.grad(disc(decode_grad(z_crop)).mean(), z_crop)`.
   (b) NEW cadence knob `pix_teacher_refresh_every=N`: the historical code
       ran the teacher EVERY step (gan_critic_grad_full_every only widened
       frame coverage); here the teacher fwd+grad runs every N steps with
       cached targets between refreshes; critic serves the generator every
       step. Reuse the `% N` gate shape from 835b1df:2808.
   (c) Keep the health diagnostics verbatim (`critic_grad_cos_sim`,
       `critic_disc_corr`) — the only honest readout for a Sobolev
       surrogate — plus a periodic direct-vs-surrogate gradient check.
   Order of operations per step (from the ancestor): D-update → critic
   distillation → generator consumes the just-updated critic.

**Sequencing note:** B1 runs standalone first (direct pixel gradients on
sampled crops — affordable at crop scale, per §6); B3 removes the per-step
decode cost when scaling coverage toward R10's ≥60%; B2 is independent of
both (latent-side backbone). B1 without B3 is a valid first arm; B3 without
B1's disc has no teacher — hence the landing order.

## R. RESEARCHER DIRECTIVES (2026-08-23, verbatim intent — supersede anything below where in conflict)

The experiments are BASED on these. Config-only items go into the validation
arms immediately; code items queue behind the in-flight A-implementation.

- [ ] **R1. Tap correctly** — early taps (`ladd_feature_blocks=[0,2,4,8,29]`
  now; patch-embedding tap when code allows). *Config-only now.*
  **IN TEST 2026-08-23: early taps live in ganv2f_all/nopatch/noaug (notaps
  is the leave-one-out control); verdict pending. Patch-embed tap = code,
  queued.**
- [ ] **R2. Spatial/token-level gradients, never frame/chunk-level** — patch
  logits end to end: `ladd_scalar_output=false`, per-token loss, positional
  gradient. *Config-only.*
  **IN TEST 2026-08-23: live in ganv2f_all/noaug/notaps (nopatch is the
  leave-one-out control), safe post-A16; verdict pending.**
- [ ] **R3. Nearest-L1 real matching DISABLED for now** —
  `ladd_gt_transition_match=false`; reals are position-matched GT / wide
  draws, never similarity-selected. *Config-only.*
  **STATUS 2026-08-23: the INTENT (no similarity selection) is live in all
  ganv2f arms via `match_pool=100000` = uniform-random over the whole ride;
  literal `match=false` is now safe post-A16 and can go in the next wave.**
- [ ] **R4. The disc must see ENOUGH real latents** — wide/uniform real
  draws + cross-ride ring stays on + the A6 diversity telemetry (unique
  rides / windows / repeat rate) proves it, every run. *Config + telemetry.*
  **STATUS 2026-08-23: config half live in ganv2f (uniform pool, k=8,
  max_real=16, ring 8192/32); A6 telemetry now covers BOTH paths and is
  enabled (`gan_real_diversity_log=true`). Closes when the first live
  readout proves ring population + diversity.**
- [ ] **R5. Prepare Wan 2.1-T2V-14B as the discriminator projection base** —
  weights verified on disk (`/scratch/u6ex/as1748.u6ex/frodobots/Wan2.1-T2V-14B`,
  dim 5120, 40 layers, in_dim 16 = same latent space; VAE symlinked to the
  1.3B's). Build = truncated-prefix load + early-exit forward in
  `WanFeatureProjector` (Point 6): tapping `[0,2,4,8]` loads 9/40 blocks
  ≈ 6 GB, cheaper than today's full-depth 1.3B forward. *Code — queued
  behind the A-implementation; the [UNVERIFIED] truncation-cleanliness check
  is its first step.*
- [x] **R6. DONE 2026-08-23 (by deletion + repair, verified): R2 deleted
  outright instead of repaired; R1 token-normalisation landed in ALL FOUR
  estimator branches (A16); fire-rate monotone counters live. Fix R1/R2** —
  token-normalised penalty
  (`ladd_r1_normalize_tokens=true`) + the cadence repair (r2-never-fires /
  fire-rate imbalance); intended fire rates stated and logged from day one.
  *Config + the A-implementation's cadence fix.*
- [x] **R7. DONE 2026-08-23 — shift 5.0 eliminated from every active
  recipe; the fixed-low-t option is taken (`ladd_disc_sample_t=false`, t0)
  in the v2 base and all four ganv2f arms. Disc timestep shift 5.0 is
  measured-bad → GONE.** Low-t mass
  instead: `ladd_disc_timestep_shift=0.35` (or fixed low t). *Config-only.*
- [x] **R8. DONE 2026-08-23 — mean/cross equalisation OFF in the v2 base
  and all four ganv2f arms; the DC-gradient-kill remains a queued escalation
  ONLY if a brightness shortcut reappears. Mean/cross equalisation is
  measured-bad → GONE.** If a
  brightness shortcut reappears, do NOT re-equalise: **kill the gradient
  there instead** (project the DC/per-channel-mean direction out of the
  GAN's generator gradient), or take a different frame. *Eq removal
  config-only; DC-gradient-kill is a small code item, queued.*
- [ ] **R9. Multi-timestep fakes + timestep-informed disc + matched GT
  noising — authorised.** If the fake is taken at multiple/varying
  timesteps, the disc must be TOLD the timestep, and the GT real must be
  noised to the SAME timestep so real and fake stay aligned (standing rule
  2). **Scoped exception to standing rule 3, granted 2026-08-23: noising GT
  is permitted ONLY as disc-input alignment for a timestep-informed
  discriminator — never for context, commit, eval, or DMD paths.**
  Distribution-matched conditioning per Point 5 (the giveaway trap). *Code —
  queued (this is B4 pulled forward).*
- [ ] **R10. Coverage: the GAN looks at the WHOLE rollout** — whichever part
  DMD supervision acts on, the GAN must act on too, improving texture at ALL
  patches. If memory forces sampling, sample — but **never below 60% of the
  DMD-supervised portion seen by the GAN per step**. First step is a VERIFY
  (A2-class): measure what fraction of the supervised band the GAN currently
  sees (flash chunk vs all rolled pairs vs `ladd_pairs_per_step` caps), then
  enforce ≥60% and LOG the coverage fraction every step. *Verify now, then
  config/code.*
- **R11. Core principles stand** — TF head only, never AR (rule 1); real and
  fake aligned in noise level / timestep / conditioning (rule 2); rule 3 as
  scoped by R9; flag-gated default-off + adversarial review before launch.

## V. 10-HOUR VALIDATION PLAN — can a latent GAN improve texture at all? (added 2026-08-23, RUNNING)

**Question under test.** With the A-fixes applied, does the latent transition
GAN improve texture over no-GAN — or is latent adversarial texture
supervision a dead end, sending us all-in on the pixel critic (B2)?
Pre-registered here BEFORE the fixed arms run.

**The instrument.** The A/B/C texture diagnostic
(`analysis/texture_abc/run_abc.sh`): 40-chunk rollout on ride 20240115085313
(raw footage survives), joint battery at C_early (~10 s) and C_late (~25 s)
— HF power fraction, fy/fx anisotropy, angular entropy, Laplacian
variance/kurtosis, luma kurtosis, Haar LH/HL/HH — read against the same A
(raw RGB) and B (decode-roundtrip) references every time. Plus the 60 s
Madrid eval (researcher's eye) and the live `cos(g_GAN, g_DMD)` /
norm-ratio telemetry (default-on, every 25 steps).

**Reference scale — RUNNING NOW on holders 6106319/6106320 (~8 min/ckpt).**
Diagnostics on the six existing eval checkpoints: `nogan200`, `raw_t0`
(current best), `wave01`, `wave_ts`, `horizon_nogan90`, `horizon_wave90`.
`strict03` already measured: C_late anisotropy **0.36**, HF **2.21×** — the
baseline pathology. This gives the quantitative scale every validation arm is
judged on — and `nogan200` vs `raw_t0` already answers whether the UN-fixed
latent GAN beats no-GAN on texture numbers.

**Validation arms — launch when the A-fixes land AND pass adversarial review
(standing rule). Fresh holders 6106989/90/91 reserved.** All on the `raw_t0`
recipe — the measured-best base (TF head, old optimizer set) — transition
critic ON: it is the only latent GAN that exists until the B-builds land, so
this is explicitly a test of the latent *pathway*, not a revocation of the
parking decision.

| arm | delta vs raw_t0 | isolates |
|---|---|---|
| V1 | `ladd_scalar_output=false` + `ladd_r1_normalize_tokens=true` + `ladd_diff_aug_policy=flip,translation` + all mean/cross equalisation OFF | the reduction / R1-scaling / information fixes (A8 core) |
| V1b | V1 at `GANW=0.03` | weight re-bracket under patch logits (0.01/0.03 bracket was measured under scalar output and does not transfer) |
| V2 | V1 + `ladd_feature_blocks=[0,2,4,8,29]` + `ladd_disc_sample_t=true`, `ladd_disc_timestep_shift=0.35` | feature location: early taps + low-t mass (Point 2) |
| (V2b) | V2 at `GANW=0.03`, only if holders allow | — |

200 steps each (~80 min incl. chained 60 s eval), then the A/B/C diagnostic
on each eval checkpoint (~8 min). Caveat, pre-registered: 200 steps
under-trains a re-scaled critic (A3 wants 400–600), so "no improvement AND
d_loss still falling at step 200" reads **inconclusive-extend**, not failure.

**Decision rule RESCINDED by the researcher 2026-08-23** — the two references
split the battery, so "beat both jointly" was ill-defined; the battery stays
as instrument, the researcher judges numbers + videos. Diagnostics are
bit-deterministic: vary `ODE_FLOW_SEED` (not `--seed`, which is ignored under
`ODE_FLOW_REC`) and use ≥3 seeds (`docs/TEXABC_REFERENCE_TABLE.md`).

**Timeline (T0 ≈ 10:15).** T0–T1h: reference diagnostics (running). When the
Opus A-changes land: adversarial review (~30–60 min). Review passed → V1 /
V1b / V2 launch in parallel on the fresh holders → +80 min evals → +10 min
diagnostics → verdict against the reference scale. Buffer ≥ 2 h inside the
10 h window for one relaunch or an extended-step follow-up.

## C. Contradictions a sub-agent could trip over (flagged, prose left intact)

- **Texture supervision is adversarial only (researcher rule, 2026-08-23).**
  The anisotropic statistics are instruments (A1 battery, A5 tripwire, A9
  measurement), never loss terms.
- **Stale phasing text**: Point 2's "Phasing note: pixel is Phase 2" and
  Point 4's "[CC] Decisions taken — Phase 1 = latent only" (and its latent
  design table) predate the A/B/C diagnostic; standing decision 1 supersedes
  them — **pixel IS Phase 1**. Point 6 carries the STATUS marker; Points 2/4
  do not. Do not build the latent critic from those sections.
- **Standing decision 4 vs resolved [CC](a)**: decision 4 ("never from
  scratch") still reads absolute; the recorded resolution allows the
  from-scratch pixel PatchGAN for Phase 1 with mitigations. The resolution
  governs.
- **Standing rule 2 scope** ("real and fake have to be aligned"): read as
  alignment of noise level / timestep / conditioning between the real and fake
  sides (the scorer-symmetry lesson) — NOT scene-level pairing, which the
  band-matched unpaired texture reals deliberately do not have. If it were
  scene pairing, the whole marginal texture design would be invalid — confirm
  with the researcher if in doubt.
- **Point 1's test arms C/D** target the parked transition critic — do not run
  them in Phase 1.

---

### Standing decisions (apply to every point below)

1. ~~**Phase 1 is latent-space only.**~~ **SUPERSEDED 2026-08-23 by the A/B/C
   diagnostic** (see the section immediately below). The diagnostic returned
   **Case 1** — the VAE can represent the texture we want, and the corruption is
   born in the student latent — which *empirically authorises* a decoded-RGB
   critic. Pixel therefore moves **into Phase 1**, and the memory objection is
   handled by latent-crop-then-decode (~8x reduction) rather than by deferral.
   The latent-phase reasoning below is retained because most of it carries over
   unchanged (patch logits, per-frame, no nearest-matching, no equalisation);
   only the *domain* changed. Phase 2 is now the **dynamical** question, not the
   pixel question.
2. **The transition / temporal critic is PARKED — entirely, for the whole
   effort, not merely deprioritised.** We do not need temporal-adversarial
   supervision at the moment. Since our existing `gt_transition` critic *is* the
   transition critic, this means **the current GAN is switched off** and Phase 1
   is a new texture critic built alongside a disabled one. Everything in
   `GAN_ARCHITECTURE_BRIEF.md` §3 and §5 — pair construction, nearest-L1
   matching, the cross-ride ring, `match_k`, mean/cross equalisation, the
   former/latter detach — is **out of scope** until it is unparked.
3. **The texture critic is the whole of Phase 1.** It is the urgent need.
4. **We never train a critic from scratch.** Every discriminator projects onto a
   **pretrained** network. In Phase 1 (latent-only) the pretrained options are
   Wan models, because they are the only nets native to this latent space; if we
   move off the current DMD teacher, we move to **Wan 2.1-T2V-14B**. Phase 2 may
   use a well-pretrained pixel-domain critic.

---

## The decisive diagnostic (2026-08-23) — what it settled and what it did not

Three distributions were measured on the same ride with the same texture
instruments — directional 2D spectrum, wavelet LH/HL/HH, Laplacian variance,
kurtosis, local covariance, scanline anisotropy:

| | |
|---|---|
| **A** | raw GT RGB |
| **B** | `decode(encode(GT))` |
| **C** | `decode(student latent)` |

**Verdict: Case 1.** Full numbers in `eval/texture_abc_strict03/REPORT.md`.
Design consequences are folded into the EXECUTION PLAN at the top of THIS
document (A3/B2) — this document is the spec.

| finding | number |
|---|---|
| A ≈ B | B/A HF power 0.86–0.89; Laplacian kurtosis 13.4 → 13.0/14.0; anisotropy ~1.0; angular entropy ~0.98 |
| C fabricates HF | 1.49× raw at ~10 s, 2.21× at ~25 s (consistent with 3.6× at 60 s) |
| directional collapse | fy/fx anisotropy A 1.10 → C_early 0.93 → **C_late 0.36**; angular entropy 0.98 → 0.84 |
| contrast wash | luma kurtosis −0.19 (A) → −1.2 (C) |

### Phase 1 implications (all now actionable, and all confirm the texture critic being built)

1. **The failure is localised.** The VAE is not the bottleneck — it merely
   *renders* the corruption. The student progressively produces pathological
   latents. **Case 2 is excluded**: the ~0.84–0.89 HF reduction is ordinary
   compression smoothing; it does not invent stripes, destroy angular diversity
   or collapse texture statistics.
2. **The student is not losing texture — it is recursively manufacturing it.**
   1.49× → 2.21× → 3.6× with rollout depth is an unstable feedback process:
   small fake texture → committed to context → treated as real structure →
   more fake detail added → committed again. This is why perceptual metrics call
   it "sharp"; there is plenty of detail, it is simply *wrong* detail.
3. **The anisotropy result is the stronger half.** 1.10 → 0.93 → 0.36 with
   angular entropy 0.98 → 0.84 means spectral energy that was broadly spread
   over orientations concentrates into a narrow family — the mathematical
   counterpart of the visible stripe. So the critic's question is **not** "is
   there enough HF" but "**is this natural multi-orientation texture or
   high-energy structured fake texture**".
4. **This retro-explains why the wavelet GAN made things worse.** A crude
   "HF = realistic" pressure is almost perfectly *anti*-aligned with our failure,
   because the model already has too much HF. Our measured result — the wavelet
   arm dying in scanline banding — is exactly what that misalignment predicts.
5. **The RGB critic now has an empirical warrant, not an architectural
   preference.** The failure can live in a latent direction the frozen teacher
   considers innocuous while the VAE renders it as stripes. Asking a frozen Wan
   representation "does this look realistic to Wan?" is a different question from
   "when rendered, does this patch look like real video?" — and they demonstrably
   diverge here. The decoded critic closes that loophole.
6. **`decode(GT)` as real, never raw RGB.** The decoder removes ~10–15 % of HF.
   With raw RGB as real, the critic learns "sharp = real, softened = fake"; the
   generator cannot change the frozen decoder, so its only recourse is to inject
   *more* HF into the latent hoping some survives — i.e. a recipe for worsening
   the exact failure being treated. Same decoder on both sides cancels the
   transfer function.
7. **Patch logits are justified by the artefact's geometry.** A stripe occupies
   part of a frame; a global scalar permits "85 % excellent + 15 % horrible =
   reasonably real". Patch logits localise the generator gradient to where the
   texture actually breaks.
8. **Clean division of responsibility**: DMD = generate correct world content;
   CARN = keep committed latent statistics from drifting; RGB PatchGAN = make
   local rendered appearance statistically natural.
9. **The success metric must be JOINT, or it is gameable.** Anisotropy → 1.0
   alone can be satisfied by replacing vertical stripes with **isotropic
   high-frequency snow**. Track together, against the **B** distribution:
   **HF power** (must come *down*), **anisotropy** (→ ~1.0), **angular entropy**
   (→ ~0.98), **kurtosis**. This is a pre-registered falsification guard, not a
   reporting preference.

### Phase 2 — the question the diagnostic explicitly did **not** settle

10. **This does not prove adversarial training cures the recurrent instability.**
    The 1.49 → 2.21 → 3.6 progression *with depth* means an underlying
    autoregressive amplification mechanism exists. A pixel GAN may teach the
    student not to enter that region of latent space — or it may merely suppress
    the visible manifestation while another recurrent error surfaces later.
11. **Therefore two separate success criteria, and both must be reported:**
    - **Local correction** — does the arm restore HF power, anisotropy, angular
      entropy and kurtosis toward the **B** distribution *at a fixed depth*?
      (The pre-registered `C_late 0.36 → ~1.0` metric covers this.)
    - **Dynamical correction** — does that improvement **persist at 10 s, 25 s
      and 60 s**, or does it merely postpone divergence? Running the A/B/C
      diagnostic at all three depths on the new checkpoint is the test.
    Criterion 2 is the deeper research object and is where the residual —
    whatever neither texture-realism nor CARN's drift correction explains —
    will show up.

### [CC] Two contradictions between the implemented design and decisions already taken

Flagging these now because both are cheap to fix before the arm runs and
expensive to discover afterwards.

**(a) `TEXTURE_GAN_DESIGN.md` §4 specifies a from-scratch PatchGAN. That
violates standing decision 4** ("we never train a critic from scratch; we project
onto pretrained networks"). The rule was stated when Phase 1 was latent-only,
where the *only* pretrained options were Wan models. **Pixel space removes that
constraint entirely** — and the researcher's own framing was that pixel and
pretrained go together ("Phase 2 we can use some really well pretrained pixel gan
kind of thing"). Now that pixel *is* Phase 1, the pretrained option is available
and should be taken: **DINOv2 is precisely what ADD used for this job**, and the
SD/VQGAN discriminator weights are a second candidate trained for exactly this
objective on decoded latents. A from-scratch 2–3 M-param critic also inherits the
budget risk noted in Point 6 — ~180 updates to learn texture statistics from
nothing, with `d_loss ≈ ln 2` indistinguishable from "design is wrong".
**RESOLVED 2026-08-23 — deferred to Phase 2 (researcher's call), and I think that
is the right trade.** Recording the reasoning so it is a decision and not a
lapse:

- The **VQGAN/SD precedent actively supports from-scratch here.** The latent space
  we operate in was itself produced by a from-scratch pixel PatchGAN. This is the
  reference implementation of the exact job, not an exotic choice.
- The hypothesis class is small and the target is low-order. A 2–3 M-param
  PatchGAN with a ~70 px receptive field learning a *marginal texture* statistic
  is a far easier learning problem than the semantic features a pretrained
  backbone would supply. Classical texture work (Gram matrices of shallow
  features) makes the same point.
- Getting a first signal quickly beats optimality. If a from-scratch critic
  separates, the design is validated and a pretrained backbone becomes an
  upgrade; if it does not, we escalate with evidence.

**But the budget risk from Point 6 transfers intact and must be mitigated, or the
arm will be uninterpretable.** ~180 D-updates to learn texture statistics from
nothing, and `d_loss ≈ ln 2` reads identically for *undertrained* and *wrong
design*. Two cheap mitigations, both recommended:

1. **Run the texture arm longer than 200 steps** (400–600) and read the `d_loss`
   **trajectory**, not its endpoint. A from-scratch critic is *expected* to sit at
   chance early; that is not evidence of anything.
2. **Add a positive control** — this is the decisive one, and it costs almost
   nothing. Periodically score `decode(GT)` against a *deliberately corrupted*
   real: `decode(GT latent + structured HF perturbation)`, or simply
   `decode(GT)` versus the C_late student output already on disk. Then:
   - critic cannot separate the obvious corruption → **undertrained**, extend
     the budget;
   - critic separates the corruption but not the student → **informative**: the
     student's texture is genuinely close at this depth, and the failure is
     elsewhere (or deeper in the rollout than the arm reaches);
   - critic separates both → healthy, read the arm normally.

   This converts the ambiguous `ln 2` outcome — the one that has burned this
   campaign repeatedly — into a decidable one. It belongs in
   `TEXTURE_GAN_DESIGN.md` §8 alongside the existing falsification criteria.

**(b) §5 keeps position-matched RpGAN at patch level.** Point 1 concluded, and
the researcher agreed, that spatially-matched relativistic loss asserts a
correspondence that does not exist between unrelated reals and fakes.

*Partial defence*: since reals are drawn **uniformly at random**, position-matched
pairing degenerates to the *shuffled-pair* RpGAN variant, which Point 1 listed as
defensible. So this is milder than it first appears.

*Residual concern*: in dashcam footage **vertical position is a strong proxy for
content class** — sky at top, buildings mid, road surface at bottom — and these
have genuinely different texture statistics. Random crops mean a fake *road* patch
can be paired against a real *sky* patch. For a marginal critic this averages out,
but it adds variance and could drag road texture toward sky texture.

**RESOLVED 2026-08-23 — do both: band-match *and* raise the sample count. But the
sample count is much smaller than the patch-grid count suggests.**

The intuition "sample so many patches that it does not matter" is right in
principle, and wrong about the arithmetic if read off the patch grid. With the
design's `pix_crop_lat=(24,32)` → 192×256 px, a stride-8 patch grid, and a ~70 px
receptive field:

| quantity | value |
|---|---|
| patch logits per image | **768** |
| non-overlapping receptive-field tiles per image | **6** |
| overlap factor | **~128 : 1** |

Adjacent patch logits share almost all of their receptive field, so they are not
independent samples of the texture distribution. Effective sample size per step:

| config | fake images/step | patch logits | **effective** |
|---|---|---|---|
| current (`crops=2, frames=2`) | 4 | 3 072 | **~24** |
| `crops=4, frames=3` | 12 | 9 216 | **~72** |
| `crops=8, frames=3` | 24 | 18 432 | **~144** |

So averaging-out has to be bought with **more crops and more frames**, not with
more patch positions inside a crop — the latter is nearly free of new information.
Crops are cheap (that is the whole point of latent-crop-then-decode), so raising
`pix_crops_per_step` and `pix_frames_per_crop` is the affordable lever; note the
G-side grad-decode cost scales with it, while the D-loop decodes stay `no_grad`.

**And band-matching is still worth doing**, because it reduces the same variance
*at fixed sample count* rather than paying compute for it. The two are
complementary: band-matching cuts the per-sample variance, more crops cut the
per-step variance. It is nuisance-matching in the sense of Point 7 (match the
nuisance covariate, never the target property), costs one index constraint, and
does not reintroduce nearest-L1.


---

## Point 1 — the global average over patch logits is the wrong reduction

### The claim

The discriminator has a *local* head — one 1x1 conv and one 3x3 residual block on
a 30x52 feature grid — which produces thousands of local logits, and then
collapses them:

```python
visual_logits = visual_logits.mean(dim=1, keepdim=True)   # model/ladd_disc.py:911-915
```

**before** the adversarial nonlinearity. With `gt_transition` inputs that is
5 taps x 6 frames x 1560 tokens = **46,800 local realism votes averaged into one
number.** It is no longer a PatchGAN.

The reduction order matters. Ours is `softplus(mean_i[D_f(i) - D_r(i)])`, whereas
a patch discriminator gives `mean_i[softplus(D_f(i) - D_r(i))]`. Since softplus is
convex, Jensen gives `softplus(E[x]) <= E[softplus(x)]`: a disastrous set of fake
local patches can be **cancelled by sufficiently normal patches before the loss
ever sees them**. For a failure that reads as "there is horizontal garbage over
15 % of the road", that is exactly the wrong reduction.

Canonical VQGAN / Stable-Diffusion autoencoder training does the opposite — the
discriminator emits a spatial patch map and the adversarial loss averages the
patch logits *after* evaluating them. The Stability/CompVis implementation
explicitly handles discriminator outputs shaped `[B, 1, H, W]`.

### [CC] A second, sharper mechanism (stronger than the Jensen argument here)

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

### [CC] Implementation — cheaper than it looks

Both halves are largely already in the codebase.

1. **`ladd_scalar_output=false`.** This is the *default*
   (`causal_action_forcing_train.py:883`); our arms explicitly set it true. With
   it false, `rpgan_d_loss` already does `.mean()` over all dims, i.e. it already
   computes `mean_i[softplus(·)]`. **The convexity fix is a flag flip.**
2. **`ladd_r1_normalize_tokens=true`.** Currently false. The inline comment at
   `causal_action_forcing_train.py:7071-7076` states that with token logits R1's
   `grad_sq` estimates `‖∇ Σ_i D_i‖²`, which "scales with ~T² over ~47k tokens,
   forcing γ to a tiny un-portable value." **This is ~~very likely the mechanism
   behind our replicated "R1 γ=1e6 pins every discriminator at ln 2" finding.**~~ **[WITHDRAWN 2026-08-23 — FALSIFIED: `_roll_holder.sh` ran `ladd_r1_gamma=1e6` TOGETHER WITH `ladd_r1_normalize_tokens=true` and token logits (no `ladd_scalar_output`), and still pinned at ln2. The T^2 artefact cannot explain it; gamma was simply ~1e6x too large. This removes the main justification for A8 fix 2 / A12.]**
   Normalising makes R1 estimate `‖∇ mean_i D_i‖²`, token-count-independent, so
   patch logits and a portable γ are compatible. This reframes a headline prior
   result as a scaling artefact rather than a law.
3. **New non-relativistic patchwise loss** — the only genuinely new code. Add
   `ladd_loss_form ∈ {rpgan, nsgan, hinge}` (default `rpgan` = byte-identical),
   with `nsgan`/`hinge` as above. R1/R2 are unchanged in form and still apply.

### [CC] Risks, middle options, and open questions

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

---

## Point 2 — the frozen feature taps are poor candidates for texture, and the latent grid may be too coarse to represent the artefact at all

### The claim

We tap blocks **6, 12, 18, 24, 29** and have **nothing before block 6**
(`causal_action_forcing_train.py:737-746`). LADD itself is materially richer: it
re-noises the latent, extracts the full token sequence **after each attention
block**, and places **independent** heads on those representations, emphasising
that **low-noise generative features carry the texture / local-detail feedback**.

So this implementation is not "LADD on Wan". It is *five sparse, relatively mature
Wan representations, globally fused and reduced*. Our earliest feature already
reflects a substantial amount of transformer processing. For a texture detector
the tap set should include the **patch-embedding output, blocks 0/1, 2/3**,
perhaps 6, and only then a couple of later layers for structure.

There is also a physical resolution issue. The VAE compresses RGB by 8x
spatially; the Wan DiT then patches the latent by a further 2x2, so a single
transformer spatial token has a **nominal 16x16 RGB footprint before the
transformer does anything**. Fine grain — demosaicing-like structure, 1-4-pixel
scanlines, edge ringing, decoder-phase artefacts — is not naturally representable
as a spatially localised decision at that granularity. Latent channels can encode
sub-token detail (that is why latent diffusion works), **but a latent token
representation is not equivalent to looking at the resulting pixels.**

### [CC] Three findings from checking this against the implementation

**(a) Earlier taps are a config change; the patch embedding is not.**
`_validate_block_indices` (`ladd_disc.py:138-147`) only requires
`0 ≤ idx < n_blocks`, so `ladd_feature_blocks=[0,2,4,8,29]` works today with no
code change. The **patch-embedding output is not reachable** — the projector's
`_find_blocks` locates only the transformer-block `ModuleList`, so hooking the
patch embed is a small addition to `WanFeatureProjector`.

Minor fidelity note: our hook fires on the **whole block output** (attention +
FFN + residual, `ladd_disc.py:231-241`), whereas LADD taps after the attention
block. Similar granularity, not identical.

**(b) A 6× channel bottleneck sits in front of every head.** `dim_teacher` is
**1536** for Wan-1.3B and CCM projects it to `ladd_proj_dim = 256`
(`ladd_disc.py:294`) — a 6× compression at every tap **before the head sees
anything**. Whatever texture information survives the deep taps then passes
through a learned 1×1 that was never asked to preserve it.

**(c) The sampled-t arms spend most of their capacity where texture does not
exist.** This is the compounding problem. `ladd_disc_timestep_shift = 5.0`
applies `t = 1000·s·u/(1+(s−1)u)` to `u ~ U[0.02, 0.98]`
(`causal_action_forcing_train.py:5776-5782`). Measured distribution:

| shift | p25 | median | p75 | fraction `t > 625` | fraction `t < 208` |
|---|---|---|---|---|---|
| 1.0 | 260 | 502 | 741 | 0.37 | 0.20 |
| **5.0 (ours)** | **637** | **834** | **935** | **0.76** | **0.03** |

**76% of discriminator samples land above t=625 and only 3% below t=208.** LADD's
point is that *low*-noise features give texture feedback; we sample overwhelmingly
*high*-noise. The `strict03` and projected arms — introduced precisely to escape
the wavelet branch's forced `t=0` — replaced it with a distribution in which
texture has largely been destroyed by noise. Both settings are wrong for texture,
in opposite directions. A texture critic wants the low-t mass: either a fixed
small `t` or `ladd_disc_timestep_shift < 1`.

### The resolution / physics objection

The VAE compresses RGB by **8× spatially**; the Wan DiT then patches the latent
by a further **2×2**. A single transformer spatial token therefore has a nominal
**16×16 RGB footprint before the transformer does anything** — and after block 6
its receptive field is global anyway (Point 1's caveat).

Fine-grain structure — 1–4-pixel scanlines, edge ringing, demosaicing-like
patterning, **decoder-phase artefacts** — is not naturally representable as a
spatially localised decision at that granularity. Latent channels can certainly
*encode* sub-token detail; that is why latent diffusion works. But a latent token
representation is not equivalent to looking at the resulting pixels.

This matters directly for our observed failure: `horizon_wave90` dies in
**horizontal scanline banding**, which is precisely a decoder-phase artefact
class. **A latent-space critic can never see the artefact it is supposed to
punish** — the artefact is manufactured downstream of everything it observes.

### [CC] The logical endpoint of this argument

If the useful taps are the earliest ones, those features are close to a linear
patch embedding of the noisy latent, carrying little beyond the raw latent
itself. At that point the DiT is not earning its place as a texture backbone, and
the honest conclusion is:

> **Stop using the frozen DiT as the backbone for the texture branch.** Keep the
> DiT critic for structure/semantics, and add a *separate, small, non-DiT texture
> critic* — a conv stack on raw latents, or on decoded pixels.

**Phasing note (decided after this was written): pixel is Phase 2.** Decoder
gradients are expensive, so Phase 1 stays in latent space — see Point 4 for the
latent texture critic that replaces this recommendation for now. The rest of
this section stands as the Phase-2 argument.

**A pixel-space critic has in-repo precedent and is not a from-scratch
capability.** `_vae_decode_grad` (`causal_action_forcing_train.py:3731-3775`)
already performs a **gradient-carrying** VAE decode with checkpointing, and is
already used to build pixel-space `pred_pix` / `gt_pix` pairs on a frame subset
(`:3812-3828`). The open question is cost, not feasibility: the decode is 4×
temporal expansion at full resolution, so a pixel critic would likely run on a
*sampled subset of frames* and possibly on random crops rather than whole frames.

### Concrete changes proposed

1. **Re-tap** (config only): `ladd_feature_blocks=[0, 2, 4, 8, 29]` or similar —
   shift the mass early, keep one late tap for structure.
2. **Hook the patch embedding** (small code change to `WanFeatureProjector`).
3. **Widen `ladd_proj_dim`** for the early taps, or use per-tap dims, so the 1×1
   is not a 6× bottleneck on the texture-bearing taps.
4. **Fix the disc timestep for the texture objective**: fixed low `t`, or
   `ladd_disc_timestep_shift ≈ 0.25–0.5` to move the mass under t≈357.
5. **Pixel-space texture critic** on decoded crops, as a separate branch with its
   own weight, reusing `_vae_decode_grad`.

### [CC] Risks

- Early DiT features may be nearly raw latent, in which case (1)–(3) buy little
  and the real answer is (5). This is a reason to run the tap sweep and the pixel
  critic as *separate* arms rather than bundling them.
- A pixel critic costs a differentiable decode per D-update. Budget it against
  the existing per-step cost (~24 s/step) before committing.
- Changing the tap set changes `dim_teacher` bookkeeping and the CSM fusion
  ordering; verify the FPN top-down path still makes sense with unevenly spaced
  taps.

### Test

Separate arms against the `nogan200` control, each with Point 1's patch-logit
fix already in place so the reduction is not confounding:

| arm | change |
|---|---|
| E | re-tap early `[0,2,4,8,29]`, disc t fixed low |
| F | E + patch-embedding tap + wider `proj_dim` |
| G | pixel-space texture critic on decoded crops (DiT critic off) |

Readout: does the scanline-banding mode disappear, and does `cos(g_GAN, g_DMD)`
stop being anti-aligned.

---

## Point 3 — a latent GAN is not inherently wrong, but it is decoder-blind

### The claim

Latent adversarial training demonstrably works. **LADD** does high-quality latent
adversarial distillation without decoding to RGB; **APT** runs its discriminator
directly on latent-space diffusion features; **AAPT** uses a latent causal
discriminator for long autoregressive video; **One-Forcing** — closest to our
setup — explicitly uses **no decoded-frame discriminator**. So "latent GAN" is not
the error, and concluding otherwise would throw away useful evidence.

But our use case has a special requirement. **The entire raison d'être of this GAN
is to detect visual texture artefacts that only matter once the VAE decoder
renders them to RGB.** A latent-only critic never sees `latent --[decoder]--> pixels`,
so **it cannot reason about the decoder's Jacobian**. A tiny adversarially
favourable change in latent space can become a large *structured* pixel artefact
after decoding, and the scanline result is highly consistent with exactly that
exploit: the GAN discovers *"the frozen Wan representation rewards moving the
latent this way"* without knowing that *"the decoder turns that direction into
horrible horizontal banding."* A pixel discriminator would know.

### [CC] Assessment

> **STATUS 2026-08-23: VINDICATED AND PROMOTED.** The A/B/C diagnostic confirmed
> the decoder-blindness argument empirically — the corruption is born in the
> student latent and is invisible to a critic that never sees the render. Pixel
> moved from Phase 2 into Phase 1 on the strength of it. The Phase-1
> no-grad decoder tripwire below is retained; for GAN arms it is subsumed
> by the pixel critic seeing the render directly.

I agree, and the distinction is worth stating precisely, because it changes what
Phase 1 has to do rather than invalidating Phase 1.

The cited systems all use latent critics for **distillation fidelity** — matching
a teacher's output distribution, where the decoder is a fixed shared postprocess
applied identically to both sides and therefore cancels. Our critic is being asked
for something different: to *suppress an artefact class produced by the decoder
itself*. The decoder cancels for LADD/APT/AAPT; it does not cancel for us. That is
the actual asymmetry, and it is why "latent GANs work elsewhere" does not license
"a latent GAN will fix our texture".

So the honest Phase-1 position is: **the latent texture critic is not expected to
be able to see the banding class directly.** It is expected to attack the
*upstream* cause — the measured high-frequency latent drift — and we must
instrument for the possibility that it gets gamed.

**Concrete Phase-1 mitigation — a no-grad decoder tripwire.** We can get decoder
*awareness* without decoder *gradients*: every N steps, decode a small sample of
generated latents under `torch.no_grad()` and run the banding/HF detector on the
result (`analysis/rollout_quality.py` already computes Laplacian-variance and
directional-band statistics). If the latent critic starts winning while decoded
HF energy climbs, we have caught the exploit in training rather than in a
60-second eval two hours later. Cost is one forward decode every N steps, no
backward — affordable in a way the gradient path is not.

**Where this argument does hold fully:** it is a decisive argument against ever
expecting the *current* latent transition critic to fix texture, and it is the
strongest reason to keep Phase 2 alive as a planned step rather than a fallback
we hope never to need.

---

## Point 4 — build a hybrid two-critic system

### The claim

Do not ask one critic to be simultaneously a motion critic, action critic,
temporal transition critic, texture critic, frequency critic and global-realism
critic. Split the problem.

**Critic A — texture critic.** One job: *does this frame locally look like real
decoded video?* A small 2D multi-scale PatchGAN on
`generated latent frame -> frozen Wan VAE decoder -> RGB -> PatchGAN`.

Crucially, **real examples must traverse the same VAE path** — GT latent through
the same frozen decoder to reconstructed-real RGB — rather than raw camera RGB.
Otherwise the discriminator simply learns *"raw camera image = real, VAE
reconstruction fingerprint = fake"*, which the generator **cannot** eliminate.
Comparing against original RGB is a separate later experiment.

Decode sparsely: one random latent frame from the latter chunk, or 2 frames per
GAN iteration. There is no need to decode a whole 1.5-second transition to
supervise road texture. Random pixel crops if memory is tight.

**Critic B — latent temporal/action critic.** Keep the latent pathway for what it
is genuinely good at: motion consistency, trajectory plausibility, action
compliance, longer temporal state. Can be the existing scalar transition
discriminator, an AAPT-like per-frame scalar critic, or One-Forcing-style register
attention. Make this one **strictly action-conditioned**; the texture critic can
be deliberately **action-blind**. That is a far cleaner decomposition than
debating whether *the* GAN should be blind or conditioned.

### [CC] Assessment — agreed, and the VAE-fingerprint point is the sharpest thing here

The reconstructed-real-vs-raw-camera insistence is the most important detail in
this section and would have been an easy trap. A PatchGAN handed raw camera RGB
as "real" gets a free, permanently-winning feature — VAE reconstruction error —
that the generator has no means to remove because it does not control the
decoder. The critic would saturate, the generator would receive a large
unfixable gradient, and we would have manufactured a much worse version of the
failure we already have.

**That trap has a latent-space analogue, and Phase 1 happens to be clean of it.**
In Phase 1 both sides are latents: GT latents are VAE *encoder* outputs, student
latents are DiT outputs. There is no encode/decode asymmetry to exploit — the
distributional difference between them *is* precisely the target. Worth stating
explicitly so nobody "improves" Phase 1 by roundtripping one side.

The A/B split also dissolves a question we have been relitigating for days.
Action-blind-vs-conditioned was never one question: **texture wants blind**
(texture realism is action-independent, and conditioning only adds a nuisance
variable), **motion wants strict**. Every arm so far forced one answer on both
jobs at once.

### [CC] Decisions taken, and what Phase 1 actually is

Per the researcher's direction:

- **Phase 1 = latent only.** Decoder gradients are too expensive to take on now.
  Critic A as specified (decode → PatchGAN) becomes the **Phase-2** design and is
  recorded here in full so it is ready to build.
- **Critic B is parked outright** (standing decision 2). Temporal/action
  adversarial supervision is not needed at the moment. This has a large
  simplifying consequence (below).
- **The texture critic is the whole of Phase 1.**

**Consequence of parking Critic B: the current GAN turns off entirely.** Our
existing `gt_transition` critic *is* Critic B — a temporal transition critic. So
Phase 1 is not a modification of the current GAN; it is a **new, small critic
built alongside a disabled one**. That is cleaner than it sounds, and it retires
a whole cluster of problems for free:

- the nearest-L1 real matching and its degenerate support
  (`GAN_ARCHITECTURE_BRIEF.md` §5) — **gone**; a texture critic's reals are just
  GT latent frames, no matching, no ring, no `match_k`;
- mean/cross equalisation stripping absolute level and per-channel DC — **gone**;
- the former/latter detach question — **gone** (single frames);
- action-blind-vs-conditioned — **settled** (blind).

### [CC] Phase-1 design: latent texture critic

Combining Points 1–3 with the phasing decision:

| property | choice | why |
|---|---|---|
| domain | **raw 16-channel latent**, not DiT features | Point 2: DiT taps are deep, globally-attended, and 6×-compressed |
| backbone | **pretrained Wan 14B, early blocks only** (truncated prefix) | standing decision 4 (never from scratch) + Point 6 (must not be the DMD teacher's weights) |
| spatial grid | 30×52 (DiT token grid) | NOTE: a conv critic on the un-patched 60×104 latent would be 2× finer, but that path is closed by decision 4 — flagged as the resolution cost of using a pretrained DiT backbone |
| output | **patch logits**, no global mean | Point 1 |
| loss | patchwise non-saturating logistic or hinge | Point 1 — no false spatial correspondence |
| temporal support | **single frames** | texture is per-frame; coherence is Critic B's job, parked |
| conditioning | **action-blind**, no prompt | Point 4 |
| reals | GT latent frames, unmatched | no ring, no nearest-L1 |
| noise level | **fixed low t** (or shift < 1) | Point 2(c): shift 5.0 puts 76% of samples above t=625 |
| R1/R2 | keep, with `normalize_tokens=true` | Point 1: makes γ portable on patch logits |
| tripwire | no-grad decode every N steps → HF/banding detector | Point 3: catch decoder-blind gaming during training |

**Build cost — VERIFIED, and it is a real build.** The `gan_disc_*` config
family (`gan_disc_in_channels=16`, `gan_disc_base_channels=64`,
`gan_disc_num_blocks=4`) describes exactly this shape, but **no implementation
sits behind it**: `gan_backbone` *raises* for any value other than
`'ladd_teacher_feat'` (`causal_action_forcing_train.py:686-692`). Those knobs are
dead config. Phase 1 therefore requires a new discriminator module plus a second
`gan_backbone` branch and its optimiser/DDP wiring — not a flag flip. It is still
a small module (a 4-block conv PatchGAN on 16 channels), and it is *independent*
of the LADD path, so it can be added without touching the parked transition
critic at all.

### [CC] Risks

- A per-frame latent texture critic has **no temporal term at all**, so it can
  reward per-frame-plausible, temporally-incoherent texture (shimmer). The
  no-grad tripwire should track a temporal statistic too, not only per-frame HF.
- Point 3 stands: this critic cannot see the banding class directly. If Phase 1
  produces a critic that trains healthily and still yields decoded artefacts,
  **that is the trigger for Phase 2**, and it is a clean, informative outcome
  rather than a failure.
- Adversarial scale must be re-bracketed from scratch: new domain, new
  reduction, new loss form. The 0.01/0.03/1.0 bracket does not transfer.

### Test

Against the `nogan200` control on the fixed-route 60 s eval, with the transition
critic off in every arm:

| arm | critic |
|---|---|
| A (control) | none |
| H | latent conv PatchGAN, per-frame, blind, low t, patch logits |

---

## Point 5 — condition the critic on the timestep the chunk came from, and cap the aggressive late gradients

### The claim

Give the GAN the **timestep at which the current chunk latent was taken**, so it
knows what noise level it is looking at and does not punish a legitimately
less-sharp early-rung latent as a texture failure. If it still over-penalises and
breaks down, **downweight the aggressive gradients that arrive near the end** of
the denoising ladder.

The transition critic handled this implicitly by carrying the **previous chunk as
a reference** — the critic judged relative change rather than absolute quality.
That reference is being removed when the transition critic is parked, so the
timestep has to take its place. Textures must work before the transition is added
back.

### [CC] Assessment — the reference argument is the important part

The observation that *the transition pair was functioning as an implicit
normaliser* is the strongest part of this, and it is easy to miss. A per-frame
texture critic has no within-sample reference at all: it must judge "is this
latent's texture right" in absolute terms, against reals drawn from other scenes.
Timestep conditioning restores a calibration axis that the parked design was
providing for free. I agree it is required, not optional, for Phase 1.

### [CC] Prerequisite — today the provenance timestep is a constant, so conditioning would be a no-op

The rollout picks a **random exit rung per step**
(`model/dmd_action_forcing.py:6559-6579`, which logs `exit_rung_t_from/to`
precisely because unlabelled random-rung renders were uninterpretable). But the
GAN's fake is **not** the random-rung sample: with `flash_dmd_enabled=true` it is
`flash_dmd_gan_chunk` — the flash pass at the fixed `flash_dmd_gan_t = 60`
(`:9545`, surfaced at `:10624-10637`).

So **every fake the critic has ever seen came from the same t=60 refinement.**
Conditioning on that provenance would feed the critic a constant and change
nothing. The proposal only becomes live if the provenance is made to vary:

- **(a) sample the flash timestep** (`flash_dmd_gan_t` drawn per step rather than
  pinned at 60), or
- **(b) take the fake from the actual random exit rung** (flash off for the GAN
  path), which is also the more faithful target — it is what the ladder really
  produces.

> **[CC] CORRECTED 2026-08-23 by A2 — both options above were wrong.**
> (a) is **not** the smaller change: `pipeline/action_forcing_training.py:2283`
> reassigns `cache_pred = flash_dmd_pred.detach()`, so the flash tensor **is**
> what training commits — sampling `flash_dmd_gan_t` would change the committed
> KV content as well as the fake.
> (b) is **doubly wrong**: the exit-rung x0 (`denoised_pred`) is *never*
> committed — the post-exit finish loop (`:2139-2165`) supersedes it — and it is
> not what inference produces either. The exit rung selects only where the
> gradient attaches; the full ladder always runs, so the commit is structurally
> exit-rung-independent.
> The change that actually restores inference parity is **turning flash off**,
> which makes the commit the ladder endpoint. See A2 for the ranked fixes.

(b) is the more honest object to supervise but changes what is committed to the
KV cache, since flash reassigns `cache_pred` (`pipeline/action_forcing_training.py`
~2229-2242); (a) is the smaller change. **[UNVERIFIED]** — worth checking whether
`utils/eval_causal_AR.py` commits the ladder endpoint while training commits the
t=60 flash prediction. If those differ, the texture critic is shaping a tensor
that is not the one whose texture appears in the evaluation video, and that
mismatch would matter more than the conditioning itself.

### [CC] The giveaway trap — conditioning must be distribution-matched

This is the same class of error as the VAE-fingerprint trap in Point 4, and it
would be easy to walk into.

GT reals have **no** generation rung. If fakes carry their true rung and reals
carry a sentinel, a default, or zero, then the conditioning field *is the label*:
the critic reads it directly, wins instantly, and the generator receives a large
gradient it can never reduce.

The rung must therefore be a **shared conditioning variable, distribution-matched
across real and fake** — each real is assigned the rung of the fake it is scored
against (or drawn from the same distribution). This is exactly how `disc_t_int`
is already handled: one scalar, shared by both halves, passed as the disc's
`timestep` argument (`causal_action_forcing_train.py:5787-5790`). The new
conditioning should follow that established pattern rather than invent a second
one.

Note this is a *second, distinct* timestep from `disc_t_int`: one is "how much
noise did I add before showing you this", the other is "which rung produced it".
Both are legitimate and they should be separately named and separately logged, or
they will be confused in six weeks.

### [CC] Cap, do not downweight

I would change one thing about the fallback. "Downweight the aggressive gradients
near the end" is right in intent but the wrong instrument.

"Near the end" is rungs **357 and 208** — and those rungs are where the student's
**final image is set**. That is precisely where texture signal is wanted most. A
fixed per-rung downweight is always on, so it removes the signal we are trying to
deliver in exactly the place we are trying to deliver it.

A **norm cap** is inactive in the normal regime and only engages on the anomalous
excursions the proposal is actually worried about. We already have the precedent
and the machinery: `dmd_grad_target_norm` is a **cap-only rescale** of the DMD
gradient (`model/dmd_action_forcing.py:2571-2592`, applied at `:5477-5511`). The
matching knob is a `gan_grad_target_norm` built the same way.

Suggested order:

1. **Cap first** — `gan_grad_target_norm`, cap-only, calibrated to the median
   `‖g_GAN‖` of a healthy run (the same calibration procedure the DMD cap needs).
2. **Per-rung weight only if the cap proves insufficient**, and then as a mild
   taper rather than a step, with the taper shape logged.
3. Either way this needs the `‖g_GAN‖ / ‖g_DMD‖` and `cos(g_GAN, g_DMD)`
   telemetry to calibrate against — it cannot be tuned blind.

### [CC] A second substitute for the parked reference

Timestep conditioning restores a *noise-level* calibration axis, but not a
*scene* reference — the transition pair gave the critic both. A per-frame texture
critic still has to judge absolute texture against out-of-scene reals.

A cheaper substitute for the scene half, which does **not** reintroduce the
transition: condition the critic on the **seed chunk's texture statistics** — the
same fixed per-ride anchor CARN already publishes from `streaming_state
["seed_latents"]`. The critic would then be asking *"is this latent's texture
consistent with the texture of the real footage this ride started from"*, which
is closer to the actual acceptance criterion (does it still look like the reality
in the seed context) than *"is this texture real in general"*. It is available at
inference, it is per-ride rather than per-pair, and it costs one extra
conditioning vector.

Flagging as a candidate, not a recommendation — it should not be bundled into the
first Phase-1 arm.

### Ordering

Per the researcher: **textures first.** The transition critic stays parked until
the texture critic works. Point 5's conditioning is part of the Phase-1 texture
critic spec (Point 4), not a separate wave.

### Test

Added to the Phase-1 ladder:

| arm | change |
|---|---|
| H | latent conv PatchGAN, per-frame, blind, low t, patch logits (baseline) |
| K | H + provenance-rung conditioning, distribution-matched across real/fake |
| L | K + `gan_grad_target_norm` cap |

K is only meaningful once the provenance actually varies (prerequisite above), so
that change lands with K, not before.

---

## Point 6 — remove the teacher projection from the texture path

### The claim

If the GAN exists to add information that DMD is missing, then building it on
**essentially the same perceptual system as DMD** is not maximally complementary.

ADD is the relevant precedent: it used an **RGB-space DINOv2 feature
discriminator**, and one stated reason LADD was developed is that ADD had to
decode latent samples to RGB, which is computationally expensive. LADD describes
the trade-off explicitly — ADD's image-space discriminator costs memory, LADD
gains efficiency by staying in latent space. **So LADD moved to latent largely for
efficiency and scalability, not because pixel adversarial information is invalid.**

And the original latent-diffusion / VQGAN autoencoder — the very thing that
*produces* the latent representation we work in — is conventionally trained with a
**decoded pixel-space PatchGAN**, precisely because reconstruction and perceptual
losses alone do not produce the desired fine image statistics. That is a very
strong precedent for this exact situation.

### [CC] Assessment — agreed, and the redundancy is literal, not approximate

I verified this and it is sharper than "essentially the same perceptual system".

The disc backbone is constructed from `_real_score = self.model.real_score`
(`causal_action_forcing_train.py:714`), which is passed to `build_ladd_disc(...)`
(`:800`) and on to `WanFeatureProjector(real_score=real_score)`
(`model/ladd_disc.py:1144-1145`). `self.model.real_score` **is the DMD real
score** — the frozen bidirectional v14e teacher. So the critic's perceptual basis
and the DMD teacher are **the same weights**, not merely the same family.

**The honest caveat.** This is not a proof of redundancy. DMD consumes the
teacher's *score output* — one particular functional of those features — while
the critic reads *intermediate representations* and learns a new head over them,
so in principle it can surface something the score discards. But the margin for
that is narrow given what Points 1 and 2 established: the taps are deep
(≥ block 6, globally attended), the CCM compresses 1536 → 256, and the result is
mean-pooled into a single scalar. The critic is looking for residual information
in a heavily-reduced view of a network whose full function the DMD gradient
already exploits.

**Where the argument is strongest** is the VQGAN precedent, and it deserves more
weight than the ADD/LADD framing. The latent space we operate in was *itself*
produced by a pixel PatchGAN, adopted for exactly our symptom: L1 + perceptual
losses gave insufficient fine image statistics. That is the same failure mode and
the same remedy, one level down the stack.

### [CC] This converges with the Phase-1 spec, which raises confidence

Point 4's Phase-1 design already chose a small conv PatchGAN on raw latents with
no DiT, arrived at from a different direction (tap depth, resolution, the 6×
bottleneck). This point reaches the same architecture from the complementarity
argument. Two independent routes to "drop the teacher from the texture path" is
meaningful support for the choice.

### [CC] Correction — "from scratch" was wrong; the replacement is Wan 14B

An earlier draft of this section proposed a from-scratch conv PatchGAN. That
**violates standing decision 4** (we never train a critic from scratch; we
project onto pretrained networks) and is withdrawn. It is recorded here rather
than deleted because the reasoning that produced it — drop the DMD teacher from
the texture path — is still correct; only the replacement was wrong.

> **STATUS 2026-08-23: SUPERSEDED for the texture critic.** The diagnostic moved
> the texture critic into **pixel** space, where far better pretrained bases
> exist (DINOv2, SD/VQGAN disc). Wan 14B remains the right answer *only* if we
> return to a latent-domain texture critic, and it remains the right answer for
> the parked transition critic. The truncated-prefix insight below (load and run
> only blocks up to the last tap) applies to any future Wan-backed critic and
> should not be lost.

**The correct Phase-1 replacement is Wan 2.1-T2V-14B as the texture backbone.**
It satisfies both constraints simultaneously:

- *pretrained* — standing decision 4;
- *different weights from the DMD teacher* — Point 6's complementarity argument,
  since the DMD teacher is the 1.3B-derived v14e;
- *native to this latent space* — it keeps us in latent for Phase 1.

**Verified on disk:** `/scratch/u6ex/as1748.u6ex/frodobots/Wan2.1-T2V-14B`
(weights symlinked to the shared project copy), `config.json`:

```
dim = 5120,  num_layers = 40,  num_heads = 40,
in_dim = 16, out_dim = 16,     model_type = "t2v"
```

`in_dim/out_dim = 16` and the VAE is symlinked to the 1.3B's `Wan2.1_VAE.pth` —
**the same latent space**, so it is a drop-in backbone for the projector. The
existing per-teacher-size tap defaults already anticipate it
(`causal_action_forcing_train.py:703`: "1.3B has 30 transformer blocks; 14B has
40") and `ladd_disc.py:618` already documents `dim_teacher` 5120 for 14B.

### [CC] The blocker, and the fix that makes 14B affordable

Naively this is expensive. A frozen 14B forward per D-update, at
`gan_updates_per_step` up to 5, on top of the student, the 1.3B teacher and the
fake score already resident — roughly **28 GB of extra bf16 weights** and a full
40-block forward each time. That is the reason to think twice.

**But the texture critic only wants EARLY blocks** (Point 2). And the projector
currently runs the **entire** backbone regardless of tap depth: `__call__` invokes
`self.real_score(**kwargs_local)` and merely hooks the tapped blocks
(`ladd_disc.py:246-262`), so today it pays full depth even though the deepest tap
is block 29 of 30.

So the enabling change is **truncation**: run — and *load* — only the prefix of
the 14B up to the last tapped block.

- Tapping `[0, 2, 4, 8]` needs **9 of 40 blocks ≈ 22 %** of the model.
- Compute per disc forward falls to roughly that fraction, making a 14B texture
  backbone **comparable to or cheaper than the current full-depth 1.3B forward**.
- Crucially the same fraction applies to **memory**: loading only the tapped
  prefix is ~6 GB rather than ~28 GB. That is the difference between feasible and
  not on an 80 GB card with everything else resident.

This is a genuine code change (a `max_block` / early-exit path in
`WanFeatureProjector`, plus prefix-only weight loading), but it is small, it is
independent of the parked transition critic, and it converts the 14B option from
"too expensive" into the cheapest of the candidates. It also compounds with Point
2 rather than competing with it: the deeper we are told texture is *not*, the
cheaper this gets.

**[UNVERIFIED]** — whether the Wan forward can be cleanly truncated without
tripping later-stage assumptions (final norm, unpatchify, output head). If a
sentinel-exception early exit is needed instead of a clean `max_block` argument,
that is still workable but less tidy.

### [CC] MEASURED (WP-14B, 2026-08-23) — the affordability claim holds, with numbers

Built and measured. Full write-up + reproduce commands in `docs/WP_14B.md`
(§6); `testing/probe_wan14b_disc_scaling.py` is the probe. One 95 GiB GPU,
production chunk (F=3, 60×104, 4680 tokens/row), 12 disc rows,
`ladd_gen_guidance_micro_batch_groups=4`:

| backbone | D-update | G-guidance | resident weights |
|---|---|---|---|
| 1.3B, 30 blocks, taps [6,12,18,24,29] — *today* | 1561 ms / 12.1 GiB | 2850 ms / 32.4 GiB | 2.64 GiB |
| 14B, 9-block prefix, taps [0,2,4,8] — *this WP* | 2088 ms / 26.1 GiB | 3525 ms / 34.2 GiB | 6.32 GiB |

So the swap costs **~1.2× the time and ~1.05× the peak memory of the disc we
run today**, for **+3.7 GiB** of resident weights. "Comparable to or cheaper
than the current full-depth 1.3B forward" was the right prediction; the
loaded prefix is 6.32 GiB (3.40 B params bf16), not 28 GB. The real-weights
load-and-forward passes on GPU under flash-attn at the production grid.

Three findings that change what the arm must be launched with:

1. **`ladd_gen_guidance_micro_batch_groups >= 4` is mandatory, not optional.**
   Un-micro-batched G-guidance costs ~8.5 GiB/row and **OOMs at 12 rows**
   (68.4 GiB already at 8). This is NOT a new 14B failure mode — the 1.3B
   path OOMs at 12 rows too, which is why `_fgan_holder.sh:474` and
   `_roll_holder.sh:426` already set 4. Keep it, and do not let a new
   launcher drop it. It is also free: at 8 rows micro-batching was
   *faster* (2337 vs 2500 ms), at 12 rows within noise.
2. **The expensive half is G-guidance, not the D-update.** In train mode the
   disc input carries no grad, so the projector never checkpoints and no
   graph reaches the backbone — the D-update is cheap and near-linear
   (26.1 GiB at 12 rows). All the memory pressure is the gen-side
   input-gradient path's checkpointed teacher recompute. Any future
   coverage increase (more rows, more frames) should be budgeted against
   the G-guidance column, not the D column.
3. **Trim room exists.** Taps `[0, 2, 4]` load 5 blocks (~3.5 GiB) instead
   of 9. `num_layers = max(taps) + 1`, so tap depth — not tap *count* — is
   what costs.

This also resolves the **[UNVERIFIED]** marker above: the Wan forward
canNOT simply run to completion on a truncated prefix, and the reason is not
the one anticipated. `unpatchify` is fine; the **head** is not — `Head.forward`
wants a per-sample `[B, dim]` modulation while the disc passes per-frame
timesteps `[B, F]` (`t_disc`), which makes `e` `[B·F, dim]` and raises a shape
error. No sentinel exception was needed: a clean `max_block` argument in
`wan/modules/model.py::_forward` returns the token tensor after the last
tapped block and skips head + unpatchify entirely (`None` default ⇒ every
existing caller byte-identical). Found by a unit test, not by inspection.

Caveat on the baseline row: it is the same projector code path driven by
stock 1.3B weights, not the live `real_score` wrapper (no action tokens, no
`clean_x`/`aug_t`, no causal-model plumbing). It isolates the cost of the
backbone swap, which is the question here; it is not a replica of the current
arm's absolute cost. Still unmeasured: in-situ memory with generator +
fake_score + real_score + optimizer resident, and whether the independent
critic helps at all (A7 `|cos(g_GAN, g_DMD)|` is the readout, and it is
*expected* to fall).

### [CC] It makes complementarity measurable, which is the real gain

This point converts an architectural preference into a **testable prediction**.

If the critic shares the DMD teacher's basis, its gradient should be strongly
aligned or anti-aligned with the DMD gradient. A genuinely complementary critic
should contribute information DMD does not have — i.e. `cos(g_GAN, g_DMD)` near
**0** (orthogonal), not near ±1.

So the prediction is: **swapping the teacher backbone for a from-scratch conv
critic should drive |cos(g_GAN, g_DMD)| down toward 0.** If it does not, the
complementarity argument is wrong and we should know that early. This is
precisely the telemetry (checklist item A3) that is being built now, and it gives
that instrument a second job beyond diagnosing destructiveness.

### Test

Slots into the Phase-1 ladder from Point 4:

| arm | backbone | prediction |
|---|---|---|
| B (current) | frozen v14e DiT taps, scalar | `|cos(g_GAN, g_DMD)|` large |
| H | Wan **14B** early-block taps (truncated prefix) | `|cos|` → 0 |
| G / Phase 2 | decoded RGB PatchGAN (or DINOv2) | deferred |

Readout: `|cos(g_GAN, g_DMD)|`, the critic's `d_loss` **trajectory**, and the
researcher's judgement of the videos.

---

## Point 7 — nearest-L1 matching is wrong for a texture critic

### The claim

The 4096-entry cross-ride ring is good, but **two of every four real partners still
come from the eight nearest latents to the student's own output**. That selects the
subset of reality most resembling the fake — asking *"which real examples look most
like my current failure mode?"* and then training against those. It reduces the
discriminator's contrast exactly where contrast is wanted.

For the texture critic, reals should be **100 % broad cross-ride sampling**,
perhaps loosely matched on day/night, weather and broad camera domain, but **never
on latent L1 to the fake**. For the temporal/action critic, action matching makes
sense. Splitting the critics makes the data policy much easier.

And the ring's population/sampling **still cannot be verified — telemetry is
absent**. Fix that before trusting any conclusion about cross-ride diversity.

### [CC] Agreed, and the parking decision already resolves most of it

With standing decision 2, the transition critic is parked, and the entire
matching apparatus goes with it: no `match_k`, no top-8 pool, no ring, no
block-diagonal. **A texture critic's reals are simply GT latent frames.** So for
Phase 1 this is resolved by construction rather than by a fix. It becomes live
again the moment the transition critic is unparked, which is why it is recorded
rather than closed.

### [CC] The design principle worth extracting: match nuisances, never the target

"Loosely match day/night/weather" is right, but it is worth stating *why*, because
the same reasoning that motivates it also bounds it.

If fakes are night scenes and reals are day scenes, the critic learns
"bright = real". The generator *can* reduce that loss — by making night look like
day. That is a real failure mode, not a harmless shortcut, so loose domain
matching is protective, not cosmetic.

But tighten the matching and you walk back into the nearest-L1 problem: the reals
converge on the fake and the contrast disappears. The sweet spot is a clean rule:

> **Match on nuisance covariates (exposure, time of day, weather, camera domain).
> Never match on the property being judged (texture).**

Nearest-L1 violates this in the worst way — it matches on a similarity metric
dominated by the very content whose realism is in question.

### [CC] Telemetry — accepted, with a scope change

Agreed that this must not be trusted without instrumentation, and the existing
`[UNVERIFIED]` stands. But note that with the ring parked, the Phase-1 version of
this telemetry is different: what needs logging for a texture critic is
**real-sample diversity per batch** — unique rides, unique source windows,
repeat rate — not ring occupancy. Both are cheap; the point is to log the one
that matches the critic actually running.

---

## Point 8 — equalisation is over-sanitising the texture task

### The claim

Current equalisation deliberately removes **absolute brightness** and
**per-channel DC** before the discriminator. That was understandable when the
critic kept taking cheap brightness shortcuts. But style/texture realism *includes*
local contrast statistics, tone relationships, colour-channel noise and
road-surface luminance distributions.

Do not pair-equalise the texture critic. Use ordinary augmentations applied
**independently and symmetrically to both distributions** — mild exposure jitter,
translation/crop, horizontal flip where valid. That teaches D not to key solely on
exposure *without* forcing fake and real to share first moments. Keep equalisation
if it is useful in the latent dynamics critic.

### [CC] Agreed — and the mechanism is coupling vs broadening

The distinction that makes this correct is worth naming explicitly:

- **Equalisation couples the two distributions** — it forces real and fake to
  shared moments, and the information is *destroyed* before the critic sees it.
  Neither side can ever be judged on it again.
- **Augmentation broadens each distribution independently** — the critic becomes
  *invariant* to the nuisance without the nuisance being removed. Anything the
  augmentation does not span remains discriminable.

For a texture critic the second is strictly better, for exactly the reason given:
tone and local contrast are part of the target, not nuisance.

### [CC] We turned off the augmentations and turned on the equalisation

Worth recording as history, because it looks like the wrong branch was taken at a
fork. The config default is

```
ladd_diff_aug_policy: "flip,cutout,translation"    # configs/action_forcing_phase3_dmd.yaml:98
```

but **every arm we have run narrows it to `flip`** (`sbatch/run_full_carn_probe.sh`,
all modes). So we disabled translation/crop — the augmentation that would have
delivered shift-invariance for a stationary property like texture — and then added
mean/cross equalisation to suppress the shortcut instead. Restoring
`translation` is a config change.

Two refinements on the augmentation list:

- **`cutout` is probably harmful here** and should stay off: it deletes texture
  regions, which is precisely the evidence a texture critic needs. It is a
  regulariser for object-level GANs, not for stationary-statistics ones.
- **"Exposure jitter" in latent space is not benign.** Our latents are not RGB;
  the nearest analogue is shifting per-channel means — which is *exactly* the
  quantity CARN corrects and the drift probe measures as the dominant AR drift.
  Jittering it would make the critic blind to genuine DC drift. This is a real
  tension, not a detail: we want invariance to *legitimate* exposure variation
  and sensitivity to *drift*, and in latent space those look alike. Recommend
  translation/flip in Phase 1 and treating exposure jitter as a Phase-2
  (RGB-domain) tool where the separation is cleaner.

---

## Point 9 — the wavelet experiment isn't implementing the most useful WGSR idea

### The claim

WGSR is highly relevant, but there is a subtle mismatch. **WGSR trains the
discriminator directly on fixed high-frequency wavelet subbands**, coupled with
explicit wavelet-domain generator fidelity losses. We instead do

```
latent HF SWT -> learned 1x1 adapter -> frozen Wan DiT -> teacher features -> GAN
```

That introduces two opportunities for the signal to stop meaning "realistic HF":
**the learned adapter can discover an adversarial encoding**, and **the teacher
representation can reinterpret or discard the frequency structure**. And the
result — horizontal scanline banding — is exactly the sort of evidence that the
system learned a **frequency-domain shortcut**, not genuine detail.

If wavelets are kept at all, put them **on the decoded RGB / luma side** with a
**direct small discriminator over LH / HL / HH**. No learned adapter.

### [CC] Assessment — this is the mechanism behind our worst result, and the code confirms every step

This point was missing from an earlier draft of this document; it is restored here
because it is the most *specific* causal account anyone has offered for the
banding, and the implementation matches it exactly.

**Both leak points are real and verifiable in our code:**

- The adapter exists as described: `model/wavelet_hf.py` decomposes the 16-channel
  latent into LL/LH/HL/HH = 64 channels via Haar SWT, then a **learned 1x1 conv
  adapter re-maps back to 16 channels** so the downstream Wan projector "sees an
  in-distribution shape". It is initialised at gain 0.1 and is **trained jointly
  with the critic** — so it is free to learn whatever encoding best helps the
  discriminator win, which need not be "realistic HF" at all.
- The teacher stage then follows, with the deep taps and the 6x channel
  compression from Point 2, and finally the 46,800-to-1 mean from Point 1.

So the wavelet path stacks **four** successive opportunities for "HF realism" to
stop meaning HF realism: adapter → frozen-teacher reinterpretation → deep-tap
abstraction → global mean. WGSR has **none** of them; it discriminates the
subbands directly.

**Why this is the strongest available explanation of the scanline result.** The
banding appeared *specifically* in the wavelet arms (`wave01`, `horizon_wave90`),
which is what a frequency-domain shortcut predicts and what a generic
"GAN too strong" account does not — the critic was applying only **0.6-1.2 %** of
the DMD gradient while producing a highly structured artefact. A weak critic
producing a *specific, directional* pathology is the signature of a shortcut being
exploited, not of excessive pressure.

**It also composes with the diagnostic.** The A/B/C result showed the corruption
is born in the student *latent* and is invisible to anything that never sees the
render. A latent-domain wavelet critic behind a learned adapter is about as far
from the render as it is possible to get while still nominally being a "texture"
critic.

**Consequence for the current build.** `TEXTURE_GAN_DESIGN.md` already drops the
wavelet stage entirely for the pixel critic, which satisfies this point by
construction. But the recommendation has a *positive* half that is not yet
adopted and should be recorded as a candidate: **if wavelets return, they belong
on the decoded RGB/luma side as a direct LH/HL/HH discriminator with no learned
adapter.** That is cheap — `model/wavelet_hf.py` already contains the `_HAAR_LH`,
`_HAAR_HL` and `_HAAR_HH` kernels; they would simply be applied to decoded luma
instead of to latents, and fed to a small critic directly rather than through an
adapter and a frozen DiT.

---

## Point 10 — the measured 3.6× HF gain changes the objective

### The claim

This is perhaps the most important scientific observation in the whole document.
We do **not** need *"a GAN that creates more high frequencies."* We need **a loss
that distinguishes plausible natural HF structure from fabricated HF structure.**
Those are completely different objectives.

The measurement consequence: **not just a radial spectrum, because horizontal
scanline banding is highly anisotropic and radial averaging can hide it**. The
instruments must track, against GT/reconstructed-real distributions: **log
power in 2D frequency bins**; separate **horizontal / vertical directional
bands**; **Haar LH / HL / HH power**; **spatial kurtosis** of HF coefficients;
**channel covariance** of HF coefficients.

### [CC] This is the sharpest point in the batch, and it invalidates part of our own design

Two consequences, one of which we would otherwise have walked straight into.

**(a) Our designed corrector is blind to our observed failure.** `CARN_V2_DESIGN.md`
candidate (b) — "the aimed shot", the radial-spectrum re-anchor — is explicitly
**radially averaged**: 8 bins over `r ∈ [0, 0.5]`, applied as a *"zero-phase
radial filter"* with *"radially-smooth interpolation"* (`CARN_V2_DESIGN.md:245-252`).
A pure horizontal banding pattern has energy concentrated on the `f_y` axis;
averaging over orientation at fixed radius smears it into bins shared with benign
isotropic detail. **The corrector we designed as the aimed shot at texture drift
could not see the scanline mode at all**, and a radial anchor could even be
satisfied while banding worsens. Candidate (b) needs to become **anisotropic**
before it is built.

**(b) It fixes the measurement, not just the loss — and that failure already cost
us.** My `rollout_realism.py` CLIP probe ranked `horizon_wave90` *above*
`horizon_nogan90`; the researcher's eye ranked them the other way, because the
GAN arm ends in scanline banding. My `rollout_quality.py` uses **isotropic**
Laplacian variance, which cannot distinguish "melted ridges" from "horizontal
banding" either. **Both of my instruments are orientation-blind, which is exactly
why they inverted the verdict.** Adding directional band power (H vs V) and the
Haar LH/HL/HH split to the *metric* is cheap, needs no training run, and would
likely have caught this. That is the first thing to build.

**Implementation note — the Haar statistic is free.** `model/wavelet_hf.py`
already implements single-level Haar SWT with explicit `_HAAR_LH` (HF along
width), `_HAAR_HL` (HF along height) and `_HAAR_HH` kernels, undecimated so
resolution is preserved. The LH/HL/HH power statistic needs no new mathematics —
only a different consumer. The direction-selectivity we need is already sitting
in the file.

---

## Point 11 — the transition framing dilutes the texture signal

### The claim

The critic sees a **1.5-second, six-latent-frame transition** and only the latter
chunk receives generator gradient. That framing was designed for temporal
coherence; forcing texture supervision into the same representation makes little
sense. **AAPT moves the opposite way** for autoregressive discrimination —
emitting a discriminator logit **for every frame** rather than one whole-clip
score, explicitly enabling multi-duration supervision. So *per-frame texture D*
plus *transition/trajectory dynamics D* is much closer to what the problem wants.

### [CC] Agreed; already adopted, and it composes with Points 1 and 5

This is the same decomposition as Point 4 and it is already in the Phase-1 spec
(single-frame texture critic, transition parked). Recording the AAPT precedent is
useful because it is the strongest external support for the per-frame choice.

Two interactions worth making explicit:

- **With Point 1**: per-frame *and* patch logits gives a per-frame, per-patch
  logit map — AAPT's per-frame idea crossed with a PatchGAN. Neither reduction
  (over frames or over space) happens before the nonlinearity. That is the full
  version of the fix, and it is the Phase-1 target.
- **With Point 5**: going per-frame is precisely what *removes* the transition's
  implicit reference, which is why provenance-timestep conditioning is required
  rather than optional. Points 5 and 10 are two halves of one change and should
  land together.

---

## Immediate actionable item arising from this batch

Point 9(b) identifies a defect in the **measurement** that is cheap to fix, needs
no GPU, and already produced one inverted verdict. Before any further arms are
judged, `analysis/rollout_quality.py` should gain:

- directional HF band power (horizontal vs vertical) and their ratio;
- Haar LH / HL / HH power via the existing kernels in `model/wavelet_hf.py`;
- spatial kurtosis of HF coefficients;

and `rollout_realism.py`'s survival score should be reported *alongside* a
banding indicator rather than alone, since CLIP at 224x224 is blind to it.


> **[CC] VERIFIED 2026-08-23 — the flag flip ALONE is not safe.** R1 and R2 are
> normalised *asymmetrically* in the code. In the micro-batched path (the one our
> arms use, `ladd_disc_micro_batch_groups=2`):
>
> ```python
> gsq_terms  = ((d_r_pert_g.sum(1) - d_r_owned.sum(1)) / (_r1_sigma * _r1_tok)).pow(2)  # R1: /(sigma * TOKENS)
> gsq2_terms = ((d_f_pert_g.sum(1) - d_f_g.sum(1))     /  _r2_sigma          ).pow(2)  # R2: /sigma ONLY
> ```
>
> `_r1_tok` is the token count when `ladd_r1_normalize_tokens=true`; **R2 has no
> token normaliser anywhere**. Both sum over the token axis, so R1 would estimate
> `||grad MEAN_i D_i||^2` while R2 still estimates `||grad SUM_i D_i||^2`.
>
> **Why no arm has been bitten yet:** with `ladd_scalar_output=true` the logit
> tensor is `[B, 1]`, so `T = 1` and the two are consistently scaled.
> **The bug is latent and this backlog would trigger it**: setting
> `ladd_scalar_output=false` *and* `ladd_r1_normalize_tokens=true` together — the
> exact pair recommended here — leaves R2 larger than R1 by ~T^2 (T ~ 46,800, so
> ~1e9x) at equal gammas.
>
> **Therefore this is a CODE change, not a config flip:** add the matching
> normaliser to R2 (all three branches: micro-batched, FD, autograd) before
> flipping either flag. Stacks with the A10 *cadence* imbalance — that one is
> about how OFTEN each fires, this one about how BIG each is.
