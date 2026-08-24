# TASK: wire the surrogate consumption path — dark, gated, switch-ready

**For: WP-SURROGATE** (owner — your module, your §4.2–4.4 spec).
**Reviewer: WP-PIXGAN** for the two B1-touching points (§3.1, §3.2 below —
they offered the teacher-side guidance and own those helpers).
**Authorized by: the researcher, 2026-08-24 morning** — "have the surrogate
arm be disabled at the beginning and with the flick of a switch we can
enable it… to not waste time waiting idly." MAIN began this, was redirected
to write this ask instead; everything below is from MAIN's completed
reconnaissance of your module, the trainer, and B1's landed code.

## 1. Objective

Land WP_SURROGATE.md §4.2–4.4 (the parts marked planned-NOT-applied) so
that direct-pixgan-gradients → surrogate-gradients is a config flip.
NOTHING RUNS: `surrogate_critic_enabled` stays default-false; the arm
gates (KV verdict, trained teacher, cadence-asserted smoke) are unchanged.

## 2. What already exists (verified by MAIN)

- Your module is complete for this: `LatentSurrogateDistiller.step(...)`
  (model/latent_texture_critic.py:812 — takes z_real/z_fake crops,
  teacher_value_fn, current_step, optimizer, origins; handles
  refresh-vs-replay and n_teacher_refresh internally),
  `generator_surrogate_loss` (:1116 — frozen-critic idiom, returns
  (loss, logs), expects the UNWRAPPED critic), `two_stage_gen_weight`
  (:1158), `compute_teacher_targets` (:625), `TeacherTargetCache` (:554).
- Your build step is landed (trainer:1328–1380): `latent_texture_critic`,
  `latent_critic_optimizer`, `latent_texture_distiller` attributes +
  `surrogate_critic_enabled` derived from the build.
- Config block landed (13 keys, yaml EOF), byte-identity + RNG tests green.

## 3. Integration points — exact, with anchors

### 3.1 Generator consumption = a branch INSIDE `_compute_pixel_texture_g_loss` (trainer:6254)

This is the key structural decision, and it buys three properties at once:
- **The (weighted, raw, logs) contract is preserved.** `raw` MUST be the
  UNWEIGHTED surrogate loss tensor — the A7 probe plumbing differentiates
  `raw` to publish the weight-1.0 ratio, so the weight-probe protocol works
  for the surrogate unchanged. `weighted = w*raw`, or None when w==0
  (never add 0.0*g — zero-gradient backward for nothing).
- **Both call sites work untouched** (outer ~:13986 and the transition-GAN
  second path ~:14152) — no new call-site plumbing.
- **Alternatives-never-summed is satisfied by the branch**: when
  `surrogate_critic_enabled`, take the surrogate path and return; the
  direct decode/disc path never executes.
Inside the branch:
- Fake latents via `self._pix_select_fake_latents(info)` (:5240–5330) —
  REUSE, do not reimplement: it is mask-aware (selects on
  `finish_denoised_chunk_grad_mask`, fail-loud on absence/all-false) and
  handles both fake sources. This satisfies your own §4.3 masking
  requirement for consumption.
- Weight via `self._pix_gen_weight(current_step)` (:6221) rather than a
  second `two_stage_gen_weight` instantiation — one schedule for both
  alternatives keeps the direct-vs-surrogate comparison clean. If you
  disagree, diverge deliberately and record why.
- Call `generator_surrogate_loss(self.latent_texture_critic, fake_lat, weight=1.0)`
  → that return is `raw`; scale for `weighted`. Pass the UNWRAPPED critic
  (your own docstring's DDP warning).
- Emit your `train/surrogate_g_*` logs; OMIT the pix_g decode-path keys
  (they measure a computation that did not run — forgeable-zero rule).

### 3.2 Distillation step = new `_maybe_run_surrogate_distillation(info, out, *, current_step)`

Called immediately AFTER `_maybe_run_pixel_texture_d_updates` (:6356) at
its call site, same gating style. Body per your §4.2/§4.3:
- Gate: `surrogate_critic_enabled` and `pixel_texture_disc` present and
  `current_step >= gan_disc_start_step`.
- Fake crops: `_pix_select_fake_latents(info)` then DETACH, then
  `_pix_take_crops_with_origins` (:5022) — mask selection happens BEFORE
  the crop draw (your zero-gradient trap: a crop from a detached frame
  teaches "no gradient here", and under normalization the all-zero target
  drives the denominator to its 1e-12 floor — numerically loud, wrong).
- Real crops: same source the pixel D-loop uses (`_pix_real_pool`,
  :5452/:5628 — read `_maybe_run_pixel_texture_d_updates` for the exact
  draw; reuse its pattern, not a new supply).
- Teacher closure: your §4.2 closure verbatim — `self._vae_decode_grad`
  (graph-on; NEVER the no_grad twin — B1's constraint), border trim from
  `self._pix_resolve_cfg()`, `self.pixel_texture_disc`, reshape to
  [N,F,1,h,w]; `torch.cuda.empty_cache()` before the decode (the step-2
  allocator failure is measured-real, trainer:10833).
- `self.latent_texture_distiller.step(z_real=…, z_fake=…, teacher_value_fn=…,
  current_step=…, optimizer=self.latent_critic_optimizer,
  origin_real=…, origin_fake=…)`; merge the returned logs into `out`.

### 3.3 ORDERING DEVIATION — your explicit call needed

Your §4.3 demands D-update → distill → consume within one step. The
trainer computes the G-term EARLIER in the step than the D-loop
(B1's own docstring at :6276–6279 states this; B1's direct path handles it
by snapshotting the pre-update disc, `_pix_g_snapshot_disc` :6173). Under
the branch design of §3.1, the generator consumes the critic distilled at
the END of the PREVIOUS step — one step stale, symmetric with B1's
snapshot semantics, so the two alternatives stay comparable. Options:
(a) accept + log the staleness explicitly (MAIN's recommendation — least
invasive, comparable); (b) restructure the step order (invasive; touches
B1's structure). Decide as spec owner, record the decision in your doc.

### 3.4 Save/resume — FAIL-LOUD per your §4.4

- Save: parent block in trainer/causal_rolling_staircase_train.py (~:1623
  region — the explicit-key dict). Add `latent_texture_critic` +
  `latent_critic_optimizer` keys, gated on the attributes existing (NOT on
  `gan_enabled` — that's the B1(d) trap).
- Resume: trainer/causal_action_forcing_train.py ~:2834–2871. RAISE (not
  warn) when `surrogate_critic_enabled` is true and the keys are absent —
  your own words: a silently re-initialized surrogate hands the generator
  a zero gradient that reads as "GAN term present and quiet".

## 4. Standards (all already standing; listed so nothing is re-derived)

- Echoes: print/WARNING or the pipeline's dual pattern. NOTE: MAIN
  root-caused and FIXED the INFO suppression (basicConfig no-op under
  pre-existing root handlers; explicit `setLevel` added in
  causal_rolling_staircase_train.py — landed, compile-verified, mechanism
  proven). Belt-and-braces still preferred for load-bearing lines.
- Resolved-value echoes read LIVE objects (optimizer param groups, live
  distiller attrs), never cfg re-reads; omit-and-flag on disagreement.
- Tests: cadence assert via `n_teacher_refresh` == ceil(steps/N) (never
  raw teacher calls — your own 2×-under-checkpoint finding); byte-identical
  -off via your RNG-state pattern incl. the has-teeth mutation companion;
  planted companions use fixed constants; config keys only through
  `build_from_config`'s single read site (extend it if new keys appear —
  no second read site).
- Grid protocol: announce lock take in the INBOX (flock), release
  explicitly. MAIN's partial lock-take for this task is RETRACTED — the
  file is free for you.

## 5. Explicitly out of scope

Running anything; changing arm gates; touching B1's direct path beyond the
branch; weight/γ values (measurement-only, per protocol); the KV
instrument; DDP wrapping of the critic (build step already decided this).
