# WP-PIXGAN (B1) — pixel-space texture PatchGAN + the A23 grad path

Progress log for work package **B1**. Spec = `docs/TEXTURE_GAN_DESIGN.md`
(complete; `pix_*` names mandatory) governed by `docs/GAN_REDESIGN.md`
section B item 1 and directives A19–A24. **Only this doc is written by
WP-PIXGAN**; `GAN_REDESIGN.md` is the main agent's.

Status legend: `[ ]` not started · `[~]` in flight · `[x]` landed + tested.

---

## 0. Build decomposition (main agent, 2026-08-23)

The package was split into three tracks with **disjoint file ownership** so
they can land independently. Tracks 1 and 2 have no shared files and run in
parallel; track 3 is gated on both, and additionally on WP-14B's trainer
build branch (`trainer/causal_action_forcing_train.py:694-937`) landing
first per the section-B LANDING ORDER.

| track | scope | files owned | gating |
|---|---|---|---|
| **T1 — A23 grad path** | launch gate: inference-parity fake | `pipeline/action_forcing_training.py`; `model/dmd_action_forcing.py` **:9513-9516 only**; `testing/test_a23_finish_grad.py` | none |
| **T2 — the critic** | §4 arch, §5.1 loss, §5.3 R1, §5.4 counters, §8.1 C1 control | new `model/pixel_texture_disc.py`; new `testing/test_pixel_texture_disc.py` | none |
| **T3 — wiring** | trainer attrs/optimizer/gate/save-resume, real+fake supply, §7 telemetry, config block | `trainer/causal_action_forcing_train.py`; `trainer/causal_rolling_staircase_train.py` **save block :1623-1637 only**; `configs/action_forcing_phase3_dmd.yaml` **appended block only** | T1 + T2 + **WP-14B** |

**Not touched by any track** (other packages own them):
`model/ladd_disc.py`, `wan/modules/model.py`, `model/latent_texture_critic.py`,
`model/disc_holdout_probe.py`, `analysis/texture_stats.py`,
`analysis/compare_commit_tensors.py`.

## 1. Frozen interface contracts this package must honour

1. `model/pixel_texture_disc.py::PixelTextureDisc`,
   `forward(px [N,3,H,W] in [-1,1]) -> [N,1,h,w]` patch-logit map; trainer
   attribute named exactly **`pixel_texture_disc`** — with that name
   `model/disc_holdout_probe.py:78-100,207` needs zero edits, and its
   `_reduce_scores` already mean-reduces a `[N,1,h,w]` map, matching §5.3.
2. `info["finish_denoised_chunk_grad"]` (graph-on, flash-off path). The
   existing detached `finish_denoised_chunk` key is **unchanged**.
3. Optimizer/attr names: `pix_optimizer` (Adam, betas (0.0, 0.9),
   lr=`pix_gan_lr`), gate `gan_pixel_texture_enabled` — **not** a new
   `gan_backbone` value.

## 2. Spec decisions already fixed (do not re-open)

- **R1 only. R2 does not exist** — no `pix_r2_*` key of any kind (§5.3).
- **R1 differentiates the MEAN patch score**, single finite-difference
  estimator, `pix_r1_sigma=0.01` on the `[-1,1]` pixel scale, `γ=1.0`.
  The mean reduction is **not a flag**. LADD penalty code is **not** reused
  (four divergent estimators live there — A16/A17).
- **`pix_r1_every_n=1`, target `pix_r1_rate = 1.00`**; bound cost by
  subsampling (`pix_r1_num_samples`), never by skipping.
- **Patchwise NS-logistic / hinge, non-relativistic** (§5.1). No RpGAN.
- **Band matching stays COARSE** — thirds, `pix_band_count=3` (A24).
  Exact-y matching and every approach toward it is forbidden.
- **A20**: one *independently sourced* real per fake per D update
  (12-and-12 at the §3.7 spec). Frame-within-crop expansion is a
  **fake-side** device only.
- **A21**: total real support is a *measured* claim off
  `pix_real_support_frames`; floor ≥4,096 distinct source frames.
- **`pix_gan_weight` has no default** — 0.03 is withdrawn (§5.5); it is
  calibrated from `gan_dmd_grad_ratio` into the 5–20 % band.
- Scope freeze (§ "B2 SCOPE FREEZE") holds: no ADA, no timestep
  conditioning, no pretrained pixel backbone, no multi-scale/temporal, no
  wavelet, no retrieval-matched reals, no action conditioning, no
  transition pairing.

## 3. Known traps carried into the build

- **P = 660, not the doc's 768.** 24×32 latent → 192×256 px → 8-px border
  trim → 176×240 → /8 → 22×30 = 660. Mean reduction makes it harmless;
  compute P from the tensor shape and log it.
- The parent save block in `trainer/causal_rolling_staircase_train.py:1623-1637`
  is gated on `gan_enabled` — **silent critic re-init on resume is the
  known trap**. Pixel-critic save/resume keys must FAIL LOUD.
- DDP `find_unused_parameters=False` vs the R1-subsample branch.
- 60 `no_grad` decodes/step at `pix_gan_updates_per_step=5` puts the loader
  on the critical path (§6) — measure step time in the smoke.
- Unknown dotlist keys merge **silently** (OmegaConf); the trainer's
  `_OVERRIDE_GUARD_PREFIXES` already covers `pix_` (grep the symbol).

## 3a. Stale-doc corrections found while reading (2026-08-23)

- `GAN_REDESIGN.md` **A22 "REOPENED"** is stale relative to the code: the
  fail-closed hardening has landed. `model/disc_holdout_probe.py` now
  carries `INVALID_NO_HOLDOUT_ROOT` / `INVALID_NO_HOLDOUT_RIDES` /
  `INVALID_NO_TRAIN_RIDES` / `INVALID_POOL_MISMATCH` / `INVALID_GEOMETRY` /
  `INVALID_POOL_EMPTY` and `VERDICT_SIGN_FLIPPED` / `VERDICT_STUDENT_HARD` /
  `VERDICT_D_UNDERPOWERED`. Do not design B1 around the old failure modes.
  (Flagged to the main agent — `GAN_REDESIGN.md` is not WP-PIXGAN's to edit.)
- `TEXTURE_GAN_DESIGN.md` §4 states **768** patch logits per image; the
  post-trim geometry gives **660**. See §3 above.

## 4. Track status

- [x] **T1 — A23 grad path. LANDED, 20/20 tests pass.**
      Gate `pix_finish_grad_enabled` (default false, byte-identical off).
      Grad-on FINAL finish rung under `_ckpt(..., use_reentrant=False)` with
      a detached input (one-rung bound is structural, not incidental);
      `cache_pred` immediately re-detached so the flash pass, `output`,
      `clean_chunk` and the §3.3 K/V commit all see the pre-A23 tensor.
      Publishes `finish_denoised_chunk_grad` **and**
      `finish_denoised_chunk_grad_mask`. `finish_denoised_chunk` and its
      `.detach()` unchanged (frozen contract 2).
      `model/dmd_action_forcing.py` diff is **purely additive**.
- [x] **T2 — `PixelTextureDisc` + tests. LANDED, green.**
      `79 passed, 32 subtests passed`. Measured: **661,953 params**,
      forward `[2,3,176,240] -> [2,1,22,30]`, **P = 660**. API in §7.
      Extended 2026-08-23 so the §8.1 control harness is reusable by LATENT
      critics — see §18.
- **T3 — trainer wiring. SPLIT into three chunks** after three consecutive
  session restarts killed the single large task before it wrote anything.
  Each chunk is now sized to finish inside the restart cadence and is
  instructed to write incrementally rather than batching edits to the end.
  - [x] **T3-A — LANDED, green.** 37/37 own suite; 100 passed + 20 subtests
        in the two neighbouring suites; byte-identical off **proven
        including RNG state**. Gate + `pixel_texture_disc` +
        `pixel_texture_disc_ddp` + `pix_optimizer` (Adam, betas (0.0,0.9),
        lr `pix_gan_lr`); `[mem-inventory]` registration; **fail-loud
        save/resume** under its own gate with `_maybe_resume`'s early-return
        extended to admit the pixel case; explicit `pix_*` override-guard
        registration (see §11). Checkpoint keys `pixel_texture_disc` /
        `pix_optimizer`, asserted equal on both sides by test.
  - [~] **T3-B** — real supply (A20 1:1 independent source frames, A21
        support floor, A24 coarse bands), the grad-on crop helper returning
        `(crops, ys, xs, bands)`, mask-selected fake, the D-loop. In flight.
  - [ ] **T3-C** — G-term into `gen_gan_loss`, full §7 telemetry, the
        `pix_*` config block appended at EOF after WP-14B's block.
        **Extra requirement (committed to the main session):** the A7
        telemetry block must also report an **UNWEIGHTED**
        `pix_gan_grad_ratio` — see §12.

### Coordination log

- **2026-08-23** — WP-14B confirmed it has **not** yet edited
  `trainer/causal_action_forcing_train.py`; its hunk lands inside the
  `if self.gan_enabled:` LADD build branch at :694-937 (three small edits:
  a provenance line after `ladd_blocks = list(...)`, an optional 14B-prefix
  block before the `build_ladd_disc(` call, and a `backbone=` kwarg). It
  will shift **everything below :937 by ~40-60 lines**. WP-PIXGAN is
  therefore off that file entirely until WP-14B signals, and T3 must anchor
  on greppable **symbols**, never on line numbers computed before that
  edit: `_compute_r3gan_losses`, `gen_gan_loss`, `register_fake_source`,
  `_vae_decode_grad`, `_vae_decode_nograd`,
  `_sample_critic_grad_frame_indices`, `_OVERRIDE_GUARD_PREFIXES`.
- Config append order on `configs/action_forcing_phase3_dmd.yaml` is being
  sequenced with WP-14B so the two end-of-file blocks cannot collide.
- **2026-08-23** — WP-SURROGATE (B3) confirmed it touches **none** of
  WP-PIXGAN's files, does **not** import `model/pixel_texture_disc.py`, and
  stays off `configs/action_forcing_phase3_dmd.yaml` until B1's block has
  landed. Its teacher is an injected callable
  `teacher_value_fn(z_crop [N,F,C,h,w]) -> patch logits`, implemented at the
  trainer call site as `pixel_texture_disc(decode_grad(z_crop))`, reduced to
  the per-sample MEAN patch logit — contract 1 is satisfied unchanged.
  **Three commitments B1 owes B3** (all folded into T3's brief):
  1. The fake-side decode is **graph-on back to the latent crop** —
     `_vae_decode_grad`, never the probe's `no_grad` `_decode_crops` (that
     helper is REAL-side only, where §5.2 mandates `no_grad`). The grad-on
     variant is a **separately named helper**, not a `no_grad` function with
     a flag, so no caller can land on the wrong one by default.
     Caveat passed on: `_vae_decode_grad` checkpoints the VAE forward
     (`use_reentrant=False`), so `autograd.grad` triggers a decoder
     recompute; `use_checkpoint=False` exists if double-backward is ever
     needed, at ~5-10 GB more activation.
  2. **Crop origin is surfaced.** `model/disc_holdout_probe.py` is not
     WP-PIXGAN's to edit and `take_crops` draws `x` internally and discards
     it, so B1's own crop helper draws **both** offsets explicitly and
     returns `(crops, ys, xs, bands)` — `ys` still from `band_plan`, so the
     A24 coarse-band constraint and the real/fake band pairing are
     unchanged.
  3. The **exact built API** is reported into this doc (B3 reads it here
     rather than pinging).
- **A7 telemetry sharing**: the pixel G-term is added into `gen_gan_loss`,
  so `train/gan_grad_norm` / `gan_dmd_grad_ratio` / `gan_dmd_grad_cos`
  measure LADD + pixel whenever both are on. In the `ganfix_pixtex` arm the
  transition GAN is off (standing decision 2), so they never overlap in
  practice; a `pix_`-namespaced readout is the fallback if WP-14B needs
  those keys pure.

## 5. Acceptance

- **Launch gate (A23(a))**: `analysis/compare_commit_tensors.py` re-run
  gives `d_vote <= d_null` in **every** depth stratum.
- Flag-gated, default-off, **byte-identical off**, with tests, throughout.


---

## 6. Findings that change how the arm must be read

### 6.0 A23 IS OVERTURNED — path (a) is no longer a launch gate

**Reported by the main GAN-redesign session 2026-08-23 and independently
confirmed here.** The A23 "FAILED late, d=0.109" verdict was **single-seed**.
On the same checkpoint and ride, seeds 42/43/44 give late
`d = .1093 / .0113 / .0128`, and the >=3-seed protocol returns **EQUIVALENT in
every stratum**. The proposed mechanism ("the extra t=60 forward partially
repairs the banding") was also refuted — a bare extra forward AMPLIFIES it.

**Corroboration found here, which the report did not cite:**
`analysis/compare_commit_tensors.py:118` sets
`VOTERS = ("hv_anisotropy", "haar_HL_LH_ratio")`, so the d=0.109 verdict was
computed on `hv_anisotropy` — **one of the exact statistics the campaign's own
noise-floor protocol already flags as noise-dominated** (C_late anisotropy /
angular entropy carry sd up to ±0.34; only HF power and Laplacian kurtosis
separate at ~3σ; never rank close arms on them single-seed). A23's original
verdict was a single-seed read on a voter already documented as unsafe to read
single-seed, and the .1093/.0113/.0128 spread is what that warning predicts.

**Consequence for B1: nothing is wasted, and nothing is blocked.** The A23
grad path is built, tested 20/20, and **flag-gated default-off**
(`pix_finish_grad_enabled`). The fake source is now an **experimental
variable**, not a prerequisite:
- `pix_finish_grad_enabled=false` -> fake is `flash_dmd_gan_x0` (cheaper: no
  retained rung graph, no frame-dilution mask to select on);
- `pix_finish_grad_enabled=true` -> fake is the inference-parity ladder
  endpoint.

**Related seeding trap (verified in code):** `utils/eval_causal_AR.py:973-979`
builds `_fr_gen` ONLY inside `if _fr_dir:`, so without `ODE_FLOW_REC` set it
stays `None` and `torch.randn(..., generator=None)` draws from the global RNG.
**Varying `ODE_FLOW_SEED` alone is a guaranteed no-op** — bit-identical output
is the expected result, not evidence of determinism. Seed-average with
`--seed` (`compare_commit_tensors.py --seed`, default 42).

`GAN_REDESIGN.md`'s A23 entry still reads "FAILED late ... **B1 must take path
(a)**" and still lists A23(a) as STILL-OPEN item 1. That text is stale; the
main session has been asked to correct it (that file is not WP-PIXGAN's).

### 6.1 The A23 acceptance criterion is a TAUTOLOGY under path (a)

`GAN_REDESIGN.md` states B1's acceptance as *"re-run
`analysis/compare_commit_tensors.py`; `d_vote <= d_null` in every depth
stratum"*. That criterion was written for path **(b)** — the equivalence
probe, where flash and ladder are genuinely different tensors. Under path
**(a)**, which A23 forces, the critic's fake **is** the ladder endpoint, so
both arms of the comparison are the same object and `d_voters` and `d_null`
both collapse to exactly `0.0`. Every stratum passes by construction.

**The run is therefore a harness self-check, not evidence about the model.**
Taken together with §6.0, the honest position is that **A23 never justified a
launch block at all**: the original failure was a single-seed artefact on a
known-noisy voter, and the acceptance test for the remedy is vacuous.
It should only ever be reported paired with the unmodified-default control
run (which must still reproduce the late `d ~ 0.109` vs `d_null ~ 0.016`), so
that one report shows *switching the fake source* is what closes the gap.

**The substantive A23 acceptance is provenance, and it is locked by test:**
`finish_denoised_chunk_grad` is the output of the forward at
`denoising_step_list[-1]` under the same KV context inference commits, and
with `flash_dmd_enabled=false` it is **bit-identical** to
`finish_denoised_chunk`
(`test_flash_off_grad_buffer_matches_clean_chunk_values`,
`test_flash_on_grad_buffer_is_ladder_endpoint_not_flash`).

A narrow ownership exception was granted to add a default-`flash`
`--fake_source {flash,ladder}` flag to `compare_commit_tensors.py` — without
it the stated criterion is not runnable at all. Existing invocations are
byte-identical (`sbatch/run_commit_tensor_probe.sh` passes no `--fake_source`);
`--help` exits 0; the file parses. `ladder` mode prints the caveat in **four**
places — startup banner, the `gate_A23` verdict string, the terminal header,
and a `fake_source_note` field in `summary.json` — because the unguarded
verdict string would otherwise have read *"path (b) SATISFIED — B2 may use the
flash tensor as its fake"*, which is the exact opposite of the truth.

> **WARNING — `analysis/` is GITIGNORED** (`.gitignore:98`). This change is
> **not under version control** and will not survive a clean checkout. The
> same is true of `compare_commit_tensors.py` itself. If the `--fake_source`
> flag matters beyond this campaign it must be moved somewhere tracked.

**Ready-to-paste acceptance pair** (both need a COMPUTE node — the script
SIGKILLs on the login node; env preamble mirrors
`sbatch/run_commit_tensor_probe.sh`). Use the **same `--student_ckpt` for
both** or the pair means nothing:

```bash
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate && conda activate arrwm
cd /scratch/u6ex/as1748.u6ex/ARRWM
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache
export HF_HUB_CACHE=$HF_HOME HUGGINGFACE_HUB_CACHE=$HF_HOME TRANSFORMERS_CACHE=$HF_HOME
export TMPDIR=/tmp OMP_NUM_THREADS=8 WORLD_SIZE=1 LOCAL_RANK=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# CONTROL (path b, default — byte-identical to what has always run)
python analysis/compare_commit_tensors.py \
  --student_ckpt logs/<B1_RUN>/eval_step0200.pt \
  --fake_source flash --flash_t 60 --ar_gen_chunks 100 \
  --out analysis/commit_tensor_probe/a23_control

# ACCEPTANCE (path a — the self-check)
python analysis/compare_commit_tensors.py \
  --student_ckpt logs/<B1_RUN>/eval_step0200.pt \
  --fake_source ladder --ar_gen_chunks 100 \
  --out analysis/commit_tensor_probe/a23_patha
```

Expected: control reproduces late `d ~ 0.109` vs `d_null ~ 0.016`
(NOT-EQUIVALENT); acceptance prints `d_voters = d_null = 0.0000`,
`gate = 0.1000`, EQUIVALENT in every stratum, `underpowered=false`.
Two readings that make the tautological run *not quite* worthless:
`d_adj` (chunk *c* vs chunk *c-1* ladder) stays **non-zero**, proving the
instrument is alive rather than returning zeros because it broke; and an
**exact** `0.0000` confirms the decode path is deterministic, since ladder
mode decodes the identical tensor three times.

**NOT RUN** — no GPU on this node, and jobs are not submitted without the
user's explicit instruction.

### 6.2 Frame-level grad dilution — measured, and now visible

The published grad buffer is **partially detached** by design, and before T1's
mask it was **silently** so (buffer-level `requires_grad` reads `True` while
individual frames carry no graph). A critic averaging its G-loss over the whole
chunk would dilute the adversarial signal with dead frames and nothing would
say so. Two sources:

| source | rate | why |
|---|---|---|
| trailing block of MULTI-block calls | **14.3 %** measured (live 18/21 on a 7-block rollout) | mirrors `flash_grad_active`; the documented "6 chunks instead of 7" memory intent for the heavy iter-1 rollout |
| no post-exit rung to attach (block exits at the last rung) | **~1/K only under a UNIFORM exit draw — in the campaign's actual configs it is EFFECTIVELY ALWAYS. See §34.** | `range(exit_index+1, K)` is empty |

**Consumers MUST select on `finish_denoised_chunk_grad_mask`.** Counted as
`pix_finish_grad_frames` / `pix_finish_grad_frames_total`.

**Decision (main agent):** the no-rung fallback stays **detached**. Reusing the
exit rung's already-grad-carrying output would recover those frames but shares
a graph with the DMD path, making a second `.backward()` on the pixel term a
double-backward error. Counted + masked + logged is the correct end state. The
`generate_and_sync_list(exclude_last_rung=...)` call site governs DMD's
exit-rung distribution and is **not** WP-PIXGAN's to change.

### 6.3 The critic is 0.66 M params, not the spec's "2-3 M"

`TEXTURE_GAN_DESIGN.md` §4 claims ~2–3 M params, but its own layer table gives
**661,953**: conv1 3,136 + conv2 131,200 + GN 256 + conv3 524,544 + GN 512 +
conv4 2,305. The table is exact and is what is implemented; the parameter
claim is arithmetically wrong. The build pins the **measured** number.
This does not change §4.1's from-scratch argument (a smaller hypothesis class
strengthens it), but it does mean the "~180 D-updates to learn texture from
nothing" budget risk should be re-read against 0.66 M, not 2–3 M.

### 6.4 `TEXTURE_GAN_DESIGN.md` §5.3's "zero references repo-wide" was wrong

§5.3 asserts `ladd_r2*` / `r3gan_r2*` / `_ladd_r2_fires` have zero references
repo-wide. In fact `model/dmd_action_forcing.py:1936-1974` still assigned four
`ladd_r2_*` attributes (dead stores — nothing reads them),
`configs/action_forcing_phase3_dmd.yaml:94-97` still sets the keys, and
`sbatch/_fgan_holder*.sh` still pass them. A sub-agent deleted the dead stores
as "hygiene"; that was reverted as out-of-scope for B1. **This is LADD-side
cleanup for whoever unparks that path, not WP-PIXGAN's.**

### 6.5 Operational: CPU tests need thread limits on this node

`nproc = 144`. Torch grabs all 144 threads and thrashes on these small CPU
convs: the pixel-critic suite takes **>30 minutes and looks hung**. With
`OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8` the *same* suite
runs in **21 seconds**. The default `python` also has no torch — use
`/scratch/u6ex/as1748.u6ex/miniforge3/envs/arrwm/bin/python`.

### 6.6 Untested in production (T1)

No GPU, no DDP, no real `WanModel` was exercised — the A23 suite runs a dummy
generator on CPU. Specifically unverified: DDP bucket behaviour with the new
grad-on phase-LoRA dispatch site inside `_finish_fn` (it mirrors `_exit_fn`, so
it should be stationary, but it is a new grad-on site); and real `_ckpt`
recompute against a live KV cache. The RNG guarantee, by contrast, **was**
checked against this torch's `utils/checkpoint.py` rather than assumed: the
forward only reads RNG state, and recompute runs inside `fork_rng`, so the
global stream is untouched even with the gate ON.


---

## 7. Built API — `model/pixel_texture_disc.py` (STABLE, code against this)

Published for WP-SURROGATE (B3) and the trainer wiring, per the coordination
log commitment. Everything takes an explicit `generator` where it draws, so
every consumer can make its draws DDP-deterministic.

```python
class PixelTextureDisc(in_channels=3, base_channels=64, groups=8,
                       negative_slope=0.2, *, use_spectral_norm=True,
                       use_norm=True)
    # forward(px [N,3,H,W] in [-1,1]) -> [N,1,H/8,W/8]   (frozen contract 1)

# §5.1 losses — patchwise, nonlinearity BEFORE any averaging, non-relativistic
d_loss(real_logits, fake_logits, loss_form="nsgan") -> Dict[str, Tensor]
g_loss(fake_logits, loss_form="nsgan") -> Tensor

# §5.3 R1 — ONE finite-difference estimator on the per-image MEAN patch score
r1_penalty(disc, real_px, *, gamma=1.0, sigma=0.01, num_samples=None,
           generator=None, indices=None, eps=None) -> Dict[str, Any]
r1_subsample_indices(n, num_samples, generator=None) -> Optional[Tensor]
#   `indices` / `eps` are injectable so the caller can broadcast the draw
#   from rank 0 rather than trusting per-rank RNG.

# §5.4 cadence counters — MONOTONE, not gauges
class PixCounters(r1_every_n=1)

# §7 telemetry
patch_logit_telemetry(logits, prefix="pix_") -> Dict[str, float]
patch_logit_spatial_variance(logits) -> float
patch_logit_separation(pos_logits, neg_logits, *, n_boot=1000, ci=0.95,
                       generator=None) -> Dict[str, float]
roc_auc_fast(pos, neg) -> float

# §8.1 positive control (C1 only; C2 deliberately not implemented — mp4
# round-trip contaminates the very HF being measured)
c1_structured_hf(x, amplitude, *, generator=None, hf_cut=0.55,
                 wedge_half_width=pi/10, orientation="column",
                 scale_ref=None) -> Tensor          # perturbs the LATENT
sweep_c1_amplitude(x, amplitudes, *, seed=0, decode_fn=None, hf_cut=0.25,
                   targets=None, target_frac=0.5, c1_kwargs=None)
c1_calibration_verdict(clean, corrupted, *, targets=None,
                       terms=("hv_anisotropy", "angular_entropy"))
positive_control_readout(disc, gt_px, corrupted_px, student_px=None, *,
                         n_boot=1000, ci=0.95, generator=None,
                         prefix="pix_poscontrol_") -> Dict[str, float]

# §5.5 — pix_gan_weight has NO default; absence is loud by design
resolve_gan_weight(value, *, strict=True) -> float

measure_receptive_field(input_size=129, ...) -> Dict[str, int]
```

**Not present, deliberately and permanently:** any `pix_r2_*` symbol, any
`pix_r1_normalize` knob, any un-normalised or autograd or per-patch R1
variant, any relativistic/RpGAN loss, any global-scalar output option.
A spec-drift tripwire test asserts this against the module source with
comments and docstrings stripped (so the module may still *document* what it
refuses to do), and a companion test proves the tripwire fires on a planted
violation.

### Note for WP-SURROGATE

`c1_structured_hf` perturbs the **latent**, before decode — matching §8.1.
The teacher-side reduction B3 needs (mean patch logit) is what
`disc_holdout_probe._reduce_scores` already does and what `r1_penalty`
uses internally, so the two are consistent by construction. The crop-helper
signature `(crops, ys, xs, bands)` is owned by T3 and will be appended here
when it lands.


---

## 8. A18-D3 — the override guard already reserves `pix_`

A registration hook landed in the trainer on 2026-08-23 (grep `A18-D3` and
`_OVERRIDE_GUARD_REGISTRY_NAMES`) and it explicitly anticipates this package:
`pix_` is a **guarded prefix**, and the comment names
`model/pixel_texture_disc.py`'s `PIX_DEFAULTS` table as the motivating case.

Why it matters to B1: unknown OmegaConf dotlist keys merge **silently**, and a
module that reads its knobs through a *variable* key has no literal
`getattr(config, "<key>", ...)` site for the textual scan to find. Before the
hook, every such knob was reported as "merged silently, no effect" — and under
`strict_override_keys=true` that FALSE POSITIVE **kills the job before the
trainer is constructed**. With ~15 `pix_*` knobs specced, the blast radius was
the entire arm.

The hook claims to auto-harvest `DEFAULTS`-style tables. **T3-A is instructed
to verify that against `model/pixel_texture_disc.py` by actually running the
scan, not to trust it**, and to register the keys explicitly under one of the
accepted registry names if the harvest misses them.


---

## 9. §6 memory-budget correction — decode scales with CROPS ONLY

`TEXTURE_GAN_DESIGN.md` §6 says *"the grad-decode cost scales linearly with
`crops × frames`"* and tells the reader to **"re-measure at `crops=8,
frames=3` before committing to the upper end"**. Both the multiplier and the
tuning advice are wrong.

Verified in `model/disc_holdout_probe.py::_decode_crops`: it calls
`_vae_decode_nograd(sub)` on the **whole latent crop**, and only *then* draws
`sel = randperm(f_pix)[:k]`. **Frame selection happens strictly AFTER decode.**

Therefore:
- decode cost (the expensive part, both `no_grad` real-side and grad-on
  fake-side) scales with **`pix_crops_per_step` only**;
- `pix_frames_per_crop` costs **nothing** at decode and scales only the critic
  forward — ~0.66 M params on 176×240 crops, i.e. negligible.

**Practical consequence, which is the opposite of what §6 implies:** raising
`pix_frames_per_crop` is nearly free, and it is the cheap way to buy the §3.7
"effective sample count". `pix_crops_per_step` is the knob that actually costs.
Note this does **not** relax the A20 real-side rule — frame-within-crop
expansion remains a **fake-side** device, because three frames from one
decoded crop are not three independent draws of the real distribution.

(Reported to the main GAN-redesign session for the spec; `TEXTURE_GAN_DESIGN.md`
is not WP-PIXGAN's to edit.)


---

## 10. GPU verification — the recorded 14:26 failures are STALE

`GAN_REDESIGN.md`'s STILL-OPEN item 2 records test failures found on holder
GPUs at 2026-08-23 14:26 and calls green suites *"a merge precondition for
T3"*: `test_pixel_texture_disc.py` 4F/14, `test_a23_finish_grad.py` 1F/5,
`test_latent_texture_critic.py` 1F/40.

**Those counts are against older files.** The suites now hold 71 and 20 tests
respectively (not 14 and 5), and every file involved was modified AFTER that
run (`model/pixel_texture_disc.py` 15:00, `testing/test_pixel_texture_disc.py`
15:03, `testing/test_a23_finish_grad.py` 14:43).

**Re-run on a real GPU** (`srun --overlap --jobid=6108075 --gpus-per-task=1`,
same arrwm env, thread limits applied):

```
testing/test_pixel_texture_disc.py + testing/test_a23_finish_grad.py
    91 passed, 20 subtests passed in 5.06s
testing/test_latent_texture_critic.py        (WP-SURROGATE's, run not edited)
    44 passed, 1 warning in 22.17s
```

All three suites are green on GPU. The merge precondition for T3 is met; item
2 can be closed. Reported to the main session — that ledger is not
WP-PIXGAN's to edit.

**Method note worth keeping:** these were run on an EXISTING holder via
`srun --overlap`, not by submitting a job. Note also the CPU-vs-GPU timing:
the same combined suite takes 5.06s on a GPU and ~50s on CPU with thread
limits — and over 30 minutes on CPU *without* them (nproc=144, see §6.5).


---

## 11. A18-D3 does NOT harvest `PIX_DEFAULTS` — verified, and it would have
## killed the arm

T3-A reported, and **I confirmed independently** by running the trainer's own
`_ast_config_sourced_keys` against `model/pixel_texture_disc.py`:

```
harvested from model/pixel_texture_disc.py : EMPTY SET
after explicit registration, sourced from the trainer (15):
  gan_pixel_texture_enabled, pix_band_count, pix_crop_lat,
  pix_crops_per_step, pix_decode_border_trim, pix_frames_per_crop,
  pix_gan_betas, pix_gan_lr, pix_gan_weight, pix_loss_form,
  pix_r1_every_n, pix_r1_gamma, pix_r1_num_samples, pix_r1_sigma,
  pix_reals_per_fake
```

**Root cause:** the `DEFAULTS`-table harvest is gated on the module reading
config through a *variable key*. `model/pixel_texture_disc.py` reads config
**not at all** — by design, per its own contract ("this module never reads
yaml") — so the harvest's precondition can never be satisfied there. The
A18-D3 comment is true of `disc_holdout_probe.py` and false of this module.

**Consequence had it been missed:** under `strict_override_keys=true` all 15
`pix_*` keys would have been reported "merged silently, no effect" and the job
would have died **before the trainer was constructed**. Registered explicitly
from the trainer via `_OVERRIDE_GUARD_KEYS`.

**Bookkeeping debt this creates:** a registered key is allowlisted *whether or
not anything reads it*, which masks a future regression. As T3-B and T3-C add
real consumers, the matching entries must be **deleted** from that table so the
guard goes back to proving the keys are genuinely read. T3-B is instructed to
report which it removed.

## 12. `pix_gan_weight` — calibration protocol, not a number

§5.5 withdraws the 0.03 default deliberately: it was measured under a **scalar
latent** critic and none of it transfers (pixel domain, 660 patch logits with
the nonlinearity applied *before* the mean, new loss form). Shipping a number
without measurement re-imports the constant §5.5 threw out.

The ratio is **linear in the weight**, so the calibration is cheap and exact:

1. **Weight-free probe** — ~25–50 steps; read the A7 `gan_dmd_grad_ratio` for
   the pixel term **computed at weight 1.0 but not applied to the loss**.
2. **Solve** — `pix_gan_weight = 0.10 / r_probe`, targeting the middle of
   §5.5's 5–20 % band.
3. **Bracket** — ×1/3, ×1, ×3 around it, per §5.5.

**Step 1 requires a build change, which is why it is recorded here**: the A7
block must report an **unweighted** `pix_gan_grad_ratio`. Without it, a
weight-free probe reads exactly `0.0` (weight 0 ⇒ no contribution to
`gen_gan_loss` ⇒ no gradient), so "just run it at weight 0" does not work.
Assigned to T3-C.

*Scale-setting only, not a recommendation:* historical LADD arms measured
`gan_dmd_grad_ratio` at **0.0005–0.012** (0.05–1.2 %), far under the intended
band — those critics were small and near-irrelevant. If the pixel critic lands
in the same decade, step 2 implies a weight of order **10–200**, which is
wildly far from 0.03 and is exactly why inheriting the old constant would have
produced an inert arm. That is arithmetic on a different critic; the probe
settles it properly in one short run.

**No safety net:** A4 `gan_grad_target_norm` was DELETED, so there is no
GAN-side gradient cap to catch an over-large weight. That argues for taking the
probe step rather than launching on an estimate.


---

## 13. Holder protocol for test steps (agreed with the queue owner)

Short pytest runs may use an **existing** holder via
`srun --overlap --jobid=<id> -N1 -n1 --gpus-per-task=1`. **Never submit
(`sbatch`) or allocate (`salloc`) anything, and never `scancel`/`kill`
anything** — that is a hard standing rule.

**Before using any holder**, check for `logs/.holder_running_<jobid>.sh`.
**Marker present = a live command is running there — stay off.** Confirm with
`squeue -s -j <jobid>` (job STEPS, not jobs).

As of 2026-08-23 ~18:10:

| holder | use |
|---|---|
| 6108076 | **safe for short test steps** (~40 min left) |
| 6109490 | **designated spare** — safe once the pending set starts |
| 6108075 | busy — `smoke_pixtex` |
| 6109486 | busy — `pixtex300` |
| 6109488 | busy — `noanchor300` |
| 6109489 | **reserved** for the first adversarial pixel arm |

If nothing is free, run CPU-only with the thread limits and **label it
honestly in the report** rather than disturbing a live run. Also weigh
walltime: a suite that would be cut off mid-run is worse than a labelled CPU
result.

**Incident, recorded for completeness:** two short pytest steps (~5 s and
~22 s) were run on 6108075 via `srun --overlap` *before* it was known that
`smoke_pixtex` had been scheduled there. The queue owner confirmed no harm —
`smoke_pixtex` booted clean afterwards with preflight seeing clean GPUs. The
marker-file check above exists so this cannot recur.

## 14. First adversarial arm — agreed configuration

- **Fake source: `pix_finish_grad_enabled=false`** (fake =
  `flash_dmd_gan_x0`) — cheaper, no retained rung graph, no frame-dilution
  mask to select on. The A23 grad path is a **follow-up variable**, now that
  A23 is overturned (§6.0). Both paths are built and both must stay tested.
- **`pix_gan_weight`: probe first, no exceptions.** The weight-free-probe
  protocol of §12 is the plan of record; the unweighted `pix_gan_grad_ratio`
  telemetry (T3-C) is load-bearing for it.
- `strict_override_keys` is not set by the pre-queued arms, and they import
  the trainer at holder start, so they inherit T3-A's `pix_*` registration.

**Standing test principle adopted across this package:** any guard asserting
something is ABSENT (`pix_r2_*` symbols, band mismatch == 0,
`pix_real_repeat_frac == 0`, holdout rides absent) ships with a
**planted-violation companion test** proving the guard fires. A guard nobody
has seen trip is not evidence.


---

## 15. Decoupling thesis (GAN_REDESIGN_TWO) — WP-PIXGAN's review, and outcome

The researcher's binding diagnosis (2026-08-23): every latent GAN failure
traces to projecting the disc onto the DMD teacher's own weights, so the
critic is blind to exactly DMD's mistakes. Three options: (A) Wan 14B prefix,
(B) stock Wan 1.3B, (C) this package's pixel critic + surrogate.

Views appended to `### WP-PIXGAN` in `docs/GAN_REDESIGN_TWO.md` (that file's
other subsections are not ours). **Both objections were adopted in full and
the execution plan is now rev 2.**

### 15.1 Objection 1 — the blindness evidence is confounded with weight-starvation

The A7 numbers cited as support, `gan_dmd_grad_ratio` **0.0005–0.012**, are
**0.05–1.2 %** against §5.5's intended **5–20 %** band: those critics carried
~1/100th of the intended gradient share. A critic at that share is
near-irrelevant *whatever basis it projects onto*, so **"blind because
coupled" and "inert because ~100× underweighted" predict the same null**, and
no arm on record separates them. (poolrich-vs-raw_t0 is independently
confounded on ~7 axes and cannot break the tie either.)

Not a refutation of the thesis — an objection to the **experiment**: A and B
launched at inherited weights would very likely land at 0.05–1.2 % again and
the null would be recorded as *"decoupling didn't help"* when decoupling was
never tested at a weight where any critic could matter.

**Adopted:** the weight-free probe (§12) is now a **prerequisite for every
decoupled arm — A, B and C.** Two new Opus items created: generalise the
unweighted-ratio telemetry to the LADD path (so A/B can run the probe at all),
and build the latent-side positive control.

### 15.2 Objection 2 — readout criterion (i) is undecidable for A and B

The plan's *"d_loss trajectory that actually separates, not pinned at ln 2"*
is exactly the reading `TEXTURE_GAN_DESIGN` §4.1/§8.1 establishes is **not
available on its own**. The three criteria were presented flat; they are not:
**(iii) the positive control GATES (i) and (ii)** — §8.1's decision table.

And **(iii) did not exist for A or B**: `poscontrol` / `positive_control`
appear **only** in `model/pixel_texture_disc.py`; there is no latent-side
control in the tree. A and B as planned would have produced precisely the
undecidable ln-2 reading the control was built to eliminate.

**Adopted:** the (iii)-gates-(i)/(ii) hierarchy is now explicit, and until a
latent control lands the plan states A/B's `d_loss` is **non-informative** —
they are judged on battery + video only.

### 15.3 Ordering and the rest

- **Ordering adopted: B → C → A.** B now (config-only, OOM-immune, free first
  data point); **C not sequenced behind A** (A is blocked on an OOM fix of
  unknown duration, C on work already in flight, and C is the only
  full-strength test of the thesis). B is the weakest decoupling — same
  family, same scale, merely un-fine-tuned — so a B null is ambiguous between
  "decoupling doesn't help" and "not enough decoupling".
- **First C arm:** fake = `flash_dmd_gan_x0`; `pix_finish_grad_enabled=true`
  booked as the **FIRST** follow-up variable, on the argument that A23's
  statistical claim collapsed but its *mechanism* (an extra denoising forward
  yields a cleaner tensor) remains plausible — and training a **texture**
  critic against a systematically-cleaner-than-deployed tensor is the failure
  mode we would least like, because cleanliness is the judged property.
- **The thesis STRENGTHENS §4.1's from-scratch exception** — a from-scratch
  critic is the only one here with provably *zero* shared basis with the DMD
  teacher. Consequence booked: the **B5 escalation** to DINOv2/SD weights
  stays decoupled from the teacher but loses that property, so it is **not a
  pure upgrade**.
- **Residual coupling in C, deliberate and bounded:** real and fake both
  decode through the **same frozen Wan VAE** the teacher uses (§2), so the
  pixel critic cannot see what the decoder masks. Intentional — same-decoder
  cancels the transfer function, and raw-RGB reals would reward out-sharpening
  the decoder, the exact failure being treated — and bounded by the Case-1
  verdict (~15 % HF softening; structure, tails, isotropy survive). Booked so
  "full escape from the latent-teacher family" is not over-read as "zero
  shared components with DMD".

**Nothing in the B1 spec changes.** Option C is unchanged in content, only
re-motivated. T3-B/C continue as planned; the unweighted `pix_gan_grad_ratio`
in T3-C is now load-bearing for the whole campaign, not just this arm.


---

## 16. THE SEAM BUG — `pix_finish_grad_enabled` never reached the pipeline

Found by the live `smoke_pixtex` 60-step 2-node run (2026-08-23), via the
override guard reporting the key as *"merged silently, NO effect on this run"*.
**That warning was a TRUE POSITIVE.**

### The defect

| reader | object read |
|---|---|
| `pipeline/action_forcing_training.py:1986` | `getattr(self, "pix_finish_grad_enabled", False)` — the **PIPELINE object** |
| `trainer/causal_action_forcing_train.py:~4597` (T3-B) | `getattr(self.config, "pix_finish_grad_enabled", False)` — the **CONFIG** |

Grep across `trainer/`, `model/`, `pipeline/` for any assignment or `setattr`
of that attribute onto the pipeline: **zero hits.** The two halves read
different objects and nothing connects them.

**Consequence:** with `pix_finish_grad_enabled=true` on the launch line the
trainer's `use_grad_src` goes True and demands
`info["finish_denoised_chunk_grad"]`, but the pipeline gate stayed False, never
built the buffer, and published `None`. The ladder-fake path **cannot run at
all** today.

**It fails LOUD, not silent** — T3-B's consumer raises naming this exact seam,
and refuses to fall back to `flash_dmd_gan_x0` because the fake source is the
experimental variable (§6.0). So `pixtex300` as pre-queued with
`pix_finish_grad_enabled=true` would **crash at the first pixel-GAN step**
rather than mis-train. Arm 1 is specced `false` anyway (§14), so the launch
script should be set to `false` — flagged to its owner, not edited here.
Currently `true` in `sbatch/run_pixtex300.sh:23` and
`sbatch/run_smoke_pixtex.sh:23`.

### What the smoke does and does NOT certify

- **DOES**: T3-A build, DDP wrapper executing for the first time with **zero
  unused-parameter errors** (retiring the largest structurally-unverifiable
  risk — `find_unused_parameters=False` vs the R1-subsample branch), the A22
  probe, and `step_peak` 55.94 GB at byte-parity with the noanchor base.
- **DOES NOT**: certify T1 in vivo. The A23 grad path never engaged.

### Two lessons, both generalisable

**1. Do not silence an unread-key warning by registering the key.** The offered
remedy — add it to `_OVERRIDE_GUARD_KEYS` "so the warning stops crying wolf" —
would have suppressed a correct bug report and hidden the defect indefinitely.
Registration is correct ONLY after verifying the key genuinely is read through
an idiom the scan cannot see (§11 registered 15 keys only after running the
scan and measuring the empty result).

**2. Both halves were tested; the SEAM was not.** Every existing test passed
while this bug was live — T1's tests set the attribute directly on a stub
pipeline, T3-B's read the config. Stub-to-stub agreement is not evidence of
wiring. **A flag is not wired until one test drives it end to end from config
to consumer**, and that test must be shown to FAIL on the pre-fix code.

This is the same shape as the gradient-share finding (§15.1) and
WP-SURROGATE's decorative Sobolev term: *a thing present, plausible-looking and
logged, but not actually connected to what it is supposed to affect.* Three
instances now, in three packages.

## 17. Cross-package findings received (not WP-PIXGAN's to fix)

- **Option-A OOM — CORRECTED TWICE; the settled answer is last.**
  1. The original recorded cause (graph-on 14B forward fed the padded
     18,721-token seq_len) was **falsified** by WP-14B from the smoke's own
     log: the failing 366.00 MiB alloc is `rope_apply`'s `view_as_real` at
     **S=9360**, the ACTUAL chunk token count; the padded budget would give
     731.29 MiB, off by 2x.
  2. The replacement theory — that `ladd_gen_guidance_micro_batch_groups`
     was left at its default of 1 — was **also wrong, and I corroborated it
     wrongly.** `sbatch/run_smoke_14bdisc.sh:25` launches through
     `sbatch/_fgan_holder.sh`, whose **own fixed arg list** passes the knob
     at `:474` (`$DEXTRA` merges after and does not carry the key). The
     resolved value was **4**. I checked what the script's DEXTRA contained
     and never checked how the script launches — one level too shallow.
  3. **Settled cause: 4 is not enough at that geometry.** The original
     ~8.5 GiB/row was measured on 3-frame chunks; the smoke runs 6-frame
     `2*npb` transition chunks at 9360 tokens/row, ~2x. Measured at F=6:
     16 rows at `groups=4` = **79.8 GiB** disc-side (`groups=8` -> 51.0,
     `groups=16` -> 36.7). Recommended line
     `ladd_gen_guidance_micro_batch_groups=16 ladd_gt_transition_match_max_real=8`
     in DEXTRA, so it overrides the holder's `:474` without editing a holder
     mid-run.
  *Generalisable, and unchanged through all three revisions*: the two
  micro-batch knobs are not interchangeable and the expensive one is the
  **G** side — in train mode the disc input carries no grad, so nothing
  checkpoints and no graph reaches the backbone.
  **This is the strongest argument yet for standing rule 4** (echo the
  RESOLVED value): every launcher-level check made by either of us was
  accurate, and the conclusion still came out wrong, because the effective
  value is assembled across two files. Only the resolved value settles it.
- **Option B needs no code** (stock 1.3B measured through the Option-A
  machinery on GPU; 0.83 GiB prefix). **Trap:** at dim 1536 the disc's CCM/head
  shapes MATCH the v14e-projected arms, so the resume guard does not fire and
  **a resumed disc silently carries the coupled basis the arm exists to
  remove.** Launch `decouple13b` with a fresh disc and verify in the startup
  log. Invisible with the 14B, where 1536→5120 forces the mismatch.
- **Unweighted-ratio telemetry split agreed with WP-14B**: WP-PIXGAN lands it
  as a **reusable helper** (loss term + params in, ratio out) in T3-C; WP-14B
  adds the LADD call site after. Avoids a second implementation of the same
  statistic — the exact divergence that produced four inconsistent LADD R1
  estimators (A16/A17).


---

## 18. §8.1 control harness is now reusable by latent critics

Landed in `model/pixel_texture_disc.py` (additive; `79 passed, 32 subtests`):

- `patch_logit_telemetry` accepts `[N]` and `[N,1]` alongside `[N,1,h,w]` —
  **the same accepted-shape set as `disc_holdout_probe._reduce_scores`**, so
  the two agree by construction. Grid-less inputs return
  `{patch_mean, patch_std, patch_count}` and **OMIT `patch_spatial_var`**
  rather than reporting `0.0`. Shapes outside the set now raise instead of
  silently producing nonsense grid keys.
- `patch_logit_spatial_variance` is **untouched** and still raises on `[N]` /
  `[N,1]` — asking a spatial statistic of scalars is a caller bug. A raising
  function plus an omitting caller is the right division of labour.
- `positive_control_readout`'s docstring now states explicitly that `disc` may
  be **any** scorer callable, pixel- or latent-space, and that the `*_px`
  parameter names are historical rather than a constraint.

**The `[N,1,h,w]` path was verified byte-identical**, not merely asserted: the
pre-change implementation was run side by side against the new one over four
grid inputs × three prefixes, checking identical key set, identical key
**order**, and exact float equality. An in-suite regression guard pins the same.

**Why omit rather than zero-fill** (the rule, from WP-SURROGATE, now applied in
both packages): a spurious `patch_spatial_var = 0.0` reads as *"the critic is
not using locality"* — a real §7 diagnostic — and that must not be forgeable by
a tensor shape. A test pins that a **genuinely flat grid** still reports a real
`0.0`, so the diagnostic the omission protects is itself locked.

**Consequence for the campaign:** the LADD path emits token logits by default
(`ladd_scalar_output` defaults False), i.e. the `[N,1,h,w]` shape that already
worked. With the `[N]` case now hardened, the latent positive control for
Options A and B is a **wire-up** — a scorer callable plus a latent-domain
amplitude calibration — not a build, in either disc output shape.

### Two §8.1 semantics clarifications found while testing

1. **AUC does not saturate for the token-map scorer.** A uniform shift is exact
   in the *gap*, but pooled patch logits spread wider than the shift, so
   `AUC = 0.949` where a per-sample scorer gives 1.0. Pin the gap and bound the
   AUC; do not expect 1.0 from a patch-map critic.
2. **`rank_ok == 0` is the calibration alarm, and it fires correctly.** A
   fixture that made the control *less* separable than the student returned
   `rank_ok = 0` — which is exactly §8.1's *"if that ordering inverts, the
   calibration is wrong, not the critic"*. Kept as a dedicated inversion test
   rather than being tuned away.


---

## 19. Diagnostics vs SAFETY CLAIMS — omission is not always right

Refinement to §16's forgeable-zero rule, drawn by WP-SURROGATE. B1 has two
keys that are **not diagnostics at all**:

| key | what it asserts |
|---|---|
| `pix_holdout_leak` | an **A22 data-contamination claim** — on a failure that has already occurred once in this campaign |
| `pix_a21_support_ok` | the **4,096 distinct-source-frame floor** was cleared — the thing A21 says must be *measured*, never assumed from the loader |

"Omit when uncomputed" is right for a diagnostic, where a missing key honestly
reads as *"not measured"*. It is **wrong** for a correctness assertion, because
**silence and "clean" are too easy to confuse**: a reader or a script seeing no
`pix_holdout_leak` key concludes there was no leak — the same false all-clear,
reached by a different route.

> **A diagnostic answers "how is it going" and MAY be silent when uncomputed —
> omit it, with a regime flag.**
> **A safety claim answers "is this run valid" and MAY NOT be silent.** It must
> be computed on every path where the guarded operation ran; if it cannot be
> computed, the run **fails loudly**, naming what could not be verified.

`model/disc_holdout_probe.py` already implements this posture — it fails
CLOSED with `INVALID_NO_HOLDOUT_ROOT` / `INVALID_NO_TRAIN_RIDES` /
`INVALID_POOL_EMPTY` rather than a quiet pass. B1 matches that convention
rather than inventing a second one. Omission is retained for the genuine
diagnostics (`pix_real_repeat_frac`, `pix_band_mismatch`, the patch stats, the
finish-grad frame counts).

## 20. Seam tests require a MUTATION CONTROL

A seam test is evidence only once it has been **demonstrated to fail on the
broken code**. WP-SURROGATE set the standard: they cut the seam on a scratch
copy of the module (never the shared tree, which is under live pytest) and
measured that the new seam test detects the break, the pre-existing tests all
still pass, and telemetry still reads healthy. Required for B1's
`pix_finish_grad_enabled` seam test and any future one.

Without it, *"I added a seam test"* is an assertion, not a result — which is
the same category error as the defects being hunted.

## 21. The pattern, named — five instances in three packages

| # | package | instance | detection gap |
|---|---|---|---|
| 1 | WP-PIXGAN | `gan_dmd_grad_ratio` at 0.05–1.2 % vs the 5–20 % band | value, not **share** |
| 2 | WP-SURROGATE | Sobolev term decorative (cos −0.002 vs +0.986 normalised) | value, not **share** |
| 3 | WP-SURROGATE | `critic_grad_cos_sim = 0.0` in value-only mode, **with a test pinning the 0.0** | placeholder, not **absence** |
| 4 | WP-PIXGAN | `pix_finish_grad_enabled` never reaches the pipeline | both endpoints, not the **path between** |
| 5 | WP-SURROGATE | `latent_origin` plumbing — cutting it left 44/44 green | both endpoints, not the **path between** |

**One mechanism, three detection gaps.** In every case a component is present,
plausible and logged, and the telemetry answers a question *adjacent* to the
one that matters — measuring what is easy to measure and reading it as the
thing you care about. And in every case **the correct reading was available and
nobody computed it**, which is why a standing telemetry rule beats vigilance.

A sixth instance, and the one that cost the most people the most time: the
gen-guidance micro-batch knob. Its **effective** value is assembled across two
files (a launch script's DEXTRA *and* the holder's own fixed arg list) and then
derived from three inputs (`_gmg`, `n_rows`, `disc.training`) — and nothing
logs the result. Two packages independently reached a wrong conclusion about it
from accurate launcher-level evidence. See §17.

**B1's own exposure to the same thing**: `pix_decode_batch` is read as
`getattr(cfg, "pix_decode_batch", 4)` and controls the grad-decode micro-batch
size — i.e. the G-side memory, the expensive side — and its resolved value was
not logged. T3-B is adding a one-time resolved-value echo for every `pix_*`
knob, including `pix_band_count` (which §7 already requires echoed every run,
so a silently tightened A24 banding is visible in the trace and not only in a
diff), and echoing **derived** values rather than requested ones wherever the
two can differ.


---

## 22. INSTANCE #8 — the seam FIX was itself a silent no-op

The most instructive instance of the whole set, because **the pattern recursed
into its own remedy.**

§16 recorded that `pix_finish_grad_enabled` never reached the pipeline. The fix
propagated the config value onto **`self.model`**. That was the wrong object:

| fact | evidence |
|---|---|
| the gate read lives in `ActionForcingTrainingPipeline` | `pipeline/action_forcing_training.py`, `getattr(self, "pix_finish_grad_enabled", False)` inside `generate_chunk_with_cache` |
| the trainer holds that instance as **`self.pipeline`** | `trainer/causal_action_forcing_train.py:1308` |
| `self.model` is `ActionForcingDMD` — a different object | — |
| and the two are cross-linked, which is what makes them easy to confuse | `:1336  self.model.inference_pipeline = self.pipeline` |

So the fix **compiled, ran, logged a cheerful confirmation, and did nothing** —
silently indistinguishable from the bug it was fixing. Worse: **the seam test
passed**, because its stub was named `model`. The test verified the value
*moved*; it never verified it landed on an object that anything reads.

Now propagated to `self.pipeline` (`:1372`) with the in-gate re-assert
(`:5151`).

**The check that would have caught it, and now does:**
`test_flag_lands_on_the_object_whose_class_reads_it` **parses which class
performs the read**, **parses which trainer attribute is assigned an instance of
that class**, and requires the two to match. That converts "two things that must
agree, with nothing enforcing it" into something self-enforcing — the same
remedy as deriving the stub's method list off the real class rather than
hand-maintaining it.

**Answering the rebuild question directly:** `self.pipeline` is assigned exactly
once (`:1308`) and never rebound — no resume, eval or EMA path reassigns it.
Only sub-modules (`model.generator.model`, `model.fake_score.model`,
`model.real_score.model`) are swapped for `torch.compile`/DDP wrappers, and
those are different objects, so they cannot strand the flag. **Construction is
the load-bearing path in production; the in-gate re-assert is insurance** — and
the rebuild path is covered by test anyway.

**Lesson, distinct from the earlier ones:** a seam test must assert the value
arrives at *the object whose class performs the read*, not merely that it
arrives somewhere. A stub named after the wrong attribute will validate a no-op
fix indefinitely.

## 23. GPU verification status — honest limits

| suite | CPU | GPU |
|---|---|---|
| `test_pixel_texture_disc.py` | green | **green** (with `test_a23_finish_grad.py`, 91 passed + 20 subtests) |
| `test_a23_finish_grad.py` | green | **green** (same run) |
| `test_pixgan_trainer_wiring.py` | green (37/37) | **NOT VERIFIED** |
| `test_pixgan_trainer_supply.py` | in flight | **NOT VERIFIED** |

The two trainer suites cannot currently be collected on a compute node:
`from pipeline import SelfForcingTrainingPipeline` fails with
*"unknown location"* — the namespace-package symptom, i.e.
`pipeline/__init__.py` did not execute. Reproduced three times on holder
6108076 (bare `srun`, with `PYTHONPATH` set, and with an explicit `cd`), so it
is not a cwd or path problem. Note `testing/test_a23_finish_grad.py`'s own
docstring records that `pipeline/__init__.py` -> `utils.wan_wrapper` ->
`wan.modules.t5` touches `torch.cuda.current_device()` at import time, which is
why that suite stubs the wrapper; the trainer suites import the real trainer and
cannot.

**This is a test-infrastructure limit, not a product defect** — but it must not
be reported as "GPU-verified". The trainer wiring is **CPU-verified only**; its
GPU-specific behaviour (DDP wrapper execution, CUDA placement, real VAE decode)
is covered instead by the live `smoke_pixtex` 60-step 2-node run, which
exercised T3-A's build with zero DDP unused-parameter errors.


---

## 24. WEIGHT-STARVATION CONFIRMED ON REAL DATA — and the weight must NOT be copied

The main session ran the retroactive division (they have wandb access; WP-14B
and I did not). WP-14B supplied the identity, I verified it in code (§ below),
they supplied the data:

```
r_unweighted = gen/train/gan_dmd_grad_ratio  /  gen/train/r3gan_g_weight_gtxn
  notaps   0.125
  nopatch  0.153
  all      1.84 median   (n=2, range 0.14-3.55)
rows with gen/train/dmd_grad_norm_shared == 0  EXCLUDED
```

**VERDICT: weight-starved, not blind.** The intrinsic gradient share sits in or
above §5.5's 5-20 % band; `w=0.03` was crushing it ~30x. **Probe arms are
cancelled** — a division over the first ~50 steps of any arm suffices.

**The zero-denominator filter fired.** Several rows had
`dmd_grad_norm_shared == 0`. Unfiltered they would have dragged the ratio toward
zero — manufacturing exactly the *"the critic contributes nothing"* reading that
keeps the blindness thesis alive, **on an artefact**. That is the forgeable-zero
pattern (§21) actively distorting the adjudication *of* the pattern. Anyone
re-running this division must apply the filter.

### What it settles, stated precisely

The decoupling thesis is **unsupported by the ratio evidence, but UNTESTED —
not falsified.** Every DMD-projected critic on record ran ~30x underweighted, so
*"coupled critics don't work"* was never tested; it was confounded with
*"critics contributing ~0.3 % of the gradient don't work"*. Decoupled-vs-coupled
**at a properly set weight** is the clean experiment. With n=1–2 qualifying rows
per arm (the ratio logs every 25 steps) and a 0.14–3.55 spread, no stronger
claim is supportable in either direction.

### The weight is per-arm. Copying it is PROHIBITED.

The three measurements span **more than an order of magnitude within one
family**. Implied weights for a 10 % target:

| arm | r_unweighted | implied w |
|---|---|---|
| notaps | 0.125 | ~0.80 |
| nopatch | 0.153 | ~0.65 |
| all | 1.84 | **~0.054** |

At `w=1.0` the `all` arm's GAN gradient would be **~1.8x the DMD gradient** —
almost certainly destructive, and **A4's cap is deleted so nothing catches it**.
So "~0.7–0.8 for this family" is right for two arms and wrong by ~15x for the
third. This must be a **prohibition, not a preference**: a single quoted number
is exactly how `0.03` propagated across the whole campaign.

### And NONE of these transfer to the pixel arm

`pix_gan_weight` comes from the pixel critic's **own** division. Different
domain (decoded RGB, not latents), different reduction (660 patch logits,
nonlinearity applied *before* the mean), different loss form, different critic
(0.66 M params, from scratch). There is no reason its intrinsic share should
resemble a LADD projector's, and inheriting 0.7–0.8 would be the identical
mistake as inheriting 0.03 — just with a fresher number.
B1 keeps `resolve_gan_weight(..., strict=True)` so absence stays loud, and T3-C
lands the resolved-weight echo plus the unweighted ratio so the pixel arm's
first ~50 steps produce its own value.


---

## 25. ADVERSARIAL REVIEW — stable half (T1 + T2). 7 defects, 2 HIGH.

Suites are genuinely strong: **mutation score 18/18 caught, zero survivors**, and
the R2 spec-drift tripwire was *demonstrated* to trip on both a plain planted
`pix_r2_gamma = 0.0` and a smuggled string-literal form. But seven real defects.

### F1 (HIGH) — `gsq` is NOT scale-free in P. §5.3's claim is FALSE.

§5.3 asserts the mean reduction is *"scale-free in `P` by construction"* and
that this is *"the whole justification for γ=1.0 meaning anything"* — the
replacement justification adopted **after** the ln-2 rationale was falsified.
It does not hold either.

Measured on the shipped `PixelTextureDisc`, and **independently reproduced by
the main agent**:

| P | 165 | 660 | 2640 |
|---|---|---|---|
| `gsq` | 6.01e-3 | 1.53e-3 | 1.86e-4 |

fitted `gsq ∝ P^α` → **α = −0.989, then −1.519** (reviewer measured −0.98 /
−1.11 / −0.93 on noise and −0.91 / −0.93 on smooth+texture). Claimed α = **0**.
The sum reduction measures **α ≈ +1.0**, not the +2 the spec asserts.

**Mechanism, and it is analytically correct:** patch gradients are local and
near-independent, so `‖Σᵢ∇Dᵢ‖² ≈ P·‖∇Dᵢ‖²`; the mean divides by `P²`, giving
`∝ 1/P`. The spec's `P²` reasoning assumes perfectly **correlated** patch
gradients, which a local conv critic does not produce.

**The estimator itself is correct** — FD/autograd agreement 1.04 over 12 trials
— so this is a false *claim*, not broken math. **Consequence: γ=1.0 is tied to
the crop size it was calibrated at.** At the spec crop (P=660) it stands; at a
2× linear crop the same γ delivers ~4× weaker R1, 16× at 4×.
**Both justifications for the mean reduction have now been falsified.** It
remains the specced behaviour (and is not a flag), but it should be understood
as a convention, not a portability guarantee.

### F2 (HIGH) — the test pinning F1 is vacuous, three ways

`TestR1::test_scale_portability_in_P`, docstring *"THE JUSTIFICATION FOR THE
MEAN … the MEAN-based one is invariant"*:
1. acceptance window `0.05 < ratio < 4.0` — **80× wide** for a quantity called
   invariant; the measured 4×-P ratio of **0.265** sails through;
2. it runs `_SmoothNet`, not `PixelTextureDisc` — and `_SmoothNet` itself
   measures α = −0.958, so the substitute already refutes the claim;
3. the final assert, commented *"so the test is not vacuous"*, reduces
   algebraically to `P²x/P²y == x/y` — **it returns 1.0 on random floats with
   no network involved.**

### F3 (MED-HIGH) — the receptive field is measured on a net that never trains

`measure_receptive_field` runs `use_norm=False` → a clean 38×38. With
**GroupNorm ON, the training configuration**, a single centre patch logit's
input-gradient spans the **whole 176×240 image**, 5.7 % of gradient energy
outside a 50×50 box — GroupNorm normalises over the full spatial extent, so
every patch logit depends on every pixel. §4's *"receptive field ≈ 70 px — a
local texture question"* is **not a property of the shipped critic**. Whether to
drop the norm is a **researcher decision**; the code must stop claiming
locality it does not have.

### F4 (MED) — the A23 fallback writes the FLASH tensor into the grad buffer

The comment says the slice is written with `finish_grad_pred`, *"never
`cache_pred`"*; the else-branch does exactly `_grad_slice = cache_pred.detach()`
and by then `cache_pred = flash_dmd_pred.detach()`. Demonstrated on a
9-frame/3-block flash-ON call: `clean_chunk_grad[:, 6:9]` is **bit-identical to
the t=60 flash output**. The guarding assert sits in the other branch. The mask
protects a compliant consumer, but **flash-ON with a detached block was
untested**, and any consumer or viz decoding the whole buffer sees precisely
the tensor A23 exists to exclude.

### F5–F7 (LOW / LOW-MED)

- **F5** — `patch_logit_telemetry`'s "same accepted shapes as `_reduce_scores`"
  is false: `_reduce_scores` has **no** dim restriction, telemetry allows only
  dim ∈ (1,2,4). At `[6,1,7]` the probe returns `(6,)` and telemetry raises —
  and a **3-D token map is the realistic latent case**, which is the stated
  reason the relaxation exists.
- **F6** — `pix_r1_grad_sq` single-draw relative std is **135 % at N=1** (300
  draws spanned 1.7e-8 to 2.3e-2), yet §5.3/§5.4 say calibrate γ from it on the
  **first logged row**. Unbiased, so the mean is right; one sample is not.
- **F7** — the override guard harvests **`pix_r2_gamma`** from planted-violation
  fixture *strings*, allowlisting the one symbol B1 forbids.

### Reviewer's own disclosure, in the campaign's spirit

Its first mutation battery reported 17/18 **survivors**. That was its own
harness bug — `Path(__file__).resolve()` followed a symlink back into the repo,
so both the scanned source and the imported module were the pristine ones. *A
check that ran, reported, and measured the wrong object.* Rebuilt with physical
copies plus an isolation guard; the real answer is 18/18. Ninth instance of the
§21 mechanism, found inside the instrument built to hunt it.

### Untested risk flagged (not a demonstrated defect)

`_finish_fn`'s backward-time checkpoint recompute re-enters the generator with
`kv_cache=self.kv_cache1`. A23 adds a **second** KV-writing checkpointed forward
alongside `_flash_fn`, and recomputes run in reverse order at backward, so
post-backward cache state need not match post-forward. The code's safety note
addresses recompute **reads**, not **writes**. The flash path already has this
shape, so it is pre-existing rather than introduced — but it is unverified, and
it needs a GPU run to settle.


---

## 26. RESEARCHER DECISIONS 2026-08-23 — arm 1 is BLOCKED until both land

The §25 review produced two findings that were **false claims in the spec**, not
code bugs. Both were put to the researcher rather than decided here. Three
decisions taken:

### D1 — γ is calibrated from MEASUREMENT, not assumed to be 1.0

§5.3's "scale-free in P" is false (§25 F1, α = −0.915, independently
reproduced). Rather than normalising — which §5.3 forbids via "the mean
reduction is NOT a flag" — γ now follows the **same probe-then-set discipline as
`pix_gan_weight`** (§12):

1. run ~50 steps at γ=1.0;
2. read `train/pix_r1_grad_sq_mean` (**the running mean, not a single row** —
   single-draw relative sd is 126 %, so §5.4's "read it on the first logged row"
   was never usable);
3. set γ for the target R1 magnitude;
4. bracket ×⅓ / ×1 / ×3.

The mean reduction stays as specced. It is now understood as a **convention**,
not a portability guarantee — both of its justifications have been falsified.

### D2 — the GroupNorm is DROPPED. Authorised deviation from §4.

§4's "receptive field ≈ 70 px — a local texture question" is false of the normed
critic (§25 F3: full-image gradient support, 4–9 % of L1 mass outside 38×38).
**Decision: remove GroupNorm(8) from blocks 2 and 3**; spectral norm, LeakyReLU
and all layer geometry unchanged; `use_norm` retained so the normed variant
stays constructible, but the **default flips to norm-off**.

Rationale, and it is the whole point of the arm: the B2 hypothesis is *"can
**LOCAL** decoded-pixel adversarial feedback suppress fabricated directional
texture?"* A global critic does not test it, and a null result would be
uninterpretable.

**Recorded as an AUTHORISED DEVIATION from §4's layer spec** — not drift. Do not
quietly restore the norm.

**Everything the change invalidates is being re-measured, not carried forward:**
param count (661,953 → expect 661,185), receptive field, the **α exponent**
(may move — the norm may be part of why patch gradients looked near-independent),
and **§3.7's effective-sample arithmetic**, whose ~6 tiles / ~128:1 assumed a
~70 px RF. That last is a **launch-configuration input**: it feeds
`pix_crops_per_step` / `pix_frames_per_crop`.

**Recorded risk:** an unnormalised from-scratch PatchGAN may train less stably.
Being smoke-tested synthetically on CPU, which is **not** evidence about the
live arm. Watch `pix_d_loss` early.

### D3 — arm 1 does NOT launch until D1 and D2 are both resolved

An arm run on a critic whose R1 strength and receptive field are not what anyone
intended cannot attribute its result to the pixel-adversarial term. Holder
6109489 is held. **Options A and B are unaffected.**

Order is load-bearing: **drop the norm first, then calibrate γ on the final
architecture.** γ measured on the normed critic would not transfer.

### Spec status after this

`TEXTURE_GAN_DESIGN.md` now contains measured-false claims in **§4** (locality)
and **§5.3** (scale-freedom), and **§3.7**'s arithmetic depends on the §4 one.
That file is not WP-PIXGAN's to edit; flagged to the main session for the ledger.


---

## 27. INSTANCE #11 — queue-time vs START-TIME skew (found by the main session)

A pre-staged arm's config can be **correct when written and wrong when it
starts**, because the code underneath it changed in between. Nothing is
individually wrong at any instant — not the config, not the code, not the
telemetry.

**The case:** `sbatch/run_pixtex300.sh` was queued as a 300-step *control* but
carried `gan_pixel_texture_enabled=true` from its smoke ancestry. Harmless while
T3-C was unlanded (the gate reached nothing). But a **late holder start after
T3-C landed** would have booted it as an **accidental adversarial arm** — on the
pre-D2 architecture, with γ unmeasured — or crashed on the absent
`pix_gan_weight`. Found and fixed by the main session; now
`gan_pixel_texture_enabled=false` + `pix_finish_grad_enabled=false`, a pure
control invariant to B1's landing schedule.

**This is a distinct class from the other ten.** The rest are "requested vs
resolved" in *space* (across files, objects, derivations). This one is
requested-vs-resolved across **TIME**: the resolved meaning of an unchanged
config line moved when new code landed. Pre-staged work is the exposure, and a
long-running queue is what makes it likely.

**Residual, flagged to the script owner (sbatch/ is not WP-PIXGAN's lane):**
`sbatch/run_smoke_pixtex.sh` is the only launcher in the tree that still sets
`gan_pixel_texture_enabled=true`, with **no `pix_gan_weight`**. It already ran
*before* T3-C, so nothing is queued on it — but a re-run today is the same
accidental-adversarial-arm case. It would **crash** rather than mis-train
(`resolve_gan_weight(..., strict=True)` raises on the absent weight), which is
the fail-loud working — but a crash on a holder is still worth closing.

**The general defence is standing rule 4.** A pre-staged arm that echoes its
**resolved** gate values at step 0 makes queue-vs-start skew visible in the first
log lines rather than at whatever step it bites. `_pix_echo_resolved_config`
already does this for every `pix_*` knob, derived-not-requested — the instrument
exists; someone has to read line one.

### Running tally — one mechanism, five detection gaps

| gap | reading taken | reading needed |
|---|---|---|
| value, not share | the term's magnitude | its **fraction of the gradient** |
| placeholder, not absence | a fake 0.0 in range | **omission** (or an out-of-range sentinel) |
| endpoints, not the path | both halves tested | the **seam** driven end to end |
| requested, not resolved | what the launch line asked for | the **derived effective** value |
| queue-time, not start-time | correct when written | correct **when it starts** |

Eleven instances, four packages, one day. In every case the correct reading was
available and nobody computed it — which is why standing telemetry rules beat
vigilance.


---

## 28. γ=1.0 WAS INERT ON BOTH ARCHITECTURES — and how it surfaced

**Launch-critical, and stronger than §26/D1 assumed.** `pix_r1_gamma = 1.0`
was never delivering meaningful R1 — not on the norm-free critic, and **not on
the normed one either**.

Measured `R1/d_loss` at init (`d_loss = 2·ln2 = 1.3863` for nsgan), matched
inits, verified independently by the main agent:

| seed | norm ON | norm OFF | ratio |
|---|---|---|---|
| 0 | 6.488e-4 | 5.257e-6 | 123× |
| 1 | 5.958e-4 | 6.264e-6 | 95× |
| 2 | 8.741e-4 | 7.646e-6 | 114× |

Every norm-ON value is already **below 0.1 % of `d_loss`**. So:

> **The GroupNorm removal made an already-inert term ~100× more inert. It did
> not create the problem.**

§26/D1 was taken on the understanding that γ was *becoming* stale. It was
already stale, and the decision is better-founded than the reasoning that
produced it: γ=1.0 is not a slightly-outdated default, it is **inert on both
builds**.

**It would have shipped looking perfectly healthy** — `pix_r1_gamma` reads 1.0,
the R1 term appears in the loss, `pix_r1_rate` reads 1.00, counters monotone,
`pix_r1_grad_sq` logs a plausible magnitude. Nothing in the config surface or
the telemetry would have said the penalty does nothing. That is §21's mechanism
exactly — **value present, effect absent** — and it is the reason `pix_r1_grad_sq`
needed its *share* alongside its magnitude (§15.3).

### How it surfaced — the planted-companion rule earning its keep

A planted-violation companion **FAILED**. It asserted that the same negligibility
test *should not* hold under the norm — "it was NOT negligible under the norm,
which is why γ=1.0 was ever a defensible number". That premise is false, so the
`assertRaises` never fired.

The guard did exactly what §20's rule exists for: **it fired on a false premise
in its own documentation.** Had the companion been written as a bare assertion
rather than a discriminating one, the test would have passed and the finding
would have stayed buried.

### Consequence

`pix_r1_gamma` joins `pix_gan_weight` in **probe-then-set** (§12), off a measured
`pix_r1_grad_sq_mean` — the **running mean**, since a single row carries 126 %
relative sd (§25 F6). Two knobs, the same discipline, **neither inheritable**,
and neither with a safety net now that A4's cap is deleted.

**For the mechanics smoke:** R1 will be inert in it. `pix_r1_rate` reading 1.00
means the code path fired, **not** that R1 did anything. "R1 fired" ≠ "R1 worked".


---

## 29. A21 LAUNCH BLOCKER — found, confirmed, fixed

Found by the trainer-half adversarial review, arithmetic confirmed independently.

**`pix_a21_support_ok` could NEVER be true.** The arm would have certified its
own A21 violation on every logged step, for its entire life.

`_grow = min(max(0, pool_target - pool_now), ...)` goes to **0** the moment
`len(pool) == pix_real_pool_windows`, and distinct-source-frame support only
grows on admission — so support froze. Ceiling was
`pix_lat_frames_per_crop × pix_real_pool_windows` = 2 × 2048 = **4096, exactly
the A21 floor with ZERO margin** — and overlapping starts within a ride (a
window at `s` contributes `{s, s+1}`) pushed it *below*. Measured ceilings
**4069 / 4071 / 4081 / 4085** across seeds; `a21_support_ok` never reached 1.

**Two cascading defects, same root cause:**
- the real pool **froze** into a 2,048-window mini-dataset —
  `pix_real_cache_refresh_total` flatlining from D-update ~253, FIFO eviction
  **dead code**, `pix_real_pool_refresh` an inert knob. **Precisely the
  memorisation hazard A21 exists to prevent**, and the docstring promised the
  opposite ("never a frozen mini-dataset D can memorise").
- `pix_holdout_leak` degenerated into a **stale republished value**: its only
  writer was reachable only from the pool fill, so once that stopped, the
  safety block passed its `if _checked < 1.0: raise` gate on a sticky attribute
  and re-emitted a leak value computed hundreds of steps earlier, every step,
  as if fresh.

**FIX (both halves necessary):** refresh now admits past cap so FIFO actually
evicts, **and** `pix_real_pool_windows` 2048 → **4096** (ceiling 8192, real
margin). Raising the window count alone clears the floor but leaves the pool
frozen, so F2/F3 would have survived.

### A THIRD defeat route for safety claims — STALENESS

§19 established that a safety claim may not be **omitted** and may not be
**forged**. This adds a third: a safety claim can be defeated by going
**stale** — computed once, then republished unchanged forever, indistinguishable
from a fresh measurement.

> **A safety claim must be FRESH per emission.** Not merely present, not merely
> non-default — *recomputed*, or explicitly marked as not-recomputed.

Adopted campaign-wide into the standing-rule block, citing this measurement.

## 30. SEAM MUTATION CONTROL — **VERIFIED 2026-08-24** (was: unverified)

**The box is NOT ticked.** The seam tests
(`test_flag_lands_on_the_object_whose_class_reads_it`,
`test_gate_re_asserts_the_flag_after_a_pipeline_rebuild`) remain **asserted, not
demonstrated**, to catch a disconnected flag.

I attempted the control myself, on a shadow package that correctly resolved the
mutated module (`find_spec` verified: mutated file → SHADOW, siblings → REAL).
The three-way run returned "seam tests pass on M3 (both propagation sites cut)"
— which would mean the guard does not work.

**That reading was an artefact of my harness, not a property of the code.** The
seam test does not import the trainer to check the seam — it **parses the
source**, and it derives the path from
`_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`. With the
test file living in the real repo, `_ROOT` is the real repo, so it parsed
`$SRC/trainer/causal_action_forcing_train.py` **regardless of my mutant**.

**This is exactly the trap that gave an earlier reviewer 17/18 false survivors**
(§25) — a check that ran, reported, and measured the wrong object — and I walked
into it after warning three other agents about it. It was caught only by
disbelieving an identical PRISTINE/M3 result.

**Correct method, for whoever finishes it:** the test file must ALSO live in the
shadow tree, so `_ROOT` resolves there and it parses the mutated trainer. The
shadow needs `trainer/`, `pipeline/` (for `_PIPE_SRC_PATH`) and `testing/`, with
each `__init__.py` appending the real directory to `__path__` so siblings fall
through. Verify by printing the resolved `_ROOT` before trusting any result.


---

## 31. SEAM MUTATION CONTROL — RESULT: THE GUARD WORKS

§30's negative is **closed**. The seam tests **can fail**; the disconnected-flag
defect cannot recur undetected.

| variant | seam tests | full file (90) |
|---|---|---|
| PRISTINE | 3 passed | **90 passed** |
| **M1** construction cut | **1 FAILED** — `test_flag_lands_on_the_object_whose_class_reads_it` | 5 failed |
| **M2** re-assert cut | **1 FAILED** — `test_gate_re_asserts_the_flag_after_a_pipeline_rebuild` | 1 failed |
| **M3** both cut | **2 FAILED** — both | 6 failed |

M1's failure is `assert direct` → *"nothing in the trainer assigns
pix_finish_grad_enabled"*. **Both sites are INDEPENDENTLY guarded** — cutting
either alone is caught, so the insurance is itself tested and cannot silently
rot.

**Load-bearing site confirmed:** `self.pipeline` is bound exactly once
(`trainer/causal_action_forcing_train.py:1611`) and never rebound, so
construction carries the flag in production and the in-gate re-assert is
insurance against a rebuild that does not currently exist.

`test_the_reader_is_the_pipeline_not_the_dmd_model` passes on all four variants
— it pins the reader-side class name only. **It is documentation, not a
mutation detector.** The two named above are the real detectors.

### WHY THE FIRST FOUR ATTEMPTS RETURNED A FALSE "PASSES ON M3"

§30 blamed `_ROOT`. That was **necessary but not sufficient**, and the real
cause is worse: **the repo contains import-time `sys.path.insert(0,
Path(__file__).resolve().parents[1])` calls** — at
`testing/test_pixel_texture_disc.py:64` (which the supply suite imports *before*
the trainer) and at `trainer/causal_action_forcing_train.py:51`. They run on
import and shove the **real repo ahead of the shadow**, so `PYTHONPATH` ordering
and a neutral cwd are both defeated.

**The fix that actually works:** a shadow `testing/conftest.py` that
**pre-imports** the trainer and pipeline modules under the shadow, pinning
`sys.modules` before any later `sys.path` reordering can matter — plus shadowing
`testing/test_pixel_texture_disc.py` and `model/pixel_texture_disc.py`, which
that helper reaches for. Package `__init__` order is not uniform either:
`pipeline/__init__.py` imports submodules eagerly so `__path__.append` must go
**first**, while `model/__init__.py` opens with `from __future__ import
annotations` so its append must go **last**.

**Three isolation proofs are now asserted by an autouse fixture inside every
run**, so no number can be a harness artefact: (1) `_ROOT` and both parsed
source paths resolve under the shadow; (2) `find_spec` AND the imported
modules' `__file__` are shadow, while a sibling resolves to the real repo
(proving a true overlay, not a wholesale copy); (3) the mutant markers are
present on disk. Zero symlinks in the tree.

### Three incidental notes

- **The repo file changed mid-run** (+28 lines; sites moved 1647→1675 and
  6376→6404). Caught rather than silently skewing, because the mutator
  re-copies from the repo and asserts exactly one occurrence per cut. Matrix
  re-run against the current sha256 — identical outcome.
- **Two full-file runs died with SIGKILL (137), reproducibly at the same test**,
  which looked like a real M2-specific effect. It was **node memory pressure**
  (~134/237 GB used by other work); that test uses 1.2 GB in isolation and M2
  later completed 89/89. **Not a finding** — recorded so nobody mistakes it for
  one.
- Minor: `test_gate_re_asserts_the_flag_after_a_pipeline_rebuild`'s docstring
  cites `:1342` for the bound-exactly-once audit; the actual binding is
  **:1611**. Claim true, line number stale.


---

## 32. THE RULE-4 INSTRUMENT WAS ITSELF INVISIBLE

Found from a passing remark by MAIN (*"INFO is suppressed in run logs"*), and it
landed inside the very instrument built to catch this class.

`_pix_echo_resolved_config` emitted via **`logging.info`**. It is the instrument
for standing rule 4 — the thing that makes requested-vs-resolved visible, and
the defence against queue-time-vs-start-time skew. **Computed correctly, and
quite possibly never seen.**

**Verified mechanism, worse than a policy choice:**
`trainer/causal_rolling_staircase_train.py:332-337` calls
`logging.basicConfig(level=INFO if main else WARNING, ...)` **with no
`force=True`**. `basicConfig` is a **silent no-op if the root logger already has
handlers** — so if any import, launcher or torch/DDP path installs one first,
the level is never applied, root stays at Python's default WARNING, and **every
`logging.info` in the trainer vanishes.** Whether it bites depends on import
order.

**Fix (B1 side only):** new `_pix_emit_actionable(msg)` does
`logging.warning(msg)` **and** `print(msg, file=sys.stderr, flush=True)` — the
pair already used at `pipeline/action_forcing_training.py:701-702`.
Main-process-gated, once-per-run latch unchanged, message pre-rendered so a `%`
in a resolved value cannot break the formatter.

**Deliberately NOT fixed: the logging config itself.**
`causal_rolling_staircase_train.py` is not WP-PIXGAN's, other packages depend on
its behaviour, and adding `force=True` could clobber a handler someone installed
on purpose. Diagnosis handed to MAIN/researcher.

**Three more sites promoted to WARNING** (all carry actionable information):
the critic-built line (wiring claim + §12 param counts), the optimizer-built
line (lr/betas at their one resolution point, a §12 calibration input), and —
sharpest — **the A23 gate-propagation line**, which is *the only runtime
evidence the seam works*. As the implementer put it: **a proof-of-connection
that is itself only conditionally connected proves nothing.**

Left at INFO deliberately: the resume-restored lines (the absence path already
`raise`s — the fail-loud guard is the instrument, the log is confirmation),
`[mem-inventory]`, and cache-hit chatter.

**Flagged to WP-SURROGATE, not changed:** `Latent surrogate critic built`
(~:1382) is at INFO and carries resolved knobs (`pix_teacher_refresh_every`,
loss weights) — same exposure, their package, their guard file.

10 new visibility tests, proven to fail pre-fix (5 failed / 5 passed when the
emit was reverted). Note the structure: the **absence** guard correctly still
PASSES pre-fix, while its planted-violation companion FAILS — which is what
shows the companion is load-bearing rather than decorative.

## 33. OPERATIONAL — `inspect.getsource` flakes on Lustre right after an edit

Observed during §32: **transient, nondeterministic `inspect.getsource`
misalignment** — two runs returned a *different method's* source for
`_maybe_run_pixel_texture_d_updates`, producing 1 then 3 spurious failures,
followed by three consecutive clean runs. A per-test teardown watcher recorded
**no source-state change**. The repo is on Lustre and the flaking runs were the
ones immediately following file edits: this looks like `linecache` racing Lustre
metadata propagation, **not a code defect**.

**Consequence, and it is broad:** a large share of this package's guards are
source scanners (the resolution-point guard, the echo guard, the banned-constant
guards, the seam parser). **Any of them can flake in the first run after an
edit. Re-run before believing a source-scanner failure** — and, symmetrically,
do not trust a single clean source-scan run immediately after an edit either.


---

## 34. A23 IS INCOMPATIBLE WITH THE CAMPAIGN'S EXIT-RUNG CONFIG — measured in vivo

**§6.2's "~1/K" no-rung rate is WRONG for the configs this campaign actually
runs.** True for a uniform exit-rung draw; false in practice. Found only by
running it — no CPU test could have.

`smoke_pixgrad` on 6110540, `pix_finish_grad_enabled=true`, died at step ~1 on
all ranks:
```
RuntimeError: pix_finish_grad_enabled=true but info['finish_denoised_chunk_grad']
is absent. ... Refusing to fall back to flash_dmd_gan_x0 silently
```
preceded by **16 ×** the `[A23] ... but` **no_rung** one-shot.

**Mechanism.** A23 attaches its graph to the **last post-exit finish rung**. If a
block's exit rung *is* the last rung, `range(exit_index + 1, K)` is empty — no
finish rung, nothing to attach. With **`dmd_rolling_ctx_last_rung=true`** and
**`dmd_sample_at_rungs=false`** the exit is driven to the last rung, so **no
block ever has a finish rung**: nothing attaches, the buffer carries no graph,
T1's publish correctly drops it to `None`, and the consumer correctly refuses
rather than silently substituting the flash tensor.

**Every layer behaved exactly as designed. The design is incompatible with this
sampler configuration.** Three launches, three loud early failures on all ranks
with the reason in the error text — no silent mis-training, no wasted 200 steps.

### Consequences

1. **`pix_kv_commit_match` cannot be produced under this config.** The KV
   recompute-order tripwire fires on the A23 grad path, which never engages
   here. **Any conclusion gated on that verdict must be un-gated or
   re-pointed.**
2. **The A23 path needs `exclude_last_rung=True`** on the exit-rung draw (the
   parameter exists — `pipeline/action_forcing_training.py:720`, sampling
   `[0, K-1)`), **or** `force_exit_step < K-1`. Either reserves a finish rung.
   **This changes DMD's exit-rung distribution, so it is a RESEARCHER decision**,
   not a config tweak.
3. **Arm 1 is unaffected** — it uses the flash fake
   (`pix_finish_grad_enabled=false`).

### What the window did bank

**The import-collection diagnostic came back `IMPORT-OK` on all three modes**
(plain / PYTHONPATH / sys.path). See §35 — the "CPU-only" caveat is **lifted**.

## 35. CORRECTION — the "cannot be collected on a compute node" claim was WRONG

I reported, and repeated in the handoff and the grid, that the trainer suites
**cannot be collected on a compute node** (`from pipeline import ...` →
"unknown location", reproduced 3×). **That was a misattribution.**

The diagnostic on 6110540 returned `IMPORT-OK` **plain**, **pythonpath** and
**sys.path**, with `pipeline/__init__.py` resolving normally.

**The real cause was one I had already diagnosed separately**: running
`testing/test_a23_finish_grad.py` first **poisons that import for any suite in
the same pytest process** — which is exactly why the standing instruction is
*one process per suite*. I filed one fact as two, and generalised a **pytest
collection symptom** into a **node capability** claim.

Same failure class as everything else in §21, applied to my own reporting:
**three reproductions of a symptom felt like confirmation of a cause.**

**Consequence:** the trainer suites are very likely **GPU-verifiable**, one
process per suite. Not yet claimed as GPU-green — that needs an actual run.


---

## 36. RESEARCHER DIRECTIVES 2026-08-24 — exposure, and the ring withdrawn

Three directives, received directly:

1. **Restoring-force framing for the dynamical question.** The critic does not
   need to break the AR loop per step: if it learns the axes along which
   in-distribution content drifts OOD and its gradient points back inward, the
   correction compounds over many steps. One-or-two-step weakness is fine.
2. **Much more D exposure to fakes and equivalent reals.** Landing:
   `pix_crops_per_step` 4→8, `pix_gan_updates_per_step` null=follow
   `gan_updates_per_step` (→5) — **120 LIVE fakes / 120 band-matched fresh
   independent reals per step** (was 12/12), with monotone exposure totals,
   a live-fake depth histogram, and measured decode/step-time cost.
3. **Style/content only** — no contrast/global-stat terms; other parts of the
   model constrain those.

### The fake replay ring — proposed, then WITHDRAWN by the researcher

My first implementation of (2) included a SimGAN-style fake history buffer.
**Withdrawn on researcher correction**, reasoning verbatim in substance:

> The axes along which texture collapse happens are always changing. It is the
> POSITIVE samples that stay in the same style — that is what we want to get
> to. The negatives will always be changing in style; we need to know the
> CURRENT degenerate texture collapse, not all historic ones. Long-term memory
> on the fake side makes the discriminator's job HARDER.

**Principle: cache the stationary class, never the moving one.** Real style is
stationary → the A21 fresh real pool (8192-ceiling, continuously refreshed) is
correct. The fake distribution is non-stationary **by direction** → D must
sharply detect where the generator IS; a history buffer spreads its boundary
over dead failure modes. This also pre-answers any future buffer proposal for
other arms: the argument is not "buffers are bad", it is that THIS negative
class moves directionally and current-mode sharpness is the product.

Nothing ring-shaped ships — no code, no config key, no telemetry (a stray key
would merge silently; absence is grep-proven in the build report).
