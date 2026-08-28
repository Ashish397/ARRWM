# ONBOARDING PROMPT — switch decoder pullback experiments from Q1 0.6 to Q1 0.99

Copy everything between the lines to the agent currently working on the older
Q1~0.6 decoder-shaped surrogate.

---

You are taking over the decoder-shaped WAN pullback work in:

`/lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM`

Your current Q1~0.6 configuration is now obsolete as an experimental baseline.
Preserve its outputs as a historical control, but **do not spend further runs
improving or tuning it**.  Switch all subsequent decoder-pullback experiments
to the measured all-residual Q1~0.99 bundle described below.

## Critical live-GAN correction (28 August)

The offline Q1~0.99 result and artifact are valid. The first live VGG-GAN
calibration/active waves are not. Their detached pixel-decode cache identified
fresh temporary micro-group tensors by `data_ptr`; allocator reuse generated
false hits and stale real/fake/R1 pixels. Do not use `3rlvysdv` to authorize
`.10`, and do not treat `4vo4kwys/ududui2a/1wyle11y/u3i9p6ya` or `fzuybrrm`
as parameter-selection evidence.

All new runs must resolve `ladd_pixel_decode_cache=false` and log
`ladd_pix_decode_cache_enabled=0` while decode calls rise. Four corrected
weight-zero calibrations are running on holders `6168503/6168505`: DC
reference R1=10 `br6ivuzw`, DC reference R1=1 `zoud58dc`, raw reference R1=10
`ob5gqkvo`, and DC aux-minus R1=10 `xlf99yu7`. Inspect those before choosing
any nonzero weight. The safe optional cache now requires
exact retained tensor identity and mutation version, but production keeps it
off because the current micro-group geometry has no legitimate reuse.

## Read first

Before editing or launching anything, read these completely:

1. `analysis/gan_tuning/EXACT_RESIDUAL_LADDER_2808.md`
2. `analysis/gan_tuning/LOCAL_VJP_MAX_Q1_NO_RECAPTURE_2808.md`
3. A17/A18 in `docs/GAN_REDESIGN_2608.md` for the older live Q1~0.6 work
4. `model/decoder_shaped_pullback.py`
5. `model/tied_residual_vjp.py`
6. `trainer/causal_action_forcing_train.py`, especially the
   `surrogate_decoder_shaped_*` construction and consumption paths

Do not re-derive or rerun the completed ladder unless you are explicitly asked
for a new split/rotation.  The artifacts are complete.

## What changed

The older bundle made only residual stages 6 and 7 exact.  It measured about
0.60 paired Q1 and strict-transfer worst-seed Q1 0.508.  That was enough for the
old 0.50 gate, but it is no longer the best design.

We traced the remaining chain rotation to learned residual stages and ran a
predeclared full-chain ladder while freezing every checkpoint/model selection:

| Configuration | Paired worst-seed Q1 | Cartesian worst-seed Q1 |
|---|---:|---:|
| Old exact 6+7 configuration | ~0.60 | 0.508 |
| Exact mid-low 5--7 + mid-high 9--11 | 0.846409 | ~0.853 |
| Above + exact 13 | 0.888658 | 0.889639 |
| Above + exact 2 | 0.898560 | 0.903012 |
| Above + exact 3 | 0.928749 | 0.935435 |
| **All residuals exact** | **0.990999** | **0.992351** |

The final numbers are measured composed pixel-to-latent results over 81 paired
test examples and 108 unrelated Cartesian z/v pairs, not projections from
local block scores.  Per-seed paired Q1 is
`0.990999 / 0.991116 / 0.991590`; Cartesian Q1 is
`0.992351 / 0.992798 / 0.993054`.  The paired minimum is 0.982646, so the claim
is Q1>=0.99, not that every individual example is >=0.99.

## Switch to this bundle

Use this exact artifact for all further experiments:

`/scratch/u6ex/as1748.u6ex/ARRWM_data/gan_aligned_discrimination/surrogate_field_2708/local_vjp_granular_audit_r0_h6158256/exact_residual_ladder_2808/decoder_shaped_all_residual_seed0.pt`

- Size: 20,668,763 bytes (~20 MiB)
- SHA-256:
  `de82ec3211ec408f34cca883be53c15d835e19d5a3860a0b7fe8840dad450357`
- Exact residual stages:
  `1,2,3,5,6,7,9,10,11,13,14,15`
- Exact fixed stages: `4,8,12,16`
- Exact public latent prefix
- Only stage 0 (`low_core`) remains learned; its local Q1 is about 0.99

The bundle passed a production-loader smoke through
`DecoderShapedPullback.from_bundle`: correct 12-stage metadata, graph-free
output, captured-state Q1 0.99999988, and four-example field Q1 0.990512.

Some legacy launchers still default to the old file
`decoder_shaped_exact_midlow12_seed0.pt` and contain stale `q1p60` naming.
Do **not** rely on those defaults.  The guarded launcher
`sbatch/run_carn_decoder_surrogate_node.sh` now defaults to the Q1~0.99
artifact and refuses a wrong SHA or exact-stage list.  For any other interim
launcher set:

```bash
DECODER_PULLBACK_BUNDLE=/scratch/u6ex/as1748.u6ex/ARRWM_data/gan_aligned_discrimination/surrogate_field_2708/local_vjp_granular_audit_r0_h6158256/exact_residual_ladder_2808/decoder_shaped_all_residual_seed0.pt
```

At boot, require the resolved log to print exactly:

```text
exact_residual_stages=[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15] graph_free=true
```

If it prints `[6, 7]`, you accidentally launched the obsolete bundle.  Stop
only your child step, never the holder allocation, and correct the override.

## The replacement GAN is back on Flash

Do not carry forward the temporary DMD/finish-frame GAN routing.  The current
Q1~0.99 replacement GAN is explicitly Flash-aligned on both sides:

```text
flash_dmd_enabled=true
flash_dmd_gan_t=60
ladd_fake_sample_source=flash
pix_finish_grad_enabled=false
pix_flash_grad_select_enabled=true
```

`flash_dmd_enabled=true` by itself is not sufficient.  The discriminator must
train on the Flash slab (`ladd_fake_sample_source=flash`) and the generator
query must select the latest contiguous graph-live frames from that slab
(`pix_flash_grad_select_enabled=true`).  Otherwise a run can claim Flash is
enabled while the GAN still consumes the old DMD/finish surface.

The guarded launcher pins this contract in
`sbatch/run_carn_decoder_surrogate_node.sh`. It keeps
`gan_updates_per_step=1` while the cache-correct campaign is rebuilt. The
structurally protected candidate under retest is:

```text
PIXEL_FILTER=dc
STAT_ANCHOR=1.0
```

Spatial DC rejection still has an analytic zero-brightness-cotangent
guarantee, so it is a reasonable protected candidate rather than a measured
winner. The former `fzuybrrm`/raw/SWT/no-anchor comparison is invalidated by
the stale decode cache and must not select a production recipe. Do not enable
the raw-image nearest matcher; it reinstates the exposure cue that DC removes.
The historical measurements and invalidation are documented in
`analysis/gan_tuning/FLASH_BRIGHTNESS_WAVELET_2808.md`.

At runtime require all of these proofs:

```text
pix_flash_grad_select_active=1
pix_flash_grad_frac>0
surrogate_decoder_current_teacher=1
surrogate_decoder_state_detached=1
surrogate_staleness_steps=0
surrogate_exact_all_residual=1
```

## Why this is also the fastest-converging/reconverging design

Do not describe this as a better-trained neural student.  The key result is
that sensitive residual backward stages are no longer trained at all:

- their VJPs are analytic and tied directly to the frozen WAN decoder weights;
- they need **zero optimizer updates, zero warm-up, zero replay and zero
  convergence steps**;
- the current discriminator supplies a fresh pixel cotangent every generator
  step, so discriminator/head drift is consumed immediately;
- telemetry should continue to show
  `surrogate_decoder_current_teacher=1`, `surrogate_staleness_steps=0`, and
  detached state;
- because the VJP is structurally linear in cotangent, a changed GAN head does
  not require refitting the decoder transport.

In time-to-quality terms, Q1~0.99 is present on the first evaluated field.  Its
reconvergence time after a changed pixel cotangent is also zero surrogate
updates.  Only a change to the frozen VAE itself would invalidate this claim.

This structural fact is stronger than trying DAgger, RL loss scheduling or
additional local fitting on the old 0.6 system.  Keep those only as historical
ideas/ablations; do not use them as the base for future experiments.

## Cost

Measured combined detached-capture plus reverse costs are:

| Path | Alignment Q1 | Median time | Incremental peak working memory |
|---|---:|---:|---:|
| 0.846 graph-free hybrid | 0.846409 | 145.38 ms | 4.32 GB |
| **0.99 all-residual graph-free** | **0.990999** | **188.29 ms** | **4.09 GB** |
| Full forward + exact autograd VJP | 1.0 | 208.59 ms | 11.67 GB |

The 0.99 reverse part is slower than the approximate reverse, but the complete
graph-free path still beats exact autograd on time and is dramatically lower
memory.  Detached decoding is already needed for current discriminator pixels.

## Your next experiment

Switch the treatment, not merely the label:

1. Freeze the old Q1~0.6 run as a historical control.  Do not cancel a holder
   or delete its artifacts.
2. Use the all-residual bundle above as the sole decoder-pullback baseline for
   new work.
3. Keep the explicit Flash source/query contract above; never infer it from
   `flash_dmd_enabled` alone.
4. First run a **weight-zero production-geometry calibration** and prove from
   resolved logs/counters that the 12 exact stages loaded, the current teacher
   supplied the cotangent, state remained detached, staleness was zero, and all
   intended generator tensors were reached.
5. **Do not reuse the old `PIXW=1.18928` blindly.**  That weight was calibrated
   from the Q1~0.6 field RMS/gradient ratio.  Re-measure field RMS and the
   unweighted generator-gradient share for this new field, then derive a new
   weight.
6. After calibration, run the smallest nonzero smoke with the existing warm-up
   and safety telemetry.  Do not enable the in-trainer exact audit; it previously
   OOMed beside the full DMD graph.  Offline paired/Cartesian gates are the
   accuracy authority.
7. For convergence/reconvergence comparisons, report quality at the first
   field and after teacher/head transitions with **zero surrogate fitting
   updates**.  Compare against the old run only as a control.

Do not change production defaults or launch a nonzero generator weight until
the new weight-zero calibration is inspected.  Preserve the dirty worktree and
unrelated user changes.  Use durable holder command files, never a long-lived
login-shell `srun`, and never cancel a holder itself.

## Evidence and outputs

Main report:

`analysis/gan_tuning/EXACT_RESIDUAL_LADDER_2808.md`

Raw summaries:

`/scratch/u6ex/as1748.u6ex/ARRWM_data/gan_aligned_discrimination/surrogate_field_2708/local_vjp_granular_audit_r0_h6158256/exact_residual_ladder_2808/ladder_summary.json`

`/scratch/u6ex/as1748.u6ex/ARRWM_data/gan_aligned_discrimination/surrogate_field_2708/local_vjp_granular_audit_r0_h6158256/exact_residual_ladder_2808/cost10.json`

Plain instruction: **retire Q1~0.6 as the active experimental base and switch
all further decoder-pullback work to the all-residual Q1~0.99 bundle.**

---
