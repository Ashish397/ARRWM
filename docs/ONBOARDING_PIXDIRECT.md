# ONBOARDING — DINO pixel-feature GAN (pixdirect), 2026-08-26

Written for a fresh agent picking up the texture work. Read this top to bottom
before touching anything. The previous builder ran out of context; nothing here
is inferred, every number below was measured on GPU.

---

## 1. THE GOAL

Shift generated texture toward the **dataset style**. Not sharpness — the
researcher has been explicit twice: *"sharpness can get the thing to go really
blocky and pixelated so don't optimise for sharpness"* and *"it's not just
sharpness, our main objective has to be style shift to the dataset style"*.

`gt_vs_fake` (real GT vs generated, "does this look like real footage?") is the
pair mode that carries texture information. The idea under test: feed it
**DINOv2 features of decoded PIXELS** instead of Wan transformer latent features.

**Why this should work.** The LADD discriminator taps Wan transformer features at
**patch resolution**, so an over-driven disc prints a **16-pixel lattice** of dots
into the decoded output (16 px = one Wan patch). DINOv2 ViT-S/14 has stride
**14**, coprime with 16, so it cannot trivially relearn the same grid.

---

## 2. WHAT IS PROVEN WORKING (do not rebuild)

The DINO feature basis, from a 90-step probe
(`wandb/wandb/run-20260826_125159-mttksk6m`):

| key | value | meaning |
|---|---|---|
| `ladd_pix_src_forwards` | 210 | DINO encoder ran |
| `ladd_pix_images` | 1080 | images scored |
| `ladd_pix_grid_h/w` | 13 / 17 | exactly as predicted for 176x240 |
| `ladd_pix_wan_projector_calls` | **0** | never falls back to Wan taps — the central claim |
| `r3gan_disc_updates_total_gt` | 60 | gt_vs_fake disc really updates |
| `r3gan_d_loss_gt` | 0.511 | disc learning (below ln2=0.693) |

**The direct decode-gradient route is measured and working**
(`pixdirect_frozen`, 32 steps, holder 6135660):

- peak **42.38 GB** at step 31 — below the 51 GB projection and the surrogate's 63 GB
- no OOM, no decode micro-batching needed
- **`gan_dmd_grad_ratio = 0.258`, NON-ZERO** — the generator receives adversarial
  gradient through the VAE decode

> **Counter polarity, because it is inverted between the two routes.** On
> **pixdirect**, `ladd_pix_decode_grad` **must be NON-ZERO** — it is the proof
> that the direct decode-gradient route is live (the completed 200-step arms
> read **204**; `analysis/gan_tuning/PIXDIRECT_200_RESULTS.md` §6). A
> "must be 0" expectation for this key belongs to the **SURROGATE** arm, where
> the generator route is the latent critic. Do not carry it across, and do not
> read a non-zero value here as a failure.

---

## 3. THE SURROGATE — WAS DEAD, FIXED 2026-08-26

> **SUPERSEDED.** The section below stands as written for the record, but
> its verdict ("do not revive") and its root cause (`base_unreachable`)
> are both wrong. The real cause is measured in
> `analysis/gan_tuning/PIXEL_FEATURE_SOURCE.md` §7c "RESOLVED": **the
> latent critic was never trained.**
>
> `_maybe_run_surrogate_distillation` runs only on GENERATOR iterations
> (`self.step % dfake_gen_update_ratio == 0`, =5) and not below
> `gan_disc_start_step=20`, so a 90-step run gives it **14** calls, of
> which `should_refresh` (step % 2) fires on **7** — exactly the
> `surrogate_n_teacher_refresh = 7` that run reported. Fourteen critic
> gradient steps. Held-out gradient cosine vs critic steps, measured on
> CPU: 0.38 at 7, 0.57 at 14, 0.80 at 120, 0.84 at 500 (realistic
> teacher); and −0.003 at 14 rising to 0.71 by 500 on a worst-case
> white-noise teacher. **Production's 0.0097 is just what 14 steps looks
> like.**
>
> Fixed by `surrogate_distill_substeps` (default 1 = byte-identical),
> which takes N critic steps per call against the CACHED teacher targets,
> so `pix_teacher_refresh_every` and the teacher's cost are unchanged.
> Prove it fired from `train/surrogate_substeps_achieved` (must read ~N,
> not 1.0), and judge from `train/surrogate_check_cos_sim`.
> `sbatch/pixdino_frozen.sbatch` / `pixdino_online.sbatch` carry the fix
> and are launchable; they have NOT been submitted.
>
> Note also that §7c's own table reports `surrogate_param_n_reached_sur =
> 825` — "not severed, not off-path" — which directly contradicts the
> `base_unreachable` claim below. Treat the parameter-space probe as the
> stronger reading, and see the falsifiable prediction in §7c.

### Superseded text, kept verbatim


**The surrogate route delivers EXACTLY ZERO gradient to the generator.**
`surrogate_param_grad_norm_unweighted = 0` across **825/825** reachable params,
while DMD's own base grad norm was 9.53. Gradient cosine 0.0097 (healthy ~0.986).

Root cause: reason key `base_unreachable`. `pix_fake_source_ladder = 0` means the
surrogate's fake is `flash_dmd_gan_x0`, a **different sub-graph** from the tensor
DMD scores, so `generator_loss` has no path to it. This is **One-Forcing
divergence 3, third occurrence**. Recorded in
`analysis/gan_tuning/PIXEL_FEATURE_SOURCE.md` §7c.

Had we launched, every curve would have looked plausible while the generator
trained completely unchanged.

Note the researcher originally said: *"check if you have enough memory for
gradients and if not just use the surrogate gradient method"*. Direct was always
first choice; memory allows it (42 GB of 95). So direct is correct, not a
deviation.

---

## 4. THE LIVE BLOCKER — your first job

**`pixdirect_online` will crash on DDP.** Confirmed on the surrogate online arm
(`logs/holdersmoke_pixdino_online_h6136513.log`, step 16):

```
RuntimeError: Expected to have finished reduction in the prior iteration...
Parameter indices which did not receive grad for rank 0: 2
```

### FIXED 2026-08-26 — and the diagnosis below was WRONG. Read this first.

**The superseded diagnosis was:** "DINO taps blocks `[0,1,2,3]`
(`ladd_pixel_dino_n_taps=4`) but the whole 22.06M ViT-S — 12 blocks — is wrapped
trainable. Blocks 4-11 never receive gradient, so DDP's reducer aborts.
Preferred fix: truncate the ViT after the last tapped block."

**Why it was wrong.** `[0,1,2,3]` is the list of LADD **tap indices**, which is
what the `[LADD-PIXFEAT]` boot line prints as `taps=`. It is NOT the list of ViT
**block** indices. `_build_dinov2_trunk` spaces the taps evenly over the depth,
so at `n_taps=4` on 12 blocks they resolve to **`[2, 5, 8, 11]`**. Block 11 *is*
tapped, every one of the 12 blocks is on the backward path, and truncation
therefore drops **zero** blocks and could not have been the fix.

**The measured cause.** A forward+backward through `get_intermediate_layers`
leaves exactly **one** parameter with `grad is None`: DINOv2's `mask_token`.
`prepare_tokens_with_masks` reads it only under `if masks is not None`, and this
call site never passes masks, so it is unreachable by construction. DINOv2
declares `cls_token`, `pos_embed`, `mask_token`, so with `pixel_source`
registered first on `LADDDiscriminator`, `mask_token` sits at reducer index
**2** — exactly the index the traceback names.

Confirmed both ways on CPU: undoing the freeze reproduces
`did not receive grad for rank 0: 2` verbatim at DDP iteration 2; with the
freeze, 3 iterations pass at `find_unused_parameters=False`.

**The fix** (`model/ladd_pixel_features.py::_freeze_ddp_unreachable`) discovers
the unreachable set from a real build-time backward and freezes it — discovered,
never hard-coded by name, so an upstream rename fails a test instead of silently
reinstating the crash. It is gated on `encoder_trainable`, so the frozen arm does
not even run the probe. Truncation was implemented too (it is the right fix for a
shallow explicit `ladd_pixel_dino_layers`) but is **inert at `n_taps=4`** and
reports `blocks_dropped=0` rather than pretending otherwise.

**Proof-of-fire keys** (`_ladd_pixel_logs`, wandb only):
`ladd_pix_encoder_unreachable_frozen` = 1.0 on a trainable-encoder arm, 0.0 on
the frozen arm; plus `ladd_pix_encoder_blocks_kept` / `_dropped`.

Frozen-arm invariance was **verified, not assumed**: all 4 tap tensors sha256-
identical pre/post fix in both arms, grid `(13,17)` unchanged,
`trainable_params` 0 → 0 frozen and 22,056,576 → 22,056,192 online (the delta is
exactly `mask_token.numel()` = 384). Regression tests in
`testing/test_ladd_pixel_feature_source.py`.

> **CORRECTED 2026-08-26 — the runtime contradicts this, and it leaves the
> original crash unexplained.** ~~`find_unused_parameters=True` was not
> used.~~ **Both** 200-step arms emit *"find_unused_parameters=True was
> specified in DDP constructor, but did not find any unused parameters in the
> forward pass"*, and it appears on the **FROZEN** arm too, so it is a
> pre-existing trainer-wide setting rather than anything the pixel work
> introduced. This does not invalidate the `mask_token` freeze or any
> verification above. But if `find_unused_parameters` was already on, the
> original reducer-index-2 crash **should not have occurred** — so that crash
> is **NOT fully explained**. Recorded as an **OPEN QUESTION**, no resolution
> asserted:  `analysis/gan_tuning/PIXDIRECT_200_RESULTS.md` §7b.

---

## 5. HOLDERS AND WHAT IS ON THEM RIGHT NOW

Pattern: write `logs/.holder_cmd_<jobid>.sh`; the holder polls every 10s and runs
it AS the batch job. **Never `srun` directly from an agent shell — it gets
SIGTERM'd when the turn ends.** Never `scancel` the holder itself, only its step.

| holder | time left | running |
|---|---|---|
| **6140644** | ~5h10 | `pixdirect_frozen` — the real arm, direct gradient |
| **6136514** | ~1h29 | `pixdirect_frozen` **A/A replicate** (different node = the replicate this campaign has never had). **NOT "seed 2"** — both arms are `seed=1234`, hardcoded at `sbatch/pixdirect_frozen.sbatch:281` with no override. Still the correct noise estimate (the §7 CVs were measured exactly this way), but the seed label is wrong and must not propagate: `PIXDIRECT_200_RESULTS.md` §7a. |
| **6140643** | ~5h07 | **FREE** — reserved for `pixdirect_online` once you fix DDP |
| 6136513 | ~1h25 | FREE |
| 6135661 | ~29m | FREE (too short for a 200-step arm) |
| 6135660 | ~8m | expiring |
| 6143212/13 | PENDING | queued |

**Launch through `sbatch/run_smoke_on_holder.sh`, NOT
`sbatch/run_pixdino_smoke.sh`.** The latter hardcodes
`DEXTRA="... sample_interval=1000 ..."` which suppresses all video. The normal
runner sets no DEXTRA, inherits `sample_interval=15`, and is **proven** to emit
`pred_image_rollout` (verified at step 16 on 6136514).

```bash
cat > logs/.holder_cmd_<HOLDER>.sh <<'EOF'
#!/bin/bash
cd /scratch/u6ex/as1748.u6ex/ARRWM
export PIXW=0.0
HOLDER=<HOLDER> SMOKE=pixdirect_online PORTOFF=<unique> bash sbatch/run_smoke_on_holder.sh
EOF
chmod +x logs/.holder_cmd_<HOLDER>.sh
```

`PORTOFF` must be unique per concurrent run or two jobs collide on one rendezvous.

---

## 6. CALIBRATION — the researcher has already ruled

`gan_dmd_grad_ratio = 0.258` sits above the 0.05-0.20 target band, and the
previous builder recommended cutting `gan_loss_weight` 1.5-3x. **Do not.**

1. The researcher's verdict today: **"w003 is bad, w1 is better"**. `gan_loss_weight`
   stays at **1.0**.
2. `gan_loss_weight` scales **only the G side**. Cutting it leaves `gan_lr`,
   `gan_updates_per_step=5` and `r1_gamma` at weight-1.0 values, converting an
   over-driven-G problem into a **D-wins** problem — measured on `w003`:
   `d_loss <= 0.135` for five consecutive logged steps against a 0.25-0.55 band.
3. `gan_dmd_grad_ratio` has a **50% noise floor** and 0.258 is n=1, so it is not
   reliably out of band at all.

If over-drive genuinely needs fixing later, move `gan_lr` / `gan_updates_per_step`,
or reinstate the removed `gan_grad_target_norm` controller — something that scales
**both** sides.

---

## 7. STATISTICAL RULES — these are hard, and violating them has produced 8 documented false conclusions

Measured run-to-run coefficient of variation for **identical configs on different
nodes** (no determinism enforcement, TF32 on, different NCCL reduction order;
amplifies ~250x over 90 steps):

| key | CV |
|---|---|
| `d_real` / `gan_cos` | **95-119%** |
| `r1` | **76%** |
| `gan_dmd_grad_ratio` | **50%** |
| `gen_loss` / `d_loss` | 14-17% |
| `roll_mae` / `dmd_err` / `gt_dist` | 3.5-5.3% |

- **Never** draw an n=1 conclusion on ratio, r1, d_real or cos.
- Use **medians past step 50**, never a single step, never the final step alone.
- Report n and spread for every number. A bare point estimate is not acceptable.
- Texture metrics must be computed on **TEXTURED crops** (tree crown, road) — NOT
  sky. Sky is flat and past analysis was misled exactly this way.
- Texture references: `fold2d` P=16 dotfrac GT = **0.13**; mod-8 row fold A8y
  GT = **1.42**. Two-sided — undershooting (too smooth) is as wrong as
  overshooting (blocky).

---

## 8. DOCUMENTATION MAP

**Read first**
- `docs/GAN_REDESIGN_2608.md` — **the 2026-08-26 handover**: the arc of the
  day in the order it happened, the completed six-arm `d_loss` table
  (`pixdirect_onlinelr` is the best healthy arm at 0.4560), the **D-WINS
  COLLAPSE** that hid inside `pixdirect_strong`'s healthy-looking 0.3632
  median, the surrogate route's **falsification** at `substeps=24` (do not
  launch surrogate arms), the correction to §2's "42.38 GB", the binding
  researcher rulings, the proof-of-fire counter table for the new pooled
  VGG/rn50 arms, the holder-collision operational lessons, and the ordered
  WHAT TO DO NEXT
- `analysis/gan_tuning/TEXTURE_BASIS_BENCHMARK.md` — **the frozen-feature
  texture-sensitivity benchmark (2026-08-26, holders 6144621 / 6144623)**:
  the offline backbone inventory (EfficientNet, kymatio and P-S are NOT
  available; VGG16 and DINOv2 are), the two-sided U-shape curves against a
  crop-translation nuisance control, the ranked basis table, the pixel-
  gradient conditioning numbers, and the answer to "did DINOv2 discard the
  texture cue or was the head badly optimised" (**neither** — DINO probes at
  AUC 0.927 but its fused 4-tap representation moves *more* under an 8 px
  crop shift than under any texture perturbation, and its pixel gradient is
  only 0.39 shift-consistent against VGG relu1_2's 0.884). Carries the
  **GO** decision and the three specifications for the VGG arm
- `analysis/gan_tuning/DRIFT_SEPARABILITY.md` — **the follow-up that tests
  whether that 0.927 is accumulated rollout drift (2026-08-26, CPU only)**:
  AUC vs rollout chunk index is FLAT at ~1.000 from chunk 0, so drift is
  NOT the separability source and "the disc only sees one chunk" is a dead
  explanation for the near-chance `d_loss`. It also documents a confound in
  the U-test itself — the two halves of `pred_image_7_chunk` are DIFFERENT
  84-frame windows of the ride (GT = the context, student = the
  continuation), and two disjoint windows of *undisturbed real footage*
  separate at AUC 0.86-1.00 on the same probe
- `analysis/gan_tuning/PIXDIRECT_200_RESULTS.md` — **the 200-step results for
  the two frozen arms + online**: the lattice reduction that cleared its own
  A/A floor (§1), the near-chance DINO disc and why ONLINE proves it (§2), the
  quantified adversarial dose (§3), two falsified hypotheses (§4), the
  telemetry blackout (§5), completed-run counters (§6), and the corrections to
  *this* file (§7)
- `analysis/gan_tuning/PIXEL_FEATURE_SOURCE.md` — this workstream; §7b telemetry
  defects, §7c the zero-gradient root cause
- `analysis/gan_tuning/ARMS_200_REVIEW.md` — latest arm results (w003, carn_fwd,
  carn_ctrl); the `carn_ctrl` inert-noiser finding
- `analysis/sharpness/TEXTURE_REVIEW.md` — texture metric definitions
- `CAMPAIGN_STATE.md` — overall campaign synthesis

**Researcher-facing (protocol: put questions here, they answer inline)**
- `COMMENTS_FOR_USER.md` — live questions needing sign-off
- `RESOLVED_TODOs.md` — answered items, moved out of COMMENTS
- `docs/THINGS_TO_DO.md`, `docs/THINGS_STILL_TO_DO.md`

**Background**
- `docs/ONE_FORCING_PORT.md` — the 5 divergences; divergence 3 is what killed the surrogate
- `docs/WP_SURROGATE.md`, `docs/WP_PIXGAN.md` — the two prior pixel routes
- `analysis/gan_tuning/CARN_TRANSITION_REVIEW.md` — CARN/transition-GAN coupling
- `.claude/dmd_gan_stage_reference.md` — DMD/GAN stage architecture
- `docs/HOLDER_GRID.md` — holder mechanics

---

## 9. STANDING HAZARDS

- **`OmegaConf.from_dotlist` creates unknown keys silently.** Two of the previous
  builder's three `DEXTRA` keys did not exist and were accepted without a murmur
  (`mem_snapshot_every`, `holdout_eval_interval`). Verify every override key has a
  read site.
- **Overrides are last-wins.** Arms set the same flag twice deliberately (baseline
  block, then the arm's block). Always resolve the LAST value before believing a config.
- **Flags that silently do nothing.** 21 arms once trained a noiser that never
  touched data; `slide12` was an A/A replicate because its flag was inert under
  `chain_levels`; `carn_ctrl` trained a CARN nothing reads. **Prove a flag fired
  from a counter, never from the patch.**
- **wandb-only keys.** `train/ladd_pix_*` are emitted at
  `trainer/causal_action_forcing_train.py:7073-7075` and go to wandb, NOT the log.
  `sbatch/check_pixdino_smoke.sh` greps the log for them and reports false FAILs —
  it once called jitter "phase LOCKED" when jitter was moving. Read
  `wandb/wandb/<run>/files/wandb-summary.json`.
- **Holder LR0 OOM launch race** — ~50 GB unaccounted on GPU0 on the second node,
  ~60-90s in. Not config. Retry up to 3x with a fresh PORTOFF.
- **Never noise the clean context**; **never** `dmd_ar_head_weight=1.0` (TF head
  always); **real_guidance_scale=0.0** always.

---

## 10. DEFINITION OF DONE

Two arms, single-variable (frozen vs online DINO backbone at 0.1x lr), direct
decode gradient, `gan_loss_weight=1.0`, **emitting rollout videos**, run long
enough to judge (researcher wants ~2 hours), with texture measured two-sided on
textured crops against GT 0.13 / 1.42 — and **replicated**, because n=1 cannot
clear the noise floors in §7.
