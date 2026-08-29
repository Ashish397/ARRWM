# Sol onboarding — definitive stationary Phase 3

Status timestamp: 28 August 2026, approximately 21:10 UTC.

Repository:

`/lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM`

You are taking over a project whose strongest components have already been
found and integrated. Do not restart the search from the older Q1~0.6 GAN or
the historical CARN permutations. The current job is to finish and evaluate
the two definitive stationary trials, preserve their checkpoints, and prepare
the later rolling phase only after the stationary result is reviewed.

## Read first, in this order

Read each file completely before editing or launching work:

1. `docs/PHASE3_STATIONARY_DEFINITIVE_2808.md` — exact promoted recipe,
   current trial and no-GAN control.
2. `docs/DEFINITIVE_CARN_GAN_SURROGATE_KNOWLEDGE_TRANSFER_2808.md` — why the
   CARN, harmony bridge, GAN, surrogate, stat anchor, brightness policy and
   action critic are shaped this way.
3. `docs/ONBOARDING_DECODER_PULLBACK_099.md` — complete Q1~0.99/live-GAN
   history, invalidations and required runtime proofs.
4. `analysis/gan_tuning/EXACT_RESIDUAL_LADDER_2808.md` — measured Q1 ladder,
   artifact, time and memory.
5. `analysis/gan_tuning/CARN_HARMONY_2808.md` — coherent F/G geometry,
   normalized bridge and calibration.
6. `analysis/gan_tuning/FLASH_BRIGHTNESS_WAVELET_2808.md` — brightness,
   discarded wavelet route and the accepted `raw_grad_hp` treatment.
7. `analysis/gan_tuning/CARN_BASE_DECODER_SURROGATE_2808.md` and section 7 of
   `docs/CARN_VX_2708.md` — historical CARN/GAN evidence.

Treat older reports as history when a newer report explicitly invalidates
their live-GAN results. The offline Q1 ladder remains valid.

## What is currently considered best

The word “best” has three evidence levels here:

- The Q1~0.99 decoder pullback is a measured offline result on paired and
  Cartesian gates.
- The harmony bridge and `raw_grad_hp` brightness treatment passed targeted
  calibrations and visual mechanism screens.
- Their complete integration into a stationary 200-step Phase 3 is the live
  definitive trial. Do not call that integrated recipe fully validated until
  its step-100/200 checkpoints and videos have been reviewed.

The parent selected by the researcher is W&B `htfhu37j`. The definitive
treatment adds:

- coherent CARN harmony bridge;
- aux-minus/internalisation weight `.25`;
- recurrent commit alpha `.25`, off through step 100 and ramped over the next
  100 steps;
- DC-off/raw-forward brightness filter `raw_grad_hp`;
- stat anchor `1.0`;
- stationary first-seven-chunk training for 200 steps;
- full resumable checkpoints at steps 100 and 200.

It starts from the Phase 2 KL-ODE checkpoint:

`/scratch/u6ex/as1748.u6ex/ARRWM/logs/ode14e_pilot/run3_flip2_rollkl10k/action_ode_step0000400.pt`

## Live trials: do not restart or cancel them

| holder | child step / node | W&B | treatment |
|---|---|---|---|
| `6170183` | `6170183.15` / `nid010221` | `kv9axmpu` | definitive Flash GAN + Q1~0.99 pullback + harmony + DC-off brightness |
| `6170182` | `6170182.6` / `nid010187` | `yi4iwouc` | matched stationary no-GAN ablation |

Both holder allocations end at `2026-08-28T22:39:18` UTC. At the timestamp
above the GAN trial was around step 45 and the no-GAN trial around step 10.
Refresh rather than trusting those step numbers:

```bash
squeue -j 6170182,6170183 -o '%.18i %.9T %.10M %.10l %.24R'
squeue --steps -j 6170182,6170183 -o '%.20i %.10M %.24N'
```

The two reserve two-node/six-hour holders `6175078` and `6175087` are queued.
They were pending on priority at handoff.

### Definitive GAN trial

Launcher:

`sbatch/run_phase3_stationary_definitive_node.sh`

Local log:

`logs/carn_surrogate_q1p99_sharedefinitive_htfhu37j_harmony_dcoff_h6170183_nid010221_2808phase3stationarydefinitive_h6170183.log`

W&B:

`https://wandb.ai/ashish397-university-of-exeter/longlive-phase-3/runs/kv9axmpu`

Its resolved config must keep all of these true:

```text
gan_enabled=true
flash_dmd_enabled=true
flash_dmd_gan_t=60
ladd_fake_sample_source=flash
pix_flash_grad_select_enabled=true
ladd_pixel_decode_cache=false
ladd_defer_disc_update=false
surrogate_decoder_fresh_disc_order=true
surrogate_decoder_shaped_enabled=true
```

It uses `PIXW=.0175`, D LR `1e-3`, U1, R1 `1`, the frozen action critic at
guidance `.3`, and the immutable all-residual pullback bundle.

### Matched no-GAN trial

Launcher:

`sbatch/run_phase3_stationary_definitive_nogan_node.sh`

Local log:

`logs/carn_surrogate_q1p99_sharedefinitive_htfhu37j_harmony_dcoff_nogan_h6170182_nid010187_2808phase3stationarydefinitive_nogan_h6170182.log`

W&B:

`https://wandb.ai/ashish397-university-of-exeter/longlive-phase-3/runs/yi4iwouc`

The log already proves the adversarial path is genuinely absent:

```text
gan_enabled=false
gan_loss_weight=0.0
surrogate_decoder_shaped_enabled=false
surrogate_decoder_fresh_disc_order=false
pix_gan_weight=0.0
ladd_gt_vs_fake_enabled=false
ladd_gt_transition_enabled=false
ladd_adjacent_chunks_enabled=false
```

Flash DMD, stat anchor, CARN F/G cycle, aux/internalisation, staged commit,
stationary geometry and checkpoint schedule remain. The discriminator-side
harmony transition objective is necessarily inert in this no-GAN control;
that is expected, not a wiring failure.

## Stationary and checkpoint contract

Both runs resolve:

```text
max_steps=200
num_training_frames=21
rollout_frames=21
streaming_chunk_size=18
max_rolls_per_ride=1
dmd_only_first_chunk_per_ride=true
dmd_42f_rolling_sup_new=false
dmd_42f_allroll_student_ctx=false
num_chunks_roll_forward=0
sample_7chunk_enabled=true
sample_7chunk_alias_rollout=true
checkpoint_interval=100
save_full_checkpoint=true
keep_last_n_checkpoints=4
```

The stationary geometry has no native multi-roll accumulator because
`num_chunks_roll_forward=0`. Its long sample is the independent 168-frame
`pred_image_7_chunk` evaluation (seven GT context chunks plus seven generated
chunks). The alias flag publishes the same bytes under
`sample/pred_image_rollout` and logs
`sample/pred_image_rollout_alias_from_7chunk=1`; do not mislabel it as a
rolling-phase accumulator. The live runs were backfilled under that exact key
at source steps 76 (`kv9axmpu`) and 46 (`yi4iwouc`) without a restart.

The required checkpoint filenames beneath each resolved `log_dir/run_name`
are:

```text
phase1_step0000100.pt
phase1_step0000200.pt
```

Do not merely see a filename and declare success. Verify each file is nonzero,
can be opened with `torch.load(..., map_location='cpu')`, contains `step` equal
to 100/200, and retains generator, optimizer, GAN state where applicable,
CARN forward/reverse state and EMA state. The no-GAN checkpoint should not be
expected to contain a discriminator.

## Immediate responsibilities

1. Monitor both child steps without modifying them. Look for NaN/OOM/error,
   advancing `mem-end step`, and W&B/video cadence.
2. Confirm `pred_image_7_chunk` and its provenance-marked stationary
   `pred_image_rollout` alias continue to upload on future launches. The two
   already-running processes only have the verified backfilled alias because
   they loaded the old trainer before the repair.
3. At step 100, verify and inventory both full checkpoints before allowing the
   holder time to be repurposed.
4. At step 200, verify the final checkpoint and record exact file size/SHA.
5. Compare GAN versus no-GAN at matched steps, with special attention to
   texture, bright patches, motion around the seed-context boundary, action
   adherence and reconvergence after a stall.
6. Report the integrated trial honestly. The no-GAN arm is a causal control,
   not an expected production winner.
7. Do not begin rolling Phase 4 until the stationary step-200 output is
   reviewed and the researcher authorizes continuation.

## Things that must not be revived

- Q1~0.6 as an active baseline; it is historical only.
- Wavelet/SWT as the production texture route; the user discarded it.
- The standalone `former_plus` treatment; it is an unfinished ablation and
  previously accumulated lattice/chromatic degradation.
- Literal `latter_minus` as a default.
- Pooled-VGG action conditioning; it learned real/fake texture while ignoring
  action compatibility.
- Stronger frozen action guidance `.6` or the online 59.3M action critic as a
  production default. Their visual screens appeared to break trajectory/model
  behaviour. The retained setting is the previously trained frozen critic at
  `.3`.
- Raw-image nearest matching; it reinstates exposure as an easy cue.
- Pixel decode caching in this micro-group geometry.
- A deferred D update before the Q1 field.
- In-trainer exact decoder audits beside the full graph; this previously OOMed.

## Operating rules

- The worktree is intentionally dirty and contains user/other-agent changes.
  Never reset or discard unrelated edits.
- Holder jobs are allocations, not disposable child runs. Never `scancel` a
  holder unless the user explicitly says to cancel the allocation.
- Launch durable work by writing `logs/.holder_cmd_<jobid>.sh`; the holder
  batch process owns the resulting `srun`. Do not leave a login-shell `srun`
  as the load-bearing client.
- Stop only a child step when correction is necessary.
- OmegaConf overrides are last-wins and unknown keys can be accepted silently.
  Inspect the resolved config and runtime counters, not the launcher text.
- Preserve W&B IDs, logs, videos and checkpoints even for negative controls.
- Distinguish measured results, visual researcher rulings and hypotheses in
  every report.

## Immutable Q1 artifact

```text
/scratch/u6ex/as1748.u6ex/ARRWM_data/gan_aligned_discrimination/surrogate_field_2708/local_vjp_granular_audit_r0_h6158256/exact_residual_ladder_2808/decoder_shaped_all_residual_seed0.pt
```

SHA-256:

`de82ec3211ec408f34cca883be53c15d835e19d5a3860a0b7fe8840dad450357`

Required exact residual stages:

`[1,2,3,5,6,7,9,10,11,13,14,15]`

The measured gates are paired worst-seed Q1 `0.990999` and Cartesian
worst-seed Q1 `0.992351`. Only stage 0 (`low_core`) remains learned.
