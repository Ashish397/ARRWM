# Definitive stationary Phase 3 recipe — 28 August 2026

Incoming agents should begin with
`docs/ONBOARDING_SOL_PHASE3_DEFINITIVE_2808.md`; the architectural explanation
is in `docs/DEFINITIVE_CARN_GAN_SURROGATE_KNOWLEDGE_TRANSFER_2808.md`.

This phase starts from the KL-ODE Phase 2 step-400 checkpoint:

`/scratch/u6ex/as1748.u6ex/ARRWM/logs/ode14e_pilot/run3_flip2_rollkl10k/action_ode_step0000400.pt`

The promoted 200-step treatment is based on W&B `htfhu37j`, with the harmony
bridge and the DC-off brightness policy added. It trains only the fixed first
seven chunks (21 latent frames): one seed chunk followed by six autoregressive
chunks. Rolling-window training is a later phase, not part of this run.

## Definitive treatment

- Launcher: `sbatch/run_phase3_stationary_definitive_node.sh`
- Live child step: `6170183.15` on `nid010221`
- W&B: `kv9axmpu`
- Steps: 200
- Full checkpoints: steps 100 and 200
- Checkpoint retention: four files
- Samples: every 15 steps, plus the first-step seven-chunk video
- GAN: Flash-60 source/query with the measured Q1~0.99 decoder pullback
- GAN weight / D LR / D updates / R1: `.0175 / 1e-3 / 1 / 1`
- Brightness: `ladd_pixel_input_filter=raw_grad_hp`, SWT off, stat anchor 1.0
- CARN: harmony bridge, aux/internalisation `.25`, commit alpha `.25`, commit
  starts at step 100 and ramps for 100 steps
- Frozen action guidance: `.3`

The stationary geometry is pinned by:

```text
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
```

### Stationary rollout video contract

Stationary mode deliberately sets `num_chunks_roll_forward=0` and
`max_rolls_per_ride=1`. The trainer's native rolling accumulator is therefore
not populated and cannot emit its usual `sample/pred_image_rollout` key. This
does **not** mean that the long evaluation failed: `pred_image_7_chunk` is a
168-frame evaluation containing seven ground-truth context chunks followed by
seven generated chunks.

For stationary launches, `sample_7chunk_alias_rollout=true` publishes that
same encoded 7+7 clip under both `sample/pred_image_7_chunk` and the expected
`sample/pred_image_rollout` dashboard key. The run also records
`sample/pred_image_rollout_alias_from_7chunk=1` and the source step, so it
cannot be confused with a native rolling-phase accumulator.

The two already-running definitive trials loaded the trainer before this fix.
Their latest clips were backfilled without restarting training:

- `kv9axmpu`: 168-frame source step 76, committed at W&B history step 77.
- `yi4iwouc`: 168-frame source step 46, committed at W&B history step 52.

Both API records remained `running` after the backfill.

`raw_grad_hp` exposes raw RGB, including its evolving mean, to the
discriminator while suppressing broad/DC generator-side brightness gradients.
It is the promoted "brightness DC off" treatment; the stat anchor remains on
to protect global statistics without freezing the mean.

Checkpoint files are written beneath the run's resolved `log_dir/run_name` as:

```text
phase1_step0000100.pt
phase1_step0000200.pt
```

## Matched no-GAN ablation

- Launcher: `sbatch/run_phase3_stationary_definitive_nogan_node.sh`
- Live child step: `6170182.6` on `nid010187`
- W&B: `yi4iwouc`

This is matched to the definitive treatment except that the adversarial path
is removed with last-wins settings:

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

Flash DMD, the CARN F/G cycle, aux/internalisation, staged commit, stat anchor,
stationary geometry, samples and checkpoint schedule remain enabled. Because
the harmony transition bridge is a discriminator-side objective, it is
necessarily inert in this no-GAN control; the underlying CARN cycle and shared
reverse operator remain trained.

## Submitted reserve holders

Two additional two-node, six-hour holders were submitted as jobs `6175078`
and `6175087`.
