# DMD and GAN Session Log: 2026-08-22

## Objective

Turn the strong stationary KL-DMD result into a stable causal rolling model while
preserving the existing causal student. The target is a useful 60-second rollout,
using the already bidirectionally trained 14e model for DMD scoring and avoiding
student retraining.

The intended contributions are retained:

- Ground-truth feeding and temporal clean matching for DMD supervision.
- CARN correction of generated context before it is committed to the rollout.
- A transition discriminator, including wavelet and projected variants.
- Ground-truth transition matching for the discriminator.

## Main Findings

### The 4.5-second freeze is an attention-horizon boundary

The repeated freeze and jitter in `rollcarn700*` and `ganmvp700*` occurs at almost
exactly the point where the original seed leaves the student's local attention
window:

- Total local attention window: 21 latent frames.
- Current generated chunk inside that window: 3 latent frames.
- Remaining past-context capacity: 18 latent frames.
- Wan VAE temporal stride: 4 RGB frames per latent frame.
- Logged sample rate: 16 RGB frames per second.
- Past-context horizon: `(21 - 3) * 4 / 16 = 4.5 seconds`.

The attention window is still sliding correctly. Its 21-frame span includes the
three-frame query chunk, so only the previous 18 latent frames are available as
context. After this boundary, every past context frame is generated. The inherited
step-700 model enters a low-motion attractor and jitters around nearly the same
state. This is a much stronger explanation than clean matching "running out."
Clean matching is recalculated for every training window and is not active during
inference, so it has no fixed wall-clock expiry.

The training KV buffer is physically deeper at 33 frames, but attention is sliced
to the latest 21 frames throughout. Physical FIFO eviction begins later than the
functional attention-window transition and is not the cause of the 4.5-second
boundary.

There is a pre-existing standalone-evaluator mismatch to correct separately. With
`cache_chunks=7` and a three-frame incoming chunk, `eval_causal_AR.py` allocates a
24-frame buffer and currently also assigns that buffer size to
`max_attention_size`, despite reporting `local_attn_size=21`. The Infinity-RoPE
path slices on `max_attention_size`, so those evaluations effectively attend 24
frames, while training attends 21. This can move the all-generated-context
transition from 4.5 to 5.25 seconds. It was not introduced by this session's CARN,
DMD, or GAN changes.

Clean matching can still bias learned speed if it repeatedly chooses a shifted
target. Several old step-700 runs did select offsets around +6 and at the search
boundary, so it remains a possible contributor. It does not explain the precise,
shared 4.5-second transition as well as seed eviction does.

### The step-700 experiments were confounded

The `rollcarn700*` and `ganmvp700*` arms inherited the same model and its rolling
attractor. They tested whether a short continuation could repair an existing
failure, not whether CARN or the GAN could prevent it from fresh initialization.
Those jobs were stopped after this distinction was clarified. New CARN/GAN arms
start at global step zero from the good KL ODE checkpoint.

### CARN helps, but must also be applied at inference

Fresh `dmd10k_fullcarn_bidir_kl_nogan200` training remained stable and produced
substantially more useful motion than the earlier dark-collapse line. Its old
61.8-second evaluation nevertheless accumulated texture, then geometric, errors.

That evaluation used CARN-trained weights but did not apply CARN to generated
chunks at inference. The evaluator now applies the same seam-affine correction as
training, using prefill latent statistics and lambda 0.5 before each generated
chunk is emitted and committed. A matched 60-second CARN-on evaluation is queued.

The current evidence therefore supports CARN as useful, but the prior long video
was not a definitive test of the complete CARN method.

## DMD Repairs

### Correct 14e ODE schedule

The evaluator and ODE regression code had stale assumptions from a 48-step
schedule. They now use the checkpoint-specific 14e KL schedule:

- `N = 20`
- Snapshot indices `[0, 15, 18, 19, -1]`
- Training indices `[0, 15, 18, 19]`
- Corresponding times approximately `[1000, 625, 357.14, 208.33]`

Invalid or incompatible schedule indices now fail explicitly instead of silently
selecting the wrong denoising states.

### Correct causal DMD normalization

For a causal score head, the DMD normalization denominator now uses the causal
residual on the supervised band:

`mean(abs(student_band - real_score_ar_band))`

The old path could normalize a causal gradient using the privileged
teacher-forced residual. Both denominators and floor rates are logged, and the
legacy behavior remains selectable for controlled comparisons.

The `arlocal300` logs confirm that the repair is active: its median causal
denominator was about 0.098 while the teacher-forced denominator was about 0.192.
Using the old denominator would have reduced the causal update by roughly half in
that run.

### Current DMD recipe

- Fresh initialization from the good KL ODE step-400 checkpoint.
- No automatic inheritance from a previous DMD campaign.
- Bidirectional 14e real and fake score heads.
- KL/DMD3-style objective, not MSE.
- Continuous DMD times from 20 to 980.
- Causal generator rollout with random depth from 2 to 6.
- Ground-truth feeding, clean matching, and drift correction.
- CARN seam-affine correction with lambda 0.5.
- Minimal evaluation checkpoints to avoid repeated 17 GB full-state writes.

### DMD verdict

The schedule and normalization repairs work mechanically: unit tests pass, the
new quantities appear in W&B, and fresh training is stable. Behaviorally, the
combined fresh recipe delays the old catastrophic collapse and preserves useful
motion longer. It is not yet a solved 60-second rollout; texture feedback remains
the first major failure in the no-GAN control.

## GAN Repairs

### Discriminator contract

The discriminator can now return one scalar logit per sample. Generator loss,
discriminator loss, R1, and R2 therefore optimize the same scalar quantity, and
the adversarial scale no longer changes with feature-map resolution.

The pretrained projector mixing layers are frozen while discriminator heads are
trained. This prevents the critic from cheaply rewriting the teacher feature
basis that supplies its perceptual signal.

### Transition construction

- Real and fake transitions use the same sampled diffusion time.
- Wavelet mode takes precedence and compares clean `t=0` transitions.
- Mean equalization preserves the transition delta instead of independently
  recoloring both endpoints.
- Current wavelet runs do not perform per-channel standard-deviation matching.
- Ground-truth transitions can be temporally clean-matched to generated motion.
- Action-blind and strict action-conditioned transition critics are both
  available.
- Alternating R1/R2 regularization is applied without changing the scalar-logit
  contract.

### GAN verdict

The repaired GAN works mechanically: gradients flow through the intended heads,
losses are finite, and `ganfix_blind` learned a modest real/fake separation. This
does not yet establish that the GAN improves long rollouts.

In `ganfix_blind`, the weighted generator GAN term was only about 5-6 percent of
the median DMD term. In `fullkl_bidir_ganxeq`, it was about 2 percent and the
critic stayed close to random-chance loss. The latter GAN was effectively inert.
The current wavelet arm is active. Its step-91 rollout keeps moving for the full
nine-second clip, unlike the old frozen-frame attractor, but texture and geometry
start warping after the approximately 4.5-second seed horizon. The implementation
repairs are real; the scientific objective has not yet passed the rollout test.

## Four-Run Comparison

### `dmd10k_ganfix_blind` (`zdyrlib9`)

- Fresh causal run, CARN 0.5, corrected causal normalization.
- GAN weight 0.03, five discriminator updates, action-blind transition critic.
- Median DMD loss about 0.422; late value about 0.504.
- Median rollout-window MAE about 0.466; late value about 0.468.
- Discriminator loss moved from about 0.652 to about 0.542.
- Real/fake logits developed a modest gap.
- Median weighted generator GAN term about 0.024 versus DMD about 0.422.

The GAN definitely executed and learned a discriminator signal. Its generator
pressure was small, and an action-blind critic can accept visually plausible but
motion-inappropriate transitions. Visual similarity to `arlocal300` is expected
because both share the same fresh causal DMD/CARN base.

### `dmd10k_fullkl_bidir_ganxeq` (`32tvstpn`)

- Bidirectional DMD scoring, CARN 0.5.
- GAN weight 0.01, one discriminator update, action blind, mean equalization.
- Median discriminator loss about 0.689, close to random-chance 0.693.
- Real and fake logits remained very close.
- Median weighted generator GAN term about 0.007 versus DMD about 0.349.
- Late DMD loss rose to about 1.408; late rollout-window MAE about 0.527.

This GAN path ran, but the critic learned almost no useful distinction. It should
be described as mechanically active and scientifically near-inert. The visual
difference from the no-GAN run is not evidence that it corrected texture.

### `dmd10k_fullkl_bidir_nogan` (`qylm9ft4`)

- Closest no-GAN comparison to `ganxeq`.
- Median DMD loss about 0.525; late value about 0.813.
- Median rollout-window MAE about 0.463; late value about 0.449.
- Visually better than `ganxeq`, consistent with its better late rollout MAE.

The no-GAN result is currently stronger. Because these campaigns are separate
stochastic trajectories and inherited their own lineages, this is strong warning
evidence rather than a perfectly paired ablation.

### `dmd10k_arlocal300` (`1hh870sx`)

- Fresh causal DMD/CARN run without a GAN.
- Median DMD loss about 0.383, but late value about 1.102 and occasional spikes.
- Median rollout-window MAE about 0.481; late value about 0.731.
- Corrected causal denominator is visibly active in the logs.

Its sampled clips can look good even though late training metrics deteriorate.
Different clips and routes make visual similarity to `ganfix_blind` insufficient
to prove a GAN improvement. A fixed-route, fixed-seed long evaluation is required.

## Campaign Notes

| Campaign | What it tested | Outcome |
| --- | --- | --- |
| `rollcarn700` | CARN continuation of an already trained rolling model | Initially promising, then inherited freeze near 4.5 s |
| `rollcarn700_bidir_*` | Bidirectional/GAN changes after step 700 | Confounded by the inherited attractor; stopped |
| `ganmvp700*` | GAN repair after step 700 | Same inherited freeze; not a clean prevention test |
| `fullkl_bidir_nogan` | Bidirectional DMD without GAN | Visually stronger than paired GANxeq line |
| `fullkl_bidir_ganxeq` | Conservative equalized transition GAN | Critic remained near chance; near-inert GAN |
| `ganfix_blind` | Aggressive action-blind transition GAN | GAN learned modest separation, rollout benefit unproven |
| `ganfix_strict` | Action-conditioned transition GAN | Reached step 50 before Slurm cancellation; rerun queued |
| `arlocal300` | Corrected fresh causal DMD/CARN baseline | Good sampled clips, but volatile late metrics |
| `dmd3kl_v14` | Fresh v14 KL/DMD3 attempt | No saved videos found; visually inconclusive, not a negative result |
| `fullcarn_bidir_kl_nogan200` | Fresh best DMD plus CARN | Strong current baseline; texture still accumulates |
| `fullcarn_bidir_kl_wave01` | Same baseline plus clean wavelet GAN | Running; motion survives 9 s at step 91, but texture/geometry warp after the seed horizon |
| `fullcarn_bidir_kl_proj01` | Same baseline plus projected GAN | Queued on holder `6100986` |

The original KL checkpoint remains a good initialization. The absence of videos
for `dmd3kl_v14` means that campaign should not be rejected on visual grounds.

## Current Queue

- Holder `6100009`: running `fullcarn_bidir_kl_wave01`; afterward, it will run the
  matched no-GAN CARN-on 60-second evaluation.
- Holder `6100985`: pending; queued to rerun `ganfix_strict` from fresh KL
  initialization for at most 200 steps.
- Holder `6100986`: pending; queued for the projected-GAN full-CARN arm.

No holder allocation has been canceled.

## What Went Well

- Identified and repaired the stale 14e schedule mapping.
- Corrected the causal DMD normalization and added diagnostics.
- Rebuilt the main experiment from fresh KL initialization instead of step 700.
- Preserved the bidirectional 14e score model while keeping the student causal.
- Made the discriminator objective internally consistent and testable.
- Established from W&B that `ganfix_blind` really trains its GAN, while `ganxeq`
  mostly does not learn a critic.
- Found the exact 4.5-second past-context boundary behind the repeated freeze.
- Added inference-time CARN parity and fixed long-action evaluation handling.
- Reduced checkpoint storage pressure and kept experiments inside short holders.

## What Did Not Work Yet

- Short continuation from step 700 did not repair the inherited rolling attractor.
- Conservative equalized GAN settings were too weak to learn useful separation.
- Action-blind GAN pressure has not demonstrated improved long-horizon motion.
- The no-GAN full-CARN control still develops texture and structural feedback.
- The current wavelet GAN preserves motion in its step-91 sample but has not
  prevented texture and geometry feedback after the seed horizon.
- Earlier long evaluations omitted inference-time CARN and must not be treated as
  definitive CARN evaluations.

## Verification

Focused tests cover the schedule, ODE checkpoint compatibility, scalar
discriminator output, long action loading, and inference-time CARN. The current
suite passes with:

```bash
PYTHONPATH=.:action-forcing pytest \
  testing/test_action_forcing_checkpoint.py \
  testing/test_eval_carn.py \
  testing/test_eval_chain_actions.py \
  testing/test_ladd_scalar_output.py \
  testing/test_ode_schedule.py
```

Result: 12 tests passed. Python compilation and shell syntax checks also passed.

## Immediate Decision Rule

Judge each GAN against the same fresh full-CARN baseline on the same Madrid route,
seed, actions, checkpoint step, and inference-time CARN. A GAN passes only if it:

1. Learns a sustained real/fake margin rather than staying near chance.
2. Applies a non-negligible but controlled generator update.
3. Extends motion and texture stability beyond the 4.5-second seed horizon.
4. Improves the fixed 60-second video without exchanging texture failure for
   freezing, action mismatch, or geometric collapse.

The strict action-conditioned rerun is the most informative next check. If its
critic learns but motion still freezes, the transition representation/objective
is insufficient. If it preserves action-consistent motion while the wavelet arm
controls texture, a combined discriminator is justified by evidence rather than
by a larger sweep.
