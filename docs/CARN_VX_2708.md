# CARN VX comparison — 27 August 2026

Status: active experiment record. No CARN reflow is authorised by this
document; the purpose of the campaign is to choose the winning consumer first.

## 1. Question

The CARN learns the causal-AR texture drift between two rollouts. A forward
model learns rollout1 -> rollout2 (`+1` drift); reversing the training pairs
learns rollout2 -> rollout1 (`-1` drift). We are comparing three ways to turn
that model into a generator improvement:

1. show a transition discriminator a negative drift direction;
2. apply the learned correction to generated autoregressive memory directly;
3. use the corrected output as a stop-gradient auxiliary target so the
   generator itself internalises the negative direction.

The transition construction has two sign-equivalent candidates which are also
being compared: `+1` on the real former chunk versus `-1` on the real latter
chunk.

## 2. Pinned base

The requested base is W&B run `dmd10k_carncommit_long_off_j6144623`
(`g5ndc0fz`), produced from `sbatch/carncommit_long.sbatch`. The run completed
300/300 steps on holder 6144623. Its resolved configuration and runtime trace
establish:

| Item | Resolved behaviour |
|---|---|
| GAN | enabled, `ladd_teacher_feat`, fake-score feature source |
| Fake latent source | `ladd_fake_sample_source=dmd`: graph-bearing DMD-scored band, not the separate flash-60 slab |
| Real latent source | detached dataset GT window |
| Disc diffusion noise | none: `ladd_disc_force_clean=true` sets `disc_t=0` for both real and fake; flip/translation DiffAug remains |
| Pair modes | `gt_vs_fake=true`; `gt_transition=false`; `adjacent_chunks=false` |
| Action critic | `action_critic_aux_enabled=true` and checkpoint-loaded, but frozen: no optimiser/DDP/teacher regression; generator guidance remains active at weight 0.3 after its 50-step ramp |
| CARN in GT-vs-fake | deliberately bypassed; runtime `[FN-GTVF-CLEAN]` confirms positives stay clean GT |
| Commit correction | off (the matched treatment switch) |

“Real vs fake” here is the LADD discriminator's GT-vs-fake pair mode, not an
action-critic classifier. The frozen action critic supplies action-consistency
guidance on generated chunks; it is not separately trained to classify real
and fake in this run.

Important hidden base behaviour: `reverse_noiser_dedrift_enabled=true` already
corrected the loss-computation copy of `train_chunk` before DMD/GAN scoring in
both members of the old commit pair. It did **not** change the emitted chunk or
the KV memory unless `reverse_noiser_dedrift_apply_to_commit=true`.

## 3. Completed direct/commit arm

The matched treatment is W&B run
`dmd10k_carncommit_long_on_j6144621` (`e0609ow2`). It is the same source file,
seed and recipe as the OFF base; the treatment is
`reverse_noiser_dedrift_apply_to_commit=true`.

Researcher visual verdict (26 August): **marked improvement; this arm works**.
This completes method 2. A direct W&B-config comparison found only the intended
commit switch, the run/log names, and the ON run's 299-step cap versus the OFF
run's 300-step cap; there is no hidden GAN, critic, seed or optimiser change.
Mechanistically, the result also resolves the older
"correct every generator output but nothing changes" observation: correcting a
scoring copy lets the composite corrector carry the burden and leaves the
autoregressive state raw. The successful site corrects the tensor written into
KV memory, so chunk N+1 and every later chunk condition on corrected history.

The mechanism fired rather than merely resolving in config: the ON log reports
commit correction `rel|dz|=0` at calls 1 and 10 (expected from zero-init), then
`0.00774795` at call 50 and `0.00334163` at call 200. The OFF run has no commit
proof by construction.

## 4. Code faults found before the remaining launch

1. `forward_noiser_reverse=true` trained the requested R2 -> R1 single network,
   but `_dedrift_with_reverse_noiser` only selected the corrector for
   `fn_pair_mode=rollout_to_gt` or cycle mode. Reverse and cycle modes are
   mutually exclusive, so the direct reverse configuration returned the input
   object unchanged.
2. The internalisation helper then saw `z_dedrifted is z_raw` and returned an
   exact zero. Even if given a distinct tensor, it rejected non-cycle
   `r1_vs_r2` mode. This was a second independent aux no-op.
3. The master de-drift gate unconditionally replaced `train_chunk`, so the
   existing "aux" configuration necessarily stacked direct de-drift-then-score
   with the aux loss. It was not an aux-only test.
4. A single scalar level was applied to a multi-chunk streaming slab even
   though the FN is trained on 3-frame chunks with a condition per pair. The
   new adapter splits the slab and derives the condition from each chunk's
   absolute rollout position.
5. FN training historically preferred the flash-60 slab, while the pinned
   GAN consumes the DMD band. The aux arm now trains and applies the reverse FN
   on the same raw rollout-output domain.

The fixes are default-compatible and covered by
`testing/test_carn_vx_routes.py`; the old commit and decoupled-GT coverage is
also rerun before launch.

## 5. Remaining matched arms

All launch through `sbatch/run_carn_vx_on_holder.sh`, which derives the runtime
script from `sbatch/carncommit_long.sbatch` and injects the listed CARN values
as final, last-wins overrides. The DMD recipe, clean DMD-band GAN inputs, frozen
action critic, optimiser settings, seed and evaluation cadence remain the
requested base.

| Mode | One GAN pair mode | CARN training | CARN consumer |
|---|---|---|---|
| `tx_plus_former` | transition | R1 -> R2 | real pair `[F(+1, GT former), clean GT latter]`; generator former detached |
| `tx_minus_latter` | transition | R2 -> R1 | real pair `[clean GT former, R(-1, GT latter)]`; generator former detached |
| `aux_minus` | GT-vs-fake | R2 -> R1 on raw rollout source | masked `0.25 * L1(raw, stopgrad(R(-1, raw)))`; direct score replacement, flash replacement and commit replacement all off |

For single-mode parity the transition arms replace GT-vs-fake rather than stack
a second GAN loss. All three keep the base's one-mode backbone kick
(`ladd_fake_backbone_grad_scale=0.2`).

Before results, `tx_plus_former` is the simpler/safer transition construction:
it reuses the established forward R1 -> R2 CARN and makes the negative
direction relationally (`drifted former -> clean latter`). The literal
`tx_minus_latter` is worth testing, but has an unavoidable domain caveat:
R was learned on drifted rollout2 inputs and is being evaluated on a clean GT
latter chunk, so it may extrapolate past the clean manifold. Both restore the
input chunk's per-channel mean and mean magnitude after CARN, preventing
brightness/statistics leakage from deciding the comparison.

## 6. Holder ledger

| Holder/job | Purpose | State |
|---|---|---|
| 6145507 | aux mechanism smoke, then full `tx_plus_former` | full arm completed 300/300 as `p68q87lv` |
| 6145508 | replacement full `aux_minus` | full arm completed 300/300 as `334wiw34` |
| 6148536 | full `tx_minus_latter`, then GAN scalar-surrogate smoke | CARN completed 300/300 as `ykxizqvl`; scalar surrogate completed 90/90 as `9u5rb6hr` without overlap |
| 6148537 | GAN evidence experiments | VGG and RN50 direct connections both proved but both OOMed after step 35; serialized weight-zero direct-gradient surrogate then completed 90/90 as `zffrfznl` |
| 6150252 | GAN current-teacher control | `ttxdfxu8`: held-out cosine 0.154→0.268→0.342→0.484; below 0.50; learned and tangent parameter gradients both zero |
| 6150866 | GAN aligned-source control | `wucn3u1w`: 0.147→0.313→0.318→0.440; alignment did not improve final field; same zero-gradient route |
| 6150867 | main no-Flash t0 CARN holder | commit+aux treatment completed 300/300 as `4guvi6s2`; allocation deliberately remains a holder |
| 6153178 | active two-node CARN allocation | restored Flash-era `tx_both` is healthy beyond step 10 as `y1kq6xup` |
| 6153179 | idle two-node CARN allocation | Flash-aligned commit+aux reached step-0 sampling, then the `nid010944` GPU-0 defect exhausted memory; allocation left untouched and direct job `6157938` is the clean replacement |
| 6156228 / 6156229 | two queued two-node CARN jobs | armed with the restored Flash-era bundle for the combined campaign described in section 9 |
| 6158612 | direct two-node CARN job | queued full permutation: tx both + aux-minus + committed-memory correction, restored Flash-era routing |
| 6157129 | six-hour holder running commit+aux | first child `cam20fbh` failed before step 0 from an asymmetric hidden GPU-0 reservation; `NCCL_NVLS_ENABLE=0` retry `8wh1djnr` has balanced memory, completed step 0, and emitted/uploaded its first videos |
| 6157130 | six-hour holder running selective-t0 commit+aux | `6dn8cq2k`: Flash t=60 retained for action guidance/anti-collapse, aux CARN R1 uses rollout t0 and KV commit uses ladder-endpoint t0; NVLS workaround active, step 0 and first videos completed |
| 6156642 | cancelled replacement allocation formerly pinned to `nid010673,nid010691` | cancelled externally before use; its residual command file is Flash-aligned if the allocation is ever re-created under that ID |
| 6150253 | former full `aux_minus` backup | cancelled after `6145508` replacement went live |

Final run IDs, proof lines, matched-window metrics and visual verdicts will be
added below as the holder jobs finish.

Action-critic check: the CARN VX runs on `6145507` and `6145508` both inherit
the requested frozen action critic (`action_critic_aux_enabled=true`,
checkpoint loaded, `action_critic_freeze=true`, guidance weight 0.3). The old
`carncommit_of_on` task on `6145508` did **not**: it had
`action_critic_aux_enabled=false` and guidance weight 0.0.

## 7. Result table

| Arm | Run | Mechanism proof | Quantitative result | Visual result |
|---|---|---|---|---|
| Commit OFF | `g5ndc0fz` | commit proof absent by design | baseline | baseline |
| Commit ON | `e0609ow2` | nonzero commit correction by call 50 | late H/V texture anisotropy 1.42 vs 6.13 OFF; angular entropy 0.942 vs 0.864 OFF; short-horizon MAE tied | **marked improvement (researcher)** |
| Transition +former | `p68q87lv` | resolved transition-only; `[CARN-FORMER]` confirms +1 on real former; fake former detached | completed 300/300; final logged D loss 0.1894, D real/fake -1.467/-3.195; final rollout gate fake/real 0.7170/0.7178 | late seven-chunk output develops visible lattice/soft chromatic cast; final detail/HF retention 0.62/0.74 |
| Transition -latter | `ykxizqvl` | `[CARN-LATTER-REVERSE]` confirms clean former plus reverse-CARN(-1) GT latter; GT transition pairs built; frozen action critic loaded | completed 300/300; final logged D loss 0.1402, D real/fake +1.310/-0.733; final rollout gate fake/real 0.5549/0.5511 | late directional texture/haze; final detail/HF retention 0.70/0.80 and H/V anisotropy 0.756 |
| Aux -drift smoke | `fd5fc65z` | correct reverse net selected; levels 2..7; initial zero-init correction; FN reverse flag 1 and grad norm 0.04297 | completed 25/25; aux loss 0.00169373 at step 11 and 0.00186920 at step 21 | n/a |
| Aux -drift full | `334wiw34` | resolved GT-vs-fake only; reverse noiser R2 -> R1 on rollout source; `reverse_noiser_internalize_weight=0.25`; direct score/flash/commit replacement off; frozen action critic loaded | completed 300/300; final internalisation loss 0.006958, gate mean 1; final rollout gate fake/real 0.6361/0.6436 | **cleanest of the three matched arms**; final detail/HF retention 0.90/1.01, H/V anisotropy 1.030 |

The completed commit pair's logged scalar sample is sparse (27 ten-step
training records and 54 five-step rollout records after step 30), and the two
arms are effectively tied: ON/OFF DMD error is 0.14862/0.15123 and rollout MAE
is 0.55185/0.54954. These diagnostics are not a contradiction of the visual
verdict. They mostly measure the current short training slab, whereas commit
correction changes the autoregressive history and is intended to show up over
the long rollout videos.

The texture diagnostics do see the same improvement as the researcher. Over
steps 151 onward, ON/OFF H/V anisotropy is 1.4168/6.1337 (1 is the isotropic
target), angular entropy is 0.9418/0.8643, and HF power is 0.0958/0.0865.
Commit correction therefore removes a strong directional texture pathology
without buying that result through a short-horizon MAE change.

The aux weight is 0.25 by gradient calibration, not by comparing raw loss
values. Mean-L1 contributes an output-space signed gradient of scale 0.25; the
base's observed post-normalisation DMD mean-gradient scale is roughly
0.45--0.8. Thus the aux direction is approximately a 30--55% competitor after
the DMD warmup even though its value is small once `R(raw)` is close to raw.

### Matched visual verdict (2026-08-27)

The final `step_0000296_pred_image_7_chunk` videos were compared at the same
12 evenly spaced frames, with special attention to the generated second half.
Aux-minus is the clear practical winner: it remains sharp and visually clean.
Transition +former visibly accumulates a fine lattice and a soft chromatic
cast; literal transition -latter accumulates directional texture/haze. On
eight evenly spaced generated-half frames, output/GT-half Laplacian retention
is 0.897/0.621/0.703 and HF-power retention is 1.007/0.742/0.804 for
aux/+former/-latter respectively. Their final H/V anisotropy is
1.030/1.029/0.756 (target 1).

Across steps 256, 276 and 296, aux also preserves detail best on average
(Laplacian ratio 1.057, HF ratio 1.101), versus +former (0.628, 0.861) and
-latter (0.992, 0.989). One aux scene at step 276 makes the orientation-only
battery pessimistic despite no corresponding visible lattice; this is why the
verdict uses the matched frames plus detail retention rather than a single
scene-confounded orientation scalar.

Recommendation: make `internalize_target` / aux-minus the default CARN
consumer in the reflow. Keep +former only as an explicit transition-GAN
ablation and do not promote literal -latter: it adds the clean-GT extrapolation
caveat without winning visually. This is a recommendation, not authorization;
no reflow has been applied.

## 7.1 Q1~0.99 GAN transport hand-off (28 August)

**Cache invalidation.** The Q1~0.99 decoder pullback itself remains the GAN
transport choice, but the first live VGG-GAN waves below are not valid CARN
comparisons. An unsafe `data_ptr` decode-cache key collided across fresh
temporary discriminator micro-groups and supplied stale decoded pixels.
Accordingly `.10`, the four 100-step rankings, the stopped transition/R1
wave, and the brightness-screen ranking are withdrawn pending cache-off
repetition. This does not change section 7's standalone aux-minus result,
which did not depend on this VGG decode cache. Corrected weight-zero
calibrations are running on all four nodes of holders `6168503/6168505`; only
their child steps may be stopped, never the holders.

The GAN consumer used for the eventual CARN reflow is no longer the learned
Q1~0.60 exact-6+7 transport.  New runs use the hash-pinned all-residual WAN
pullback measured at paired Q1 `.990999` and strict Cartesian Q1 `.992351`.
All 12 residual VJPs are analytic/tied and consume the current Flash-t=60
pixel cotangent with detached state and zero staleness; no surrogate fitting
or reconvergence occurs in training.  The definitive launcher retains only
the CARN reference/commit/aux-minus/commit+aux/former-plus/latter-minus
surface and explicitly disables the older pixel-texture, OF and latent-
surrogate GAN paths.  Weight-zero production calibration selected
`LR=1e-3,U1` and a conservative first live `PIXW=.10`; the active smoke is
W&B `mpt1nx4y`.  This changes the GAN transport used to evaluate CARN, not the
section-7 ruling that aux-minus is the preferred CARN consumer.

The invalidated first matched Q1~0.99 wave completed as reference `4vo4kwys`, commit
`ududui2a`, aux-minus `1wyle11y`, and commit+aux `u3i9p6ya`.  All four pass
the exact-stage/current-teacher/video safety gates.  The short 100-step
interaction does not identify a CARN improvement over reference: correct D
ordering is 9/11, 8/11, 6/11 and 9/11 respectively, while aux-minus has a
single 34.9% applied-share excursion and combined ends wrong-order.  Preserve
the longer section-7 aux-minus ruling for the CARN reflow, but do not claim
that adding it improves this newly calibrated GAN without a longer matched
run.  The transition pair is now running as `ir2yhn2f/jrcll5m8`; independent
reference arms `sf4skqwf/14anf43a` test lower R1 and U2 discriminator tracking.
These are raw-VGG matched controls.  The independent brightness screen already
selects Flash/U1 + spatial-DC rejection + stat anchor (`fzuybrrm`) as the
production GAN front end; a winning head change must be reconfirmed with that
filter before the CARN reflow uses it.

## 8. Reflow decision (awaiting researcher authorization)

The three requested arms are complete and aux-minus is the recommended
consumer. Do not reflow CARN until the researcher authorizes that choice. The
eventual reflow
should retain one training direction abstraction and one explicit consumer
policy (transition target, memory correction, or internalisation), eliminate
the current implicit master-gate side effects, and make the chosen output
domain/level schedule part of the typed contract rather than scattered flags.

The reviewed shape of that reflow is:

1. replace `forward_noiser_reverse` plus cycle/pair-mode conditionals with an
   explicit training direction (`add_drift`, `remove_drift`, or dedicated
   cycle pair) and a named source domain (`flash`, `rollout`, `GT`);
2. centralise chunk splitting, absolute-position level derivation, frozen
   parameter handling and optional moment restoration in one CARN operator;
3. make consumers mutually exclusive and explicit (`transition_former`,
   `transition_latter`, `score`, `commit_memory`, `internalize_target`) rather
   than letting `reverse_noiser_dedrift_enabled` mutate several sites;
4. extract real/fake pair construction from the monolithic LADD routine into
   a single `GanPairSpec`, so GT-vs-fake and transition mode cannot be stacked
   accidentally and backbone scaling derives from the resolved spec;
5. require every non-identity consumer to emit source, level range, active
   rows and relative displacement on a fixed cadence.

Which consumer becomes the default—and which legacy sites can be deleted—is
the only result-dependent part. No item above has been applied as a reflow.

## 9. Combined-permutation Flash-era campaign (realigned 2026-08-27)

Every campaign arm is a real two-node job with four GPUs per node. Holder-backed
arms invoke one two-node `srun`; the direct batch arm requests two nodes
itself. The launcher refuses to start on any allocation whose resolved node
count is not two. Every arm has `max_steps=300`, `sample_interval=15`, and
`checkpoint_interval=200` last-wins.

The requested `tx_both` and `commit_aux_tx_minus` queued commands enter through
`sbatch/run_carn_vx_snapshot.sh`, which copies the reviewed launcher before
execution so later source edits cannot mutate a live Bash process.

| Holder | Intended W&B run name | Treatment |
|---|---|---|
| `6150867` | `dmd10k_carnvx_commit_aux_minus_noflash_t0_full2708_main_j6150867` ([W&B `4guvi6s2`](https://wandb.ai/ashish397-university-of-exeter/longlive-phase-3/runs/4guvi6s2)) | completed historical no-Flash result; retained for comparison but not part of the realigned queue |
| `6153178` | `dmd10k_carnvx_tx_both_flash60_flashrestore2708_j6153178` ([W&B `y1kq6xup`](https://wandb.ai/ashish397-university-of-exeter/longlive-phase-3/runs/y1kq6xup)) | transition only: cycle F applies `+1` to the real former and cycle G applies `-1` to the real latter |
| `6153179` | `dmd10k_carnvx_commit_aux_minus_flash60_flashrestore2708_placementaudit_j6153179` ([W&B `th9c2olq`](https://wandb.ai/ashish397-university-of-exeter/longlive-phase-3/runs/th9c2olq)) | Flash-aligned but failed after step-0 sample on defective `nid010944` GPU 0; superseded by direct job `6157938` |
| `6156228` | `dmd10k_carnvx_commit_aux_minus_tx_minus_flash60_flashrestore2708_j6156228` | requested full minus stack: commit + aux-minus + transition-minus |
| `6156229` | `dmd10k_carnvx_aux_minus_tx_minus_flash60_flashrestore2708_j6156229` | control: aux-minus + transition-minus without commit, isolating the commit contribution to the full minus stack |
| `6156642` | n/a | allocation was cancelled externally and is not in the queue |
| `6157938` | `dmd10k_carnvx_commit_aux_minus_flash60_tuned_flashrestore2708_j6157938` | direct two-node, three-hour matched follow-up: commit+aux with the known Flash-era auxiliary domains restored; clean DMD-band LADD and ladder-endpoint KV commit retained |
| `6158612` | `dmd10k_carnvx_commit_aux_minus_tx_both_flash60_flashrestore2708_j6158612` | requested full stack: transition `+former` and `-latter`, aux-minus at 0.25, and committed-memory correction; two pair modes use calibrated backbone scale 0.1 |

The active and queued campaign now restores the old Flash-included routing as
one atomic contract: `flash_dmd_enabled=true`, `flash_dmd_gan_t=60`, Flash-rung
anti-collapse, action guidance from `flash`, no differentiable finish-rung
replacement, and the legacy rollout2 source. Transition-only modes train R1
from Flash. Modes containing aux-minus train R1 from raw rollout, matching the
completed clean aux-minus reference, while R2 remains `legacy`. The LADD GAN
continues to use the clean DMD-scored band (`ladd_fake_sample_source=dmd`,
`ladd_disc_force_clean=true`); this restoration does not reinterpret Flash as
the LADD fake. KV commit remains the ladder endpoint under the pipeline's
default-true commit contract.

The discarded no-Flash attempts on `6153178` and `6153179` both failed
primarily from CUDA OOM before producing useful campaign results. Both Flash
replacement Hydra dumps independently confirm the contract above. `6153178`
is healthy beyond step 10. On `6153179`, an instrumented run found exactly one
rank per GPU but an additional, unattributed ~43 GiB unavailable only on
`nid010944` GPU 0; the run reached its step-0 sample and then failed when that
device had 151 MiB free. The allocation was left intact rather than changing
scientific knobs to fit a bad GPU; direct queued job `6157938` is the clean
commit+aux replacement. Failed names remain only as W&B/local provenance and
must not be compared as completed arms.

The two-pair-mode arms use `ladd_fake_backbone_grad_scale=0.1`; with five
updates in each of two modes this keeps the total backbone kick equal to the
single-mode arms' `5 * 0.2`. The new transition route logs monotone transformed
row counts for both signs, including an explicit cycle-source proof, rather
than relying only on resolved config.

## 10. No-Flash result and matched Flash follow-up

The main no-Flash run `4guvi6s2` completed 300/300 cleanly and wrote
`fn_rev_step0200.pt`. It did not numerically collapse: the DMD error stayed in
the prior arms' range, CARN/internalisation remained finite, and no NaN, OOM,
or traceback occurred. It did, however, regress visually. On the matched final
step-296 seven-chunk rollout, the generated tail has 2.049x the real seed's HF
power and rises from seed median brightness 93.1 to 160.2 while contrast falls
from 50.8 to 42.1. The completed Flash-era aux-minus reference has 0.985x HF
power, median 111.8, and near-unit directional ratios. The no-Flash output is
therefore over-sharpened/brightened rather than merely insufficiently denoised.

This result does **not** establish that the LADD GAN needs Flash: both runs use
the same `ladd_fake_sample_source=dmd` and `ladd_disc_force_clean=true`, and the
pixel GAN is disabled. The changed surfaces are the frozen action critic, the
anti-collapse anchor, and the CARN R1/R2 domains; commit+aux is also a new stack
relative to the two individually successful treatments. The endpoint CARN is
live but weakly trained (late forward-noiser gradient about 0.037 versus 0.135
in the Flash-era aux run), so it is signal-starved rather than hard-dead. The
highest-risk replacements are the out-of-domain frozen action critic and the
multi-rung ladder anti-collapse gradient; their combination is consistent with
the excess isotropic HF.

Job `6157938` is the matched bundle test. It restores `flash_dmd_enabled=true`,
`flash_dmd_gan_t=60`, action guidance from `flash`, Flash-rung anti-collapse,
and the known-good aux CARN domains (`R1=rollout`, `R2=legacy`). It does not
move the tuned LADD fake off the clean DMD band, and the pipeline's default-true
contract still commits the ladder endpoint to KV memory. Thus a recovery says
the Flash-era auxiliary bundle matters; it must not be misreported as evidence
that the LADD GAN itself was consuming Flash.
