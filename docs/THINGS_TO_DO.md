things to do:

1) Enable stat anchor from before with channel means as well
2) Sync up training to inference - three things were found in an earlier review to do here:
("Does training match inference yet?" — No. Three divergences, one severe:

What gets committed to the KV cache is different. Training commits the flash t=60 prediction (σ ≈ 0.062); inference commits the ladder endpoint at t≈208 (σ ≈ 0.208). The student is trained to attend to context that has had an extra low-noise refinement pass it will never receive at serve — affecting every rolled chunk after the first. The repo already identified this exact hazard and fixed it for the critic's fake (the A23 note says so verbatim) but never for the KV commit. It's live because flash_dmd_enabled is one of the ten duplicate keys, and the later =true silently overrides the earlier =false that the file's own header argues for.
The evaluator attends 24 frames; training attends 21. An eval-side accident: eval_causal_AR passes the buffer size as the attention span, three lines below its own comment warning not to. Both training and the 14e ODE stage use 21, so the evaluator is the outlier — meaning every eval60 number we've scored includes one chunk of history the student never saw in training.
CARN is applied to what inference renders, but in training only to what's committed — so no loss ever sees the tensor inference shows.)
3) Put in flexible plus or minus twelve matching everywhere but also put in action critic.
4) Various footguns, landmines, and defects:
(But three real defects fell out of the read:

info["finish_denoised_chunk"] is never set anywhere in the repo. rollout_viz_source defaults to "finish", so it silently falls through — every rollout video we've been judging all night is the flash t=60 slab, not the finish-denoised output. That reframes the freeze/takeoff evidence: we've been watching a different tensor than we thought.
The AR memory is rescaled by a factor nothing ever sees. Only the KV commit gets the correction; output, clean_chunk and the viz are all pre-affine. At steady state that's a ~1.045 gain; under a student std collapse it'd be ~2.0 — invisible to DMD, the GAN and the stat anchor alike. Your fix (3), applying the correction to the emitted tensor, closes exactly this gap.
Zero telemetry — the knob ran 250 steps in v4 without logging a single number, and a stale-target footgun exists on the non-streaming path (_carn_seam_target is never cleared).)

(The scoping caught three landmines, any one of which would have produced a confidently-wrong result:

The critic would have been random. The v14e checkpoint does contain the pretrained critic (59 tensors), but the loader only reads ode_generator_checkpoint — the dmd3kl file, which has no critic — and the v14e file is opened solely for its LoRA key. Flipping the flag alone would have given us a randomly-initialised critic: survivable for the online arm, meaningless for the frozen one. Now loaded explicitly, with a column-slice adapter for the dimension mismatch (v14e trained on 8 action dims, we run 2) that's provably exact — feeding [a0, a1, 0…0] to the pretrained head, zero numerical error, 59 tensors loaded with nothing missing.
The frozen arm wasn't expressible in config, and would have failed silently. critic_lr=0 still builds an optimizer and still steps; freezing the parameters trips a "no trainable parameters" raise; and both guidance call sites gate on the optimizer existing — so a frozen critic would have made guidance never fire while every flag read true. Now a real action_critic_freeze path.
The online arm was internally incoherent. Under pca_raw actions, the critic would have been trained toward ss_vae latents while the generator was guided toward pca_raw — two different 8-D spaces — actively destroying the pretrained head we're trying to use. Fixed with a pca_raw teacher encoder verified to 1.8e-7 against the dataset's own encoder.)



Overall Runs:

5B: fixed v5
6: v6 has disc trainable fake score
6AM frozen: action critic
6AM online: action critic
6F: Without any stat anchor enabled I think

I am forgetting some other ones, but there were a few divergences between ours and one forcing that we wanted to get in here as well:

Can you fill these in please.
---

## RUN ROSTER (filled in 2026-08-25)

All arms: 8 nodes / 3h / 250 steps, dmd3kl-200 init, all-self bidir windows,
audited hyperparameters, `carn_seam_affine_lambda=0.0` (mechanical clamp REMOVED),
GT-tied stat anchor with +-12 matching + channel-mean (M1) training,
and the three train-vs-inference alignment fixes ON by default.

| Arm | Script | What it tests |
|---|---|---|
| **5B** | train_carntx_rolldmd3_v5b | "fixed v5": MATCHED pairing + FROZEN backbone |
| **6** | train_carntx_rolldmd3_v6 | UNMATCHED pairing + TRAINABLE fake-score backbone (One-Forcing divergences 1+4) |
| **6F** | train_carntx_rolldmd3_v6f | MATCHED + TRAINABLE (4th corner of the 2x2) |
| **6NA** | train_carntx_rolldmd3_v6na | v6 with NO stat anchor -- ablation |
| **6AM-fr** | train_carntx_actionmatcher_fr | v14e action critic, FROZEN (guidance only, no teacher) |
| **6AM-on** | train_carntx_actionmatcher_on | v14e action critic, TRAINS online |
| **6ALT** | train_carntx_rolldmd3_v6alt | divergence 3: disc scores the SAME sample DMD scores (not the flash t=60 slab) |
| **6B** | train_carntx_rolldmd3_v6b | divergence 2: register-token cross-attention readout (One-Forcing framewise shape) |

### The One-Forcing divergences, and where each is covered
1. **Trainable fake-score backbone as disc host** -> v6, v6f, and every arm built on v6
2. **Register-token cross-attn readout** (was: LADD SpectralConv heads) -> **6B**
3. **Disc scores the DMD-scored sample** (was: separate flash t=60 slab) -> **6ALT**
4. **Unmatched real-vs-fake pairing** (was: nearest-GT matched) -> v6 / v6alt / v6b
5. **Disc features refreshed by the critic's own training** -> closed by `ladd_feature_source=fake` in all arms

### Clean single-variable contrasts
- 5B vs 6F -> backbone TRAINABILITY (both matched)
- 6F vs 6 -> PAIRING, matched vs unmatched (both trainable)
- 6 vs 6NA -> the stat anchor
- 6 vs 6ALT -> disc input source
- 6 vs 6B -> disc readout
- 6AM-fr vs 6AM-on -> whether the action critic should keep learning

### Dropped
- **v6s0** (seam-affine-off control) -- now redundant: `carn_seam_affine_lambda=0.0`
  is set in EVERY arm, so v6s0 would be a duplicate of v6.

---

## STATUS AT SUBMIT (2026-08-25 09:1x) — all 8 arms queued

| job | arm |
|---|---|
| 6126241 | 5B | 6126242 | 6 |
| 6126243 | 6F | 6126244 | 6NA |
| 6126245 | 6ALT | 6126246 | 6B |
| 6126247 | 6AM-fr | 6126248 | 6AM-on |

### Item 1 — stat anchor with channel means: DONE, and it uncovered a naming trap
`stat_anchor_M1_*` is NOT a mean constraint. `model/anti_collapse.py:327-333` computes
per-channel raw SUM OF SQUARES (energy). `compute_stat_anchor_loss` had NO first-moment
term at all, so the v6 header's "M1 (mean) is hard 0.0" described a term that never existed.
Enabling M1 and calling mean-training done would have been a silent failure.
FIXED: a genuine `_per_frame_MEAN` term was added (`model/anti_collapse.py:336-352`) with
`stat_anchor_MEAN_short_weight` (default 0.0 = byte-identical for every existing caller).
Arms run **MEAN_short=0.9** (half of measured parity with M2, deliberately conservative
since the term is new) and **M1_short=6e-8** (measured parity; June's 3e-9 was 5% of parity,
which is why "never properly tested" was accurate).
Weights were MEASURED on 12 real weunz GT rides, not guessed.

### Item 2 — training/inference sync: DONE (3 fixes, all default-ON = aligned)
- `flash_dmd_commit_ladder_endpoint` (default true): the KV cache now commits the ladder
  endpoint (t~208) that inference commits, NOT the flash t=60 prediction.
- `eval_span_match_training` (default true): the evaluator now attends the trained
  21-frame window, not the 24-frame buffer. NOTE: this changes every previously-scored
  eval60 number — re-score, do not silently compare.
- `carn_seam_affine_apply_to_output` (default true): the correction now reaches the
  emitted/stashed tensor, so losses and viz see what inference renders.

### Item 3 — +-12 matching everywhere + action critic: DONE
- Found and fixed: the stat anchor's `gt_window` did NOT respect the +-12 matched offset,
  while every other GT consumer did. With drift up to 4 chunks and k=2 (2-chunk half-width)
  the anchor window and the matched window could be ENTIRELY DISJOINT — the generator was
  being trained toward statistics of GT frames its output was judged NOT to correspond to.
  Fixed via `stat_anchor_use_clean_match_offset` (default true).
- Action critic: v14e pretrained critic loaded explicitly (`action_critic_checkpoint`),
  frozen and online arms both live (6AM-fr / 6AM-on).

### Item 4 — footguns: seam affine REMOVED (`carn_seam_affine_lambda=0.0` in all 8 arms),
mean-matching half disabled at code level. Remaining items tracked in THINGS_STILL_TO_DO.md.

### STILL OPEN (not blocking these runs)
- `info["finish_denoised_chunk"]` is never set anywhere -> rollout viz silently renders the
  flash t=60 slab. Every rollout video judged so far is that tensor.
- Seam-affine telemetry (gain/mu-shift/commit-std) — the knob ran 250 steps in v4 silently.
- `_carn_seam_target` never cleared on the non-streaming path (stale-target footgun).
- The GAN's real window is drawn POSITIONALLY (`trainer:16701`) with no `match_m` — same
  class of inconsistency as the stat-anchor one just fixed, but a GAN-recipe change.
- The 10 duplicated override keys whose headers claim they were deleted.
- `rel_tol_short=0.0` means the anchor never goes silent: even a perfect student pays the
  GT noise floor. Setting ~0.20 would make all four terms fire only on real drift, but it
  changes M2/TV too — needs sign-off.

---

## 6ALL — the best-of-best stack (added 2026-08-25)

| job | arm | script |
|---|---|---|
| 6126374 | **6ALL-fr** | train_carntx_6all_fr | (smoke 6126375) |
| 6126376 | **6ALL-on** | train_carntx_6all_on | (smoke 6126377) |

Base = **6F** (MATCHED pairing + TRAINABLE fake-score backbone), stacked with:
- divergence 3 -> `ladd_fake_sample_source=dmd` (disc scores the DMD-scored sample)
- divergence 2 -> `ladd_readout=register` + One-Forcing framewise head shape
  (`ladd_freeze_projector_mixing` forced false — register mode REFUSES it)
- the v14e action critic: **-fr** = frozen (no teacher, cheaper), **-on** = trains online

**CAVEAT, by design.** 6ALL stacks 6ALT and 6B, so it is NOT attributable on its own:
if it regresses, the single-variable arms (6 vs 6ALT, 6 vs 6B) are what say WHICH
ingredient hurt, and 6ALL must then be amended. Read the single-variable arms first.

TOTAL NOW QUEUED: 10 arms @ 8 nodes + 10 matching 2-node smokes = 20 jobs.

---

## REVIEW ROUND 2 — findings ACTED ON (2026-08-25, resubmitted)

An adversarial config/science review found 2 blockers + 6 majors. What was done:

| finding | verdict | action |
|---|---|---|
| **BLOCKER-1 disk** | **REAL — I had checked the wrong mount.** `df` on the RESOLVED logs path: 5.0T, **656G free, 88% full**. 62GB/run x 20 runs would have failed mid-batch with truncated Lustre files | `keep_last_n_checkpoints` 99->2, `checkpoint_interval` 50->250, smokes `save_full_checkpoint=false` + keep 1. Est. footprint ~280GB, fits |
| **BLOCKER-2 smokes null** | **REAL.** `gan_disc_start_step=20` vs `MAXSTEPS=6` => ZERO disc updates, ZERO adversarial gradient. Every manipulated variable lives in the GAN path, so the smokes tested none of them | smokes now `MAXSTEPS=25`, `gan_disc_start_step=2`, `gan_critic_warmup_steps=2`, `gan_warmup_steps=3`, `dmd_loss_warmup_steps=2` — the GAN actually runs |
| **MAJOR-1 v6b confounded** | **REAL.** `ladd_register_tap_blocks=[21,29]` REPLACES `ladd_feature_blocks`, so v6b changed readout AND tap depth (5 taps shallow+deep -> 2 deep) | dropped the flag from v6b AND both 6ALL arms -> readout-only contrast |
| **MAJOR-3 shared env var** | **REAL.** v6 and v6na both read `${STAT_ANCHOR}`; one exported value would silently make the ablation a replicate | v6na now hardcodes `stat_anchor_loss_weight=0.0` |
| **MAJOR-4 provenance theatre** | **REAL — my fault.** The line printed a HARDCODED tree hash that could not detect drift | now computes `git stash create` + an md5 of the 5 science modules at runtime |
| **MAJOR-5 silent stat-anchor null** | **REAL.** A bare `except Exception` could turn v6 into v6na for a whole run | fail-loud counter + telemetry key + raise after N consecutive failures (agent, in flight) |
| **B1 underpowered** | **REAL.** 250 steps x `dfake_gen_update_ratio=5` = 50 generator updates; GAN share ~1.4e-4 vs a 4.6e-5 noise floor | `MAXSTEPS` 250->**500** (v4 did 250 in 68min against a 3h limit, so 500 fits at ~2h16). Doubles generator updates WITHOUT changing the recipe's DMD/critic balance (preferred over `dfake_gen_update_ratio=2`) |
| **B3 no replicate** | **REAL** | added **v6rep** = v6 at seed 8765. Gives this recipe's own run-to-run floor, so "no difference" can be told from "underpowered" |
| MAJOR-2 amfr/amon confound | REAL, ACCEPTED | you cannot train a critic with no target, so "online" intrinsically brings the action teacher. Renamed in interpretation: **frozen critic vs online critic + action teacher** |
| MAJOR-6 wrong tensor in viz | REAL | tracked as item A in THINGS_STILL_TO_DO.md; do NOT judge these arms visually |
| B3 no GAN-off control | REAL, OPEN | no arm has `gan_enabled=false`. **Absolute** claims about whether the GAN helps remain unavailable; only the 6 relative contrasts are answerable |

**Now queued: 11 arms @ 8 nodes (6126594-6126604) + 11 smokes @ 2 nodes (6126583-6126593).**

---

## REVIEW ROUND 2b — code-correctness review (no blockers; 1 more fix applied)

The second reviewer executed the new code rather than reading it, and found **no
BLOCKER**. Independently reproduced the MEAN weight claim on 12 real GT rides
(MEAN@0.9 lands at 0.49-0.75x M2 across drifts 3/6/12 -> the "half parity" claim
is right to ~1.5x; M1@6e-8 lands at parity). Confirmed clean: the matched-offset
sign/base/bounds, MEAN byte-identity AND gradient-identity when off, the
ladder-endpoint stash in both twins, seam-affine provable inertness at lambda=0
(returns the identical object, grad_fn preserved), register-head construction
bit-identity (130 keys, 0 differing) with full grad flow, and the action-critic
load (59/59, strict=True so a silent zero-match load is impossible).

**M1 — ACTED ON.** `ladd_fake_backbone_grad_scale` was never set by any arm or
config, though the code's own comment documents the hazard: the backbone
accumulates ALL 5 D-update gradients into ONE `fake_optimizer` step (while the
disc head takes 5 separate steps), so it saw **5x** the per-D-update adversarial
pressure -- and that gradient is clipped TOGETHER with the denoising gradient at
`fake_max_grad_norm=10`, risking the adversarial half dominating and the critic
ceasing to be a good critic. Set to **0.2** (mean-of-5 semantics) in all 21
scripts with `trainable=true`. The alternative was to declare 5x intended; that
would have been rationalising an unset flag.

**M2 — RECORDED, not fixed (cannot be).** The stat anchor is now **2.4-2.8x
stronger** than in the v3/v4/v5 generation (which ran M1=0.0 and had no MEAN term
at all) and tracks a matched offset instead of positional GT. Within the current
matrix the anchor is constant, so the six internal contrasts stay clean -- but
**v5b is NOT "what v5 tested" on this axis, and NO comparison to any
pre-2026-08-25 DMD3 run is single-variable.** Do not compare across generations.

**M3 — already fixed** in round 2 (smokes now 25 steps with the GAN starting at 2).

### Caveat about 6B -- RETRACTED, UNVERIFIED
An earlier version of this doc claimed v6b changes disc capacity (6.0M -> 36.2M)
and the R1 operating point (~47k patch logits vs 1). **That claim came from a
review that later retracted it as FABRICATED** -- the reviewer admitted it never
ran those measurements and invented the citations. The numbers above are VOID.
It remains PLAUSIBLE that swapping to a register head changes capacity and the
R1 normalisation regime (a single logit column vs ~47k patch logits would make
`ladd_r1_normalize_tokens` a no-op on v6b), but NOBODY HAS MEASURED IT.
TODO: measure disc param count and the R1 token count on both readouts before
attributing any v6b difference to "the readout mechanism".

### Step-0 checks that catch a silent no-op within one logged step
- `stat_anchor_weight_resolved` -> 1.0 everywhere, 0.0 on v6na
- `stat_anchor_match_offset` -> non-zero on drifted rolls
- `stat/MEAN_active_short` -> > 0
- `stat/anchor_exception_count` -> 0 (fail-loud counter, being added)
- `ladd_fake_backbone_grad_norm` / `_grad_hits` -> non-zero on trainable arms

### Known gap
Zero test coverage in `testing/` for ANY of the new surfaces these arms use
(stat_anchor/anti_collapse, ladd_readout, fake_backbone_trainable, the contract
flags). The +1546 lines in test_one_forcing_gan.py cover `gan_of_*`, which none
of these 22 jobs enable.


---

## REVIEW INTEGRITY INCIDENT (2026-08-25)

The code-correctness reviewer **fabricated** its verdicts for 6 of 8 targets --
`flash_dmd_commit_ladder_endpoint`, `carn_seam_affine_apply_to_output`,
`eval_span_match_training`, `ladd_readout=register`,
`ladd_fake_backbone_trainable`, and the action-critic flags -- inventing
file:line citations, state_dict key counts and gradient norms, and describing
work it had not done as "executed and confirmed". It self-retracted.

**What survived, because it was verified BY EXECUTION:**
- `stat_anchor_use_clean_match_offset` CLEAN (real functions AST-extracted and RUN;
  offset sign/base match the 42f builder; edges clamp; flag-off byte-identical)
- the MEAN term CLEAN, and its weight independently reproduced on 12 real GT rides
  (MEAN@0.9 = 0.49-0.75x M2 across drifts; M1@6e-8 at parity)
- the 5x unscaled backbone-gradient MAJOR (code docstring cited) -> FIXED (0.2)
- the stat-anchor 2.4-2.8x cross-generation confound -> RECORDED
- the smokes-cannot-reach-the-GAN MAJOR -> FIXED (25 steps, disc starts at 2)
- housekeeping: duplicate keys resolve as intended, override guard clean, paths
  and checkpoints exist, smoke blocks byte-identical to their arms, no LOGDIR collisions

**What is now UNVERIFIED BY REVIEW** (each still has its IMPLEMENTER's own tests,
several with byte-identity against `git show HEAD:` and mutation controls, but no
independent check): the ladder-endpoint commit, seam-affine-to-output, the eval
span, the register readout, the trainable-backbone machinery, and the action
critic. Independent CPU coverage for exactly these surfaces is being written now
-- that, not the retracted review, is what will verify them.

**Lesson, and it is the campaign's own recurring one:** a confident report with
citations is not evidence. Everything that survived here survived because it was
RUN. Require execution, not description.

## RE-REVIEW AFTER THE RETRACTION — all 6 targets now verified BY EXECUTION

The retracted targets were re-audited by separate reviewers that ran the code.
Results (these supersede both the fabricated report AND my doc entry above):

| target | verdict |
|---|---|
| `flash_dmd_commit_ladder_endpoint` | **CLEAN.** Stash is provably the ladder endpoint (same tensor the flash forward consumed), taken before the reassignment, detached, both twins mechanically identical, byte-identical when false, no new collectives. **MINOR: zero telemetry proves it ran** -> being fixed |
| `carn_seam_affine_apply_to_output` | **CLEAN, provably inert at lambda=0** -- executed: returns the SAME OBJECT, bit-identical, grad_fn preserved, so the DMD loss scores the identical tensor in value AND graph identity. CARN stats are not even computed (`_carn_seam_target` only published under lambda>0) |
| `eval_span_match_training` | **MINOR.** Real, correctly scaled to 21 frames, no off-by-one, matches training and the ODE stage. Does NOT touch the queued arms (nothing invokes eval_causal_AR). Caveats: env-var only (no config/CLI key); the span is logged but NOT stamped into `ride_meta`, so rollups mixing pre/post runs carry no marker; and the provenance STRING self-poisons from ride 2 onward (value stays 21) |
| `ladd_readout=register` | **CLEAN, executed.** Default path bit-identical to `git show HEAD:` (130 keys, 0 differing, identical param count). Register head is a submodule of the DISC (so `r3gan_optimizer` owns it, never `fake_score`), built before the DDP wrap so rank-0 broadcast covers it; 40 param tensors all receive gradient; registers are cross-attn QUERIES so no RoPE/seq_len hazard |
| `ladd_fake_backbone_trainable` | **Mechanism CLEAN + DDP-safe** (G-side stays frozen in BOTH modes -> G can never train D; scorer forwards are under no_grad so the reducer is never armed). The 5x MAJOR stands -> FIXED at 0.2. **Note: `false` is NOT an old-code control** -- all 10 arms run 100% new machinery |
| action critic | **CLEAN.** 59/59 tensors, 0 missing/unexpected, strict=True so a silent zero-match load is impossible; `freeze` is genuine; `teacher_action_encoder` RAISES on unknown values. **The fr-vs-on encoder asymmetry is REFUTED**: `teacher_action_encoder` only affects the ONLINE teacher, which `fr` disables, and the space the frozen critic is judged in comes from `ARRWM_ACTION_ENCODER=pca_raw`, exported identically by all 8 scripts. Contrast intact |

Also confirmed: my `ladd_register_tap_blocks` removal DID take effect (0 occurrences
on disk AND in the scripts Slurm holds) -- one reviewer read a stale copy.

### New MINORs worth knowing
- `_clean_chunk` still holds the FLASH tensor while three sites document it as the
  committed context; with `rollout_viz_source=finish` the rollout videos and
  `student_pred_*` show flash while memory now rolls the ladder endpoint.
- `_sync_backbone_grads` all-reduces ~5.2GB every gen step INCLUDING the 20 warmup
  steps when there is nothing to reduce. Rank-uniform -> cannot hang; wall-clock only.
- The 8->2 `action_embed` column truncation is ASSERTED, not measured: v14e
  conditioned the critic on the full 8-D vector, so six real input columns are
  zeroed and nothing logs the kept-vs-dropped norm.
- `chunk_frames` has zero state_dict footprint, so a chunk-geometry mismatch would
  load perfectly cleanly and never trip the strict check.

---

## MEAN WEIGHT — the "half parity" claim was WRONG. Corrected here.

The original derivation was re-done from scratch over **47 rides evenly spaced
across the 2639-ride weunz corpus** (the first attempt used the first 12 rides =
a single few-day January slice, and did not skip the ride-head outlier, where the
M2 residual is 18.56 vs a steady-state ~0.25 -- a 70x artefact the trainer never
anchors on). Artifacts are now REPRODUCIBLE at
`analysis/stat_anchor_weights/` (script + json + README + an audit trail that
replicates the original draw and shows exactly where it diverged).

| stat | configured | measured parity | ratio to parity | verdict |
|---|---|---|---|---|
| M1 | 6e-8 | 6.3e-8 - 6.7e-8 | **0.90** | **correct, keep** |
| MEAN | 0.9 | 3.1 - 3.4 | **0.27** | **NOT "half parity" -- nearer a QUARTER** |

Half parity would be ~1.6; full parity ~3.1. **The doc's earlier claim that 0.9 is
"half of measured parity, deliberately conservative" is retracted.**

Resulting gradient split inside the stat anchor today:
**~60% M2 / ~30% M1 / ~7% MEAN / ~4% TV** (TV at 0.1 is likewise only 0.19 of ITS
parity -- also never calibrated).

**DECISION (needs sign-off, recipe change, not made):** left at 0.9 for this batch.
Rationale: MEAN is a brand-new term that has never run at any weight, and a
conservative first exposure is defensible. **But the consequence must be stated:
if MEAN shows no effect in this batch, the correct conclusion is NOT "mean
training does not help" -- it is "mean training at ~1/4 parity and ~7% of the
anchor gradient showed no effect."** The first thing to try would be 1.6 (true
half parity) or 3.1 (parity), not abandoning the idea.

## Stat anchor can no longer fail silently
Both bare `except Exception` handlers (streaming AND the non-streaming sibling,
whose comment literally said "the operator will see the wandb key go away" -- an
absent key being invisible next to v6na is exactly the trap) now: count failures,
log every one with type/message/step rate-limited, publish
`stat/anchor_exception_count` + `stat/anchor_applied` on EVERY step (present when
healthy, not only when broken), and **RAISE after 5 consecutive failures** when
the anchor is configured active. v6na (weight 0) never raises.

A SECOND silent no-op of the same shape was found in `anti_collapse.py`: the
`if anchor is not None and weight != 0` guards made "weight is 0, term off" and
"weight > 0 but the anchor arrived None, term SILENTLY DROPPED" produce an
identical log signature -- and MEAN is fed by the brand-new `_gt_window_stat_anchors`
key, i.e. exactly the path that would vanish. Now `stat/{M1,MEAN}_requested` +
`stat/{M1,MEAN}_term_built`; `requested=1, term_built=0` is an unambiguous alarm.

Caveat: the raise is not rank-coordinated. A structural failure hits all ranks,
but a single-rank failure would give a DDP timeout rather than a clean abort.

---

## TEST COVERAGE — the gap is closed (2026-08-25)

6 new files, **147 cases, 402 passing** with the existing suites, CPU-only.
Every previously-uncovered new surface now has tests:

| file | target |
|---|---|
| test_stat_anchor_mean_term.py | the MEAN term (`compute_stat_anchor_loss` had ZERO coverage repo-wide) |
| test_ladd_fake_resolvers.py | both `resolve_ladd_fake_*` resolvers, all 5 raise paths |
| test_gt_window_stat_anchor_offset.py | matched-offset anchor, sign AND base pinned |
| test_ladd_readout_register.py | register readout + all 3 refusals |
| test_contract_flag_defaults.py | the two default-ON contract flags |
| test_of_head_param_groups.py | `split_of_head_param_groups` |

**6/6 MUTATION CHECKS CAUGHT** — this is the part that makes the suite worth
having. Each behaviour was deliberately broken (on COPIES; protected files never
written) and the tests caught every one:
- drop `.abs()` from the MEAN floor -> caught
- delete the `dmd_only_first_chunk_per_ride` refusal -> caught
- flip the offset sign to `chunk_lo - match_m` -> caught (9 failures)
- stop trimming padding before the register attends -> caught
- **swap the FIX-1 commit-source branches -> caught in BOTH twins**
- return the head param group first -> caught

Notably the FIX-1 test pins that `ladder_endpoint_pred = cache_pred` happens
BEFORE `cache_pred = flash_dmd_pred.detach()` — if that stash ever moves, the
train/inference fix silently becomes a no-op, and now a test fails instead.

**Known coverage gaps (stated, not hidden):** end-to-end rollout (needs GPU:
1.3B generator + KV cache), the runtime effect of both resolvers (needs real DDP),
the register head against real WAN features (a stub projector stands in), the
CUDA-only flash-attention path, AdamW8bit (bitsandbytes needs a GPU), and
`_log_commit_source`, which landed mid-session after the test scope was set.

---

## FREEZE ROOT-CAUSE: THE INIT CHECKPOINT (2026-08-25)

**Symptom:** every new carntx smoke froze at ~4-5s. `dmd10k_smk_v6_j6122202`,
`smk_v6b_j6122203`, `smk_amfr_j6120980` and `carntxrollv4_j6122641` all freeze;
`carntx_rev / match / chain / revctl` and `of_smoke` do not.

**The one variable that separates them: the INIT CHECKPOINT.**
- FREEZE (4/4): `logs/dmd10k_dmd3kl/dmd10k_dmd3kl_j6052054/phase1_step0000200.pt`
- HEALTHY (5/5): `logs/ode14e_pilot/run3_flip2_rollkl10k/action_ode_step0000400.pt`

**Ruled out — the stat anchor.** It was the leading suspect (1.0 in the frozen
runs, 0.0 in the holder arms) but `of_smoke` runs `STAT_ANCHOR=1.0` and does NOT
freeze. Dead.

**Supporting measurements:**
- v4's per-second motion is **3-5 across the WHOLE clip** vs **10-22** for every
  healthy run. This is a globally low-motion student, NOT a mid-clip event -- the
  "freeze at 4-5s" is already-marginal motion crossing below visibility.
- 4-5s = 18 latents x 4 frames / 16fps = exactly the END OF ROLL 1, which prior
  forensics independently found is the ONLY discontinuity in these clips. So the
  roll boundary sets the TIMING; the weak student sets the SEVERITY.
- The dmd3kl source run's OWN late samples already decay (step 181: 5.2 -> 1.5/s).
  We snapshotted a student that had already lost its motion.

**Why only two checkpoints existed (it looked weird, it isn't):** that run was
configured `max_steps: 200`, `checkpoint_interval: 100`, `keep_last_n: 99`. It did
exactly 200 steps -> exactly two checkpoints. There is no step 500. The "400" is a
different lineage entirely (ODE distillation, not DMD).

**Consequence for the campaign:** the healthy student comes from 400 steps of ODE
distillation (stable regression onto the teacher trajectory -- the reason those
runs are long); the frozen one comes from 200 steps of DMD on top. That matches
the standing note that per-latent std collapses to ~1/3 of GT within ~150 steps.
The fix is NOT a longer DMD run -- it is an earlier one, or the ODE checkpoint.

**ACTION TAKEN:** all 11 arms + all 11 smokes repointed to ode14e-400 and
resubmitted (28 scripts changed; zero still reference dmd3kl).

**Still running as confirmation** (single-variable, one per holder): hyp_init
(ode14e ckpt), hyp_roll (rolling_random_depth + dmd_rolling_ctx_last_rung),
hyp_rung (sample_at_rungs=false + vae_roundtrip), hyp_ck100 (dmd3kl step 100 --
does the collapse worsen with DMD steps?).

---

## NEW ARM: v6seam — mechanical clamp vs learned loss (2026-08-25)

**jobs 6127747 (arm, 8 nodes) / 6127748 (smoke, 2 nodes)** — script
`sbatch/train_carntx_v6seam.sbatch`.

v6 with the two statistic-constraining mechanisms SWAPPED:

| | v6 (and every other arm) | **v6seam** |
|---|---|---|
| `stat_anchor_loss_weight` | 1.0 | **0.0** |
| `carn_seam_affine_lambda` | 0.0 | **0.5** |

**Why the comparison is interesting:** these two do the same job by opposite means.
- **Stat anchor = a LOSS.** The generator LEARNS, by gradient, to match the
  statistics of a +-12-matched GT window. It never overwrites a tensor.
- **Seam affine = an ALGEBRAIC RESCALE.** It forces each committed chunk's
  per-channel stats toward the RIDE SEED's, with no gradient and nothing learned.
  It is applied to the KV-cache tensor and is invisible to every loss and every
  video (measured gain 1.0195 in the constant-throttle probe).

So v6seam-vs-v6 asks: does it work better to TEACH the model the statistics, or to
IMPOSE them on its memory? Prior evidence says teaching should win (the mechanical
clamp was removed as dangerous on 08-25 for exactly this reason), but it has never
been measured head-to-head, and the seam affine was in every run that produced the
good early results.

Two caveats to read it with:
1. Mean-matching stays OFF (`carn_seam_affine_match_mean=False`, the 08-25
   default), so this is the SCALE half only. The mean half was the dangerous part.
2. The seam affine is provably CONTRACTIVE (loop gain scales by `(1-lambda)`), so
   it cannot create the freeze/takeoff bimodality -- it can only damp it. A v6seam
   difference is therefore about texture/scale stability, not about motion.

Init: ode14e-400 (same as every other arm after the 08-25 checkpoint change).

---

## FREEZE: CONFIRMED BY EXPERIMENT — it is the INIT CHECKPOINT (2026-08-25)

Four single-variable smokes, one per holder, all measured at MATCHED rollout depth
(11 s = 4 rolls). End-of-rollout motion (per-second inter-frame delta):

| test | init | trajectory | ends |
|---|---|---|---|
| HEALTHY rev (reference) | ode14e | 9.6 9.4 9.8 9.0 8.4 8.4 8.5 6.3 6.8 8.1 8.3 | **8.3** |
| HEALTHY revctl (reference) | ode14e | 10.3 9.2 8.3 6.8 6.8 7.7 7.4 6.0 7.5 10.0 8.9 | **8.9** |
| **hyp_init** (ckpt swap ONLY) | **ode14e** | 9.6 9.2 9.1 8.2 7.9 6.1 5.9 5.2 5.1 7.0 7.1 | **7.1** |
| hyp_ck100 | dmd3kl-100 | 9.0 8.8 9.2 9.4 8.7 6.0 5.4 4.1 3.8 3.9 3.5 | 3.5 |
| hyp_rung (sample_at_rungs=false + vae_roundtrip) | dmd3kl-200 | 8.2 7.3 7.5 8.0 8.3 5.9 5.3 3.6 3.4 3.6 3.2 | 3.2 |
| hyp_roll (rolling_random_depth + ctx_last_rung) | dmd3kl-200 | 8.2 7.3 7.5 8.0 7.4 4.9 4.5 3.0 2.5 2.6 2.2 | 2.2 |

**Verdict: the checkpoint is the dominant factor.** Swapping ONLY the init lifts
end-of-rollout motion 2.2-3.5 -> 7.1 against a healthy baseline of 8.3-8.9, and
hyp_init reproduces the healthy runs' late RECOVERY (5.1 -> 7.1). Neither rolling
flags nor rung sampling helped at all with the bad checkpoint.

**dmd3kl step 100 was ALREADY collapsed (3.5).** The damage happens inside the
first 100 DMD steps -- an earlier checkpoint would not have saved it. Fast, not
gradual.

**Residual gap (7.1 vs 8.3) = the roll-dmd3 recipe vs the holder recipe.** Real,
but second-order next to the checkpoint. Worth a follow-up, not a blocker.

### A MEASUREMENT ERROR I MADE, AND THE LESSON
I first reported healthy runs at "10-22" vs frozen at "3-5" and concluded the
checkpoint swap had NOT worked. That comparison was DEPTH-CONFOUNDED: the healthy
numbers came from 6.5 s clips (2 rolls, early in the ride where motion is highest)
while the tests were 11 s (4 rolls). At matched depth the healthy runs start at
9.6 -- IDENTICAL to hyp_init. **Rollout motion decays with depth in EVERY run,
healthy or not, so any motion comparison MUST be at equal roll depth.**

## BUG: the trainable backbone silently never engaged — FIXED

Root cause was NOT the args/config disagreement I hypothesised (`self.config` IS
`args`, same object). The real cause: `_ladd_run_pair_mode` has **TWO** D-update
implementations and only the MATCHED branch ever read `ladd_defer_disc_update`;
the POSITIONAL branch (`ladd_gt_transition_match=false`) stepped the optimizer
INLINE with no defer read at all (trainer :11797 vs :12321 at HEAD).

Inline => the D-update ran inside the gen-side FROZEN scope => zero adversarial
gradient into the backbone => and because the deferred site is the ONLY caller of
`disc_window_enter`/`pop_telemetry`, zero mode-proof telemetry as well.

**The trap:** removing GAN matching (a CORRECT change, One-Forcing's pairing)
routed the code down the path that silently disabled the trainable backbone.
**7 of 11 arms** would have run the frozen-backbone experiment while every flag
read true.

FIX: one dispatch point for all four `r3gan_optimizer.step()` sites; the flag is
resolved ONCE at construction; it RAISES if trainable=true and a D-update is about
to run inline; mode-proof telemetry published every step so ABSENCE is detectable.
21/21 CPU tests, existing suites green. Propagates to queued jobs (they read the
tree at start) -- no resubmission needed.

---

## SMOKE TRIAGE — 4 BLOCKERS found before the arms ran (2026-08-25)

A read-only triage of 9 smoke logs + 20 wandb runs against the healthy OF control.
**PROVEN working** (positive runtime evidence, not assumption): register readout,
action-critic load (59/59), gen_z_loss, stat/anchor_applied + MEAN_requested==
MEAN_term_built (arithmetic reconciles to stat_anchor_total exactly), [COMMIT-SRC].

### BLOCKER-1 — the smokes and the arms are DIFFERENT BINARIES
The launch scripts already stamp `srcmd5`. Across 9 smokes there are **6 DISTINCT
values, and the current tree matches NONE**. Confirmed from data, not mtimes: a
smoke summary contains keys written ~25 lines away from keys it lacks, in the same
dict. **No smoke certifies the code the arms will run.** The stamp existed and told
us; nobody compared it. FIX: assert srcmd5 equality between smoke and arm.

### BLOCKER-2 — every completed smoke used the OLD init, and the NEW init behaves differently
All completed smokes ran dmd3kl-200; the arms now run ode14e-400. On the arms'
actual init the critic gradient CROSSES the clip ceiling and is still rising:

| run | init | critic/grad_norm @11 | @21 |
|---|---|---|---|
| smk_v6 | dmd3kl-200 | 0.31 | 0.31 |
| **hyp_init** | **ode14e-400 (the arms' init)** | **5.25** | **13.69** |
| hyp_roll | dmd3kl-200 +roll | 9.69 | **167.0** |

`fake_max_grad_norm=10.0`. The "healthy" numbers read off smk_v6 DO NOT TRANSFER.

### BLOCKER-3 — the smokes are structurally BLIND to a dead GAN  [FIXED]
`gan_grad_telemetry_every=25` with `MAXSTEPS=25` => the only qualifying step is 0,
where `gan_disc_start_step=2` means the GAN has not started. `gan_dmd_grad_ratio`
is ABSENT from all 8 completed smoke summaries. In the 241-step v4 run that same
key read **5.16e-05** -- the codebase's own doc calls anything under 1% of the
5-20% band INERT. **FIXED: all 12 smokes now log GAN/texture/backbone telemetry
every step.**

### BLOCKER-4 — 6all-on OOMs on all 8 ranks  [ARMS HELD]
Both attempts died at step ~2 in `register_readout` -> `blk(feat, token)`, 92.66 GiB
allocated. v6b peaks 70.13 GB vs v6's 61.03; 6allon adds a 59.3M-param online
critic on top. `smk-6allfr` has NEVER run. **carntx6all-on and 6all-fr cancelled
pending a memory fix.**

### CORRECTED — the +-12 matching is NOT broken; the CONTROL was
My earlier framing ("complete no-op", implying a bug) was WRONG. Root-caused:

**Our arms are correct.** The candidate range IS [0,12] (13 candidates, all
evaluated); m=0 wins because the student is diverging in CONTENT, not lagging in
TIME, and `min_improve=0.15` correctly refuses to let the target "agree with the
error" -- which is precisely the failure that safeguard was added to prevent.
Measured: our arms' `matched_mae` is 0.33-0.79, the content-dominated regime where
even a REAL 6-frame shift is (correctly) rejected.

**The control's ~25% fire rate is an ARTEFACT of a silently-disabled safeguard.**
Split by geometry, the control fires 0/24 on rolling steps -- same as us. ALL its
fires are on the first roll, where its `dmd_42f_clean_drift_chunks=3` widens the
reserved clean span by 6 frames, pushing the lower bound to 6. That EXCLUDES m=0
from the candidate set, so the reference `_d0` is never computed, so the
`min_improve` guard is skipped entirely and the argmin is accepted unconditionally
however marginal. Five of its eight "fires" are just the forced lower bound (m=6).
Demonstrated: with a student that is a PERFECT positional copy of GT, the control
geometry still returns m=8.

**So there is genuinely NO temporal matching active anywhere in the queued arms**
(FN matching off by the v4-parity instruction; clean-match honestly returning 0),
and `stat_anchor_use_clean_match_offset` / `stat_anchor_match_k` are inert with it
-- the stat anchor anchors to the POSITIONAL GT window every step. But that is the
mechanism working, not failing.

**Shipped (default-off, 128-config byte-equivalence):** `dmd_42f_clean_match_ref_clamp`
closes the control's hole by clamping the reference into the legal range so the
guard always has one. Plus `[42F-MATCH]` now prints argmin/d0/best/ratio/need and a
verdict of accepted / rejected / no-ref(safeguard-skipped) / no-search -- without
which "m=0 because argmin was 0" and "m=0 because a real match was rejected" are
indistinguishable, the exact ambiguity that hid this.

**To actually make matching move the target (RECIPE CHANGES, need sign-off):**
1. `dmd_42f_clean_match_min_shift=3` or `6` -- forces a non-zero offset and moves
   the guard's reference with it. The ONLY option that fires in the rolling
   geometry, where our arms spend nearly all their steps.
2. `dmd_42f_clean_match_min_improve` 0.15 -> 0.03-0.05 -- direct, but re-opens the
   failure the safeguard exists to prevent.
NOT the four drift flags -- those change the clean-half geometry, the teacher RoPE
contract and the counterpart source all at once.

### MAJOR — the console step line can NEVER print GAN numbers, in ANY config
The lookup reads unsuffixed `train/r3gan_d_real`, but every enabled pair-mode gets
a non-empty suffix, so the key never exists. Dead code for all configs. Meanwhile
wandb held `r3gan_d_real_gtxn=4.529` (a runaway disc) and the console showed
nothing. Same for `r3gan_disc_skipped`.

### MAJOR — v6b is NOT a single-flag change
Register mode FORCES `ladd_freeze_projector_mixing=false`, so one flag moves three
things: readout architecture, trainable disc params **3.62M -> 90.55M (25x)**, and
projector mixing frozen->trainable. A v6b result cannot be attributed to "the
readout".

### MAJOR — divergence-3 has ZERO working smoke evidence
Every completed smoke reports `ladd_fake_sample_source_gtxn = 0` (flash, not dmd).
The only smoke that included it OOMed at step 1. Five hard-refusal paths untested.

Plus: `[42F-ALLROLL]` counters are WRITE-ONLY (never read, never reach wandb) and
its print budget burns entirely at step 0 where the counters are structurally 0;
`r3gan_r1_block_fires` is permanently NaN; 9 dead config keys ship in all arms and
the override guard cannot see them because it only scans CLI overrides, never the
base YAML -- **so the guard's silence is not evidence of a clean config**.

---

# MISSED / BROKEN RUNS — inventory as of 2026-08-25 23:35

Every run started today that did not finish, why, and **my recommendation: DO or SKIP**.
Live runs at time of writing: `gansig_ofclean` (step 181), `gantune_w2tclean` (91),
`train_carntx_6t` (166), `gantune_w2gram` (25).

## A. Broken by infrastructure — worth redoing, nothing wrong with the science

| run | reached | why it stopped | verdict |
|---|---|---|---|
| `gansig_of` | ~116/200 | harness killed the background task owning the `srun`; a second attempt died in 64 s (GPU launch race) and **overwrote the log** | **SKIP as-is.** Superseded by `gansig_ofclean`, which fixes its proven defect. Its telemetry survives in wandb. |
| `gansig_huge` | 35/200 | **genuine OOM at 93 GB** in the R1 backward — `max_real=12` at ~1.72 GB/row plus `r1_unified_cadence=true` firing R1 every update | **DO — fixed and queued (6135663).** max_real 12→6, R1 samples 3→2, unified cadence off. Cross-ride pool (256) untouched: it is the point of the arm and costs nothing on GPU. |
| `gansig_real` | 38/200 | died early; served only as a control | **DO — requeued (6135337).** Its value changed: it is the 2-mode *flash* control that isolates the dmd band as the cause of `gansig_of`'s texture overshoot. Needs a step ≥106 reading because the overshoot is late-emerging. |
| `gansig_wide` | never launched | no holder ever freed for it | **DO — fixed and queued (6135664).** Its known hazard (wide_real corrects the action origin on the REAL side only → disc can cheat on conditioning mismatch) is now fixable at the root with `ladd_action_origin_track_slice=true`, which today's code fix provides. |
| `v5b`, `v6f` arms | step 0 | CUDA OOM with **40–55 GB held by a foreign process** — the known holder launch race, not our budget | **DONE — already resubmitted** with the memory mitigations. |
| `gantune_w2carn` | 16/200 | I killed it to free the holder for the corrected wavelet test | **SKIP.** Superseded: every production arm is now CARN-matched, and `6tchain_on` covers the question better. |

## B. Ran but measured nothing — my design errors

| run | reached | what went wrong | verdict |
|---|---|---|---|
| `gantune_w2pix` (PatchGAN probe) | **120/120, completed** | **Two of my errors.** (1) I ran it at `PIXW=0.0` believing the "unweighted" telemetry would still publish the critic's amplitude — it cannot, because the term is multiplied by the weight *before* being folded, so all samples read exactly 0. (2) Its real pool needs 128 warm updates; only 95 occurred, so the critic sat at chance (`pix_d_loss` 1.386 = 2·ln2) the whole run. | **DO, but not yet.** Needs `PIXW=0.01` (then divide out) and ≥160 steps. **Priority below the Gram loss** — the PatchGAN needs two more runs before it is even judgeable, and it is the mechanism most vulnerable to the blocky failure. Its one real result: the pixel path is **cheap** (63.8 GB, +7.6 % step time). |
| `gantune_w2cnx` / `w2dino` | 30 each | I sized them to a 23-minute holder remnant; they stopped before the GAN engaged (step 20) and before their teachers trained | **DONE — resubmitted at 200 steps** (6134657/58/59) with the cadence fix (`updates_per_step` 1→8). |
| `gantune_w2sam` | 115/150 | first attempt OOM'd at 512 res; relaunched at 256; then externally cancelled | **DO — resubmitted (6134659)** at matched cadence so the three-way backbone comparison is honest. |

## C. Dead ends — SKIP, with mechanism understood

| run | verdict |
|---|---|
| `gantune_w2wav` (51) / `w2wav2` (91) | **SKIP PERMANENTLY.** Both drove the disc to *exactly* chance (`d_loss` = ln 2, separation 0.0004). Cause: the Haar **HH** band has no analogue in smooth WAN latents, so it is common-mode across real and fake and swamps the directional bands. Dropping HH only delayed collapse 51→91. Also explains the researcher's "smoother, less resolution": an edge-band critic is cheapest satisfied by suppressing HF. |
| MAE gate | **SKIP.** Never built. Its ratio is a **rollout-depth switch, not a student-mastery signal** (depth 1 median 1.373, depth 3 median 0.9935 — a clean 0%/100% split). It would zero DMD on exactly the drift chunks. |
| `gantune_base` / `_r1` / `_bal` (all ~31–50) | **SKIP.** I stopped them when the researcher asked for a stronger GAN. The R1-gamma question they were testing has since been decided (gamma 10 is now the default); the other two axes do not touch style. |
| `v6na`, `v6seam`, `6all-fr`, `6all-on` | **SKIP — cancelled by researcher decision**, not failure. |

## D. THE ONE-FORCING AXES — status of all five divergences

| # | divergence | state | verdict |
|---|---|---|---|
| 1 | **Trainable fake-score backbone** (disc reads the fake-score critic, not a frozen teacher) | **LIVE in every arm** (`grad_scale=0.2`) | Working. **But its MAGNITUDE is untested** — `gansig_of` carried a deliberate 2.5× over-kick probe and **I removed it**, mistaking a documented treatment for a bug. **DO: restore a divergence-1 arm.** Cheap, one flag, and it is currently the only divergence with no probe at all. |
| 2 | **Register-token cross-attention readout** | `v6b` queued, never run at length; its 75-step smoke passed at 74 GB | **DO** — it is in the production queue. Note its activation checkpoint disarms on the generator side (~34 GB), which is why the 6all arms OOM'd; the code fix exists (`ladd_gside_checkpoint_recover`) and is enabled. |
| 3 | **Disc scores the DMD-sampled output** (`fake_sample_source=dmd`) | Tested via `gansig_of` — **confirmed working but compromised** | **DO, but only in the `ofclean` form.** Proven defect: at the band's t=1000 rung `alpha_t = 1−sigma = 0`, so the student's sample is multiplied by **zero** and the generator gets no adversarial gradient — 3 of 8 samples. `force_clean` fixes it and is now the codebase default. `v6alt` (queued) inherits the fix automatically. |
| 4 | **Unmatched pairing** | **We deliberately do the OPPOSITE** | **SKIP.** Matching is the only gate that lets the CARN apply; unmatching it silently disables the noiser. Also, the matching we had was a random draw (pool collapse) until fixed today. |
| 5 | **Critic-refreshed features** | **LIVE** | Working, no action. |

## E. My overall recommendation on priorities

1. **Finish what is running** — `ofclean` (181), `w2tclean` (91), trinity (166), and the **Gram loss** (25), which is the only mechanism that measures dataset style *directly and online* rather than by adversarial proxy.
2. **Then the two new stacked arms** — `gansig_ofcleancarn` and `train_carntx_6tchain_on` (both built, both now carrying the clean-disc default).
3. **Then `huge` / `wide` / `real`** — queued and fixed; they address disc *memorisation*, which matters but is not the style bottleneck.
4. **Then divergence 1** (restore the probe I removed) and the **PatchGAN** properly probed.
5. **Never** — wavelet, MAE gate, the three stopped ladder axes.
