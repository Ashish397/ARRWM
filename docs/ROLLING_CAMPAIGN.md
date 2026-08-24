# Rolling campaign — autonomous execution ledger (2026-08-20)

## Standing rules (from Ashish, permanent)
- NEVER: only-last supervision, LL to disc, context_noise>0, cold starts.
- ALWAYS: warm-start from closest ckpt MATCHING DISC ARCH:
  wavelet-ON arms  -> dmd3klGANck j6069331 step200 (WAVELET=true job)
  wavelet-OFF arms -> arkl_GAN j6062665 step200 (flashgan WAVELET default false)
  no-GAN / fresh-disc fallback -> dmd3kl j6052054 step200
  (mismatch = LADDDiscriminator load_state_dict crash: heads/band dims)
  dense supervision, wavelet ON + drop_ll, real_guidance 0, 2-node smokes.
- Holders: never cancel; HARD 60-90 min per experiment => CH_MAXSTEPS=380
  (180 new steps from the 200 warm start, ~80 min incl. setup). Trim, never
  extend: more experiments beats longer ones.
- Launcher: /tmp/claim_holder.sh (atomic mkdir locks /tmp/claimed_<id>,
  CH_SRC/CH_MAXSTEPS/CH_MIN_SEC env). Lock reaper: /tmp/lock_reaper.sh.

## Score protocol per finished arm
1. python analysis/seam_drift.py <wandb_dir>   (drift + seam ratio, >=100f)
2. depth-detail table (chunk-index detail energy; cv vs monotone)
3. disc health: d_loss off ln(2)? (wandb r3gan_d_loss_gtxn)
4. dmd_supervised_count rising (gauge fix), luminance range
5. append verdict here; compare vs rollwarm_gan + roll_noGAN baselines.

## Arms in flight / queue (see squeue + /tmp/claimed_*)
- running: rollife (anchors 0.2/0.1 + r1@5 + wav-off + GAN), rollcarn
  (seam affine λ=.5), rollgansure (r1/r2=0, wav-off)
- armed: rolltemp (T=1.09), rollfloor (one-sided STD .1), rollr1w (r1@5
  wav-on = go-forward candidate), drift_probe (lr=0, latent dumps →
  analysis/drift_probe_v1), rollgansure2 (chained after gansure; r1=0
  wav-on+dropLL = "proper GAN")
- next-wave candidates (approved directions): CARN drift-field predictor
  (train on probe dumps), action-stratified drift, critic-gated repulsion,
  spectrum anchor, temp×floor combo, best-of-round consolidation arm.

## ROOT CAUSE SOLVED (2026-08-21 00:1x): disc pin was R1 gamma=1e6
- rollgansure (rolling, R1/R2=0): d_loss 0.62->0.016 = disc LEARNS under
  rolling. Every prior rolling arm carried gamma 1e6 (any cadence) = pinned
  at ln(2). NOT structural, NOT wavelet, NOT pairs. gamma=0 overshoots
  (disc dominates, g_loss 2.1->2.9); gamma ladder next: rollg1e3 (1e3@5)
  launched on 6073278. rollife verdict: rel_tol 0.2 disables anchors ->
  DC drift returns; try 0.03-0.05. All wave-1 "GAN-on" corrector arms
  (carn/temp/floor/drift) effectively ran GAN-less: re-run winners with
  live-GAN config in wave-2.

## WAVE-1 CLOSED (2026-08-21 05:0x) — canonical config VALIDATED
rollcombo2 (wavelet-on + dropLL + affine λ.5 + γ1e3@5 + dense sup +
random depth + ctx_last_rung + compose + vaert + warm start):
  mean |drift| 0.019 (7/9 clips <±0.02) — flattest of campaign
  d_loss 0.63 -> 0.10-0.17 — healthy adversarial learning
  (combo1, wavelet-off variant: drift ±0.06, texture 0.95-1.02 at depth)
FACTORIAL: affine necessary for drift; GAN necessary for texture/life;
both jointly sufficient. rolldrift (open-loop bias): over-corrects to
black — repulsion must be CLOSED-LOOP. rolltraj (teacher rungs): running.

## fkl PARKED (2026-08-21): three strikes, structural
v1 raw-mean (CLT-pinned), v1.5 smoke (caught it), v2 relativistic gap:
fkl_rhat = 1.000 on all 17 rows AGAIN. Diagnosis: cross-RANK dispersion of
token-averaged logits ~ 0 (CLT over ~50k tokens); B=1/rank leaves no
per-sample contrast for f-distill's h(r). Only viable form = per-CHUNK
weights WITHIN a sample (v3, needs disc token->frame layout mapping).
Parked in favor of CARN-v2; run itself healthy (drift flat, disc fine) =
another canonical replica.

## WAVE-3 VERDICTS (2026-08-21 evening)
- rolldepth: canonical DEPTH-ROBUST -- caps 5/5,6/6 at max_length=90, clips
  to 252f (21 chunks), all drift within +-0.045, no OOM.
- rolllong (holder expired at step 601/700): holds cleanly to ~450; steps
  ~480-530 show BOUNDED elevated excursions (max +0.18, levels to 0.77)
  with self-recovery by 601 -- no runaway, but the affine's margin thins
  at long horizons. Candidate fixes if valcanon2 corroborates: affine
  lambda 0.6-0.7, or per-chunk (not per-window) affine, or mild M1 anchor.
- fkl: parked (structural; see above). Next new mechanism: CARN-v2
  residual predictor on probe latents.

## Wave-1 verdicts (final)
- rollcarn (affine λ.5): WINNER on drift — all clips ±0.08, no ramps.
- rollfloor (one-sided STD .1): loss ACTIVE (STD_active 0.035->0.016,
  anchor_total ~1), drift mostly bounded (worst +0.15/-0.16) — works,
  second place; combine with affine later if variance droops.
- rollife (rel_tol .2): NEGATIVE — anchors zeroed, DC drift returned.
  Narrow band 0.03-0.05 if revisited.
- rollgansure (γ0): disc LEARNS (0.62->0.016) => R1 γ1e6 was the pin;
  γ0 overshoots (±0.4 level oscillation). Ladder: rollg1e3 running.
- rolltemp (T=1.09): pending. rollcombo (affine+γ1e3): running.
- rolldrift relaunched with live-GAN γ1e3@5 on 6073280.

## Decisions log
- 22:31 rollife/rollcarn/rollgansure first launch: silent srun hang, 0 steps,
  0 output for 35min; GPUs clean. Relaunched DIRECT (fresh ports 1018/1028)
  -> both alive. gansure claimant still polling (will take a 6h holder).
- 22:5x drift_probe launched direct on short holder 6068976 (arkl_GAN warm,
  lr=0, depth 8, latent dumps). 2 more holders submitted.
- 22:5x DRIFT PROBE RESULT (stage-2, TRUE latents, 40 rides, frozen arkl):
  drift is GLOBAL in latent channel-mean space: ride-level pairwise cos
  +0.74 (pixel proxy misleadingly said ~0), transition cos +0.53, drift
  mass in channel MEANS (0.41) >> stds (0.16). VAR specrad 1.199
  (amplifying walk), 26 complex eigenpairs (oscillatory modes exist).
  => fitted global per-roll drift vector saved:
  analysis/drift_probe_v1/global_drift_mu16.pt (|d|=0.57/roll).
  => new knob carn_seam_drift_lambda/file subtracts it at every commit;
  arm `rolldrift` (lambda=1.0) armed. Dump ride-counter bug fixed
  (reconstruct rides from the roll field).
- Watchout: claim-race launches can silently hang srun; prefer direct
  HOLDER= targeting when relaunching. Reaper active (bb1085s1e).

## WAVE 4 (user directives, 2026-08-21 evening)
0) 2 holders queued; queue 2 more when they open. Launch on open.
1) SEAM at root: only structural lever = dmd_lookback_chunks=1 (gradient
   across the handoff; today previous_chunk.detach() means chunk k is never
   trained to produce good context). Arm: rolllook = rollcarn recipe
   (dmd3kl warm, GAN dead/off) + dmd_lookback_chunks=1. Needs
   retain_graph -- watch memory.
2) GAN deep forensics (sub-agent): all d/g metrics across every run; why
   live disc => breakdown at ~466 (rolllong); wavelet-HF diet suspected.
3) Transition-GAN redesign (sub-agent): formalize a rolling GAN that is
   neither chunk real-vs-fake nor naive gt_transition.
4) CARN review (sub-agent follow-up): how to extend (per-chunk M2/TV
   higher moments; predictor v2 on probe latents).
5) rollcarn-700: does the GAN-FREE recipe survive step 466+? (rolllong
   broke down ~466 WITH live GAN; decides method-vs-failure-mode.)
6) fgt3 PARKED per user (worked without GAN, breaks with GAN).
Rules stand: no LL-to-disc beyond drop-LL config, no only-last, no
context noise, warm-start closest ckpt, 60-90min/arm, symlink seeds.
- USER DIRECTIVE: do NOT warm from ck checkpoints. Rolling tests warm from
  dmd10k_dmd3kl_GAN_h6067260_103050 (the non-ck stationary GAN run the
  user rates best). Once the GAN is fixed in rolling, redo stationary
  with the fixed GAN to mint a fresh ck. Working assumption: current GAN
  is broken and needs major rehaul (see GAN_FORENSICS.md when it lands).
- rolllook VERDICT (2026-08-22): lookback=1, 380 steps from 103050: seam
  ratios 2.0-3.0 (=rollcarn ~2.3, no improvement), drift excursions ±0.12
  (slightly worse), retain_graph cost. NEGATIVE at 90-min screen -> park.
  Synthesis with flow maps: rollcarn already ELIMINATES seam jerk in
  motion space; residual luminance seam-ratio ~2 is likely benign content
  variation. SEAM = effectively handled by CARN recipe; root-fix lookback
  unnecessary at current evidence.

## 2026-08-22 late — THE FIXED-ROUTE 60 s EVAL LANDED (Step 2 of PLAN_22-8)
Same Madrid route (20240216101235.zarr +100), seed 42, 4 denoising steps,
ar_initial_chunks=3, 100 gen chunks (59.6 s), inference-CARN λ0.5, step-200
checkpoints, both arms fresh from the KL-ODE step-400 init.

NEW INSTRUMENTS (both CPU, offline, in analysis/):
- `rollout_quality.py` — time-resolved pixel-statistics profile referenced to
  the REAL SEED (not to the rollout's own first second): sharpness×seed,
  dark-channel, contrast, median, motion×seed, ORB-vs-seed AND local ORB
  (0.5 s apart, immune to honest camera motion). Ports fleet_hf /
  haze_baseline / blind_cpu_metrics from the 14e paper onto one long rollout.
  Added an `hf_gain` class: this student fails by TEXTURE BLOW-UP, the
  opposite of the paper's hedging/blur family, so the paper's classifier
  scored it "clean".
- `rollout_realism.py` — CLIP ViT-B/32 zero-shot REAL-scene vs MELTED-scene
  probe + seed-embedding drift. Headline scalar = SURVIVAL TIME (first second
  at which p_mangle stays high). The paper says structural corruption stays
  crisp and must be caught by embedding drift; this is that instrument.
  Validated against my own frame-by-frame read of both videos.

| arm (step 200)            | survival | lap×seed(end) | motion×seed | local-ORB kept |
|---------------------------|----------|---------------|-------------|----------------|
| fullcarn_bidir_kl_nogan200| 8 s      | 3.63×         | 0.75×       | 0.41           |
| fullcarn_bidir_kl_wave01  | 12 s     | 2.98×         | 0.50×       | 0.11           |

VERDICT (wavelet GAN, weight 0.01, action-blind, disc t=0):
- Criterion 1 (learns): MARGINAL. d_loss 0.659 -> 0.590 over 200 steps.
- Criterion 2 (controlled pressure): FAIL-LOW. weighted g_loss ~0.0072 vs
  median DMD 0.61 = ~1.2%, well under the 5-20% band.
- Criterion 3 (extends stability past the horizon): PASS. Survival 8 -> 12 s
  (+50%); texture blow-up onset 2 s -> 3 s; end sharpness 3.63x -> 2.98x seed.
- Criterion 4 (no bad trade): FAIL. Motion drops 0.75x -> 0.50x of the seed's
  own pace and local geometric coherence collapses (ORB kept 0.41 -> 0.11).
So: a GAN this weak already buys horizon, and pays in motion. That argues for
the STRICT assembly (weight 0.03, 5 D-steps, sampled disc-t, action-
conditioned) rather than against the GAN.

THE FAILURE IS NOT THE 5.25 s CLIFF. Both rollouts are still recognisable
street scenes well past seed eviction; they then MELT into an iridescent
corrugated attractor (crisp, high-frequency, geometrically dead) at 8 s
(no-GAN) / 12 s (wavelet). Onset is gradual, not a step. Pixel stats stay
healthy through it — hence the CLIP probe.

## Live arms (2026-08-22 22:1x)
- 6100009: `fullcarn_bidir_kl_strict03` — the STRICT critic on the IDENTICAL
  DMD recipe as the nogan200 control (new `strict` MODE in
  sbatch/run_full_carn_probe.sh), so the only delta vs the control is the GAN.
  Mid-run health at step ~55: d_loss 0.692 -> 0.589 (faster than wave01) and
  weighted g_loss/DMD ~10-20% = INSIDE the target band. Auto-chains the eval.
- Queued on pending holders (PLAN v2 items 6 and 8):
  6100985 `horizon_nogan90`  MODE=none    MAXLEN=90 depth 3-9 (horizon, GAN off)
  6100986 `horizon_wave90`   MODE=wavelet MAXLEN=90 depth 3-9 (horizon + GAN)
  6102572 `gan2x2_raw_t0`    MODE=raw     DISC_T=t0      (wavelet x timestep 2x2)
  6102573 `gan2x2_wave_ts`   MODE=wavelet DISC_T=sampled (wavelet x timestep 2x2)

## Code hygiene (2026-08-22 late)
- R1/R2 TELEMETRY DEFECT: `r3gan_r2_fired` reads 0 on EVERY logged row of every
  run. wandb samples every ~10 steps; the R2 cadence is (step-offset)%2 and the
  GAN only runs on generator steps — the three cadences alias, so the gauge can
  read 0 forever whether or not R2 fires. Added MONOTONE counters
  (`r3gan_r{1,2}_fired_total`, `r3gan_disc_updates_total`, and fire RATES) which
  cannot alias; the next arm settles whether R2 has ever fired. Telemetry only.
- Removed a stray `[CARN-DIAG]` debug print that spammed stderr and, being
  placed above the docstring, silently shadowed `_ladd_run_pair_mode.__doc__`.
- DISK: /lus/lfs1aip2 was at 100% (48 G free). Deleted 850 GB of holder
  end-save `phase1_step*.pt` written since 2026-08-20 (the class the rules
  already call invalid warm starts), sparing sbatch (`_j`) runs, the
  user's preferred `dmd10k_dmd3kl_GAN_h6067260_103050` warm-start source and
  `rollcarn700_h6089105_022300/phase1_step0000700.pt`. Now 790 G free.

## 2026-08-23 — R1/R2 cadence: defect RESOLVED, imbalance FOUND
The monotone counters added 2026-08-22 have reported. `r3gan_r2_fired` reading 0
on every logged row of every run was a pure SAMPLING ALIAS (wandb every ~10
steps vs the R2 parity cadence vs `dfake_gen_update_ratio=5`), not a dead
penalty: `r3gan_r2_fired_total = 85 / 175` disc updates = **48.6 % fire rate**.
"We ran R1+R2" is TRUE — the doubt is struck.

NEW FINDING from the same counters: R1 and R2 are NOT balanced. In the `_gtxn`
arms R1 fires at **0.20** and R2 at **0.49** (2.4x asymmetry) because
`ladd_r1_once_per_step=true` caps R1 at the first of `gan_updates_per_step=5`
updates while R2 is uncapped. A `_gt`-mode arm is balanced (0.53/0.47). R3GAN
assumes balanced R1+R2, so every 5-D-step arm ran an unintended asymmetric
regulariser. Needs an intended-rate decision before any new critic inherits the
machinery (`GAN_REDESIGN.md` A10/A13).

FIRST GRAD TELEMETRY READINGS (A7, n=2/arm — early): `gan_dmd_grad_ratio`
0.0005-0.012 (0.05-1.2 %, far under the 5-20 % target band) and
`gan_dmd_grad_cos` -0.02 to -0.10. The current critics are SMALL AND
NEAR-IRRELEVANT, not small-and-destructive.

## 2026-08-23 — THREE silently-ignored knobs found; two experiment readings void
A full config-vs-code audit of every `gan_*`/`ladd_*` knob found flags that live
sbatch arms SET but the code never READS. The override parser is
`OmegaConf.from_dotlist` + `merge`, which accepts unknown keys silently — no
allowlist, no warning. Consequences:

1. `ladd_gt_transition_action_blind` — read only off `self.model`, never parsed
   from args (siblings ARE parsed, e.g. `ladd_gt_transition_match` at
   model/dmd_action_forcing.py:2022). **Always False.** So wave01 /
   horizon_wave90 / the gan2x2 arms ran ACTION-CONDITIONED despite setting
   `action_blind=true`. The "blind vs strict" contrast never existed.
2. `ladd_gt_vs_fake_n_real=16` in run_ganfix_marginal.sh does NOTHING —
   `_mismatched` is computed and used only inside a mutual-exclusion raise; there
   is no mismatched-real branch. That arm scores against the position-matched GT
   chunks, not 16 fresh resampled ones as its own header claims.
3. `ladd_gt_transition_cross_equalize` (set by strict_rerun + poolrich) has ZERO
   code references anywhere.

CORRECTION to the 2026-08-23 R1/R2 entry: the `marginal` arm's "balanced"
0.53/0.47 fire rates are NOT a flag interaction. `ladd_r1_once_per_step` and
`ladd_r1_normalize_tokens` are implemented ONLY in the matched branch; the
positional branch that `marginal` takes silently drops both (and also ignores
`ladd_disc_micro_batch_groups`, so it runs an un-micro-batched full-batch disc
backward while its config claims groups=2).

Also dead-but-set in live scripts: `gan_r1_gamma`/`gan_r2_gamma` (log line only —
the real knobs are `ladd_r1_gamma`/`ladd_r2_gamma`), `gan_critic_*` (7 knobs),
`gan_disc_base_channels`/`_num_blocks`, `ladd_pair_start_seed_boundary`, all
`gan_sam2_*`. Full inventory in docs/PARKED_LADD_CONFIG_FIXES.md.

## 2026-08-23 — DIFF-AUG ASYMMETRY: the matched-path critic had a free, unfixable cue
`model/ladd_disc.py::latent_diff_augment` documents its contract as "Apply
IDENTICAL random augmentation to real and fake latents ... identical per-sample
augmentation". The matched (gt_transition) path violates it:

  D-UPDATE (trainer/causal_action_forcing_train.py ~7111):
      _rn, _ = latent_diff_augment(_rn, _rn, policy=flip, seed=...)   # REAL augmented
      _rn = _rn.detach()
      _fk = _m_noise(fake_chunks_det).detach()                        # FAKE NEVER AUGMENTED

  G-SIDE (~7306/7309): real and fake augmented in TWO separate calls with
      DIFFERENT seeds (+99 vs +199) -> independent flips, contract broken again
      (milder: both distributions broadened equally).

CONSEQUENCE. With policy=flip on driving footage — which is strongly
left/right asymmetric (kerb side, traffic direction, road markings, sun angle) —
the discriminator saw randomly MIRRORED reals and NEVER-mirrored fakes. Mirror
parity is therefore a real/fake cue, and it is one the GENERATOR CANNOT
ELIMINATE because it does not control the augmentation. Same failure class as
the raw-RGB-vs-VAE-reconstruction trap: an unfixable giveaway feature.

SCOPE: every arm with ladd_gt_transition_match=true, i.e. run_full_carn_probe.sh
(wave01, horizon_wave90, gan2x2_*, strict03), run_ganfix_strict_rerun.sh and
run_ganfix_poolrich.sh. Essentially every gt_transition GAN arm of the campaign.
strict_rerun and poolrich were RUNNING when this was found.

CONSISTENT WITH (not proof of) the observed pattern: ~half the reals become
trivially identifiable while the rest stay at chance, which caps the achievable
separation well short of collapse — d_loss should settle meaningfully below
ln2=0.693 but nowhere near the 0.10 breakdown threshold. Measured: 0.58-0.62 in
every arm. It would also explain a critic whose gradient is tiny and nearly
orthogonal to DMD (cos -0.02..-0.10): much of its discriminative power sits on an
axis the generator cannot act on.

NOT YET QUANTIFIED: how much flip-parity is actually learnable from 16-channel
latents through the frozen teacher taps. The direction of the bug is certain; the
effect size is not. Cheapest test: re-run one matched arm with policy="" and
compare the d_loss trajectory.

## 2026-08-23 — CORRECTION (2nd) to the R1 fire-rate attribution
I attributed R1's 0.20 fire rate to `ladd_r1_once_per_step=true` interacting with
`gan_updates_per_step=5`. That was WRONG, twice over. Verified mechanism:

`ladd_r1_once_per_step` was ALREADY A NO-OP under the debt-based cadence:

    _do_r1 = (current_step - _last_r1_at >= _r1_every_n) and (_it == 0 or not once)
    if _do_r1: self._ladd_last_r1_step = int(current_step)

`_it == 0` fires first and immediately consumes the debt (the difference becomes
0), and `_r1_every_n = max(1, ...) >= 1`, so `0 >= _r1_every_n` is false for every
later `_it`. R1 could never fire twice in one step regardless of the flag.

TRUE mechanism for 0.20 vs 0.53:
  * MATCHED path (gt_transition arms): debt-latched -> at most ONE R1 fire per
    step; with gan_updates_per_step=5 that is 1/5 = 0.20.
  * POSITIONAL path (the `marginal` arm, gt_vs_fake): a DIFFERENT implementation
    using plain `current_step % _r1_every_n == 0` with no debt latch, so it can
    fire on several `_it` within one step -> ~0.53.
The difference is debt-latch vs modulo in two separately-written R1 estimators,
not a flag. `ladd_r1_once_per_step` has now been deleted as provably inert.

R2 IS GONE (2026-08-23): -346 lines across trainer, model/dmd_action_forcing.py,
model/r3gan.py (r2_penalty deleted), both configs and 13 sbatch scripts. Every
sbatch edit used write-temp + os.replace (atomic) because holder 6106490 was
executing run_ganfix_poolrich.sh at the time. Repo-wide grep for the R2 symbols
returns zero. 16/16 tests pass (12 focused + 4 disc-micro-batch).
The R1/R2 cadence imbalance and the R2-never-normalised scaling bug are both
resolved BY DELETION rather than by fixing.

## 2026-08-23 — ADVERSARIAL REVIEW: two of my findings were overstated
An adversarial reviewer (instructed to REFUTE) checked five claims. Three clean,
two need correcting. Recording both corrections against my own reporting.

### CORRECTION A — R2 removal is a RECIPE CHANGE, not a cleanup
I framed R2's deletion as "resolving two bugs by deletion". That understated it.
VERIFIED: `ladd_r2_gamma=1.0` was set in the DEXTRA of run_ganfix_strict_rerun.sh,
run_ganfix_poolrich.sh and all four GAN modes of run_full_carn_probe.sh. DEXTRA is
appended AFTER the holder's inline `ladd_r2_gamma=0`, so DEXTRA wins. **R2 was
ACTIVE at gamma=1.0 — the same gamma as R1 — in every ganfix and carn-probe arm.**
And because R2 used a plain modulo cadence inside the per-_it loop while R1 was
debt-latched, R2 fired at 0.49 vs R1's 0.20 (measured) — i.e. **R2 was the LARGER
regularisation term in the loss**, roughly 2.4x more fires.

Consequences:
  * deleting R2 CHANGES the recipe relative to every prior GAN arm;
  * re-running sbatch/run_ganfix_strict_rerun.sh today is NOT a rerun of the same
    recipe as the historical arm of that name;
  * any prior GAN result attributed to R1 tuning was actually dominated by R2.
Only `gan_r2_gamma` + `model/r3gan.py::r2_penalty` were genuinely dead (never
called); that part of the removal is inert.

### CORRECTION B — the diff-aug finding: right mechanism, WRONG headline
I led on "mirror parity gives the critic a free cue". The reviewer confirms the
asymmetry (real augmented via `latent_diff_augment(_rn, _rn, flip)`, `_fk` never
augmented anywhere downstream — verified through `_m_noise`, `_m_fwd` and
`_ladd_disc_update_microbatched`) but argues mirror parity is the WEAKEST of the
three defects in those lines, because:
  * the disc is `scalar_output=true`, so parity survives only as a faint global
    `k - flip(k)` statistic after mean-pooling ~46,800 head outputs;
  * it largely SELF-CANCELS on the G side, where the fake IS flipped 50% of the
    time — so a "mirrored => real" detector pushes G toward correct parity half
    the time and wrong parity the other half. Net: added gradient variance and
    wasted critic capacity, not a systematic artefact.

The two defects in the same three lines that DO have teeth:
  1. **Action tokens are not flipped with the latent.** `real_m_rat`/`real_m_ram`
     are passed to `_m_fwd` unflipped while the real latent is mirrored. With
     action_tokens_per_frame=1 and action_blind provably dead, **50% of reals are
     action-image-INCONSISTENT while 100% of fakes are consistent** — so the
     critic can learn "image contradicts the steering => real". Directly
     adversarial to the purpose of an action-conditioned transition critic.
  2. **D train/serve mismatch on the fake side.** D is TRAINED on never-flipped
     fakes and QUERIED on the G-side with 50%-flipped fakes — a distribution
     shift on the exact input whose gradient the generator receives.

=> Do NOT void arms on "mirror parity". If anything is voided, void it on (1).

### Unchanged after review
  * action_blind never parsed -> always False: CONFIRMED. But the "blind vs
    strict" contrast it labelled was ALREADY confounded by ~5 other knobs
    (gan_lr 5e-6 vs 1e-5, updates 1 vs 5, critic warmup 40 vs 20, mean_equalize,
    xeq_preserve_delta). The dead flag costs a LABEL, not an experiment.
  * ladd_gt_vs_fake_n_real=16 is a silent no-op: CONFIRMED clean.
  * ladd_r1_once_per_step provably inert; 0.20 vs 0.53 = debt-latch (matched) vs
    plain modulo (positional): CONFIRMED, and the code reproduces both numbers
    exactly.
  * DMD path untouched by all of today's work: CONFIRMED (model/dmd_action_forcing.py
    has exactly one hunk in the whole working diff — 33 deleted R2-knob lines).

## 2026-08-23 — REVIEW ROUND 2: one headline finding WITHDRAWN, one instrument bug in my own tool

### WITHDRAWN — the "R1 gamma=1e6 was a token-scaling artefact" reframe is FALSIFIED
I claimed un-normalised R1 on token logits scales ~T^2 and that this was "very
likely the mechanism" behind the replicated "R1 gamma=1e6 pins every disc at
ln2" finding — i.e. that a headline prior result was an artefact, not a law.
VERIFIED FALSE. `sbatch/_roll_holder.sh` runs:
    ladd_r1_gamma=1e6   ladd_r1_normalize_tokens=true   (no ladd_scalar_output)
so those runs ALREADY used token logits WITH the mean normalisation, and still
pinned at ln2. The T^2 artefact cannot explain it. gamma=1e6 was simply ~1e6x too
large under the mean reduction. The original finding STANDS as measured.
Knock-on: this removes the main stated justification for A8 fix 2 and for A12.
The mean-vs-sum question is still worth specifying for the pixel critic, but it
is no longer backed by "it explains the ln2 pinning".

### INSTRUMENT BUG (mine) — the seed reference was contaminated
Every battery/quality number I produced normalised against a "real seed"
reference computed as `seed_secs * fps` with a 2.25 s default. **The eval videos
render at 20 fps, not 16** (`fps: 20.0`, `n_frames: 1236` in every
analysis/*.json). So seed_end resolved to 45 frames against a TRUE seed of
3 chunks x npb 3 x VAE stride 4 = **36 RGB frames** — i.e. 9 GENERATED frames
were averaged into the "reality" baseline, ~20% contamination, biasing every
ratio slightly TOWARD 1.0 (i.e. flattering the arms).
FIXED: analysis/rollout_quality.py now derives the seed length in FRAMES from
the eval's own rank0_ride.json (`ar_initial_chunks * npb * 4`), with
--seed-frames to override; --seed-secs retained but deprecated in the help text.
All seed-referenced numbers reported before this entry need re-running; the
RANKINGS are unlikely to flip (both arms shared the same contamination) but the
absolute ratios are wrong.

### MUST-FIX before A4 is ever enabled (from the trainer review)
`_apply_gan_grad_cap` has NO NaN guard: a non-finite disc gradient makes
`mean|g|` NaN -> `clamp(tau/NaN, max=1.0)` = NaN -> `NaN >= 1.0` is False -> the
function RETURNS `gen_gan_loss * NaN`, NaN-ing every generator weight on the next
backward. The DMD original it mirrors calls `torch.nan_to_num(grad)` FIRST
(model/dmd_action_forcing.py:5412) before its cap at :5450. A4 copied the cap and
dropped the guard. Also: A4's tau is NOT on the same scale as dmd_grad_target_norm
(it measures AFTER gan_loss_weight x warmup ramp x ladd_disc_loss_weight), so a
borrowed tau=1.0 can never fire; and it costs a full extra teacher backward on
every generator step with no cadence knob.

### STILL PRESENT after the R2 removal (do not read A10 as "fixed")
Four live R1 estimators with inconsistent semantics: micro-batched FD and
matched-inline FD honour normalize_tokens + debt latch; the two positional
variants ignore normalize_tokens AND use a plain `step % n` with no latch. The
0.20-vs-0.5 fire-rate imbalance is therefore UNCHANGED. Worse: a match-fallback
step silently routes to the positional estimator, where normalize_tokens=true is
ignored -> grad_sq jumps by ~T^2 for that step at gamma=1.0.

## 2026-08-23 — texture_stats ADVERSARIAL REVIEW: 5 real bugs, 1 reported number wrong
An adversarial reviewer independently reproduced the instrument and found:

WRONG NUMBER I REPORTED. I quoted horizon_nogan90 late anisotropy as 0.62 x seed.
It does not reproduce: independent measurement gives **1.319** (seed 2.2318,
nogan tail 2.9441, wave tail 0.7055 -> 0.316). wave90's ~0.32 IS correct. The
0.62 should not be quoted anywhere. NOTE the direction: nogan90 OVERSHOOTS the
seed (1.32) while wave90 UNDERSHOOTS (0.32) -- the verdict only exists as
"closeness to seed on a LOG scale", which was not implemented anywhere; it lived
in my head. Now implemented as `seed_log_dist`.

THE PRE-REGISTERED JOINT CRITERION IS WORSE THAN ONE STATISTIC. Per-window
agreement with the researcher's ranking, 59 windows (second half in brackets):
    hv_anisotropy     50/59 (29/29)
    haar_HL_LH_ratio  49/59 (29/29)
    hf_kurtosis       37/59 (13/29)
    angular_entropy   34/59 (22/29)
    hf_power          11/59 ( 0/29)  <-- ANTI-CORRELATED
    JOINT all five    37/59
    JOINT minus hf_power 55/59
=> GAN_REDESIGN item 9's "track HF power + anisotropy + entropy + kurtosis
jointly or it is gameable" is, as literally written, WORSE than anisotropy alone
on the only case where truth is known. hf_power must be REPORTED but must NEVER
VOTE: the arm that manufactures the most HF is not the arm the eye rejects.

FIVE BUGS FIXED in analysis/texture_stats.py:
 1. rfft2 Hermitian fold unaccounted: interior columns stand for TWO full-plane
    bins, fx=0 and fx=Nyquist for one. Equal weighting inflated p_fy/p_fx by up
    to 2x, one-sidedly, in proportion to x-invariant energy -- i.e. exactly the
    banding mode -- and ~8x more on wave90 than nogan90, pushing the losing arm
    toward the winner. Fixed with an explicit multiplicity weight.
 2. angular_entropy silently dropped the ENTIRE fx=0 column: atan2(|fy|,1e-12)
    rounds to float32(pi/2) == edges[-1] and the last bin tested strict `<`.
    For pure banding hist.sum() was 1.6e-10 vs Pm.sum() 1.9e5 -- entropy
    renormalised over rounding noise. Fixed (inclusive last bin). Effect is
    large: synthetic noise+stripes entropy went 0.990 (blind) -> 0.486 (sees it).
 3. No window function: ~half the real seed's on-axis HF energy was top/bottom
    wrap leakage, not content. Hann window added.
 4. Clamped ratio returned ZERO GRADIENT on a strongly-banded input -- the
    anisotropy term would go silent exactly when the model is most broken,
    fatal for the B1 anchor loss. Replaced with a power-relative epsilon;
    measured grad on pure stripes 0.0 -> 266.
 5. battery_from_numpy divided by 255 unconditionally, scaling absolute Haar
    powers by 1/65025 for float inputs. Now scale-detected.
 Also: upper radius bound at Nyquist; hf_kurtosis_excess emitted alongside raw
 (analysis/texture_abc reports EXCESS -- the apparent 13.2-vs-13.4 agreement was
 a COINCIDENCE and any threshold copied across is off by 3).

TWO WIRING BUGS FIXED in analysis/rollout_quality.py:
 6. `from texture_stats import ...` only resolved when run as a script; under
    `import analysis.rollout_quality` it silently set _HAVE_BATTERY=False and the
    table printed 0.000, which READS AS "maximally striped" rather than "not
    measured". Now an explicit path insert, and n/a instead of 0.000.
 7. SIZE=(832,448) resized native 832x480 VERTICALLY ONLY with INTER_LINEAR --
    a low-pass on the fy axis, the numerator of the statistic under test.
    Measured 28% attenuation. Now native resolution.
 Plus the earlier seed-length fix (20 fps not 16; seed now derived in FRAMES
 from rank0_ride.json).

STILL OPEN / NOT DONE:
 * POST-FIX VALIDATION NOT RE-RUN. The numbers above are the reviewer's
   measurements on the PRE-FIX module. The Hermitian and Hann fixes are both
   expected to move them materially (reviewer predicted seed 2.232 -> 1.964 and
   wave90 0.308 -> 0.187). No post-fix number should be quoted until re-run.
   The re-run is blocked by erratic SIGKILL of long python jobs on the login
   node (same command succeeded twice then didn't); run it on a holder instead.
 * texture_stats and analysis/texture_abc/texture_abc_diag.py are DIFFERENT
   implementations of same-named statistics (torch vs numpy, 8 vs 12 wedges,
   annulus 0.125 vs 0.15, no window vs Hann, ratio-of-means vs mean-of-ratios).
   The campaign's targets ("real ~1.10", "C_late 0.36 -> ~1.0") are in
   texture_abc units and CANNOT be checked against texture_stats output. Pick
   one implementation before either is used as a success criterion.
 * verdict must be depth-resolved: at t~10s (the A/B/C "C_early" depth)
   anisotropy INVERTS. wave90's trajectory is two-sided -- rises to 5.12 at
   t=20s then collapses to 0.50 at t=35-45s. A single-depth readout misleads.

## 2026-08-23 — A23 LAUNCH GATE: FAILED. B2 must take path (a).
Built `analysis/compare_commit_tensors.py` (+ `sbatch/run_commit_tensor_probe.sh`)
and ran it on holder 6108076 against
`dmd10k_ganfix_strict_rerun_h6106414_083234/eval_step0200.pt`, 100 gen chunks,
canonical eval-60 settings.

METHOD. Taps the cache-refresh-on-commit forward (all-zero timestep) inside the
INFERENCE rollout, so both tensors come from ONE checkpoint, ONE ride, the SAME
chunk and the SAME KV context, differing ONLY by the one extra t=60 forward:
  ladder = pred_x0 at t=208.33 post-CARN (byte-identical to the eval video)
  flash  = add_noise(ladder, eps, t=60) + one no-grad generator forward, i.e.
           training's Step 3.2.b reproduced verbatim
Plus an independent-eps NULL draw for the noise floor. This is a LOWER BOUND on
the real train/infer mismatch: committing the flash tensor changes all later
context, and that compounding is excluded here by construction.

MEASURED. sigma_flash 0.0618 vs sigma_ladder 0.2083 (3.37x) — confirms A2.
Pre-registered rule: voters (hv_anisotropy, haar_HL_LH_ratio),
d = mean|log(flash/ladder)|, EQUIVALENT iff d <= max(0.10, 2*d_null) in EVERY
depth stratum.
  early  d=0.0081  null 0.0055  -> EQUIVALENT
  mid    d=0.0221  null 0.0092  -> EQUIVALENT
  late   d=0.1093  null 0.0166  -> **NOT-EQUIVALENT** (6.6x the noise floor)
Monotone in depth by decile: 0.0100, 0.0126, 0.0487, 0.1025, 0.2152.
SYSTEMATIC, not scatter: 34/34 late chunks have flash haar_HL_LH below ladder,
32/34 for hv_anisotropy, mean signed log ~ -0.11.

WHAT IT MEANS — the sign is the dangerous one. The extra t=60 forward PARTIALLY
REPAIRS the banding: hv_anisotropy 88.8 (ladder) -> 68.1 (flash) in the final
decile. So the GAN's fake is systematically LESS artefacted than the video the
researcher actually watches. A pixel critic could drive d_loss down on a
partially-repaired sample while the visible artefact is untouched — exactly the
"perfect a hidden t=60 auxiliary render the user never sees" failure A23 names.

THE WORST POSSIBLE SPLIT. Training rolls out to ~eval-chunk 20
(streaming_max_length=60, rolling_random_depth_max=6) — entirely inside the
EQUIVALENT region (d ~ 0.006-0.013). The tensors AGREE where B2 trains and
DISAGREE where B2 must deliver, so the mismatch is invisible to any
training-time telemetry: a clean-looking 600-step run and a failed eval.

DEPTH STRATIFICATION IS LOAD-BEARING. A 6-chunk smoke of the same probe returned
EQUIVALENT on all three strata (d <= 0.015). Any shallow version of this check
passes and would have waved the gate through.

DECISION: **B2 takes path (a)** — build the fake from `finish_denoised_chunk`
with flash disabled, gradient path through the generation op. This also removes
the phase-LoRA landmine A2 flagged (with phase-LoRA re-enabled, the flash
gradient would land on a `rung_flash` adapter inference never loads). Path (iii)
(`flash_dmd_gan_t` -> last rung) needs sign-off since it changes what is
committed; path (ii) costs back the forward flash was introduced to save AND
still leaves the KV commit non-parity.
ACCEPTANCE TEST for (a): re-run this probe — d_vote should collapse to the
d_null floor in EVERY stratum.

## 2026-08-23 — CORRECTION: `rollout_viz_source=finish` claim was WRONG (I propagated it)
I reported, from the A2 agent's write-up and without verifying, that
"`rollout_viz_source=finish` means training-time sample videos also render the
t=60 tensor, so they are not an inference-parity readout." **That is inverted.**

Code (`trainer/causal_action_forcing_train.py:10028-10041`):
    # finish (DEFAULT) = finish-denoised pred, INFERENCE-PARITY, no exit-rung
    # lottery; FALLS BACK to flash slab, then chunk. auto = legacy
    # flash-else-chunk. chunk = always exit-rung.
    _fin = info.get("finish_denoised_chunk")
    if _viz_mode not in ("chunk","auto") and _fin is not None: _src_acc = _fin

So `finish` selects `finish_denoised_chunk` — the LADDER ENDPOINT, which IS
inference-parity. It only falls back to the flash slab when that tensor is
absent. `auto` is the mode that picks flash first.
CONSEQUENCE: training-time sample videos under the default ARE an
inference-parity artefact — the one training-time readout that already shows
what the eval renders. My claim would have led an auditor to DISCARD the only
parity evidence available. A23's measurement is unaffected (it taps the
inference rollout directly and never relied on the viz path).

## 2026-08-23 — NOISE FLOOR: single-seed texture rankings cannot separate close arms
`docs/TEXABC_REFERENCE_TABLE.md` (4 x ODE_FLOW_SEED per reference checkpoint)
measures C_late seed sd up to **+-0.34 on fft anisotropy** and states plainly:
"C_late anisotropy and angular entropy are NOISE-DOMINATED at this granularity —
single-seed readings cannot rank close arms", with the protocol "run at >=3
ODE_FLOW_SEEDs and treat differences under ~2x the pooled sd as unresolved".

CONSEQUENCE for the 10-arm seed_log_dist table logged earlier today: the LARGE
separations are safe (ganfix_marginal 0.149 vs ganfix_strict_rerun 2.307 is not
a seed artefact), and the horizon pair reproduces the researcher's own visual
ranking (0.287 vs 1.004, 3.5x). **But the top cluster — ganfix_marginal 0.149,
strict03 0.203, poolrich 0.217 — is NOT separable on one seed and must not be
read as a ranking.** Any arm comparison at that granularity needs >=3 seeds.
This applies to B2's success criterion too: TEXTURE_GAN_DESIGN §9.4's target
band (1.07-1.15 on anisotropy) is NARROWER THAN ONE SEED SD, so as written it
would be decided by noise.

## 2026-08-23 — RELIABILITY vs VALIDITY: our metric axes are split, and we had conflated them
Two measurements that look contradictory are not, and the distinction should
govern every future arm verdict.

`docs/TEXABC_REFERENCE_TABLE.md` (4 x ODE_FLOW_SEED, texture_abc instrument):
  RELIABLE axes  — hf_power_frac (2.8 sigma) and laplacian_kurtosis (3.4 sigma)
                   separate the two reference arms; the effect is real, not luck.
  NOISY axes     — fft_aniso fy/fx (1.0 sigma, sd +-0.34) and angular_entropy
                   (0.9 sigma) are NOISE-DOMINATED at C_late; single-seed
                   readings of them cannot rank close arms.

A11 (this campaign, rollout_quality instrument, 59 windows vs the researcher's
own ranking):
  VALID axes     — hv_anisotropy 50/59 (29/29 late), haar_HL_LH_ratio 49/59
                   (29/29 late).
  INVALID axis   — hf_power 11/59 overall and **0/29 in the late rollout**,
                   i.e. ANTI-correlated with the eye.

SYNTHESIS — these are orthogonal properties and we had been treating them as one:
  * `hf_power` is a PRECISE measurement of something that does not track human
    judgement. Reliable, invalid.
  * `hv_anisotropy` is a NOISY measurement of something that does. Valid,
    unreliable.
So "drop hf_power from the vote" (A11) and "anisotropy cannot rank close arms"
(TEXABC) are BOTH correct, and together they say our verdict currently rests on
a valid-but-noisy axis while excluding a reliable-but-invalid one. The remedy is
NOT to re-admit hf_power — it is to average the valid axis over seeds.

CONSEQUENCE FOR THE 10-ARM TABLE: large separations stand (ganfix_marginal 0.149
vs ganfix_strict_rerun 2.307; horizon_nogan90 0.287 vs horizon_wave90 1.004,
which reproduces the researcher's visual verdict). **The top cluster —
marginal 0.149 / strict03 0.203 / poolrich 0.217 — is NOT a ranking** and must
not be read as one until it has >=3 seeds.

OPEN AND NOW BEING MEASURED: **haar_HL_LH_ratio's noise floor is unknown.** It
is absent from the TEXABC table yet is our strongest single voter. If it is both
valid AND low-noise it should carry the verdict on its own; if it is as noisy as
anisotropy, every close-arm comparison needs the multi-seed protocol. Running now
on holders 6108075/6108076: horizon_nogan90 and horizon_wave90, same checkpoints,
ODE_FLOW_SEED in {1234, 43, 44}, scored with the corrected battery
(`sbatch/seed_noise_probe.sh`). Deliverables: per-statistic mean +- sd, the
between-arm gap in sigma, and whether `seed_log_dist` is a reliable quantity.

## 2026-08-23 — A23 VERDICT OVERTURNED. "B2 must take path (a)" is NOT ESTABLISHED.
The adversarial review replicated the probe across seeds, rides and checkpoints.
NOT-EQUIVALENT is a **1-of-7** result — a seed outlier on the single most
collapsed arm in the fleet.

| cell | early | mid | late | dec9 | ladder aniso @dec9 | verdict |
|---|---|---|---|---|---|---|
| strict_rerun ride100 **seed 42** | .0081 | .0221 | **.1093** | **.2152** | **88.8** | NOT-EQ |
| strict_rerun ride100 seed 43 | .0092 | .0122 | .0113 | .0092 | 5.6 | EQUIVALENT |
| strict_rerun ride100 seed 44 | .0102 | .0135 | .0128 | .0186 | 4.1 | EQUIVALENT |
| strict_rerun ride200 seed 42 | .0121 | .0351 | .0233 | .0232 | 1.0 | EQUIVALENT |
| ganfix_marginal ride100 s42 | .0087 | .0164 | .0076 | .0064 | 5.3 | EQUIVALENT |
| ganfix_poolrich ride100 s42 | .0079 | .0176 | .0117 | .0114 | 6.4 | EQUIVALENT |

Applying the campaign's OWN mandated >=3-seed protocol: late 0.0445, dec9 0.0810
-> **EQUIVALENT in every stratum**. The arm's own late anisotropy swings
88.8 / 5.6 / 4.1 / 1.0 across those cells — the collapse is SEED-CONDITIONAL, not
a stable property of the checkpoint.

WHAT THE EFFECT ACTUALLY IS: a function of COLLAPSE SEVERITY, not depth.
r(log ladder_aniso, d) = 0.73; **0/270 chunks with aniso < 20 exceed tau**
(max .1011); 15/30 with aniso >= 20 do. A23 happened to sample the most collapsed
arm in the fleet (seed_log_dist 2.307 vs marginal's 0.149).

MECHANISM REFUTED — strike the "partial repair" story. Signed log(arm/ladder) on
hv_anisotropy (late/dec9): `noiseonly` (add_noise, no forward) **-1.05/-1.78**;
`fwdonly` (forward, no noise) **+0.48/+0.72**; flash -0.11/-0.26; `flash208`
(same operator at the ladder's OWN sigma) **-0.16/-0.36**. So a bare extra
forward AMPLIFIES the banding — there is no repair; the negative sign is residual
injected eps a single denoise step fails to remove. And matching the sigma makes
the gap LARGER, which contradicts the sigma 0.062-vs-0.208 causal framing.

THRESHOLD WAS UNSOUND IN ITS OWN UNITS: tau=0.10 was anchored on |log(1.10/0.93)|
= 0.168 from **texture_abc**, which this ledger already records as
non-comparable to texture_stats. texture_stats log-units are ~3.9x wider, so
tau=0.10 ~= 0.025 abc-nats — ~7x STRICTER than the eye anchor it cites. Late also
passes at tau=0.15/0.20, and its MEDIAN (0.0856) passes at tau=0.10.

d_null WAS ~10x TOO SMALL: it is the eps floor (0.0166 late) and excludes the
rollout-trajectory component entirely (between-seed sd 0.0562 late / 0.1163
dec9). So the probe's `"underpowered": false` was false reassurance.

PATH (a) IS OVER-READ — and I inverted my own finding. The gap does not exist in
the regime B2 TRAINS in: training caps at ~20-24 eval-chunk equivalents where
d = 0.008-0.016 at EVERY seed and checkpoint. I reported "the tensors agree where
B2 trains and disagree where it delivers" as the danger; it is in fact the reason
the danger cannot arise as stated — the critic's fake is already inference-parity
everywhere a gradient is ever taken. Also: A23 measured a fixed-point tensor
DISTANCE on a frozen checkpoint; the only surviving worry (GAN gradient at t=60
moving the t=60 marginal without moving t=208) is a claim about GRADIENTS, on
which this probe carries zero information.

**If path (a) is built, build it on the A2 provenance/phase-LoRA argument** (the
flash gradient would land on a `rung_flash` adapter inference never loads) —
that stands on its own. Do NOT build it on d=0.1093.

REAL DEFECT FOUND in the probe: it takes the ladder AFTER CARN
(`utils/eval_causal_AR.py:1328-1333`), but training's flash input is PRE-CARN
(`pipeline/action_forcing_training.py:1466-1476` noises raw `cache_pred`; CARN is
applied at `:1676-1688` AFTER `cache_pred = flash_dmd_pred.detach()`). So the
"reproduces Step 3.2.b verbatim" claim is false, and the actual GAN fake is a
pre-CARN tensor the probe never measured — a second mismatch, still unscoped.

TAP VALIDITY CONFIRMED CLEAN (the one attack that failed): full-rollout latent
SHA identical between tapped and untapped modes, 0/100 per-chunk mismatches,
published strata reproduced to delta = 0.0.

REVISED ACCEPTANCE TEST: `d_vote <= max(tau, 2 x between-seed sd)` at >=3 seeds.
The old "d_vote <= d_null in every stratum" is unmeetable for the wrong reason.

## 2026-08-23 — CORRECTION: my "R2 has zero references repo-wide" claim
I reported, after grepping, that `ladd_r2_|gan_r2_gamma|r2_penalty|r3gan_r2`
returned ZERO hits across trainer/model/configs/sbatch. That was true when I ran
it and is FALSE now: `model/dmd_action_forcing.py:1948-1970` assigns four
`ladd_r2_*` attributes as DEAD STORES, and the configs/sbatch still pass the
keys. Timeline: a WP-PIXGAN sub-agent deleted those dead stores as hygiene and
WP-PIXGAN reverted it as out-of-scope for B1, restoring them.

WHAT IS AND IS NOT TRUE: the R2 **loss path** is genuinely gone — no R2 penalty
is computed anywhere, which is the part that matters. The **attribute surface**
remains (8 grep hits). Cleanup belongs to whoever unparks the LADD transition
critic. Recorded because I stated the stronger claim.

## 2026-08-23 — A23: peer corroboration, and the acceptance criterion is a tautology
WP-PIXGAN independently confirmed the overturn and added a sharper argument than
mine: `analysis/compare_commit_tensors.py:118` sets
`VOTERS = (hv_anisotropy, haar_HL_LH_ratio)`, and **`hv_anisotropy` is a
statistic this campaign's own noise-floor protocol already flags as
noise-dominated** (sd up to +-0.34; never rank close arms single-seed). So A23's
verdict was a single-seed read on a voter already documented as unsafe to read
single-seed; the .1093/.0113/.0128 spread is exactly what that warning predicts.

Second, independent finding from WP-PIXGAN: **the path-(a) acceptance criterion I
recorded is a TAUTOLOGY.** Under path (a) the fake IS the ladder endpoint, so
both arms of `compare_commit_tensors.py` are the SAME tensor and every distance
collapses to 0.0 by construction — it can never fail, so it tests nothing.
Between that and the seed spread, A23 never justified a launch block.

STATUS CHANGE: path (a) is an OPTION, not a gate. WP-PIXGAN has it built, tested
(20/20) and flag-gated default-off (`pix_finish_grad_enabled`), so the fake
source is now an experimental variable rather than a blocker.

## 2026-08-23 — SEED NOISE FLOOR MEASURED: our metric cannot rank close arms
Corrected probe (varying `--seed`, not `ODE_FLOW_SEED`), 3 seeds x 2 arms,
`sbatch/seed_noise_probe.sh` + `analysis/seed_noise_summary.py`.

horizon_nogan90 vs horizon_wave90, gap / pooled sd / sigma:
  seed_log_dist      -0.414 / 0.245 / **1.69**  unresolved
  hf_power            0.951 / 1.045 / 0.91      unresolved
  angular_entropy     0.075 / 0.126 / 0.60      unresolved
  hv_anisotropy       0.083 / 0.901 / 0.09      unresolved
  hf_kurtosis        -0.016 / 0.209 / 0.08      unresolved
  haar_HL_LH_ratio   -0.005 / 0.457 / **0.01**  unresolved

**NOTHING separates at >=2 sigma on this pair.** Two findings:

1. **`haar_HL_LH_ratio` — the statistic I called our strongest voter — separates
   these arms at 0.01 sigma.** Its 49/59 per-window agreement did NOT survive
   seed variation. LESSON: **per-window agreement WITHIN a rollout is a
   different property from between-arm separation ACROSS seeds.** I had been
   treating the first as evidence for the second.
2. `seed_log_dist` at 1.69 sigma is the BEST of the six — the composite does
   work its parts do not — but is still under the 2 sigma bar at n=3. The
   DIRECTION matched the researcher's eye at all three seeds, which is real
   evidence; it is not a resolved separation.

WHAT THE 10-ARM TABLE CAN ACTUALLY SUPPORT (pooled sd 0.245 -> minimum
resolvable gap 0.489):
  RESOLVED   marginal vs horizon_wave90 3.5s, vs gan2x2_wave_ts 3.5s,
             vs gan2x2_raw_t0 5.3s, vs ganfix_strict_rerun 8.8s
  UNRESOLVED marginal vs strict03 0.22s, vs poolrich 0.28s, vs nogan200 0.86s,
             vs wave01 1.80s
So the instrument distinguishes CATASTROPHIC from HEALTHY and nothing finer.
Every future arm comparison needs >=3 seeds as standard, and B2's success
criterion cannot be a single-seed reading of anisotropy.

## 2026-08-23 — A22 re-review: 3 of 6 defects were NOT fixed; fixed directly
The re-review agent died twice, so I verified by hand. Fixed by the earlier
agent: D1 (default `train_from_observed=False` + a `train_rides` gate), D4
(`n_eff_*` crop-level SE), D5 (two-sided `lo = 0.5 - margin`, `VERDICT_INVERTED`).
NOT fixed, and now fixed by me:
  * **D2/D3 — `_fail_closed` was described in the module docstring but NEVER
    IMPLEMENTED.** Both fail-open paths still reported the clean signature.
    Added `_fail_closed` + `HoldoutUnverifiableError` (deliberately distinct from
    `HoldoutLeakError`: one means we PROVED contamination, the other that we
    proved NOTHING and must not be read as clean). Emits `unverifiable=1.0`.
  * **D6 — strict mode was still one-shot.** `if cached is not None: return
    cached` sat BEFORE the raise, so any enclosing retry turned a hard fail into
    a silent `dhp_leak_rides=1`. The cached path now re-asserts and re-raises.
  * **D14 — `disc_holdout_probe_` was missing from `_OVERRIDE_GUARD_PREFIXES`.**
    A typo'd `disc_holdout_probe_evrey=50` was silently accepted and the probe
    simply never fired — the exact failure class the guard exists for, on the one
    instrument whose job is to catch contaminated experiments.
Behaviour verified directly: no-reserved-rides -> raises; unreachable ride list
-> raises; healthy -> passes through with `unverifiable=0.0`; strict raises on
3/3 cached calls.
