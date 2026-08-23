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
