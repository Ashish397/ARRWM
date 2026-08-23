# 22/8 PLAN — finishing the rolling campaign
(companion to OPUS_HANDOFF_PLAN.md, which holds full context/rules/state)

## Goal
A stable, lively 60-second causal rollout: canonical DMD recipe + working GAN,
judged on the fixed-route Madrid eval with inference-CARN.

## Step 1 — Land the decisive GAN test (today)
- 6100985 (`ganfix_strict` rerun) = first FULL assembly (cross-ride bank 4096,
  sampled disc-t shift5 shared real/fake, scalar logit, frozen projectors,
  R1+R2 γ=1, weight 0.03, strict action conditioning, symmetric bidir scorers,
  grad cap). Verify launch → d_loss leaves 0.693, settles 0.3–0.6, never
  sustained <0.10 → ratio ‖g_GAN‖/‖g_DMD‖ in 5–20%.
- 6100986 (projected-GAN variant) runs the same protocol as the arch alternative.

## Step 2 — Run THE evaluation (the part never done)
Fixed route, fixed seed, same checkpoint step, 60 s, inference-CARN ON:
strict-GAN arm vs matched no-GAN control (fullcarn_bidir lineage) vs wavelet arm.
Pass/fail per DMD_GAN_SESSION_LOG "Immediate Decision Rule". This decides
"does the GAN help" once and for all.

## Step 3 — Branch on the verdict
- GAN passes → merge into canonical; mint fresh 8-node stationary GANck with the
  fixed recipe (user's ordering: rolling first, then re-mint stationary).
- Critic learns but motion still freezes → transition representation is the
  limit → trajectory-level critic OR non-adversarial drifting-stat matching
  (spectrum/covariance/kurtosis — CARN_V2_DESIGN lists exact targets).
- Critic never learns → escalate: the bank/conditioning assembly gets the
  factorial (raw vs HF-fixed × t60 vs sampled-t) from the research review.

## Step 4 — DMD consolidation (parallel, cheap)
- Score the completed trio: rolldmdfix (1:1) vs 51 (5:1) vs 51f0 (floor0):
  drift table + cartoon/texture by eye + denominator D histograms by (t, depth).
- Calibrate dmd_grad_target_norm τ from rollcarn700's median grad norm
  (provisional 1.0 is in the queued configs).
- Finish A/B/C: run roll-mode=last arm once a holder frees.
- If random-roll confirmed best → implement Causal-rCM coherent-target
  (frozen-trajectory scoring, slice per chunk) as wave-5.

## Step 5 — The 5.25 s cliff attack (after GAN verdict)
Exposure-bias cliff = seed eviction from the 21-frame window. Fixes in order:
symmetric-scorer + causal-denom retrain (in flight) → dense supervision beyond
roll 3 (already: random depth ≤6) → verify with the 60 s eval; if the cliff
persists, add post-eviction-focused supervision (bias random-roll draw toward
deep rolls) — one-line change to the roll-mode sampler.

## Step 6 — Close-out hygiene
- wandb sync all new runs; flow timelines for the last 2 arms; score rollnocarn;
  review dbg_inputs videos (clean-match reference quality) with the user;
  update ROLLING_CAMPAIGN.md verdicts; delete stray holder end-saves.

## Ordering constraint
Do NOT bundle new DMD changes into the GAN arms (isolation per arm), and never
trust a "looks good" without the Step-2 eval — that lesson is paid for.

---
# 22/8 PLAN v2 — research-recommender triage (numbered, action in order)

## Corrections to the recommender (already done, VERIFY at runtime don't re-implement)
- Cross-ride real bank IS in the strict-rerun config (ladd_real_pool_cross_ride=4096,
  push_per_ride=8) — recommender read a log that omitted it. VERIFY it actually
  populates/samples at runtime (log bank size + unique-real count per update).
- Generator EMA IS in config (ema_weight=0.99, ema_start_step=0). VERIFY the 60s
  eval can use EMA weights; evaluate raw vs EMA.
- Fake-score init = exact 14e teacher weight copy (825/825 keys) + resume-skip: DONE.
- Fake-score EMA: traced = pull-toward-GENERATOR, set 0.0: DONE (worse than lag —
  it was melting the critic into the student).
- Strict rerun floor is 1e-6, NOT 1.0 → the causal-denominator repair is LIVE there.
  (Recommender's bypass warning is right in general: floor=1.0 + causal denom
  = repair silently disabled. Never combine.)

## A — verify/cheap, do before anything else
1. FLOOR POLICY: treat 1.0 strictly as "no-amplification ablation". Default going
   forward: small floor (1e-5..0.05) + dmd_grad_target_norm as the gain governor
   (they compose: eq8 relative weighting intact, absolute gain capped). Run the
   4-point floor ladder {1e-5, 0.05, 0.1, 1.0} only if collapse reappears.
2. MAE GATE: add explicit dmd_mae_gate_enabled=false to every rolling config +
   a startup log line asserting it (historical default was off, but with
   continuous low-t sampling an active gate would silently kill exactly the
   texture gradients we just enabled).
3. GRAD TELEMETRY (recommender's strongest cheap ask, still missing): log at a
   shared late generator block ||g_DMD||, ||g_GAN||, ||g_CARN||, ratio, and
   cos(g_GAN, g_DMD). Loss ratios ≠ update shares; the cosine detects a small
   GAN that is destructive (cos≈-0.8) vs irrelevant (ratio≈0.01).
4. R1/R2 ALTERNATION: r1/r2_every_n_steps=2 with phase offset = each penalty at
   HALF per-step strength vs One-Forcing's both-each-step. Either double both
   gammas or run both every step; document intended effective strength.

## B — adopt as the next code wave (in this order)
5. HiAR FIRST-JUMP REGULARIZER (new, targets the low-motion attractor directly):
   L += 0.1 * teacher-trajectory loss at the FIRST denoising jump only,
   computed in bidirectional mode. No density ratio, no B=1 problem (unlike the
   failed f-distill). Assets exist (14e teacher, 4-rung schedule). Note: distinct
   from the retired dmd_real_traj (that was all-rung segment MSE, stationary-era,
   dead-disc era); this is first-jump-only, motion-mode targeted. Flag:
   dmd_teacher_first_jump_weight=0.1.
6. TRAIN THE FAILURE HORIZON (AAPT pattern; recommender's new #1): long
   student-forced rollouts (30-60s; raise streaming_max_length, accept fewer
   rides/step) with the scalar D applied to short windows SAMPLED AT MULTIPLE
   DEPTHS (2s/7s/15s/30s...), progressive duration extension. The generator must
   LIVE through deep self-context; D need not see 60s at once. This directly
   attacks the 5.25s cliff — training currently barely crosses the eviction
   boundary (depth 2-6 ≈ 27-63 latent frames vs 21-frame window).
7. MULTI-DEPTH DISCRIMINATION (part of 6, standalone if 6 is staged): keep the
   scalar contract, change temporal support — D(prefix ending at chunk k) for
   several k, not one 2-chunk transition.
8. DECONFOUND WAVELET×TIMESTEP: the wavelet branch still forces disc t=0 (APT:
   t=0 diffusion features collapse discrimination). Run the 2x2: {raw, wavelet-
   fixed} × {t=0, sampled shift-5}. Much of "wavelet behavior" may be timestep.
9. ACTION-BUCKETED CROSS-RIDE RETRIEVAL: strict conditioning + tiny local pool
   = correct labels with catastrophically low diversity. Retrieve reals by
   action trajectory first, visual similarity second, from the 4096 bank; never
   let nearest-latent collapse support to the same handful.

## C — keep exactly as-is (recommender concurs)
- GAN weight 0.03 (bracket measured: 0.01 inert / 0.03 learns / 1.0 destroys).
- RpGAN + R1 + R2 (AAPT validates the combination; don't relitigate).
- The strict-rerun + fixed-route 60s eval remains THE decisive test (Steps 1-2).

## D — discard / defer, with reasons
- ADA: defer until real-pool diversity + horizon training are in (it would
  compensate for a wrong data distribution rather than fix it).
- De-bursting the 5 D-steps: legitimate but strictly below pool/horizon in
  expected effect; revisit after 6.
- Projected-vs-wavelet architecture choice: last among GAN questions (both
  recommender and our data agree the failure has migrated to horizon/state).
- Pushing GAN weight back toward 1.0: contradicted by our own bracket.
- f-distill-style density-ratio forward KL: stays dead (structural B=1 failure,
  three strikes); HiAR item 5 is the mode-covering replacement.
