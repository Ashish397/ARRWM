# Deep-research prompt: audit our DMD (Distribution Matching Distillation) implementation against the state of the art

You are a research agent with web access. Your task is to audit the DMD
(distribution-matching distillation) training recipe of a driving world model
against how the field actually implements DMD-family distillation, and to tell
us **where our implementation makes silly assumptions, skips standard steps, or
deviates from what works** — ranked by how suspicious each deviation is.

Everything below is a faithful transcription of what our code ACTUALLY does
(file:line citations refer to our private repo, branch `dmd_one_step`; you
cannot see the code, so treat the transcription as ground truth). Read all of
it before answering — the failure modes we observe are described at the end,
and several of our design choices exist because of them.

---

## 1. The system, as if you were joining the team

### 1.1 Model family and data

- **Backbone**: Wan2.1-family video diffusion DiT (~1.3B class), converted to
  a **causal, KV-cache streaming, action-conditioned** video model
  (`CausalWanModel`). Latent space: Wan VAE, **16 channels, 60x104 spatial**,
  ~4x temporal compression. One latent "chunk" = `num_frame_per_block` = **3
  latent frames** (~12 pixel frames). Each latent frame is ~1,560 tokens
  (`frame_seq_length`, action-token aware).
- **Conditioning**: text prompt (cross-attn), plus per-frame **actions**
  (2-D: col0 = throttle, col1 = steer) injected two ways: (a) action tokens
  appended per frame, (b) per-frame FiLM/AdaLN "action modulation" of the DiT
  blocks. Data = ~2,500 rides ("frodobots" sidewalk-robot driving videos)
  stored as zarr latents; a training "ride" is a long window (up to 900
  latent frames) streamed chunk by chunk.
- **Scheduler**: `FlowMatchScheduler` (rectified-flow / flow matching), shift
  = 5.0, `sigma_min = 0.0`, 1000 train timesteps; the network predicts flow,
  converted to x0 (`_convert_flow_pred_to_x0`).

### 1.2 Teacher and student

- **Teacher ("14e")**: a LoRA fine-tune of the *bidirectional* Wan model,
  trained teacher-forced on a **joint 42-frame layout**: 21 clean context
  frames concatenated with 21 noisy frames; clean half at RoPE `[0,21)`,
  noisy half at RoPE `[3,24)` (`tf_rope_offset_frames = npb = 3`). Frozen
  during DMD (merged LoRA, `real_teacher_train_online=false`).
  `real_guidance_scale = 0.0` — **no CFG on the teacher** (CFG scale 3.0
  reliably collapsed the student; scale 0 is a hard project rule).
- **Student**: the same architecture served **causally with a KV cache**,
  distilled to a 4-rung ladder `denoising_step_list =
  [1000, 625, 357.142857, 208.333333]`. These rungs are an exact subsample
  (grid indices [0, 15, 18, 19]) of the 20-step Euler chain the teacher used
  to generate its ODE dataset. The student was **initialized by offline ODE
  regression** ("ODE-distill") on ~10-12k teacher trajectories (MSE or
  local-KL objective variants; the current best init is the "KL" arm,
  `run3_flip2_rollkl10k/step400`), then DMD fine-tunes it online.
- **fake_score (the "critic")**: a full copy of the student architecture,
  initialized from the same weights, trained online as the fake-distribution
  score estimator (standard DMD2 arrangement). Optional EMA on it
  (`fake_score_ema_weight = 0.95` in current runs). Everything runs bf16 with
  gradient checkpointing; **batch size = 1 ride per GPU rank**, 8-32 ranks
  (2-8 nodes x 4 GPUs), DDP.

### 1.3 The streaming training loop

Per trainer iteration (`trainer/causal_action_forcing_train.py:1890`):

```
train_generator = (step % dfake_gen_update_ratio == 0)   # ratio = 5
if train_generator: generator step (rollout + DMD + aux losses + backward)
ALWAYS:            critic step   (rollout detached; fake_score denoising loss)
then: optimizer.step() for whichever ran (gen lr 1e-5, betas (0.0, 0.999),
      wd 0.01, 8-bit Adam; fake_lr 2e-6, betas (0.0, 0.999), wd 0.01)
```

So the fake_score gets **5 updates per generator update** (DMD2's two
time-scale rule, implemented as "gen every 5th iter" rather than "critic 5x
per iter"). `dmd_loss_weight = 1.0`, ramped over the first 20 steps
(`dmd_loss_warmup_steps = 20`).

A generator step does a **KV-cache streaming rollout**: seed context frames
are prefilled from real (GT) latents (`seed_prefill_mode=real`, clean, t=0 —
"never noise the clean context" is another hard project rule,
`context_noise=0`), then the student autoregressively generates chunks with
the 4-rung sampler; a random exit rung per step
(`generate_and_sync_list`) decides which rung's x0 carries gradient. Between
chunks the student's own denoised prediction is committed back into the KV
cache at t=0 (self-forcing-style AR training; commit path is `no_grad`).
Stationary arms train on the first rolled window only
(`dmd_only_first_chunk_per_ride=true, max_rolls_per_ride=1`); rolling arms
keep rolling forward through the ride (random depth 2-6 rolls) and apply DMD
at every roll.

### 1.4 The "42f" joint scoring window (how the teacher scores the student)

Because the teacher is bidirectional and teacher-forced, the DMD scorers do
NOT score the student's raw causal window. Instead
(`model/dmd_action_forcing.py:9623`, `_build_42f_scoring_inputs`) we rebuild
the teacher's exact training geometry every DMD step — a 21-frame noisy half
plus a 21-frame clean half ("42f"):

- **Stationary layout** (first roll): noisy half = `[n_ctx GT frames |
  ns student chunks | gt_after GT frames]`. Current arms: `dmd_42f_num_chunks
  = 4` student chunks rolled, `gt_after_chunks = 1`; all student chunks
  except the newest are supervised (the newest sits at the structurally-OOD
  last slot RoPE [21,24), which has no clean counterpart in the teacher's
  training distribution — it gets ~2x teacher MAE — so it is
  gradient-masked). Variants exist: `rand_sup_slot` (random supervised slot),
  `allsup` (supervise all counterpart slots), `seed_last` (GT scaffold after
  the supervised chunk).
- **Rolling layout** (`dmd_42f_rolling_sup_new`, roll k>=2): the whole 42f
  window rolls with the student. Noisy half = `[n_ctx STUDENT overlap frames
  (detached) | new student chunks (graph-on, supervised) | 3 GT future
  frames at the masked OOD slot]`. The clean half remains **positional GT**
  and rolls forward with the window.
- **clean half geometry knobs**: `clean_drift` ramps the clean half's offset
  from -3 frames (v14 parity, clean half BEHIND the noisy half) toward +3k
  frames (clean half AHEAD = forward/future GT context) over a step
  curriculum, optionally coupling RoPE (`couple_rope`); `clean_match`
  replaces the positional GT slice with the *content-matched* GT window
  (offset `match_m` chosen once per step by an L1 match over the ride, so a
  drifted student is compared against the GT region it actually resembles);
  `clean_match_drift_compose` composes both. **Every good June-era DMD run
  gave the teacher a FORWARD (future) clean half**; this is why these knobs
  exist.
- The **critic is trained under the exact same 42f conditioning** the
  generator is scored under (`_compute_critic_loss_streaming_gtfix`,
  `model/dmd_action_forcing.py:11116`): same input builder, same timestep
  sampler, denoising (flow-matching) loss on the student's detached chunk,
  same gradient mask.

### 1.5 The DMD loss, exactly as implemented

`model/dmd_action_forcing.py:6233` (`compute_distribution_matching_loss`) and
`:4941` (`_compute_kl_grad`):

```
t  ~ sampler (below), one t per sample, broadcast over all 21 frames
x_t = add_noise(x0_student, eps, t)                      # x0_student = full 42f noisy half
pred_fake = fake_score(x_t, cond, t, clean_x=...)        # x0-prediction
pred_real = real_score(x_t, cond, t, clean_x=...)        # x0-prediction, frozen teacher, no CFG
grad = pred_fake - pred_real                             # :5199  (reverse-KL score difference)
normalizer = |x0_student - pred_real|.mean(dim=[1,2,3,4], keepdim=True)   # DMD eq.(8)
grad = grad / normalizer.clamp_min(0.05)                 # :5227  denom floor 0.05 (caps amplification at 20x)
grad = nan_to_num(grad)
dmd_loss = 0.5 * MSE( x0_student[mask], (x0_student - grad).detach()[mask] )   # :6403
```

Notes on this core:

- Both scorers see the SAME `x_t`; only the clean-half conditioning can
  differ (`clean_x_real` = lightly-noised GT view in `dmd_context='GT'`
  mode).
- The gradient mask restricts the loss to the supervised student chunks (3-9
  of the 21 noisy frames).
- The eq.(8) normalizer can be computed band-locally
  (`dmd_normalization_band_local`) — ablated, no difference.
- An optional **f-distill forward-KL mixing** (`dmd_fkl_mix`,
  `:5233-5251`) reweights `grad` by a density-ratio estimate `r = exp(disc
  relativistic gap)` normalized across ranks. **This failed structurally
  three times**: with B=1/rank and token-averaged disc logits, the
  cross-rank dispersion of the ratio is ~0 (CLT over ~50k tokens), so
  `r_hat = 1.000` on every row. Parked.

### 1.6 Timestep sampling

`_sample_dmd_timestep` (`:5348`): uniform in
`[min_score_timestep=150, 1000]`, clamped to `[20, 980]`, with SD3-style
shift (`timestep_shift = 0.5` in current runs, i.e. skew toward LOW noise).
**But** the current recipe sets `dmd_sample_at_rungs = true` (`:5395`): t is
drawn uniformly from the student's own 4-rung ladder {1000, 625, 357, 208}
(no clamp — clamping had silently turned rung 1000 into 980) and the
shift/min/max knobs become dead code. One t per sample, broadcast across
frames (`uniform_timestep=True`). The critic uses the same sampler with an
optional separate shift (`critic_timestep_shift`; ablated, no difference).

### 1.7 The MAE gate (teacher-reliability throttle)

`_dmd_mae_gate_weight` (`:4817`): per step, on the supervised slots,

```
m_real = |pred_real - GT|.mean();  m_fake = |x0_student - GT|.mean()
r = EMA(m_fake / m_real);   w = clamp( ((r-1)/(r_full-1))^0.75, 0, 1 ),  r_full = 2.0
dmd_loss *= w      # detached scalar
```

i.e. DMD is scaled to zero as the student matches the teacher's own
GT-oracle accuracy. Current stationary runs have it ON (exponent 0.75,
EMA=0); the DMD3 study concluded it should be OFF because it is monotone in
t and throttles DMD to w~0.03 exactly at the low rungs where the image
forms. (See §3 for why `m_fake` itself is a misleading metric.)

### 1.8 The AR head (dual-serving DMD)

The TF head above scores through the teacher's bidirectional joint window —
which **contains the band's own GT future in the clean half**, making the
teacher a near-oracle, hence mean-seeking. To get a genuinely causal
conditional we added an AR head (`_ar_score_band`, `:5729`;
`dmd_ar_head_weight`, current best arm: AR=1.0, TF=0.0):

- BOTH `real_score` and `fake_score` are served **autoregressively with a
  local KV cache**: prefill the past-only context chunk-by-chunk at t=0,
  then score the N supervised band chunks in one shared-cache pass, all
  noised to the same rung t; between chunks, **commit the student's own
  chunk** (`dmd_ar_head_commit="student"`; committing GT instead measurably
  destroys the student — off-manifold unreachable targets).
- `grad_ar = pred_fake_ar - pred_real_ar`, normalized by the SAME TF eq.(8)
  normalizer (deliberately shared support so head weights are comparable),
  then `ar_loss = 0.5*MSE(band, (band - grad_ar).detach())` on band slots
  (`:6676`), added as `dmd_ar_head_weight * ar_loss` — NOT gate-scaled.
- The critic gets a matching AR-served training term
  (`dmd_ar_critic_weight`, `:11163`): same timestep/noise draws as the TF
  critic term, scoring forwards grad-enabled through the same
  prefill/score/commit schedule.
- Since the scorers were built bidirectional, a weight-sharing causal twin
  is attached so kv_cache kwargs actually work; RoPE offsets are matched to
  the TF head (+npb), infinity-RoPE (rotates at local cache indices) is
  active. Known bounded inconsistency: the TF clean half sits npb frames
  behind its content position, which the causal cache cannot reproduce, so
  TF-vs-AR deltas carry a positional component.
- Cost: ~16 extra sequential scorer forwards per DMD step at the 3|3|1
  geometry.

### 1.9 Anti-collapse machinery (the load-bearing part)

- **Stat anchor** (`model/anti_collapse.py:493`,
  `compute_stat_anchor_loss`): per-frame summary stats of the student
  rollout — STD, M2 = sum_c sigma_c^2, TV = sum_c mean|delta x| — anchored
  by MSE-with-tolerance-floor to targets from the seed window or a
  k-nearest matched GT window (`stat_anchor_mode=gt_window, match_k=2`).
  Current winning weights: `weight=1.0, M2_short=0.1, TV_short=0.1`,
  everything else 0, and **rel_tol pinned to 0** (no tolerance band — the
  floor formula is `loss = clamp(mse - (rel_tol*anchor)^2, min=0)`, so
  rel_tol=0.2 was found to disable the anchors entirely and DC drift
  returned). Ablations found this is **the ONLY component whose removal
  collapses training** (to black by ~step 61); six other components read
  null in sequence.
- **CARN seam ops** (`pipeline/action_forcing_training.py:1643-1685` and
  `:2308-2350`): at every KV-cache commit site the committed context chunk
  passes (no_grad, inference-parity, gradient-free) through: (1)
  temperature `x <- mu + T(x-mu)` (counter the per-chunk variance
  contraction), (2) global drift counter-bias `x <- x - lambda_d * d_hat`
  (d_hat = fitted 16-channel per-roll drift vector, |d|=0.57/roll), (3)
  **seam affine** `x <- (x-mu)/sigma * (lambda*sigma* + (1-lambda)*sigma) +
  (lambda*mu* + (1-lambda)*mu)` toward the ride seed's per-channel stats,
  lambda=0.5. The affine is measured ~80% sufficient against DC drift and
  is an AR(1) with pole (1-lambda) on the committed-stat residual.
- **Flash-DMD slab** (`model/dmd_action_forcing.py:10312`): optional extra
  near-clean forward at t=60 per block whose x0 feeds the GAN and gets its
  own anti-collapse term (`flash_dmd_enabled` — on in GAN arms, off in
  pure-DMD arms).

---

## 2. Known prior results and analyses (treat as established)

1. **Reverse-KL DMD is stable** in this setup; the pure-DMD + stat-anchor +
   CARN recipe rolls out flat (drift within +-0.02..0.05) for 380+ steps and
   survives to 700 with bounded excursions.
2. **The teacher's TF conditional leaks the future**: the bidirectional
   teacher reads the band's own GT via the clean half; measured
   `m_real` 1.7-3x better than the student — the teacher is a near-oracle
   in TF serving and mean-seeking pressure follows. The AR head was built to
   fix this and wins by eye.
3. **`m_fake` (student MAE vs GT under GT context) is ANTI-correlated with
   sample quality** across arms: MAE rewards regression to the conditional
   mean. Metric ranked mse > kl > ar_kl; the eye ranks the reverse.
4. **AR variance-contraction law**: per-chunk latent std contracts as
   `s_c = s_GT * k^(c+1)` with k set by sampler depth (k~0.917 at 4 rungs,
   0.987 at 48 steps); rung-0 (t=1000 single-jump) output is
   mean-collapsed. The CARN temp/affine exist to counter exactly this.
5. **Drift anatomy** (frozen-model probe, 120 roll transitions): drift is
   GLOBAL in latent channel-mean space (ride-level pairwise cos +0.74);
   ranked residuals the affine does NOT fix: radial power-spectrum tilt
   (high-band loss + low-band gain, z=1.33), channel-covariance
   eigenspectrum collapse (z=1.18), per-frame variance structure, kurtosis
   decay (Gaussianization). This is the remaining "texture drift / cartoon"
   failure.
6. **Timestep artefacts dominate naive metric reads**: the DMD scorers'
   MAE-vs-GT is monotone in t; an apparent 35-step "collapse" was fully
   explained by the random t draw (mse/kl flat over 800 steps once t was
   regressed out).
7. **Trajectory distillation** (refining pred_real by continuing the
   teacher's Euler chain along the rung segment) improved the teacher's
   m_real but did NOT move the student, at +48% wall clock. Reverted.
8. **June-era working DMD runs** all shared: forward (future) clean half for
   the teacher, 18-frame seed, 6-supervised/6-detached chunk split,
   stat_anchor=1.0. GAN/flash/noiser were NOT needed then.
9. **f-distill port failed structurally** (B=1/rank, token-averaged logits
   -> ratio estimates pinned at 1 by CLT). Only per-chunk weights *within* a
   sample would carry signal here.

---

## 3. What we want from you

Compare our recipe, step by step, against the actual published + released
implementations of:

- **DMD** (Yin et al.) and **DMD2** (Yin et al., "Improved Distribution
  Matching Distillation") — including their two-time-scale update rule,
  their removal of the regression loss, their GAN term placement, their
  eq.(8) normalizer handling, their timestep sampling range (e.g. the
  `min_step/max_step = 0.02/0.98 * T` convention), backward-simulation
  details, EMA usage, and their few-step (4-step) student scheduling.
- **f-distill / forward-KL variants** (Xu et al.) — the h(r) weighting, the
  batch-normalized density ratio, batch-size requirements.
- **CausVid** (Yin et al.) and **Self-Forcing** (Huang et al.) — the closest
  published systems to ours: causal AR video students distilled from
  bidirectional teachers with DMD; how they serve the teacher (do they score
  full windows bidirectionally? how do they avoid the future-leak we hit?),
  their rollout/gradient-truncation choices, KV-cache handling, ODE-init,
  and their critic training conditioning.
- Any other SOTA one/few-step video distillation with relevance (SiD, SIM,
  Seaweed/APT-style adversarial post-training, MagCache-era Wan distills,
  LightX2V/Wan2.1-distill recipes, etc.).

Then answer, concretely and with citations:

1. **Per-step diff**: for each stage — (a) timestep sampling, (b) noising
   and scoring, (c) score-difference gradient construction and
   normalization, (d) loss surrogate + masking, (e) critic training
   objective/conditioning/update ratio, (f) student rollout & gradient
   path, (g) EMA / initialization / optimizer settings — what do the SOTA
   implementations do differently from us, and which of our deviations are
   known in the literature to matter?
2. **What are we missing entirely?** Candidates we suspect but have not
   tried properly: timestep-dependent weighting of the DMD gradient (SNR /
   sigma^2 weightings), two-time-scale learning rates rather than update
   counts, generator EMA for eval/serving, CFG on the teacher done right
   (e.g. smaller scales, or CFG only at high t), scheduled rung curricula,
   proper regression-loss anchors (DMD1-style) instead of our stat anchors,
   backward simulation of student states for the critic, larger effective
   batch via gradient accumulation, noise-shared (paired) scoring. Tell us
   which of these the field considers load-bearing for stability and which
   are cargo cult.
3. **Judge our idiosyncratic parts** against any precedent you can find:
   the 42f joint-window scoring of a causal student through a bidirectional
   TF teacher (and the clean-half drift/match curricula), the MAE gate, the
   denom floor 0.05, sampling t only at the student's rungs, the AR-served
   dual head with commit=student, the stat anchor with rel_tol=0, and the
   CARN commit-site affine. For each: is there a standard technique that
   solves the same problem more cleanly?
4. **Rank the top 5-10 most suspicious deviations** from field practice, by
   (expected impact on our observed failure modes: mean-collapse /
   variance contraction, texture drift/cartoonification, long-horizon DC
   drift) x (confidence that the field's way is better). For each, name the
   exact change you would make and the source that justifies it.

Format: a structured report with sections matching questions 1-4, explicit
paper/repo citations (arXiv IDs, GitHub paths where possible), and a final
one-page executive summary. Where our transcription is ambiguous, state your
assumption instead of asking.
