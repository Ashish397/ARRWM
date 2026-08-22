# DMD_FIX_REPORT — four research-directed DMD fixes (arm `rolldmdfix`, 2026-08-22)

Full cycle: implement → adversarial subagent review → address findings → launch.
All changes flag-gated; every default reproduces prior behavior byte-identically.
Files touched: `model/dmd_action_forcing.py`, `trainer/causal_action_forcing_train.py`,
`trainer/causal_rolling_staircase_train.py`. Backups in `/tmp/*.bak_dmdfix`;
session diff at scratchpad `dmdfix.diff` (~360 lines). `ast.parse` + `py_compile`
clean after every edit.

---

## Fix 1 — Score-timestep distribution (continuous path revived)

**What the code actually did before.** `dmd_sample_at_rungs=true` drew t
uniformly from the student's own 4-rung ladder {1000, 625, 357, 208} — 25% of
the gradient budget at t=1000 (maximum mean-collapse pressure) and zero mass
below 208 (where texture/covariance form). The older continuous path
(`_sample_dmd_timestep`, `model/dmd_action_forcing.py:~5470`) was intact but
bypassed: uniform in `[min_score_timestep=0, max_score_timestep=1000)` →
SD3-style `timestep_shift` (top-level key, **5.0** from
`configs/action_forcing_phase1.yaml:561` — distinct from
`model_kwargs.timestep_shift`, which stays the scorers' sigma-grid shift) →
clamp to hard-coded `[int(0.02·T), int(0.98·T)] = [20, 980]`. `ts_schedule` /
`ts_schedule_max` are both false in the config (CF parity), so no
pipeline-driven clamp interferes.

**Change.** New knobs `dmd_score_t_min` / `dmd_score_t_max` bind to
`self.min_step` / `self.max_step` (the final clamp). Defaults = the legacy
0.02/0.98 convention → byte-identical when unset. Validation
`0 ≤ min < max ≤ T`, None-tolerant (a present-but-null YAML key no longer
`int(None)`-crashes). The rung path deliberately still bypasses the clamp
(clamping had silently turned rung 1000 into 980).

**Verified.** With `dmd_sample_at_rungs=false` the run draws the continuous
shift-5.0 distribution (Self-Forcing's exact shift): p25≈625, med≈833,
p75≈938, ~5% of draws below t=208, clamped to [20, 980]. The shift knob
exists and is live (`timestep_shift`, read only by `_sample_dmd_timestep`).
**Review caveat (documented in-code):** `_sample_dmd_timestep` also serves the
critic's training-t and the aux-teacher pass, so these knobs move those draws
too — deliberate DMD symmetry; in this run the values equal the old defaults
so nothing shifts.

## Fix 2 — Supervise one random roll per ride (`dmd_supervise_roll_mode`)

**Before.** DMD fired at EVERY roll: with random depth 2–6, that is 2–6
correlated reverse-KL terms per ride — a hidden LR multiplier that also
over-weights shallow (near-GT-context) rolls relative to inference.

**Change.** `dmd_supervise_roll_mode` = `"all"` (default) | `"last"` |
`"random"`:
- `"last"` maps onto the existing `dmd_only_last_chunk_per_ride` machinery
  (both directions: legacy flag ⇒ mode "last"; mode "last" ⇒ raises the legacy
  flag so every existing gate fires unchanged). Mutexes: random×only_last,
  last/random×only_first → loud `ValueError`.
- `"random"`: target roll drawn ONCE per ride, **rank-0 + `dist.broadcast`**,
  at the same site/trigger/device/dtype pattern as the `rolling_random_depth`
  draw (`trainer/causal_action_forcing_train.py:~9250`), uniform in
  `[1, max_rolls]` (post-random-depth cap), stamped on the model
  (`_dmd_supervise_target_roll`). The generator skip
  (`compute_generator_loss_streaming`) and the mirrored
  `compute_clean_match_offset` guard reuse the exact only-last
  `_dmd_scorer_skip_this_roll` path: scorer forwards skipped on non-target
  rolls, connected-zero loss returned, critic still trains on EVERY roll,
  clean_match offset zeroed (never stale) on skipped rolls.
- **Rank-uniformity** (the scorer contains collectives): both skip operands
  are uniform — target broadcast; `chunks_in_ride` lockstep via the
  MAX-reduced reset. Reviewer confirmed the draw sits above any early return
  in `_streaming_step` and fires on identical steps on every rank.
- **Self-healing (review finding):** effective target =
  `min(target, max_rolls_this_step)` in BOTH guards, so a mid-ride
  MIN-reduced capacity-clamp shrink degrades to "deepest reachable roll"
  instead of blacking out the ride (the measured only-last pathology).
  Target 0 (unset) fails closed to "all". Logged:
  `dmd_supervise_target_roll`, plus existing `dmd_supervised_this_roll` /
  `dmd_supervised_count`.

## Fix 3 — fake_score init from the diffusion teacher

**Before.** `_mirror_generator_into_fake_score`
(`model/dmd_action_forcing.py:~4230`) cloned the fake_score from the 4-step
ODE-distilled student — the wrong prior for a score model (DMD2 initializes
the fake score from the base diffusion model).

**Change.** `fake_score_init_from_teacher` (default false = byte-identical).
When true, AFTER the generator mirror the fake DiT is overwritten with the
merged v14e teacher's weights (`real_score.model` post
`_load_real_score_with_v14_lora` — init order verified: load-real → mirror →
fake-LoRA wrap). Architecturally clean: real_score and fake_score are the
SAME `WanDiffusionWrapper(is_causal=False)` 1.3B class with the same action
patches (`model/base.py:150-240`) — keys map 1:1; fake-only keys (e.g.
`head_alt`) keep the generator-mirror value. Fails LOUD: shape mismatches
raise inside `load_state_dict` even with strict=False, and a ≥90%-keys-loaded
audit raises otherwise (a silent fallback to the ODE clone is exactly the
failure mode the flag exists to kill).
**Review MAJOR (fixed):** under peft 0.18 the online-teacher branch
(`real_teacher_train_online=true`) leaves `real_score.model` peft-wrapped and
`get_base_model().state_dict()` is `base_layer/lora_`-keyed → ~50% keys would
miss. Now: prefer the bare merged `real_score_frozen` copy when the
frozen-pass branch built one; otherwise refuse the peft-wrapped source with a
clear `RuntimeError`. (Not hit by this run — frozen teacher.)

**Resume interaction (would have been a silent no-op).** The warm-start
symlink + `auto_resume=true` path (`_maybe_resume`,
`trainer/causal_rolling_staircase_train.py:~1760`) restores
`ckpt["fake_score"]` — verified present in the warm-start checkpoint — which
would have silently overwritten the teacher init. New knob
`resume_load_fake_score` (default true = byte-identical): when false, skips
restoring fake_score + fake_optimizer (both, verified — teacher weights with
another parameterization's warm Adam moments is its own bug), touches nothing
else (generator/optimizer/EMA/step restore unaffected; the
ActionForcingDMDTrainer resume-append path never re-loads fake_score). Safe
here because `CKPT_EVERY=100000` ⇒ no mid-run checkpoints, so a requeue
restarts from the same warm-start symlink anyway; do NOT set false on runs
relying on periodic checkpoints.

**Critic-only warmup — what `dmd_loss_start_step` actually does.** Verified:
below `dmd_loss_start_step` the generator's DMD term is 0-weighted
(`_resolved_dmd_loss_weight`; the AR head is inside `dmd_loss` before the
weighting, so AR=1.0/TF=0.0 is gated too), while `compute_critic_loss_streaming`
+ `fake_optimizer.step()` run on every active step ungated → fake-only warmup
works with NO new gate needed. Two discoveries that changed the launch value:
1. **The warm-start checkpoint stores `step=300`** (the symlink's
   `phase1_step0000200.pt` name does not change the stored step). The
   prescribed `dmd_loss_start_step=250` would therefore have been a silent
   no-op (300 ≥ 250 ⇒ DMD active from the first resumed step). Launched with
   **`dmd_loss_start_step=350`** = 50 steps of critic-only warmup after
   resume (+ the base config's 20-step ramp to full weight by 370).
2. **The "5:1 two-time-scale" of the review doc does not hold in streaming
   K=1 mode.** `_fwdbwd_streaming_step` makes the standalone critic iter a
   no-op; gen loss AND critic loss both run inside `_streaming_step`, which
   only runs on `step % dfake_gen_update_ratio == 0`. So with ratio 5, 4 of 5
   outer steps do nothing and the TRUE critic:gen update ratio is 1:1 per
   active step. The 50 warmup steps are therefore **10 actual fake-score
   updates**, not 50. Acceptable: with fake ≡ teacher at init,
   `pred_fake − pred_real ≈ 0`, so the DMD gradient starts near zero and
   grows only as the critic specializes — DMD2 itself uses no critic-only
   warmup at all. But the ratio finding is worth its own follow-up (the
   two-time-scale rule we believed we had, we do not have).

## Fix 4 — fake_score EMA: the explicit answer

**`fake_score_ema_weight` does NOT supply `pred_fake` and is NOT
checkpoint-only.** It is a Flash-DMD §3.3-style post-step pull
(`_maybe_ema_fake_score_from_generator`,
`trainer/causal_action_forcing_train.py:1683-1740`, applied at `:2038` after
the fake optimizer step on generator iters): `ψ ← w·ψ + (1−w)·θ` — the LIVE
fake_score params are dragged toward the GENERATOR's params. `pred_fake`
always comes from the live fake_score; there is no shadow copy (the
`_real_score_ema_swap` mechanism is the real-score side only). At the
previous runs' 0.95 this blended 5% generator into the critic per generator
step — with teacher-init (Fix 3) it would have erased the teacher prior with
a ~13-gen-step half-life. **Run config sets `fake_score_ema_weight=0.0`**
(disables the pull; also the trainer-side default), which is the correct
setting given Fix 3, not merely optional.

## Additional run-config delta (user directive, 2026-08-22)

**`dmd_normalization_denom_floor=1.0`** added to the DEXTRA (knob
pre-existing and verified bound at `model/dmd_action_forcing.py:2474`; used by
BOTH the TF normalizer clamp and the AR-head normalizer). Semantics: the DMD
gradient divides by `max(|x0 − pred_real|.mean(), floor)`. The legacy floor
0.05 only guards the extreme cusp — it still AMPLIFIES the gradient up to 20×
whenever the per-sample normalizer < 1. Floor 1.0 = divide by `max(|f|, 1)`:
normalizers ≥ 1 behave as the standard eq.(8) down-scaling, but once the
student nears the teacher (|f| < 1) the division is by 1 — no amplification
at the cusp of convergence. Composes with the overpowered-DMD-step diagnosis.

## Adversarial review — findings and resolutions

Reviewer (subagent) briefed with the session diff + hazard classes
(DDP-collective divergence, silent config no-ops, stale state, only-last
back-compat, fake-init shape mismatch). **No BLOCKER.** Findings:

1. **MAJOR** — teacher-init crashes at construction under
   `real_teacher_train_online=true` (peft `get_base_model()` state_dict is
   `base_layer.`-keyed; empirically verified against peft 0.18.1). *Fixed:*
   prefer `real_score_frozen`; loud refusal otherwise. Latent (not this run's
   path), but was a guaranteed future crash with a misleading comment.
2. **MINOR** — `dmd_score_t_min/max` also reshape the critic/aux-teacher t
   draws (shared sampler). *Resolved:* documented in-code + here; run values
   equal the old defaults so no behavior change this run.
3. **MINOR** — mode "random" (unlike only-last) did not self-heal when the
   capacity clamp shrank below the frozen target → unsupervised rides.
   *Fixed:* `min(target, cap)` in both guards, rank-uniform.
4. **NIT** — `dmd_score_t_min: null` would `int(None)`-crash. *Fixed:*
   None-tolerant parse.
5. **NIT** — comments overstated "42f builder skipped" (only the scorer
   forwards are skipped; builder runs and feeds the critic at offset 0 —
   identical to the only-last precedent). *Fixed:* comments corrected.

Reviewer's rule-outs (verbatim conclusions): draw + skip provably
rank-uniform incl. resume/exhaustion paths; every launch knob traced to a
live consumer; no early return leaves `_dmd_scorer_skip_this_roll` or the
target stale (flag consumed between set and clear; cleared at the end of the
gen loss; the critic call after it never sees it set); legacy only-last
configs byte-identical; clean_match mirror and generator gate read the same
1-indexed counter and the same MIN-reduced cap; resume gate encloses exactly
fake_score+fake_optimizer and is rank-guarded.

## Launch

Holder **6093810** (hold-2n-6h, RUNNING, 0 steps via `squeue -s`, ~4h18m
left at claim; locked via `/tmp/claimed_6093810`). Run dir
`logs/dmd10k_rolldmdfix/dmd10k_rolldmdfix_h6093810_095750`, warm-start
symlink `phase1_step0000200.pt → dmd10k_dmd3kl_GAN_h6067260_103050/
phase1_step0000300.pt` (stored step=300). Command = prescribed
`_roll_holder.sh` invocation with three verified adjustments:
- `critic_lr` → **`fake_lr=4e-7`** (the fake optimizer reads `fake_lr`,
  `trainer/causal_rolling_staircase_train.py:736`; the config key
  `critic_lr` is the unrelated ACTION-critic LR at 3e-4 — using it would
  have been a silent no-op on the fake_score AND a 100× cut to the action
  critic). Pairing gen `lr=2e-6` / fake `4e-7` exactly as prescribed
  (Self-Forcing's pairing).
- `dmd_loss_start_step=350` (not 250 — see Fix 3, checkpoint stores step 300).
- `+ resume_load_fake_score=false` (see Fix 3 — without it, teacher-init is
  overwritten by the resume) and `+ dmd_normalization_denom_floor=1.0`
  (user directive).

### Launch status — VERIFIED STEPPING (2026-08-22 10:03 UTC)

Every new mechanism observed live in `logs/exp_rolldmdfix.err`:
- `fake_score initialized from the merged v14 teacher (real_score): 825/825
  keys loaded, missing=0, unexpected=0` (09:59:01).
- `Auto-resuming from .../phase1_step0000200.pt` → resumed at **step 300**
  (as predicted from the stored step) with
  `resume: resume_load_fake_score=false — SKIPPING the checkpoint's
  fake_score/fake_optimizer restore` (teacher init survived the resume).
- Critic-only warmup (steps 300–349): e.g. step 311 `critic_loss=0.1338
  fake_grad_norm=0.6523 gen_grad_norm=0.0000`; critic loss fell
  0.134→0.045 across the warmup; stat-anchor gradient reached the
  generator on some rolls (step 331 `gen_grad_norm=3.02`) as designed
  (anti-collapse is deliberately not gated by the DMD start step).
- DMD activation on schedule: step 351 log (work of step 350, ramp weight
  0) `gen_grad_norm=0.0`; step 361 log (work of step 360, ramp weight 0.5,
  a target roll) `gen_loss=1.4688 gen_grad_norm=49.5` (clipped to
  max_grad_norm=1.0) — generator DMD flowing.
- `gen/dmd_supervise_target_roll` confirmed present in the wandb stream
  (`wandb/wandb/run-20260822_095909-utiu3nsr`); rides cycling with random
  depth (caps 2/3/4/5 observed) and `reset=1(cap)` at ride ends; 13 rolls
  in the first ~4 min of active training; peak 42–48 GB; no NCCL stalls,
  no failure signatures. **The reviewer's self-healing `min(target, cap)`
  fix is already load-bearing:** `streaming_max_length=60` capacity-clamps
  many rides to 3 rolls while the random depth draws up to 6 — without
  the cap-min, targets 4–6 would have blacked those rides out.

Run: `logs/dmd10k_rolldmdfix/dmd10k_rolldmdfix_h6093810_095750`, wandb run
`utiu3nsr` (longlive-phase-3), holder 6093810 (~4h15m at launch; 700 steps
from 300 ≈ 80 active steps ≈ well within budget). No relaunch needed.

## Design context recorded (do NOT implement yet) — Causal-rCM "replayed backprop"

Wave-5 candidate, natural successor to `dmd_supervise_roll_mode=random` if
the random-roll arm shows improvement. Causal-rCM: (1) full causal rollout
entirely `no_grad`, saving per-chunk pre-final-denoise state + timestep +
detached KV; (2) ONE teacher+fake scoring pass over the WHOLE generated video
producing ONE coherent DMD target; (3) replay each chunk's final student
forward and backward against its SLICE of that global target using readonly
saved KV. The sharp question for our design: our rolling applies DMD per-roll
over OVERLAPPING 42f windows whose teacher conditioning (clean-match offset,
drift position) changes at each roll — should the target instead be computed
once from a coherent on-policy trajectory with fixed conditioning, then
sliced? Caveat: our rides (2–6 rolls × 9–18f) exceed one 21f teacher window,
so a direct transplant needs per-21f-segment targets over the frozen
trajectory — the transferable property is TARGET COHERENCE (one scoring world
per ride, frozen trajectory, consistent conditioning) vs our per-roll
re-matched windows. Random-roll is the cheap approximation (one target, one
location, no overlap correlation); replayed-slicing is the full version.

## Knob summary (all new, all default = previous behavior)

| knob | default | this run | effect |
|---|---|---|---|
| `dmd_score_t_min` / `dmd_score_t_max` | 0.02·T / 0.98·T (=20/980) | 20 / 980 | final clamp of the continuous DMD-t sampler (shared with critic/aux draws) |
| `dmd_supervise_roll_mode` | `all` | `random` | which roll(s) get generator DMD; `last` ≡ legacy only-last |
| `fake_score_init_from_teacher` | false | true | critic starts as the merged v14e diffusion teacher, not the ODE student |
| `resume_load_fake_score` | true | false | skip ckpt fake_score/fake_optimizer restore on auto_resume warm starts |
| (existing) `fake_score_ema_weight` | 0.0 | 0.0 | generator→critic EMA pull OFF (was 0.95 in prior runs) |
| (existing) `dmd_normalization_denom_floor` | 0.05 | 1.0 | eq.(8) floor: no gradient amplification below normalizer 1 |

---

# ADDENDUM (2026-08-22 ~10:45 UTC) — TRUE 5:1 two-time-scale (`rolldmdfix51`)

**Motivation.** Follow-up on the ratio discovery above: in streaming K=1 mode
the standalone critic iters of the outer loop are no-ops
(`_fwdbwd_streaming_step` returns early), so `dfake_gen_update_ratio=5`
never delivered DMD2's 5:1 — the true fake:gen update ratio was **1:1 per
active step** and 4 of 5 outer steps did nothing (also why `rolldmdfix`
finished 700 "steps" in ~28 min).

**Implementation** (`trainer/causal_action_forcing_train.py`, one block after
the main `critic_loss.backward()` in `_streaming_step`; backup
`/tmp/causal_action_forcing_train.py.bak_51`, diff `dmdfix51.diff`).
New flag **`streaming_fake_updates_per_gen`** (int, default 0 =
byte-identical): N extra SEQUENTIAL fake_score updates per active step —
each inner iter first consumes the gradient already on the fake params
(clip at `fake_max_grad_norm` + step + zero), then runs a fresh
`compute_critic_loss_streaming` on the same detached chunk with **fresh
(ε, t) draws** and backwards it; the last inner gradient is consumed by the
outer loop's existing fake step ⇒ **N+1 sequential updates per gen update**
(N=4 ⇒ 5:1). Mirrors the reviewed `teacher_cadence='fake'` inner-loop
pattern.

**Design decision (documented in-code): inline in the active step, NOT
rolling new chunks on the idle non-gen iters** (the coordinator-preferred
alternative — considered and rejected):
1. The generator is FROZEN between gen updates — extra rollouts would sample
   the same policy at ~5× rollout wall-clock; what the critic needs is
   (ε, t) coverage, which each inner update gets fresh.
2. Rolling on critic iters advances `chunks_in_ride`, so under
   `dmd_supervise_roll_mode=random` the per-ride target roll lands on a
   non-gen iter ~4/5 of the time — rides of depth <5 could contain ZERO
   gen-iter rolls ⇒ structurally unsupervised rides, gutting the arm this
   is A/B'd against.
3. Ride/reset/capacity/supervise-gate logic stays byte-identical ⇒
   single-variable A/B vs `rolldmdfix`.

**Adversarial review #2** (subagent, on the 5:1 diff): no BLOCKER on the run
path. Verified: streaming state provably alive at the insertion point (all
ride teardown is Stage 5, after this Stage-4 block; setup at next step's
top); each inner call is a self-contained DDP cycle (AR critic term resets
its local KV cache + RoPE memo per call; standard DDP wrap, no
static_graph); grads flow untouched to the outer fake step (grepped every
intervening block); gtfix critic path never writes streaming state /
eval stash / debug dumps; rank-uniform (config-driven N, allreduced grads);
memory = one critic graph per inner iter, freed each backward. Findings
fixed: **MAJOR(latent)** legacy (non-42f/non-asym) critic path would
double-backward a freed shared cond graph → loud guard requiring the gtfix
path; **MAJOR(latent)** forward-noiser mse-fold would silently accumulate
(N+1)× FN gradient into one FN step → loud guard; **MINOR**
`fake_score_updates_enabled=false` + flag>0 would silently run zero fake
updates → loud guard; NITs: null-tolerant flag parse, GPU sync only on last
inner iter, corrected reset-timing comment. Known cosmetics: rate-limited
42f debug prints consumed ~5× faster; `fake_updates_total` gauge not
checkpointed (resets on resume).

**Holder 6093812 cleanup.** Inspected both nodes via `srun --overlap`
(surgical, per-PID approach — no stray `torchrun`/trainer processes found;
all 8 GPUs at 1–3 MiB / 0%). Nothing to kill; the partial `rolldmdlast`
srun never started.

**Launch — `rolldmdfix51`, VERIFIED.** Holder 6093812 (PORTOFF 1448), run
dir `logs/dmd10k_rolldmdfix51/dmd10k_rolldmdfix51_h6093812_103318`, wandb
`krzk0tee`. EXACTLY the `rolldmdfix` config + `streaming_fake_updates_per_gen=4`.
Observed live: teacher-init 825/825, resume-skip, resumed step 300;
**5× machinery confirmed** by (a) the `gen/critic_extra_updates` /
`gen/fake_updates_total` / `gen/critic_extra_loss_last` keys present in the
wandb stream (emitted only from inside the executed loop) and (b) active-step
wall time ~25–35 s vs the sibling's ~17 s — the cost of 4 extra critic
passes; critic-only warmup 300–349 (critic_loss 0.107→0.056); DMD opened on
schedule (step 361 log: `gen_grad_norm=11.1`, a target roll at ramp weight
0.5); peak 42.7 GB; zero failure signatures. Early signal: critic_loss at
the same point is lower than the sibling's (0.035 vs 0.045) — consistent
with 5× critic adaptation.

**Sibling status.** `rolldmdfix` (holder 6093810) **completed step 700** at
10:26 UTC, final checkpoint
`logs/dmd10k_rolldmdfix/dmd10k_rolldmdfix_h6093810_095750/phase1_step0000700.pt`
(the post-completion torchelastic exit-barrier timeout in its log is benign
teardown noise). Holder 6093810 has ~3.5 h remaining and is free again.
A/B pair for scoring: `rolldmdfix` (1:1) vs `rolldmdfix51` (5:1), all else
identical.
