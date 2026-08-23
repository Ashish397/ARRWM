# OPUS HANDOFF PLAN (2026-08-22, written at Fable token exhaustion)

Read first: docs/ROLLING_CAMPAIGN.md (ledger+rules), DMD_GAN_SESSION_LOG_2026-08-22.md
(successor session), GAN_FORENSICS.md, CARN_V2_DESIGN.md, DMD_FIX_REPORT.md,
PROMPT_DMD_REVIEW.md / PROMPT_GAN_REVIEW.md (system description w/ file:line).

## Standing rules (user, permanent)
- Never: only-last supervision, context_noise>0, cancel holders, checkpoints on holders
  (2-node optimizer state = invalid warm starts; end-of-run saves happen anyway - ignore/delete).
- Always: warm-start from closest ARCH-MATCHED ckpt via SYMLINK named phase1_step0000200.pt
  (never cp - a cp spree filled the 5T volume once); real_guidance_scale=0.0; 60-90 min/arm
  on holders (~380 steps warm from 200); submit real jobs to sbatch queue with ckpts.
- KL ODE ckpt + 14e are the trusted theoretical baseline. Pre-June DMD/GAN assumptions: void.
- Holders: never edit a bash script a live run is executing in place (lazy-read corruption);
  clear GPU zombies before reusing a freed holder (pkill python via srun --overlap; the
  reaper pattern in /tmp/lock_reaper.sh did this - may need restarting in a new session).

## Current holders/runs (check squeue first; this snapshot ages fast)
- 6100009 RUNNING: fullcarn_bidir_kl_wave01 (wavelet GAN arm); afterwards its command file
  chain was meant to run the matched no-GAN 60s CARN-on eval.
- 6100985 PENDING: ganfix_strict rerun - THE decisive GAN assembly (see below). Launch file
  logs/.holder_cmd_6100985.sh -> sbatch/run_ganfix_strict_rerun.sh (I just patched it).
- 6100986 PENDING: projected-GAN arm via sbatch/run_full_carn_probe.sh (also patched).

## What I changed today (verify then rely on)
1. fake_score_causal_mask knob (model/dmd_action_forcing.py, near real_score mask assignment):
   fake score's TF mask was NEVER set -> defaulted CAUSAL while real ran BIDIRECTIONAL =
   asymmetric conditionals in grad = pred_fake - pred_real (biased non-vanishing term; same
   class as the documented AR-head v1 bug). Default true = historical byte-identical.
   BOTH queued launches now append: real_teacher_causal_mask=false fake_score_causal_mask=false.
   -> EVERY new DMD arm must carry both flags.
2. Both queued launches also got dmd_grad_target_norm=1.0 (cap-only rescale of the DMD grad,
   implemented earlier; tau=1.0 is PROVISIONAL - calibrate to the median dmdtrain_gradient_norm
   of a healthy run (rollcarn700 wandb run-20260822_042334-0sp23h3c) and update).
3. Recommended normalizer stack now IN the strict rerun: causal denom (dmd_ar_normalization_source=ar)
   + grad-norm target + floor ~0 (1e-6). These compose: correctness + gain governor.

## The exposure-bias cliff (5.25 s) - explanation to carry forward
local_attn_size=21 latent frames x 4 RGB/latent / 16 fps = 5.25 s. Sliding attention IS active.
At 5.25 s the seed is fully evicted: every attended token is self-generated. Training gave the
model dense supervision mostly on GT-anchored context; its own outputs carry a small statistical
signature (channel-mean walk, HF loss - measured, specrad 1.2 amplifying), so fully-self context
is OOD -> the model falls to its conditional-prior attractor (low-motion jitter). Not "dislikes
its rollouts" - insufficiently trained on the fully-self distribution + previously biased gradients
(fake-causal asymmetry, wrong denominators, dead knobs). Fixes stack: symmetric bidirectional
scorers, causal denom, grad cap, dense rolling supervision past depth 21/9=3 rolls, CARN at
inference (now in evaluator), and eventually a healthy GAN for texture.

## PRIORITY 1 - GAN: assemble properly, test properly (user directive)
The strict rerun (6100985) is the first full assembly: cross-ride real bank 4096 (no nearest-L1
in bank; match_k retained only for GT-transition variant), sampled disc t (shift 5, [20,980]) same
for real+fake, scalar logit + frozen projector mixing, logistic-family loss w/ R1+R2 gamma=1.0
token-norm OFF, weight 0.03, lr_D 1e-5, strict action conditioning, symmetric bidir scorers,
gen EMA 0.99. ACTIONS:
a) When 6100985 starts, verify launch, verify d_loss learns (must leave 0.693; healthy target:
   settle 0.3-0.6, NEVER sustained <0.10 - breakdown threshold measured repeatedly).
b) Log/verify ratio ||g_GAN||/||g_DMD|| in 5-20 pct band (metric may need adding - check).
c) THE TEST (user: "test it properly"): fixed-route fixed-seed 60 s eval, SAME Madrid route/actions/
   step/checkpoint, inference-CARN ON, GAN arm vs matched no-GAN control
   (fullcarn_bidir_kl_nogan200 lineage). Pass criteria in DMD_GAN_SESSION_LOG "Immediate
   Decision Rule" (margin sustained; controlled G pressure; motion+texture beyond 5.25 s; better
   60 s video without trading failure modes).
d) If strict critic learns but motion still freezes -> transition representation insufficient ->
   escalate to trajectory-level critic (GAN_FORENSICS ranked designs) or feature-stat matching
   (CARN_V2 non-adversarial control: match the MEASURED drifting stats - spectrum/covariance/kurtosis).
e) Action-blind vs strict comparison already half-done (ganfix_blind learned margin 0.65->0.54).

## PRIORITY 2 - DMD recipe consolidation
Current best-known stack (all in strict rerun DEXTRA): fresh KL-ODE init, bidir 14e real+fake
(BOTH flags), continuous t [20,980] shift 5, supervise ONE random roll, teacher-init fake score
(+resume_load_fake_score=false), fake EMA 0.0, 5:1 via streaming_fake_updates_per_gen=4,
lr 2e-6 / fake_lr 4e-7, causal denom, grad target, floor 1e-6, CARN 0.5 (+ inference CARN),
gt feeding + clean match + drift compose. OPEN: A/B/C roll-mode comparison unfinished ("last"
mode never run); Causal-rCM coherent-target (one scoring pass per frozen trajectory, slice per
chunk) is the designed wave-5 successor if random-roll wins; verify dmd_grad_target_norm tau.

## PRIORITY 3 - open threads
- rolldmdfix (1:1) vs rolldmdfix51 (5:1) vs rolldmd51f0 (floor 0): completed runs, scoring
  possibly unfinished - compare drift/texture/cartoon + d(t) histograms (research wants the
  denominator D distribution by t and roll depth logged before trusting any floor default).
- 51 was MORE cartoonish than 1:1 (user) - consistent w/ better critic => stronger reverse-KL
  step; the grad cap is the intended cure - verify on strict rerun.
- Flow timelines: 3/5 made (analysis/flow_rolling/), rollnocarn + one more pending.
- rollnocarn (CARN ablation, GAN off) completed - score vs rollcarn700 (466-wall + drift).
- Debug input dumps exist: logs/dmd10k_rolldbg/*/samples/dbg_inputs_step*.mp4 + docs/DBG_INPUTS.md
  (user wanted eyes on clean-match refs for DMD + GAN pairs).
- wandb cloud sync fails silently sometimes: wandb sync wandb/wandb/run-<id> after runs.

## Verdict shorthand (measured, replicated - do not relitigate without new data)
- CARN seam affine lambda 0.5 = load-bearing (drift flat to 700 steps, depth 21); closed-loop
  only (open-loop bias and temp scaling both failed). EMA target + spectrum re-anchor = designed v2.
- R1 gamma 1e6 pinned every disc (ln2); weight 1.0 = 33x One-Forcing; old wavelet-kill result
  was t=0-features artifact; real-pool memorization (22 rides-local) was the disc-domination root.
- d_loss < 0.10 sustained = generator breakdown (rolllong step ~466-541 sequence: divergence
  dies -> pan/thrash -> frozen). Stat anchor = brake not cure; rel_tol 0.2 disables anchors,
  0.05/0.03 breathes, 0 pins.

---

## ONBOARDING PROMPT (start a fresh Opus agent with exactly this)

You are taking over an active research campaign in /scratch/u6ex/as1748.u6ex/ARRWM:
distilling a 14e bidirectional video-diffusion teacher into a CAUSAL few-step
autoregressive driving world model (Wan DiT, KV-cache streaming, 2-node Slurm
"holder" jobs for experiments, 8-node sbatch for validations). The user is a
hands-on researcher who reviews videos personally; their visual judgment has
repeatedly out-diagnosed our metrics — treat it as ground truth and build
metrics to match it, not the other way round.

FIRST ACTIONS (in order):
1. Read this file fully, then docs/PLAN_22-8_FINISH.md (your work queue),
   then docs/ROLLING_CAMPAIGN.md (rules + verdict ledger). Skim
   DMD_GAN_SESSION_LOG_2026-08-22.md, GAN_FORENSICS.md, CARN_V2_DESIGN.md.
2. `squeue -u $USER` — find the holders. Verify the ganfix_strict rerun
   (holder 6100985 or successor) launched with real_teacher_causal_mask=false
   fake_score_causal_mask=false dmd_grad_target_norm=1.0 in its config echo.
   If its holder expired before launch, relaunch via
   logs/.holder_cmd_6100985.sh -> sbatch/run_ganfix_strict_rerun.sh on a fresh
   holder (sbatch sbatch/hold_nodes_2node_6h.sbatch; wait for RUNNING; clear
   GPU zombies via srun --overlap pkill python first).
3. Execute PLAN_22-8_FINISH.md steps 1-2 (the strict-GAN assembly + the
   fixed-route 60 s evaluation). Everything else branches on that verdict.
4. HOLDER FLEET POLICY (standing user directive): keep 4-5 holders online at
   all times. When the currently-pending holders come online, immediately queue
   two new ones (sbatch sbatch/hold_nodes_2node_6h.sbatch); repeat whenever the
   online count drops below 4. Holders take a while to schedule, so queue
   replacements the moment existing ones START, not when they expire. Never let
   experiments idle waiting for capacity — and never let a free holder sit
   unused while scored questions remain open.

OPERATING RULES (non-negotiable, learned expensively):
- Never cancel holder jobs; never write checkpoints on holders; warm-start via
  SYMLINK from arch-matched ckpts; 60-90 min per holder experiment; verify GPU
  zombies cleared before reusing a freed holder; never edit a script a live
  bash is executing (write-temp + mv); real_guidance_scale=0.0 always;
  context_noise stays 0 forever; no only-last supervision.
- Loss-path changes: implement flag-gated default-off, spawn an adversarial
  review subagent, address findings, THEN launch. Every unreviewed loss change
  in this campaign shipped with a real defect.
- Report failures plainly, score every finished arm before drawing conclusions,
  and never call a run healthy from videos alone (a "beautiful" run once
  trained nothing — the metrics check caught it).

## GUIDING THEORY (the user's frame — keep this at the center)

If self-generated context is done WELL, it should be statistically so close to
real-world-style generation that the model is perfectly comfortable conditioning
on it — no cliff, no attractor. The teacher and the fake score are where this
becomes achievable: they already know what the world looks like, and their job
is to lend that knowledge so the student can condition on its own outputs as if
they were real.

Corollary: whenever the model treats self-context and real context DIFFERENTLY,
that divergence is itself a signal — measurable and usable, not just a nuisance.
The division of labor:
- The GAN is the detector for the TEXTURE/STYLE component of that divergence
  (does self-context look real to a well-posed critic?).
- CARN is the corrector for the AUTOREGRESSIVE-DRIFT component (statistical
  walk of the context distribution, fixed closed-loop at the commit site).
Get both genuinely working — GAN assembled + passing the fixed-route eval,
CARN already validated — and whatever failure remains is, by construction, the
part neither texture-realism nor statistical drift explains. That residual is
the next research object. Do not skip to it before the two detectors work.
