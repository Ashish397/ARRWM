# CARN redesign — session handover, 2026-08-26

Written for handover in case this session is interrupted (it already has been
once today). If you are a new agent picking this up: **read this whole
document before touching anything**, then check the "Live state to verify"
section first — several things below were true as of writing but may have
changed by the time you read this.

## TL;DR

- CARN today is scattered across 5+ application sites, all flag-gated,
  **none enabled in any live run**. This session traced all of them, found
  the mechanism the researcher actually wants (correct the generator's
  output **in place**, at the moment it's committed to context, so the
  correction is structurally part of generation rather than a side-channel
  loss tweak), wrote it up in `docs/CARN_FINAL_PLAN.md`, and implemented +
  smoke-tested it. **Both background tracks are now complete.**
- **In-place commit-time mechanism (§4): mechanically validated.** A clean,
  node-matched on/off pair confirmed the corrector fires, trains, and its
  correction magnitude grows across calls (`rel|dz|` 0 → 0 → 0.0082 at
  calls 1/10/50) without destabilising anything. Does **not** yet show
  whether it *helps* — 40 steps is too short to move quality metrics. A
  useful correction also landed: this doc's original gradient-flow risk
  (Risk 6) was wrong, benignly — the commit site was already gradient-free,
  proven bit-identical, so R can't perturb the generator's own gradient at
  all, only the future context it conditions on.
- **Eval-side inference smoke (§4a): mechanically validated, effect
  ambiguous.** Byte-identical-off proved via md5. Two checkpoints tested;
  the better-trained one made rollout variance expand instead of contract —
  plausibly the right direction (fights the known contraction failure mode)
  or plausibly slow divergence. **Needs a human to watch the output videos**
  (already sent to the researcher) before concluding anything.
- **End goal, stated explicitly by the researcher:** this in-place
  commit-time mechanism becomes **the only CARN**. Everything else
  (GAN-discriminator application sites A-E, flash de-drift, pre-scoring
  de-drift, DMD-target de-drift) is meant to be retired once this is
  validated. **That retirement has not happened and is not authorized yet** —
  do not remove the old sites without a separate explicit go-ahead. Given the
  above, "validated" so far means "runs correctly and does something
  measurable," not "shown to improve anything" — retirement should wait for
  the latter.
- A third, unrelated session landed a change that makes `gt_vs_fake` never
  see CARN-noised GT (a call-site-only bypass, does not touch CARN's
  implementation). That work is safe to build on top of; do not duplicate or
  touch it.

## Standing project rule that applies to all of this

Per this project's long-standing convention (confirmed independently by a
peer session working elsewhere in this codebase): **any training-recipe
change needs explicit researcher sign-off before it is *enabled* in a real
run — not before it is written.** Every flag introduced in this session
defaults to `False` / is byte-identical off, and nothing described here has
been turned on in a full/long training run. The researcher has authorized
implementation + **short mechanism smokes** (tens of steps, proving the
thing runs and looks sane) for the in-place commit-time mechanism
specifically. A longer run, and the retirement of the older sites, both need
a fresh go-ahead.

---

## 1. What CARN is (for orientation)

A learned network (`model/forward_noiser.py`, class `ForwardNoiser`, ~23-30M
params, FiLM-conditioned 3D convnet, zero-init output projection so it's the
identity at init) that models this streaming AR video generator's own
rollout drift. Trained via `forward_noiser_reverse`:
- `false` (default): maps `[chunk at level n] -> [chunk at level n+1]`
  (ADDS drift — a drift-adder, used to synthesize "aged GT" for the GAN).
- `true`: maps `[chunk at level n] -> [chunk at level n-1]` (REMOVES drift —
  a de-CARN denoiser).

A second, optional network (`self.reverse_noiser`, same architecture, only
built when `forward_noiser_cycle_enabled=true`) is trained via
cycle-consistency to invert the forward noiser.

## 2. Every de-drift application site that exists, and its status

All of these use (or should use) the shared stepping helper
`_dedrift_with_reverse_noiser` (`model/dmd_action_forcing.py:12528-12594`):
relaxed multi-step Euler correction, `cur = cur + alpha * G(cur, level,
residual=False)`, `alpha = alpha0 * (decay**k)`, network params frozen but
input kept differentiable.

| # | site | file:lines | flag | status |
|---|---|---|---|---|
| 1 | GAN discriminator real-input augmentation (sites A-E) | `trainer/causal_action_forcing_train.py`, multiple sites; full audit in `analysis/gan_tuning/CARN_TRANSITION_REVIEW.md` | `forward_noiser_apply_gt_former` / `_apply_decoupled` / `ladd_gt_transition_carn_former` etc. | **Live**, in the `carn_ctrl`/`carn_fwd` sweep (see §3). Being decoupled from `gt_vs_fake` by another session (see §5) — leave alone. |
| 2 | Flash-slab de-drift | `model/dmd_action_forcing.py:15789` (line shifted from `:15740` as of an earlier edit — verify) | `reverse_noiser_dedrift_apply_to_flash` | Built, default off, not enabled anywhere. Per `THINGS_STILL_TO_DO.md` item A, this tensor isn't even what gets committed to context when flash is on — low-value site. |
| 3 | Pre-DMD-scoring de-drift of `train_chunk` | `trainer/causal_action_forcing_train.py:19321-19337` | `reverse_noiser_dedrift_enabled` | Built, default off, not enabled anywhere. **This is the existing mechanism that already does "de-drift the student chunk before real_score sees it"** — the researcher's own proposed fix during this session turned out to already exist here. Deliberately does NOT touch what gets committed to context (comment explicitly preserves `_raw_train_chunk` for that reason). |
| 4 | DMD-target de-drift (teacher's `pred_real_image`) | `model/dmd_action_forcing.py`, `_compute_kl_grad`, flag registered ~`:2700-2710`, applied ~`:8237-8260` | `reverse_noiser_dedrift_apply_to_real_target` | **Built this session.** Default off, not enabled. 19 new tests + 24 existing pass. Gradient-flow traced and proven inert to the generator (the whole call sits inside `torch.no_grad()`, DMD loss uses a detached manual-gradient injection — see `testing/test_reverse_noiser_dedrift_real_target.py`). **Superseded in intent by site 5 below** — this de-drifts the teacher's *output* (out-of-domain for R, trained on student rollouts); site 3 or site 5 are the correct-domain fixes. Recommend not pursuing this further; kept only because it's tested and harmless. |
| 5 | **In-place commit-time de-drift (THE NEW WORK)** | `pipeline/action_forcing_training.py`, both `inference_with_trajectory` and `generate_chunk_with_cache` (the "rollout twins" that already share `_carn_seam_correct`) | `reverse_noiser_dedrift_apply_to_commit` (new) | **In progress** — see §4 and "Live state to verify." This is the mechanism the researcher wants as the eventual sole CARN. |
| 6 | Inference/eval-time AR-rollout de-drift | `utils/eval_causal_AR.py`, `generate_ar`, new function `_apply_reverse_noiser_dedrift` | new CLI flags (`--reverse_noiser_checkpoint` etc.) | **DONE — smoke complete, results ambiguous, needs human video review.** See §4a. |

There is also a crude, non-learned mean/std affine correction
(`_carn_seam_correct` in `pipeline/action_forcing_training.py:1148`, and
`_apply_carn_seam_affine` in `utils/eval_causal_AR.py:169`) already live at
the actual commit sites in both training and eval. It's controlled by
`carn_seam_affine_lambda`, which is `0.0` (off) in every current live arm.
Site 5/6 above add the *learned* correction alongside it, not replacing it.

## 3. The `carn_ctrl` / `carn_fwd` GAN-discriminator sweep (earlier this session, separate from the redesign)

Full detail in `analysis/gan_tuning/CARN_TRANSITION_REVIEW.md`. Short
version: `carn_ctrl` (matched control, no CARN applied to the disc's real
input) and `carn_fwd` (direction-fixed CARN applied) are the "load-bearing
pair" for testing whether applying CARN to the GAN discriminator's real
input does anything. As of this session: only one full-length (200-step)
matched pair exists (one replicate died pre-warmup at step 25); the signal
is **inconclusive** — no metric clears the noise floor established by the
review's own A/A methodology. The review's protocol requires ≥3 seeds before
any conclusion. **This sweep is independent of the redesign work in §4** and
does not need to finish before the redesign proceeds, but do not read a
"CARN doesn't help" or "CARN helps" conclusion into it — it's genuinely
unresolved.

## 4. The redesign work this session (docs/CARN_FINAL_PLAN.md)

Read `docs/CARN_FINAL_PLAN.md` in full — it is the design doc. Summary:

**Mechanism:** apply the reverse-noiser R **in place**, at the moment a
chunk is committed to the KV cache, in both `inference_with_trajectory` and
`generate_chunk_with_cache`. New flag `reverse_noiser_dedrift_apply_to_commit`
(default False), reusing the existing `_dedrift_with_reverse_noiser` helper
and its `reverse_noiser_dedrift_level`/`_min_level`/`_steps`/`_alpha0`/
`_alpha_decay` knobs.

**Why concurrent CARN training matters:** once correction is live, the
model's *actual* observed drift changes, so R must keep training against the
currently-observed (partially-corrected) residual, not a frozen snapshot.
Traced and confirmed: `_prebuild_rollout2_for_v24`
(`model/dmd_action_forcing.py:13155`) — which builds the training pairs R
learns from — calls `generate_chunk_with_cache` (`:13267`), so correcting
that method's commit site should make R's own training pairs automatically
reflect the corrected dynamics with no separate plumbing. **This claim was
assigned to be proven with a direct test, not assumed — check whether that
test exists and passes (see "Live state to verify").**

**Deliberate reversal of an existing safeguard:** site 3 above (`train_chunk`
pre-scoring de-drift) explicitly avoids feeding corrected output into FN
training pairs ("would corrupt the training pairs"). This redesign does the
opposite on purpose — see `CARN_FINAL_PLAN.md`'s "A deliberate reversal"
section for the reasoning; don't "fix" one to match the other.

**Risks documented in the plan** (bootstrapping stability, exposure bias,
diminishing training signal as residual shrinks, extra compute cost, and
gradient-flow correctness — this site, unlike site 4, is gradient-tracked,
so R's correction needs to stay differentiable back into the generator, not
just be provably inert).

### Implementation status (files touched as of this writing — verify current state)

- `model/dmd_action_forcing.py` — modified, +154 lines net (includes both
  the earlier DMD-target work and this redesign's additions).
- `pipeline/action_forcing_training.py` — modified, +142 lines. New
  `self._reverse_noiser_dedrift_commit(...)`-style wrapper expected at the
  commit site(s) alongside `_carn_seam_correct`, in both rollout twins.
- `testing/test_reverse_noiser_dedrift_commit.py` — new file, exists.
  **Has not been confirmed passing by this document's author** — the
  implementing agent was still working when the session was interrupted and
  again mid-task when this doc was written. Run it before trusting it.
- `sbatch/carncommit_smoke.sbatch` — new file, a minimal clone of
  `gansig_carn_fwd.sbatch` (the in-flight GAN sweep arm) with only: 40 steps
  instead of 200, checkpointing effectively off, `forward_noiser_cycle_enabled=true`
  (needed so a corrector network exists under the default `fn_pair_mode`),
  `reverse_noiser_dedrift_enabled=true` (the shared gate), and
  `reverse_noiser_dedrift_apply_to_commit=true` as the actual treatment —
  controlled by a `COMMITDEDRIFT` env toggle for the on/off pair.
- `utils/eval_causal_AR.py` — modified, +273 lines. New function
  `_apply_reverse_noiser_dedrift`, new `generate_ar` params, new CLI flags,
  wired into the commit site near the existing `_apply_carn_seam_affine`
  call. This is the eval-only track (§2 site 6), separate from the training
  pipeline change, using the already-completed `carn_ctrl` run's
  `forward_noiser` (trained with `forward_noiser_reverse=true`, i.e. already
  a de-CARN denoiser) as the corrector — checkpoint at
  `ARRWM_data/logs/dmd10k_gansig_carn_ctrl/dmd10k_gansig_carn_ctrl_j6135660/fn_rev_step*.pt`
  (use the latest step available).

### Smoke run status (training-side, site 5) — messy, still not resolved

This has been the hardest part of the session to get a clean read on, mostly
due to infrastructure, not the mechanism. Sequence of events, so a resuming
agent doesn't repeat the same false reads:

1. First `carncommit_on`/`carncommit_off` pair on holders 6144621/6144623:
   `on` completed 40/40 cleanly; `off` died at startup (external node
   collision with another session's job, unrelated to this work).
2. `off` relaunches repeatedly failed with `exit=1` and no
   `[HOLDERSMOKE]` line — traced to a **broken collision-guard in the
   shared launcher** `sbatch/run_smoke_on_holder.sh` (owned by another
   session): `set -euo pipefail` combined with a `grep -v` that matches
   nothing on a genuinely idle holder, so the guard failed closed exactly
   when it should have passed. Fixed by that session (`|| true`), confirmed
   in the file now.
3. **A real false-positive trap, already fallen into once — watch for it.**
   `logs/holdersmoke_<tag>_h<holder>.log` is keyed by smoke-tag + holder,
   not by run/attempt, so **any relaunch on the same holder overwrites the
   previous run's console log in place**, leaving a file *named* for one run
   but *containing* another's output. This produced a false "the treatment
   flag never fired" conclusion earlier in the session, later retracted.
   **The reliable per-run record is the wandb output**, not the shared
   console log file: `wandb/wandb/run-<timestamp>-<runid>/files/output.log`
   and `wandb-summary.json` — note the doubled `wandb/wandb/` path
   (`WANDB_SAVE_DIR=wandb` + wandb's own subdirectory) and that `find`
   needs `-L` since `wandb/` is a symlink into `ARRWM_data`.
4. Using the wandb record, the original `carncommit_on` run (job 6144621,
   wandb run `vpzi86p6`) was confirmed to have genuinely fired the
   correction: `[CARN][commit-dedrift] ACTIVE at the KV commit: level=1
   rel|dz|=0`. But that run predates a later edit to the proof-line emitter
   (`pipeline/action_forcing_training.py`, current version logs on a
   geometric ladder of calls — `1, 10, 50, 200, 1000, 5000, 20000` — with a
   `carn_commit_dedrift_applied` counter; the original run has neither), so
   it only ever produced **one structurally-uninformative sample**, at the
   one call (the first) where `rel|dz|=0` is guaranteed regardless of
   whether the mechanism works (`out_proj` is zero-init). **This run cannot
   answer whether the correction does anything — not because it failed, but
   because it only has one data point at a point that's mathematically
   forced to read zero.**
5. Meanwhile another session accidentally launched its own `carncommit_off`
   baseline onto holder 6144621 (the SAME nodes the original treatment used)
   while trying to fix something else — this is actually valuable: it's a
   genuinely node-matched baseline, which cross-node comparisons in this
   codebase's own telemetry are known to be too noisy for (`d_real`/`gan_cos`
   95-119% CV, `r1` 76%, `ratio` 50% across nodes, per earlier findings this
   session). That baseline completed cleanly at 18:22:13 (job
   `dmd10k_carncommit_off_j6144621`, wandb run `gd14x6i2`), freeing the
   holder.
6. **As of this writing**: a fresh treatment run, under the current
   ladder-logging code, was being staged onto the now-free 6144621 (to stay
   node-matched with the baseline that just finished there), using a
   **different smoke tag** than before so it doesn't overwrite that
   baseline's console log. Not yet confirmed complete.

**Resolved.** A genuine node-matched pair (nid[010178,010185], both 40/40,
clean) landed: ON = wandb run `g6cvxmgz`, OFF = wandb run `gd14x6i2` (console
logs use distinct tags — `carncommit_smoke2` for the rerun — and the original
treatment log was preserved as `logs/carncommit_TREATMENT_h6144621.log.keep`
rather than left to be clobbered again).

**Headline result — the `rel|dz|` trajectory the whole redesign hinged on:**
```
call=1   rel|dz| = 0            (guaranteed — out_proj zero-init)
call=10  rel|dz| = 0
call=50  rel|dz| = 0.00819507   <-- growing off zero
```
R's weights moved, the correction became measurably non-zero, and nothing
destabilised (`gt_dist`, `fn_rev_loss`/`fn_fwd_loss`, peak memory all within
noise between arms). **The mechanism is mechanically sound and bootstrapping
as designed** — this was the central open question in §4 and it now has a
real answer. It does **not** yet show whether the correction *helps*
anything — 0.8% displacement after 50 calls is too small to move `gt_dist`,
and 40 steps never reaches the logging ladder's next rung (`call=200`). A
longer run is needed to see whether `rel|dz|` converges, keeps growing
(Risk 1), or goes quiet (Risk 3).

**A genuinely useful correction to this document's own Risk 6.** The
implementing agent found `commit_input_clean` is `.detach()`'d and the whole
commit forward runs inside `torch.no_grad()` — i.e. **no generator gradient
ever flowed through this site, with or without the change**. Risk 6's
premise (that this path was gradient-tracked and needed careful freeze/live
handling) was wrong, benignly — proven with a bit-identical
`torch.equal(grad_off, grad_on)` check, not just argued. R influences the
generator only by changing the *context it conditions on* for future chunks,
never by contributing to this step's loss gradient — a simpler, safer
picture than the design doc assumed.

**Rollout2-propagation: confirmed**, both by AST (asserting
`_prebuild_rollout2_for_v24` still calls `generate_chunk_with_cache`) and
executably (two consecutive chunks through the real method, chunk 2 differs
with the flag on, chunk 1 doesn't). One nuance the doc didn't state:
propagation starts at chunk **N+1**, not chunk N — the correction lands
after the current chunk is emitted.

**Open items for a human before this goes further:**
1. `carn_commit_dedrift_applied` (the wandb proof-counter) never actually
   reached wandb on the **streaming** path — `_last_extension_metrics`
   folding only happens on the non-streaming DMD path. Currently the only
   proof the flag fired is the stdout `[CARN][commit-dedrift]` line. Fix
   before any longer run, per this project's own "prove a flag fired from a
   counter" rule.
2. **A design choice needs explicit sign-off**: the correction is
   commit-only (KV-cache memory), while the emitted/scored chunk stays raw —
   deliberate, matches the plan, but is the *opposite* default convention
   from the existing affine correction (`carn_seam_affine_apply_to_output`
   defaults to correcting the output too). Pinned by a test so it can't
   drift silently, but someone should consciously agree this asymmetry is
   wanted.
3. The smoke's base recipe needed `forward_noiser_cycle_enabled=true` +
   `reverse_noiser_dedrift_enabled=true` on top of the new flag (it's inert
   without a corrector network) — both arms carried them equally so the
   contrast stayed single-variable, but the base recipe differs from stock
   `gansig_carn_fwd`, worth knowing when comparing against that sweep.
4. 40 steps is far too short to answer the actual research question — next
   step is a longer run (a few hundred steps minimum) watching whether
   `rel|dz|` converges, keeps climbing, or dies.

## 4a. Eval-side smoke (site 6, `utils/eval_causal_AR.py`) — COMPLETE, results ambiguous

Separate track from §4, no training involved. Status: done, all four arms
ran clean, byte-identical-off proved rigorously, but the "does it help"
question is unresolved and needs a human to actually watch the output
videos (sent to the researcher already — see chat).

**One real bug found and fixed in the prior agent's diff**: the correction
was applying a fixed `level=1` to every chunk regardless of position. The
trainer's actual reverse pairing conditions on the **input chunk's own
rollout position** (starting at 1 for the first generated chunk, climbing
from there — the `carn_ctrl` checkpoint's training log shows it only ever
saw conditioning levels `[4,5,6,7,8]`, never 1). Fixed via new
`--reverse_noiser_dedrift_level_auto` / `--reverse_noiser_dedrift_level_max`
flags (default **off**, so nothing changes unless explicitly requested) that
reproduce the trainer's indexing: `level = min(len(generated)+1,
level_max)`. **Anyone using a `chain_levels=true`-trained checkpoint should
pass `--reverse_noiser_dedrift_level_auto`** or they're querying a level the
network never trained on.

**Byte-identical-off proved, not just claimed**: ran the actual `git HEAD`
version of the file as a control arm (`predif`) against the working-tree
version with the flag unset (`base`) — the two rollout output `.mp4` files
are **exactly md5-identical**.

**Two checkpoints tested**, both `forward_noiser_reverse=true` (genuine
de-CARN denoisers), both loading `strict=True`:
- `dmd10k_gansig_carn_ctrl/..._j6135660/fn_rev_step0175.pt` — the same
  checkpoint referenced in §2/§4. **Found to be barely trained**: 0.94%
  weight change from init, and its correction strength is essentially
  **level-independent** (`rel|dz| ≈ 0.017` at level 1, 4, or 8 — the FiLM
  step-conditioning hasn't meaningfully learned). Don't draw a verdict from
  this arm alone.
- `dmd10k_carntx_match/..._h6117225_224923/fn_rev_step0125.pt` — a better
  find: trained with `chain_levels=false`, `forward_noiser_gt_match_frames=12`
  (`FN(student chunk) → clean GT` directly), and — importantly — this run
  also saved its own student checkpoint (`eval_step0200.pt`), making it a
  genuinely **matched student/corrector pair**, unlike `carn_ctrl` whose
  corrector has no checkpoint-compatible student of its own. More trained
  (`out_proj` norm 2.3x carn_ctrl's).

**Numeric result** (per-chunk latent std over a 15-chunk rollout, same seed,
same student, `carn_seam_affine_lambda=0.0` throughout):

| arm | std @ chunk 0 | std @ chunk 14 | trend |
|---|---|---|---|
| base (no de-drift) | 0.7867 | 0.7585 | mild contraction |
| carn_ctrl corrector | 0.7879 | 0.7421 | **contracts slightly *more*** |
| carntx_match corrector | 0.8080 | **0.9586** | **monotonically expands** |

The undertrained `carn_ctrl` arm moves in the *wrong* direction if the goal
is fighting variance contraction (see `project_ar_variance_contraction_law`
in memory — contraction is the known failure mode this whole CARN effort
traces back to). The `carntx_match` arm expands variance, which is the
*right direction* if contraction is the problem — but expansion could
equally mean the correction is slowly diverging/hallucinating rather than
correcting. **The videos have not been watched by a human as of this
writing** — that's the next concrete action, not more numeric analysis.

Also flagged by the implementing agent, worth keeping in mind: both
correctors were trained with `forward_noiser_loss_mode=teacher_feat` (a
sliced-Wasserstein distributional loss), not a value-level "match GT"
loss — so the `alpha0`/`decay` stepping schedule's framing as a literal
"Euler step toward GT" doesn't strictly apply to either checkpoint tested
here. Nothing is *wrong* because of this, it just caps how literally to
read the step schedule.

Artifacts: `eval/dedrift_smoke_{base,carnctrl,carntxmt,predif}/`,
`logs/dedrift_smoke_{base,carnctrl,carntxmt,predif}.log`,
`sbatch/run_dedrift_smoke.sh`, `sbatch/run_dedrift_predif.sh`.

## 5. Concurrent work by other sessions — do not touch, safe to build on

A separate session (referred to in cross-session chatter as "the GAN-texture
session" / pixdirect / DINO campaign) landed a change making `gt_vs_fake`
never see CARN-noised GT positives: a call-site-only bypass in
`trainer/causal_action_forcing_train.py` and a default-False escape-hatch
flag read in `model/dmd_action_forcing.py` (~`:2317-2380` area). Per an
independent review it obtained: byte-identical for `gt_transition`, a
mutation-control test proves the exact-match check is real, and the CARN/
forward-noiser *implementation* (levels, chaining, weak/strong, reverse,
moment-preservation, the constructor, `model/forward_noiser.py` itself) is
untouched. Their own summary: "gt_transition is safe for the CARN owner to
build on." Two things worth knowing if you touch GAN telemetry: their proof
keys are suffixed `_gt` (`train/fn_gtvf_noise_skipped_gt`, not the bare
name — a mode-suffixing convention at `trainer.py:11986` that has already
caused confusion in several sbatch headers), and `forward_noiser_apply_in_aux`
(default True) is deliberately left out of scope of their bypass — it still
CARNs the aux real-score teacher's target, a DMD path not a disc input.

They also reported a failure mode worth watching if this redesign increases
the discriminator's effective information per step: their `pixdirect_strong`
arm collapsed into a disc-wins regime (`d_loss` 0.010/0.004/0.002/0.010 on
consecutive steps) after raising disc crop coverage. Watch **consecutive**
low `d_loss` readings, not medians, if the in-place commit correction is
ever combined with GAN work.

This project's working tree is **shared and uncommitted across multiple
concurrent sessions** right now (`git status` will show 15+ modified files
and 20+ untracked from at least three sessions). Do not `git stash`,
`checkout`, or `reset` anything — any of those would destroy other sessions'
in-progress work. Use a worktree if you need isolation.

## 4b. The 300-step smoke's base recipe had confounds — a cleaner arm is queued

After the 40-step and 300-step smokes above, the researcher asked to test
against the campaign's actual canonical arm, `sbatch/gansig_of.sbatch` (the
divergence-3-only, mode-neutral-backbone-kick recipe every `gansig_*`/
`carn_*` variant in this sweep descends from). This surfaced two things:

1. **`gansig_of` is itself the campaign's CARN-OFF baseline** — its header
   sets `forward_noiser_enabled=false` explicitly. So "add the commit-dedrift
   flags on top of it" isn't adding an isolated variable, it requires also
   flipping `forward_noiser_enabled=true` and `forward_noiser_loss_mode=
   teacher_feat` as preconditions (the model raises `ValueError` otherwise —
   `model/dmd_action_forcing.py:2774,2779` — no corrector network gets built
   without them). This is applied identically to both arms of the new pair,
   so it doesn't confound the ON/OFF contrast, but it's worth knowing that
   "CARN on" here necessarily means more than one flag flip.
2. **The 300-step `carncommit_long` smoke (§4) has SIX real, non-CARN
   confounds relative to the canonical `gansig_of` recipe**, found by a
   proper effective-value diff (not a naive line diff — both scripts have
   last-wins duplicate keys):
   - `action_critic_aux_enabled` false→true (+ an entire auxiliary loss and
     loaded v14e critic checkpoint)
   - `generator_action_z_guidance_weight` 0.0→0.3
   - `ladd_fake_backbone_grad_scale` 0.1→0.2 (double the mode-neutral kick)
   - `ladd_gt_transition_enabled` true→false (an entire disc branch missing)
   - `ladd_disc_force_clean`, `ladd_gside_checkpoint_recover`

   **Treat the §4 300-step result as exploratory, not clean** — any
   difference it shows could be partly or wholly these six things, not the
   commit-dedrift mechanism. `sbatch/carncommit_of.sbatch` was built as the
   real test: `gansig_of.sbatch` verbatim + one appended override block,
   verified **166/168 keys byte-identical**, every remaining delta is
   `forward_noiser_*`/`reverse_noiser_dedrift_*`. This is the result that
   actually answers the question cleanly.

**Design note worth keeping**: with only one holder available (6145508;
6145507 was claimed by another session), both arms were staged
**sequentially on the same two nodes** rather than in parallel on different
holders — this removes node identity as a variable entirely, stronger than
the "node-matched pair" approach used earlier, at the cost of wall-clock
(sequential, not parallel).

**Timing — this is the real blocker right now, not a bug or oversight**:
6145508 was PENDING at check time with an estimated start around 02:58 (next
day), so the `carncommit_of` result won't land until roughly 07:20. There is
no faster path available — every other holder was occupied, and the ones
freeing at 21:04 also expire at 21:04 (their own 6h `TimeLimit`), so they
can't be reused for a fresh multi-hour run either. The three `carncommit_long`
arms (§4) were left running and will still be reported as the exploratory
secondary result.

## 6. Live state to verify (do this first, this document is a snapshot)

Both background tracks reported complete as of this writing (§4, §4a). If
you're picking this up later, re-check rather than trust this document
indefinitely:

1. `squeue -u as1748.u6ex` — what's actually running/pending now (both
   holders used for the training-side smoke, 6144621/6144623, should be
   idle; nothing was left running by either agent).
2. `git status --short` and `git diff --stat` — does the file list in §4/§4a
   still match, or has something changed since?
3. Do `testing/test_reverse_noiser_dedrift_commit.py` (40 tests) and
   `testing/test_reverse_noiser_dedrift_real_target.py` still pass
   (`OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" PYTHONPATH=. pytest -q
   testing/test_reverse_noiser_dedrift_commit.py testing/test_reverse_noiser_dedrift_real_target.py`)?
4. Has the researcher watched the eval-side rollout videos (§4a) yet, and
   what did they conclude? That's the actual open question on that track.
5. Any pending `SendMessage` replies from other sessions not yet acted on.

## 7. Suggested next steps, in order

1. Confirm §6 items above.
2. **Researcher decision needed**: watch the three eval-side rollout videos
   (§4a — no de-drift / undertrained corrector / better-matched corrector)
   and judge whether the variance-expanding arm looks like correction or
   divergence. This gates whether the eval-side track continues.
3. **Fix the missing wandb counter** on the streaming path (§4, open item 1)
   before running the training-side mechanism any longer — right now its
   only proof of firing is a stdout line, not the counter this project's own
   conventions require.
4. **Run the training-side mechanism longer** (a few hundred steps, not 40)
   to see whether `rel|dz|` converges, keeps growing, or goes quiet — 40
   steps only proved the mechanism is alive, not what it converges to.
5. **Get explicit sign-off on the commit-vs-output asymmetry** (§4, open
   item 2) — the correction currently touches only what's remembered, not
   what's emitted/scored, which is the opposite convention from the existing
   affine correction.
6. Only after the mechanism is shown to help something (not just run
   safely): bring the "retire the other CARN sites" question back to the
   researcher explicitly — it is the stated end goal but is not authorized
   to execute yet.
7. Independently, the `carn_ctrl`/`carn_fwd` GAN-discriminator sweep (§3)
   still needs its 3rd seed before any conclusion — unrelated track, can
   proceed in parallel.
