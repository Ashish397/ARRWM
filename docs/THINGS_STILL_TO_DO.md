# THINGS STILL TO DO — train/infer contract audit leftovers

Written 2026-08-25, immediately after the three ordered fixes landed.
Everything below is **open**. Each item says what it is, why it matters, the
exact `file:line`, and the concrete change required.

Companion documents:

* `docs/THINGS_TO_DO.md` — the researcher's running list. Item 2 of that file
  is the three fixes; they are now DONE (see "What was fixed" below). Item 4's
  first bullet is **factually wrong** and is corrected here as item **A**.
* `analysis/train_infer_contract/sim_rope.py` — the pure-python index replay
  that produced the 21-vs-24 numbers.
* `analysis/freeze_takeoff/` — the seam-affine analysis (`seam_affine_stats.py`,
  `constant_throttle_probe.py`).

---

## What was fixed (for orientation; nothing to do here)

| Fix | Flag | Default | Legacy escape hatch |
|---|---|---|---|
| 1. KV commit = ladder endpoint, not the t=60 flash pred | `flash_dmd_commit_ladder_endpoint` | **True (aligned)** | attr `False`, or `FLASH_DMD_COMMIT_LADDER_ENDPOINT=0` |
| 2. Eval attention span = trained window, not KV buffer | `eval_span_match_training` | **True (aligned)** | `EVAL_SPAN_MATCH_TRAINING=0` |
| 3. CARN seam correction also on emitted `output` / `clean_chunk` | `carn_seam_affine_apply_to_output` | **True (aligned)** | attr `False`, or `CARN_SEAM_AFFINE_APPLY_TO_OUTPUT=0` |

Sites: `pipeline/action_forcing_training.py` (`_carn_seam_correct`,
`_contract_flag`, and both rollout twins), `utils/eval_causal_AR.py:709-777`.

**Live-arm impact, measured against the actual queued sbatch files:** all nine
queued `carntx*` arms set `carn_seam_affine_lambda=0.0`, so **FIX 3 is a
bit-identical no-op for them**; they do set `flash_dmd_enabled=true`, so
**FIX 1 is live** for all nine. `sbatch/train_of_200_8n.sbatch` sets
`carn_seam_affine_lambda=0.5` **and** `flash_dmd_enabled=true`, so both FIX 1
and FIX 3 are live for that one.

---

## A. CORRECTION: `info["finish_denoised_chunk"]` IS set — the bug is elsewhere

**Status: the premise in `docs/THINGS_TO_DO.md` item 4 is wrong; the conclusion
is right, for a different reason.**

* It **is** published: `model/dmd_action_forcing.py:14062-14065`
  (`"finish_denoised_chunk": info_finish_denoised.detach() ...`).
* The viz **does** consume it and even records which source it used:
  `trainer/causal_action_forcing_train.py:15857-15876` (`_viz_mode`, `_fin`,
  `self._rollout_viz_src_used`). There is no silent fall-through on a
  nonexistent key.

**The real defect.** `finish_denoised_chunk` is `pipe._clean_chunk`, and
`_clean_chunk` is written from `cache_pred` **after** the flash reassignment
(`pipeline/action_forcing_training.py:2748-2756`). With `flash_dmd_enabled=true`
`cache_pred` **is** the t=60 flash tensor. So the name is a lie: the "finish
denoised" buffer holds the flash slab whenever flash is on, and the coordinator's
conclusion — *every rollout video we have been judging is the flash t=60 slab* —
**is correct**. Worse, after FIX 1 the videos now show a tensor that is not even
what the KV cache was built from.

**Concrete change required.** In `pipeline/action_forcing_training.py`, alongside
the existing `ladder_endpoint_pred` stash (`:2730` and `:1814`), allocate and
publish a `_ladder_chunk` buffer written with `ladder_endpoint_pred.detach()`,
then in `model/dmd_action_forcing.py:14062` publish **that** as
`finish_denoised_chunk` (keeping the flash slab available under its honest name
`flash_dmd_gan_chunk`, which already exists at `:14061`). Flag-gate it; it
changes what every rollout video shows, so it needs sign-off, which is why it
was not done here.
**Interim mitigation (cheap, no behaviour change):** make
`_rollout_viz_src_used` say `flash` rather than `finish` when
`flash_dmd_enabled` is on — currently it reports `finish` for a flash tensor,
i.e. the telemetry actively misleads.

---

## B. Ten duplicated override keys, in files whose header says there are none

`sbatch/train_carntx_rolldmd3_v6.sbatch:4` states *"Duplicates DELETED, not
shadowed."* There are exactly **10** duplicated override keys in the argument
list, and in 8 of them the second value **contradicts** the first. Later value
wins silently.

| key | lines | first → second |
|---|---|---|
| `max_rolls_per_ride` | 345, 422 | `1` → `4` |
| `streaming_chunk_size` | 358, 425 | `12` → `18` |
| `dmd_42f_clean_match_enabled` | 397, 429 | `false` → `true` |
| `dmd_only_first_chunk_per_ride` | 398, 423 | `true` → `false` |
| `flash_dmd_enabled` | 399, 435 | `false` → **`true`** |
| `gan_enabled` | 402, 433 | `false` → `true` |
| `gan_backbone` | 403, 434 | same value |
| `forward_noiser_enabled` | 413, 441 | `false` → `true` |
| `carn_recurse` | 414, 444 | same value |
| `forward_noiser_apply_gt_former` | 415, 450 | `false` → `true` |

The same 10 appear at a +5 line offset in
`sbatch/train_carntx_rolldmd3_v6b.sbatch` and in the other `v5b/v6alt/v6f/v6s0`
clones (they are copies).

**Why it matters.** `flash_dmd_enabled` is the switch that decides whether FIX 1
does anything at all. A reader of the header believes it is off. It is on.

**Concrete change required.** Delete the *first* occurrence of each contradicting
key (the later value is the one every run has actually used, so deleting the
earlier one is behaviour-preserving), delete one of each identical pair, and add
a guard to the launcher that greps its own override list for duplicate keys and
`exit 1`s. Do **not** "fix" it by changing values — that would alter running
recipes.

---

## C. `utils/play_world_model.py` implements a THIRD, different contract

* Attention span: `utils/play_world_model.py:222-225` sets
  `local_attn_size_frames = kv_cache_chunks (default 8) * num_frame_per_block (3)
  = 24` frames and `kv_cache_tokens = 24 * frame_seq_length`, then
  `:477-481` passes `max_tokens=self.kv_cache_tokens` — i.e. a **24-frame span**,
  exactly the bug FIX 2 just removed from `eval_causal_AR.py`. Note it also
  *reads* the config's trained window into `self.cfg_local_attn`
  (`:222`) and then **ignores it**.
* CARN: there is **no** seam-affine, temperature or drift correction anywhere in
  the file (grep for `carn` returns nothing).

So we now have three contracts: training (21-frame span, CARN on commit **and**
output), `eval_causal_AR` (21-frame span, CARN on `pred_x0`), and
`play_world_model` (24-frame span, no CARN).

**Concrete change required.** Port both: (1) at `:222-225`, derive the span from
`self.cfg_local_attn` while keeping `kv_cache_chunks * npb` for the *buffer*
allocation, mirroring `eval_causal_AR.py:709-777` (reuse
`_resolve_trained_attn_window` — lift it into a shared helper rather than
copy-pasting a third time); (2) add the `_apply_carn_seam_affine` call on
`pred_x0` before emission and cache refresh, mirroring
`utils/eval_causal_AR.py:1422-1429`, gated on the same
`--carn_seam_affine_lambda` argument.

---

## D. M3 — GT-anchor depth: training prefills 12 real frames, inference 9

**Judgement: this is a REAL divergence, not a deliberate one. It should be
closed, but it is a training-recipe change and needs sign-off.**

* Training: `model/dmd_action_forcing.py:12213-12224` seed-prefills
  `cf = dmd_context_clean_frames = 9` GT frames, then
  `:12251-12274` (`dmd_42f_gt_anchor`, set to `true` in every queued arm)
  prefills a **further `npb = 3` GT frames** through the same
  `_seed_prefill_chunk` path. First supervised chunk therefore sees **12 real
  frames** of context.
* Inference: `utils/eval_causal_AR.py` prefills `initial_latents`
  (`initial_frames`, 9 under `run_eval60`) and nothing else.
* `analysis/train_infer_contract/sim_rope.py:36` encodes exactly this
  (`prefill=12` train vs `prefill=9` infer).

**Why it matters.** The GT anchor's own comment (`:12246-12250`) justifies it as
*"matching the 7-chunk inference eval"* — but it matches the eval's *cleanliness*
while adding a frame of *depth* the eval does not have. Every trained chunk gets
one extra chunk of real history than it will ever get at serve.

**Concrete change required.** Either (a) make eval prefill `cf + npb` when the
checkpoint was trained with `dmd_42f_gt_anchor=true` — record the flag in the
checkpoint and read it in `eval_causal_AR.py` around `:697`; or (b) drop the
anchor's frames from the *attendable* context in training. (a) is much cheaper
and does not touch the recipe. Do not simply set `dmd_42f_gt_anchor=false`: the
comment at `:12240-12250` documents a real generate-vs-score mismatch it fixes.

---

## E. Seam telemetry — PARTLY DONE, one piece left

**Done in this pass** (`pipeline/action_forcing_training.py:1119-1161`,
`_carn_seam_record`, flag `carn_seam_telemetry`, **default ON**, emitted only
when the affine is actually live): `carn_seam_gain` (channel-mean S/s),
`carn_seam_mu_shift` (‖M−m‖₂), `carn_commit_std` (S), `carn_pred_std` (s), plus
`carn_seam_blocks`. They ride the existing `_last_extension_metrics` channel that
`model/dmd_action_forcing.py:14025` already folds into the trainer's info dict,
so no consumer change was needed.

**Prediction to check them against.** The recursion is
`S = λ·σ_seed + (1−λ)·s` then `s' = k·S`. At the fixed point `S = s/k`, so the
**steady-state gain is exactly `1/k`, independent of λ** — `≈1.090` for the
4-rung sampler (`k = 0.917`), `≈1.013` for the 48-step sampler (`k = 0.987`).
A measured gain near `2.0` means the student's own σ has collapsed and the
corrector is doing all the work. (The `~1.045` figure in `THINGS_TO_DO.md`
corresponds to `k ≈ 0.957`; the telemetry now settles this by measurement.)

**Still to do.** Nothing logs these per-*roll-depth*. `carn_seam_gain` is
averaged over the blocks of one call, so the geometric convergence across roll
depth — the thing `analysis/freeze_takeoff/seam_affine_stats.py` Q2 wanted and
could only proxy in decoded pixels — is still not directly observable. Concrete
change: key the accumulator by roll index (the trainer already tracks
`self._chunks_in_current_ride`, `trainer/causal_action_forcing_train.py:15891`)
and emit `carn_seam_gain_roll{j}`.

---

## F. Stale `_carn_seam_target` on the non-streaming path

`pipe._carn_seam_target` is published **only** by the streaming path
(`model/dmd_action_forcing.py:13776-13782`) and is **never cleared**. The other
affine site — `inference_with_trajectory`, now
`pipeline/action_forcing_training.py:1892-1894` via `_carn_seam_correct` — has no
publisher of its own, so any call through it re-anchors to whichever ride's seed
statistics were published last.

Not on the current hot path (the queued arms all go through
`generate_chunk_with_cache`), but live and silent.

**Concrete change required.** Clear it at the top of
`inference_with_trajectory` (`:1219`, next to
`self._last_extension_metrics = {}`) unless the caller published one for *this*
rollout — e.g. stamp `self._carn_seam_target_seq` with the sequence id at publish
time and have `_carn_seam_correct` refuse (or warn-once and skip) a target whose
stamp does not match the current rollout.

---

## G. Inference CARN still mean-matches; training no longer does

`utils/eval_causal_AR.py:169-192` (`_apply_carn_seam_affine`) blends **both**
mean and std toward the seed, unconditionally. Training's default flipped to
std-only (`carn_seam_affine_match_mean` defaults **False**,
`pipeline/action_forcing_training.py:1076-1102`). That is a **new** train/infer
divergence, opened by the std-only change and not closed by FIX 3 — FIX 3 was
explicitly scoped to respect `match_mean=False` and not resurrect mean matching.

**Concrete change required.** Add a `match_mean: bool = False` parameter to
`_apply_carn_seam_affine` (`utils/eval_causal_AR.py:169-192`) and a matching
`--carn_seam_affine_match_mean` CLI flag next to
`--carn_seam_affine_lambda` (`:1719`), defaulting False so eval matches
training's new default; wire it at the call site `:1422-1429`.

Related, smaller: eval has **no** counterpart for `carn_seam_temp` or
`carn_seam_drift_lambda` at all. Both are off by default in every arm today, so
this is dormant — but the moment either is switched on, training and inference
diverge again with nothing to catch it. Either port them to eval or make the
trainer refuse to start with them non-default.

---

## H. Consequences of FIX 3 that were deliberately left in place

1. **`flash_dmd_gan_output` is NOT seam-corrected.** FIX 1's brief was explicit
   that the GAN/FN slab keeps flowing "exactly as today", so it was not touched
   (`pipeline/action_forcing_training.py:2731-2740`). Before this change
   `_clean_chunk` and `_flash_dmd_gan_output` were numerically identical (one is
   the detached other); **they now differ whenever the seam affine is live**.
   Nothing consumes both and compares them today, but the invariant is gone.
   Decide deliberately: either correct the GAN slab too (changes the
   discriminator's fake) or document the asymmetry at both write sites.
2. **`clean_chunk_grad` (A23) is NOT seam-corrected**
   (`pipeline/action_forcing_training.py:2788-2816`). That buffer exists
   precisely to be "the tensor `utils/eval_causal_AR.py` commits and renders" —
   and inference renders the *corrected* tensor. So with the affine live, A23's
   inference-parity fake is now one correction short of parity. Concrete change:
   apply `self._carn_seam_correct` to `finish_grad_pred` before the
   `clean_chunk_grad[...] = ` write, under the same
   `carn_seam_affine_apply_to_output` flag. Left undone because it changes the
   pixel critic's fake, i.e. a GAN-recipe change.
3. **Third commit site has no CARN at all.** `_seed_prefill_chunk`'s
   `seed_prefill_mode="estimate"` branch
   (`pipeline/action_forcing_training.py:950-977`) commits without any seam
   correction. Dormant — every queued arm uses `seed_prefill_mode=real`, which
   returns before that branch (`:903-921`) — but it is the one remaining commit
   path that would silently disagree.

---

## I. The three new flags are not reachable from config

`flash_dmd_commit_ladder_endpoint`, `carn_seam_affine_apply_to_output` and
`carn_seam_telemetry` are read via `_contract_flag`
(`pipeline/action_forcing_training.py:1017-1025`): **attribute → env var →
aligned default**. Nothing in `model/dmd_action_forcing.py` or
`trainer/causal_action_forcing_train.py` sets the attributes, because those files
were off-limits during this task. The env-var fallback exists so the legacy
behaviour is still reachable without an edit.

**Concrete change required.** Three lines next to the existing CARN plumbing at
`model/dmd_action_forcing.py:13772-13775`:

```python
pipe.flash_dmd_commit_ladder_endpoint = bool(getattr(
    self, "flash_dmd_commit_ladder_endpoint", True))
pipe.carn_seam_affine_apply_to_output = bool(getattr(
    self, "carn_seam_affine_apply_to_output", True))
pipe.carn_seam_telemetry = bool(getattr(self, "carn_seam_telemetry", True))
```

plus the matching `self.<name> = ...` reads from `args` near
`model/dmd_action_forcing.py:1392-1408`. Keep every default `True`.

---

## J. FIX 2 changes previously-scored eval numbers

Every `eval_causal_AR` run before 2026-08-25 attended **24** frames; from now on
it attends the trained window (**21** unless the checkpoint says otherwise). The
effective span and its provenance are printed once at eval start
(`utils/eval_causal_AR.py:761-773`, the `[AR][span]` line).

**Concrete change required.** Re-score, or at minimum re-label, any eval60 number
carried forward from before this date; and check whether
`utils/causal_chain_rollout.py` (the teacher, `local_attn_chunks=7`) and
`pipeline/ode_rollout.py` agree with the new resolver's answer for the
checkpoints in play — the resolver now reads `model.local_attn_size`, so a
checkpoint whose config says something other than 21 will now be served at *its*
window rather than at the old hardcoded 21. That is the intended behaviour, but
it is a silent change for any such checkpoint, so verify the printed
`[AR][span]` line on the first run of each arm.

---

## K. Test-environment note (not a code defect)

`testing/test_pixgan_trainer_supply.py` is OOM-killed (SIGKILL 137) partway
through when the whole 114-test file runs in one process on the shared node —
non-deterministically (it died at test 36, then 5, then 24 on successive runs,
with ~140 GB of the node's 237 GB already held by other processes). Every subset
passes when run alone, and the killed tests do not touch
`generate_chunk_with_cache` / `inference_with_trajectory`. Worth splitting the
file or adding a per-test VAE teardown so this suite is usable as a gate;
`testing/test_a23_finish_grad.py` (45 tests) runs clean in 4 s.

---

## 2026-08-25 19:10 — WHY THE dino/cnx RUNS "ONLY WENT TO 16 AND LOOKED THE SAME"

Researcher observation: `gantune_w2dino` and `gantune_w2cnx` produced videos only up to
step 16 and looked no different from each other. Diagnosed — **three separate causes, none
of which was the backbone**:

1. **They did NOT stop at 16.** Both logged `Training complete` at **step 30** (their
   MAXSTEPS, sized to the ~23 min of holder wall left). The *videos* stop at 16 because the
   sample interval is 15, so a 30-step run renders at steps 1 and 16 only.
2. **Every video compared was rendered BEFORE the GAN existed.** `gan_disc_start_step=20`.
   At step 16 the discriminator has not engaged, the adversarial term is zero, and the
   surrogate teacher has had ~2 updates. There was nothing for the two arms to differ by —
   identical output at step 16 is the *correct* result, not a null finding.
3. **Even at full length, the surrogate could not have moved the picture at the weight it
   was running.** Measured on the SAM2 run: `surrogate_g_weighted` grew 6.4e-05 → 7.0e-04
   while the LADD adversarial term is O(1) — i.e. the surrogate was **~0.1%** of the GAN
   signal. No backbone comparison is meaningful at that amplitude.

### Actions taken
- **`pix_gan_weight` 0.25 → 2.0** on all three backbone arms (`w2cnx`, `w2dino`, `w2sam`),
  so the surrogate has authority to change the output rather than being a rounding error.
  Chosen as 8× rather than 100× deliberately: the raw term also grows as the critic trains
  (`updates_per_step` is now 8, not 1), so the two effects compound and a 100× weight on a
  maturing critic risks the same runaway we measured at `gan_loss_weight=4.0`.
- **Job-name bug fixed.** All three inherited `--job-name=gt-w2sam` from the file they were
  copied from, so `squeue` showed three identical `gt-w2sam` rows — the run tags (`DARM`)
  were correct, but the queue was unreadable and the wrong job could easily have been
  cancelled. Now `gt-w2cnx` / `gt-w2dino` / `gt-w2sam`. Resubmitted as **6133669 / 6133690 /
  6133691**, all 200 steps.

### Standing lesson (add to the silent-failure taxonomy)
**A comparison rendered before its mechanism engages is not a null result, it is a
non-measurement.** Any short run must be checked against the engagement schedule before its
output is interpreted: `gan_disc_start_step=20`, `gan_warmup_steps=25`,
`z_guidance_warmup_steps=50`, sample interval 15. A run shorter than ~50 steps cannot show a
GAN or critic effect *at all*, and its videos will look identical no matter what is changed.

### Still open on this thread
- If `surrogate_g_weighted` still sits <1% of the LADD term at weight 2.0 by step ~100, the
  surrogate route is amplitude-limited and needs either a much larger weight or a different
  coupling — decide from the run, not by guessing again.
- The three-way backbone comparison is only valid because all three now share
  `updates_per_step=8`, `lr=5e-4`, `pix_gan_weight=2.0`. Do not compare them against the
  earlier SAM2 run (updates=1, weight 0.25) — different experiment.

---

# RUNS STILL TO DO — queued 2026-08-25 ~23:15

## Researcher observation driving these
**`dmd10k_gansig_ofclean` looks excellent at step 151.** That arm is:
divergence 3 (`ladd_fake_sample_source=dmd`) + **clean discriminator**
(`ladd_disc_force_clean=true`, i.e. `disc_t=0`) + the second pair mode
(`ladd_gt_vs_fake_enabled=true`), on the non-CARN w1 GAN base.

Why the clean disc matters, proven earlier today: the flow-matching scheduler gives
`alpha_t = 1 - sigma`, so at the band's `disc_t=1000` rung the student's sample is
multiplied by **zero** and the generator gets *no* adversarial gradient — measured on
3 of 8 samples in plain `gansig_of`. `force_clean` pins `alpha_t = 1.0` on every step:
maximum gradient AND maximum texture visibility. `ofclean` is that fix in action.

**Both arms below are the researcher's request: keep what ofclean does well, and add
autoregressive-drift handling (old chained CARN) plus an ONLINE action critic.**

## 1. `sbatch/gansig_ofcleancarn.sbatch` — BUILT, NOT LAUNCHED
`gansig_ofclean` + old-fashioned drift CARN + online action critic.
* CARN: `forward_noiser_enabled=true`, **`forward_noiser_chain_levels=true`** (the
  original chained/level-conditioned scheme), `forward_noiser_reverse=true` (TX),
  `apply_gt_former=true`, `apply_gt_level=1`, `carn_recurse=false`.
* **`ladd_gt_transition_match=true`** with `match_k=1`, `pool=3`, `max_real=6`,
  `lazy_cand_disc=true`. **Non-negotiable** — the noiser's ONLY application site sits
  inside the matched block, so without this the CARN trains and never touches the data
  (this morning's biggest bug; `[FN-GT-FORMER]` fired 0 times in every unmatched arm).
* Online critic: `action_critic_aux_enabled=true`, `freeze` unset (default False =
  online, verified at `trainer:1250`), `action_teacher_mode=all`,
  `critic_updates_per_step=2`, `z_loss_weight=0.5`,
  `generator_action_z_guidance_weight=0.3`, `z_guidance_warmup_steps=50`,
  checkpoint `logs/v14e_pca8_raw/causal_lora_step0005000.pt`.
* Expected peak ~65 GB (matched + mitigations), not ~90.

## 2. `sbatch/train_carntx_6tchain_on.sbatch` — BUILT, NOT LAUNCHED
"CARN-TX 6 trinity, chained + online critic". `carntx6t` (v6f base + w1 GAN defaults +
action critic) with two changes:
* **`forward_noiser_chain_levels=true`** — adds the old-fashioned drift CARN on top of
  the TX direction already present (`reverse=true`), i.e. both axes, the combination
  proven in `carntxcarnold`.
* **Critic switched frozen → ONLINE**: `action_critic_freeze=false`,
  `action_teacher_mode=all`, `critic_updates_per_step=2`, `z_loss_weight=0.5`.
* Differs from `gansig_ofcleancarn` in base: this one is the CARN/v6f production
  lineage (8-node arm script, flash disc path); that one is the gansig tuning lineage
  with the clean disc and dmd band. Running both separates "CARN production recipe +
  online critic" from "ofclean recipe + CARN".

## Caveats to carry when reading either
1. **Online vs frozen is NOT a single-variable swap.** The online recipe also sets
   `action_teacher_mode=all` (frozen uses `off`) and an explicit `z_loss_weight`. That
   is how the original pair was designed, so it is kept — but a difference cannot be
   attributed to "the critic learns" alone.
2. **Chained CARN makes noiser training sparse** — measured `tf_pairs` 5.0 → 2.0 → 0.0
   across steps 0/25/50 in the carnold smoke, because pairs come from the ride's setup
   window only. `fn_frontier_pairs=true` is the knob if that turns out to matter.
3. `gansig_ofcleancarn` inherits the dmd band, so its `gan_dmd_grad_ratio` is **not
   comparable** to flash-path arms (the DMD denominator co-varies with the rung). Judge
   it on `gan_grad_norm` or stratified by rung.

## Launch priority when a holder frees
1. `gansig_ofcleancarn` — most complete arm; direct answer to the researcher's request.
2. `train_carntx_6tchain_on` — the production-lineage counterpart.
3. `gansig_wide` — never launched all day; lowest value (needs `all_pairs`, which
   disables the memory guards, and corrects the action origin on the real side only).

---

# ABLATING THE TRANSITION GAN — is it feeding us or hurting us? (2026-08-26)

The question: `gt_transition` is the pair mode every pre-`gansig` arm ran, and the whole
`w2-*` series ran it EXCLUSIVELY. Is it contributing, or is it dead weight that also drags
the CARN along with it?

## What we know already

**1. It is the wrong question for TEXTURE, and that is now measured.**
`gt_transition`'s "real" side is a 2-chunk **transition** — it asks *"does this clip join to
itself plausibly across a seam?"*, a question about temporal continuity. `gt_vs_fake` asks
*"does this look like real footage?"*. The entire w2-* series (w2, w2pix, w2sam, w2wav2,
w2gram, w2tclean) ran transition-ONLY, so **not one of them contained a disc that was ever
asked the appearance question** — the series was underdefined by construction. The
researcher's observation that `ofclean` is the only arm with semi-realistic texture is
consistent: it is the only one of that set with `gt_vs_fake` enabled.

**2. It is load-bearing for the CARN — structurally, not by design intent.**
The CARN's apply site sits behind TWO guards: `_match_active`, and `chunks_per_pair == 2`
which **only `gt_transition` sets** (`:12412`; every other mode leaves it 1). Log census over
8,707 files: `[FN-GT-FORMER]` fired **0 times in all 24 arms** with `match=false`, and 2+
times only where true. **21 arms trained and checkpointed a noiser that never touched a
single tensor.** That coupling is now broken by `forward_noiser_apply_decoupled`, but the
decoupled path is a WEAKER variant (see D1-D3 in `THINGS_TO_DO.md`).

**3. Removing it removes an anchor, and the sign may flip.**
Coupled application noises the FORMER half of a transition and leaves the LATTER clean — a
degraded→clean anchor the disc can key on. At `cpp==1` that is *structurally impossible*:
one chunk carries no transition, so there is no quality gradient. What replaces it is
real-side instance noise (real class = ⅓ clean, ⅔ level-1 CARN'd). The shipped generator
term is RpGAN `E[softplus(d_real − d_fake)]` with `d_real` detached
(`model/r3gan.py:238-246`); its gradient w.r.t. the fake logit is `−σ(d_real − d_fake)`, so
**lowering `d_real` monotonically SHRINKS the adversarial gradient** — i.e. noising the real
side teaches the disc to *tolerate* drift, not punish it. Site B (`[CARN-MATCH-POOL]`)
shields against exactly this with a gen-side `carn=True`-on-D-update-only rule; the
decoupled path does not.

## The ablation set (minimum 4 arms, 2 pairs)

All on the `gansig_gtvf_*` base (clean disc, dmd source, frozen critic). **Pair the arms on
the same nodes and run ≥3 seeds** — `d_loss` has a 17% noise floor and the ratio 50%.

| # | arm | `gt_transition_enabled` | `gt_vs_fake_enabled` | answers |
|---|---|---|---|---|
| A1 | `gtvf_only` | **false** | true | already run (`gansig_gtvf_dmd`, 200 steps) |
| A2 | `both_modes` | **true** | true | already run (`gansig_ofclean`, 200 steps) |
| A3 | `xn_only` | true | **false** | **MISSING** — the transition-only control at matched settings |
| A4 | `neither` | false | false | **MISSING** — GAN-off floor; bounds how much either mode buys |

A1 vs A2 answers *"does transition ADD anything once gt_vs_fake is present?"* — the
researcher's live hypothesis. A3 vs A1 answers *"is transition WORSE than gt_vs_fake?"*.
A4 bounds both. **A3 and A4 do not exist yet and are the gap.**

Caveat on A1 vs A2 as currently run: they also differ in the action critic, backbone scale
(0.2 vs 0.1 — mode-neutral in both, so not a treatment) and the code-fix telemetry flags.
Only the critic is a real second treatment; strip it for a clean read.

## Judge on texture, not disc health

Per `analysis/sharpness/TEXTURE_REVIEW.md`, the certified composite ranks `of` BEST and
`ofclean` WORST — the exact inverse of the researcher's eye — because it is blind to the
phase-locked fold. Use instead, on **textured crops** (tree crown, road; NOT sky), each arm
paired against its own `clean_x_real`:
* `fold2d` P=16 **dotfrac** and residual peak-to-peak (GT: 0.13 / 3.30)
* mod-8 row fold A8y (GT: 1.42)
* two-sided distance-to-dataset — overshoot is as bad as undershoot
* reject any arm whose fold improves while HF drops BELOW the dataset (the `of` degeneracy)

## Standing warning

`of`'s "better texture" is the GAN being switched OFF 25% of the time: `disc_t` is drawn
uniformly from {1000, 250, 100, 50} and at 1000 `alpha_t = 1−sigma = 0` **exactly**, so the
adversarial gradient is multiplied by zero and the disc trains on two independent N(0,1)
draws. `ofclean` is `of` with that one line changed and reaches A8y **1.75 vs 4.26** at FULL
gradient. Any future arm that "improves texture" should be checked against this failure mode
before being believed.
