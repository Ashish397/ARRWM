# RESOLVED TODOs

Items that were proposed, signed off by the researcher, implemented, and **finished**.
Moved here out of `COMMENTS_FOR_USER.md` so that file only holds live questions.
Newest first. Anything still awaiting GPU validation stays in COMMENTS, not here.

---

## R7. Matching cost audited, approximated, and fixed — 2026-08-25 16:10
**Proposed:** matching costs +24.7 GB; audit whether that is intrinsic and approximate it.
**You said:** *"Matching should not cost that much, if it does then it is no longer worth
it and you should approximate it instead."*
**Done:** Full audit in `analysis/smoke_gate/matching_cost_audit.md`. Matching itself
costs **0.11 GB**; the 24.7 GB came from `match_k=8` colliding with a micro-batcher that
splits by *fake* (9 disc rows per group vs the unmatched path's 2, at 1.719 GB/row).
Accounting closed at 24.7 predicted vs 24.74 measured and cross-validated on a
held-out arm. **Verdict: implementation artefact, not the price of matching.**
Three defects found and fixed:
1. `match_pool=100000` collapsed the pool to the 22 existing candidates, making the
   top-k the identity — we were paying the full 24.7 GB for a **uniform random draw**.
2. Two different algorithms ran under one flag name (v6f at `K=8/pool=100000`; five
   other arms silently on defaults `K=4/M=8`, an actual top-8 matcher).
3. My own `max_real=16→6` mitigation had **no-op'd the CARN** — shared real rows get
   FN level 0, whose output is discarded (`levels=[0,0,0,0,0,0,0,0]` in the smoke).
**Applied:** `match_k=1`, `match_pool=3`, keeping `max_real=6` + `lazy_cand_disc=true`.
Predicted peak ~90 → **~65 GB**, with a *real* top-3-of-22 match and the FN firing at
full strength on 3 of 5 rows. All 10 arms resubmitted (6131750–6131759).
**Still open elsewhere:** the optional long-term fix (port the FN application block to
the positional path) remains an unanswered question in COMMENTS.

## R6. "What does matching actually do?" — answered 2026-08-25 16:05
**You asked:** *"Sorry what does this do again?"*
**Done:** Written up as section C.0 of COMMENTS: two distinct mechanisms share the name
"matching" — (i) GAN real/fake **pair matching** (`ladd_gt_transition_match`: search the
GT pool for the pair most similar to each fake, versus taking the same-position pair),
and (ii) **CARN forward-noiser application**, which is physically nested inside (i) in
the code. Conflating them is the root of the inert-CARN bug.

## R5. Section C rewritten — 2026-08-25 16:05
**You said:** *"This is really confusing - can you clean it up please? Use proper
variable names and don't skip steps like this..."*
**Done:** Section C rebuilt with real flag names, the actual nesting shown as code, a
per-arm table of who was affected, the log-census evidence (`[FN-GT-FORMER]` firing 0
times in every unmatched arm), and three separately-numbered reasons my checks missed it.
No steps skipped.

## R4. `carntxcarnold` built, smoked, and submitted — 2026-08-25 16:15
**Proposed:** "original CARN + CARN-TX" combined arm.
**You said:** *"Submit it please, call it carntxcarnold."*
**Done:** Established from code that the two are not separate networks but **two
orthogonal axes of one noiser**, and they compose: `forward_noiser_reverse=true` (TX) +
`forward_noiser_chain_levels=true` (original chained scheme) — a one-flag change from v6,
with the earlier successful "carntx chain" run as precedent. Smoke PASSED with both
mechanisms proven live in the logs (chain banners 8/6/4 firings vs **0** in the chain-off
baseline) at 65 GB, identical to baseline. Renamed to `sbatch/train_carntxcarnold.sbatch`
(+ matching smoke), found and added **both missing memory mitigations** (it would have run
at the default `max_real=12`), and submitted. **Queued as job 6131759.**
*Caveat recorded:* the 65 GB smoke predates matching being switched on.

## R3. Backbone grad-scale audit — 2026-08-25 16:00 — NO PRODUCTION BUG
**Proposed:** `ladd_fake_backbone_grad_scale` must be `1/(n_modes × gan_updates_per_step)`;
arms running 2 pair modes at 0.2 would double-kick the backbone.
**You said:** *"Check it and fix it."*
**Done:** Arithmetic verified against the code (the deferred window spans all pending
closures, so the backbone sums `n_modes × S` backwards; micro-batch groups do **not**
multiply — each group's loss is pre-divided). **All 12 queued production jobs run exactly
one pair mode, so 0.2 is correct and all 12 set it.** Switching to matched does not change
the mode count. No production fix was needed.
Two things it did catch: `sbatch/gansig_of.sbatch` was at 0.25 where 2 modes × 5 updates
requires 0.1 — a **2.5× over-kick**, fixed in both the flag and its log echo before it ran;
and a once-per-run, report-only checker was added that recomputes the correct value live
and logs loudly if config disagrees (**never auto-corrects**, inert when the backbone is
frozen).
*Standing hazard for later:* `carntxrollv5b` is the frozen-backbone control at `scale=1.0`;
flipping `trainable=true` there without changing the scale becomes a 5× kick.

## R2. `adjacent_chunks` left off — 2026-08-25 15:40
**Proposed:** leave `ladd_adjacent_chunks_enabled=false`.
**You said:** *"Good"*
**Done:** No action taken, by design. Rationale on record: its "real" side is the student
itself (no GT anywhere, action-blind), so enabling it would spend a third R1 application
on the weakest objective.

## R1. Stripe/banding fix — DROPPED by your decision — 2026-08-25 15:30
**Proposed:** land the decoded-imprint penalty at weight 1e-3 for the next generation.
**You said:** *"This is not relevant, the GAN should fix it."*
**Done:** Dropped. The patch stays unlanded in its worktree; nothing is enabled anywhere
and no arm carries it. The investigation itself is preserved for the record: the CARN
forward-noiser was proven **causally** to inject the 8 px horizontal corduroy (27× an
equal-energy white-noise control; zero-init noiser = zero stripes; amplitude growing
0.25 → 1.27 as the noiser trains), and every cheap fix was measured dead (spatial
low-pass −9%, full anisotropy matching −13%) because the stripe lives in the same
low-frequency band that carries 77% of the noiser's learned function.
Artifacts: `analysis/vertical_banding/` (incl. `imprint_probe/numbers.json`).
