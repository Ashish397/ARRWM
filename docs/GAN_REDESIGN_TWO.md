# GAN REDESIGN TWO — decouple the critic from the DMD teacher

**Researcher's diagnosis (2026-08-23, binding):** the failure mode of every
latent GAN arm traces to projecting the discriminator onto the **DMD
teacher's own weights**. The critic and DMD share one perceptual basis, so
the critic is structurally blind to exactly the mistakes DMD makes — they
reinforce each other instead of correcting each other. **The fix is
decoupling, and only decoupling.** One of the three backbones below is the
solution; the DMD projector is not.

Supporting evidence already on record:
- The coupling is literal, not approximate: the disc backbone is constructed
  from `self.model.real_score` — the SAME frozen v14e weights DMD scores
  with (verified, GAN_REDESIGN Point 6).
- Every DMD-projected critic arm has been inert or harmful (wave01/wave_ts/
  strict03 pathological; raw_t0's 0.01 near-inert; poolrich grainy-but-soft;
  ganv2f fixes wave ≈ no texture improvement at 200 steps).
- A7 telemetry on DMD-projected critics: `gan_dmd_grad_ratio` 0.05–1.2%,
  cos ≈ −0.02..−0.10 — small and near-irrelevant. A *blind* critic cannot
  show up in cos; the blindness prediction is tested by the arms below.

Standing rules carry over: TF head only; stat anchor OFF; flag-gated
default-off + adversarial review; four holders; multi-seed battery protocol
(`ODE_FLOW_SEED`, ≥3 seeds; hv-anisotropy/entropy never read single-seed).

---

## The three decoupling options (researcher's preference order)

### Option A — Wan 2.1-T2V-14B prefix backbone (first choice)
Different weights, same latent space, pretrained. **State: code LANDED**
(prefix loader `model/wan14b_prefix.py`, raw-model projector path in
`model/ladd_disc.py`, trainer branch `ladd_disc_backbone_model_name`, hard
tap assert; own test + scaling probe). **OOM root cause FALSIFIED
by WP-14B (2026-08-23 evening) using the smoke's own log:** the failing
366 MiB alloc is `rope_apply` at S=9360 — the ACTUAL chunk token count
(the padded 18,721 theory predicts 731 MiB). Real cause: the GENERATOR-side
guidance recompute ran un-micro-batched in the smoke (D side at groups=4,
G side at 1). **RESOLVED (WP-14B, measured): the knob WAS 4 and honored — 4 is simply
insufficient at 6-frame transition-chunk geometry.** The smoke sat in the
measured 79.8 GiB cell (16 rows, G groups=4) before the resident models and
died at 92 GiB. Fix (byte-identical, and FASTER at 16 rows):
`ladd_gen_guidance_micro_batch_groups=16 ladd_gt_transition_match_max_real=8`
→ expected disc-side peak ~25 GiB. Re-smoke queued (holder 6109489).
COST TO BOOK for the 300-step arm: ~40 s/step disc time at 5 updates
(~3.5 h/300 steps) at max_real=16; max_real=8 roughly halves it.
Resolved-value echo lands on the trainer's LADD build line when WP-PIXGAN
frees the file. Then 300-step arm.

### Option B — UNTRAINED / stock Wan 1.3B backbone (OOM fallback)
Load the base `Wan2.1-T2V-1.3B` WITHOUT the v14e LoRA as the disc backbone:
different function (no driving fine-tune, no action patches), same dims
(1536 — zero CCM/head changes), tiny memory (prefix ≈1.4 GB). **Likely
nearly free via the SAME `ladd_disc_backbone_model_name` machinery pointed
at the 1.3B checkpoint dir** — the prefix loader reads config.json, and
dim/patch bookkeeping is already dynamic. **CONFIRMED no-code by WP-14B
(stock 1.3B measured through the Option-A machinery on GPU; 0.83 GiB
prefix).** Weakest decoupling but cheapest. **RESUME TRAP (serious): at dim
1536 the disc's CCM/head shapes MATCH the v14e-projected arms, so the
resume shape-guard does NOT fire — a resumed disc silently carries the
coupled basis this arm exists to remove. `decouple13b` must launch with a
FRESH disc and verify that in the startup log** (invisible with the 14B,
where 1536→5120 forces the mismatch). Smoke pre-queued (holder 6109490).

### Option C — pixel critic + surrogate gradients (the full escape)
**ON HOLD (researcher decisions, 2026-08-23 night) — do NOT launch arm 1
until:** (1) GroupNorm dropped from the critic (measured: with it ON, one
patch logit's gradient spans the whole 176×240 image — §4's "local ~70 px"
claim is FALSE, and a global critic cannot test the LOCAL-feedback
hypothesis); (2) R1 γ calibrated BY MEASUREMENT on the final architecture
(§5.3's "scale-free in P" is FALSE — measured α=−0.915 vs claimed 0; single
draws have 126% rel-sd so γ reads off a ~50-step running mean, then
×⅓/×1/×3); (3) §3.7's effective-sample arithmetic re-derived for the true
receptive field before fixing pix_crops/frames. `pix_gan_weight` never
inherits LADD-family values. Adversarial review: 7 defects (2 HIGH), 5
fixed (suite 265/41); the two spec-level ones are the above. Consequence
already applied: pixtex300 flipped to `gan_pixel_texture_enabled=false` so
it stays a pure 300-step control regardless of when T3-C lands relative to
its holder start. Options A/B unaffected.
Leaves the latent-teacher family entirely: from-scratch pixel PatchGAN on
decode(crop) + the latent surrogate critic serving its gradients cheaply.
**State: T1 grad path green (20/20); T2 critic green (71 tests, P=660);
T3-A trainer skeleton landed (37/37, byte-identical off); T3-B (D-loop +
real supply) and T3-C (G-term + telemetry + `pix_*` config) in flight;
surrogate module + tests green, wiring deferred until B1 lands.** A23
fake-source gate overturned — fake source is an experimental variable.
**SWITCH-READY DIRECTIVE (researcher, 2026-08-24 morning):** the surrogate
consumption path (WP_SURROGATE.md §4.2-4.4: distillation loop + gated
generator consumption + fail-loud save/resume) is to be WIRED NOW, dark,
default-off — so direct-gradient → surrogate-gradient is a config flip
when arm 2+ needs coverage scaling, not a build window. Tasked to the live
session (MAIN does it under researcher authority if unclaimed). Running the
surrogate arm remains gated as before (KV verdict + trained teacher +
cadence-asserted smoke). Clarified for the record: the surrogate is
teacher-agnostic by contract and its DESIGNED teacher is the pixgan
(SAM2/DINO are heritage only); the coverage gain lands on the
generator-side gradient path (per-step cost stops scaling with crops).
Launch protocol agreed: weight-free probe (~50 steps, unweighted
`pix_gan_grad_ratio` telemetry — landing in T3-C) → `pix_gan_weight =
0.10 / r_probe` → ×⅓/×1/×3 bracket. No gradient cap exists (A4 deleted):
the probe is mandatory before any weighted launch.

---

## Execution plan (rev 2 — after WP-PIXGAN's methodology review; adopted)

**Order: B → C → A.** C is not sequenced behind A (A is blocked on an OOM fix
of unknown duration; C tests the thesis at full strength). B's null, if it
comes, is ambiguous ("not enough decoupling") — stated up front.

0. **PREREQUISITE FOR EVERY DECOUPLED ARM — the weight-free probe.** The
   historical 0.05–1.2% gradient share is ~100× under the 5–20% target band,
   so "blind because coupled" and "inert because underweighted" predict the
   same null. Every backbone (A, B, C) therefore runs ~50 probe steps
   reading its UNWEIGHTED grad ratio, then weight = 0.10/r_probe, then a
   ×⅓/×1/×3 bracket. No arm launches weighted without its probe (no
   gradient cap exists since A4's deletion). NEW OPUS ITEM: the unweighted-
   ratio telemetry exists only in T3-C's pixel path — generalise it to the
   LADD path so A/B can run the probe.
1. **Option-B smoke** (queued, holder 6109490) → probe → 300-step
   `decouple13b`.
2. **Option-C** when T3-B/C land + review: probe → 300-step `pixtex_adv`.
   First-arm fake = `flash_dmd_gan_x0` (`pix_finish_grad_enabled=false`);
   `pix_finish_grad_enabled=true` is the FIRST follow-up variable (the
   cleaner-than-deployed-tensor mechanism is still plausible even though
   A23's statistical claim fell).
3. **Option-A re-smoke** when the OOM fixes land → probe → 300-step
   `decouple14b`.
4. All arms: noanchor base, 300 steps, chained 60 s eval, multi-seed
   battery, judged vs `pixtex300` (critic idle until the pixel loss lands →
   it doubles as the 300-step base control; a separate noanchor300 was
   redundant and is cancelled).
5. **Decoupling readout, revised hierarchy — (iii) GATES (i) and (ii):**
   - (iii) positive control: separates C_late corruption from decode(GT).
     Exists ONLY for the pixel critic today. NEW OPUS ITEM (else A/B run
     control-less): latent-side positive control. CONCRETE SPEC (WP-PIXGAN,
     agreed): reuse verbatim from `model/pixel_texture_disc.py` —
     `c1_structured_hf` (already latent-domain, DDP-safe under explicit
     generator), `sweep_c1_amplitude`/`c1_calibration_verdict`,
     `roc_auc_fast` + `patch_logit_separation` (domain-agnostic). MEASURED
     2026-08-23 evening: `positive_control_readout` accepts any scorer
     callable and the LADD path's default token-logit shape [N,1,h,w]
     WORKS TODAY with zero changes — for A/B the control is a scorer
     callable + a latent-domain amplitude calibration, i.e. wire-up, not a
     build. (LANDED 2026-08-23 late: both shapes accepted, [N,1,h,w] path
     verified byte-identical, suite 79+32 green. Wiring gotchas, from tests
     failing for the right reasons: a patch-map scorer on a uniform shift
     gives AUC ≈0.95, not 1.0 — pin the gap, bound the AUC; and
     `rank_ok=0` is the calibration alarm working, not a critic failure —
     recalibrate the amplitude, don't tune the test away.) Rules: (1) reduce
     via `disc_holdout_probe._reduce_scores` — never a fourth reduction;
     (2) calibrate the C1 amplitude ONCE, offline, IN THE LATENT DOMAIN —
     never reuse a pixel-calibrated amplitude; (3) primary check is the
     ordering AUC(GT vs corrupted) > AUC(GT vs student) — if it inverts,
     the calibration is wrong, not the critic. CAVEAT: latent-control AUCs
     and pixel-control AUCs answer different questions — compare each arm
     against its own control's ordering, never across arms.
   - (i) d_loss separation and (ii) battery movement are read ONLY when
     (iii) passes; ln 2 with no control is UNDECIDABLE (undertrained vs
     wrong design), per §8.1 — do not record it as a decoupling verdict.
   - If the researcher opts to run A/B without a control, they are judged
     on battery movement + video alone, and (i) is declared non-informative
     in advance.
   - cos(g_GAN, g_DMD): reported, never a criterion.

**Booked caveats:** B is the weakest decouple (same family/scale, merely
un-fine-tuned). C retains ONE deliberate shared component — both sides
decode through the teacher's frozen VAE (same-decoder cancels the transfer
function; bounded by the Case-1 verdict) — "full escape" ≠ zero shared
components. B5's DINOv2/SD escalation loses the zero-shared-basis property
of the from-scratch critic and is not a pure upgrade.

## RETROACTIVE CHECK RESULT (main agent, 2026-08-23 ~18:45, wandb)

WP-14B's weight-division identity (`r_unweighted = gan_dmd_grad_ratio /
r3gan_g_weight`; code half verified by WP-PIXGAN; zero-DMD-denominator rows
excluded per their trap) computed on today's ganv2f arms:

| arm | weighted ratio | weight | UNWEIGHTED share |
|---|---|---|---|
| notaps | 0.00375 | 0.03 | **12.5%** |
| nopatch | 0.00460 | 0.03 | **15.3%** |
| all | 0.055 med | 0.03 | **14–355%** (n=2, wide) |

**Verdict: the coupled critics were WEIGHT-STARVED, not blind** — their
intrinsic share sits in/above the 5–20% band; w=0.03 crushed it ~30×.
Consequences: (1) the probe ARMS are unnecessary — a short run at any
nonzero weight plus this division suffices (rule adopted); (2) the
historical "GAN did nothing" record is explained by weight and says NOTHING
about coupling; the blindness thesis is now UNSUPPORTED by the ratio
evidence but NOT falsified — the clean test is decoupled-vs-coupled at
PROPERLY-SET weight, which the queued arms will run; (3) hitting the 10%
target needs w ≈ 0.10/0.13 ≈ 0.7–0.8 on this critic family — a ~25× raise
with NO gradient cap in the tree, so weights are set from each arm's own
first-50-step division, never copied. CAVEAT: n=1–2 rows/arm; the division
should be re-read on the first 50 steps of every new arm before trusting
its weight.

## PROPOSED STANDING RULE — gradient SHARE, not value (researcher sign-off pending)

Two independent instances of one failure class, found in unrelated code on
2026-08-23:
1. Every historical GAN arm ran at 0.05–1.2% gradient share vs the intended
   5–20% — reported "present", actually inert (the weight-starvation
   confound).
2. WP-SURROGATE: the ancestor's Sobolev gradient-distillation term was
   ~700–3000× under-scaled vs its value term — held-out gradient cosine vs
   the true teacher: ancestor config −0.002 (OFF in all but name),
   normalized +0.986. It logged a plausible loss value the entire time.

Rule: **every auxiliary loss term logs its gradient SHARE alongside its
value, and a term below ~1% of its intended band is reported as INERT, not
present.** WP-PIXGAN has already applied it to its own R1 (share landing in
T3-C alongside the magnitude).

Companion rules from the same day's failures (same class — present and
plausible-looking, but not connected to what it should affect):
- **A flag is not wired until one test drives it end-to-end from config to
  consumer.** Stub-to-stub agreement is not evidence: T1's seam bug passed
  every test because each half was tested against a stub of the other. The
  seam test must FAIL on pre-fix code before the fix lands.
- **Never register an unread-key guard warning away until the key is
  verified read** (by running the scan), an idiom the scan can't see. The
  offered "stop crying wolf" registration would have hidden the T1 seam
  bug indefinitely; the guard was a true positive.
- **The run log must echo the RESOLVED value of every knob the arm's
  conclusion depends on** — not the requested value, and not nothing. Would
  have caught: the seam bug, the G-side micro-batch ambiguity, the A18 dead
  flag.
- **Diagnostics vs SAFETY CLAIMS (WP-SURROGATE's distinction).** A
  diagnostic ("how is it going") may be omitted when uncomputed, with a
  regime flag — never a fake in-range value. A safety claim ("is this run
  valid" — e.g. `pix_holdout_leak`, `pix_a21_support_ok`) may NOT be
  silent: silence and "clean" are too easy to confuse. It must be computed
  on every path where the guarded operation ran, fail LOUDLY naming what
  could not be verified, **and be FRESH per emission — a stale republished
  value is a third defeat route beside omission and forgery (measured:
  `pix_holdout_leak` re-emitted an old value every step as if fresh)** — the fail-closed posture
  `disc_holdout_probe` already implements (`INVALID_*` codes). One
  convention, applied consistently.
- **A planted companion must be RE-PROVEN whenever the thing it guards
  changes shape** — it fails silently, by passing (measured: the A20
  companion planted into a field the reworked gauge no longer read; the
  guard had stopped guarding and nothing said so). Corollary of the same
  class: the resolved-value echo itself can report CONFIG rather than
  RUNTIME (measured twice: pix_gan_lr echoed from cfg while the optimizer
  read elsewhere; pix_r1_every_n echoed 0 while running at 1) — echoes read
  the LIVE object, omit-and-flag on disagreement.
- **A seam test is evidence only after a MUTATION CONTROL**: demonstrate it
  fails on a deliberately-broken copy (seam cut) while the pre-existing
  tests still pass and telemetry still reads healthy. "I added a seam test"
  without that is an assertion, not a result.

**MAIN RULING (23:2x) on WP-SURROGATE's Sobolev question: NORMALIZED
(relative) Sobolev loss is adopted; verbatim-restore of the 835b1df scaling
is REJECTED** (measured decorative: held-out grad cosine −0.002 as shipped
vs +0.986 normalized; a verbatim restore would re-ship the exact
silent-divergence failure this block exists to prevent). Teacher gradient
`.sum()` over crops approved (scale invariant to pix_crops_per_step). First
pixel-critic arm logs critic_grad_loss UNNORMALIZED alongside for ~50 steps
to measure the real disc∘decode scale gap (analytic-teacher caveat).

Instance count for the requested-vs-effective divergence class: SIX in one
day (gradient share; decorative Sobolev; forgeable zeros; T1 seam;
G-side `_Gm` resolution; `latent_origin` plumbing) — plus one identified
in advance: `pix_teacher_refresh_every` defaulting to 1 would leave the
surrogate's entire affordability mechanism silently off while every metric
reads healthy.

## Done so far (ledger)

| piece | state |
|---|---|
| 14B loader/projector/trainer branch | landed, tests; OOM fix queued |
| Pixel T2/T3-A | landed, green; T3-A + probe certified IN VIVO (smoke_pixtex 60/60, DDP clean, memory parity) |
| Pixel T1 (A23 grad path) | unit-green BUT in-vivo UNCERTIFIED — config→pipeline seam gap (guard TRUE POSITIVE: trainer reads config, pipeline reads its own attr, never connected); fail-loud consumer + fix land with T3-B |
| Pixel T3-B/C | in flight (WP-PIXGAN) |
| Surrogate module + tests | green; wiring waits on B1 |
| A22 holdout probe | built, fail-closed, exercised in live arm |
| Weight-probe launch protocol | agreed (plan of record) |
| ganv2f fixes wave + CARN arms + controls | trained + evaled + batteried |
| pixtex300 | pre-queued (holder 6109489) — until the pixel loss lands its critic is idle, so this IS the 300-step base control (researcher: a separate noanchor300 proved redundant, cancelled) |

## Agent weigh-in — append ONLY to your own subsection

### WP-14B
(views on Option A fixes, Option B feasibility through your machinery, and
whether anything in the loader assumes 5120)

**Headline: nothing in the loader assumes 5120 — Option B already works and I
have run it twice today on GPU. And the recorded Option-A root cause is wrong;
the real one is a config line the smoke omitted, so A is not blocked on code.**

#### 1. Option B — confirmed feasible, with measured numbers

`ladd_disc_backbone_model_name=/scratch/u6ex/as1748.u6ex/frodobots/Wan2.1-T2V-1.3B`
works today, no code change. This is not an inspection result: today's scaling
probe (`testing/probe_wan14b_disc_scaling.py`, `docs/WP_14B.md` §6) ran the
**stock 1.3B through the exact Option-A machinery**, on a GPU, twice —

| stock-1.3B config | prefix weights | D-update @12 rows | G-guidance @12 rows, groups=4 |
|---|---|---|---|
| taps `[0,2,4,8]` → 9 blocks | **0.83 GiB** | 8.49 GiB / 503 ms | 11.67 GiB / 886 ms |
| taps `[6,12,18,24,29]` → 30 blocks | 2.64 GiB | 12.14 GiB / 1561 ms | 32.43 GiB / 2850 ms |

Geometry is read from `config.json` (`dim`, `ffn_dim`, `num_heads`, `in_dim`,
`out_dim`, `num_layers`, `text_len`, `text_dim`, `patch_size`); the trainer
reads `dim_teacher` and `patch_size` back off the returned model
(`_dim_teacher = int(_ladd_backbone.dim)`), and `build_ladd_disc` asserts they
agree. The only 5120 anywhere is `_WAN14B_REFERENCE`, a readback aid — I have
just demoted it from a per-key `logging.warning` to one INFO line, so a 1.3B
load no longer emits four scary "differs from the reference" lines. I also
added `load_wan_prefix` as an alias for `load_wan14b_prefix`, since the module
is model-agnostic and the name should not imply otherwise.

**Two traps for the Option-B smoke, one of them serious:**

1. **A disc resume can silently load v14e-trained weights — but the
   conditions are narrower than I first said.** The resume guard drops
   checkpoint entries by SHAPE only, on names that already match. With the
   14B, `dim_teacher` 1536 → 5120 forces every CCM tensor out; with the stock
   1.3B it stays 1536, so nothing is dropped and the load is silent.

   **CORRECTION (mine).** I called this "the one serious trap" without
   stating two further conditions an audit surfaced. (a) The CCM and heads
   are `ModuleDict`s keyed by tap index, so the taps must ALSO match: at the
   recommended `[0,2,4,8]` the keys are `0/2/4/8` against a legacy
   checkpoint's `6/12/18/24/29` — zero overlap, nothing loads. The trap only
   bites at taps `[6,12,18,24,29]`. (b) It requires `auto_resume=true` AND a
   prior checkpoint in the same log_dir; every launcher in this family sets
   `auto_resume=false`. So the arm as planned is already immune twice over.
   Also, under `ladd_freeze_projector_mixing=true` the CCM is a frozen random
   projection that was never trained on v14e features — the heads are the
   part that would genuinely carry over. Still launch fresh and verify the
   startup log; but this is a guard-rail, not a live hazard.
2. Set `ladd_gen_guidance_micro_batch_groups=4` for B as well. It is cheaper
   than A, but the 30-block/12-row un-micro-batched number above is 35.0 GiB
   at 3-frame chunks, and the planned arm uses 6-frame transition chunks
   (2·npb), which roughly doubles it.

Caveat I would put in the plan text: my probe drives the stock 1.3B through
the raw-model path (no action tokens, no `clean_x`/`aug_t`), which is exactly
what the arm will do — but it is not the live `real_score` wrapper, so treat
the table as the cost of the swap, not as a replica of the current arm.

#### 2. Option A — the recorded root cause is falsified by the smoke's own log

The plan records: *"graph-on 14B forward fed the padded 18,721-token seq_len at
5120-dim"*. That is not what happened, and the fix queued against it is a no-op
because the behaviour it asks for has been in since the code landed.

Evidence from `logs/exp_smoke_14bdisc.err` (the 15:12 attempt, **7 of 8**
ranks — rank 4 died differently, at a 1.45 GiB FFN allocation in the same
recompute, which turns out to be the more informative one):

* The failing allocation is **366.00 MiB** in `rope_apply`'s complex multiply.
  That buffer is `S × n_heads(40) × head_dim/2(64) × 2 × 8 B` (float64 —
  `rope_apply` upcasts), and the allocator rounds large requests up to a 2 MiB
  multiple, so a reported 366.00 pins S ∈ [9320, 9369]. The only realizable
  grid there is **S = 9360 = 6 × 30 × 52**, a **6-frame** `2·npb` transition
  chunk. The log confirms it independently: `npb=3`.

  **CORRECTION (mine).** I first argued this allocation *rules out* the
  18,721-token padded budget, since that would be 731 MiB. **That inference is
  invalid** — an adversarial review caught it. `rope_apply` takes its `S` from
  `grid_sizes`, which `wan/modules/model.py:741-742` builds from the patch grid
  **before** the zero-pad at `:745-749`. The RoPE buffer is padding-invariant:
  fed 18,721, this allocation would still have been 365.62 MiB. The 366 MiB
  pins the chunk geometry and says nothing about padding.
* The padded-budget theory is still false, but on **code**, not arithmetic:
  the projector computes the actual token count itself
  (`_raw_backbone_seq_len`) and **raises** if a caller passes a `seq_len` that
  disagrees, so it never inherits the wrapper's 18721. That is unit-tested
  (`test_projector_runs_raw_backbone_at_actual_token_count`,
  `test_projector_rejects_wrapper_only_arguments`) and confirmed on real
  weights at the production grid (4680 tokens for a 3-frame chunk).

The actual failure: the frames above the OOM are
`Tensor.backward → checkpoint.unpack_hook → recompute_fn → ladd_disc._run_teacher`
inside `_streaming_train_one_chunk`, i.e. the **generator-side guidance**
forward being recomputed during the gen backward — not the D-update, which
runs its own micro-batched helper and does not appear in the trace.

**CORRECTION (mine, same day).** I first wrote that the smoke left
`ladd_gen_guidance_micro_batch_groups` at the code default of 1. That is
**wrong** — the main session challenged it and was right. `run_smoke_14bdisc.sh`
calls `sbatch/_fgan_holder.sh`, whose own fixed argument list passes
`ladd_gen_guidance_micro_batch_groups=4` at `:474`, and `$DEXTRA` (`:476`,
which does not carry the key) merges *after* it. **The resolved value was 4.**
Nor does my path ignore it: the split lives in the trainer's shared `_m_fwd`
(`:8500-8530`), which chunks the row dim before calling `disc(...)` and is
backbone-agnostic, and the gen call site explicitly does
`disc_for_guidance.eval()` (`:8905`) so the eval-gated micro-batch branch is
live. So it is neither "knob absent" nor "raw path ignores knob" — it is the
main session's option (b): **groups=4 is simply not enough at this chunk
size.**

Measured, for the smoke's ACTUAL geometry (6-frame `2·npb` transition chunks,
9360 tokens/row — my first numbers were 3-frame, which understated it ~2×):

| rows | D-update | G groups=4 | G groups=8 | G groups=16 |
|---|---|---|---|---|
| 4  | 19.5 GiB | 24.9 GiB | — | — |
| 8  | 32.4 GiB | 43.2 GiB | 28.8 GiB | — |
| 16 | 58.3 GiB | **79.8 GiB** | 51.0 GiB | 36.7 GiB |
| 32 | OOM | OOM | OOM | 66.6 GiB |

The gen-side forward carries `n_uniq + n_fake` rows, `n_uniq` capped by
`ladd_gt_transition_match_max_real=16`. Rank 4's 1.45 GiB FFN allocation pins
the actual micro-batch: `rows_micro × 9360 × 13824 × 2 B` = 1.45 GiB ⟹
`rows_micro = 6`, so with a 4-way split the batch was **21-24 rows**, not the
"~17-20" I first wrote. The smoke therefore sat *past* the 16-row / 79.8 GiB
cell (22-24 rows needs ~110-120 GiB), before generator + fake_score +
real_score + optimizer. It OOMed at 92 GiB. The measurements account for the
failure **with room to spare** — "exactly" was not earned.

**A SECOND GEN-SIDE PATH IGNORES THE KNOB ENTIRELY.** Land this before any arm
runs with `ladd_gt_transition_match=false`: the row split lives in `_m_fwd`,
which the **matched** branch calls, but the **positional** branch calls the
disc directly (`combined_g_logits = disc_for_guidance(x_noisy=combined_g, …)`)
and `ladd_gen_guidance_micro_batch_groups` appears nowhere in it. On that path
the knob is a silent no-op at any value and the un-micro-batched cost (~8.5
GiB/row at F=3, ~17 at F=6) applies in full. The smoke used `match=true`, so
this did not cause its OOM — but a `match=false` arm would OOM identically
while the knob looked correctly set.

**So Option A's blocker is still config-only, but the value matters — 4 is not
the fix, 16 is:**

```
ladd_gen_guidance_micro_batch_groups=16    # NOT 4; 4 was already in effect
ladd_gt_transition_match_max_real=8        # was 16 — halves the rows, and the
                                           # G-guidance time with them
```

Put both in the smoke's `DEXTRA`, which merges after the holder's fixed list
and so overrides `:474` **without editing `_fgan_holder.sh`** (which must not
be edited while a run is executing it). Expected disc-side peak: ~37 GiB at 16
rows, ~25 GiB at 8-10 rows.

The micro-batch remains free in time — at 16 rows groups=16 was *faster* than
groups=4 (11051 vs 11579 ms) — and byte-identical (per-row independent
concatenation, EVAL-mode only).

**One cost the plan should book before committing to a 300-step arm:** at
16 rows / F=6 the G-guidance forward+backward alone is ~11 s, and a D-update
~6 s. At `gan_updates_per_step=5` that is ~40 s/step of disc time, so 300 steps
is ~3.5 h of GAN overhead. `max_real=8` roughly halves it. This is a property
of 6-frame transition chunks at 5120-dim, not of the prefix loader.

The other two queued items are genuinely done: actual-token seq_len since
landing, and name resolution was the 15:00 retry that switched to the full
path.

Cost against the same code path at 1.3B, for budgeting: **+3.7 GiB resident**
(2.64 → 6.32 GiB), G-guidance **~1.2× time / ~1.05× memory**, D-update
**~1.3× time / 2.16× memory** (12.1 → 26.1 GiB).

**CORRECTION (mine).** I headlined this as "~1.2× time, ~1.05× peak memory
**of the disc we run today**". Two things are wrong with that framing. First,
1.05× is the G column only — the D column more than doubles. Second, the
baseline row is the raw-model path driven by stock 1.3B weights at the ACTUAL
token count; the disc we run today goes through `WanDiffusionWrapper`, which
supplies its own padded `seq_len`, i.e. ~4× the tokens. So today's real disc is
*more* expensive than my baseline row, and the true swap ratio is more
favourable than the numbers above — but it has not been measured, and the
comparison should not be quoted as "vs today's arm". A is affordable either
way; it was never an OOM of principle.

#### 3. On WP-PIXGAN's two objections

**The weight probe (their point 1): agreed, and it is the difference between
an interpretable arm and another null.** I have no stake in the pixel
implementation, but the confound is real — a critic at 0.05–1.2 % of the DMD
gradient is inert whatever basis it projects onto, so `decouple13b` and
`decouple14b` launched at an inherited weight would re-run the same
uninterpretable experiment with a different backbone. Generalising the
unweighted-ratio telemetry to the LADD path is my territory and I am ready to
do it, with one process constraint: the trainer is currently WP-PIXGAN's under
the section-B landing order, and I released it to them. **I will not edit
`trainer/causal_action_forcing_train.py` until they say it is free.** Cleanest
sequencing: they land `pix_*` unweighted ratio as a small reusable helper
(loss term + params in, ratio out), then I add the LADD call site — a few
lines, no new machinery. Say the word and I will queue it.

**The LADD unweighted ratio may already be in the logs — no probe run, and
retroactive.** Worth checking before anyone launches a probe arm. The A7
telemetry differentiates `gen_gan_loss`, which is
`gen_gan_weight × (unweighted GAN term)`, and `train/r3gan_g_weight`
(`trainer:9067`) logs that weight fully resolved — post warmup ramp, post
`ladd_disc_loss_weight`, post gate-coupling. Gradients are linear in a scalar
multiplier, so

```
r_unweighted  =  train/gan_dmd_grad_ratio  ÷  train/r3gan_g_weight_gtxn
```

**Key-name correction:** I first wrote `train/r3gan_g_weight`. Every key
returned by `_ladd_run_pair_mode` is suffixed by pair mode
(`_gtxn` / `_gt` / `_adj`) unconditionally, so an UNSUFFIXED
`train/r3gan_g_weight` is never emitted on the LADD path — a query for it
returns nothing. (The main session evidently found the right key anyway; their
read used `r3gan_g_weight_gtxn`.)

is an identity, not an approximation. If it holds up, the
weight-starvation-vs-blindness confound is decidable **on the arms already on
record**, from wandb history, without launching anything — and A/B need no
special weight-free mode, just a short run at any nonzero weight followed by
a division.

Caveats, all checkable, and revised after an audit of my own claim:

* Exact only while the GAN term is a pure scalar multiple, i.e.
  `ladd_stat_head_enabled=false` (true in these arms — the stat sideband
  carries its own weight). **Confirmed** at both gen-side return sites.
* **A second multiplier is NOT logged.** `_compute_ladd_losses` aggregates
  per pair-mode as `gen_loss_total += w * g_loss` with
  `w = ladd_gt_transition_weight` (default 1.0), while
  `r3gan_g_weight_gtxn` records only `gen_gan_weight`. The arms leave it at
  1.0 so the identity survives — but it is a silent failure mode, not a
  structural guarantee. Check it per arm.
* **The denominator is not structurally "pure DMD".** It is
  ‖∇(everything in `generator_loss` except the GAN and pixel-perceptual
  terms)‖ — which includes the action-critic, LoRA-action, state-probe and
  aux-disc gradients whenever `aux_active`. In the holder-launched arms
  `action_critic_aux_enabled=false`, so it does reduce to DMD there; verify
  `aux_active=false` per arm before dividing, and do not read the key's name
  as a guarantee.
* Only for runs after A4's cap deletion (the cap rescaled `gen_gan_loss`
  immediately before the telemetry). The cap left no git trace — it lived and
  died in the uncommitted tree — so this rests on the in-code removal note,
  not primary evidence.
* Needs `r3gan_g_weight_gtxn > 0`. My earlier step-alignment caveat can be
  **dropped**: both keys are written into the same `out` dict on the same
  call, so they are necessarily co-logged.
* The ratio is a **single-parameter** estimate — see the caution below.

**CAUTION on using `r` to SET the weight — it is measured at ONE parameter.**
Now that the identity has been read on real arms and probe arms are cancelled,
this matters more, not less. `trainer:12526-12535` walks
`self.model.generator.named_parameters()`, keeps `_p_last` — the LAST
requires_grad parameter in iteration order — and takes both
`autograd.grad(generator_loss, _p_last)` and
`autograd.grad(gen_gan_loss, _p_last)`. So `r` is a **per-tensor** gradient
share, not the model-wide one. Two consequences:

* `w = 0.10 / r_probe` assumes the share at that one tensor equals the share
  over the full parameter vector. The GAN term enters through the disc on the
  student's output latents and the DMD term through the flow-matching
  objective; they have different depth profiles through the generator, so the
  assumption is not free. It may well be fine — but it is currently unstated.
* It is a candidate explanation for the >10x spread across arms of the same
  family (`all` 1.84 vs `notaps` 0.125, `nopatch` 0.153) that has nothing to
  do with the arms differing in substance: `_p_last` need not be the SAME
  parameter across arms, since it depends on construction order and on which
  params carry requires_grad, which tap/patch variations plausibly change. If
  it differs, those numbers are not measured at the same probe point and are
  not comparable — and the per-arm weights would have been set from
  incommensurable measurements.

Cheap fix, one line: log the NAME and shape of `_p_last` beside the ratio. It
either kills this explanation or exposes a real problem. Stronger version:
probe at a small fixed set spanning depth (first / middle / last requires_grad
param) — still two `autograd.grad` calls if the params are passed as a list —
and treat the single-point recipe as safe only if they agree within ~2x.

**The latent positive control (their point 2): agreed, and I think it is
mandatory, not optional, for A and B.** Their reading is right that `d_loss ≈
ln 2` is undecidable between undertrained and wrong-design, and I confirm
their grep: there is no latent-side control in the tree. The cheap version is
genuinely cheap on my side — the C1 corruption in `c1_structured_hf` already
perturbs the **latent** before decode, so the corruption generator transfers
untouched and only the readout needs swapping to the LADD scorer (which lives
in my file, `model/ladd_disc.py`). Estimate: a scorer wrapper + a
`positive_control` entry point + a test, no trainer edit, comparable in size
to the probe I landed today. I would rather build it than have A/B ship with
criterion (i) unreadable — but it is a new build beyond WP-14B's brief, so I
am flagging it as ready-to-start rather than starting it unasked.

**On ordering** I agree with WP-PIXGAN's B → C → A, with one amendment: A is
no longer blocked on code, so if a holder is free, A's re-smoke is now as cheap
to try as B's (one config line each). Their argument that B is the weakest
decoupling — same family, same scale, merely un-fine-tuned — stands, and is the
reason not to let a B null close the question.

#### 4. Sign-off

**I, 14B agent, am finished oh great main agent.**

Everything above has since been through an adversarial review pass (four
independent reviewers: loader/projector defects, trainer+config gating,
claim-vs-evidence audit, and mutation testing of the test suite). It cost me
five published claims — the 366-MiB falsification argument, the row count, the
resume-trap scope, the cost headline, and a comment asserting the head would
crash — all corrected in place above and marked as corrections. Code defects it
found are fixed (patch_size validation, fail-loud when the `max_block` early
exit cannot be plumbed, i2v rejection, architecture flags read from
config.json, rank-0 logging); the suite went 13 -> 19 tests after mutation
testing showed 12 mutations slipping through, including a tap shifted by one
block and a wrong RoPE table. Probe artefacts are in `logs/wp14b_probe/`
rather than prose-only.

Concretely: the loader, projector, trainer branch, config block, tests,
scaling probe and both reports are landed, and after a four-agent review round
the code carries seven defect fixes and the test suite seven new tests
(**25 passed, 1 skipped** across the WP-14B suite plus the existing LADD
regression tests, CPU). The Option-A re-smoke needs the one config line in
section 2 and no code from me; Option B is confirmed working through the same
machinery, with the resume trap in section 1 now correctly bounded.

**One verification gap, stated plainly:** the real-weights GPU smoke passed
*before* the review fixes (19:0x, taps [0,2,4,8], 6.79 GiB, 4680 tokens,
flash-attn, artefacts in `logs/wp14b_probe/`). The fixes since — chiefly moving
the forward hooks inside the checkpointed region — are CPU-tested only, because
every holder expired while the review ran. **Re-run
`WAN14B_PREFIX_SMOKE=1 python testing/test_wan14b_prefix.py` on a GPU before
the Option-A re-smoke**; it takes ~4 minutes. I will do it myself the moment a
holder is running if nobody beats me to it.

Two items are flagged but NOT started, each waiting on someone else:
* the LADD call site for the unweighted-ratio helper — blocked until
  WP-PIXGAN releases `trainer/causal_action_forcing_train.py`, and it may be
  unnecessary if the `r_unweighted = gan_dmd_grad_ratio / r3gan_g_weight`
  identity in section 3 checks out against a recorded run;
* a latent-side positive control for criterion (i) — ready to build, needs a
  go-ahead since it is beyond WP-14B's brief.

Ping me if either becomes live, or if a re-smoke turns up something that looks
like the backbone rather than the budget.

**Post-review addendum.** Everything above was put through four independent
review agents (adversarial code review, trainer/config gating, a claim audit
against primary evidence, and mutation-testing of the test suite) before this
sign-off. They found real defects in both the code and my own claims; the
corrections are marked inline above, and the code fixes are listed in
`docs/WP_14B.md` §11. Three findings that touch OTHER packages, repeated here
because they are not mine to fix:

* The positional gen-side branch bypasses
  `ladd_gen_guidance_micro_batch_groups` entirely (section 2) — any
  `match=false` arm is unprotected.
* `train/gan_dmd_grad_ratio` is `_ng/_nr if _nr > 0 else **0.0**` — a logged
  0.0 can mean "the DMD gradient was zero", not "the GAN term contributes
  nothing". WP-PIXGAN flagged the same trap independently and the main
  session's read excluded those rows, which was correct.
* With `gan_enabled=false`, the whole 14B block is skipped in silence — no
  warning, no raise — so an arm cloned from a pixel-GAN sbatch would run with
  no critic at all and a config that says otherwise.


### WP-PIXGAN

**I, pix gan agent, am finished oh great main agent.**

---

#### FINAL STATUS — B1 / WP-PIXGAN, 2026-08-24

**BUILD COMPLETE. All five suites green, CPU:**

| suite | result |
|---|---|
| `test_pixel_texture_disc.py` | 99 passed, 51 subtests |
| `test_a23_finish_grad.py` | 45 passed |
| `test_pixgan_trainer_wiring.py` | 39 passed |
| `test_pixgan_trainer_supply.py` | 90 passed |
| `test_pixgan_t3c_gterm.py` | 40 passed |

Delivered: the §4 pixel PatchGAN with patchwise NS-logistic/hinge and a single
FD R1 on the MEAN patch score; the A23 inference-parity grad path with a
frame-level attachment mask; full trainer wiring (gate, `pixel_texture_disc`,
`pix_optimizer`, fail-loud save/resume, A20/A21/A22/A24 real supply, D-loop,
G-term, `grad_at`, unweighted ratio, R1 share, §7 telemetry); the `pix_*`
config block; and the `pix_kv_commit_check_every` tripwire. Flag-gated,
default-off, byte-identical when off.

#### THE ONE THING THAT IS NOT DONE — read this before ticking any box

**[CLOSED 2026-08-24 — the seam mutation control is now VERIFIED: the guard WORKS.** M1 cut fails `test_flag_lands`, M2 cut fails `test_gate_re_asserts`, M3 fails both, PRISTINE 90/90. Both sites independently guarded; construction confirmed load-bearing (`self.pipeline` bound once at :1611, never rebound). The earlier false passes were `sys.path` artefacts from repo-internal import-time `sys.path.insert` calls, not the `_ROOT` issue alone — see `WP_PIXGAN.md` §31.] ~~The seam mutation control is UNVERIFIED.~~ `test_flag_lands_on_the_object_whose_class_reads_it`
and `test_gate_re_asserts_the_flag_after_a_pipeline_rebuild` are **asserted, not
demonstrated**, to catch a disconnected flag.

I ran it and got a clean-looking three-way result, then found it was a harness
artefact: the seam test parses source via
`_ROOT = dirname(dirname(abspath(__file__)))`, so it read the **real** trainer
regardless of my mutant. **The same trap that gave an earlier reviewer 17/18
false survivors** — and I walked into it after warning three other packages
about it. Caught only by disbelieving an identical PRISTINE/M3 result.
Method for whoever finishes it is in `docs/WP_PIXGAN.md` §30.

#### VERIFICATION LIMITS — do not report these as GPU-verified

Everything above is **CPU-verified** (GPU-green pending). ~~CPU-only / cannot be collected on a compute node~~ CORRECTED 13:4x: the 6110540 import diagnostic returned IMPORT-OK in all three modes — the node was never the problem; the failure was intra-process pytest ordering (test_a23_finish_grad poisons later collection in the same process). Suites are GPU-verifiable one-process-per-suite. Original claim (struck): the trainer suites cannot even be *collected*
on a compute node (`from pipeline import ...` -> "unknown location", reproduced
3x). `pix_finish_grad_enabled=true` has **never executed in a real training
loop** — `BID PIXGAN-1` on 6110540 is what would settle it, together with the
KV recompute-order hazard (silent, pre-existing, flash-path-shaped).

#### BLOCKING THE ARM

1. **`pix_gan_weight`** — no default by design; from the arm's own unweighted
   ratio. Must NOT inherit the LADD family's 0.65–0.80 (or 0.054).
2. **`pix_r1_gamma`** — same discipline. **1.0 is INERT on both
   architectures** (R1/d_loss ~6e-4 normed, ~5e-6 norm-free). Not a weak
   setting — an absent one.
3. Researcher decisions D1/D2/D3 (`WP_PIXGAN.md` §26): γ from measurement;
   GroupNorm **dropped** (authorised deviation from §4); resolve both before
   arm 1.

#### FINDINGS WITH REACH BEYOND THIS PACKAGE

- **A21 launch blocker**: `pix_a21_support_ok` could never be true (ceiling
  4069–4085 vs the 4096 floor). Fixed.
- **Safety claims have a THIRD defeat route: STALENESS** — computed once, then
  republished forever, indistinguishable from a fresh measurement. After
  omission and forgery. Now a standing rule.
- **`gamma=1.0` inert on both builds** — would have shipped with
  `pix_r1_rate` reading 1.00 the whole time.
- **§5.3's "scale-free in P" is FALSE** (alpha = −0.915). Both justifications
  for the mean reduction are now falsified.
- **§4's "local texture question" was false of the built critic** — GroupNorm
  made every patch logit depend on every pixel.
- **§4's param count (2-3 M -> 661,185) and patch count (768 -> 660)** wrong.
- Eleven-plus instances of one mechanism: *present, plausible, logged, and not
  connected to what it is supposed to affect.* Five detection gaps —
  value-not-share, placeholder-not-absence, endpoints-not-seam,
  requested-not-resolved, queue-time-not-start-time.

**Trainer-file lock: RELEASED to WP-14B**, explicitly, as of this line.

Full detail: `docs/WP_PIXGAN.md` (30 sections).

I agree decoupling is worth pursuing and that C is the full escape. Two
problems with the plan as written, though — both would let a decoupled arm
return a null result for a reason that has nothing to do with decoupling.

**1. The blindness evidence is confounded with weight-starvation, and the
plan currently carries that confound into the decoupled arms.**

The A7 line cited as supporting evidence — `gan_dmd_grad_ratio` 0.0005–0.012,
i.e. **0.05–1.2 %** — is not evidence of blindness. It is evidence that those
critics contributed **~1/100th of the intended gradient share** (§5.5's target
band is 5–20 %). A critic at 0.05–1.2 % is near-irrelevant *whatever basis it
projects onto*; "blind because coupled" and "inert because ~100× underweighted"
predict the same null, and the arms on record cannot separate them. The
poolrich-vs-raw_t0 comparison is separately confounded on ~7 axes (lr 5×,
EMA-from-0, fake-EMA 0.0, GANW/updates, sampled-t shift, action-cond, pool), so
it cannot break the tie either.

This is not an objection to the thesis — it may well be right. It is an
objection to the **experiment**: if `decouple13b` and `decouple14b` run at
whatever weight they inherit, they will very likely land at 0.05–1.2 % again,
return the same null, and it will be misread as *"decoupling didn't help"* when
decoupling was never actually tested at a weight where any critic could matter.

**Concrete ask: the weight-free probe protocol must apply to Options A and B
too, not just C.** It is not a pixel-critic nicety; it is the control that makes
any of these arms interpretable. One ~50-step probe per backbone, read the
unweighted ratio, set weight = 0.10/r_probe, then bracket. Cheap, and without it
the decoupling readout is not available. Note this also means the unweighted-
ratio telemetry I am landing in T3-C should be generalised to the LADD path, or
A/B need their own equivalent — as things stand only the pixel arm can run the
probe. (A4's deletion means there is no gradient cap on any of these paths, so
raising the weight also needs the probe rather than a guess.)

**2. Criterion (i) is undecidable for A and B as specified.**

Readout (i) — *"a d_loss trajectory that actually separates, not pinned at
ln 2"* — is exactly the reading `TEXTURE_GAN_DESIGN` §4.1/§8.1 establishes is
**not available on its own**: `d_loss ≈ ln 2` reads identically for
*undertrained* and *wrong design*, and that ambiguity is recorded as having
"cost this campaign repeatedly". The doc lists the positive control as (iii),
which is right, but the three are presented as a flat list. They are not flat:
**(iii) gates (i) and (ii)** — that is the whole point of §8.1's decision table.

Worse, **(iii) does not currently exist for Options A and B.** I checked:
`poscontrol` / `positive_control` appear **only** in `model/pixel_texture_disc.py`
— there is no latent-side positive control anywhere in the tree. So A and B as
planned would produce precisely the undecidable ln-2 reading that motivated
building the control in the first place.

Two ways out, either fine: (a) give A/B a latent-side positive control — the C1
corruption in `c1_structured_hf` already perturbs the **latent** before decode,
so the corruption generator transfers directly; only the readout needs a latent
scorer instead of `positive_control_readout`'s pixel disc; or (b) state
explicitly in the plan that A/B ship without a control and that their ln-2 reads
are therefore uninterpretable, so they are judged on (ii) battery movement
alone. What should not happen is reading (i) as if it were informative.

**Ordering.** Run **B now** — it is config-only, immune to the OOM by
construction, and a nearly-free first decoupled data point; no argument. But do
not sequence **C behind A**. A is blocked on an OOM fix of unknown duration; C
is blocked on two chunks of my own work already in flight, and C is the only
option that tests the thesis at full strength — B is the weakest form
(same family, same scale, same latent space, merely un-fine-tuned), so a B null
is ambiguous between "decoupling doesn't help" and "that wasn't enough
decoupling". My recommended order is **B (now, free) → C (when T3-B/C land and
pass review) → A (when the OOM fixes land)**, with the caveat above that all
three need the weight probe.

**First-arm fake source: `pix_finish_grad_enabled=false`** (fake =
`flash_dmd_gan_x0`), as agreed — cheaper, no retained rung graph, no
frame-dilution mask. The fake is the student's own output either way, so it does
not interact with the coupling question at all. One caveat worth booking as an
**early** follow-up rather than a distant one: A23 showed the flash tensor is
*plausibly* systematically cleaner than the ladder endpoint inference actually
renders (the statistical claim collapsed under multi-seed, but the mechanism —
an extra denoising forward — is a priori reasonable). For a **texture** critic
specifically, training against a systematically-cleaner-than-deployed tensor is
the failure mode we would least like to have, because cleanliness is the judged
property. So: flash for arm 1, and `pix_finish_grad_enabled=true` as the first
follow-up variable, not the last.

**What the decoupling thesis changes in the B1 spec: essentially nothing, and
it strengthens one recorded exception.** §4.1's from-scratch PatchGAN was a
deliberate exception to standing decision 4, justified on three grounds (VQGAN
precedent, small hypothesis class, fast first signal). The decoupling thesis
supplies a fourth and stronger one: a from-scratch critic is the **only** critic
in this family with provably *zero* shared basis with the DMD teacher. Worth
recording, because it has a consequence — the B5 escalation (swap to DINOv2 or
SD/VQGAN weights on an UNDERTRAINED verdict) is still decoupled from the DMD
teacher but no longer has the zero-shared-basis property. Minor, but the
escalation should be taken with that in view rather than as a pure upgrade.

**One residual coupling in C, bounded and deliberate.** Real and fake are both
decoded through the **same frozen Wan VAE** the DMD teacher uses (§2), so the
pixel critic cannot see errors the decoder itself masks. That is intentional —
same-decoder comparison cancels the decoder's transfer function, and reals as
raw RGB would reward the student for out-sharpening the decoder, which is the
fabricated-HF failure we are treating. It is also bounded by the A/B/C
diagnostic's Case-1 verdict (the round-trip only softens, ~15 % HF; structure,
tail statistics and isotropy survive). Flagging it only so "full escape from the
latent-teacher family" is not read as "zero shared components with DMD" — the
VAE is shared, by design, and the diagnostic says that is safe.

### WP-SURROGATE
(views on whether the surrogate teacher should be the pixel critic only, or
whether a decoupled latent backbone (A/B) changes your teacher choice)

**Short answer: the teacher stays the pixel critic, and a decoupled latent
backbone does not change that — but not because A/B are worse. It is because
the surrogate has nothing to offer them.**

**1. The surrogate is basis-neutral. It cannot decouple anything, and it
cannot re-couple anything either.**

The surrogate is not a perceptual basis. It is a regression onto `grad_z` of
whatever teacher it is handed — from-scratch weights, no pretrained
initialisation, no relationship to the DMD teacher. So it adds **zero** shared
basis with `real_score`. It also **launders nothing**: distilling a
DMD-projected critic would reproduce a blind gradient field faithfully, and my
`critic_grad_cos_sim` would report a healthy number the whole time. Whatever
the teacher's coupling is, the surrogate's is identical. The decoupling
question is settled entirely by the teacher choice, which is why I do not
think this package has a vote on A-vs-B-vs-C beyond the note in (4).

**2. Why the pixel critic, and not A/B: the saving the surrogate exists for is
the VAE decode, and A/B do not have one.**

The mechanism's entire purpose (B3 §1) is removing `VAE.decode` from the
generator's autograd graph — that is the cost that blocks scaling coverage
toward R10's >=60 %. A latent backbone teacher has no decode: the generator can
differentiate through it directly. Interposing a surrogate there buys compute
at a fidelity cost and changes nothing structurally. Concretely, my held-out
direct-vs-surrogate gradient cosine tops out around 0.88-0.99 on analytic
teachers — good, but strictly worse than 1.0, which is what the direct path
gets for free when there is no decode to avoid. Paying that for compute you
did not need is a bad trade.

**3. The one contingency where Option A *should* take the surrogate — and it
costs zero code.**

If A's 92 GB OOM does **not** fall to the seq_len/micro-batch fixes (i.e. the
graph-on 14B forward is intrinsically too large rather than accidentally
18,721-token padded), then the surrogate is exactly the mechanism that makes a
14B teacher affordable: run the 14B every `pix_teacher_refresh_every` steps,
distil, generator consumes the small latent critic every step. This needs **no
change to `model/latent_texture_critic.py`** — the teacher is an injected
callable `teacher_value_fn(z) -> logits`, and a latent backbone is a *simpler*
teacher than the pixel one because there is no decode in the closure at all.
Worth booking as A's fallback rather than discovering it later. I am not
proposing it now: fixing a padding bug beats adding a lossy approximation on
top of it, and the surrogate should be A's last resort, not its first.

**4. Experimental-design point: keep the surrogate OUT of the first Option-C
arm.**

This is my one substantive ask, and it is the same species of objection
WP-PIXGAN raises above. If the first decoupled pixel arm runs on the surrogate
path, a null has **three** live explanations — coupling irrelevant, weight
~100x too low, surrogate transmitting only ~0.9 of the direction — and they are
**multiplicative**, not additive. The first C arm must use B1's direct path so
a null has one fewer degree of freedom. The surrogate is the *scaling*
mechanism for when coverage goes up; it is not part of the hypothesis test.
This coincides with the existing landing order, but I want it recorded as an
experimental-design constraint rather than an engineering convenience, because
those get reordered under schedule pressure and this one should not be.

**5. Concrete interaction with the weight probe (bears on WP-PIXGAN's point 1).**

When the surrogate is eventually swapped in, **the weight probe must be re-run,
or the weight divided by the measured magnitude ratio.** My
`train/surrogate_check_mag_ratio` is exactly `||g_surrogate|| / ||g_teacher||`;
it measured **0.88-0.89** in the module's fits. A `pix_gan_weight` calibrated
against the direct path via `0.10 / r_probe` therefore lands ~10-12 % low the
moment the surrogate replaces it — small next to the 100x starvation, but it is
a known bias with a measured value and there is no reason to eat it. The
telemetry to do this is already in the module and needs no new work.

**6. A second instance of WP-PIXGAN's point 1, found independently in this
package — the class of bug is the finding.**

Building B3 I measured that the two distillation terms differ in natural scale
by **~700-3000x** at the test crop (growing with crop size), because the
teacher's value is an O(1) mean patch logit while its per-element derivative is
that response spread over the crop. The ancestor shipped
`gan_critic_grad_loss_weight: 1.0` against a raw MSE
(`835b1df:configs/action_forcing_phase1.yaml`:704, unchanged through
`01ea13d^`). Measured end-to-end at 120 fit steps, held-out gradient cosine
against the true teacher:

| configuration | cos | `||g_sur||/||g_true||` |
|---|---|---|
| value-only (`grad_loss_weight=0`) | -0.003 | 0.00 |
| ancestor: raw MSE, weight 1.0 | -0.002 | 0.00 |
| normalized, weight 1.0 | **+0.986** | 0.89 |

**The Sobolev term in the deleted implementation was decorative** — statistically
indistinguishable from switching it off — and it logged a plausible-looking
`critic_grad_loss` the entire time. Fixed here by normalising it into a
relative gradient error (`grad_loss_normalize`, default ON;
`False` reproduces the ancestor). Details and sign-off request in
`docs/WP_SURROGATE.md` §2(e).

I raise it here rather than only in my own doc because **that is now two
independent instances, in two different packages, of the same failure**: a loss
term that is present, wired, logged, and contributing ~1 % or less of what its
weight implies — invisible because the telemetry reports the term's *value*
and never its *share*. WP-PIXGAN's `gan_dmd_grad_ratio` at 0.05-1.2 % is the
same bug wearing different clothes. That suggests a general rule worth adopting
beyond either package: **every auxiliary loss term logs its gradient share, not
just its value, and a term below ~1 % of the intended band is reported as
inert rather than as present.** Cheap, and it would have caught both.

**7. On WP-PIXGAN's point 2 (no latent-side positive control) — my critic does
NOT solve it, and I want to head off the natural assumption that it does.**

`LatentTextureCritic` is a latent scorer, so it looks like the missing readout
for a latent positive control. It is not: it is *distilled from the pixel
disc*, so pointing it at C1-corrupted latents measures the pixel disc's
sensitivity, not Option A's or B's. Using it that way would produce a control
that passes or fails for reasons unrelated to the backbone under test.

What A/B actually need is not a scorer — they already have one, their own disc.
They need the **harness** generalised to accept an injected scorer callable
instead of hardcoding the pixel disc, which is the same pattern B3 already uses
for its teacher (`teacher_value_fn(z) -> logits`, any batch-first shape,
reduced by mean over non-batch dims — see `reduce_patch_logits`). Since
`c1_structured_hf` already perturbs the **latent** before decode, the
corruption side transfers unchanged; only the readout needs the injection. That
is a small change in WP-PIXGAN's file and I am not proposing to make it — but
it is the cheapest route to WP-PIXGAN's option (a), and it means (iii) is
available for A and B rather than them shipping without a control.

**Status.** Module + tests green, wiring specced and deferred; two items in
`docs/WP_SURROGATE.md` §6 await main-agent sign-off (the
`grad_loss_normalize` default above, and a batch-reduction detail that is the
frozen contract's expression evaluated per crop). Nothing in this package
blocks B, C or A.

