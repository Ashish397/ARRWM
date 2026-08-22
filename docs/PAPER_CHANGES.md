# Paper change list — AnonymousSubmission2027.tex

All line numbers refer to `aaai_template/AuthorKit27_extracted/AnonymousSubmission2027.tex`
as of 2026-08-17. Every "verified" number below was recomputed with a pipeline
first validated against published values (validation evidence noted per item).

---

## A. Structural

**A1. Move the No Critic ablation into the main body.**
Lines **2736–2878** (`\subsection{Critic Ablation}` up to line 2879
`\section{Training Infrastructure and Statistical Protocol}`). Includes
Figs. `fig:no_critic_gt_flip`, `fig:no_critic_wedges` and Tables
`tab:no_critic_directional` (16) and `tab:no_critic_stationary` (17).
`\appendix` is at line **913** — the block must land before it.
All cross-references are `\ref` (no hardcoded numbers anywhere in the file),
so renumbering is safe.

**A2. Line 2792 caption** — drop "in the appendix" from *"reproduced here to
make the \texttt{No Critic} comparison explicit in the appendix"*.

**A3. Method §critic (line 472)** — the section asserts the critic is necessary
without pointing at evidence. Add a forward reference to the promoted ablation;
this is the main thing promotion buys.

**A4. Do NOT add No Critic to Table 1** (`tab:quality_all`). That is the
cross-*model* comparison (six baselines plus two released configurations);
an ablation there implies it is a shipped system. Table 3 is the right
main-body home (see C3).

**A5. Relabel `app:no_critic_ablation`** once it is no longer an appendix, and
re-read the four `\ref`s at lines 2745, 2769, 2802, 2842 so they scan as
main-body prose.

**A6. Contribution list (intro)** — contribution 2 is the critic. With its
necessity now demonstrated in the main body, state it as validated rather than
asserted.

---

## B. Number corrections (verified)

| # | line | table | current | change to |
|---|---|---|---|---|
| B1 | **1779** | 7 | `Ours (No Critic) & 240 & 92` | `240 & `**`75`** |
| B2 | **2489** | 13 | `No Critic & 0 & 2 & 11 & 0 & 16` | Reloc `11`→**`3`**, HF `16`→**`20`** |
| B3 | **2821** | 16 | Scene relocation `10.9%` | **`2.7%`** |
| B4 | **2823** | 16 | HF `16.3%` / `27.2%` | **`20.0%`** / **`5.9%`** |
| B5 | **2826** | 16 | Control fail `77.3%` | **`79.2%`** |
| B6 | **2828** | 16 | Active population `92 / 256` | **`75 / 240`** |
| B7 | **821** | 2 | `No Critic & 0 & 0 & 3 & 0 & 13` | HF `13`→**`3`** |
| B8 | **~2866** | 17 | HF degradation No Critic `12.5%` | **`3.1%`** |

Notes:
- **B1/B6** — active is defined *after* the feature-valid mask (Table 7), so it
  should be written `75 / 240`, not `X / 256`.
- **B4** — Default's `27.2%` is unreproducible at any threshold of the deployed
  metric (B>0 → 22.6%, B>−20 → 31.4%); no other fleet model scores 27.2% on the
  active basis. It is not the deployed `B>150` statistic. The correct value,
  `5.9% = 14/239`, is what Tables 13 and 14 already report.
- **B7/B8** — the manuscript's HF was computed on the wrong stationary campaign
  (plain `B_Astarts`, z=(0,0)); the null-inverse camera holds stiller and
  degrades less. Corrected value is `1/32`.
- **B3/B7/B8** — note `10.9%` is exactly Default's relocation rate and the HF
  denominators coincide with Default's; the signature of row-copying.
- **B3 threshold fragility (±1 rollout).** No Critic's relocation is **2/75 =
  2.7% (→3)** as scored on the cluster, and **1/75 = 1.3% (→1)** as scored on
  the local node. The two runs agree on 74 of 75 rollouts; the single
  disagreement is `r12_BR`, which sits *on* the 50-inlier threshold (50 vs 44) —
  cross-node decode noise, not a pipeline difference. **Use 3**, matching the
  environment the published rows were produced in, and note the ±1 sensitivity
  where the instrument is described. The same fragility affects any borderline
  rollout in any model's relocation count.

---

## C. Additions

**C1. Table 14** (block at line **2525**) — add row:
`No Critic | +34.7 | 20 | 15/75`

**C2. Table 15** (rows ~**2665–2680**) — add row:
`Ours (No Critic) | 1.8 | 0.2`
Gate passed 8/8 (Default 2.3/1.4 exact, No AdaLN 32.4/6.0 vs 32.2/6.2, REAL 0.0/1.2 vs 0.0/1.1).

**C3. Table 3** (`tab:overall_legit`, rows at lines **~866–873**) — this is a
main-paper table already listing *Ours (Default)* (0, 73) and *Ours (Batch64)*
(1, 64). If A1 moves the critic ablation into the main body, add:
`Ours (No Critic) | 79 | 15`
(control fail 79.2%, legitimate 15.0%, rounded to the table's integer
convention). This is where the critic's necessity becomes visible in the main
paper: 79 vs Default's 0 on control fail.

**C4. Figure 26** (line **2561**) — swap
`Figures/appendix_validation/hf_distribution.png` →
regenerated version including No Critic
(`grids/eval/out/hf_distribution_nocritic.png`).

**C5. Figure 28** (line **2700**) — swap `Figures/stationary_wedges.png` →
regenerated 14-panel version including No Critic
(`grids/eval/out/stationary_wedges_pca_nc.png`).

---

## D. Caption / text fixes (independent of No Critic)

**D1. Line 2689 — Table 15 caption is wrong three ways.** It says *"Movement is
the median camera drift in pixels"*. It is a **mean**, of the **CoTracker–PCA
departure from the real continuation**, in **tanh-squashed action units**, not
pixels. Suggested: *"Movement is the mean per-scene departure of the
CoTracker–PCA readout from the real continuation, in squashed action units;
lower is better."*

**D2. Same caption** — state that `REAL = 0.0` is **zero by construction**
(real minus real), not a measurement.

**D3. Lines 2644–2656 — Animation method under-specified.**
`stationary_signs.py` is referenced nowhere in the paper. The operative details
are missing: top-3 PCA mode removal, an **8×8 spatial grid**, ≥3 tracks per
cell, residual > 2 px, coherence ratio ‖mean‖/mean(std) > 1.0 — and crucially
that the count is of **alive grid cells (0–64)**, not individual movers. Since
one large mover can light several adjacent cells, "number of movers" is a
proxy; say so.

**D4. Lines 2698–2708 — Fig. 28 caption.** "radius its magnitude" is unitless;
specify it is the same PCA departure from the real continuation, not pixels.

**D5. Table 13 caption (line ~2498) and Table 16 caption** — add the warning
that a low geometry/relocation score can reflect degenerate near-static
generation rather than better structure. **No AdaLN posts the best geometry in
the grid (11%)** and **No Critic the best relocation (2.7%)** for exactly that
reason.

*Nuance — do not over-lean on Movement.* No Critic's Movement of 1.8 is only
modestly below the 2.0–2.3 cluster (the real outliers are No AdaLN 32.4 and the
drifting externals), and on the stationary axis in isolation a low Movement is
nominally **good** — the viewpoint is held under the null command. The
degeneracy is only visible when Movement is read together with **Animation 0.2
vs real 1.2** (it freezes the world) and **directional control-fail 79.2%** (it
barely responds to any command). The three together say it holds the camera
still by holding *everything* still. Present it that way, not as "lowest
Movement = degenerate".

**D6. Redefine the stationary axes on the CoTracker–PCA readout; drop the
RANSAC framing.** The deployed stationary numbers do not use RANSAC at all:

- **Movement** is `mean(hypot(pc0−pc0_real, pc1−pc1_real))` from
  `stationary_cotracker_pca.py`. The RANSAC-affine pipeline
  (`stationary_cotracker.py`, `stationary_wedges.py`) is an **older, unused**
  variant — its `camera_motion` column is what the current caption describes,
  but that is not the number in Table 15.
- **Animation** is `life_pca` (global motion removed by dropping the top-3 PCA
  modes). A `life_ransac` column is computed alongside it and **is not used**.

So RANSAC is computed in both stationary instruments and discarded in both. It
is load-bearing only in **relocation** and **static detection**, where
"geometrically verified inliers" does real work.

Restating both axes on the PCA readout (i) matches what is computed,
(ii) **unifies the stationary axes with the controllability metric** under the
same frozen CoTracker→PCA teacher used everywhere else in the paper — a
genuine simplification, not just a correction — and (iii) removes a dependency
that does no work. Then note once that RANSAC survives only for place identity.

---

## E. Methodology notes to add

**E1. Conjuration threshold sensitivity.** The flag rule is `score > 0`, and
~0.4% of fleet rollouts (13 of 3328) sit within 0.10 of threshold, concentrated
almost entirely in minWM (13 of its 51 flags). Worst case minWM moves 51→38
(19.9%→14.8%); nothing else shifts by more than one rollout, and the
minWM ≫ everything ordering is unaffected.

**E2. Animation cross-source bitrate confound.** Our variants encode at
~7.5 Mbps, the externals at ~1.6 Mbps, and `life_pca` weakly tracks bitrate
(Spearman +0.185, p=0.003, n=256). This modestly suppresses the externals'
Animation, so "Matrix-Game and WorldPlay freeze the scene" is partly
confounded. Not fixable by re-encoding — their low bitrate is inherent to
native output.

**E3. Stationary sibling-coupling.** The stationary HF and relocation metrics
are *not* invariant to the set of models scored: adding No Critic to the
campaign shifts published rows (No AdaLN HF 21→23, Reloc 3→2) because the
sibling median / reference set changes. No Critic was therefore scored against
the **frozen published 13-model reference** so existing rows stay intact. A
future joint rerun would move those two No AdaLN cells by 1–2.

**E4. Cross-hardware generation limit (reproducibility appendix).**
Autoregressive generation is metric-safe for aggregate camera drift but not for
localised mover counts. Regenerating on different GPU hardware with identical
weights and seeds gives near-identical first chunks (mean |Δ| frame-mean 0.43)
that diverge ×30 by chunk 8 (13.08) through FP amplification along the AR
chain. Movement survives (per-scene max |Δ| 2.2, r=0.995); Animation does not
(r=0.50, r07 9→0, r17 19→1). Any rerun of the stationary tables must therefore
come off the same hardware.

---

## F. Verified — no change needed

- **Tables 13 and 14** reproduce exactly: all seven Table 14 rows
  (14/239, 15/237, 25/237, 45/229, 52/232, 101/236, 95/200) and Table 13's
  relocation rates (Default 10.9%, Batch16 25.8%, No AdaLN 28.0%).
- **Table 15** all eight published rows reproduce on the null-inverse campaign.
- **Table 2** No Critic Style 0, Geometry 0, Relocation 3, Conjuration 0 all
  confirmed (only HF moves — item B7).
- **Table 13 No Critic** geometry (2) and conjuration (0) confirmed as
  genuinely computed; only relocation and HF were wrong.

---

## G. Final assembled No Critic values

| where | value |
|---|---|
| Table 7 | feature-valid 240, active **75** |
| Table 13 row | Style **0** / Geom **2** / Reloc **3** / Conj **0** / HF **20** |
| Table 14 row | median B **+34.7**, B>150 **20%**, n/N **15/75** |
| Table 15 row | Movement **1.8**, Animation **0.2** |
| Table 16 | Geom 1.6%, Reloc **2.7%**, Style 0.0%, HF **20.0%**, Conj 0.4%, control fail **79.2%**, legitimate **15.0%**, active **75/240** |
| Table 2 / 17 row | Style 0 / Geom 0 / Reloc 3 (1/32) / Conj 0 / HF **3** (1/32) |

---

## H. Making HF and Relocation invariant (optional, for a cleaner metric)

Two axes are **fleet-coupled** — their value for one model depends on which
other models are in the set. Everything else (geometry, conjuration, style,
Movement, Animation, static/active, control-fail) is per-rollout and invariant.

| metric | why coupled | direction of the effect |
|---|---|---|
| **HF** | `B = −(d_blur − per-scene sibling median)` | **non-monotone** — the median can move either way (No AdaLN 21→23 when No Critic joined) |
| **Relocation** | `score = max inliers to any *other* member` | **monotone** — adding a member can only raise the max, so published relocation rates can only **fall** as the fleet grows (No AdaLN 3→2) |

Affected: Tables **1, 2, 3, 13, 14, 16, 17**. Invariant: Tables **7, 15**.

### H1. HF — TESTED AND REJECTED; keep the sibling median

The proposed invariant form replaced the per-scene sibling median with the real
footage as anchor: `B_real = −(d_blur − d_blur_real)`. Tested against human
labels (`human_tiers.csv` free-text notes over 84 ours-variant rollouts;
HF-positive = notes citing haze/blur/wash/mashy/soft, n=20, haze-only n=12).

*Circularity note:* `fleet_pixscan.csv`'s `label` column is **metric-derived**
(`classify()` thresholds `x_haze`/`x_median`/`x_blur` directly) and is therefore
unusable as ground truth for this test.

| GT cut | n / pos | B (deployed) | B_real (invariant) | diff |
|---|---|---|---|---|
| HF-broad | 84 / 20 | **0.730** | 0.643 | +0.087 |
| haze-only | 84 / 12 | **0.873** | 0.779 | +0.094 |
| HF-broad, excl. dirty | 78 / 17 | **0.781** | 0.674 | +0.107 |

`B_real` is ~0.09 lower on every GT definition, but the bootstrap 95% CI of the
difference is **[−0.029, +0.219]** (93% of resamples favour the deployed rule) —
it includes zero, so discrimination is **inconclusive**, blocked from
significance by thin ground truth rather than by parity.

**The flag-count behaviour is decisive.** Recalibrating `B_real` to match the
deployed total (B>150 → 341 flags; B_real>395 → 373) still cannot reproduce
Table 14 — it inflates the clean variants ~5× and compresses the worst,
inverting the variant ordering:

| variant | B>150 (published) | B_real>395 (recalibrated) |
|---|---|---|
| Default | 5 | 26 |
| Batch64 | 5 | 28 |
| pca4 | 5 | 29 |
| pca2 | 46 | 60 |
| Batch16 | 105 | 82 |
| No Action Tokens | 39 | 59 |
| No AdaLN | 136 | 89 |

**Mechanism (CORRECTED).** An earlier draft said the real anchor fails because
"generated video degrades more than real, so the common offset swamps the
signal". That explanation is **wrong**: a genuinely *common additive* offset
shifts every system equally, leaves AUC and rankings unchanged, and can be
absorbed by recalibrating the threshold. Since the ordering *did* invert after
total-matched recalibration, the real-to-generated gap must vary **by item**
and/or interact with system, i.e. it is **reference-domain mismatch**, not a
constant. Subtracting a well-matched control removes nuisance variance;
subtracting a poorly-correlated one *adds* it. The peer median is the better
control because it estimates item difficulty under the generative-model response
distribution, which real footage does not. Phrase it as blocking / nuisance
variance, never as "common-mode offset".

**Consequence for the caption, independent of invariance:** HF as deployed is a
**relative** measure — "degrades more than the fleet median does" — not an
absolute degradation rate. A reader will otherwise read "HF 6%" as "6% of
rollouts degrade". Say which it is.

### H2. Relocation — TESTED AND REJECTED; keep the sibling consensus

A fleet-invariant replacement was designed and tested against the 80-rollout
human-labelled set (`results_scene_cand.csv`, 27 positives). The candidate —
**chained place identity**: anchor at the last real context frame, step through
the generated span at chunk boundaries, score = min RANSAC-verified ORB inliers
over *consecutive* links — is fleet-invariant and fixes the turning penalty, but
is **significantly less sensitive** and is not a viable replacement.

| rule | accuracy | TP | FP | FN | TN | AUC |
|---|---|---|---|---|---|---|
| `consensus_inl < 50` (deployed, coupled) | **88.8%** | 23 | 5 | 4 | 48 | **0.864** |
| `chain_min < 150` (best invariant) | 70.0% | 13 | 10 | 14 | 43 | 0.757 |

AUC difference +0.108, bootstrap 95% CI **[+0.044, +0.180]**, 100% of 2000
resamples favour the deployed rule. Not a within-noise match.

**Why it fails is structural, not a tuning issue.** The false negatives
concentrate in the smooth-drift models (WorldPlay 0→4, Yume 1→5, which hold 13
of the 27 positives). A model that *wanders gradually* into an invented street
never breaks any single adjacent link, so the chain minimum stays high while the
endpoint is somewhere else entirely. Chained continuity measures **local
smoothness**; relocation is a **global endpoint property**. No threshold
recovers a signal the statistic does not contain, so a direction-normalised
variant would not help either.

**Corollary worth stating in the paper:** the sibling coupling is *load-bearing,
not incidental*. Siblings are other attempts at the same command from the same
seed, so they define what "the right place after this command" looks like —
external information no self-referential metric has. Invariance here would cost
real discriminative power.

Two secondary findings from the same test:
- `chain_min` and `gen_min` are **exactly tied** at every threshold, so the real
  seed anchor contributes nothing; a self-consistency metric needs no reference
  frame at all. (Interesting, but it does not rescue sensitivity.)
- The invariant does deliver on its motivation: per-direction medians for our
  variants are near-flat (forward 810, turns 745, worst turn 458) versus the
  deployed rule's forward 541 / turns 412 / worst turn **92** — close to the
  <50 flag. So the deployed rule's turning margin is tight but adequate;
  worth a sentence, not a redesign. (Only ~2 rollouts per direction — suggestive.)

Test script: `grids/eval/reloc_invariant_test.py`.

### H3. RECOMMENDED — declare a frozen reference panel, and ship its statistics

Both invariance attempts failed for the *same* reason: these two metrics need an
**external reference class**. Relocation needs other attempts at the same command
to define "the right place after this command" (self-consistency is structurally
blind to smooth drift); HF needs peers to isolate *relative* degradation
(anchoring to real collapses the between-variant spread).

Neither result shows the reference class must **vary**. Stability is therefore
achievable even though reference-freedom is not:

> **Protocol.** The 13 models of the original fleet are the permanent reference
> panel. Every model scored afterwards is measured against that frozen panel and
> never joins it.

This is already what was done ad hoc for No Critic; it needs promoting to a
stated protocol. It costs nothing, changes no published number, and makes future
additions exactly reproducible. Caveat to state: the panel is historical, so the
anchor ages as the models it contains are superseded.

**What the coupling costs today — three things:**

1. **Precision (small, measured).** Adding one model to the 13-model panel moved
   No AdaLN's stationary HF 21→23 and relocation 3→2 — 1–2 rollouts in 32.
   Relocation is **monotone** (published rates can only fall as the fleet grows);
   HF is **non-monotone**. The directional equivalent was not measured — say so
   rather than assume it matches.
2. **Reproducibility (currently total, fixable in one sentence).** The paper does
   not state which panel produced the numbers, so they cannot be reproduced even
   internally. Declaring the panel fixes this.
3. **Adoptability (the real cost).** Because both metrics are sibling-relative,
   an external researcher cannot compute our HF or relocation for their own model
   **without our entire fleet of 3,328 videos**. This is invisible in the current
   write-up and is a serious barrier to the instruments being reused.

### H4. Release the frozen reference statistics (fixes adoptability)

Ship the reference *statistics*, not the videos:

- **HF — trivial.** The per-scene sibling median is recoverable from
  `fleet_hf.csv` as `d_blur − x_blur`: **256 floats**. Publishing that one file
  makes HF fully reproducible by anyone, with no fleet access.
- **Relocation — heavier but tractable.** Publish the reference-panel ORB
  descriptors (subsampled if size demands) per scene, so a newcomer matches
  against descriptors rather than videos.

**BUILT AND VALIDATED** — artifacts in `grids/eval/out/`:

- `hf_reference_median_directional.csv` (256 scenes),
  `hf_reference_median_stationary.csv` (32 scenes) — per-scene
  `sibling_median_d_blur`, taken as the median of `d_blur` over the 13 panel
  models (cleaner than `d_blur − x_blur`, which carries stored rounding).
  *Validated:* recomputing `B` from the CSV alone reproduces `fleet_hf.csv`'s
  `B` to max |err| = 0.0000 and every Table 14 flag count exactly
  (14/239, 15/237, 25/237, 45/229, 52/232, 101/236, 95/200); stationary
  No AdaLN 21/32 likewise.
- `reloc_reference_pack_3000.npz` (358 MB) — per-scene ORB keypoints and
  descriptors for the 13 panel models' +6 s end frames plus the 4 real context
  frames. *Validated:* scoring No Critic against the pack alone, with no other
  model's video, reproduces same-node live scoring at r = 0.993 with 74/75
  flags identical. Builder/validator: `grids/eval/reloc_reference_pack.py`.
- **Ship the 3000-keypoint pack, not the 1000-keypoint subsample.** The smaller
  pack (146 MB) systematically undercounts inliers (mean −233), drops
  correlation to 0.909, and pushes borderline rollouts across the threshold; it
  would need its own recalibrated threshold rather than being a drop-in.
- **Release text must note the ±1-rollout threshold fragility** (see B3): a
  newcomer reproducing 1/75 where we report 2/75 is seeing metric noise at the
  50-inlier boundary, not a broken artifact.

This converts the coupling from a reproducibility hole into a documented,
shippable constant — a better outcome than the invariance originally sought,
since both tests showed invariance costs real discriminative power.

### H5. Manuscript note — state the limitation explicitly

Add to the quality-methodology section (and reference it from the Table 13/14
and Table 2 captions):

> Two of the five quality axes, **high-frequency degradation** and **scene
> relocation**, are *reference-relative*: each rollout is scored against the
> other models' rollouts of the same scene rather than in isolation. HF measures
> sharpness loss relative to the per-scene fleet median; relocation measures
> place identity against the best-matching sibling continuation. They therefore
> differ in kind from style, geometry and conjuration, which are per-rollout and
> absolute, and must be read as *comparative* statements about a model relative
> to the evaluated set, not as standalone properties.
>
> We tested reference-free formulations of both. For **relocation**, a chained
> self-consistency score (no external reference) lost discrimination
> significantly: AUC $0.864 \rightarrow 0.757$ against $27$ human labels,
> bootstrap $95\%$ CI of the difference $[+0.044, +0.180]$. Its failures
> concentrate on models that drift *gradually*, which never break local
> continuity while still ending elsewhere — a global property a local measure
> cannot recover. For **HF**, anchoring to the real footage instead of the fleet
> median gave a lower point estimate that is **not statistically separable**
> (AUC $0.730 \rightarrow 0.643$; bootstrap $95\%$ CI $[-0.029, +0.219]$,
> spanning zero, on only $20$ human labels); the decisive evidence there is
> instead distributional — after recalibrating the threshold to match the total
> flag count, the real-anchored variant still inflates the cleanest variants
> roughly fivefold and compresses the worst, inverting the variant ordering,
> because an absolute anchor removes the between-model normalisation the axis
> depends on.
>
> We therefore report both metrics against a **frozen reference panel** and
> release its per-scene statistics, and leave reference-free formulations to
> future work.

*Do not* state the two results symmetrically: relocation's loss is significant,
HF's is inconclusive on AUC and rests on the flag-count/ordering argument. Note
too that HF's AUC is ground-truth dependent and thin (0.730 on broad blur/haze,
n=20; 0.873 haze-only, n=12).

Also note in the HF caption that it is a **relative** measure — "degrades more
than the fleet median does", not an absolute degradation rate — since "HF 6%"
otherwise reads as "6% of rollouts degrade".


## I. Fleet-coupling vs hardware-coupling are different axes

These are often conflated; they are not the same set of metrics.

- **Fleet-coupling** (which models are scored): **HF and relocation only** — see H.
- **Hardware-coupling** (which GPU *generated* the video): affects *any* metric
  scored on generated video, because different hardware produces different
  pixels. What was measured is which metrics are **robust** to it:
  **Movement survives** (per-scene r = 0.995, max |Δ| 2.2) and
  **Animation does not** (r = 0.50; r07 9→0, r17 19→1).

So they overlap only incidentally. Note the hardware-robustness of HF,
relocation, geometry, conjuration and style was **not** tested — the generator
gate only covered Movement and Animation. Any claim about those five being
hardware-stable would be unsupported.


---

## J. Precedent, terminology and protocol (from the methodology review)

An external review of the peer-normalised metrics returned **defensible-with-
changes**. The frozen-panel plan (§H3) is not ad hoc — it matches established
practice. Cite these rather than describing the scheme in home-made language:

- **Blocking / randomized block designs** (NIST) — matched inputs are blocks,
  systems are treatments; blocking removes item-difficulty nuisance variance.
  This is the correct framing for *why* peer normalisation helps.
- **Fixed-item IRT calibration / common-item equating** (ETS) — item parameters
  estimated on a calibration sample, then frozen so later systems are placed on
  the existing scale. Operationally identical to §H3.
- **fRMA (frozen RMA)** — invented for exactly our problem: cohort-dependent
  normalisation is powerful but makes independently-processed cohorts
  incomparable; the fix is to estimate reference effects once and freeze them.
  The closest analogue to what we propose.
- **Tukey two-way / median polish**, **crossed mixed-effects models** — the
  principled form of what our median approximates.
- **Demšar within-dataset ranks**, **Bradley–Terry / Elo** — precedent that
  deliberately pool-relative comparison is legitimate *when the construct is
  declared relative*.

**J1. Terminology.** Describe the scheme as **peer-referenced, blocked
evaluation with frozen calibration** (psychometrics: *norm-referenced*, as
opposed to *criterion-referenced*). Important: a column labelled "% of rollouts
that relocated" reads as criterion-referenced; if the decision boundary is
panel-relative, either rename it (e.g. *peer-referenced relocation risk*) or
justify the threshold against the human labels so it carries an absolute
meaning.

**J2. Do not call HF "leave-one-out".** `fleet_hf.py` computes
`groupby("scene").d_blur.transform(lambda s: s - s.median())`, so the panel
median **includes the evaluated system**; relocation *does* exclude it. Either
make them consistent or state the difference. Measured impact of switching HF to
a true leave-one-out median: ±6 rollouts of ~238 per variant (pca8 14→18,
Batch64 15→17, pca4 25→19, noatok 45→46, pca2 52→51, Batch16 101→102,
No AdaLN 95→94), no systematic direction, median |ΔB| ≈ 14 against a threshold
of 150. Ordering is preserved; the cleanest three shuffle within their cluster.
**Recommendation: disclose, do not re-run.**

**J3. Version the calibration.** Name it (`HF-Cal-v1`, `Reloc-Cal-v1`) and never
silently update it. If a v2 is ever needed, keep overlap anchors and publish a
v1↔v2 linking analysis rather than overwriting historical scores.

**J4. Declare that the metrics are transductive.** They use other systems'
outputs at evaluation time — side information a stand-alone no-reference metric
does not have. Not leakage if declared; present it as a design feature.

**J5. Diagnostics worth running (cheap — the 13×256 table already exists).**
- **System × item interaction (DIF).** The two-way model assumes "hard item" is
  common across systems. If model families have different failure modes, a
  single item difficulty is not invariant. This is the key validity check before
  claiming the frozen item effects generalise.
- **Leave-one-family-out sensitivity** on the reference panel, not just
  leave-one-model-out (anchor-selection effects do not vanish with sample size).
- **Paired bootstrap over the 256 inputs**, preserving pairing across systems.

**J6. Estimator upgrade (future work, changes all numbers).** Replace the median
with a robust two-way calibration `d_si = μ + α_s + β_i + ε_si`, freeze `μ̂` and
`β̂_i` **but not `α̂_s`** (system effect is the quantity being measured, not
nuisance), and score future systems as `B = −[d − (μ̂ + β̂_i)]`. Gives a clean
estimand, uncertainty, and non-additivity diagnostics. Not a submission-time
change.

**J7. Relocation — harden the reference set (future work).** Replace "max over
any peer" with a **frozen consensus anchor bank**: build the per-scene geometric
agreement graph, keep clusters supported by multiple independently-developed
systems and/or the real reference, and require **top-k support rather than a
pure max** so a single erroneous anchor cannot validate a candidate. The pure
max is the manipulable form — one peer suffices to validate another, and
correlated failures across similar architectures are consensus without truth.

**J8. Strongest objections to pre-empt.** Norm-relative truth (if all 13 fail,
the median calls failure normal); correlated peer errors (systems are not
independent annotators); system×item interaction; benchmark aging after
freezing (anchors have a validity lifetime — psychometrics calls this item
parameter drift).

**J9. Release catch — scalars are enough for HF, not for relocation.** Publishing
per-item reference *statistics* lets an outsider score HF (`β̂_i` is a number).
It does **not** let anyone score relocation: you cannot RANSAC a frame against a
scalar. The shipped `reloc_reference_pack_3000.npz` (ORB keypoints +
descriptors, reproduces live same-node scoring at r=0.993, 74/75 flags
identical) **is** the metric definition, not a convenience artifact, and the
release text must say so — together with the exact verification procedure
(ORB config, RANSAC threshold, inlier count) and the ±1-rollout threshold
fragility already noted in §H4.

**J10. Triage — what actually has to happen before submission.**

*Free (wording only, no numbers change):* J1 terminology; the corrected §H1
mechanism; J2 self-inclusion disclosure; J3 versioning; J4 transductive
declaration; J6/J7 stated as future work in the limitations paragraph alongside
the reference-free note from §H5.

*Cheap (existing 13x256 table, no new generation):* the §J5 diagnostics. The
system x item interaction test is the load-bearing one — it decides whether
"hard scene" is a property common to all systems (frozen `β̂_i` generalises) or
whether model families have disjoint failure modes (a single item difficulty is
not invariant and the whole calibration is on sand). Run it first.

*Not before submission:* J6 (robust two-way estimator) and J7 (consensus anchor
bank + top-k). Both change every published number.

---

## K. MEASURED: system x item interaction and panel composition (run 2026-08-20)

The §J5 diagnostic, run on the complete 13x256 `fleet_hf.csv` table
(`scratchpad/dif.py`). Three results, one of them serious.

**K1. Blocking is justified — item difficulty is the largest systematic term.**
Least-squares two-way partition of `d_blur`: **item (scene) 27.9%**, system
12.5%, residual 59.6%. Scene difficulty explains more than twice the variance
of the system effect we are trying to measure. Removing it is not a
convenience; an unblocked absolute score is dominated by which scene you drew.
This is the number to quote when justifying peer normalisation.

**K2. Item effects are moderately reliable, not weak.** Split-half reliability
of `beta_i` over 500 random splits of the panel: Spearman rho = 0.653
(95% range 0.468-0.745) at 6-7 systems per half, giving a Spearman-Brown
full-panel reliability of **0.790**. Tukey's 1-df non-additivity test is
significant (F(1,3059)=9.51, p=0.002) but accounts for only **0.31%** of the
residual — with 3,328 cells significance is expected; the effect is negligible.
Additivity is a reasonable model.

**K3. SERIOUS — family DIF, and our variants hold 7 of 13 panel seats.**
Item difficulty estimated from our 7 variants vs from the 6 external baselines
correlates at only **Spearman rho = 0.383** (Pearson 0.494). That is *below the
2.5th percentile of random within-panel splits* (0.468), so "hard scene" is
partly family-specific, not a universal constant. Leave-one-family-out confirms
our family dominates the flat median: dropping the externals leaves
rho = 0.858 against the full-panel beta, dropping ours leaves only 0.697.

The consequence is a **self-favouring bias**. Our 7 near-identical variants
(same base model, small ablations) are 7 highly-correlated votes on what
"normal" degradation looks like, so they collectively define the reference they
are then scored against. Re-weighting to **one vote per family** (median over
families of per-family medians; rho = 0.816 vs the flat median, no systematic
shift, IQR of per-scene shift 41.2) changes the flagged counts as follows:

| system | flat median (deployed) | family-balanced | delta |
|---|---|---|---|
| Ours Default (pca8) | 14 | 40 | +26 |
| Ours Batch64 | 15 | 33 | +18 |
| Ours pca4 | 25 | 53 | +28 |
| Ours No Action Tokens | 45 | 70 | +25 |
| Ours pca2 | 52 | 71 | +19 |
| Ours Batch16 | 101 | 112 | +11 |
| Ours No AdaLN | 95 | 113 | +18 |
| astra | 18 | 19 | +1 |
| matrixgame | 16 | 16 | 0 |
| minWM | 3 | 3 | 0 |
| worldcam | 83 | 90 | +7 |
| worldplay | 23 | 26 | +3 |
| yume | 25 | 27 | +2 |

Every one of our variants moves by +11 to +28; every external baseline moves by
0 to +7. **What survives and what does not:**

- **SURVIVES — all ablation claims.** The within-family ordering is essentially
  unchanged (only the adjacent pairs pca8/Batch64 and Batch16/No AdaLN swap).
  "Reduced PCA supervision increases high-frequency degradation" (line ~783)
  and every Table 14 comparison *among our variants* is robust to the
  re-weighting. This is the paper's actual contribution and it is safe.
- **DOES NOT SURVIVE — the absolute rate and cross-family HF comparison.**
  Default's headline "$6\%$ high-frequency degradation" (line 746) becomes
  ~17% under a defensible alternative weighting. Worse, under the flat median
  Default (14) beats astra (18), matrixgame (16), worldplay (23) and yume (25);
  under family balancing it loses to all four. Any reading of Table 14 as
  "we degrade less than the baselines" is an artifact of panel composition.

**Required manuscript action.** The existing hedge at lines 746-749 ("trained
on FrodoBots footage while the baselines are not... robustness on this
deployment distribution rather than a domain-neutral ranking") is good but
covers a *different* mechanism (training distribution). Add a second, explicit
sentence to the same paragraph and to the instrument definition (Supplementary
~line 1105): the reference median is computed over a panel in which seven of
thirteen members are variants of a single model, so the high-frequency rate is
**calibrated to our own family** and cross-family absolute comparisons should
not be read as a ranking. State the measured sensitivity (rho = 0.383 across
families vs a 0.468 within-panel chance floor; our rates roughly double under
one-vote-per-family) rather than hedging vaguely. Present it alongside §H5's
reference-free-metrics-as-future-work note.

**Do not re-run the numbers.** Family balancing is defensible but not uniquely
correct, and switching would change every published HF value at submission
time. Report the deployed flat-median numbers, disclose the sensitivity, and
fold the balanced variant into the §J6 future-work estimator.

**K4. MEASURED: relocation is more family-biased than HF** (local agent,
2026-08-20; artifact `out/reloc_family_bias.csv`, per-rollout `c_any`,
`max_peer`, `max_peer_fam`, `samefam_max`, `c_extreal`, `c_onerep`).
Full per-scene pairwise RANSAC-verified ORB inlier matrix recomputed from the
frozen pack (13 systems + 4 real references x 256 scenes; no video decode).

*Who supplies the max, for our systems:* **same-family 80.4%**, real 17.3%, all
externals combined ~2%. Per variant the max is same-family 77-94% of the time
(Default 94%, pca4 92%). The exception is No AdaLN at 34% — it is degraded
enough that it matches real/externals instead. Externals are structurally 0%
same-family, each being its own family.

*Flag rate (relocated, <50) under three panels:*

| system | n | (a) any peer | (b) 1 rep/family | (c) externals+real |
|---|---|---|---|---|
| Default | 239 | 10.0% | 15.1% | **20.1%** |
| Batch64 | 237 | 8.0% | 11.0% | 17.3% |
| pca4 | 237 | 13.5% | 17.7% | 28.3% |
| pca2 | 232 | 15.1% | 19.0% | 25.4% |
| Batch16 | 236 | 25.0% | 32.2% | 41.5% |
| No Action Tokens | 229 | 13.5% | 19.7% | 30.6% |
| No AdaLN | 200 | 27.0% | 29.0% | 30.0% |
| minWM | 187 | 3.7% | 3.7% | 3.7% |
| Matrix-Game | 240 | 95.4% | 95.8% | 95.8% |
| WorldPlay | 238 | 77.3% | 78.2% | 78.6% |
| WorldCam | 216 | 87.5% | 87.5% | 87.5% |
| Astra | 215 | 67.0% | 68.4% | 68.8% |
| Yume | 237 | 58.2% | 59.1% | 59.9% |

Removing same-family peers roughly **doubles every one of our rates** while
moving every external by **at most one rollout**. That asymmetry is the control:
it rules out a threshold artifact, because a mis-set threshold would move
everyone. No AdaLN is the lone ours exception (27.0->30.0) precisely because it
does not rely on same-family propping.

*Direct propping measure:* **189/1356 = 14%** of our not-relocated rollouts
survive *only* because another of our own variants matched them (Batch16 22%,
No Action Tokens 20%, pca4 17%, pca2 12%, Default 11%, Batch64 10%,
No AdaLN 4%). So the metric is not wholly family-internal — but the bias is
material and strictly one-directional, and externals cannot access it at all.

**Same verdict as K3, structurally stronger.** Relocation is a `max` over peers
and therefore monotone in panel size, so this bias can only ever favour the
family with the most seats. Ordering among our variants survives; the absolute
rate and the cross-family comparison do not.

**K5. Manuscript text to add** (Table 14 caption and/or the relocation
instrument definition, Supplementary ~line 1105 region):

> Relocation is a peer-referenced statistic: a rollout is scored by its best
> geometric agreement with any other panel member. Because seven of the
> thirteen panel members are variants of a single model while each baseline is
> architecturally unique, our variants benefit from same-family agreement that
> the baselines cannot access -- the best-matching peer is another of our own
> variants for 80.4\% of our rollouts. Scoring only against the external
> baselines and the real references roughly doubles our relocation rates
> (Default 10.0\% to 20.1\%, pca4 13.5\% to 28.3\%) while moving every baseline
> by at most one rollout; the relative ordering among our variants is
> unaffected. Cross-family relocation rates are therefore panel-dependent and
> should not be read as a domain-neutral ranking.

The parallel HF sentence is in §K3. Both belong next to the §H5 note that
reference-free formulations are left to future work.

**K6. OPEN — reconcile 10.0% against the published 11%.** The agent's
"any peer" Default rate is 10.0% (n=239); Table 14 as published says 11%. The
13-system panel excludes No Critic in both, so the likely cause is the frozen
pack reproducing live scoring at r=0.993 (74/75 flags identical), i.e. ~2
rollouts. **Must be settled before §K5 goes in**, because the caption quotes
"10.0% to 20.1%" and the base has to match the table. If the pack is the cause,
either quote the live-scored base (11% -> ~22%) or state that the debiased
figures are pack-scored.
