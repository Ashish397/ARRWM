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

**Mechanism (structural, not calibration).** The sibling median measures
degradation *relative to peers*; the real anchor measures it in *absolute*
terms. Generated video degrades more than real footage broadly, so anchoring to
real compresses the between-variant spread — exactly the signal Table 14 exists
to display. No threshold recovers it.

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
> absolute, and they must be interpreted as *comparative* statements about a
> model relative to the evaluated set, not as standalone properties. We tested
> reference-free formulations of both — chained self-consistency for relocation,
> a real-footage anchor for HF — and both lost discriminative power
> (relocation: AUC 0.864 → 0.757, bootstrap 95% CI [+0.044, +0.180]; HF:
> AUC 0.730 → 0.643 with a threshold recalibration that inverts the variant
> ordering). The reference class appears to carry information that a
> self-referential measure cannot recover. We therefore report both metrics
> against a **frozen reference panel** and release its per-scene statistics, and
> leave reference-free formulations to future work.

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
