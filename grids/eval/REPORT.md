# Grid Video Eval Testbench — Report

**Goal:** reference-free automated metrics that reproduce the human-eye judgments in
`GRID LAYOUT.txt` for the 7 ablation variants (pca8/pca4/pca2/16node/4node/noatok/noadaln),
plus a start-vs-end style-shift detector (also intended for external models that
deliberately restyle videos into games).

**Data:** 9 human-annotated grids (r01/02/04/05/06/10 BR + r15_R/r17_BL/r18_BR,
the last three annotated as a held-out round) split into 63 tile videos
(832×448 after cropping the label strip, 108 frames @16fps) in `tiles/`.
Human notes were encoded into `gt.json` as per-variant `quality` (0-10),
`style_shift` (0/1), `action_follow` (0/1).

## What was run (all reference-free, all local on the 5090)

| Family | Metrics |
|---|---|
| NR-IQA (pyiqa, 16 frames/video) | musiq, clipiqa(+), brisque, niqe, maniqa, topiq_nr, liqe, arniqa, hyperiqa, dbcnn, paq2piq, nima, cnniqa, tres, unique, qualiclip+ |
| NR-VQA | DOVER (aesthetic/technical/overall) |
| Temporal | RAFT warping error (occlusion-masked, EvalCrafter-style), flow magnitude, CLIP-Temp, DINOv2 consistency, frame-diff/shimmer/block-flicker |
| Style start-vs-end (first 16 vs last 16 frames) | VGG-Gram distance, CLIP drift, CSD (contrastive style descriptor) drift, DINOv2 drift, MUSIQ-drift (Rolling-Forcing ΔDriftQuality), MS-SWD color drift, Lab/saturation/contrast/dark-fraction/dark-channel-haze/sharpness deltas, HF/LF spectral-ratio drift |
| VLM judges | VideoScore-v1.1 (Mantis-8B, 48 frames, 5 dims), Qwen2.5-VL-7B rubric judge (12 frames, 7 artifact dims) |

Correlation vs GT: `analyze.py` → `corr_quality.csv`, `corr_style.csv`, `merged_scores.csv`.
Composites: `composite.py` (ridge, leave-one-grid-out CV so numbers are honest).

## Headline results

### 1. Human-eye quality surrogate

**No single off-the-shelf metric works** (best singles: CSD style-drift ρ≈-0.38
within-grid, NIQE ≈+0.37, UNIQUE ≈+0.33; DOVER ≈-0.13; VideoScore ≈+0.18;
Qwen rubric ≈-0.21). This matches the literature: zero-shot photographic-quality
models are blind to generative artifacts (melt/warp/uncanny structure).

**A 7-feature ridge composite works well** — LOGO-CV Spearman **+0.73 global /
+0.74 mean within-grid**, and **89.3% pairwise accuracy on clear human calls**
(pairs the human separated by ≥1.5 quality points). Round-2 additions after the
misalignment autopsy: UNIQUE (whole-video) and dark-channel haze. Also computed
in round 2 but not selected: windowed start/end/drift versions of 8 IQA metrics
(unique_w_drift is the best *single* feature found, within-grid ρ=0.46),
stride-1 warp-error variants incl. p95 worst-pair (warp1_p95_max, ρ=0.31), and
two AI-image-detector "GAN critics" (their fake-prob rises over rollouts and
tracks blur failures, but adds nothing over the composite).

Original 5 features (weights refit with all 7):

| feature (grid-median-relative) | weight | intuition |
|---|---|---|
| CSD style drift (start→end) | −1.44 | artifact accumulation shows up as style drift from the in-distribution start window |
| DINOv2 frame consistency | −1.09 | negative weight: "too static" rollouts (4node makes nothing, noadaln ignores actions) get docked |
| NIQE | +0.63 | complements learned metrics on texture statistics |
| dark-fraction drift | +0.57 | black-region failures (noatok) |
| RAFT warping error | −0.18 | melt/shimmer |

| UNIQUE | +0.49 | learned perceptual quality, complements NIQE |
| dark-channel haze | +0.17 | haze veil (4node failure mode) |

Key insight: because each rollout starts from real context frames, the start
window is an *implicit reference* — most predictive signals are start-vs-end
drift measures, not whole-video quality scores.

### Where the composite still misaligns with the human eye (9/84 clear pairs)

Per-grid within-ρ: r01 +0.85, r02 +0.90, r04 +0.56, r05 +0.54, r06 +0.76, r10 +0.81.
The residual misses fall into three buckets:

1. **noadaln context-dependence (4/9)**: the human's noadaln score mixes axes —
   forgiven when "looks good but ignores the action" (r04: GT 7.0), slammed when
   "warping really bad" (r05: GT 3.0). A reference-free quality metric cannot see
   action compliance; this bucket needs the action-conditioned check
   (commanded action vs RAFT flow direction).
2. **Haze/holes vs blur trade-off (3/9)**: r10 4node ("hazy, holes, uncanny",
   GT 5.0) is ranked below pca4/pca2 ("very blurry, bad", GT 3.0-3.5); r05 pca2
   ("good", GT 8.0) initially under-ranked. The metrics penalize haze more, the
   human penalizes blur more; these are 1-2 rank-position judgment calls.
3. **Semantic hallucination (2/9)**: r10 pca8 vs 16node ("makes a building,
   8node better") and r04 noadaln vs noatok ("building strange, uncanny") —
   require knowing what *should* have been generated; invisible without
   references or a fine-tuned semantic judge.

Rank-target fitting and pure-within-grid greedy searches were tried and are
worse (0.65/0.60 within); the 7-feature quality-target model was the plateau for
n=42 labels.

### Round 3 (9 grids, 63 videos)

The user annotated r15_R/r17_BL/r18_BR. **Held-out test of the round-2 model**:
global ρ=+0.66, within +0.67 (r15 +0.82, r17 +0.38, r18 +0.80) — generalized as
promised; "4node good but hazy" caught in all 3 grids (4/4 style hits, 0 missed,
3 false alarms where the detector fires on mangling rather than haze).
Diagnosed blind spot: **localized sharp structural mangling** ("big messy
structure on the right") that frame means average away. Added a worst-region
patch battery (`run_patch_battery.py`, 4×2 regions, worst/spread of
NIQE/MUSIQ/CSD/Gram per region).

**Current production model** (10 features, recalibrated on 9 grids): base7 +
LIQE + `p_csd_drift_spread` (worst-region CSD drift minus median region —
the localized-mangling detector, selected by CV) + raw CSD drift.
LOGO-CV: **global +0.69, within +0.73, pairwise 85.0% (108/127)**;
per-grid ρ: r01 .67 / r02 .92 / r04 .63 / r05 .47 / r06 .76 / r10 .85 /
r15 .71 / r17 .73 / r18 .87. Style composite on 63 videos: **AUC 0.86,
balanced acc 0.83** at z>1.28 (threshold updated in style_stats.json).
r05 remains the worst grid (noadaln "warping really bad" + pca2 "good" both
misjudged). Persisting misses concentrate on: 4node-vs-blurry trade-offs,
noadaln context-dependence, and still-partial sensitivity to one-sided mangling
(r17 pca2 improved from rank 2 → mid-pack but not to your rank 6).

### 2. Style-shift detector (start vs end)

Best singles: **MUSIQ-drift AUC 0.83** (this is exactly Rolling-Forcing's
ΔDriftQuality, published on 832×480@16fps AR rollouts; CausVid=2.18 is the known
worst offender), grid-relative **VGG-Gram distance 0.82**, **MS-SWD 0.80**.

**Production composite** (in `score_video.py` / `style_stats.json`): sum of
z-scores of MUSIQ-drift + Gram-distance + MS-SWD → **AUC 0.83 raw / 0.89
grid-relative, balanced accuracy 0.79** at threshold z>1.77. All three components
are reference-free and content-robust, so the same checker transfers to external
restyling models: a deliberate game-restyle will push Gram+MS-SWD far past the
threshold while "honest" world models stay below it.

Fitted-feature alternative (clipiqa+/brisque/photometric/cnniqa) hits CV-AUC 0.98
but greedy selection on n=42 is optimistic; the 3-component composite is the
defensible choice.

### 3. Negative results (worth knowing)

- **DOVER, VideoScore-v1.1, zero-shot Qwen2.5-VL rubric judging**: all near
  chance on this GT. VideoScore saturates (all tiles 3.3-4.1/4.0); Qwen gives
  pca2 and pca4 identical scores. Consistent with Q-Bench-Video/TempGlitch
  findings that VLMs under-detect temporal/AIGC artifacts.
- Whole-video mean IQA scores are much weaker than start-vs-end *drift* of the
  same scores.
- `action_follow` (noadaln ignoring actions) is invisible to every reference-free
  metric tried; it needs action-conditioned evaluation (e.g., flow-direction vs
  commanded action — flow_mag was the only weak proxy).

## Round 4: VLM-as-judge investigation (12 grids, 215 cross-tier pairs)

Built per the ChatGPT lit-review design (`vlm_judge.py`, `qwen3_judge.py`,
`vlm_eval.py`; results in `vlm_judge_results.jsonl`). Metric: cross-tier
pairwise accuracy within grid (ties = 0.5), macro over 12 grids.

**Standings:**

| judge | acc |
|---|---|
| MiniCPM-V-4.5, sequential frame packs (any prompt) | ~54% (89% "answer B" position bias) |
| MiniCPM-V-4.5, side-by-side composites, direct rules | 63.8% |
| MiniCPM-V-4.5, priority rules v2 | 50% (verdict collapses to 100% "B" — token-level prompt fragility) |
| Qwen3-VL-8B, side-by-side + LOGIT debiasing (P(A)/P(B) averaged over both orders) | 62.7% |
| Qwen3-VL-8B logit + priority rules v2 + change-map evidence | **67.4% (best VLM)** |
| VLM ensembles (any combination) | ≤72.4% |
| **pixel composite alone (held-out)** | **78.3%** |
| composite + best VLM (any weight) | 78.5% max (noise) |

**Findings:**
1. Sequential pairwise video prompting is unusable: 89% positional verdicts.
   Side-by-side per-timestep composites (A left / B right) fix it.
2. Discrete verdict tokens are fragile (a rules edit flipped MiniCPM to 100%
   "B"); logit expectation over {A,B,T} with both-order averaging is robust to
   the same edits and is the right substrate for prompt iteration.
3. Change-map (|frame-diff|) evidence images + priority-ordered rules
   ("structural integrity dominates; sharpness does not excuse mangling;
   haze is minor") gave the best VLM: Qwen3-VL-8B at 67.4%.
4. Zero-shot 8B VLM judges plateau ~64-67% on the plausibility criterion and
   add nothing to the pixel composite in ensembles. The composite remains the
   production qualifier.

Best-VLM recipe (if a judge is wanted anyway): Qwen3-VL-8B-Instruct, 24
side-by-side frames + 3 change-map composites, `rules_v2` prompt, logits on the
forced `{"overall": "` continuation, p(x) = ½[P(A|x left) + P(B|x right)].

**Round 5-6 (plateau confirmation):**
- rules_v3 (object-inspection + protect-soft-videos rules) at 640px hi-res:
  66.3% — no gain; per-grid profile bimodal (6 grids 82-87%, 6 grids 40-55%),
  and the hard-grid failures are stable across all prompt variants.
- Prompt-margin ensembles over 4 logit runs: 65-66% (errors correlated on hard
  grids — capability ceiling, not wording).
- Tie-threshold (eps) sweep: 0.05 already optimal.
- Absolute Q-Align-style rating-token expectation (84 single-video calls,
  no positional bias possible): 61.8%; best 3-way combo with pairwise +
  composite: 75.6% — still below composite alone.

**Round 9 (third judge + relational):** InternVL3.5-8B-HF, same recipe
(rules_v2, sxs+diff, logit debias): 65.9% eps-tied / 68.4% margin-sign on 12
grids — marginally the best single VLM (Qwen3 67.4/67.8). Two-model logit
ensemble: 68.9% (+1 pt; 54% of errors are SHARED between the two models —
task-level ceiling confirmed). Note: InternVL OOMs on some grids at 32GB
(dynamic tiling); fix with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
7-way relational mosaic ranking (all variants in one labeled panel image,
2 permutations): 51.0% = chance — scores follow panel position; panels too
small at 7-up. Pairwise remains the only viable VLM protocol.

**Round 7 (detection framing, per user direction):** haze declared a don't-care
in all prompts (handled by dark-channel + style-z detectors).
- `real_check` ("does anything seem off / is this the real world", P(real)
  logits): pairwise 61.7%, global rho +0.33 / within +0.37 — best absolute VLM
  signal found, but composite+real_check (77.3-77.7%) < composite alone.
- `find_mistake` (find worst mistake, rate severity): severity saturates (all
  4-5); useless as a score BUT the descriptions accurately localize real
  artifacts (r14 melted van, deformed backpacker, r05 melting scooters).
- Pairwise comparison of the mistake descriptions (text-only, logit-scored):
  46.6% — the model narrates a severe mistake for every video including 9/10s.
- Pixel-stat tracker (`pixel_stats.py`): per-frame mean/std/median curves +
  0->4s deltas for every tile (`pixstat_curves.csv`). Best signal:
  `px_std_d4s` (contrast collapse over 4s), within rho +0.34, style AUC 0.68.

**Round 8 (reference-anchored calibration, per user design):** first-second
frames labeled REFERENCE ("real, by definition 10/10"), continuation rated as
deviation, severity from three behavioral yes/no logit probes (casual-viewer
notice / major-object break / persistence).
- `ref_anchor` (optional find): calibrates the TOP perfectly (every user 9-10
  scores 10.0, "none found") but says "none found" for most bad videos too —
  53.7% pairwise. `ref_anchor2` (forced find): finds real artifacts everywhere
  including in 9/10s — probes saturate to Yes — 59.8%. The model has no middle
  register between "nothing" and "catastrophe" in any framing.
- avg(ra1, ra2): 61.0% alone; hand-weighted blend with composite reached 80.0%
  on 12 grids, but this does NOT survive honest validation: on identical grids
  under LOGO-CV, composite alone = 81.7% pairwise vs composite+vlm = 81.1%.

**Conclusion: zero-shot VLM judging plateaus at ~67% (pairwise) on this
criterion; detection-native and reference-anchored framings don't change that —
the model can DESCRIBE artifacts (well!) but cannot CALIBRATE severity in any
framing tried, and no VLM signal survives honest CV on top of the composite.
The calibrated pixel composite remains the automatic qualifier.** Remaining escalations if ever needed: 30B-class judge (uncertain,
quantization-fiddly on 32GB) or fine-tuning on the tier labels (the only
data-supported route past ~78%).

## Round 10: external-baseline campaign (6 models x 256 scenes + ours)

Timing convention (user-specified): shared REFERENCE = last real frame of our
0.56s context (frame 8 of our tile); generation window = each model's own
real-context frame count (astra 4 / matrixgame 1 / minwm 13 / worldcam 65 /
worldplay 1 / yume 1 / ours 9) to +6s of generation; midpoint = half of endpoint.

**Instruments and validated accuracies:**
- Uncanny probe V2-strict ("casual viewer notices in 1s, ignore small artifacts",
  logit read): destroyed-vs-ours AUC 0.86 on 10 scenes; clean externals at 0.00.
- Scene relocation: RANSAC-inlier place identity vs sibling consensus
  (thr < 43 inliers), 88.8% acc / 86.2% LOSO vs 80 hand-labeled videos.
  Known biases: matrixgame restyle destroys keypoints (over-flagged); feature-poor
  scenes set a ~10-17% false-positive floor (minwm 9.4% ~ floor).
- Staticness: first-gen vs 6s-horizon frame NCC>0.9; matrixgame freezes 9%.
- VLM sequence probes are blind to yume-style coherent wandering (p 0.03 vs true
  0.6); geometry sees it (56 vs 300+ inliers). Failed scene candidates: reworded
  probes, 3-way classification, retrieval margin, spatial patch consistency.

**User-graded blind test (9 unseen videos, predictions stated before viewing):**
8/9 fully correct (worldplay relocation+wreck, worldcam static+style-drift,
worldplay clean rollout, yume relocated/borderline/clean trio, minwm borderline
+clean). 1 miss: minwm r21_R flagged RELOCATED but actually same-place — the
pre-registered turn-scene false-alarm mode; the video instead spawns a bright
yellow car mid-frame (novel-content class: same place, not uncanny — the
novelty probe's territory, graded "acceptable deviation" by the user).
Static-by-inliers replaced the broken NCC checker (user-caught: worldcam r27_F
static at NCC 0.72): threshold 600 inliers first-gen-vs-horizon; finds
astra/minwm ~21% static at 4s window — the degenerate do-nothing solution
partially explains minwm's clean record.

**Regeneration (yume full, matrixgame BL/BR):** yume relocation 74->60% (scene
infidelity persists = model property); matrixgame BL/BR 12.5->100% — the old
"good" diagonals were the buggy pathway barely moving; fixed, they wreck like
the rest, lifting matrixgame to 95% fleet relocation (worst).

**FINAL fleet scorecard (2,048 videos, 6s convention, regenerated data), also
in final_scorecard.csv:**

| model | style | novel | uncanny | reloc% | static% |
|---|---|---|---|---|---|
| worldplay | .67 | .44 | .79 | 80 | 1 |
| matrixgame | .74 | .45 | .71 | 95 | 0 |
| worldcam | .83 | .68 | .69 | 80 | 9 |
| astra | .27 | .36 | .50 | 62 | 10 |
| yume | .07 | .35 | .08 | 60 | 1 |
| minwm | .00 | .51 | .01 | 9 | 21 |
| ours_pca8 | .04 | .22 | .17 | 17 | 1 |
| ours_16node | .06 | .25 | .25 | 17 | 2 |

## Round 11: four-agent verification of the final scorecard

**Agent 1 (validity):** every scorecard cell re-derives exactly; regen propagation
verified; bootstrap CIs — all load-bearing differences significant. Corrections:
deployed reloc config (thr<43) validates at 87.5% (88.8% belongs to the round-3
instrument at thr 26 — scene_reloc_threshold.txt still says 26); pca8 vs 16node
significant ONLY on p_uncanny; minwm static magnitude threshold-sensitive;
probe means are saturated fire-rates (40-60% of values at exactly 0/1), not
probabilities; scene bootstrap slightly optimistic (directions share runs).

**Agent 2 (ranges):** consensus_inl is bimodal with a natural valley 43-150 —
recommended T=50, stable in [43,60] (<6pp any model). Texture floor located
precisely: grids r11+r21 = all 16 feature-poor scenes (<500 ORB kp vs median
4500); they cause 36% of ours' flags; ORB-gate (>=1000 kp) before rates.
Static: T 700-800 stable for all EXCEPT minwm (no valley — report its
distribution, not a flag; >600 means "low camera motion", not frozen). p_novel
unusable for model ranking (no separation); x_blur sign inverted in any
consumer assuming high=hazy. Turn scenes: FP 24% vs 5% straight — report
direction groups separately, never headline turn gaps <7pp.

**Agent 3 (our-model deep-dive):** failures scene-driven (41/44 scenes shared
between variants) and directional (uncanny L 0.41 / BL 0.39 vs F 0.06-0.08 —
corruption where imagination is demanded). True ours reloc ~8-12% after floor
adjustment; genuine failures concentrate in runs r03+r26 (murk-dissolve
signature: high uncanny, zero novelty, MURK — model dissolves the scene until
geometry loses lock). Uncanny gap vs minwm real per-unit-content (0.33 vs 0.03);
yume's clean rate survivor-biased; minwm novelty action-blind (flat across
directions; spawned-car exemplar). Flagship = pca8 (only significant axis).
Fix-first: long-horizon lateral/backward synthesis (NOT capacity — saturated);
adaln load-bearing; action encoding saturates ~pca8; patch r03/r26 data.

**Agent 4 (visual audit, 28 fresh videos):** style probe 96% agreement;
static 93% (zero FPs, undercounts); reloc 89% (high-confidence <20 inliers
7/7 correct; both FPs in the 25-43 borderline band); uncanny 89% (9/9 high
flags correct; misses mild late-horizon melt); novel = "inserted object"
probe, blind to relocation-driven novelty. Clean-composite selector 8/8.

**CORRECTED HEADLINE TABLE (ORB-gated r11/r21 excluded, T=50, 240 scenes):**
reloc: matrixgame 95 / worldcam 80 / worldplay 79 / astra 60 / yume 60 /
ours 12.5-13.3 (straight 6.7, turn ~15) / minwm 4.2.
Flagship claim (repaired): "the only model that generates ACTION-CONDITIONED
novel content while staying in the scene; baselines either abandon the scene
(60-95%) or stay by refusing the task (minwm: static tail + action-blind
novelty)."

## How to use

```bash
cd /home/ashish/ARRWM/grids/eval
# score sibling variants of one scenario together (enables grid-relative features):
./venv/bin/python score_video.py tiles/r01_BR__*.mp4 --out scores.csv
# split new grids first:
./split_tiles.sh /path/to/rXX_DIR_grid.mp4 tiles
```

Outputs `quality_score` (~0-10, calibrated to the human scale) and
`style_shift_z` + `style_shift_flag` per video, plus raw components.

## Files

- `gt.json` — encoded human ground truth
- `run_iqa_battery.py`, `run_flow_battery.py`, `style_shift.py`, `style_shift2.py`, `run_videoscore.py`, `run_qwen_judge.py` — metric batteries (all append to `results_*.csv`)
- `analyze.py` — single-metric correlations; `composite.py` — LOGO-CV composites
- `score_video.py` + `quality_model.json` + `style_stats.json` — production scorer
- `venv/` — overlay env on flash (torch 2.8/cu128; transformers pinned 4.49 for Mantis+Qwen)
- `DOVER/`, `MS-SWD/` — cloned repos

## Caveats / next steps

- n=42 videos from 6 grids; the quality composite is CV-validated but would
  benefit from annotating more grids (even coarse 0-10 per tile) — the pipeline
  ingests new rows by just editing `gt.json` and re-running.
- GT axes are entangled (quality notes include semantic judgments like
  "hallucinated a blue tarp" that no reference-free metric can see).
- If a stronger learned judge is wanted later: fine-tune a small VLM on your own
  annotations (WorldModelBench showed a 2B judge beating GPT-4o after fine-tuning),
  or try Q-Eval-Score (Qwen2-VL-7B, trained on 40K generated videos, CC-BY-NC).
