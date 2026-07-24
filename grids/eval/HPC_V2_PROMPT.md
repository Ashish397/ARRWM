# Prompt for HPC instance — Eval Testbench V2

Copy the whole `grids/eval/` directory to the HPC alongside this file, then give the
following prompt to the agent/engineer there.

---

## PROMPT (paste from here)

You are building **V2 of a reference-free video evaluation testbench** for
autoregressive driving world models. V1 was built and calibrated on a local
RTX 5090 against 12 human-annotated scenes (84 videos, 7 ablation variants each,
832×448 @16fps, ~7s, first ~1s = real context frames). V1 lives in the `eval/`
directory you have been given; its full history is in `eval/REPORT.md`. Your job:
(1) reproduce V1's metrics at scale, (2) upgrade the judge tier using HPC-class
GPUs, and (3) score EXTERNAL world models' videos — including models that
deliberately or accidentally restyle the video (photoreal → game-like) — on two
axes: content plausibility and style shift.

### What V1 established — do not re-litigate these

1. **The start of each rollout is an implicit reference.** The first second is
   real context, so start-vs-end *drift* measures beat whole-video quality
   scores everywhere. Take baselines at t=1s (never frame 0; warm-up artefacts).
2. **Group-relative measurement is essential.** All videos generated from the
   same scene+actions form a sibling group; subtracting the group median from
   any metric cancels legitimate content change and source-video defects
   (~12% of source clips have water/smudge on the lens — detect via Laplacian
   variance at t=1s below the fleet 12th percentile, and exclude those from
   degradation claims). For external models, the sibling group = all models'
   outputs on the same scene.
3. **Haze metric (KEEP AS-IS, it is signed off):** sibling-relative Laplacian
   sharpness loss between t=1s and end of generation. AUC 0.86 vs human haze
   annotations; a 3-signal z-sum with dark-channel drift and contrast loss
   reaches 0.89 (`fleet_pixscan.py`, `haze_baseline.py`). CPU-only, trivially
   parallel — run it on everything first.
4. **Quality composite:** 10-feature ridge (`quality_model.json`, features in
   `best_combo.json`, extraction in `score_video.py`): CSD style-embedding
   drift, dark-fraction drift, DINOv2 consistency, NIQE, RAFT warping error,
   UNIQUE, dark-channel, LIQE, worst-region patch CSD spread — all
   group-relative. Cross-validated agreement with human tier rankings:
   Spearman ~0.73, 78–82% cross-tier pairwise accuracy. This beat every
   zero-shot VLM and every VLM ensemble in V1.
5. **VLM judging protocol (the ONLY configuration that works):** pairwise,
   side-by-side composites (one image per timestep, A left / B right, 24
   frames + 3 |frame-diff| "change map" images), the priority-rules prompt
   (`rules_v2` in `vlm_judge.py`), and **logit scoring**: force the reply
   prefix `{"overall": "` and read P(A)/P(B) at the next token, averaged over
   both presentation orders: p(x) = ½[P(A|x left) + P(B|x right)].
   Best 8B results: InternVL3.5-8B 68.4%, Qwen3-VL-8B 67.8%, 2-model logit
   ensemble 68.9% (54% of their errors are shared).
6. **Known dead ends — do NOT retry at 8B:** sequential frame presentation
   (89% position bias); generated verdict tokens (prompt-fragile, collapses);
   7-way mosaic ranking (chance; scores follow panel position); triplet/Fano
   and quadruplet covering designs (54%/61% — panel dilution); absolute 1–5
   rating (saturates); optional-find framings ("none found" on bad videos);
   forced-find framings (catastrophe on good videos); zero-shot style-shift
   perception questions on subtle drift (AUC 0.50 = blind).
7. **The human criterion** (from the annotator, encoded in `gt.json` notes):
   plausibility beats sharpness; haze/softness is minor; hallucinated content
   is fine if well-formed; structural corruption (melting, mangling,
   vehicle-parts-becoming-boxes) is the cardinal failure. 8B VLMs can
   *describe* these artifacts accurately but cannot *calibrate* severity —
   that is the wall V2 must break.

### V2 upgrades (in priority order)

1. **Scale the judge.** The single most promising unexhausted lever. Run the
   V1 pairwise-logit protocol unchanged with Qwen3-VL-30B/72B-class and
   InternVL3.5-38B-class models (bf16 on HPC GPUs — no quantization games).
   Evaluate on `gt.json` (12 grids, cross-tier pairwise accuracy, macro over
   grids) before trusting it. Success = clearly beating the 78–82% composite;
   68% → ~75% alone would already justify ensembling.
2. **Fine-tune a judge on the tier labels.** `gt.json` has 84 tier-labeled
   videos → ~600 cross-tier pairs ×2 orders. LoRA a video-capable VLM on the
   pairwise task with the rules_v2 prompt, evaluate leave-one-grid-out. V1
   evidence says the models see the artifacts; they lack the severity
   calibration — which is exactly what fine-tuning supplies. This is the
   data-supported route past the composite.
3. **External-model scoring protocol.** For each scene, build sibling groups
   containing every model's rollout (+ our variants if available). Run, in
   order: (a) fleet pixel scan (haze metric, murk, pixel-stat curves incl.
   mean/std/median with 0→4s deltas), (b) quality composite, (c) VLM pairwise
   judging *across models within the scene group*, Bradley-Terry aggregation
   to per-model scores, (d) mistake-description generation (`find_mistake`
   mode) for human-readable artifact reports — descriptions were accurate in
   V1 even though its numeric severity was not.
4. **Style shift for external models — the primary discriminator.** Expect
   external models to pass content metrics but fail style consistency. Layer
   three instruments, all reference-free, all comparing the first second to
   the rest: (a) the pixel detectors (style-z composite in `style_stats.json`:
   MUSIQ-drift + VGG-Gram + MS-SWD, AUC 0.86 on subtle drift; plus the haze
   metric); (b) CSD contrastive-style-descriptor drift (content-invariant
   artistic style embedding — expected to fire hard on photoreal→game shifts);
   (c) the VLM perception question ("does the style of the final second depart
   from the first second?") — this was blind (AUC 0.50) on V1's subtle drift
   but MUST be re-validated on categorical shifts before use; if any external
   model is known to restyle, use it as the positive control and report the
   validated AUC. Calibrate thresholds on known-clean vs known-shifted videos,
   not on V1's thresholds (population stats in `style_stats.json` are from our
   ablation fleet and will not transfer to other generators).
5. **Efficiency at scale.** Do NOT use triplet/quad designs. Use the
   composite as a prescreen and spend VLM pairwise calls only on pairs the
   composite ranks within its uncertainty band; batch pairs per GPU; the
   protocol is embarrassingly parallel across scenes.
6. **Keep the evaluation honest.** Any new metric or prompt is selected by
   leave-one-grid-out cross-validation on `gt.json` and reported as
   cross-tier pairwise accuracy (ties = half credit) macro-averaged over
   grids. Never tune a blend weight and evaluate on the same pairs (V1 once
   "beat" the composite this way; it did not survive CV). If new human
   annotations are collected, tier format (per-scene, 1–10 with ties) is
   fast and sufficient.

### Files you have

- `gt.json` — 84 tier-labeled videos with the annotator's notes (ground truth).
- `score_video.py` + `quality_model.json` + `style_stats.json` — production
  composite scorer (needs the venv-equivalent: pyiqa, timm, open_clip, torch
  cu-appropriate; see `REPORT.md` for the env notes).
- `fleet_pixscan.py`, `haze_baseline.py`, `pixel_stats.py` — CPU pixel tier.
- `vlm_judge.py` (prompts incl. `rules_v2`, side-by-side pack builders,
  change maps), `qwen3_judge.py` (logit pairwise + absolute/ref-anchor/
  detection modes), `internvl_judge.py`, `multi_judge.py` (triplet/quad,
  kept for the record), `vlm_eval.py` (scoring vs gt.json).
- `split_tiles.sh` — 3328×960 grid video → 7 labeled tiles.
- `REPORT.md` — complete V1 history including every negative result.

Start by reproducing V1's numbers on `gt.json` exactly (composite ~78–82%
pairwise, InternVL3.5-8B 68.4%) to verify the port, then proceed down the
V2 list. Report per-model scorecards for the external models as: content
plausibility (composite + VLM-BT score), haze rate, murk rate, style-shift
rate (three instruments separately), with per-scene tables and a fleet
summary.

---

## END PROMPT
