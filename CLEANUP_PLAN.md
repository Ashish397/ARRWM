# code_release cleanup plan

Consolidated from five independent audits (training core / utils+data-prep / eval+figures /
testbench_v2 / configs+sbatch+hygiene). Nothing has been changed yet.

Governing constraint: **the computation that produced the paper's results must stay
bit-identical.** Every item below is classified:

- **SAFE** — cannot change numerics or control flow (comments, dead code, unused imports,
  local renames, formatting, path *resolution* that currently resolves to nothing).
- **RISKY** — could alter a reported number. Needs explicit sign-off + the stated proof.
- **DECISION** — not a code question; needs an author's answer.

---

## Phase 0 — Release blockers (the release does not currently run)

Two defects, **both introduced by me**, not by the original development.

### 0.1 Path placeholders don't expand (my path scrub)
Replacing author paths with `${AF_ROOT}`-style placeholders put *shell* syntax into YAML
and Python, where it does not expand. ~52 occurrences across 31 files.

| symptom | verified |
|---|---|
| `OmegaConf` reads `${X}` as **interpolation** → `InterpolationKeyError` | yes |
| `utils/wan_wrapper.py:20` loads config **at module scope** → importing any trainer raises | yes |
| `fleet.py:47,50` → `discover_fleet()` returns **0 videos**, runners print "wrote 0 rows", **exit 0** | yes |
| `fleet.py:50` → no real refs → `style_shift` silently uses its *fallback* threshold (clean-fleet p90 instead of real-ref p95) — **a different metric, no error** | yes |
| `headtohead_extract.py:16` → writes an **empty CSV, exit 0** → 7 published figures dead | yes |
| sbatch `CACHE_DIR='${HF_HOME}'` single-quoted → creates a literal directory | yes |
| `#SBATCH --output=${AF_ROOT}/...` → Slurm does not expand vars in directives | yes |

Fix (**SAFE** — the current values resolve to nothing, so nothing is altered, only enabled):
- YAML: `${DATA_ROOT}` → `${oc.env:DATA_ROOT}` (8 files). Verified working.
- Python: `os.environ` / the idiom already in-repo at `analysis/testbench_v2/dust3r_metrics.py:30`
  (`ARR = os.environ.get("AF_ROOT", <repo root from __file__>)`).
- Shell: unquote `CACHE_DIR`; make `#SBATCH --output` a relative path.
- Add a guard where failure is silent (`fleet`, `headtohead_extract`): raise if 0 videos found.

### 0.2 Broken package `__init__` files (a hole in my import-closure check)
My closure tool skipped relative imports, so these were missed. **21 broken references, 2 files:**
- `pipeline/__init__.py` — imports 7 modules that were not shipped. `import pipeline` fails
  outright, which also breaks the released `inference.py:15`.
- `model/__init__.py` — `_MODULE_MAP` names 14 DMD modules that do not ship (lazy, so it
  fails on attribute access rather than import).

Fix (**SAFE**): trim both to what actually ships. Then **DECISION**: either ship
`pipeline/causal_inference.py` so `inference.py` works, or drop `inference.py`.

### 0.3 Artifacts referenced but absent
| artifact | needed by | consequence |
|---|---|---|
| `analysis/pca_evr.npy` | `utils/pca_components_fig.py` | **fig:pca_act unreproducible** (exists in main repo; just ship it) |
| `noadaln` entry in `sbatch/launch_inject.sh` | phase-A rollouts | **`wedge_g0/g1` noadaln panels unregenerable** (MAP has 6 of 7 runs) |
| `pre_encode_local.py` | `utils/pre_encode_direct.py:28` | that script cannot import at all |
| `gen_lmdb.py` | `gen_lmdb_14e.py:86` | import error |
| `analysis/eval_final/**`, model checkpoints, encoded zarr, Wan2.1 weights | everything | expected to be absent — but must be **documented** as download/generated |

`requirements.txt` is missing **torch, torchvision, numpy, pandas, pyiqa, ruptures,
ultralytics**; `opencv-python` is duplicated; ~20 listed packages (ONNX/TensorRT/Flask) are
never imported.

---

## Phase 1 — SAFE cleanup (no sign-off needed)

**1.1 Delete unreachable code — ~8,000 LOC.** All grep-proven zero-caller, most also
un-importable, so it cannot execute.
- `trainer/distillation.py`, `model/streaming_training.py`, `model/anti_collapse.py`,
  `pipeline/streaming_training.py`, `pipeline/streaming_switch_training.py` (~6,900 LOC) —
  the DMD/self-forcing tier. No config selects `score_distillation`; the files import 14
  `model/dmd*.py` modules plus `latent_actions/` and `cotracker/` that were never shipped.
- `trainer/diffusion_train.py` (1,422 LOC) — no config sets `trainer: lora_diffusion`.
  *(Confirm no baseline number came from it.)*
- ~1,030 of 1,207 lines of `model/action_model_patch.py` — the bidirectional/TF-only half;
  only `apply_action_patches` is imported externally. Needs one smoke test after.
- `utils/zarr_dataset.py`: `ZarrSequentialDataset` + memory-log helpers (~279 lines, 22%).
- `utils/wan_wrapper.py`: dead `impose_stat` / `cls` / `rgs` branch API (~160 lines).
- `action_query/pca_motion.py`: the ~180-line exploratory half (keep `load_pca`/`transform`).
- 5 superseded v14b data-prep scripts (~540 LOC), four of which `os.chdir('${AF_ROOT}')`:
  `pre_encode_direct.py`, `build_weunz_manifest.py`, `build_curated_pool.py`,
  `score_all_windows.py`, `harvest_backward_windows.py`.
- 4 unpublished figure scripts (~441 LOC): `following_phaseA.py`, `ndof_plots.py`,
  `response_gain_eval.py`, `analysis/style_shift/detect_style_shift.py`.
- `gen_lmdb_14e.py` — self-titled "DRAFT (review before submission)", referenced by nothing.
- 7 `__pycache__` dirs.

**1.2 Rewrite the AI-dev-log register — ~450 comment lines.** Keep every methodological
constraint and every number; drop the incident narration and the instructions-to-an-agent.
- Delete: SLURM job IDs (`job 5290964 deadlocked...`), dates (`post-2026-05-06`), commit
  hashes, two `chatgpt.com/share` links, `✓` emoji, `####` banners, a quoted user remark.
- Reframe: `"V1-validated, do not change without sign-off"` → "these window definitions are
  fixed; metrics from different definitions are not comparable". `"Honesty rules
  (non-negotiable)"` → "Model-selection protocol". `"DEAD ENDS — do NOT retry"` → "Judge
  designs evaluated and rejected" (**keep the list and accuracies** — checklist provenance).
- Strip version archaeology: `v11/v12/v14/v14b/v14d/v21`, `critic8`, `loo_f3`, `weunz`,
  `phase1/phase3`, `Study 1 / AOO`, `8free`.
- Fix docstrings that misdescribe the code — notably `utils/zarr_dataset.py`, which
  documents an **ss_vae** path the released configs never use (they run `pca_raw`).
- **Do not touch** three shouty strings that are *printed report output*, not comments:
  `style_shift.py:131`, `scorecard.py:168`, `map_real_refs.py:79`.

**1.3 One shared figure module.** Seven divergent label maps currently exist; the
inconsistency is visible **in the submitted paper** (one model reads "Ours (batch 64)",
"batch size 64", and "ours batch size 64" across three figures). Create
`utils/figure_common.py` with `RUN_LABEL`/`RUN_COLOR` + the shared `DIRS`/`DNAME`/`CMD` +
one `load_headtohead()` (that loader is copy-pasted 4×). Requires re-rendering 7 figures.
**Leave every membership list alone** — see 2.5.

**1.4 Hygiene.** Unused imports/locals (~25 lines); attribution headers to match `NOTICE`
(`scheduler.py` lacks its Self-Forcing header; `memory.py:log_gpu_memory` is a local
addition inside vendored code; two files carry an NVIDIA block but are described as new
work in `NOTICE` — a licensing statement, worth a human eye); reformat
`melt_vlm_bench.py`, the one file not in house style.

**1.5 Provenance banners.** One line per rejected/unadopted candidate stating it was
evaluated and rejected, is retained per the reproducibility checklist, and is used for no
reported number: `dust3r_metrics.py`, the `epi_*` block in `geometry_metrics.py`,
`melt_vlm_bench.py`, `lora_judge_ft.py`. Plus a generation banner on each `noop_*` file
(gen-1 appendix / gen-2 main paper / gen-3 instrument / diagnostic).

---

## Phase 2 — RISKY (sign-off + proof required)

**2.1 `TB2_CORRECT_BOUNDARY` — two live window definitions.** `fleet.py:31` switches
CTX/BASE/END and `_GEN_START`; legacy CTX includes frame 12 (the first *generated* frame),
corrected stops at 0.75 s. Every window-delta metric, IQA drift, CSD drift and VLM frame set
depends on which ran — and **no sbatch script or config in the release sets a single `TB2_*`
variable**, so the environment that produced the numbers is unrecorded. *Proposal:* add a
`boundary_mode` column to every CSV and serialise the resolved config (additive, SAFE);
retire the dead branch only once the answer is known.

**2.2 The `except Exception` + `print("[skip] …")` handlers — correctness, not style.**
~20 sites. A skipped video is absent from the feature CSV → `groupnorm` NaNs the sibling
group below `MIN_GROUP=3` → the **p12 dirty threshold and the fleet MAD z-scores are
computed over survivors**. So one decode failure can move a *threshold*, not just an n; and
rates are `.mean()` over `dropna()` with no recorded denominator (`noop_scorecard.py:202`
prints a hardcoded `n=32/model` that is never checked). *Proposal:* adopt the pattern
already used at `noop_final_eval.py:305` — emit a `missing=True` row with a reason and
assert expected n. *Proof first:* row counts per model in every shipped CSV, and grep the run
logs for `[skip]`. **If skips occurred, affected rates need restating.**

**2.3 Freeze, don't change, the figure constants.** `SETTLE=2`, `LASTN=1000`,
`MIN_BIN_N=12`, 13 bins, the **132-sample cap**, `TH=0.1`, `rolling(9, min_periods=3)` (7
call sites), the `3*27` offset filter, the p95 wedge rim. Note `MIN_BIN_N` and the 132 cap
are **quoted in the paper prose**, so they are published values. *Proposal:* hoist to named
constants with values untouched + assert every plotted series lies inside the axis limits
(a series outside them silently vanishes from a figure).

**2.4 Retire the "reward" legacy fallbacks** (`causal_diffusion_teacher_train.py:419-427`).
All 7 configs set the outer keys; proven inert. Recommended **change**.

**2.5 Do NOT consolidate three things that look duplicated but are not.**
- The **five RANSAC fits** differ deliberately (affine vs homography vs fundamental;
  thresholds 2.0/3.0; grid 8/16/24) — distinct instruments.
- The **two mangle ensembles** differ on the record: `noop_scorecard.py:82-90` documents
  why (z-scoring against real refs' tight std sends matrixgame to z≈39).
- `wedge_plots.ABL_SET` lists `pca8_8node` **twice on purpose** (shared reference panel in
  both rows); a naive dict rewrite turns `wedge_g0.png` from 8 panels into 7.

**2.6 Do NOT rename the run keys** (`pca8_8node` → `Default`). Two agents independently
recommend against: the key is simultaneously a dict key, a filesystem path component, and a
value inside already-computed CSVs, spanning 40+ files with no test to catch a miss — and a
miss fails as a **silently dropped model row in a paper figure**. The reader-facing benefit
is fully available at the label layer (1.3).

**2.7 Coordinated deletions** (span two agents' scopes): the `dual_view`/AOO rear-camera
path (~80 lines across trainer + `causal_teacher_streaming` + `zarr_dataset`), whose
`flip_reverse_z(dims=(2,7))` default encodes the **retired z2/z7 convention** and is a live
footgun under the 14e layout; and the `ARRWM_MANIFEST_PICKLE` env path in `zarr_dataset.py`
(a second, undocumented manifest-cache mechanism — 3 grep hits, all internal).

---

## Phase 3 — Author decisions (not cleanup)

1. **Ablation confound — CORRECTED after checking the `_cont` configs and checkpoint
   timeline.** The audit's "three differences" claim was based on base configs only and is
   wrong as stated. Every variant reaches step 5000, and every variant runs with
   `glitch_mask_period: 151` + `fixed_shuffle: true` at some point. The real inhomogeneity is
   *when*:

   | variant | steps 0 → ~1500-2000 | ~2000 → 5000 |
   |---|---|---|
   | Default, 16node, 4node, pca4, pca2 | filter **off**, reshuffled | filter **on**, fixed order (`_cont`) |
   | noatok, noadaln | filter **on**, fixed order (from step 0) | same |

   So the two *architectural* ablations trained on glitch-filtered data in a fixed order for
   **all** 5000 steps, while the other five did so only for roughly the last 60-65%.
   `noadaln`'s `stop_at_step: 3300` is not a truncation — `_noadaln_cont` resumes the same
   logdir to 5000 (checkpoint `causal_lora_step0005000.pt`, Jul 5). **No re-run is needed to
   reach 5000.**

   Interpretation: the qualitative conclusion is likely safe — `noadaln` is flat at ~0.298
   realized throttle regardless of command, i.e. it ignores the action entirely, which an
   early-training data-filtering difference is very unlikely to cause. But the comparison is
   not recipe-matched and should be disclosed; re-running the two architectural ablations with
   the Default schedule (filter off for the first phase) is the clean fix if time allows.

   Separately and still true: "5000 steps" appears in **no config** for 5 of 7 variants — it
   comes from `launch_inject.sh` hardcoding `step0005000.pt`.
2. **`lora_judge_ft.py`** — was it ever run? If not it is not a "rejected candidate" but an
   unexecuted plan, and shipping it as method is misleading.
3. **`labels/canonical_static_mask.csv`** (3,329 rows) — **no consumer in the release**.
   Missing script, or leftover?
4. **`noop_null_sweep_score.py`, `noop_evidence_grid.py`, `make_validation_reels.py`** — do
   these render figures that appear in the paper?
5. **`trainer/diffusion_train.py`** — confirm no reported number came from it before deleting.
6. **W&B key names** (`train/critic_z2_mse` etc.) use the retired z2/z7 convention; the code
   itself warns the labels are mislabelled. Renaming desynchronises archived run logs —
   recommend leaving, but it is a presentation choice.

---

## Totals

| | |
|---|---|
| SAFE deletions | ~8,000 LOC (57% of trainer/model/pipeline is unreachable) |
| SAFE comment/doc rewrites | ~450 lines |
| Blocker fixes | ~70 lines across 33 files |
| RISKY items needing sign-off | 7 |
| Author decisions | 6 |

Suggested order: **0.1 → 0.2 → 0.3 → 1.1 → 1.2 → 1.3/1.4/1.5 → 2.x** after sign-off.
Phase 0 first because nothing in the release can be verified until it runs.
