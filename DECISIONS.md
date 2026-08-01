# Autonomous decisions taken during the code-release cleanup

Decisions I made without you, per your instruction to "take a decision to make it
just better and note it down". Each has a rationale and a reversal cost. Review
and confirm or overturn.

Working copy is `code_release/`. Reference baseline for verifying that cleanup
does not alter computation: `tools/release_baseline.py` + `tools/baseline.json`
(`--save` before edits, `--check` after each batch).

## Verification status: green

Checked the full post-cleanup tree against the pre-cleanup snapshot
(`tools/baseline.json.bak`) on GPU node nid010513, 7/7 probes live:

| probe | result |
|---|---|
| `scheduler` | no drift |
| `action_modulation` | no drift |
| `action_critic` | no drift |
| `pca_basis` | no drift |
| `tanh_squash` | no drift — confirms the `zarr_dataset` trim left `_tanh_squash` bit-identical |
| `api` | drift = **exactly** the 10 `pca_motion` functions removed in D19, nothing else |
| `configs` | drift proven to be **only** the harness's own hermetic-root change (see below) |

The `configs` entry is a harness artifact, not a config change: substituting the
probe roots back (`/probe/data`→`/tmp/data`, `/probe/af`→`/tmp/af`, ...) makes
all 8 resolved configs byte-identical to the pre-cleanup snapshot. All 60
differing lines are `encoded_root` / `caption_root` / `motion_root` / `logdir`.

The baseline has since been re-saved on the current tree and re-checked: 7/7
live, no drift. Two harness fixes went in alongside: a `RELEASE_DIR` /
`BASELINE_JSON` override so another revision can be probed without touching the
working tree, and a structural (key-wise) drift diff — the old line-by-line diff
became unreadable the moment a key was removed, because every later line shifted.

Note the login node SIGTERMs the torch probes (exit 143), so `--check` has to run
on a compute node or inside a live allocation.

---

## D1. Ablation confound: not disclosed (your call, recorded here)
You decided no disclosure is needed. Recorded for completeness: the two
conditioning ablations (`noatok`, `noadaln`) ran with `glitch_mask_period: 151`
and `fixed_shuffle: true` from step 0, whereas the other five enabled them at
their continuation checkpoint (step 1500–2000 of 5000). The mask zeroes the loss
gradient on ~0.66% of latent frames; `fixed_shuffle` changes data order only.
`noatok` is an internal control (same recipe as `noadaln`, behaves like Default),
so the AdaLN-vs-tokens comparison is unaffected. **No action taken.**

## D2. Configs unified to a single-leg recipe
All seven configs now carry `glitch_mask_period: 151`, `fixed_shuffle: true`,
`stop_at_step: 5000`, differing only in their ablation dimension. Previously the
five non-architectural variants expressed none of these, so the release could
not describe the recipe that was actually run — "5000 steps" existed only as a
hardcoded checkpoint name in `launch_inject.sh`.
*Rationale:* you confirmed the training curves show no discontinuity at the
continuation points, so the merged recipe is the honest single-leg description.
*Reversal:* re-split into base + `_cont` pairs (the originals are untouched in
the main repo).

## D3. Shipped `pipeline/causal_inference.py`
`inference.py` was in the release but `pipeline/__init__.py` imported seven
modules that were never shipped, so `import pipeline` failed outright.
*Decision:* ship `causal_inference.py` (343 LOC, no new dependencies — it needs
only `wan_wrapper`, `memory`, `debug_option`, all already present) rather than
delete `inference.py`. A reader can now actually generate a rollout.
*Reversal:* delete both files.

## D4. Deleted the dead DMD / self-forcing tier (−6,269 LOC)
Removed `trainer/distillation.py`, `trainer/diffusion_train.py`,
`model/streaming_training.py`, `model/anti_collapse.py`,
`pipeline/streaming_training.py`, `pipeline/streaming_switch_training.py`;
collapsed `train.py`'s three-way dispatch to the one trainer the configs select.
*Verified before deleting:* every config sets `trainer: causal_lora_diffusion`;
the deleted trainers were reachable only from `train.py`'s other two branches;
`anti_collapse` was gated on `_impose_stat_mode`, whose only setter
(`configure_impose_stat`) has zero callers repo-wide. Several of these files were
also un-importable in the release (they import 14 `model/dmd*.py` modules that
were never shipped).
*Rationale:* unreachable and un-importable code cannot change results, and this
was the loudest development-history signal in the tree (a 14-generation
`DMD2RealMSELAM_Actions` naming ladder).
*Reversal:* restore from git history.

## D5. Removed the `impose_stat` surface from `utils/wan_wrapper.py`
Deleted `configure_impose_stat`, `set_impose_stat_target`, their init block and
the `forward`-path branch (~55 lines) — all unreachable, and the only remaining
references to the deleted `model/anti_collapse`.
*Note:* my first attempt wrote a broken file (a regex clipped an `elif`); caught
by the syntax check and repaired. I now validate before writing.

## D6. Minimised `model/__init__.py`
Was a lazy import map for 14 non-existent DMD modules. Nothing in the release
imports those names (only the deleted `distillation.py` did). Replaced with a
docstring pointing at direct submodule imports.

## D7. Path placeholders made functional
My earlier anonymisation put shell-style `${VAR}` into YAML and Python, where it
does not expand — the release could not load a config or import a trainer, and
`fleet.py` silently discovered zero videos while `style_shift.py` silently fell
back to a different threshold rule.
*Fix:* YAML uses `${oc.env:VAR}`; Python uses `os.environ.get(VAR, <repo root
from __file__>)`. Env contract: `AF_ROOT`, `DATA_ROOT`, `WAN_MODELS`, `HF_HOME`.
*Classification:* computation-neutral — the previous values resolved to nothing.

## D8. Stripped the wandb key from the two new `nocritic` configs
`configs/..._v14e_nocritic{,_full}.yaml` were untracked and carried the live API
key inherited from the base config; committing them would have re-exposed it in
fresh files on a public repo.
**Still outstanding for you: rotate the key.** It remains in existing git
history, so removing it from new files does not revoke it.

## D9. Force-added the frozen PCA basis to git
`.gitignore` contains `checkpoints/`, which was silently excluding
`code_release/action_query/checkpoints/ss_vae_8free.pt` — the frozen action
basis, i.e. the core learned artifact of the method and a hard requirement for
reproducing the action space. 122 KB.

## D10. Remote switched from HTTPS to SSH
`git push` failed because GitHub dropped password auth and `credential.helper=
store` held a stale password. Your existing `~/.ssh/id_ed25519` is already
registered on your account (verified via `ssh -T`), so I repointed `origin` at
`git@github.com:...`. No credentials were created or handled.

## D11. Kept `LICENSE` and `NOTICE` despite "no .md files"
Renamed from `LICENSE.md` / `ATTRIBUTION.md` to extension-less plain text rather
than deleting them. The inherited CC-BY-NC-SA-4.0 (via LongLive ← Self-Forcing)
is share-alike **and** attribution-required, so distributing without them would
breach the licence the code was received under. `NOTICE` is the conventional
filename for exactly this and is not documentation.
*Reversal:* delete if you handle attribution in your own README.

## D12. Deleted 7 superseded scripts; kept two the audits wanted gone
Deleted (zero importers, and verified against the paper's actual 22-figure list
extracted from the tex): `following_phaseA.py`, `ndof_plots.py`,
`response_gain_eval.py`, `style_shift/detect_style_shift.py`,
`build_weunz_manifest.py`, `build_curated_pool.py`, `gen_lmdb_14e.py`.
`response_gain` appears in the tex only as `\ref`/`\label` for a composite float
wrapping the two `response_curves_eval` images, so no figure is lost.
**Rejected two audit recommendations:** `score_all_windows.py` writes the
`--dumps_glob` inputs `build_balanced_pool` requires (it is the live pool chain),
and `harvest_backward_windows.py` documents the 6x reverse oversampling the paper
describes. The checklist commits to shipping preprocessing code, so both stay.

## D13. Shipped two missing inputs
`analysis/pca_evr.npy` — without it `pca_components_flowfields.png`, a real
paper figure, cannot be regenerated. `utils/pre_encode_local.py` — unbreaks
`pre_encode_direct.py`, which imported it but was shipped without it.

## D14. Comment register cleaned, technical content preserved
Rewrote the instruction-to-an-agent phrasing ("V1-validated, do not change
without sign-off", "Honesty rules (non-negotiable)", "DEAD ENDS - do not
re-add", "never reuse V1 population stats") into neutral prose. Every AUC,
rejected configuration and methodological constraint is retained; only the
imperative and the addressee are gone. Also removed two `chatgpt.com/share`
derivation links (the algebra above each was already complete), one SLURM job ID
from a watchdog comment, and one emoji. Residual count for all of these: 0.

## D15. Relabelled internal codenames instead of deleting the feature
"Study 1 / AOO" (the rear-camera dual-view path) is now "dual-view rear camera
(unused by the released configs)". The audits recommended deleting the surface
outright (~80 lines), but `dual_view` also doubles `wrapper.seq_len`, so removing
it is exactly the kind of structural change you asked me to avoid. Relabelling
achieves the readability goal at zero risk. Same treatment for `v14b`/`v14d`
pool tags. Vendored `wan/` left untouched.

## D16. Left the retired z2/z7 naming alone
`score_all_windows.py` still documents `turn (|z2|)` / `motion (|z7|)`, and the
code indexes `z[:,2]`/`z[:,7]`. Per the standing instruction that these legacy
comments are wrong but should be left, I changed only the version tag. **This
will confuse a reader of the release** — see open item 7.

## D17. Trimmed `model/action_model_patch.py` (1,207 → 133 lines, −1,074)
Removed eight top-level functions with no callers — the bidirectional/critic
patch surface (`_patch_bidirectional_self_attn_for_action`,
`_bidir_forward_with_action_tokens`, `patch_bidirectional_wan_model_for_action`,
`_tf_only_forward_for_bidir_wan`, `patch_bidirectional_wan_model_for_tf_only`,
`apply_action_patches_critic`, `apply_tf_only_patches_critic`,
`_prepare_tf_block_mask_cached`) — plus the imports and the unused
`_flex_attention_compiled` object they alone required. This surface existed only
for the DMD critic path removed in D4. What remains is the three functions the
docstring actually describes.

This was the deferred item; the two concerns behind the deferral are resolved.
*Dynamic resolution:* every `getattr` in the module targets a **model
attribute** (`_action_tp_patched`, `action_tokens_per_frame`, `tf_rope_offset`),
never a module-level function name, so nothing is reachable by string lookup.
*Verification without torch:* four static checks, all on the login node —
(1) an AST call-graph closure from the sole exported entry point
`apply_action_patches`, following bare `Name` nodes so monkey-patch assignments
(`Cls.forward = _fn`) count as references, reaches only the 3 kept functions;
(2) the other 8 names appear in no other file in the release (the one external
importer, `trainer/causal_diffusion_teacher_train.py:45`, imports
`apply_action_patches` only); (3) no module-level statement references them;
(4) the 3 survivors' **ASTs are identical** to their pre-trim versions, and the
free names they resolve at module scope (`types`, `wraps`, `FSDP`) are still
imported by the same statements. Identical ASTs plus proven-unreferenced
deletions means the trim cannot change computation. Release-wide import closure
re-run afterwards: 106 files, 0 syntax errors, all imports resolve.
*Textual changes:* the docstring no longer advertises the deleted critic path,
and two step comments renumbered ("2)/3)" → unnumbered) since step 1 was in the
removed half. These are the only diffs inside the kept functions.
*Reversal:* restore from git history.

## D18. Trimmed `utils/zarr_dataset.py` (1,276 → 995 lines, −281)
Removed `ZarrSequentialDataset` and the three memory-logging helpers
(`_build_memory_log_message`, `_read_proc_status_memory`, `_format_gib`). These
form one dead cluster: the helpers are called only from inside the sequential
dataset, so they die with it. The live dataset is `ZarrRideDataset` — imported
by 11 modules, including the trainer — and `ZarrSequentialDataset` has zero
importers anywhere in the release, in any config, and in any sbatch script.
The module docstring described the removed class, so it now describes
`ZarrRideDataset` instead (one sample per ride, lazy latents, deferred motion
encoding). No other prose changed.

## D19. Trimmed `action_query/pca_motion.py` (439 → 224 lines, −215)
Removed ten exploratory helpers (`plot_example_grids_per_direction`,
`collect_transformed`, `motion_grid_type`, `direction_from_visible`,
`direction_from_flat`, `smoothness_score_batch`, `grid_energy`, `whiten_pca`,
`unwhiten_pca`, `centroid_to_motion`). The module has **no importers** in the
release — it is a standalone script whose `main()` fits and saves the frozen PCA
basis — and none of the ten is reachable from `main()`. They existed for the
`tsne_motion` / `kmeans_motion` clustering exploration, which is not part of the
release and not part of the paper.
*Checked before deleting:* the paper figure `pca_components_flowfields.png` is
produced by `utils/pca_components_fig.py` (which reads `analysis/pca_evr.npy`),
**not** by the deleted `plot_example_grids_per_direction`, so no figure loses
its generator. The docstring's pointer to the unshipped `tsne_motion.py` /
`kmeans_motion.py` is gone with them.
*Baseline effect, confirmed:* the `api` probe snapshots public signatures of
`action_query.pca_motion`, so `--check` reports drift there — and the drift is
exactly these ten names and nothing else. The basis-fitting path
(`find_all_motion_files` → `get_valid_file_shapes` → `fit_pca` → `save_pca`) is
untouched, and `pca_basis`, which hashes the actual frozen basis and a fixed
projection through it, came back green.

## D20. Trims are gated by a tool, not by hand
`tools/trim_dead.py` deletes named top-level defs and refuses to write unless
(a) the file still parses, (b) every surviving def's AST is **identical** to its
pre-trim version, and (c) the trim introduces no unresolvable name.
*Why (c) exists:* my first version of the tool pruned "unused" imports using a
usage set that subtracted import-bound names wholesale, which silently deleted
`os`, `argparse`, `sys`, `time`, `Path` and `numpy` from `pca_motion.py`. The
AST check did not catch it because it only compares function bodies, not module
imports. I restored both files from git and rewrote the usage walk to skip
import *statements* rather than subtract their names. The unresolvable-name
guard is the regression test for exactly that class of mistake, and the
corrected trims came out 13 and 7 lines smaller because the imports now survive.

## D21. One shared figure label map (`utils/figure_labels.py`)
Six figure scripts each carried their own label dict, so the same run appeared
under four different names across figures in one paper — `pca8_8node` was
"Default", "Ours (Default)", "ours Default" and "batch size 32 (top8)", and
`16node` had five spellings. A reviewer comparing two figures could reasonably
read those as different models. All six now import one map, which also handles
the key aliases the scripts inherited (`pca8` / `pca8_8node` / `8node8pca` /
`8node` all denote the default model).

The canonical names are the ones you specified: Default, batch 64, batch 16,
No Action Tokens, No AdaLN, with PCA4 / PCA2 unchanged. `label(run, ours=True)`
adds the "Ours (...)" prefix only for figures that mix our runs with external
baselines, and only for our runs.

**Which rendered figures this changes, if you re-run them:**
- `wedge_all_*.png`, `wedge_*.png` — **no change**; these were already canonical,
  which is why I made them the reference.
- `response_complete{,_steer}.png` — "batch size 64"→"batch 64", "batch size
  16"→"batch 16", "pca4"→"PCA4", "pca2"→"PCA2", "no action tokens"→"No Action
  Tokens", "no AdaLN"→"No AdaLN".
- `response_curves_eval_{steer,throttle}.png` — "ours Default"→"Ours (Default)",
  "ours batch size 64"→"Ours (batch 64)"; baselines gain proper names
  ("minwm"→"minWM", "matrixgame"→"Matrix-Game", ...) instead of raw keys.
- `following_*_by_REALdir*.png` — the "(top8)" / "(top2, batch32)" suffixes go;
  runs take their canonical names.
- `following_FAMILY_*_by_REALdir*.png` — canonical names, except that the
  default model keeps a parenthetical naming where it sits on that family's axis
  ("Default (batch 32)" in the nodes family, "Default (AdaLN + tokens)" in the
  injection family), since without it the varied axis is unreadable.

**I did not re-render anything.** The figures currently in the paper were built
with the old strings, so regenerating now would change their legends. That is an
improvement in consistency but it is a camera-ready decision, not a cleanup one
— tell me to re-render and I will. No computation is affected either way; these
are display strings only, and no baseline probe covers the figure scripts.

## D22. Removed 23 unused imports; fixed the `zarr_dataset` encoder docstring
`tools/prune_imports.py` (same guards as `trim_dead.py`, plus it skips
`__init__.py` and anything in `__all__`, since a re-export is used by importers
rather than by the file itself) removed 23 unreferenced top-level imports across
14 files. Each candidate was additionally grep-checked to appear exactly once in
its file — the import line itself — so none is reached through a string or
`getattr`. `pipeline/__init__.py`'s `CausalInferencePipeline` was deliberately
kept: it is the package's public re-export. Release-wide unused-import count is
now 0 (vendored `wan/` excluded). The `api` probe confirms no signature moved.

Separately, the plan flagged `utils/zarr_dataset.py` as documenting an encoder
path the released configs do not use, and it was right. The docstring claimed
motion is "encoded through the ss_vae to 8D latents, tanh-squashed per
dimension", but all seven configs set `teacher_action_encoder: pca_raw`, which
is the raw top-8 PCA projection with **no** squash (the code itself labels it
"pre-squash"). The docstring now says which of the three `ARRWM_ACTION_ENCODER`
paths the release actually runs, and notes the env var must match the config
since the critic target and the eval read share the basis. The `ss_vae` code
default is unchanged — only the description.

## D23. Provenance banners — roles, not a generation taxonomy
Added a short status line to the four candidate files: `dust3r_metrics.py`
(evaluated, not adopted), the `epi_*` block in `geometry_metrics.py` (evaluated
and rejected — the rigidity residual did not separate the severity labels),
`melt_vlm_bench.py` (a model-selection benchmark, and the source of the finding
that the strongest melt scorer alone *lowered* ensemble AUC, which is why Qwen
was kept), and `lora_judge_ft.py` (not adopted). Each says plainly that no
reported number depends on it and that it is retained because the checklist asks
for the alternatives that were tried.

**Deviation from the plan.** The plan asked for a generation banner on each of
the twelve `noop_*` files (gen-1 appendix / gen-2 main paper / gen-3 instrument /
diagnostic). I could not substantiate that taxonomy — `NOOP_FINAL.md` names only
`noop_final_eval.py`, and two separate files call themselves "final" — so rather
than assign generation numbers I could not defend, each file now states its
**role**, which is verifiable from its own docstring and from what it outputs:
which one produces the main-paper numbers, which are measurement or reporting
stages, which are diagnostics, and which only render media. For
`lora_judge_ft.py` I deliberately did not write "rejected", because whether it
was ever run is still open item 4; the banner claims only what is checkable,
that no reported number comes from a fine-tuned judge.

---

## Phase 2 verification: green (job 5833400, vs `baseline_prephase2.json`)

7/7 probes live. Drift appeared in exactly three places, all of them intended:

| probe | result |
|---|---|
| `scheduler` | no drift |
| `action_modulation` | no drift |
| `action_critic` | no drift |
| **`pca_basis`** | `scale` REMOVED, `latent_ch` ADDED — **`pca_mean`, `pca_comp` and `projection` are absent from the drift list, i.e. bit-identical** |
| `api` | the two `action_query` modules REMOVED, `utils.zarr_dataset` + `utils.wan_wrapper` ADDED — no signature changed |
| `configs` | the key rename only (proof below) |

The `pca_basis` line is the one that matters: removing the ss_vae did **not**
disturb the frozen action basis, and a fixed synthetic displacement field
projected through it gives the identical hash before and after.

The `configs` drift was then checked line by line against an allow-list. Across
all 8 configs the only differences are `-ss_vae_checkpoint`,
`+pca_basis_checkpoint` and `-discriminator_lr_multiplier`; every other resolved
key and value is byte-identical.

---

# Phase 2 (authorised): strip everything 14e does not use

## D24. Removed the ss_vae — but the basis it carried is preserved exactly
**The trap:** `action_query/checkpoints/ss_vae_8free.pt` was not just a VAE. It
carried `pca_mean` (200,) and `pca_comp` (16, 200) — the frozen PCA action basis,
the core learned artifact of the method — alongside 33 tensors of ss_vae network
weights and an `hparams` dict. Deleting the file would have destroyed the method.

**What I did instead:** rebuilt it as `preprocessing/checkpoints/pca_basis.pt`
holding only `pca_mean`, `pca_comp` and `latent_ch` (8, lifted out of
`hparams["LATENT_CH"]`, which was live). `pca_mean` and `pca_comp` are verified
**byte-identical** by SHA-256 before and after (`6186d92f…` / `4bf1c76d…`).
119 KB → 15 KB. Dropped: `model_state`, the rest of `hparams`, and `scale` (the
ss_vae input scale, used only by the removed encoder).

**Code removed:** `action_query/ss_vae_model.py` (95 lines); in `zarr_dataset.py`
the `load_ss_vae` import, `_encode_motion_ss_vae`, `_share_ss_vae`, the
`_ss_vae`/`_ss_scale`/`_ss_dev` state, `_ENCODE_BATCH`, and the three-way encoder
dispatch (now unconditional `pca_raw`); in the trainer the `load_ss_vae` import,
`_frozen_ss_vae`/`_frozen_ss_vae_scale`, the `_teacher_action_encoder` switch and
the ss_vae half of `_motion_to_action_z`. Also removed `_encode_motion_pca` (the
affine "pca" mode into VAE z-space) and `_PCA_SLOT_COMP`: a third encoder no
config selects. `_tanh_squash` + `_ZACTION_SCALES` went with them — their only
caller was the trainer's ss_vae branch.

**Renames:** `ss_vae_checkpoint` → `pca_basis_checkpoint` in all 7 configs, the
dataset, the trainer and 8 eval scripts; `_share_ss_vae` → `_share_basis`;
`--ss_vae` → `--pca_basis`. `action_query/` no longer exists. Grep for `ss_vae`
outside vendored `wan/` now returns nothing.

## D25. Removed the DMD/GAN surface from `utils/wan_wrapper.py` (835 → 619)
Deleted `adding_cls_branch` (the GAN classifier head), `adding_rgs_branch` (the
regression head), `enable_alt_head` / `has_alt_head` (the v21 `fake_score` alt
head), and `ResidualMLPBlock` (the classifier's residual block, orphaned once the
heads went). Then `forward` lost its `classify_mode` / `regress_mode` /
`compute_alt_head` parameters and the four dead branches and returns that went
with them.

*Why this was safe:* those `forward` branches referenced `self._cls_pred_branch`,
`self._gan_ca_blocks`, `self._rgs_pred_branch` and `self._gan_ca_blocks_rgs` —
attributes **only** `adding_cls_branch`/`adding_rgs_branch` ever created, so
after D4 removed the DMD trainers they could only ever have raised
`AttributeError`. Nothing outside `wan/` and the wrapper itself passes any of the
three kwargs; the 14e trainer never does. The state-token and **state-probe**
branches were deliberately kept — `adding_state_probe_branch` and
`adding_state_token_branch` are called by the trainer at lines 769 and 790, and
the state probe is part of the paper's contribution.

*Not touched:* `wan/`. The `gan_ca_blocks` plumbing inside `wan/modules/model.py`
is now unreachable from the release, but that tree is vendored Wan2.1 as modified
through the LongLive / Self-Forcing lineage and is covered by `NOTICE`. Editing
it risks the live training path for no reader benefit. Flagged below.

## D26. New `preprocessing/` section
Created `preprocessing/` holding the data-preparation chain, per your request:
`pre_encode_local.py` and `pre_encode_direct.py` (videos → Wan-VAE-encoded zarr),
`pre_encode_motion.py` (CoTracker motion extraction), `pre_encode_text.py`
(caption/prompt generation), `fit_pca_basis.py` (fits the frozen PCA basis; moved
from `action_query/pca_motion.py` and renamed, since "pca_motion" said nothing
about what it produced), and `checkpoints/pca_basis.pt`.

**Two reproducibility bugs fixed while moving it.** `fit_pca_basis.py` wrote a
`.npz` with keys `mean`/`components`, but the dataset and trainer load a `.pt`
with keys `pca_mean`/`pca_comp` — so running the shipped fitter produced a file
the shipped trainer could not read. And its `N_COMPONENTS` was 12, while the
shipped basis has 16 components. Both fixed: `save_pca`/`load_pca` now read and
write the exact format the loaders expect, `N_COMPONENTS = 16` matches the
released artifact, and `ACTION_DIMS = 8` names the exposed slice. The
explained-variance ratio is stored in the same file so the PCA figure is
reproducible from it.

## D27. Other unused code removed
- `utils/loss.py` (98 lines) — five denoising-loss classes, **no importer**.
- `utils/dataset.py` 477 → 41 lines: deleted `VideoLatentCaptionDataset` (the old
  LMDB dataset, superseded by `ZarrRideDataset`), `TwoTextDataset` and
  `MultiTextDataset`. Kept `TextDataset` (used by `inference.py`) and `cycle`
  (used by the trainer).
- `configs/default_config.yaml`: `discriminator_lr_multiplier` removed.

## D28. Baseline probe realigned to the live path
`p_tanh_squash` covered a function that no longer exists, so it is replaced by
`p_action_encode`, which drives `_encode_motion_pca_raw` on a fixed synthetic
CoTracker field and applies the released `pca_raw_scales`. This pins the entire
motion → action-vector map that training actually conditions on, rather than one
activation — strictly more coverage than the probe it replaces. `p_pca_basis`
now reads the new basis path and also records `latent_ch`. The `api` probe swaps
the two deleted `action_query` modules for `utils.zarr_dataset` and
`utils.wan_wrapper`, the two files this phase changed most.

## D29. dual_view rear camera removed
Gone from all three files it spanned: the trainer's `dual_view`/`reverse_root`
config surface, ride filtering and 2x `seq_len` sizing; the batcher's rear-window
constraint, alignment-map slot fields and rear latent/z concatenation; and the
dataset's `load_reverse_latent_chunk`, `load_alignment_map`, `aligned_span` and
`flip_reverse_z`. No released config enabled it, and its `flip_reverse_z`
default carried the retired legacy slot convention, so it was a live footgun.

## D30. `ARRWM_MANIFEST_PICKLE`, `data_blacklist.txt`, `canonical_static_mask.csv`
`ARRWM_MANIFEST_PICKLE` was a second, undocumented manifest cache doing exactly
what the documented `cache_path` already does — removed. `data_blacklist.txt`
(450 ride ids) had no Python consumer. `canonical_static_mask.csv` (3,328 rows of
model/scene/direction/static/active) also had no consumer: the script that read
it was never shipped, and the stationary evaluation recomputes the mask itself
in `noop_final_eval.py`. Both files deleted.

## D31. All GAN/DMD removed, including from vendored `wan/`
Previously I had left `wan/` alone. Now removed there too: `WanGanCrossAttention`,
`GanAttentionBlock`, `RegisterTokens`, `WanModel.enable_alt_head` and
`CausalWanModel.enable_alt_head`, the `head_alt` attributes, and every
`classify_mode` / `regress_mode` / `compute_alt_head` parameter, branch and
return path in both models' `_forward`. `wan/modules/model.py` 919 → 672.

**Bigger win: 4,396 lines of vendored Wan that the release never reaches.** A
module-level reachability closure from what the release actually imports
(`causal_model`, `model`, `t5`, `tokenizers`, `vae`) showed 19 unreachable
modules. Deleted: `image2video.py`, `text2video.py`, `modules/clip.py`,
`modules/xlm_roberta.py`, and the whole of `wan/utils/` (fm_solvers,
fm_solvers_unipc, prompt_extend, qwen_vl_utils), `wan/configs/`,
`wan/distributed/`. `wan/__init__.py` and `wan/modules/__init__.py` were trimmed
to match — the old `wan/__init__.py` imported the T2V/I2V pipelines, so merely
importing `wan.modules.model` dragged all of it in.

## D32. Legacy "reward" config fallbacks collapsed (plan item 2.4)
`action_critic_z_loss_weight`, `generator_action_z_guidance_weight` and
`z_guidance_warmup_steps` each fell back to a `*_reward_*` key. All 7 configs set
the modern key and none sets a legacy one, so the fallbacks were provably inert.
Also changed `action_critic_emphasis_dims`' default from `[2, 7]` to `[0, 1]`:
every config sets `[0, 1]` explicitly, so the default never fired, but `[2, 7]`
was actively misleading under the PCA convention.

## D33. z2 / z7 renamed to PC0 / PC1 in prose — but NOT in the data keys
`z2`/`z7` were slot names in the retired 8-D VAE latent. Under the released
recipe there is no VAE: the action vector is the top-8 PCA projection, and the
old correspondence (documented in the dataset) was `z2 <- PC1`, `z7 <- PC0`.
Every comment, docstring and local variable now says PC0 (throttle) / PC1 (steer).

**The JSON keys `tz2`/`tz7`/`cz2`/`cz7`/`corr_z2`/`corr_z7` were deliberately
left alone.** They are written into `metrics_r*.jsonl` and read back by
`following_by_realdir.py`, `chunk_metrics.py` and `response_complete.py`;
renaming them would make the code unable to read a single archived eval run.
A comment at the write site now states the mapping explicitly: the `"z2"` key
holds column 0 = PC0 = throttle, and `"z7"` holds column 1 = PC1 = steer — note
this is *swapped* relative to what the legacy names imply, which is exactly why
the comment is there.

## D34. Minimalist pass: unadopted evaluation candidates deleted
Removed on your instruction: `dust3r_metrics.py`, `lora_judge_ft.py`,
`melt_vlm_bench.py`, and the `EpipolarRigidity` class plus its three `epi_*`
features and their call site in `geometry_metrics.py`. Also removed the scripts
that render media rather than produce numbers (`make_validation_reels.py`,
`noop_compare_video.py`, `noop_excursion_video.py`, `noop_evidence_grid.py`) and
the pure diagnostics (`noop_drift_diagnose.py`, `noop_null_sweep_score.py`,
`noop_align_eval.py`). Ten files in total; no dangling reference remains.

Hygiene: stripped trailing whitespace and removed commented-out debug `print`
calls. Release is now **77 files / 19,160 lines**, from 117 files / ~32,000 at
the start.

## D35. The wandb key is not in the release
Confirmed by scan: `code_release` contains no API key. The configs carry
`wandb_entity: YOUR_WANDB_ENTITY` placeholders, and `wandb_key` is only read
from config (`getattr(config, "wandb_key", None)`) with no value committed. The
key remains in the *private* ARRWM repo's history, which is only an exposure
risk if that repo is ever made public — the airgapped release folder is clean.
Closing this item; no action needed.

## D36. GAP: ten of the paper's 31 figures have no generator anywhere
Extracted every `\includegraphics` path from `AnonymousSubmission2027.tex`
(31 unique figures) and mapped each to a generating script. Twenty-one map to
shipped code. **Ten do not, and their code is not in `code_release`, not in the
main ARRWM repo, and not anywhere on this machine** — only the rendered PNGs
exist, under `aaai_template/.../Figures/`:

`stationary_wedges.png`, `hf_distribution.png`, `hf_fleet_distribution.png`,
`hf_validation_internal.png`, `geometry_validation_fleet_{internal,external}.png`,
`scene_validation_fleet_{internal,external}.png`,
`style_validation_fleet_{internal,external}.png`, `conjuring_validation.png`,
plus the qualitative stills `pca8_showcase_2.png`, `madrid_18813_2p5s.png`,
`taipei_17683_2p5s.png`.

These are the scene-relocation, conjuration and fleet-validation instruments —
the same gap flagged as open item 2, now quantified. They appear to have been
produced on the local machine. **The release cannot reproduce them as it
stands**, which matters because the reproducibility checklist is the point of
the release. This needs those scripts copied in.

## D37. Fixed two import-time CUDA calls that made the release CPU-hostile
Found because the `api` probe kept recording `utils.wan_wrapper:
skipped:RuntimeError` — the file this cleanup changed most was the one the
baseline did not cover. The cause was two module-level CUDA calls:

- `utils/memory.py:9` — `gpu = torch.device(f'cuda:{torch.cuda.current_device()}')`
  at import time.
- `wan/modules/t5.py:478` — `device=torch.cuda.current_device()` as a **default
  argument**, so it is evaluated when the class body runs, i.e. on import.

Either one makes `import utils.wan_wrapper` (and transitively the trainer)
raise `RuntimeError: No CUDA GPUs are available` on any machine without a GPU.
For a public supplementary release that is a real defect: a reviewer on a CPU
box cannot even import the code to read it. Both now resolve lazily and fall
back to CPU; on a GPU host each yields exactly the previous value (`cuda:0`).

Verified afterwards on a CPU-only login node: `utils.wan_wrapper`,
`utils.zarr_dataset`, `model.causal_teacher_streaming`, `model.action_critic`,
`model.action_modulation`, `utils.scheduler`, `utils.causal_chain_rollout`,
`pipeline.causal_inference` and `trainer.causal_diffusion_teacher_train` all
import cleanly. This also restores `wan_wrapper` to the api probe's coverage.

## D38. A "green" baseline run that proved nothing — recorded so it isn't trusted
Job 5836191 reported `7/7 live probes, OK - no drift`, and that result is
**void**. Two jobs were queued together and SLURM ran the `--save` (5835755,
14:18:19) before the `--check` (5836191, 14:21:17), so the check compared the
post-change tree against a reference saved from that same tree. The give-away
was in the artifact itself: `baseline.json` was timestamped after the edits and
its stored `LockstepRideBatcher` signature already lacked the `dual_view` and
`reverse_root` parameters that batch removed.

Lesson applied: never queue a save and a check together, and confirm the
reference predates the change (`ls -l` the json, or check a signature that
should have moved) before believing a green result.

**The valid comparison (job 5837698, vs `baseline_prephase2.json`) is green.**
Drift appeared in exactly three probes, all intended, and the same probe that
was silent in the void run now correctly reports the batcher signature change —
which is the proof that the earlier green was vacuous rather than the probe
being blind:

| probe | result |
|---|---|
| `scheduler` | no drift |
| `action_modulation` | no drift |
| `action_critic` | no drift |
| `pca_basis` | `scale` removed, `latent_ch` added; `pca_mean`, `pca_comp`, `projection` identical |
| `configs` | verified line-by-line: **only** the `ss_vae_checkpoint` → `pca_basis_checkpoint` rename and the dropped `discriminator_lr_multiplier` |
| `api` | the two `action_query` modules removed; `utils.wan_wrapper` + `utils.zarr_dataset` added; `LockstepRideBatcher` lost exactly `dual_view` and `reverse_root` and gained nothing |

So removing the ss_vae, dual_view, the GAN/DMD surface and 4,396 lines of
unreachable vendored Wan changed no numeric output anywhere the harness can see.

## D39. Pre-ship audit — five things found and fixed
1. **De-anonymisation leak (blocking).** `preprocessing/pre_encode_local.py`
   carried four argparse defaults containing `/home/ashish/...`. For an
   anonymous AAAI submission that is a real-name leak in the supplementary
   material. Replaced with `DATA_ROOT` / `WAN_MODELS` env-derived defaults, in
   line with the rest of the release. A repo-wide scan for the author name,
   username, email, home paths and cluster paths is now clean.
2. **Missing input.** `utils/build_balanced_pool.py` defaults `--ride_attrs` to
   `paper_assets/v14d_ride_attrs.json`, which was not shipped. Copied in
   (1.7 MB). The four other unresolved asset paths were checked and are all
   *outputs* the scripts write, not inputs.
3. **`.gitignore` would have dropped the core artifact.** The parent repo's
   ignore rules (`checkpoints/`, `analysis/`) match
   `preprocessing/checkpoints/pca_basis.pt` — the frozen PCA basis — and
   `labels/human_tiers.csv`. Harmless once the folder is its own repo, but a
   silent data-loss trap if anything is added from the parent. Added explicit
   negations for the basis, the labels and `paper_assets/`.
4. **`requirements.txt` was inherited and wrong.** It listed `flask`,
   `flask-socketio`, `nvidia-tensorrt`, `pycuda`, `onnx*`, `dashscope`,
   `pycocotools`, `dominate`, `starlette` and `lmdb` (whose dataset this
   cleanup deleted) while **omitting `torch` and `zarr`**. Rewritten from the
   29 third-party imports the code actually makes, grouped by purpose. The
   flash-attn note is accurate: the causal model every released config uses
   runs on PyTorch `flex_attention` and never calls it; only the unused
   bidirectional `WanModel` asserts it.
5. **`launch_inject.sh` omitted `noadaln`** — 6 of the 7 runs, though "No AdaLN"
   is a paper ablation. Added; the script and its `bash -n` check pass.

Also verified clean: no TODO/FIXME/XXX, no `.md` files, no `__pycache__`, all 8
configs resolve under placeholder env vars, and every referenced sbatch script
exists. Baseline re-saved on the final tree (job 5838314, 7/7 live).

## D42. Preprocessing validated; `pre_encode_motion.py` restructured
All 8 `preprocessing/` modules now import cleanly on a CPU-only node. Seven did
already; `pre_encode_motion.py` did not, and the reason was structural rather
than environmental: it had **no `main()` and no `__name__` guard**, so the entire
ride-processing loop — including
`torch.hub.load("cotracker3_offline").to("cuda")` — executed at import time.
Any import, on any machine without a visible GPU, raised
`RuntimeError: No CUDA GPUs are available` before the script could do anything,
and even `--help` would have tried to download and load a model.

Fixed by wrapping the loop in `main()` under `if __name__ == "__main__":` (so the
model loads inside it) and resolving `device` to CPU when CUDA is absent. The
operations, their order and the constants are untouched — this only stops the
module from running as a side effect of being imported. It now matches how every
other preprocessing script in the release is structured.

`fit_pca_basis.py` was separately validated by actually running it against the
real corpus — see D41.

## D43. Local evaluation takes priority over the HPC testbench
Per instruction: the evaluation of record was produced on the local machine, so
`evaluation/quality/` (from the local-agent merge) is authoritative and
`analysis/testbench_v2/` is the superseded HPC-era version.

Mapped the dependency before touching anything. The local suite is largely
self-contained via its own `fleet_common.py`, but it does import four modules
from the HPC testbench, which therefore cannot simply be deleted:

| kept (local suite imports it) | superseded (15 files) |
|---|---|
| `fleet.py`, `groupnorm.py` | `noop_*` (7), `scorecard.py`, `composite.py` |
| `geometry_metrics.py` (`DepthField`) | `bradley_terry.py`, `judge_gate.py`, `vlm_pairwise.py` |
| `style_shift.py` (`VGGStyle`, `gram_distance`, `read_frames`) | `cpu_metrics.py`, `gpu_metrics.py`, `mangle_metrics.py`, `map_real_refs.py` |

**Also found: the merge shipped only 21 of the 113 scripts in `grids/eval/`, and
the missing ones include every generator for the paper's validation figures.**
Confirmed by mapping figure → script in the local tree:

| figure | generator (present locally, not shipped) |
|---|---|
| `hf_distribution.png` | `hf_distribution.py` |
| `hf_fleet_distribution.png` | `hf_fleet_distribution.py` |
| `hf_validation_internal.png` | `hf_ablation_reels.py` |
| `geometry_validation_fleet_*.png` | `geom_fleet_reels.py` |
| `scene_validation_fleet_*.png` | `scene_fleet_reels.py` |
| `style_validation_fleet_*.png` | `style_fleet_reels.py` |
| `conjuring_validation.png` | `conjuration_fleet_reels.py` |
| stationary numbers | `stationary_{fvd,refmetrics,freeze,signs}.py` |

I copied these in, then **reverted the copy** on instruction that the local eval
will be pushed properly. The map above is recorded so the next push can be
checked for completeness — these are the scripts that must arrive for the
release to regenerate the paper. Note they carry `/home/ashish/...` defaults
that will need the same de-anonymisation treatment as D39.

## D41. FINDING: the shipped PCA basis cannot be regenerated by the shipped fitter
Ran `preprocessing/fit_pca_basis.py` against the real motion corpus
(`/projects/u6ex/fbots/frodobots_motion`, 7,590 `motion.npy` files), 400k frames
randomly sampled across files by size, and compared the result to the shipped
`preprocessing/checkpoints/pca_basis.pt`.

**The components reproduce essentially exactly:**

| | |cosine| |
|---|---|
| PC0 (throttle) | 0.999999 |
| PC1 (steer) | 0.999993 |
| PC2–PC7 | 0.9985 – 0.99995 |
| top-8 subspace (min principal angle) | 0.9986 |

**The mean does not.** Direction agrees (cosine 0.999999) but magnitude is off by
15x: shipped `||pca_mean|| = 2.200`, refit `= 33.256`. The empirical mean-field
norm of the corpus sits near the refit value, and per-ride norms range from 0.06
to 88.4, so the corpus-wide mean depends strongly on which rides are included —
the shipped value looks like it came from a different (or much smaller, or
stationary-heavy) subset than `motion_root` currently holds.

**Measured impact** on the squashed action vector over 2,000 real frames:

| channel | MAE | correlation |
|---|---|---|
| PC0 throttle | **0.2600** | 0.9964 |
| PC1 steer | 0.0162 | 0.9999 |
| all 8 dims | 0.0623 (max 0.3260) | — |

A 0.26 offset on throttle is large against a dataset range whose maximum forward
command is +0.5. The *shape* of the signal is right (corr 0.996) — it is a
near-constant offset, not noise.

**This does not invalidate the trained models.** Training used the shipped basis
for both the conditioning signal and the critic teacher target, so the runs are
internally consistent. What it means is narrower and still serious: a third
party who regenerates the basis with the shipped script gets a measurably
different throttle channel. Either the fitter's sampling/corpus differs from
what produced the shipped file, or the shipped mean was computed on
already-centred or differently-scaled data (the retired ss_vae hparams carried a
`MAG_SCALE: 0.75`, which does not account for 15x).

**RESOLVED — it is a sampling difference, and it does not matter.** Sweeping the
mean over motion-filtered subsets of the corpus (28,975 frames, 120 rides)
reproduces the shipped value exactly where you would expect:

| subset | ‖mean‖ | cos vs shipped |
|---|---|---|
| all frames | 35.35 | 0.9995 |
| slowest 90% | 26.47 | 0.9989 |
| slowest 70% | 14.72 | 0.9946 |
| slowest 50% | 5.19 | 0.9829 |
| **shipped** | **2.20** | — |
| slowest 30% | 0.29 | 0.7585 |

The shipped mean sits between the 30% and 50% quantiles, so it was fitted on a
sample weighted toward low-motion frames — a curated or stationary-heavy pool —
rather than a size-weighted random draw of the whole corpus. Nothing is corrupt;
the two fits simply saw different motion distributions.

**Why it is not a problem.** `pca_mean` only sets the *origin* of the action
space: the encoder computes `(flat - pca_mean) @ pca_comp.T`. Training used the
shipped basis for all three of the conditioning signal, the critic teacher target
and the eval read-back, so the whole system lives in one self-consistent
coordinate frame. Every claim in the paper concerns the *relationship* between
commanded and realized action, and both sides are measured through the same
basis, so a shifted origin cancels. The components — which set the axes, and
therefore what "throttle" and "steer" mean — reproduce to |cos| ≥ 0.9985.

**The only consequence** is for someone who re-fits the basis instead of using
the shipped file and then compares absolute action values to ours: their
throttle origin would differ. Since `pca_basis.pt` ships and is the artifact of
record, that path is avoidable. Worth one sentence in the release —
*use the shipped basis; the fitter is included for provenance, and re-fitting on
a different motion sample shifts the throttle origin* — and no further work.

## D40. `utils/` split into four packages
`utils/` had become four unrelated things in one folder — a model library, an
evaluation pipeline, a plotting suite and a data-prep stage: 29 files, 4,322
lines. Now:

| package | files | lines | role |
|---|---|---|---|
| `utils/` | 11 | 2,242 | training/inference library — imported, never run directly |
| `evaluation/` | 4 | 716 | the eval pipeline producing the CSVs the figures read |
| `figures/` | 8 | 766 | generates 18 of the 21 reproducible paper figures |
| `selection/` | 5 | 551 | window scoring and train/eval pool selection |

Also deleted `decode_seed_windows.py` (47 lines): it wrote seed clips as mp4s
and produced no reported number — the same category as the media renderers
removed in D34.

**Named `evaluation/`, not `eval/`.** A top-level module named after a Python
builtin is a smell linters flag, and the longer name costs nothing.

*Mechanics:* 17 files moved, 12 rewritten (8 import statements plus prose and
two sbatch command lines). Each new directory carries an `__init__.py` matching
`utils/`. The moved scripts derive the repo root as
`dirname(dirname(__file__))`, and the new folders sit at the same depth, so
those paths resolve untouched.

*Verified:* all 79 files compile; import closure clean; and every library module
plus one script from each new package actually imports (15 modules, 0 failures),
including both cross-package imports (`figures.following_family_realdir` →
`figures.following_by_realdir`, `evaluation.chunk_metrics` →
`evaluation.ndof_following`). No probed module left `utils/` or `model/`, so the
baseline should be clean — and `tools/baseline.json` (16:16, 30th) was confirmed
to predate the split (23:06, 31st) before trusting that result, per D38.

1. **Rotate the wandb API key** (in public git history). Not something I can do.
2. **Scene-relocation and conjuration instruments** are described in the paper
   but have no code in `code_release/` — they appear to live in `grids/eval/`
   from the local machine's work. They need folding in for the reproducibility
   checklist to hold.
3. **`analysis/pca_evr.npy`** is not shipped, so `fig:pca_act` cannot be
   regenerated. One file copy fixes it; I have not done it because I could not
   confirm it is the exact array the published figure used.
4. **`lora_judge_ft.py`** — was it ever run? If not it is an unexecuted plan
   rather than a rejected candidate, and shipping it as method is misleading.
5. **`labels/canonical_static_mask.csv`** has no consumer in the release —
   missing script, or leftover?
6. **`launch_inject.sh` omits `noadaln`** (6 of 7 runs) and hardcodes
   `step0005000.pt`. One-line fix, but it changes an eval launcher, so I have
   left it for you.
7. **Retired z2/z7 naming in `score_all_windows.py`** (see D16). Under the 14e
   convention the columns are 0 = throttle, 1 = steer, but the docstring says
   `turn (|z2|)` / `motion (|z7|)` and the code indexes 2 and 7. Either the
   indices are correct and only the labels are stale, or the script predates the
   convention change and scores the wrong columns. I could not determine which
   without a data run, and it feeds the training-pool selection, so it needs your
   answer rather than my guess.
8. **`model/action_model_patch.py` trim** (D17) — done, −1,074 lines, verified
   statically (identical ASTs for the kept functions). Review if you want, but
   no action is needed from you.
