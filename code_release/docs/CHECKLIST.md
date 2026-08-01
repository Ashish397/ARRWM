# AAAI reproducibility checklist — supporting files

Where each answer is backed up in this release. Entries marked **gap** are
answers the code does not currently fully support; they are open items, not
claims.

## Theoretical contributions

| checklist item | answer | support |
|---|---|---|
| Assumptions stated formally | yes | Paper Sec. Method; the control-affine hypothesis and the tracked-motion model |
| Claims demonstrated empirically | yes | `eval/` produces every reported figure and table |
| Code used to eliminate or disprove claims included | yes | Rejected instruments are retained with provenance banners, e.g. `eval/geometry_metrics.py`, `eval/blind_depth.py` (depth curvature, AUC 0.52, unused). **Partial gap:** `blind_depth.py` needs a `fleet` harness module not in this release, so it does not run as shipped. |

## Dataset usage

| checklist item | answer | support |
|---|---|---|
| Motivation for the dataset | yes | Paper Sec. Introduction |
| Datasets from literature cited | yes | FrodoBots-2K, cited |
| Datasets publicly available | yes | FrodoBots is public; `assets/train_windows.json` gives the exact ride/window split so the subset is reconstructible |

## Computational experiments

| checklist item | answer | support |
|---|---|---|
| Hyperparameter ranges and selection criterion | yes | `configs/` — seven self-contained variants; the ablation axes are batch size, critic-supervision dimensionality, and conditioning pathway |
| Pre-processing code included | **yes** (was "no") | `preprocessing/`: `pre_encode.py` (video → Wan VAE latents), `pre_encode_motion.py` (CoTracker → displacement fields), `fit_pca_basis.py` (basis fit), `score_all_windows.py`, `build_weunz_manifest.py`, `harvest_backward_windows.py`, `build_balanced_pool.py`. **Caveat:** `fit_pca_basis.py` documents the method but does not reproduce the shipped basis (`docs/DECISIONS.md` §1), so "partial" is the defensible answer if that is not resolved before submission. |
| All source code for experiments and analysis included | yes | `trainer/`, `model/`, `utils/`, `eval/`. **Gap:** the joint legitimacy table (control-fail / legitimate) has no located producer script; four of the five quality-table columns have no confirmed recipe. See `docs/DECISIONS.md` §9. |
| Code will be publicly available | yes | — |
| New methods commented with references to the paper | yes | `trainer/causal_diffusion_teacher_train.py::_compute_action_critic_losses` cites Eqs. 7–9; `utils/zarr_dataset.py` documents the action encoder; `tests/test_action_space.py` asserts the published constants |
| Seed-setting described sufficiently | yes | `seed: 0` in every config; `utils/misc.py::set_seed`. **Qualification:** seeding makes step 1 exact but the pipeline is not bit-reproducible thereafter — see README, Reproducibility notes |
| Computing infrastructure specified | yes | `docs/RUNNING.md` |
| Evaluation metrics formally described | yes | Paper appendix; implemented in `eval/` |
| Number of runs per reported result | yes | Paper Sec. Evaluation: 32 contexts × 8 commands = 256 directional rollouts per model, plus 32 stationary |
| Variation / confidence reported | yes | SEM bands on response curves; IQR arcs on the wedges |
| Statistical significance tests | yes | Paper appendix, instrument validation against human annotation |
| All final hyperparameters listed | yes | `configs/`, verified against the originals by `tests/test_configs.py` |

## Verification available to a reviewer

```bash
pytest tests/          # 44 tests: action space, modules, split, config equivalence
```

Runs in about a second with no GPU, weights or data. It asserts, among other
things, that the action basis reproduces the paper's `a_null`, that the split
matches the reported 2,577 rides / 63,792 windows, and that every released
config is semantically identical to the config that was actually run.

With data and a GPU:

```bash
python tests/training_smoke.py --config <reduced> --logdir <dir> --check
```

runs the pipeline end to end and compares per-step losses against a recorded
reference.
