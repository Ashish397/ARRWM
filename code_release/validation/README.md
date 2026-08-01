# Validation

Evidence that this release reproduces the paper's reported numbers, and that
moving the evaluation here did not change what it computes.

Open `report.html` for the visual summary. Raw outputs are in `results/`.

## What was checked

**The paper's quality columns, recomputed.** All five, plus control failure and
the joint legitimacy table, are traced to an exact recipe and reproduce the
published figure for every model:

| column | rule | population | result |
|---|---|---|---|
| geometric corruption | `p_uncanny > 0.5` in `results_external_vlm.csv` | all 256 directional rollouts | 8/8 models exact |
| scene relocation | `consensus_inl < 50` in `fleet_scene_consensus.csv` | feature-valid and non-static | 8/8 models exact |

Both are asserted in `tests/test_paper_tables.py`, so a future edit that swaps
the artefact or the threshold fails the suite rather than quietly moving a
reported number.

The stationary table is the one that does not reproduce from the shipped data:
relocation and geometry trace to the older no-op results, style, conjuration and
high-frequency do not, and the stationary rollouts were re-rendered after those
results were written.

**The move is behaviour-preserving.** `scene_consensus.py`, the instrument that
produces the relocation column, was run from this tree and from the tree it was
developed in over the same validation scenes: **130/130 identical** consensus
counts and relocation flags. A second instrument on a different decode path
(`fleet_common` frame extraction feeding the Laplacian sharpness metric) matched
exactly too, including the de-tiling arithmetic that would otherwise silently
return a neighbouring model's video.

Against the shipped fleet CSV, 124 of those 130 rows are identical and six differ
by 1–7 inliers. All six sit at counts of 12–22, far below the threshold of 50, so
no relocation flag and no reported percentage changes. That is the instrument's
own RANSAC jitter at marginal match counts, and it appears equally in both trees.

An earlier version of this validation tested `fleet_scene_scan.py`, which was
subsequently removed from the release as superseded. Testing a script that does
not ship proves nothing about what does, so it was redone against
`scene_consensus.py`.

**Training.** A separate end-to-end gate (`tests/training_smoke.py`) runs the
real pipeline and compares per-step losses. Step 1 is bit-reproducible and
asserted exactly; later steps carry a tolerance because the backward pass uses
non-deterministic CUDA kernels — measured at ~5e-4 between identical runs, not
assumed.

## What it turned up

An older relocation file, `fleet_scene_reloc.csv`, held rows computed against
rollouts that had since been re-rendered — all 256 yume rollouts and
matrixgame's BL/BR set. The scans append and skip any `(scene, model)` already
present, so those rows were never recomputed and the file mixed two generations
of video.

It feeds no reported number: the paper's relocation column comes from
`fleet_scene_consensus.csv`, recomputed after the re-renders, which is why it
reproduces exactly. The stale file and its producer have been removed from the
release, and the surviving instruments are the ones the paper used.

**The hazard is general.** Any scan that resumes has the same exposure. Delete
the output file before rescanning whenever an input video has changed.

## Files

| file | contents |
|---|---|
| `report.html` | visual summary |
| `results/pytest.txt` | full test run, 35 passed / 6 skipped |
| `results/paper_tables.txt` | recomputed columns against the published ones |
| `results/ab_scene_consensus_release.csv` | scene_consensus run from this tree |
| `results/ab_scene_consensus_original.csv` | the same run from the development tree |

Skipped tests need the rollout videos, which are too large to distribute. Set
`AF_FLEET_DIR` to enable them.

---

# Cluster-side validation

The section above was produced on the machine the evaluation was developed on.
This section is the complementary half: the training and preprocessing path,
exercised on the HPC cluster against the real corpus. Charts in `charts/`, raw
logs in `results/`.

## Preprocessing reproduces the shipped data

`charts/motion_ab.png`. Two rides were re-extracted from their raw `.ts`
recordings with the released `preprocessing/pre_encode_motion.py` and compared
against the `motion.npy` the original pipeline produced for the same rides.

| ride | shape | max abs diff | min per-chunk cosine |
|---|---|---|---|
| ride_22757 | 499x100x3, matches | 0.0024 | 0.99956 |
| ride_22763 | 9x100x3, matches | 0.0040 | 0.99939 |

508 chunks compared, mean cosine 0.999846. Not bit-identical, and should not be:
CoTracker's reductions are non-deterministic across GPUs. The agreement is at
the level that difference explains.

## The action basis reproduces

`charts/pca_basis_shipped_vs_refit.png`. Refitting the frozen PCA basis from
scratch on the real motion corpus reproduces every component direction to
|cos| >= 0.9985 (PC0 throttle and PC1 steer to 0.99999). The mean differs
because the shipped basis was fitted on a stationary-weighted sample; since it
sets only the origin of the action space, and conditioning, critic target and
eval read-back all share the shipped file, it cancels in every
commanded-vs-realized relationship the paper reports. Use the shipped
`pca_basis.pt`; the fitter is included for provenance.

## Training is behaviour-preserving

`charts/ab_compare.png`, raw numbers in `results/training_ab.txt`. The same
recipe was trained from step 0 under the pre-cleanup repo and under this
release: same seed, same node, same GPUs, and the same shared ride manifest so
both iterate rides in identical order. Configs differ only in the deliberate key
rename (`ss_vae_checkpoint` -> `pca_basis_checkpoint`).

Over 12 logged steps: **max absolute difference 0.0053, mean 0.0011**, against
losses spanning 0.05-0.26.

The number that matters is not the magnitude but the sign. The per-step
difference oscillates around zero -- +0.0010, -0.0053, +0.0011 -- with no drift
in either direction. A behavioural change shows up as a consistent sign or a
widening gap; run-to-run non-determinism looks exactly like this. For scale, the
loss moves by more than 0.09 between adjacent steps, so the residual is well
inside the step-to-step variation of a single run.

## Self-checks, run here

| file | result |
|---|---|
| `results/pytest_cluster.txt` | the suite above, re-run on the cluster: 37 passed, 4 skipped (the 4 need the rollout grids, which live on the other machine) |
| `results/imports_and_configs.txt` | 21/21 modules import on a CPU-only node; all 8 configs resolve under the documented env contract |
| `results/release_gate.txt` | no de-anonymisation leaks, shared figure label map intact, no figure script left without its upstream producer |

## What running it actually caught

Three defects survived every static check -- compilation, imports, AST-identity
proofs, 37 passing tests and the numeric baseline probes -- and were found only
by executing the pipeline:

- `NameError: pca_basis_ckpt` in `_build_frozen_evaluator_modules`. Removing the
  ss_vae left one use of a variable whose definition went with it. **The release
  could not train at all.** No test reaches that function.
- `assets/train_windows.json` shipped machine-independent `${DATA_ROOT}/...`
  paths, which JSON does not expand, so the trainer matched none of its 63,792
  windows.
- `wan_model_path` read `DATA_ROOT` while the release documents `WAN_MODELS`
  for checkpoints.

All three are fixed. They are recorded here because they are the argument for
running a release rather than inspecting it.
