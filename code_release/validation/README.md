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
