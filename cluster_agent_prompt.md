# Task for the cluster-side agent (Isambard, `/scratch/u6ex/as1748.u6ex/ARRWM`)

## Context

We are assembling a **code appendix** for an AAAI submission (the "Action
Forcing" paper — world model trained on unlabelled FrodoBots video via a
CoTracker+PCA egomotion basis). The appendix must contain the code for both
**training and validation**, be anonymous, and be runnable by a reviewer.

The plan is to export a curated, cleaned tree into a **fresh repo with no
inherited git history** — not a filtered clone — because the wandb key and the
cluster paths are in the existing history.

A second agent is working on the local workstation, which holds
`causal_lora_step0005000.pt` (the checkpoint the paper evaluates), a local
Wan2.1 copy, one encoded zarr ride, and an RTX 5090. **All refactoring,
tree-building and verification happens locally.** Your job is the things that
can only be done on the cluster.

## Your tasks

### 1. Commit your working tree

Commit whatever uncommitted work exists, so nothing is lost and so the local
agent can diff against a stable reference. Before committing:

- Report the current branch and `git status --porcelain` summary.
- **Do not** sweep large outputs into git — no rollout mp4s, reels, model
  weights, HF caches, venvs, or `logs/`. Add them to `.gitignore` instead. The
  existing `.gitignore` documents the convention: eval **scripts, tables and
  result CSVs** are tracked; heavy artefacts are not.
- Note anything you deliberately left untracked and why.

Then push if there is a remote, and report the resulting commit SHA.

### 2. Inventory and package `third_party/`

`third_party/` exists only on the cluster and holds the authored baseline
control interfaces the paper's appendix describes:

- Astra: `_shims/`, backward `cam_type` branches 8/9/10
- WorldCam: `worldcam_runner.py` with the eight-direction trajectory generator
- HY-World 1.5 WorldPlay: the JSON pose interface driver (diagonals)
- Matrix-Game 2.0: the keyboard/mouse command mapping
- Yume-1.5: the caption clause mapping
- minWM: the pose-increment driver

For each of the six, report:

- the **file paths** holding *our authored* code (as opposed to the upstream
  release, which we must not redistribute),
- roughly how entangled it is with the upstream checkout — i.e. can our code be
  lifted out as standalone files plus documented invocation, or does it only
  make sense as a patch against their repo,
- any hardcoded absolute paths, checkpoint locations, or credentials.

Then produce **`third_party_authored.tar.gz` containing only our authored
code** — no upstream repos, no weights, no venvs, no outputs. It should be
small (single-digit MB). Report its path and size so the local agent can pull
it.

If any of the six turns out to have no separable authored code, say so plainly.
That changes what the paper's appendix can claim, and we need to know now.

### 3. Find the Table 5 aggregator (highest value)

The paper's headline table reports **Control fail %** and **Legitimate %** —
a rollout is "legitimate" if it follows the command (realised motion within 90°,
not near-static) **and** passes all five quality axes (style shift, geometric
corruption, scene relocation, conjuration, HF degradation). Ours (Default)
= 73% legitimate, 0% control fail; minWM = 59% / 22%.

The local checkout contains **no script that computes this joint**. Nothing in
`grids/eval` or `utils` combines the 90°-control criterion with the five axes,
and the local `final_scorecard.csv` has a different schema and different numbers
(astra reloc 62 vs the paper's 67), so it is from an older pass.

Please determine which is true:

- (a) an aggregator script exists on the cluster — find it, and identify the
  exact input CSVs it consumes;
- (b) it was done ad hoc (shell one-liner, notebook, pandas in a REPL) — say so,
  and recover whatever remains: shell history, notebook, the CSVs the numbers
  came from;
- (c) it was assembled by hand from per-axis outputs — say so.

Also locate the **CSVs that actually produced the paper's final tables**
(Table: quality failure rates; Table: stationary failure rates; Table:
quality denominators — feature-valid 240 / per-model active counts). Report
their paths, and confirm whether they match the numbers in the paper. This
matters more than anything else in this list: the checklist claims all analysis
code is included, and this is the table the headline result rests on.

### 4. Report local↔cluster divergence

The fleet runs happened on the cluster, so the cluster's copies of the analysis
scripts may be **ahead of** the local ones. For these, report whether the
cluster version differs from the local committed version, and if so paste the
diff (or summarise if large):

- `grids/eval/`: the `blind_*`, `conjure_*`, `fleet_*`, `hf_*`, `melt_*`,
  `stationary_*` scripts, plus `composite.py`, `ensemble_table.py`
- `utils/`: `wedge_plots.py`, `following_select.py`, `following_by_realdir.py`,
  `following_family_realdir.py`, `following_phaseA.py`, `response_complete.py`,
  `response_curves_eval.py`, `pca_components_fig.py`

We must ship the version that reproduces the published figures. Where they
differ, say which one produced the paper's numbers, if you can tell.

### 5. Confirm figure provenance

Using the fleet outputs you have (the ~3.8 GB of baseline rollouts and our
rollouts), confirm which script regenerates each of these, and whether it still
runs:

`wedge_all_g0.png`, `wedge_all_g1.png`, `wedge_all_g6.png`, `wedge_g0.png`,
`wedge_g1.png`, `following_8node8pca_by_REALdir{,_select}.png`,
`following_16node_by_REALdir.png`, `following_noadaln_by_REALdir.png`,
`following_FAMILY_{injection,nodes,encoders}_by_REALdir.png`,
`response_complete{,_steer}.png`, `response_curves_eval_{throttle,steer}.png`,
`pca_components_flowfields.png`, `stationary_wedges.png`, `pca8_showcase_2.png`

`pca8_showcase_2.png` in particular has no identifiable producer locally — if it
was assembled by hand, say so.

You do **not** need to re-run the full fleet. Confirming provenance and
runnability is enough.

## Explicit non-goals — please do NOT

- **Do not refactor, reformat, split, or rewrite any code.** The local agent is
  doing that, verified against fixed-seed numerical equivalence, and parallel
  edits to the same files would conflict and would not be verifiable. Report
  problems; don't fix them.
- **Do not delete anything**, including apparently dead code or stale outputs.
- **Do not rotate or strip the wandb key.** Rotation is deliberately deferred by
  the author and tracked in `RELEASE_TODO.md`. Just don't introduce new copies.
- **Do not build the export tree** or start a new repo. That happens locally.
- **Do not re-run training or the full evaluation fleet.**

## What to report back

A single written report covering, in order:

1. Branch, commit SHA, what you committed, what you deliberately left out.
2. `third_party/` inventory per baseline + path and size of
   `third_party_authored.tar.gz`.
3. Table 5 aggregator: (a), (b) or (c), with paths and the input CSVs.
4. Divergence list: which scripts differ, and which version is authoritative.
5. Figure provenance table: figure -> script -> runs / doesn't run / no producer.
6. Anything you found that we have not anticipated — especially analysis whose
   code no longer exists, or figures whose inputs you cannot locate.

Be blunt about gaps. A known missing aggregator is a fixable problem; one we
discover after submitting the appendix is not.
