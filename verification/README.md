# Verification evidence for `code_release`

Visual artifacts you can inspect directly. Each was produced by running the
**released** code against the **real** data on this machine — not by static
analysis. Sits outside `code_release/` because it is evidence for us, not part
of what a reader needs to reproduce the paper.

Regenerate anything here with the commands listed under each section.

---

## `figures/` — the figure layer ✅ PASS

28 PNGs produced by running `figures/wedge_plots.py` and
`figures/response_curves_eval.py` from `code_release` against the real
head-to-head CSVs in `analysis/eval_final/`.

Includes all 5 wedge figures the paper uses (`wedge_all_g0`, `wedge_all_g1`,
`wedge_all_g6`, `wedge_g0`, `wedge_g1`) plus both response curves. Check the
legends: they should read "Ours (Default)", "Ours (batch 64)", "Matrix-Game",
"WorldCam" — the shared label map, which the local-agent merge had reverted and
which has been re-applied.

```bash
cd code_release
AF_ROOT=$PWD/.. PYTHONPATH=. WG_OUT=../verification/figures python figures/wedge_plots.py
AF_ROOT=$PWD/.. PYTHONPATH=. RC_OUT=../verification/figures python figures/response_curves_eval.py
```

## `pca/` — the PCA basis ✅ PASS (finding resolved)

`pca_basis_shipped_vs_refit.png`. Refit the frozen action basis from scratch on
the real motion corpus (7,590 `motion.npy` files, 400k frames sampled across
files by size) and compared it to the shipped
`preprocessing/checkpoints/pca_basis.pt`.

**Left panel — the components reproduce.** PC0 (throttle) |cos| = 0.99999+,
PC1 (steer) = 0.99999, worst of the top-8 = 0.9985. The fitter is correct.

**Right panel — the mean does not.** Shipped `||pca_mean||` = 2.20, refit =
33.26, same direction (cos 0.999999). That residual offset moves the throttle
channel by **MAE 0.260** in squashed action units, against a dataset whose
maximum forward command is +0.5. Steer is unaffected (0.016).

**Resolved.** Sweeping the mean over motion-filtered subsets reproduces the
shipped value where you would expect: all frames 35.35, slowest 90% 26.47,
slowest 70% 14.72, slowest 50% 5.19, **shipped 2.20**, slowest 30% 0.29. The
shipped basis was fitted on a stationary-weighted sample, not a size-weighted
draw of the whole corpus. Nothing is corrupt.

It does not affect any result. `pca_mean` sets only the *origin* of the action
space, and training used the shipped basis for the conditioning signal, the
critic teacher target and the eval read-back alike — so it is one consistent
frame, and every commanded-vs-realized relationship the paper reports measures
both sides through it. The components, which define what "throttle" and "steer"
mean, reproduce to |cos| >= 0.9985.

The one consequence: re-fitting instead of using the shipped file shifts the
throttle origin. `pca_basis.pt` ships and is the artifact of record, so that
path is avoidable; the fitter's docstring now says so. See D41.

```bash
python tools/pca_verify_plot.py \
  --shipped code_release/preprocessing/checkpoints/pca_basis.pt \
  --refit <your-refit>.pt --out verification/pca/pca_basis_shipped_vs_refit.png
```

## `ab/` — training A/B ⏳ PENDING

`ab_compare.png` once jobs 5855705 (`ab_main`) and 5855706 (`ab_release`)
finish. Both run the **same** recipe, seed and node count — one under the
pre-cleanup repo, one under `code_release` — with configs identical except the
one deliberate key rename (`ss_vae_checkpoint` → `pca_basis_checkpoint`).

Three panels: total loss, flow loss, and the per-step difference against a zero
line. **Read it this way** — traces lying on top of each other, and a difference
panel flat at zero, means the cleanup preserved the computation. Any systematic
drift in the third panel is a real behavioural change that the static checks and
the baseline probes missed.

Node count changes the global batch, so this compares the two codebases against
*each other*, not against the historical 8-node run. For scale, that run logged
step 10 → 0.233, step 50 → 0.056, step 80 → 0.133.

```bash
python tools/ab_compare.py logs/ab_main_5855705.err logs/ab_release_5855706.err \
  --labels "main repo" "code_release" --out verification/ab/ab_compare.png
```

---

## Not yet covered

- Inference / rollout (`inference.py`, `pipeline/causal_inference.py`)
- The eval pipeline head (`evaluation/inject_eval.py` → `chunk_metrics.py`)
- Video encoding, motion extraction and captioning in `preprocessing/`
- The 10 paper figures whose generators arrived in the local-agent merge
