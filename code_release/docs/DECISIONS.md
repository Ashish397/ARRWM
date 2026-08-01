# Release decisions taken without sign-off

Each entry is a call made while preparing the code appendix, with the reasoning
and what it would take to reverse. Review and confirm or overturn.

Status key: **[settled]** low risk, proceed unless you object ·
**[confirm]** needs your yes/no · **[paper]** implies a change to the paper text.

---

## 1. The action basis ships from the ss_vae checkpoint, not `pca.npz` — [settled]

`action_query/pca_motion.py` writes `checkpoints/pca_motion/pca.npz` (12
components). The training and evaluation code does **not** read it. The basis
actually used lives inside `action_query/checkpoints/ss_vae_8free.pt`, under the
keys `pca_mean` / `pca_comp` (16 components), loaded via
`utils/zarr_dataset.py:556`.

They are different fits. Encoding an all-zero displacement field through each:

| basis | a_null (throttle, yaw) | meaning |
|---|---|---|
| `ss_vae_8free.pt` | `(-0.0234, -0.0013)` | matches the paper exactly |
| `pca.npz` | `(-0.3390, -0.0201)` | a hard reverse, past the -0.3 training support |

Shipping the npz would have produced a release whose "stop" command drives the
rover backwards. The checkpoint ships as `assets/pca_basis_ss_vae_8free.pt` and
is pinned by `tests/test_action_space.py`.

**Consequence:** `preprocessing/fit_pca_basis.py` (the old `pca_motion.py`) does not
reproduce the shipped basis. It is retained as the documented method, with a
header saying so. Refitting from the full corpus and confirming it reproduces
`a_null` would close this properly — worth doing if time allows.

## 2. Reverse motion is 1.04% of training windows, not "<1%" — [paper]

632 distinct reverse windows out of 60,632 distinct windows = **1.04%**. The
same figure by duration (~55 min against the ~88 h subset) gives 1.04% too. The
paper says "less than 1%" in the abstract and conclusion.

Pinned at the measured value in `tests/test_data_split.py`. Suggest the paper say
"approximately 1%" — the claim is not weakened, and the manifest ships, so a
reviewer can compute it.

## 3. The `_cont` configs are merged into one config per variant — [confirm] [paper]

As instructed. The `_cont` files add `glitch_mask_period: 151` (encoder-glitch
filter), `fixed_shuffle: true`, and a `stop_at_step` cap; the released configs
carry those settings from step 0.

What actually ran (per the checkpoint timeline, corroborated by
`CLEANUP_PLAN.md` Phase 3):

| variant | steps 0 → ~2000 | ~2000 → 5000 |
|---|---|---|
| Default, Batch64, Batch16, pca4, pca2 | filter off, reshuffled | filter on, fixed order |
| No-Act-Tokens, No-AdaLN | filter on, fixed order | same |

So the released single-phase recipe describes exactly what the two architectural
ablations did, and the *final* phase of the other five. A reviewer re-running it
gets a model filtered from step 0 rather than a bit-exact reproduction of the
released checkpoint.

The training curves show no discontinuity at the phase boundary, and `noadaln`
is flat at ~0.298 realised throttle regardless of command — it ignores the action
entirely, which an early-training data-filtering difference is very unlikely to
cause. So the qualitative conclusions look safe.

**Recommendation:** ship the merged config and add one sentence to the appendix
noting the two-phase schedule. That keeps the code simple and the record
accurate. Say the word if you would rather ship both phases explicitly.

Related: `_noadaln`'s `stop_at_step: 3300` is **not** a truncation —
`_noadaln_cont` resumes the same logdir to 5000. All variants reach 5000, as the
paper states.

## 4. Layout mirrors the source repo — [settled]

No re-architecting into a package. Module boundaries stay as they were so the
behaviour-preservation tests remain meaningful and the diff against the verbatim
first commit stays readable. Two renames only:
`action_query/ss_vae_model.py` → `model/ss_vae.py`, and the basis to
`assets/pca_basis_ss_vae_8free.pt`.

## 5. Vendored `wan/` is left untouched — [settled]

7,072 lines of upstream Wan2.1. Not ours to rewrite; edits would break fidelity
with upstream and are out of scope for a paper appendix. Line-count targets
apply to the ~13.5k lines we wrote.

## 6. Run keys are not renamed — [settled]

`pca8_8node` stays `pca8_8node` rather than becoming `Default`. The key is
simultaneously a dict key, a filesystem path component, and a value inside
already-computed CSVs across 40+ files, with no test that would catch a miss —
and a miss shows up as a silently dropped model row in a paper figure. The
reader-facing benefit is available at the label layer instead. (Independently
recommended by two of the five audits in `CLEANUP_PLAN.md`.)

## 7. Baseline harness: mine and the cluster's are merged — [settled]

Two behaviour-pinning harnesses existed. `tools/release_baseline.py` (cluster)
hashes component outputs and resolved configs; `tests/` (here) uses pytest with
JSON goldens and adds property tests — sign, linearity, composition, chunk
independence — plus assertions against the paper's own published constants.

Kept both, since they fail differently: hashes catch any drift at all, property
tests say which invariant broke. The pytest suite is the gate for cleanup.

## 8. Ride-manifest scan is pathologically slow — [confirm]

Building the ride manifest scans every zarr under `encoded_root` regardless of
which rides the window manifest names. Over 2,911 local rides it had not finished
in 25 minutes; restricted to the 16 rides a smoke run needs, it completes
promptly. A reviewer pointing this at a full corpus will conclude the release
hangs.

Restricting the scan to rides named in the window manifest is the obvious fix,
but it changes which rides enter the dataset pool, so it is **not** provably
behaviour-neutral and I have not applied it. Options: (a) leave as-is and
document, (b) add progress logging only, (c) restrict the scan and prove the
resulting window set is identical. Recommend (c) if the proof holds, (b)
otherwise.

## 9. Ten eval scripts could not import; four missing modules shipped, one cannot be — [confirm]

`style_shift.py`, `vlm_external.py`, `pal_local.py` and `melt_vlm_bench.py` were
imported by ten shipped scripts but were not themselves shipped. They now are.

`geometry_metrics.py` imports a module named `fleet` that exists nowhere in the
repository — not under `grids/eval`, `utils` or `analysis`. It appears to belong
to an earlier harness generation whose successor is `fleet_common.py`, which has
an incompatible API. So `geometry_metrics.py` and its dependant `blind_depth.py`
cannot run as shipped.

Both are retained with a header saying exactly that, because they implement the
depth-curvature instrument the paper itself reports at AUC 0.52 and excludes,
and the checklist undertakes to ship rejected-instrument code. If the original
`fleet.py` can be recovered from the cluster, shipping it closes this cleanly.

## 10. Training-time video overlays moved to `trainer/video_logging.py` — [settled]

150 lines of W&B overlay drawing lived in the trainer. Pure diagnostics: no
model effect, no paper figure. Moved verbatim to a sibling module. Verified by
the end-to-end gate.

## 11. Training is not bit-reproducible past step 1 — [paper]

Two runs of identical code in an identical environment: step 1 matches to the
last digit, steps 2 and 3 differ by ~5e-4. The backward pass uses
non-deterministic CUDA kernels (flex-attention backward, 3D-conv atomics), which
seeding cannot fix. Measured on an RTX 5090, torch 2.8 / CUDA 12.8.

The checklist answers "yes" to seeds being described sufficiently to replicate.
That remains defensible — seeds are set and documented — but "replicate" here
means statistically, not bit-exactly. The README states this. Consider one
sentence in the appendix so a reviewer whose loss curve does not overlay ours
knows it is expected.

## 12. Preprocessing tracked a 20x20 grid; corrected to 10x10 — [settled]

`preprocessing/pre_encode_motion.py` set `grid_size = 20` at module scope,
overriding the function default of 10. That emits 400 tracked points, while
every shipped `motion.npy` holds 100 and the action basis is fitted on
200 = 100 x (dx, dy). The paper states a 10x10 grid.

So the script as inherited did not reproduce the data it supposedly generated:
a reviewer rerunning preprocessing would have got motion files the encoder
cannot project. Set to 10, matching the data, the basis and the paper, and
pinned by `tests/test_preprocess_contract.py`.

The 20 looks like a leftover from an experiment. Worth confirming no shipped
motion data came from it — all 400 files sampled here are 100-point, so the
evidence says no.

## 13. ss_vae removed; the golden had to be re-recorded — [confirm]

Removed on request: `model/ss_vae.py`, the `ss_vae` and affine-`pca` encoder
branches, the GAN discriminator heads in `wan_wrapper` (`adding_cls_branch`,
`adding_rgs_branch`, `ResidualMLPBlock`, `RegisterTokens`, `GanAttentionBlock`),
and the DMD module map in `model/__init__.py`, which named 14 modules that were
never shipped and so made `import model` fail on attribute access.

The PCA basis was stored *inside* the ss_vae checkpoint, so it was extracted to
`preprocessing/checkpoints/pca_basis.pt` first and verified identical: same mean, same components,
byte-for-byte the same encoded actions, and the same published
`a_null = (-0.0234, -0.0013)`.

**But the end-to-end gate then failed at step 1**, which nondeterminism cannot
explain. Cause: `load_ss_vae` constructs the model — randomly initialising the
weights before loading them — and that consumes torch RNG draws. Every
subsequent sample, including the training noise and sampled timesteps, shifted
by removing it.

So the reported runs' noise sequence depended on constructing a model the
released configs never use for anything. The action encoding is provably
unchanged; only the random stream moved. The golden was re-recorded on that
basis.

Worth knowing rather than acting on: it means a run cannot be reproduced from
the config and seed alone unless the setup path is identical too. It does not
affect any conclusion, since nothing depends on a particular noise draw.

## 14. Open: four of five quality-table columns have no confirmed recipe — [confirm]

Reproduced exactly from the shipped CSVs: **geometric corruption** =
`p_uncanny > 0.5` over all 256 directional rollouts (every model matches the
published column to the rounding).

Not yet reproduced: style shift, scene relocation, conjuration, HF degradation,
and the joint legitimacy table. Instruments exist for each; the exact thresholds
and populations are not yet pinned down. This matters because the reproducibility
checklist claims all analysis code is included.

Tracked in `RELEASE_TODO.md` in the main repo.
