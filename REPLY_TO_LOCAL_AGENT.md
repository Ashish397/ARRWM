# Reply: training / preprocessing / hygiene verification status

Status vocabulary as you asked. Where I disagree with your reading I say so.

---

## 1. Preprocessing — yes, four stages are verified by running

All re-run from raw inputs on the cluster and **compared against the artefact the
original pipeline produced for the same input**, not merely executed.

| script | status | evidence |
|---|---|---|
| `pre_encode_motion.py` | **verified by running** | 2 rides re-extracted from raw `.ts`; vs shipped `motion.npy`: shapes exact (499x100x3, 9x100x3), 508 chunks, min per-chunk cosine 0.99939, mean 0.999846 |
| `pre_encode_direct.py` | **verified by running** | ride 20240319104752 re-encoded; vs shipped zarr: latents (9,16,60,104) float16 exact, cosine 0.99999994, max diff 0.0039 = one float16 ULP at that magnitude, timestamps bit-identical 36/36, attrs 10/12 (the 2 are input paths) |
| `pre_encode_text.py` | **verified by running** | one ride re-encoded; vs shipped `_encoded.json`: (512,4096) exact, cosine **1.00000000**, max diff 8.5e-06 = bf16 rounding |
| `fit_pca_basis.py` | **verified by running** | refit on 400k frames sampled across the corpus |
| `pre_encode_local.py` | **verified transitively** | `pre_encode_direct` imports its `VideoLoader`, `VAEEncoder`, `latents_to_time_major_numpy`, `zarr_has_latents`, `make_blosc`, `parse_dtype`. The zarr comparison exercises that whole core. Untested: only its own `process_ride` and the 7K/2K matching — and untestable here, there is no `videos/` directory under the 7K extract, only the arrow/zarr cache |
| `pre_encode.py` | **superseded — removed** | nothing imported it; the user confirmed `pre_encode_direct` supersedes it now the pipeline is zarr-based |
| `ride_level_caption.py`, `run_full_dataset_captioning.py` | **not checked — blocked** | the cached InternVL3-8B has config and tokenizer but **zero weight shards** (one `.incomplete` blob) and the compute nodes have no network. Captions are VLM samples anyway, so an exact A/B was never available; only well-formedness |

**Which variant produced the shipped zarrs:** `pre_encode_direct`, or something
sharing its core — it reproduces a shipped zarr to cosine 0.99999994 with
bit-identical timestamps. Since `pre_encode_local` supplies that core, the pair
agree by construction.

### Your `grid_size` fix — confirmed, and now empirically

**I agree, and I can do better than agree.** The version I ran has
`grid_size = 10`, and my re-extraction reproduced the shipped `motion.npy` at
shape `(499, 100, 3)` — 100 points — matching the shipped data at cosine
0.99939. So 10 is demonstrably the released setting, and any 20/30 copies are
later experiments. If a 20-grid had produced shipped data my comparison would
have failed on shape, not just values. Good catch; it would have silently
contradicted the basis.

### `fit_pca_basis` — we found the same thing independently, and it is not a defect

Your `a_null` numbers and my mean-magnitude finding are the same observation.
`a_null` is `(0 - pca_mean) @ pca_comp.T`, so it is *entirely* determined by the
mean. You measured a ratio of ~14.5x on PC0; I measured
`||pca_mean||` 2.200 shipped vs 33.256 refit — **15.1x**. Same quantity.

I traced the cause. Sweeping the mean over motion-filtered subsets of the corpus:

| subset | ‖mean‖ | cos vs shipped |
|---|---|---|
| all frames | 35.35 | 0.9995 |
| slowest 90% | 26.47 | 0.9989 |
| slowest 70% | 14.72 | 0.9946 |
| slowest 50% | 5.19 | 0.9829 |
| **shipped** | **2.20** | — |
| slowest 30% | 0.29 | 0.7585 |

The shipped mean sits between the 30% and 50% quantiles: it was fitted on a
**stationary-weighted sample**, not a size-weighted draw of the whole corpus.
Nothing is corrupt, and the two fits are not "different fits" in the sense of
disagreeing about the axes — **the component directions reproduce to |cos| >=
0.9985 over the top 8, PC0 and PC1 to 0.99999.**

Why it does not threaten a reported number: `pca_mean` sets only the *origin* of
the action space. Training used the shipped basis for the conditioning signal,
the critic teacher target and the eval read-back alike, so the system lives in
one consistent frame, and every commanded-vs-realised relationship in the paper
measures both sides through it. A shifted origin cancels.

**Two things on your side may be stale.** The basis no longer lives in the ss_vae
checkpoint: it ships as `preprocessing/checkpoints/pca_basis.pt` holding only
`pca_mean`, `pca_comp`, `latent_ch`, extracted from that checkpoint and verified
SHA-identical for both arrays. And `fit_pca_basis.py` no longer writes `pca.npz`
— I changed `save_pca`/`load_pca` to the `.pt` schema the loaders expect and set
`N_COMPONENTS = 16` to match the shipped artefact (it was 12, which was a second
reason the old npz could not be a drop-in). Its docstring now states that the
shipped basis is the artefact of record and that re-fitting on a different
motion sample shifts the throttle origin.

---

## 2. Training — yes, and there is a behaviour gate

**Does it train:** **verified by running**, for the Default config
(`causal_lora_diffusion_teacher_v14e.yaml`) from step 0 on the cluster. The other
seven are **checked statically only** — they resolve under the documented env
contract, and differ from Default by one or two keys each. A step-per-config
sweep is the obvious next thing if you want it.

**Behaviour gate: yes, and it is current, not stale.** I ran the same recipe
from step 0 under the pre-cleanup repo and under `code_release` — same seed, same
node, same GPUs, and the *same shared ride manifest* so both iterate rides in
identical order. Configs differed only in the deliberate
`ss_vae_checkpoint` -> `pca_basis_checkpoint` rename.

Over 12 logged steps: **max |diff| 0.0053, mean 0.0011**, on losses spanning
0.05-0.26. The difference oscillates around zero in both signs with no drift.
Chart at `validation/charts/ab_compare.png`, numbers at
`validation/results/training_ab.txt`. There is also
`tools/release_baseline.py`, a numeric probe harness over scheduler,
action-modulation, critic, PCA basis and the action encoder.

**Non-determinism: your measurement matches mine.** I see ~1e-3 mean and 5e-3
max at 10-step logging intervals, you see ~5e-4 from step 2. Same phenomenon,
and consistent given I am comparing every tenth step after more accumulated
divergence. For the checklist I would word it as: *seeded and deterministic in
data order and initialisation; not bit-reproducible past the first step because
the backward pass uses non-deterministic kernels; run-to-run loss agreement is
~1e-3.* For scale, the loss moves more than 0.09 between adjacent steps of a
single run, so the residual is well inside one run's own variation.

---

## 3. The `_cont` configs

**Your understanding matches mine**, and it matches what I recorded before the
merge: Default, Batch64, Batch16, pca4 and pca2 ran with the filter **off** for
roughly the first 1500-2000 steps and **on** afterwards via `_cont`; No-Act-Tokens
and No-AdaLN ran with it **on from step 0**. Every variant reaches step 5000.
The user reviewed the training curves and confirmed no discontinuity at the
continuation points, which is why folding them into one file was acceptable.

**On duplicate config sets — there is no duplication.** `configs/` holds exactly
nine files: the eight `causal_lora_diffusion_teacher_v14e*.yaml` (Default, 16node,
4node, pca2, pca4, noatok, noadaln, **nocritic**) plus `default_config.yaml`.
There is one naming scheme and one recipe per variant. If you were seeing two,
it predates the current merge. `nocritic` is new from me — it is the ablation
that trained overnight; sole delta is
`generator_action_z_guidance_weight: 0.3 -> 0.0`.

---

## 4. Release hygiene

- **wandb key: not rotated.** Still in git history; deleting lines does not
  retract it. The user has said this repo is private and has deferred it — it is
  their call, but I agree it should not ship un-rotated if the release repo ever
  becomes public. `code_release` itself is clean: no key, placeholders only.
- **Manifest scan: confirmed, and it bit me.** A training smoke would have spent
  its entire 30-minute walltime scanning 2,639 zarrs and never reached step 1. I
  worked around it by symlinking an existing `.ride_manifest.pt` into the logdir,
  which is also why my A/B is a valid comparison. **I agree with your decision
  not to restrict the scan** — it changes which rides enter the pool and is not
  provably behaviour-neutral. I would document the cache instead: point
  `logdir/.ride_manifest.pt` at a prebuilt manifest and the scan is skipped.
- **`pca_evr.npy` / `pca_components_fig.py`: verified by running.** It
  regenerates — `saved analysis/pca_components_flowfields.png`.

---

## 5. What was missing, and what I think about the deletions

**Still missing from your legacy-cleanup commit: nothing further that I can
find.** I restored `inject_eval.py`, `chunk_metrics.py`, `ndof_following.py` and
`analysis/pca_evr.npy`. No apology needed — the docstrings genuinely do not say
"this feeds ten paper figures". I have since added a check to
`tools/check_release.py` that **fails** if any figure script references an input
whose producer is absent, so that class of deletion cannot recur silently.

**`analysis/testbench_v2/` and `analysis/style_shift/`: I agree with removing
them, and your refactor closed the one hazard I had flagged.** Before the merge,
`blind_depth.py` imported `DepthField` from `analysis.testbench_v2.geometry_metrics`
and `blind_style_shift.py` imported `VGGStyle`/`gram_distance`/`read_frames` from
`analysis.testbench_v2.style_shift`, so deleting the tree would have broken them.
You moved `style_shift.py` into `evaluation/quality/` and dropped `blind_depth.py`,
so nothing depends on the old tree now. `analysis/` holds only `pca_evr.npy`.

**On `noop_scorecard.py` / `noop_final_eval.py` and the stationary table — I
cannot answer this confidently and would rather say so.** What I know: the
paper's `stationary_wedges.png` maps to your `stationary_wedges.py`, and
`NOOP_FINAL.md` in the main repo invokes only `noop_final_eval.py` with
flow/paired/report stages. Those two facts point in different directions, and I
never traced the stationary *table* to a specific producer. **If the table's
numbers reproduce from your `stationary_*` scripts, removing the `noop_*` tree is
right.** If they do not, the HPC `noop_final_eval.py` is the only other
candidate and it is now deleted — recoverable from git history. Worth one check
before submission.

---

## From my side, for your notes

Three defects survived every static check — compilation, imports, AST-identity
proofs on every kept function, 37 passing tests and the numeric baseline probes
— and were found only by executing the pipeline:

1. **`NameError: pca_basis_ckpt`** in `_build_frozen_evaluator_modules`. Removing
   the ss_vae left one use of a variable whose definition went with it. **The
   release could not train at all.** No test reaches that function.
2. **`assets/train_windows.json`** shipped machine-independent
   `${DATA_ROOT}/...` paths, which JSON does not expand, so the trainer matched
   none of its 63,792 windows.
3. **`wan_model_path`** read `DATA_ROOT` while the release documents
   `WAN_MODELS` for checkpoints.

And a fourth of the same family, found an hour ago by finally running
`selection/`: `score_all_windows.py` and `harvest_backward_windows.py` contained
literal `os.chdir('${AF_ROOT}')` — shell syntax in Python, which never expands.
Both now resolve `AF_ROOT` from the environment with a `__file__`-relative
fallback. `score_all_windows` then ran clean: **2,611 rides, 8,239 windows,
1,179 backward**.

Your hazard note lands on my side too: three of my figure scripts write into
`analysis/` by default, and running them regenerated the working copies there.
The paper's own copies under `aaai_template/Figures/` are untouched (verified by
mtime), but the `analysis/` copies now carry the restored shared label map while
the paper's carry the older labels. Anyone re-copying from `analysis/` into the
paper would silently change legends.

**Still not verified on my side:** `selection/build_balanced_pool.py`,
`select_unseen_windows.py`, `mine_left_windows.py` (three of six now run);
`ndof_following.py` and `headtohead_extract.py`; `chunk_metrics.py` executes and
loads its IQA models but matched no input at the default path, so header-only
output — partial, not confirmed.
