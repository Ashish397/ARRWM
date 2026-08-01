# Verification status check — training and preprocessing side

We are splitting the code-release verification: I am covering evaluation
(instruments, paper tables, the reference artefacts) and you are covering
training, preprocessing and release hygiene. I need to know what on your side is
**actually verified by running it** versus checked by reading, so the release
notes and the reproducibility checklist say only what we can support.

Please answer each item with one of: **verified by running**, **checked
statically only**, **not checked**, or **not applicable / superseded** — plus
what you ran, if anything. Blunt is more useful than reassuring; an unverified
item we know about is fixable, one we discover after submission is not.

## 1. Preprocessing — has any of it been executed end to end?

Nothing in `code_release/preprocessing/` has been run on my side. Specifically:

- `pre_encode.py` / `pre_encode_local.py` / `pre_encode_direct.py` — video to
  Wan VAE latents. Three variants ship; which one produced the zarrs actually
  used, and do the others still work or are they superseded?
- `pre_encode_motion.py` — CoTracker grid displacements to `motion.npy`.
- `ride_level_caption.py` + `run_full_dataset_captioning.py` — InternVL3
  captions.
- `pre_encode_text.py` — Wan T5 embedding of those captions.
- `fit_pca_basis.py` — the PCA basis fit.

**A specific defect I fixed, please confirm you agree.** `pre_encode_motion.py`
set `grid_size = 20` at module scope, overriding the function default of 10.
That emits 400 tracked points per chunk, but every shipped `motion.npy` holds
100 and the PCA basis is fitted on 200 = 100 points x (dx, dy), which is also
what the paper states. I set it to 10. Copies elsewhere carry 20 and 30 (the 30
writing to a separate `frodobots_motion_30`), so I read those as later
experiments rather than the released setting — please confirm, because if any
shipped motion data came from a 20 or 30 grid the basis and the paper disagree.

**Second item to confirm.** `fit_pca_basis.py` does not reproduce the shipped
basis. The basis that every reported number uses lives in the ss_vae checkpoint
(`pca_mean` / `pca_comp`), not in the `pca.npz` that script writes; the two are
different fits, and encoding a zero displacement field through the npz gives
`a_null = (-0.339, -0.020)`, a hard reverse, against the paper's
`(-0.0234, -0.0013)` from the checkpoint. Has the basis been refitted from the
corpus and confirmed to reproduce the checkpoint, or does the script remain
documentation of the method only?

## 2. Training — does the released code still train, after all the merges?

- Does `train.py` with each of the released configs start and run steps on the
  cluster as merged? A single successful step per config would settle it.
- Is there a behaviour gate on your side — anything that would catch a cleanup
  changing the computation? I had one comparing per-step losses, but it was
  built against my standalone tree and is stale after the merge, so I am not
  currently able to claim the merged trainer is unchanged.
- Training is not bit-reproducible past step 1 in my measurements: identical
  code and environment diverge by ~5e-4 in loss from step 2, because the
  backward pass uses non-deterministic kernels. Does that match what you see? It
  affects how the seeding answer in the checklist should be worded.

## 3. The `_cont` configs

The released configs fold the `_cont` settings (`glitch_mask_period: 151`,
`fixed_shuffle: true`) in from step 0, so each variant is one self-contained
file. I verified every training-relevant key against base-merged-with-cont, but
**never by running** — the equivalence is textual, not empirical.

Two things to confirm:

- Does that match your understanding of what actually ran, i.e. Default,
  Batch64, Batch16, pca4 and pca2 had the filter off for roughly the first 2000
  steps and on afterwards, while No-Act-Tokens and No-AdaLN had it on from step
  0?
- You restored `configs/causal_lora_diffusion_teacher_v14e*.yaml` in a recent
  commit. Are those the intended released configs, or do they coexist with the
  cleaned single-phase ones? Right now both naming schemes are present and I do
  not want two sets of configs in the release disagreeing about the recipe.

## 4. Release hygiene

- Has the wandb key been rotated? It is still in git history, so deleting the
  lines does not retract it. Tracked in `RELEASE_TODO.md`; deliberately
  deferred, but it should not ship un-rotated.
- Does the manifest scan still take tens of minutes? It scans every zarr under
  `encoded_root` regardless of which rides the window manifest names; over 2,911
  rides it did not finish in 25 minutes here. A reviewer pointing it at a full
  corpus will conclude the release hangs. I did not change it, because
  restricting the scan changes which rides enter the pool and is not provably
  behaviour-neutral.
- `analysis/pca_evr.npy` — you added it for `pca_components_fig.py`. Does that
  figure now regenerate?

## 5. Anything I broke

I deleted files in a legacy-cleanup pass and you restored several
(`ndof_following.py`, `chunk_metrics.py`, `inject_eval.py`) — thank you, and my
deletion was wrong; I judged them from their docstrings without the context you
had. Is anything else still missing that I removed? The commit is
"Keep only the local evaluation; delete the cluster's legacy copies".

Also please confirm the removal of `analysis/testbench_v2/` and
`analysis/style_shift/` is right. My reading is that they are a superseded
evaluation generation producing no reported number, and that the current numbers
come from the workstation instruments, but you have context on the no-op and
scorecard scripts in there that I do not. In particular `noop_scorecard.py` and
`noop_final_eval.py` — is the paper's stationary table computed by those, or by
the workstation `stationary_*` / `noop_*` scripts I kept?

## What I can tell you from my side

Four of the paper's columns now reproduce exactly from the shipped files under a
stated rule, and are pinned by tests: geometric corruption
(`p_uncanny > 0.5`, all 256), scene relocation (`consensus_inl < 50`, active
population), style shift (`dino_drift > 0.72`, active population) and control
failure (near-static or realised motion more than 90 degrees from the command).
The per-model active counts also match the published denominators. Conjuration
and high-frequency degradation are not yet traced, so the joint legitimacy
figure is still out of reach.

One hazard worth knowing on your side too: three evaluation instruments wrote a
shipped reference artefact by default, so running any of them overwrote a file
backing a published number. All three did it here in one session. They now take
an explicit output path.
