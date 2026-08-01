# Release TODO — must clear before the code appendix ships

## Blocking

- [ ] **Rotate the wandb API key.** The key is out of the release tree (grep of
      `code_release/` is clean), but it remains in the *main repo's* git
      history, so it must still be revoked before anything public: wandb User
      Settings -> Danger Zone -> API keys, then supply via `WANDB_API_KEY` or
      `~/.netrc`. Deliberately deferred 2026-07-29; the credential is live
      until this is done. This is a user action, not a code change.
- [x] Strip `wandb_entity` from configs — verified: no `998657fa` /
      `university-of-exeter` match anywhere in `code_release/`.
- [x] Anonymize paths and names — verified: no `/home/<user>`, `/scratch/u6ex`,
      or author/institution strings in `code_release/`; pinned by
      `tests/test_eval_pipeline.py::test_no_author_paths_anywhere_in_the_release`,
      which walks the whole tree.
- [ ] Retrieve `third_party/` from the cluster (Astra `_shims`,
      `worldcam_runner.py`, WorldPlay JSON pose driver, Matrix-Game and Yume
      mappings). The appendix's baseline-control-interface section describes
      code that exists only there. Still absent from the release.

## Verification

- [ ] Extract the finished zip to a clean directory and run from *there*, in a
      fresh env: PCA fit reproduces the shipped basis; training smoke config
      runs ~10 steps; one eval script runs end to end on shipped example data.
      (The in-tree equivalents pass — 63 tests — but the from-zip fresh-env run
      has not been done.)
- [x] Grep the output tree for the key, the paths, and the names — clean, and
      now enforced by a test rather than a one-off grep.

## Paper consistency

- [ ] Reproducibility checklist: preprocessing-code answer can honestly flip
      `no` -> `yes` (checklist edits are handled elsewhere, not in this repo).
      Basis: all pipeline stages ship under `preprocessing/` (captioning, text
      encoding, video->latent, motion tracks, PCA fit) plus `selection/`; all
      import cleanly given the release requirements and documented
      `DATA_ROOT`/`WAN_MODELS`; the HPC side reproduced the shipped
      `motion.npy` through `pre_encode_motion` (cosine 0.99939). The script
      names in the old item text were the pre-restructure ones.
- [ ] Confirm `LICENSE.md` permits redistributing the vendored `wan/` code.
      The release ships CC BY-NC-SA 4.0 at top level; `wan/modules/*` carry
      their upstream (Alibaba/Apache-2.0) headers. Apache-2.0 permits
      redistribution, but the tree should state the dual licensing explicitly —
      a `wan/LICENSE` or a NOTICE line in the top-level LICENSE saying `wan/`
      remains Apache-2.0.
