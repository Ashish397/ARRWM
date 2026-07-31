# Release TODO — must clear before the code appendix ships

## Blocking

- [ ] **Rotate the wandb API key.** `wandb_key: 998657fa...` is committed in
      plaintext across ~79 config files and is in git history, so deleting the
      lines does not retract it. Revoke via wandb User Settings -> Danger Zone
      -> API keys, issue a new one, and supply it via `WANDB_API_KEY` or
      `~/.netrc` instead of in-repo. Deliberately deferred 2026-07-29; the
      credential is live until this is done.
- [ ] Strip `wandb_entity: ashish397-university-of-exeter` from configs — it
      deanonymizes an anonymous AAAI submission independently of the key.
- [ ] Anonymize paths and names in the exported tree: `/home/ashish` (44
      tracked files), `/scratch/u6ex/as1748.u6ex` (~965), and
      `ashish|exeter|isambard` (~133).
- [ ] Retrieve `third_party/` from the cluster (Astra `_shims`,
      `worldcam_runner.py`, WorldPlay JSON pose driver, Matrix-Game and Yume
      mappings). The appendix's baseline-control-interface section describes
      code that exists only there. Cluster was down 2026-07-29.

## Verification

- [ ] Extract the finished zip to a clean directory and run from *there*, in a
      fresh env: PCA fit reproduces the shipped basis; training smoke config
      runs ~10 steps; one eval script runs end to end on shipped example data.
- [ ] Grep the *output* tree (not the source) for the key, the paths, and the
      names.

## Paper consistency

- [ ] Reproducibility checklist: flip the preprocessing-code answer from `no`
      to `yes` once `preprocess/` is confirmed present and runnable
      (`pre_encode.py`, `pre_encode_motion.py`, `pca_motion.py`,
      `score_all_windows.py`, `build_weunz_manifest.py`,
      `harvest_backward_windows.py`, `build_balanced_pool.py`).
- [ ] Confirm `LICENSE.md` permits redistributing the vendored `wan/` code.
