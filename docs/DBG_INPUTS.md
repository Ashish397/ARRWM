# DBG_INPUTS — flag-gated dump of what the DMD scorer and GAN disc actually see

Motivation: under rolling training the suspicion is that the clean-match
references fed to the DMD scorers (`clean_x`) and the GAN discriminator's
"real" transition pairs are bad (temporally misaligned, wrong content, or
degenerate). This tool decodes the EXACT tensors those losses consume to a
single side-by-side grid video every N steps, so the references can be
eyeballed against the student band.

## Flag

`debug_dump_scorer_inputs_every` (int, default `0` = off; pass via
`--override` / `DEXTRA`). Cadence is DEBT-based (same rationale as the R1
cadence fix): a dump fires when at least N trainer steps have elapsed since
the last one, because gen/GAN work only runs on generator iterations
(`dfake_gen_update_ratio`) and a plain `step % N == 0` can be missed
forever when the residues never align. Consequence: the FIRST gen step
after (re)start always fires — useful for smoke checks.

Everything is main-rank-only on the consume side, detached, copied to CPU
at stash time, and try/except-wrapped end to end: the dump can never
crash or perturb training (no training tensor is modified; only reads).

## Where the stashes are written

* DMD side — `model/dmd_action_forcing.py`, in
  `compute_generator_loss_streaming` immediately after
  `f42 = self._build_42f_scoring_inputs(chunk, info)`: detached fp32 CPU
  copies of `noisy_x`, `clean_x`, `gt_target` → `self._dbg_scorer_dump`.
  Within a firing step the stash is overwritten per rolled chunk, so the
  LAST (deepest-drift) chunk of the step wins — the interesting one under
  rolling.
* GAN side — `trainer/causal_action_forcing_train.py`,
  `_ladd_run_pair_mode`:
  * matched branch (`ladd_gt_transition_match=true`, what the rolldbg run
    uses): inside `_run_disc_updates` at `_it == 0`, right after `_rn`
    (matched reals, post CARN/noise/diff-aug) and
    `_fk = _m_noise(fake_chunks_det)` are built → `self._dbg_gan_dump`
    with `mode="<pair_mode>/match"`.
  * positional branch: right before `combined_in` is assembled in the
    D-update loop (`real_chunks_det_noisy` / `fake_chunks_det_noisy`),
    `mode="<pair_mode>/positional"`.
  Both stash points are PRE-wavelet — the wavelet-HF transform lives
  inside the disc forward — so the slabs decode through the VAE directly.
  Slabs are `[n_rows, 2*npb(=6), 16, 60, 104]` latent transition pairs.

## Where the video is written

`trainer/causal_action_forcing_train.py :: _maybe_dump_debug_inputs`,
called once per trainer step from the main loop (right before the
sample-video block). It decodes each latent slab with the SAME VAE pathway
as the sample videos (`vae.decode_to_pixel(lat, seed_first=True)`) and
writes ONE grid mp4:

```
<log_dir>/samples/dbg_inputs_step<NNNN>.mp4
```

## Row layout (top → bottom), labelled by 8px LEFT-BORDER colour

| # | border  | row              | content |
|---|---------|------------------|---------|
| 1 | RED     | `dmd_noisy_x`    | the scorer's noisy-half CONTENT (pre-noising): GT context frames + the rolled student chunks, as assembled by `_build_42f_scoring_inputs` (21 latent frames) |
| 2 | GREEN   | `dmd_clean_x`    | the clean half the scorers are conditioned on — under the rolldbg config this is the matched + drifted GT (`dmd_42f_clean_match/drift` chain) (21 latent frames) |
| 3 | BLUE    | `dmd_gt_target`  | GT at the student chunk's window positions (chunk-size latents; short — freeze-frame padded in time) |
| 4 | YELLOW  | `gan_real_pairs` | up to 3 of the disc's REAL transition slabs (6-latent pairs), hstacked left→right |
| 5 | MAGENTA | `gan_fake_pairs` | up to 3 of the disc's FAKE transition slabs, hstacked left→right |

Rows shorter in time are padded by repeating their last frame; narrower
rows are padded with black on the right. DMD rows come from the last
rolled chunk of the firing step; GAN rows from the first D-update of the
firing step (they can come from different steps if the GAN skipped the
step where the DMD stash fired — the filename uses whichever stash carries
a step first, DMD side preferred).

## What to look for

1. **Temporal misalignment between the student band and clean_x** (rows
   1 vs 2). The 42f contract shifts clean_x back `npb` frames from the
   noisy half; with `clean_match + clean_drift(3 chunks) + couple_rope`
   the matched window should still track the noisy half's content up to
   that shift. If row 2 shows a visibly different scene / different
   time-of-ride than row 1's GT context, the clean-match retrieval is
   feeding the teacher a wrong reference and the DMD gradient pulls the
   student toward unrelated content.
2. **gt_target sanity** (row 3 vs the newest frames of row 1): row 3 must
   be the GT continuation at exactly the supervised chunk's positions.
3. **GAN 'real' pairs plausibility** (row 4): each 6-frame slab should be
   a plausible GT transition (smooth ego-motion across the 2-chunk seam).
   Matched retrieval + CARN-former means the FIRST half may look slightly
   degraded on purpose (Req-1 drift cap); it must NOT be a different
   scene, a static repeat, or garbage.
4. **Whether the fake pairs are the flash slab** (row 5): under
   `flash_dmd_enabled=false` (this recipe) the fakes must be the
   student's own rolled chunks — i.e. row 5 should look like row 1's
   newest chunks. If row 5 looks like a re-noised / flash-t rendering
   instead, the disc is scoring the wrong fake tensor.

## Debug run

rolldbg (clone of rolllong, 2-node holder, resume from the
dmd3kl_GAN step-300 checkpoint symlinked as step 200,
`debug_dump_scorer_inputs_every=50`):
`logs/dmd10k_rolldbg/dmd10k_rolldbg_h6092099_<STAMP>/samples/dbg_inputs_step*.mp4`
