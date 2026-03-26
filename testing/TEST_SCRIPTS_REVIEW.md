# Testing Scripts Review

## Overall Verdict

The three Python scripts in `testing/` are **not all the same kind of test**:

- `test_context_shifted_teacher_forcing.py` is a **real regression/unit test suite** and is broadly apt.
- `test_lockstep_batcher.py` is a **real unit test suite** and is also apt.
- `test_zarr_dataloader.py` is **not really an automated test suite**. It is a **manual integration/debug visualizer**. It is useful, but it should be thought of as a diagnostic script rather than a CI-style test.

So the short answer is:

- two of them are apt as automated tests
- one of them is apt as a manual inspection tool

## 1. `test_context_shifted_teacher_forcing.py`

### Is it apt?

Yes, mostly.

This file is doing the right kind of checking for the causal teacher / teacher-forcing path:

- shifted clean/noisy frame splitting
- teacher-forcing attention-mask visibility
- RoPE temporal offset behavior
- dataset context accounting
- action-dimension slicing
- clean/noisy action alignment
- clean-side adaLN routing
- action-token helper behavior
- PEFT-wrapped clean-side adaLN regression

### Main strengths

- It checks **semantics**, not just shapes.
- It covers the highest-risk logic in the causal teacher path.
- It includes targeted regression tests for the clean/noisy shift and the PEFT clean-side adaLN issue.
- It keeps most tests CPU-friendly and synthetic, which is good for repeatability.

### Limitations

- Some tests are **logic reconstructions** rather than end-to-end execution of the full trainer/model stack.
- `TestActionConditioningModeParsing` mostly re-states boolean logic rather than testing the actual trainer constructor/config parser.
- `TestActionTokenInsertionAndStripping` validates the intended tensor layout logic, but not by driving the full model through a realistic forward pass.
- Several tests use monkey-patching/spies and synthetic tensors, so they are best understood as **focused guardrails**, not proof that the full training path is perfect.

### What each test group means

- `TestShiftedSplitLogic`
  - Verifies that clean context uses frames `0..20` and noisy targets use `3..23`.
  - Also checks that fixed-window and streaming logic agree.

- `TestTeacherForcingMask`
  - Verifies that the teacher-forcing mask only exposes the intended clean history and current noisy block.
  - This is critical for causal correctness.

- `TestRopeOffset`
  - Checks that RoPE temporal offset shifts the noisy half so it lines up with its true absolute frame positions.

- `TestSequentialDatasetContextAccounting`
  - Verifies that the dataset counts windows correctly when context frames are included.
  - Also checks that a dataset item returns `window + context`, not just the supervised target frames.

- `TestActionDimSlicing`
  - Checks that `action_dims=[2, 7]` really selects the intended 2D action subset from the full 8D action latent.
  - Also verifies clean/noisy shifted slicing for actions.

- `TestActionModulationProjectionWith2D`
  - Smoke test for `ActionModulationProjection` output shape.

- `TestCleanSideModulationThreading`
  - Verifies that the clean branch uses `_action_modulation_clean`, not the noisy modulation.

- `TestActionTokenProjectionShape`
  - Checks output shape and basic non-degeneracy of the action-token projector.

- `TestActionConditioningModeParsing`
  - Checks intended mode logic for `adaln`, `action_tokens`, and `both`.
  - Useful, but weaker than testing the real config parsing path.

- `TestActionTokenInsertionAndStripping`
  - Checks the tensor bookkeeping for inserting one action token per frame and then stripping it before the head.

- `TestActionTokenMaskDimensions`
  - Verifies that mask construction still behaves correctly when `tokens_per_frame` increases because of action tokens.

- `TestSeparateMergeActionTokens`
  - Checks that helper functions for separating and re-merging action tokens are inverses.

- `TestPEFTCleanSideAdaLN`
  - Regression test for the PEFT ordering/threading issue: the clean side must still use clean modulation.

- `TestActionTokenCleanNoisyAlignment`
  - Checks that clean/noisy action-token source slices remain shifted correctly.

### Bottom line

This is the strongest of the three files and is appropriate as a regression suite for causal teacher-forcing logic.

## 2. `test_lockstep_batcher.py`

### Is it apt?

Yes.

This is a good unit test suite for `LockstepRideBatcher`. It covers the core ride-group/window bookkeeping without depending on zarr, GPUs, or model code.

### Main strengths

- Good coverage of ride-slot lifecycle and window progression.
- Uses synthetic rides, so failures are easy to interpret.
- Checks important edge cases like too-short rides, exact minimum length, batch-size mismatch, and truncation to the shortest ride.
- Includes coverage for `load_z_actions_batch()` and prompt-embedding batching.

### Limitations

- The blockwise timestep tests reproduce the helper logic rather than calling the trainer helper directly, so they guard the intended rule more than the exact implementation path.
- It does not test integration with the actual dataset object or training loop.

### What each test group means

- `test_batch_size_1_sequential_windows`
  - Checks that a single ride produces the expected context-prepended windows in sequence.

- `test_batch_size_2_lockstep`
  - Checks that two rides advance in sync and share the same window index.

- `test_truncation_to_min_ride`
  - Verifies that a group is limited by the shortest ride.

- `test_max_windows_per_ride`
  - Checks that the optional per-ride cap is honored.

- `test_is_first_window`
  - Confirms first-window bookkeeping.

- `test_needs_new_group_initially`
  - Confirms an empty batcher correctly asks for a new group.

- `test_wrong_batch_size_raises`
  - Checks defensive error handling.

- `test_ride_too_short_for_one_window`
  - Verifies that rides shorter than `window + context` produce no usable windows.

- `test_exact_minimum_length_gives_one_window`
  - Checks the boundary case where a ride is just long enough.

- `test_z_actions_batch`
  - Verifies that `load_z_actions_batch()` slices the expected `[start:end]` action window for each ride slot.

- `test_z_actions_batch_requires_encode_fn`
  - Checks expected failure mode if no action encoder callback is provided.

- `test_prompt_embeds_batch`
  - Verifies prompt embeddings are batched in slot order.

- `test_summary_string`
  - Checks human-readable summary output.

- `test_multiple_groups`
  - Confirms that after one group is exhausted, loading a second group resets state correctly.

- `TestBlockwiseTimesteps`
  - Verifies the intended rule that timesteps are identical within each 3-frame block and vary across blocks.

### Bottom line

This is a good, focused unit test file and appropriate for automated runs.

## 3. `test_zarr_dataloader.py`

### Is it apt?

Yes as a **manual integration/debug script**, no as a **unit/regression test suite**.

This file:

- loads real rides
- loads real zarr data
- loads the Wan VAE
- decodes latents
- overlays action/motion annotations
- writes MP4s with `ffmpeg`

That makes it useful for **visual validation**, but not suitable as a normal automated test.

### Why it is not a normal test suite

- It has no `unittest.TestCase` or `pytest`-style tests.
- It depends on external data paths.
- It depends on model weights and decode behavior.
- It depends on `ffmpeg`.
- It may depend on GPU availability for practical runtime.
- It produces artifacts for humans to inspect rather than asserting pass/fail conditions.

### What it is good for

- verifying that ride grouping and window slicing make visual sense
- checking that decoded windows align with motion/action overlays
- confirming that context-prepended windows look right
- spotting obvious dataset alignment bugs that unit tests might miss

### What the script does

- builds a `ZarrRideDataset`
- groups rides with `LockstepRideBatcher`
- loads `window_size + context_frames` latent windows
- decodes them through the Wan VAE
- overlays:
  - motion arrows
  - full latent bars
  - a steering/forward latent dial
  - slot/window/frame metadata
- writes per-window or combined videos

### Bottom line

Keep this file, but treat it as a **debug visualizer** or **manual integration check**, not as part of a fast regression suite.

## How To Run Them

These commands assume you are at the repo root.

If you use the project conda environment:

```bash
conda activate arrwm
```

### Run the real automated test suites

```bash
python -m unittest testing.test_context_shifted_teacher_forcing -v
```

```bash
python -m unittest testing.test_lockstep_batcher -v
```

### Run both together

```bash
python -m unittest \
  testing.test_context_shifted_teacher_forcing \
  testing.test_lockstep_batcher -v
```

### Run the manual visualizer

```bash
python testing/test_zarr_dataloader.py \
  --num_rides 4 \
  --windows_per_ride 3 \
  --batch_size 2 \
  --output_dir testing/outputs
```

Useful optional flags:

```bash
python testing/test_zarr_dataloader.py \
  --num_rides 4 \
  --windows_per_ride 2 \
  --batch_size 1 \
  --context_frames 3 \
  --combined \
  --device cuda:0 \
  --output_dir testing/outputs
```

## Recommended Usage

- Use `test_context_shifted_teacher_forcing.py` after changing:
  - teacher-forcing logic
  - mask construction
  - action slicing
  - RoPE offsets
  - adaLN/action-token threading

- Use `test_lockstep_batcher.py` after changing:
  - `LockstepRideBatcher`
  - streaming window logic
  - ride grouping
  - z-action batch slicing

- Use `test_zarr_dataloader.py` after changing:
  - zarr loading
  - alignment assumptions
  - VAE decode window semantics
  - visual/debug overlays

## Recommended Interpretation

If you want a concise classification:

- `test_context_shifted_teacher_forcing.py`
  - **Apt** as a regression suite

- `test_lockstep_batcher.py`
  - **Apt** as a unit test suite

- `test_zarr_dataloader.py`
  - **Apt only as a manual integration/visual debugging script**

## Suggested Follow-up Improvement

If you want these files to be cleaner conceptually, the main thing I would change next is organizational:

- keep `test_context_shifted_teacher_forcing.py` and `test_lockstep_batcher.py` as automated tests
- rename `test_zarr_dataloader.py` to something like `debug_zarr_dataloader.py` or `visualize_zarr_dataloader.py`

That would make it much clearer which scripts are expected to pass/fail automatically and which are meant for human inspection.
