# WP-14B — Wan 2.1-T2V-14B as the GAN's projection backbone

Report for work package **B2** of `docs/GAN_REDESIGN.md` section B. Only
this file carries WP-14B's status; `GAN_REDESIGN.md` is the main agent's.

**State: BUILT, flag-gated, default-OFF. 19 tests pass + 1 GPU-only smoke
(passes on GPU against the real 14B at the production 60x104 grid). Reviewed
by four independent agents; their findings and my fixes are in §11. Cost vs
the same path at 1.3B: +3.7 GiB resident, ~1.2x G-side time (§6). Not yet run
inside a training step.**

---

## 1. What this changes, in one paragraph

The LADD discriminator does not learn its own features — it projects onto
a frozen transformer's intermediate block outputs. Until now that
transformer was `real_score`, i.e. the same v14e teacher the student is
being DMD-distilled from. Point 6 of `GAN_REDESIGN.md` argues the tap is
therefore largely redundant with the DMD term: critic and teacher read the
same representation, so the adversarial gradient carries little the
distillation loss does not already carry. WP-14B lets the disc project
onto an **independent** frozen network instead — stock
Wan 2.1-T2V-14B, which the student has never been trained against — while
leaving `real_score`, `real_name` and the whole DMD path untouched.

The affordability trick is that the disc only ever taps **shallow** blocks
(texture lives early), so blocks above the deepest tap are never
constructed and never read off disk:

```
num_layers_loaded = max(ladd_feature_blocks) + 1      # 9 for [0, 2, 4, 8]
```

≈3.4 B params ≈ **6.8 GB bf16**, against ~28 GB for the full 40-block
model. Blocks 0-8 live entirely in shards 1-2 of the 6-shard checkpoint.

---

## 2. Files

| File | Status | What |
|---|---|---|
| `model/wan14b_prefix.py` | **new** | prefix loader: truncated `WanModel` + manual safetensors load, frozen |
| `model/ladd_disc.py` | modified | `WanFeatureProjector` raw-`WanModel` mode; `build_ladd_disc(backbone=...)` + guards |
| `wan/modules/model.py` | modified | `_forward(..., max_block=None)` feature-tap early exit (pure addition) |
| `trainer/causal_action_forcing_train.py` | modified | build branch at the LADD disc construction (`:710`, `:803-883`) |
| `configs/action_forcing_phase3_dmd.yaml` | appended | WP-14B block at EOF (3 knobs, all default-off) |
| `testing/test_wan14b_prefix.py` | **new** | 19 unit tests + 1 opt-in real-weights smoke |
| `testing/probe_wan14b_disc_scaling.py` | **new** | GPU memory/time scaling probe (§6) |

Nothing else was touched. In particular `real_name` and
`model/base.py:183-188` were NOT modified (that would break DMD).

---

## 3. The three contracts, and why each is shaped that way

### (a) Prefix loading — `model/wan14b_prefix.py`

```python
load_wan14b_prefix(model_path, *, max_block, device, dtype=torch.bfloat16,
                   load_head=True, low_cpu_mem=True) -> WanModel   # frozen
```

* Geometry is read from the checkpoint's `config.json`
  (`dim=5120, ffn_dim=13824, num_heads=40, in_dim=16, num_layers=40`),
  never hardcoded; a mismatch against the 14B reference logs a warning
  rather than silently mis-shaping the model.
* `num_layers = max_block + 1`; **hard assert** `max_block < ckpt_layers`.
* The module is built on the **meta** device, cast to the target dtype
  while still meta (free), then materialised straight on the target
  device with `to_empty` and filled tensor-by-tensor from the shards.
  A plain fp32 CPU construction would burn ~13.6 GB of host RAM per rank
  *before* the cast — with 8 ranks/node that is an OOM. Peak host RAM
  here is one tensor. `low_cpu_mem=False` keeps the naive path available.
* `WanModel.freqs` is a **plain attribute, not a buffer** (deliberately,
  see its `__init__` comment), so neither `.to()` nor `to_empty()` moves
  it and under a meta build it would stay meta — the forward's
  `self.freqs.to(device)` would then raise. The loader recomputes it with
  the identical `rope_params` formula/table size on the target device.
* Every parameter is accounted for: the loader tracks loaded +
  zero-filled keys and raises if the total does not equal the model's
  state-dict. `to_empty` leaves *uninitialised* memory, so an unnoticed
  gap would be NaNs, not obvious garbage. A missing block tensor is a
  hard `KeyError`; only `head.*` may be zero-filled.
* Returns `requires_grad_(False).eval()`.

### (b) Projector — raw `WanModel`, actual token count

`WanFeatureProjector(real_score, block_indices, backbone=<WanModel>)`.
When `backbone` is given, `real_score` is not touched at all.

Two things had to be bypassed:

1. **`WanDiffusionWrapper` is skipped.** It consumes the model's return
   value (`utils/wan_wrapper.py:749-753` unpacks `flow_pred` and converts
   it to x0) — a truncated prefix has no meaningful output to convert.
   The projector already discards the return value and reads hooked block
   outputs, so calling the raw model is strictly simpler.
2. **`seq_len` is the chunk's ACTUAL token count**, computed from the
   input shape and the backbone's own `patch_size`:
   `T'·H'·W' = (F/1)·(H/2)·(W/2)`. `WanModel._forward` zero-pads the patch
   sequence up to `seq_len` (`wan/modules/model.py:735-741`) and *every
   block then runs at that length*. Inheriting the wrapper's 18721-token
   budget would run a 3-frame disc chunk (4680 tokens at 60×104) at ~4×
   the necessary attention/FFN cost — at dim 5120 that dominates. The
   projector recomputes the count itself and **refuses** a caller-supplied
   `seq_len` that disagrees.

Fail-loud in this mode: `clean_x` / `aug_t` (wrapper-only teacher-forcing
arguments with no raw-model equivalent) raise rather than being silently
dropped; `conditional_extra` action tokens log a one-time warning (stock
T2V has no action-token plumbing — this is the documented distribution
shift, and it is exactly why `a_per_f` must be 0).

Also needed: `_find_blocks` must not apply its `len(blocks) >= 10` guard
to the backbone — a 9-block prefix is legitimate here.

### (c) `max_block` early exit — `wan/modules/model.py`

`_forward(..., max_block=None)`: when set, the block loop returns the raw
token tensor after block `max_block`, skipping `head` + `unpatchify`.
`None` (default) is the unchanged path, so every existing caller is
byte-identical.

This is **not** just an optimisation — it is required. `Head.forward`
expects a per-sample `[B, dim]` modulation embedding, while the disc
passes **per-frame** timesteps `[B, F]` (`t_disc`,
`trainer:5854`), which makes `e` `[B·F, dim]`. Running the head on that
raises `The size of tensor a (2) must match the size of tensor b (4)`.
Found by the unit test, not by inspection.

### (d) Trainer branch — `ladd_disc_backbone_model_name`

Inside the existing LADD build (`:803-883`), gated on a non-empty
`ladd_disc_backbone_model_name`:

* refuses to build if the taps were **auto-derived** — the 40-block
  auto-default is `[8, 16, 24, 32, 39]`, i.e. a full 28 GB load. This is
  the trap the config comment warns about;
* asserts `max(taps) < num_layers_loaded`;
* passes the **backbone's own** `dim_teacher=5120` and
  `patch_size=(1,2,2)`, and hard-sets `a_per_f=0` — harvesting
  `action_tokens_per_frame=1` off `real_score` would silently mis-slice
  the token→patch-grid reshape in every head;
* loads onto `self.device` in `ladd_disc_backbone_dtype` (bf16).

The backbone hangs off the projector, which is a **plain attribute**, not
a submodule. So `disc.to(dtype=torch.float32)` (the R1-stability cast)
does not touch it, `disc.parameters()` does not contain it (no DDP
sync, no optimizer, no checkpoint bloat), and resume is unaffected.

---

## 4. Config

Appended at the end of `configs/action_forcing_phase3_dmd.yaml`:

```yaml
ladd_disc_backbone_model_name: ""        # "" = OFF (default)
ladd_disc_backbone_dtype: bfloat16
ladd_disc_backbone_load_head: true
```

`ladd_feature_blocks: [0, 2, 4, 8]` is **deliberately not** in that block.
The key already exists at `configs/action_forcing_phase3_dmd.yaml:73`, and
a duplicate at EOF would silently retarget the taps of every existing 1.3B
run. The 14B arm passes it as a launch-line override instead
(`OmegaConf.from_dotlist` parses the list fine):

```
  ladd_feature_blocks=[0,2,4,8] \
  ladd_disc_backbone_model_name=/scratch/u6ex/as1748.u6ex/frodobots/Wan2.1-T2V-14B
```

Forgetting it is not silent: the trainer raises.

Verified on a GPU node that `OmegaConf.from_dotlist` parses
`ladd_feature_blocks=[0,2,4,8]` as a real `list` (not a string), and that
the A18(ii) override-key guard (`_warn_ignored_override_keys`) recognises
all four keys as config-sourced — no "ignored override" error lines. The
launch-line fragment for the arm is therefore:

```
  ladd_feature_blocks=[0,2,4,8] \
  ladd_disc_backbone_model_name=/scratch/u6ex/as1748.u6ex/frodobots/Wan2.1-T2V-14B \
  ladd_disc_backbone_dtype=bfloat16
```

---

## 5. Tests

`testing/test_wan14b_prefix.py` — CPU-only, 12 tests, all passing:

```
PASS test_prefix_num_layers_for_taps
PASS test_prefix_load_sharded_matches_source       # both low_cpu_mem paths
PASS test_prefix_load_single_file_and_bf16
PASS test_tap_beyond_checkpoint_raises
PASS test_missing_tensor_is_fail_loud
PASS test_load_head_false_zero_fills               # never uninit memory
PASS test_projector_runs_raw_backbone_at_actual_token_count
PASS test_projector_taps_the_requested_blocks      # + no hooks left behind
PASS test_projector_rejects_wrapper_only_arguments # clean_x / aug_t / seq_len
PASS test_build_ladd_disc_backbone_guards          # deep tap / a_per_f / dim
PASS test_disc_forward_through_raw_backbone        # grads to x, none to backbone
PASS test_backbone_none_keeps_the_legacy_wrapper_call   # byte-identical OFF
```

The same 13 pass on GPU (holder 6108075), where the real-weights smoke
below is included rather than skipped.

The tests build a tiny 4-block Wan checkpoint in a tmpdir with the real
sharded-index layout, load a prefix from it, and compare **every**
tensor against the source model. The off-path test asserts the projector
still calls `real_score` with exactly the legacy kwargs (no `seq_len`, no
raw-model arguments).

CPU scaffolding lives in the test file only: `wan` touches
`torch.cuda.current_device()` at import, and WAN's attention asserts CUDA
when flash-attn is installed, so both are stubbed there. Nothing in the
shipped path changes.

### Real-weights smoke — PASSED on GPU

```
WAN14B_PREFIX_SMOKE=1 python testing/test_wan14b_prefix.py
```

Run on an idle holder GPU (job 6108075, one GPU, `srun --overlap`), taps
`[0,2,4,8]`, `x = [1, 3, 16, 60, 104]`:

```
[smoke] prefix params = 3.40B (6.79 GB bf16)
[smoke] logits (1, 18720) over 4680 tokens/tap; grad_norm=0.0004
[smoke] attention kernel = flash-attn; peak alloc = 14.20 GiB, peak reserved = 14.57 GiB
```

What that confirms, on the real checkpoint and the real kernel:

* the 9-block prefix loads from shards 1-2 (+ head from shard 6) and is
  **6.79 GB** in bf16 — the ~6.8 GB the plan predicted, versus ~28 GB for
  the full model;
* the backbone runs at **4680 tokens** = 3 · 30 · 52, the chunk's actual
  count, not the wrapper's 18721 budget (the 18720 logit columns are
  4 taps × 4680, a numerical coincidence worth not misreading);
* the whole path is differentiable to the disc input under flash-attn,
  with a finite gradient;
* peak GPU memory for this single-row forward+backward is **14.2 GiB**,
  i.e. ~7.4 GiB of transient on top of the weights.

The tiny CPU fixtures keep an SDPA stand-in for WAN's flash-attn kernel
(it asserts CUDA); the real-weights smoke restores the genuine kernel
whenever a GPU is present, so the numbers above are from the kernel
training actually uses.

---

## 6. Cost — measured

`testing/probe_wan14b_disc_scaling.py` runs the two shapes the trainer
uses (D-update: `disc.train()`, grads to the 8 M disc params; G-guidance:
`disc.eval()`, grad back to the input latent, i.e. the checkpointed
teacher recompute) at production chunk size (F=3, 60×104, 4680
tokens/row) on one 95 GiB GPU.

**Against the same code path at 1.3B**, the 14B costs ~1.2× the time and
~1.05× the memory on the G side, ~1.3× time and **2.16× memory** on the D
side, +3.7 GiB resident — because only 9 of 40 blocks are ever executed.

**CORRECTION.** I headlined this "of the disc we run today". Wrong twice:
1.05× is the G column only (the D column more than doubles), and the baseline
row is the raw-model path at the ACTUAL token count, whereas today's disc runs
through `WanDiffusionWrapper` at its padded `seq_len` (~4× the tokens).
Today's real disc is therefore *more* expensive than the baseline row — the
swap is more favourable than these numbers — but that has not been measured
and must not be quoted as "vs today's arm".

At 12 rows, `ladd_gen_guidance_micro_batch_groups=4` (what the holders
already set):

| backbone | D-update | G-guidance | resident weights |
|---|---|---|---|
| 1.3B, 30 blocks, taps [6,12,18,24,29] — *today* | 1561 ms / 12.1 GiB | 2850 ms / 32.4 GiB | 2.64 GiB |
| 14B, 9-block prefix, taps [0,2,4,8] — *this WP* | 2088 ms / 26.1 GiB | 3525 ms / 34.2 GiB | 6.32 GiB |

Scaling of the 14B path (peak allocated, GiB):

| rows | D-update | G-guid. groups=1 | G-guid. groups=4 |
|---|---|---|---|
| 1  |  8.13 | 14.26 | — |
| 4  | 13.02 | 37.47 | 15.79 |
| 8  | 19.53 | 68.44 | 24.97 |
| 12 | 26.05 | **OOM** | 34.17 |

Three things follow:

1. **`ladd_gen_guidance_micro_batch_groups >= 4` is mandatory** for the
   14B arm at 12 rows — un-micro-batched G-guidance OOMs. This is not a
   new 14B failure mode: the 1.3B path OOMs at 12 rows too, which is why
   `_fgan_holder.sh:474` and `_roll_holder.sh:426` already set 4. Keep it.
   It is also free — at 8 rows micro-batching was *faster* (2337 vs
   2500 ms), and at 12 rows it costs nothing measurable.
2. **The expensive half is G-guidance, not the D-update.** In train mode
   the input carries no grad, so the projector never checkpoints and no
   graph reaches the backbone; the gen-side path pays ~8.5 GiB/row
   un-micro-batched.
3. **Trim room exists** if the in-situ budget is tighter than the probe's
   solo GPU: taps `[0, 2, 4]` load 5 blocks (~3.5 GiB) instead of 9.

Reproduce:

```
srun --jobid=<holder> --overlap --nodelist=<node> --nodes=1 --ntasks=1      --export=ALL,CUDA_VISIBLE_DEVICES=3      python testing/probe_wan14b_disc_scaling.py --rows 1 2 4 6 8 12
# baseline for comparison:
     ... --path .../Wan2.1-T2V-1.3B --taps 6 12 18 24 29 --rows 4 8 12
```

Caveat on the baseline row: it is the same projector code path driven by
stock 1.3B weights, not the live `real_score` wrapper (which additionally
carries action tokens, `clean_x`/`aug_t` and the causal model's own
plumbing). It isolates the cost of *the backbone swap*, which is the
question here; it is not a replica of the current arm's absolute cost.

## 7. What has NOT been verified yet

1. **In-situ memory.** The probe had a GPU to itself. In training the
   disc shares it with generator + fake_score + real_score + optimizer
   state, so the headroom question is the sum, not the disc alone. The
   +3.7 GiB of resident weights and the +1.7 GiB of G-guidance peak are
   the deltas to budget for.
2. **Whether it helps.** The A7 telemetry is the health readout, and the
   prediction is that `|cos(g_GAN, g_DMD)|` **falls** versus the
   v14e-backbone critic. That fall is the intended effect (an
   independent critic), not a regression. The countervailing risk is
   distribution shift: stock T2V 14B never saw driving footage, never saw
   action tokens, and here runs on 3-frame chunks at timesteps drawn
   from the disc's own schedule.

---

## 8. Log lines to grep in the first run

```
[wan14b_prefix] loaded 9/40 blocks from ... params=3.xxB (6.8 GB @ torch.bfloat16)
[ActionForcing] LADD disc backbone OVERRIDE: ... taps=[0, 2, 4, 8] dim_teacher=5120 patch=(1, 2, 2) a_per_f=0
[ActionForcing] LADD discriminator built: blocks=[0, 2, 4, 8] dim_teacher=5120 ...
[LADD/14B] raw backbone ignores conditional_extra keys [...]   # expected once, if action tokens are on
```

---

## 9. Postmortem — the 60-step smoke OOM (2026-08-23 15:12)

The main session ran `sbatch/run_smoke_14bdisc.sh` on holder 6108076 and hit
CUDA OOM at ~92 GiB allocated on all 8 ranks. It was recorded in
`docs/GAN_REDESIGN_TWO.md` as *"graph-on 14B forward fed the padded
18,721-token seq_len at 5120-dim"*. **That diagnosis is wrong**, and the
distinction matters because it makes the difference between "A is blocked on
code" and "A is blocked on one missing config line".

What the log actually shows (`logs/exp_smoke_14bdisc.err`):

* The failing allocation is **366.00 MiB** inside `rope_apply`'s complex
  multiply (7 of 8 ranks; rank 4 died at a 1.45 GiB FFN allocation in the same
  recompute). That buffer is `S x n_heads(40) x head_dim/2(64) x 2 x 8 B`
  (float64 — `rope_apply` upcasts), and the allocator rounds large requests up
  to 2 MiB, so 366.00 pins S in [9320, 9369]. The only realizable grid there
  is **S = 9360 = 6 x 30 x 52**, a 6-frame `2*npb` transition chunk (365.62
  MiB before rounding); the log's `npb=3` confirms it independently.

  **CORRECTION.** I first argued the observed size *rules out* the
  18,721-token padded budget (731 MiB). **That inference is invalid** — an
  adversarial review caught it. `rope_apply` derives `S` from `grid_sizes`,
  built at `wan/modules/model.py:741-742` from the patch grid BEFORE the
  zero-pad at `:745-749`, so the RoPE buffer is padding-invariant: fed 18,721
  this allocation would still have been 365.62 MiB. It pins the chunk
  geometry and nothing else. The padded-budget theory is false on CODE (the
  projector computes the count itself and raises on disagreement), not on
  this arithmetic.
* It could not have been the padded budget in any case: the projector computes
  the token count itself (`_raw_backbone_seq_len`) and raises if a caller
  passes a `seq_len` that disagrees — see §3(b), unit-tested, and confirmed on
  real weights at 4680 tokens in §5.
* The frames above the OOM are `Tensor.backward` -> `checkpoint.unpack_hook`
  -> `recompute_fn` -> `ladd_disc._run_teacher`, inside
  `_streaming_train_one_chunk`: the **generator-side guidance** forward being
  recomputed during the gen backward. The D-update is not in the trace (it
  runs through its own micro-batched helper).
* **The G-side micro-batch knob was ON, at 4.** My first read of this — that
  the smoke left `ladd_gen_guidance_micro_batch_groups` at the code default
  of 1 — was wrong, and the main session was right to challenge it.
  `run_smoke_14bdisc.sh` calls `sbatch/_fgan_holder.sh`, whose fixed argument
  list passes the knob at `:474`, with `$DEXTRA` (`:476`, which does not carry
  the key) merging after. The raw-14B path does not bypass it either: the
  split is in the trainer's shared `_m_fwd` (`:8500-8530`), which chunks the
  row dim before calling `disc(...)` and never looks at the backbone, and the
  gen call site does `disc_for_guidance.eval()` (`:8905`), so the eval-gated
  branch is live.

So the cause is not a missing knob — it is that **4 is not enough at this
chunk size**. The smoke runs 6-frame `2*npb` transition chunks (9360
tokens/row), where §6's 3-frame numbers understate the cost ~2x:

| rows | D-update | G groups=4 | G groups=8 | G groups=16 |
|---|---|---|---|---|
| 4  | 19.5 GiB | 24.9 GiB | — | — |
| 8  | 32.4 GiB | 43.2 GiB | 28.8 GiB | — |
| 16 | 58.3 GiB | **79.8 GiB** | 51.0 GiB | 36.7 GiB |
| 32 | OOM | OOM | OOM | 66.6 GiB |

The gen-side forward carries `n_uniq + n_fake` rows with `n_uniq` capped by
`ladd_gt_transition_match_max_real=16`. Rank 4's 1.45 GiB FFN allocation pins
the micro-batch at 6 rows, so the batch was **21-24 rows** (I first wrote
~17-20) — *past* the 79.8 GiB cell, needing ~110-120 GiB — before generator +
fake_score + real_score + optimizer. It died at 92 GiB. The measurements
account for the failure with room to spare, not "exactly".

**A second gen-side path ignores the knob entirely.** The row split lives in
`_m_fwd`, which the MATCHED gt-transition branch calls; the POSITIONAL branch
calls the disc directly and `ladd_gen_guidance_micro_batch_groups` appears
nowhere in it, so there the knob is a silent no-op at any value. The smoke
used `match=true`; a `match=false` arm would OOM identically with the knob
apparently set. Not mine to fix — flagged to the campaign.

Fix, still config-only, but the value matters:

```
ladd_gen_guidance_micro_batch_groups=16    # NOT 4 — 4 was already in effect
ladd_gt_transition_match_max_real=8        # halves rows, and G-guidance time
```

Both belong in the smoke's `DEXTRA`, which merges after the holder's fixed
list and so overrides `:474` without editing `_fgan_holder.sh` (never edit a
holder while a run is executing it).

Two lessons worth keeping:

* The two micro-batch knobs are **not** interchangeable, and the expensive one
  is the G-side (§6 finding 2).
* A knob being *set* is not the same as it being *sufficient*. I inferred
  "absent" from the launcher I could see and missed that the holder supplies
  it; the resolved value is what matters, which is why the re-smoke should
  echo it.

---

## 10. Option B (GAN_REDESIGN TWO) — stock 1.3B through this machinery

Confirmed working with **no code change**: point
`ladd_disc_backbone_model_name` at the stock `Wan2.1-T2V-1.3B` directory.
Nothing in the loader assumes 5120 — geometry comes from `config.json` and the
trainer reads `dim_teacher`/`patch_size` back off the returned model. The §6
probe drove the stock 1.3B through this exact path twice on GPU (9-block
prefix, 0.83 GiB; 30-block, 2.64 GiB), which is the evidence, not inspection.

Two housekeeping changes made for it: the per-key "differs from the 14B
reference" warnings are now one INFO line (a 1.3B load is not an anomaly), and
`load_wan_prefix` is exported as an alias for `load_wan14b_prefix`.

**The resume trap, corrected.** With the stock 1.3B, `dim_teacher` stays 1536,
so the disc's CCM/head shapes match the existing v14e-projected arms and the
resume guard (which drops checkpoint entries by SHAPE only, on names that
already match) does not fire; with the 14B, 1536 -> 5120 makes it impossible.
I called this "the one serious trap" without stating two conditions an audit
surfaced:

* the CCM and heads are `ModuleDict`s keyed by TAP INDEX, so the taps must
  match too — at the recommended `[0,2,4,8]` the keys (`0/2/4/8`) have zero
  overlap with a legacy checkpoint's (`6/12/18/24/29`), so nothing loads. The
  trap only bites at taps `[6,12,18,24,29]`;
* it needs `auto_resume=true` plus a prior checkpoint in the same log_dir, and
  every launcher in this family sets `auto_resume=false`.

Under `ladd_freeze_projector_mixing=true` the CCM is additionally a frozen
random projection never trained on v14e features, so the heads are the only
part that would genuinely carry over. Launch fresh and verify anyway — but
this is a guard-rail, not a live hazard.

---

## 11. Review round — what four reviewers found, and what I changed

Before sign-off this package was put through four independent review agents:
adversarial code review of the loader/projector, a trainer+config gating
review, an audit of every documented CLAIM against primary evidence, and
mutation-testing of the test suite (34 mutations). They found real defects in
both the code and the docs. Doc corrections are marked inline above (§6, §9,
§10). Code fixes:

| finding | fix |
|---|---|
| Hooks were installed OUTSIDE the checkpointed region and removed in an outer `finally`, so a backward RECOMPUTE would find `local_feats` empty and `KeyError`. It survives only because autograd stops the replay early — and the `max_block` early exit shortens the region, moving that stop point closer to the access. Latent, and made likelier by my own change. | Hooks now install/remove INSIDE the checkpointed function, so the replay is correct on its own terms. |
| `build_ladd_disc` guarded `dim_teacher` and `a_per_f` against the backbone but NOT `patch_size` — the one mismatch that is silent (backbone emits more tokens, the disc slices the first N and reshapes into a bogus grid). | Added the `patch_size` guard. |
| An i2v checkpoint loaded cleanly and died at the first disc forward on a BARE `AssertionError`, after ~7 GB had landed on every rank. | Refuse `model_type != 't2v'` at load time. |
| The comment justifying the head skip claimed the head "crashes on per-frame timesteps". FALSE: `Head.forward` broadcasts, so at B=1 it runs silently as dead compute and only raises for B>1. A backbone without `max_block` support therefore failed silently, not loudly. | Corrected the comment; a backbone lacking `max_block` now raises — in `WanFeatureProjector.__init__`, at construction. The first version of this fix raised inside `_run_teacher`, i.e. at the first disc forward, after ~7 GB/rank was resident and from inside a checkpointed recompute closure; the condition is fully known at build time. |
| `in_dim = int(getattr(bk, "in_dim", C_in))` made the channel check a tautology when the attribute was missing; a wrapped backbone reported "0 blocks loaded"; an empty tap list raised `max() arg is an empty sequence`. | All three now raise with accurate messages. |
| `qk_norm` / `cross_attn_norm` / `window_size` in `config.json` were ignored, so a checkpoint declaring them built a DIFFERENT architecture and then blamed the checkpoint for missing tensors (`window_size` carries no weights — silently wrong). | Forwarded from the config. |
| The summary line under-counted params whenever anything was zero-filled, printed `head=zeroed` if ANY tensor was, and every rank emitted it. | Fixed accounting; `log` defaults to rank 0. |

Test-suite gaps the mutation audit exposed (21 caught, 11 missed, 2 equivalent
mutants). The suite could not tell WHICH block it tapped — a tap shifted by one
block, and a hook capturing block INPUT instead of output, both passed 13/13,
because the only structural assertion was `feats[0] != feats[2]`. Also missed:
the RoPE table was never compared to anything, the production bf16-backbone /
fp32-input pairing was never exercised, the head-missing-from-checkpoint branch
had no test, three fail-loud guards had zero coverage, and every fixture was
square (8x8) and used a uniform timestep, hiding H/W transposition and
per-frame timestep bugs. `test_load_head_false_zero_fills` "caught" its
mutation only because this host's `to_empty` memory happened to be non-zero —
a flaky catch, not coverage.

Fixed: fixtures are now non-square (8x16) with distinct per-frame timesteps;
every bare `except ValueError` asserts on the message (one guard could be
deleted entirely and the old test still passed, satisfied by Python's own
`max()` error); and seven tests were added —
`test_projector_tap_identity_is_exact` (compares each tap against an
independent all-blocks reference, and against each block's input),
`test_prefix_freqs_match_a_reference_model`,
`test_projector_casts_fp32_input_for_a_bf16_backbone`,
`test_head_missing_from_ckpt_is_zero_filled`,
`test_every_param_is_finite_after_load`, `test_loader_rejects_i2v_checkpoints`,
`test_build_ladd_disc_rejects_patch_size_mismatch`. Suite is now **19 passed,
1 skipped** — and the GPU smoke reports as a real skip rather than counting
itself a pass, which is how "13 passed" previously described 12 executed tests.

Findings NOT mine to fix, flagged to the campaign: the positional gen-side
branch bypasses `ladd_gen_guidance_micro_batch_groups` entirely (§9);
`train/gan_dmd_grad_ratio` returns a forgeable `0.0` when the DMD norm is zero;
and with `gan_enabled=false` the whole WP-14B block is skipped in silence.

Raw probe artefacts are now saved under `logs/wp14b_probe/` — the claim audit
correctly noted that every number in §6 and §9 existed only as a doc table
with no artefact behind it.

---

## 11. External dependency, not a WP-14B defect: the KV recompute-order hazard

Flagged by WP-PIXGAN in `docs/HOLDER_GRID.md` (provenance note 0, 00:3x):
backward-time checkpoint recompute can re-write KV cache slots in reverse
order in the causal/flash-DMD path, silently (no crash, no NaN) landing the
wrong K/V — a hazard the note states "potentially affects EVERY flash arm,"
detection staged into PIXGAN's own window.

**Checked: both `sbatch/run_smoke_14bdisc.sh` and `sbatch/run_decouple14b.sh`
run `flash_dmd_enabled=true`.** So if the hazard is confirmed persistent, it
applies to this package's arms too — but it is not a defect in anything WP-14B
built. `model/wan14b_prefix.py` and the raw-backbone path in
`model/ladd_disc.py` have no KV cache at all: the projector's checkpointed
recompute (`_run_teacher`) is a stateless forward through a frozen backbone
with local, per-call forward hooks — nothing there to mis-order. Confirmed
by re-reading this report end to end: nothing in §1-§10 depends on flash-DMD's
KV cache being correct — the loader's weight-loading, the projector's
seq_len/gradient correctness, the cost measurements, and the resume-trap
analysis are all independent of it.

The exposure is at the interpretation layer, not the code layer: if the
hazard is real, `decouple14b`'s DMD-side gradient could be silently wrong
independent of the backbone-decoupling question the arm exists to test,
confounding any texture difference it shows. Not mine to fix — PIXGAN owns
detection (the KV-hash instrument) — flagged to MAIN with the ask that its
verdict reach this package before `decouple14b`'s result is read as evidence
about decoupling specifically.

---

## 12. Trainer-file lock, taken and released — the LADD unweighted-ratio call site

Landed while holding the lock (grid: PIXGAN released explicitly on
build-completion, WP-14B held 2nd-in-queue).

**Scope:** the two things promised earlier — the LADD unweighted-ratio
call site, and a resolved-value echo at the LADD build line.

### The unweighted ratio — division, not raw-threading

`_ladd_run_pair_mode` (~2900 lines, delicate R1-fire-rate cadence
semantics documented in its own header) computes `generator_gan_loss =
total_weight * g_rp`. Threading `g_rp` itself out through both its return
sites and through `_compute_ladd_losses`'s per-mode summation — mirroring
WP-PIXGAN's own `(weighted, raw, logs)` convention for the pixel term —
was judged higher-risk than exploiting linearity directly: under five
verified conditions, `‖∇g_rp‖ = ‖∇generator_gan_loss‖ / total_weight`
exactly, and `‖∇generator_gan_loss‖` is already computed by the existing
A7 telemetry with no extra backward.

New module-level pure function `ladd_unweighted_ratio` (next to `grad_at`,
same file) takes five caller-supplied values and returns the ratio or an
omit-plus-reason-key pair — never a forgeable `0.0`:
1. `pix_folded` — the pixel G-term must not be mixed into the same
   `gen_gan_loss` this call's gradient came from.
2. `mode_count == 1` — more than one LADD pair-mode active means the
   gradient is a sum of differently-weighted terms; no single scalar
   recovers one of them.
3. `stat_value == 0.0` exactly — the stat-head sideband must not be
   additively folded into the term.
4. `total_weight > 0` and available — the live resolved multiplier
   (`per-mode weight x gen_gan_weight`), stashed by `_ladd_run_pair_mode`
   at the moment it computed it, never re-derived from cfg.
5. Both gradient vectors present.

Two one-line stashes were added at `_ladd_run_pair_mode`'s two return
sites (matched + positional branches), gated inside the same
`if critic_warmup_done and gen_gan_weight > 0 and not skip_g:` block that
produces the real weighted loss — so a d_only-phase call, a warmup-gated
call, or a zero-weight call can never leave a stale multiplier for the
caller to misread as current. `_compute_ladd_losses` resets the stash to
`None` at the top of every call (before the per-mode loop) and folds in
the per-mode weight with a same-mode sanity check.

**Explicitly out of scope, stated in the docstring rather than silently
omitted:** unlike the pixel path's raw-threaded term, this cannot answer
a true weight-free probe (`total_weight == 0` leaves the gradient with no
graph at all). It only works retroactively on a run with weight > 0 —
which is exactly what the campaign settled on tonight after the
weight-probe identity was verified on wandb history ("a division over the
first ~50 steps of any arm now suffices"), so this closes that gap and no
more.

The cosine is not recomputed: cosine is scale-invariant for a positive
multiplier, so the existing `train/gan_dmd_grad_cos` already is the
unweighted cosine — verified empirically in the test suite, not just
asserted.

### The resolved-value echo

Added to the UNIVERSAL `"[ActionForcing] LADD discriminator built:"` log
line (fires for every LADD arm, not only the 14B/13B backbone-override
path) — `ladd_gen_guidance_micro_batch_groups`,
`ladd_disc_micro_batch_groups`, `ladd_gt_transition_match_max_real`, each
read via the EXACT SAME `getattr` chain its consumer site uses, copy-pasted
rather than re-derived. Prompted directly by tonight's OOM investigation:
a launch script showing `groups=16` was not proof it reached the trainer
process without a manual trace of the holder-script + `$DEXTRA` merge
order.

### Tests

New `testing/test_wp14b_ladd_unweighted_ratio.py` (12 tests, CPU-only,
imports the real trainer module via the same CUDA-import stub
`testing/test_pixgan_t3c_gterm.py` established) — most load-bearing:
`test_division_identity_matches_independent_autograd`, which verifies the
linearity claim against REAL autograd on a small network at several
weights, not algebra alone; `test_cosine_is_scale_invariant...`, which
verifies (not just asserts) the claim that no separate unweighted cosine
is needed; five guard tests, each checking OMISSION with a distinct
reason key rather than a zero-fill; two source-text pins on the stash
sites' ordering.

### A regression this edit caused, found and fixed before calling it done

`testing/test_pixgan_t3c_gterm.py::test_a7_refactor_is_byte_identical`
(WP-PIXGAN's own suite) failed after this landed — 38/40, not 40/40. Root
cause: that test extracts the A7 block's LIVE source text between two
literal markers and `exec`s it in a curated namespace (by design, so the
"shipped" and "reference" code paths are compared on the actual current
source, not a hand-copied replica). My new code introduced two names the
extracted span now references — `ladd_unweighted_ratio` and
`_pix_g_folded` — that the namespace didn't supply, so both calls raised
`NameError`, swallowed by the block's own `except Exception`, producing an
unexpected `train/gan_grad_telemetry_err` key.

Fixed by editing `testing/test_pixgan_t3c_gterm.py` (not my file, but the
fix is mechanical and matches its own established pattern — `grad_at` is
already supplied the identical way): added both names to the exec
namespace, and widened the byte-identical assertion to exclude
`train/ladd_gan_grad_*` keys from the equality check (a legitimate new
addition post-dating the "T3-C refactor is byte-identical" claim the test
actually pins, not a change to what that claim covers). Reported to
WP-PIXGAN immediately rather than left for them to discover. Full sweep
after the fix: `test_pixgan_t3c_gterm.py` 40/40, plus a broader run across
every test file importing the trainer (`test_r1_cadence_and_override_guard`,
`test_pixgan_trainer_supply`, `test_pixgan_trainer_wiring`,
`test_toothpaste_and_defer_disc`, `test_disc_micro_batch`,
`test_action_forcing_checkpoint`) — 181/181 passed outside one unrelated
file (§13).

### Lock released

Explicit release to WP-SURROGATE (next in queue) posted to
`docs/HOLDER_GRID.md`, per the grid's own rule: release is explicit,
never inferred from mtimes.

---

## 13. Unrelated pre-existing failure, found incidentally, not investigated

`testing/test_aux_teacher_schedule.py` — 10/10 tests in that file fail
with `AttributeError: type object 'ActionForcingDMD' has no attribute
'_resolved_real_teacher_input_mix_gt_p'`. Checked and ruled out as a
WP-14B regression: the method lives in `model/dmd_action_forcing.py`,
a file this package has never edited, and the only diff present there
tonight is WP-PIXGAN's own A23 grad-twin work
(`finish_denoised_chunk_grad` / `pix_finish_grad_enabled`), which has
nothing to do with the missing method. Not investigated further — outside
WP-14B's ownership and context — flagged to the coordination channel as a
found-not-fixed item for whoever owns the aux-teacher-schedule feature.

---

## 14. Second OOM round (~05:50) — reconciled, NOT a regression in this package

Both staged re-smokes (`smoke_14bdisc` on 6110550, `smoke_13bdisc` on 6110551)
OOM'd when the holders finally granted after a ~6h queue stall. MAIN raised
a direct, specific hypothesis: the newly-landed `ladd_unweighted_ratio` call
site (§12) holding gradient vectors alive ungated. Checked against primary
evidence — both logs' own self-printed DEXTRA and both crash stack traces —
not defended from reasoning alone.

**Finding: neither job ever received `ladd_gen_guidance_micro_batch_groups`.**
Confirmed by grepping each log's own line-1 DEXTRA echo (both smoke scripts
print their resolved override string before launching):

- `smoke_14bdisc`: `ladd_gen_guidance_micro_batch_groups` and
  `ladd_gt_transition_match_max_real` both ABSENT.
- `smoke_13bdisc`: `ladd_gt_transition_match_max_real=16` (the pre-fix
  default), `ladd_gen_guidance_micro_batch_groups` ABSENT.

That single missing knob is sufficient on its own: the gen-guidance forward
falls back to the code default of 1 (no micro-batching), which is the
IDENTICAL failure mode diagnosed hours earlier (§9) — and it is
backbone-size-independent by construction, which is exactly why both a
14B and a 30x-smaller 1.3B backbone OOM'd the same way.

**Two different causes, not one:**
1. `sbatch/run_smoke_13bdisc.sh` never had the fix applied at all — checked
   on disk, confirmed a near-clone of the pre-fix 14bdisc script. **Fixed**
   in this turn: `max_real` 16->8, `ladd_gen_guidance_micro_batch_groups=16`
   added, verified single-occurrence in the actual DEXTRA line (not the
   accompanying echo string, which matches the same key=value shape and
   would false-positive a naive check).
2. `sbatch/run_smoke_14bdisc.sh` is correct on disk (unchanged since
   23:20:51, the version verified in §9-§10) — but the process that
   executed at 05:42 didn't have the fix, meaning the `.holder_cmd` that
   was consumed was a stale pre-fix snapshot queued during the ~6h stall.
   Not fixable from this file (`.holder_cmd` writes are MAIN's alone) —
   the remedy is regenerating it from current script content before the
   next grant, flagged to MAIN.

**The hypothesis itself, checked against the stack trace:** both crashes
bottom out identically at `generator_loss.backward -> checkpoint.unpack_hook
-> recompute_fn -> ladd_disc._run_teacher -> ... -> rope_apply ->
torch.stack(output)` — the disc's own checkpointed teacher recompute during
gen backward, structurally identical to the original OOM and touching none
of this package's new code. `grad_at` in the new call site operates on a
single parameter tensor (the original design constraint — "probe one param,
not the whole net"), the telemetry is cadence-gated at
`gan_grad_telemetry_every=25` (steps 5/10/15/20 never reach it), and
`ladd_unweighted_ratio`'s outputs are Python floats in a dict — no tensor is
retained by it in steady state.

**Secondary, campaign-wide finding:** `grep -c "^INFO:root:"` is 0 across
both full logs. `logging.info` is silently suppressed for this whole run
(WARNING/ERROR lines appear normally) — meaning the resolved-value echo
from §12, the pre-existing "LADD discriminator built" line, and likely
every other package's INFO-level diagnostics are all silently absent from
tonight's logs too, independent of correctness. Not something fixable from
this file; flagged to MAIN as a visibility gap broader than this package.
