# WS-B — precision / quantisation layer

Owner file: `interactive/precision.py`. This note records the environment, the
exact set of layers each mode touches, the measured `--check` numbers, and the
caveats a caller needs to know. Everything under "Measured" came out of
`python interactive/precision.py --check` on this PC; anything not measured is
labelled pending.

## Environments

`torchao` turned out to be **already present in `flash`**, so quantisation did
not need a clone env — every number in this note was measured in `flash`, and
nothing was installed or removed there.

`flash-q` was created anyway (`conda create -n flash-q --clone flash -y`) at the
coordinator's request, because the UI workstream needs `pygame` (absent from
`flash`, so `play.py` falls back to headless) and the bench workstream needs
`lpips` (absent from `flash`, so the LPIPS column is blank). Those two were
pip-installed **into `flash-q` only**. `flash-q` is therefore the canonical env
for the full interactive rig; `flash` remains fine for anything precision-only.

| | `flash` | `flash-q` |
|---|---|---|
| python | 3.10.19 | 3.10.19 |
| torch | 2.8.0+cu128 | 2.8.0+cu128 |
| triton | 3.4.0 | 3.4.0 |
| torchao | 0.14.1 | 0.14.1 |
| pygame | absent | 2.6.1 |
| lpips | absent | 0.1.4 |
| GPU | RTX 5090, sm_120 (12, 0) | same |

Verified after the install: `flash` still reports `pygame absent`, `lpips
absent`, `torchao 0.14.1` — untouched, as required.

```bash
conda run -n flash   python interactive/precision.py --check   # precision only
conda run -n flash-q python interactive/precision.py --check   # full rig env
```

`import torchao` prints one warning on every run:

```
Skipping import of cpp extensions due to incompatible torch version
2.8.0+cu128 for torchao version 0.14.1
```

That only disables torchao's compiled C++/CUDA extension ops. Everything this
layer uses (`Float8WeightOnlyConfig`, `Float8DynamicActivationFloat8WeightConfig`,
`NVFP4InferenceConfig`) is implemented with tensor subclasses over stock
`torch._scaled_mm` / triton, so the warning is cosmetic here — all three
quantised modes build and run. It is noise on stderr, not a failure; do not
"fix" it by installing into `flash`.

If someone later runs this in an env without torchao, `precision_available()`
returns `False` for `fp8_wo`/`fp8_dyn`/`fp4_wo` and `apply_precision` logs a
warning and leaves the model in bf16 — it never raises. Only then would the
`conda create -n flash-q --clone flash -y && pip install torchao` route (in
`flash-q` only) be needed.

## Modes

| mode | what it does |
|---|---|
| `fp32` | `model.to(torch.float32)`. Matches training/eval weight precision. |
| `bf16` | `model.to(torch.bfloat16)`. Byte-for-byte the policy in `utils/play_world_model.py` (`--weights_dtype bf16`): a flat cast, no norm exceptions. The forward autocasts to bf16 regardless. |
| `fp8_wo` | bf16 cast, then torchao `Float8WeightOnlyConfig(weight_dtype=float8_e4m3fn)` on the DiT block linears. Weights stored fp8, activations bf16, dequant-on-use. |
| `fp8_dyn` | bf16 cast, then `Float8DynamicActivationFloat8WeightConfig(e4m3, e4m3)` on the same linears. Dynamic per-tensor activation scaling + fp8 weights, so the GEMM itself runs in fp8. |
| `fp4_wo` | bf16 cast, then `NVFP4InferenceConfig(mm_config=NVFP4MMConfig.WEIGHT_ONLY, use_dynamic_per_tensor_scale=False)` on the same linears. NVFP4 (e2m1 + e4m3 block scales, block size 16) — Blackwell-only, requires sm_100+. |

`keep_norms_fp32`: the reference bf16 path casts *everything* including norms,
so the default here is a flat cast too, and `describe()` reports every norm as
bfloat16. Set `cfg.extra['keep_norms_fp32'] = True` to instead cast every module
whose class name contains "norm" (`WanRMSNorm`, `WanLayerNorm`) back to fp32
after the base cast. Off by default so bf16 stays bit-comparable with the
reference path.

## What actually gets quantised

Resolved against the real module tree (`wan/modules/causal_model.py`,
`CausalWanModel`, 30 × `CausalWanAttentionBlock`). Include patterns
(`DEFAULT_QUANT_INCLUDE`, overridable via `cfg.extra['quant_include']`):

```
blocks.*.self_attn.{q,k,v,o}
blocks.*.cross_attn.{q,k,v,o}
blocks.*.ffn.{0,2}
```

That is **300 of the model's 306 Linear modules** — 30 blocks × 10 linears.

Everything else stays bf16, either because it is on the exclusion list
(`DEFAULT_QUANT_EXCLUDE`, overridable via `cfg.extra['quant_exclude']`) or
because no include pattern reaches it:

| kept in bf16 | why |
|---|---|
| `patch_embedding` (Conv3d) | first projection; not a Linear anyway |
| `head.head` (Linear) + `head.norm` | final projection into flow space |
| `head_alt.*` | training-only alt head, normally absent at inference |
| `text_embedding.0/.2` | prompt projection, runs once per session |
| `time_embedding.0/.2`, `time_projection.1` | timestep → AdaLN modulation, tiny and numerically load-bearing |
| `blocks.*.norm1/norm2/norm3`, `*.self_attn.norm_q/norm_k`, `*.cross_attn.norm_q/norm_k` | 120 `WanRMSNorm` + 30 `WanLayerNorm` |
| `blocks.*.modulation`, `head.modulation` | raw `nn.Parameter`, not modules — quantisation cannot reach them |
| `action_projection`, `action_token_projection` | zero-init action heads, live outside the DiT (`WorldModelPlayer` owns them separately, so `apply_precision` on the generator never sees them) |
| VAE (`WanVAEWrapper`) | separate module, never passed to `apply_precision` |

`quant_targets(model, cfg)` returns the matched list without modifying
anything, if a caller wants to log or assert it.

`describe()` on the real model confirms the split — 300 Linears carrying the
quantised tensor subclass, 6 Linears left in bf16 (`text_embedding.0/.2`,
`time_embedding.0/.2`, `time_projection.1`, `head.head`), and every norm in
bf16:

```
precision=fp4_wo  compile=off  quantised_linears=300  root=CausalWanModel
module class                        count  dtype(s)
------------------------------------------------------------------------------
Linear                                300 * NVFP4Tensor[torch.bfloat16],bfloat16
WanRMSNorm                            120   bfloat16
CausalWanAttentionBlock                30   bfloat16
WanLayerNorm                           30   bfloat16
Linear                                  6   bfloat16
CausalHead                              1   bfloat16
Conv3d                                  1   bfloat16
```

(`*` marks a quantised row; `fp8_wo`/`fp8_dyn` are identical with
`Float8Tensor[torch.bfloat16]` in place of `NVFP4Tensor`. The bf16 dtype inside
the bracket is the subclass's *logical* dtype — the storage is fp8/fp4.)

The "parameter storage" line in `describe()` descends into tensor subclasses
via `__tensor_flatten__`, so it reports packed bytes including the quantisation
scales rather than the logical bf16 size. Checked on a synthetic 2-block stand-in
with the same layer shapes: 141.3 MiB bf16 → 70.9 MiB fp8_wo (0.50x) →
39.9 MiB fp4_wo (0.28x, i.e. 4 bits plus block scales).

The plan text says "exclude the first/last projection layers initially" — that
is `patch_embedding` and `head`, both excluded. Block 0 and block 29 *are*
quantised; if a quality regression is traced to them, add `blocks.0.*` /
`blocks.29.*` to `cfg.extra['quant_exclude']` rather than editing the module.

## Measured — `python interactive/precision.py --check`

Stand-in checkpoint (the target run's `phase1_step0001000.pt` has not synced):

```
.../phase3_rolling/dmd10k_phase3_rolling_definitive_4n_no_aux_gbs16_3008_commitgan_rehab/
  ..._j6186193/phase1_step0001000.pt      (17.55 GB, embedded step = 1000,
                                           generator_ema present, 825 tensors strict)
```

Each mode is built **in a fresh subprocess** (`--check` re-execs itself with
`--check-one <mode>`) so quantisation, VRAM and compile state cannot leak
between modes. Per mode: build generator + action heads + VAE, apply the
precision policy, prefill the KV cache with one 3-latent-frame seed chunk, then
3 warmup forwards and 10 CUDA-event-timed forwards at the real shape
`[1, 3, 16, 60, 104]` with a zero action and a dummy `[1, 512, 4096]` prompt
embedding (no T5 encode). Output is checked with `torch.isfinite(...).all()`.

GPU idle apart from this job (checked: 1.1 GB baseline). `compile_mode=off`.

| mode | check | achieved | quantised linears | forward latency ms (median of 10) | (min) | vs bf16 | peak VRAM GB | out absmax | out std |
|---|---|---|---|---|---|---|---|---|---|
| fp32    | pass | fp32    | 0   | 171.81 | 170.20 | +3.0 % | 18.34 | 1.156 | 0.5599 |
| bf16    | pass | bf16    | 0   | 166.73 | 164.79 | —      | 15.40 | 1.172 | 0.5603 |
| fp8_wo  | pass | fp8_wo  | 300 | 171.63 | 169.98 | +2.9 % | 14.13 | 1.172 | 0.5640 |
| fp8_dyn | pass | fp8_dyn | 300 | 191.02 | 184.44 | +14.6 %| 14.12 | 1.172 | 0.5644 |
| fp4_wo  | pass | fp4_wo  | 300 | 241.85 | 237.61 | +45.1 %| 13.62 | 1.211 | 0.5649 |

"check = pass" means: the mode applied without falling back, the forward
produced the right shape `[1, 3, 16, 60, 104]`, and `torch.isfinite` held over
the whole output. Policy application itself is fast — 0.14 s for fp8, 0.28 s for
fp4 — so a precision switch costs a checkpoint reload, not the quantisation.

`out_std` moving 0.5603 → 0.5640 → 0.5649 and `absmax` 1.172 → 1.211 is the only
numerical signal available from a single forward: nothing collapses, and fp4
perturbs the output slightly more than fp8, as expected. **This is not a quality
measurement.** Whether fp8/fp4 hold up over a rolling multi-horizon ride is
WS-D's PSNR/LPIPS-vs-bf16 job; a single non-NaN forward proves only that the
kernels run.

### The headline result: quantisation is currently a slowdown, not a speedup

Every quantised mode is **slower** than plain bf16 on this box. That is a
measurement, not a bug in the policy, and the reason is structural:

* The forward already runs under `torch.amp.autocast(bfloat16)`, so bf16 is
  the thing to beat, and bf16 matmuls on a 5090 are already at full tensor-core
  rate.
* `fp8_wo` and `fp4_wo` are **weight-only**: the weight is dequantised back to
  bf16 before a bf16 GEMM. That is strictly bf16 plus a dequant, so it can only
  buy memory, never time. fp8_wo costing only +2.9 % is the dequant being nearly
  free; fp4_wo's +45 % is the NVFP4 block-scale dequant path being expensive
  without the compiled kernels.
* `fp8_dyn` *should* win — it runs the GEMM in fp8 — but it also quantises the
  activation on every call, and at batch 1 with 4683 tokens the per-call scaling
  and casting overhead exceeds the GEMM saving. It needs `torch.compile` to fuse
  the quant into the surrounding ops before it can pay off.

So on the optimisation ladder, rungs 4 and 5 (FP8/FP4) currently sit *below*
rung 1 (bf16) on speed. Their real value here is **VRAM**: 15.40 → 14.13 GB
(fp8) → 13.62 GB (fp4) peak, i.e. the DiT weights drop from ~2.7 GB to ~1.4 GB
(fp8) / ~0.8 GB (fp4). That headroom matters for a longer KV window
(`kv_cache_chunks`) far more than the latency costs, since the KV cache at the
trained 21-frame window is itself ~6 GB. Treat fp8/fp4 as a *memory* lever until
a compiled fp8 path is measured.

fp32 costing only +3 % over bf16 is the same autocast effect from the other
side: the forward casts fp32 weights down to bf16 per matmul, so fp32 buys
nothing but 2.9 GB of extra VRAM. There is no reason to run fp32 in the engine
except as an A/B reference against the training weight precision.

### torch.compile — measurement delegated to the bench matrix

`compile_model` is implemented and wired (`--compile_mode reduce-overhead`
runs), but **no compile latency is reported from this file**, by decision rather
than by omission: compile timing is owned by WS-D's bench matrix. See
**`interactive/bench_results.csv`** and **`interactive/BENCH_REPORT.md`** for the
numbers — the matrix runs bf16 x steps {4, 2} x `reduce-overhead` over full
8-horizon rollouts on the real checkpoint.

That supersedes the isolated single-forward probe here, and it also answers the
rolling-recompile question in caveat 2 more directly than my probe could: a full
8-horizon rollout crosses many distinct `current_start` values, so a recompile
storm shows up straight away in the per-horizon timings. A single forward at one
fixed `current_start` cannot see it at all.

For the record, two attempts were made here before the work was handed over, and
both were invalidated by GPU contention on this shared 32 GB card (a neighbouring
job at 22.5 GB / 95 % util, then an OOM allocating the KV cache when it grew back
to ~26 GB mid-run). Neither indicates a fault in the precision layer — the eager
sweep of all five modes passes on the same code path. The probe instrumentation
(`graphs_after_warmup`, `graphs_after_roll`, `rolling_probe_ms` in `--check`)
is left in place should anyone want it on an idle card, but the bench matrix is
the authoritative source; do not re-run it against the bench for GPU time.

What is known without timings: compiling all 30 blocks takes minutes of warmup
per mode, so a compile-enabled rebuild is not a "few seconds" UI action the way
a precision change is. The UI should show a long "compiling…" state, or compile
should be opt-in from the CLI only.

## Caveats

1. **Quantised modes are slower than bf16 here.** See above. Do not enable
   fp8/fp4 expecting FPS; enable them for VRAM headroom. If the bench harness
   shows any quality loss at all, they are currently a pure loss.

2. **`compile_model` defaults to `dynamic=None`, deliberately.** The block
   forward takes `current_start` as a plain python int that advances by 3
   frames per chunk. With `dynamic=False`, dynamo specialises on its *value*
   and recompiles the entire 30-block stack on every chunk of a rolling
   session — catastrophically worse than eager. `None` lets automatic-dynamic
   promote it to a symint after the second distinct value. A caller who knows
   better can pass `dynamic=` explicitly. **This reasoning is analysis, not a
   measurement.** The empirical check lives in WS-D's bench matrix, not here:
   an 8-horizon rollout crosses many `current_start` values, so if recompiles
   are still happening the compile rows in `interactive/bench_results.csv` will
   be slower than their eager counterparts rather than faster. Read that
   before trusting `compile_mode` in the engine.

3. **`reduce-overhead` means CUDA graphs, and the KV cache is mutated in
   place.** The cache tensors are allocated once per session so their pointers
   are stable, which is the precondition CUDA graphs need — but this has not
   been validated end-to-end over a multi-horizon ride. If a rolling session
   produces frozen or repeated frames under `reduce-overhead`, suspect graph
   capture before suspecting the precision policy, and fall back to
   `compile_mode=off` or plain `default`.

4. **Compilation is per-block, not whole-model.** The outer
   `CausalWanModel.forward` does KV-cache index bookkeeping and runs the
   infinity-RoPE python patch, both graph-break heavy. `mode='full:<inner>'`
   forces whole-model compile if someone wants to try it.

5. **torchao's cpp extensions do not load** (torch 2.8.0 vs torchao 0.14.1,
   ao issue #2919). Everything used here is tensor-subclass + `torch._scaled_mm`
   + triton, so all three quantised modes work, but some of the fp4 slowness is
   plausibly the missing compiled kernels. A torch/torchao pairing that loads
   the extensions might change the fp4 number materially. Not tested — and not
   worth breaking `flash` over.

6. **fp4 uses `use_triton_kernel=False`, `use_dynamic_per_tensor_scale=False`.**
   Conservative defaults for the weight-only path. Both are switchable via
   `cfg.extra['nvfp4_triton']` and would need re-benchmarking.

7. **The `modulation` parameters are raw `nn.Parameter`, not modules**, so no
   quantisation config can reach them regardless of the exclusion list. They
   follow the base cast (bf16). Same for the RoPE `freqs` buffer, which is
   deliberately not registered as a buffer so `.to()` never touches its dtype.

8. **Action heads are outside the DiT.** `WorldModelPlayer` owns
   `action_projection` / `action_token_projection` as separate modules and casts
   them to bf16 itself, so `apply_precision(generator, cfg)` never sees them.
   That is the desired outcome (they stay bf16), but an engine that later folds
   them into the generator module must keep `action_*` in the exclusion list.

9. **Stand-in checkpoint, not the target run.** All numbers above came from the
   `no_aux_..._j6186193` sibling's `phase1_step0001000.pt`. The target run's own
   file had not synced. Latency and VRAM will not change with the checkpoint
   (same architecture, same shapes); anything about *quality* must be re-run on
   the real checkpoint.

10. **`--check` runs the trained-ladder-mismatch warning past you.** The player
    logs `denoise=[1000.0, 625.0, 312.5, 178.6]` from the phase-1 yaml, whereas
    `engine_api.TRAINED_DENOISING_STEP_LIST` is
    `[1000, 625, 357.142857, 208.333333]`. Irrelevant to precision (one forward
    at one timestep), but WS-A needs to resolve which ladder the engine uses.

11. **Peak VRAM includes the VAE decode of the seed chunk**, since `--check`
    goes through the player's real `reset()`. It is a whole-session peak, not a
    DiT-only figure; the *differences* between modes are the meaningful part.

## API

```python
from interactive.precision import (
    apply_precision, compile_model, precision_available, describe,
)

avail = precision_available()          # {'fp32': True, 'bf16': True, 'fp8_wo': ...}
apply_precision(generator, cfg)        # cfg = EngineConfig (or duck-typed)
compile_model(generator, cfg.compile_mode)
log.info("\n%s", describe(generator))
```

`apply_precision` mutates in place and returns the same object (torchao swaps
tensor subclasses; it does not rebuild the module tree), so an engine can hold
its own reference. It accepts the `WanDiffusionWrapper`, the wrapped model, or
the raw `CausalWanModel` — `_base_dit()` walks `.get_base_model()` / `.model`
until it finds the module that owns `.blocks`.

`compile_model` is deliberately separate so the engine can call it **after**
quantisation (torch.compile must see the quantised tensor subclasses, not
replace them). Calling it twice is safe: compiled blocks are flagged and
skipped.

Changing `precision` requires a full rebuild (reload the checkpoint) — fp8/fp4
quantisation is destructive, you cannot cast back up to bf16 and recover the
original weights.
