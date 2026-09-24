# WS-F: the latent→pixel decoder zoo

Owner: WS-F. Files: `interactive/decoders.py`, `interactive/decoder_bench.py`,
`interactive/decoder_weights/`, `interactive/vendor/`, this file.
Nothing outside those paths was touched — the engine wiring is WS-A's job (see
[Engine integration](#engine-integration) for the exact call).

Date: 31 August 2026. Machine: local PC, RTX 5090 32 GB, driver 580.173.02,
env `flash` (`flash-q` for the bench, which needs `lpips`).

## Why

Decode is a fixed per-chunk cost that does not shrink when the denoising ladder
does. At 4 steps it hides behind generation; at 1–2 steps it becomes the FPS
ceiling. `EngineConfig.decoder` selects one of four backends that trade
reconstruction fidelity for decode latency.

## The interface

```python
from interactive.decoders import build_decoder

dec = build_decoder("lightvaew2_1", device="cuda", dtype=torch.bfloat16,
                    wan_model_path="/home/ashish/Wan2.1/")
dec.reset()                                   # once per ride, before chunk 0
frames = dec.decode_chunk(latent_chunk)       # uint8 [T, 480, 832, 3] on GPU
```

`decode_chunk` takes `[1, 3, 16, 60, 104]` (batch 1 enforced — streaming state
is per-stream) in the **normalized** Wan latent space, i.e. exactly what the
generator emits and what `wan_wrapper.decode_to_pixel` is normally handed. It
returns a **uint8 CUDA tensor `[F, H, W, 3]`**, not numpy — same as
`engine._decode_gpu`'s return, so the engine's existing `_drain` host-copy path
still applies. `half_res=True` reproduces `cfg.decode_half_res`.

Every backend is causal: it sees only the chunk it is given plus state left by
earlier chunks, never future context. **Frame counts match the Wan VAE
exactly**: the first chunk after `reset()` yields **9** pixel frames (the VAE's
"special first" latent frame decodes to 1 pixel frame, the other two to 4 each),
every later chunk yields **12**. All four backends are 4× temporal / 8× spatial,
so the geometry is identical across the zoo and no engine-side special-casing
changes.

## The four backends

| name | what it is | decoder params | licence |
|---|---|---|---|
| `wan` | `utils.wan_wrapper.WanVAEWrapper` — the training/eval reference | 128 M-class causal Conv3D | Apache-2.0 (Wan) |
| `lightvaew2_1` | LightX2V's pruned Wan VAE: `dim=96`, `pruning_rate=0.75`, same causal Conv3D stack | 4.62 M | Apache-2.0 (LightX2V) |
| `taew2_1` | madebyollin's TAEHV for Wan 2.1 — Conv2D + MemBlock temporal memory | 9.84 M | MIT |
| `lighttaew2_1` | LightX2V's own TAE for Wan 2.1 | 9.84 M | Apache-2.0 (LightX2V) |

`wan` imports the repo's wrapper — no copy of the Wan VAE lives under
`interactive/`.

### Latent scaling — the thing that had to be got right

The repo's latents (zarr, and everything the generator produces) live in the
**normalized** Wan latent space, `(raw_vae_latent − mean) / std`.
`WanVAEWrapper.decode_to_pixel` undoes that internally by passing
`scale = [mean, 1/std]` into the VAE, which computes `z / scale[1] + scale[0]`.

* **`wan`, `lightvaew2_1`** — both consume `scale=[mean, 1/std]` and
  de-normalize internally (LightX2V's `WanVAE_.cached_decode` has the identical
  line). Feed them the latents **verbatim**. `decoders.py` carries the same 16
  mean/std constants as `wan_wrapper.py`.
* **`taew2_1`, `lighttaew2_1`** — TAEHV's README is explicit: *"TAEHV does not
  use any latent scales / shifts (TAEHV encodes / decodes exactly what diffusion
  models use), whereas Diffusers requires explicitly applying a `latents_mean`
  and `latents_std`."* LightX2V's `WanVAE_tiny` corroborates it — `need_scaled`
  defaults to **False**, and its opt-in branch applies exactly the mean/std
  above. So TAEHV also takes our latents **verbatim**; de-normalizing first
  would be the classic washed-out/saturated bug.

Output ranges also differ and are normalized in `decoders.py`: Wan and
lightvae emit `[-1, 1]` (`0.5·(x+1)`), TAEHV emits `[0, 1]` already.

### Streaming decisions

**No compromises were needed — all four decode 3-frame chunks statefully with
no future context and no overlap window.**

* **`wan`** — the repo's approved contract, unchanged: one
  `model.clear_cache()` in `reset()`, then `decode_to_pixel(chunk, use_cache=True)`
  per chunk. Never a looped plain `decode_to_pixel`.
* **`lightvaew2_1`** — architecturally the same causal Conv3D VAE and it
  exposes the identical interface: `clear_decode_cache()` in `reset()`, then
  `cached_decode(z, scale)` per chunk. The temporal `feat_map` carries the
  genuine left-context, so the state is exact.
  (`reset()` calls `clear_decode_cache()` rather than `clear_cache()` because
  the encoder half is dropped at build time — it is dead weight for a
  decode-only engine.)
* **`taew2_1` / `lighttaew2_1`** — needed a small piece of new code,
  `_TaehvChunkStream` in `decoders.py`. TAEHV ships two paths and neither is
  what a chunked engine wants:
  * `StreamingTAEHV` is exact but strictly frame-at-a-time — one frame through
    all 23 decoder blocks per call. At 480×832 on a 5090 that is launch-latency
    bound, not compute bound.
  * `decode_video(parallel=True)` is the fast path but zero-pads each
    MemBlock's temporal memory at t=0, which is only correct for a clip that
    starts at t=0. Used per chunk it silently restarts the memory every chunk
    and the seams show.

  `_TaehvChunkStream` is the parallel path made causal: identical to
  `apply_model_with_memblocks_parallel`, except each MemBlock's memory for the
  chunk's **first** frame is the previous chunk's **last input at that block**
  (carried in `self._carry`) instead of zeros. That is precisely what the
  sequential path feeds, so it is exact while keeping a whole chunk in one
  batch. Startup trim (`frames_to_trim = 3`) is applied globally after a reset,
  which is what produces the 9/12/12/… frame pattern.

  Verified on CPU with `taew2_1`: chunked-stream output over 3 chunks is
  **bit-identical** (max abs diff `0.0`) to a single whole-clip
  `decode_video(parallel=True)` of the same 9 latents, and matches
  `StreamingTAEHV` to float noise (`1.7e-6`).

### `lighttaew2_1` — not actually a separate architecture

The researcher's note expected `lighttaew2_1` to be a cheaper Conv2D TAE
needing heavy LightX2V machinery. It is neither. Reading LightX2V's
`lightx2v/models/video_encoders/hf/tae.py` against madebyollin's `taehv.py`,
LightX2V's `TAEHV` is a fork of the base variant with the *same* block layout
(`decoder_time_upscale=(True, True)` ≡ madebyollin's default
`(False, True, True)` after its two-tuple promotion, `decoder_space_upscale`
all-True, `patch_size=1`, 16 latent channels). The state-dict key indices line
up one-for-one, and `lighttaew2_1.pth` loads **strictly** into the vendored
madebyollin `TAEHV` with an identical 9.84 M-parameter decoder.

So `lighttaew2_1` is a different *training run*, not a cheaper network: expect
`taew2_1` speed with different quality. No LightX2V dependency was pulled in
for it, and it shares `_TaehvChunkStream` with `taew2_1`.

## Vendored third-party code

Under `interactive/vendor/` (no `__init__.py`; `interactive/` is already a
namespace package):

| file | source | notes |
|---|---|---|
| `taehv.py` | `https://raw.githubusercontent.com/madebyollin/taehv/main/taehv.py`, fetched 2026-08-31 | **unmodified**. MIT — full licence text kept alongside as `taehv_LICENSE`; the README is kept as `taehv_README.md` because it is the citation for the no-latent-scaling rule. sha256 `28b452ad74f924d2d0922d6b3e805dfa7f504c523ced240435fefad5f6f650a7` |
| `lightx2v_wan_vae.py` | `lightx2v/models/video_encoders/hf/wan/vae.py` from `https://github.com/ModelTC/LightX2V` (Apache-2.0), fetched 2026-08-31 | **trimmed**: only the `nn.Module` stack the checkpoint needs (`CausalConv3d` … `WanVAE_`) is kept. Dropped the `WanVAE` outer wrapper (tiling / `torch.distributed` sequence-parallel / cpu-offload) and the lightx2v-internal imports `load_weights`, `GET_USE_CHANNELS_LAST_3D`, `AI_DEVICE`. No retained line was edited. sha256 `b9d9fc35735484b24011bdbf158f71c25a5292794a59142efd77fb530ed19963` |

**Why the LightX2V VAE had to be vendored at all**: it is *not* the repo's own
`wan/modules/vae.py`. LightX2V's `Decoder3d` halves `in_dim` at upsample stages
1/2/3, its `_video_vae` defaults to `temperal_downsample=[False, True, True]`
(upstream Wan uses `[True, True, False]`), and the checkpoint is a `dim=96`,
`pruning_rate=0.75` distillation. The Wan module cannot load these weights;
this one loads them **strictly, all keys matched**.

## Weights

Downloaded to `interactive/decoder_weights/` — 96 MB total. **These are not in
`.gitignore` yet** and WS-F commits nothing; whoever commits this work should
add `interactive/decoder_weights/` and `interactive/decoder_bench_out/` to
`.gitignore` first (the re-fetch commands below make the weights reproducible
from the SHA-256s).

| file | bytes | URL | SHA-256 |
|---|---|---|---|
| `taew2_1.pth` | 22 678 901 | `https://raw.githubusercontent.com/madebyollin/taehv/main/taew2_1.pth` | `d26151e76cdc2c9424bef988de874b33d9a53f30ef3060cd556c429c469c797e` |
| `lightvaew2_1.pth` | 32 208 043 | `https://huggingface.co/lightx2v/Autoencoders/resolve/main/lightvaew2_1.pth` | `d5d6b094d4829fb03b6179be2cf4589baab21c9fea1ca70dcaa8c85ba8183ec9` |
| `lighttaew2_1.pth` | 45 301 802 | `https://huggingface.co/lightx2v/Autoencoders/resolve/main/lighttaew2_1.pth` | `7c55c7d6b8eb3d24c7daca1295dea803fb847411e2c67bceecbb286f4dfdbef5` |

`wan` needs no download — it reads
`/home/ashish/Wan2.1/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth` through the repo's
wrapper.

Re-fetch:

```bash
cd interactive/decoder_weights
curl -L -O https://raw.githubusercontent.com/madebyollin/taehv/main/taew2_1.pth
curl -L -O https://huggingface.co/lightx2v/Autoencoders/resolve/main/lightvaew2_1.pth
curl -L -O https://huggingface.co/lightx2v/Autoencoders/resolve/main/lighttaew2_1.pth
sha256sum *.pth
```

## Bench

`interactive/decoder_bench.py` needs no DiT, no checkpoint and no T5. It reads
N consecutive GT latent frames straight out of the seed zarr, decodes them with
`wan` to make the reference frames, then replays the **same** latents through
each alternative chunk-by-chunk.

```bash
conda run -n flash-q python -m interactive.decoder_bench \
    --zarr ~/20240224003808.zarr --chunks 8 --repeats 3 \
    --wan_model_path /home/ashish/Wan2.1/
```

`flash-q` (not `flash`) because `lpips` is only installed there; without it the
bench still runs and reports PSNR only. It waits for a free GPU by default
(`--gpu_free_mb 2048 --gpu_wait_min 120 --gpu_poll_s 300`); `--skip_gpu_wait`
bypasses that.

Method notes:

* Timing is CUDA-event based with a **full warmup pass** before the measured
  passes — a whole pass rather than a repeated first chunk, so the measured
  stream still starts from a genuine `reset()`.
* The headline **ms/chunk is the median over chunks 1..7**, pooled across
  repeats. Chunk 0 is excluded from the median because it is the ragged one (9
  frames, cold temporal caches) and is reported separately as `ms_chunk0`.
* `fps-equiv` = 12 000 / (ms per chunk) — pixel frames per wall second from
  decode alone. `× realtime` divides that by the 16 fps playback rate.
* Peak VRAM is `torch.cuda.max_memory_allocated` over the measured passes, with
  the allocator reset and each decoder freed between backends so the numbers
  don't contaminate each other.
* PSNR is per-clip on the 0–255 scale; LPIPS is `lpips(net='alex')` averaged
  over frames, both against the `wan` clip. **This is decoder-vs-decoder
  agreement, not absolute reconstruction quality** — the reference is the Wan
  VAE's output, not the original video.

Artefacts land in `interactive/decoder_bench_out/`: one mp4 per decoder, a
stacked labelled `compare_f<k>.png`, plus `results.csv` / `results.md` /
`results.json`.

## Results

<!-- RESULTS-TABLE -->
Measured 2026-08-31 on an RTX 5090 (driver 580.173.02, 400 W cap), bf16,
`--chunks 8 --repeats 3`, decoding zarr latents only (no DiT in the loop).
PSNR/LPIPS are **decoder-vs-decoder agreement against the Wan clip**, not
absolute reconstruction quality — the reference is the Wan VAE's own output.

| decoder | dec params | ms/chunk (med) | ms/chunk (p90) | fps-equiv | x realtime | 6-chunk horizon | speedup | peak VRAM | PSNR vs Wan | LPIPS vs Wan |
|---|---|---|---|---|---|---|---|---|---|---|
| `wan` | 73.3 M | 508.2 | 509.5 | 24 | 1.5x | 3.05 s | 1.00x | 4.19 GB | reference | reference |
| `lightvaew2_1` | 4.6 M | 116.0 | 117.4 | 103 | 6.5x | 0.70 s | 4.38x | 0.96 GB | 32.25 dB | 0.0847 |
| `taew2_1` | 9.8 M | 23.8 | 23.8 | 505 | 31.6x | 0.14 s | 21.39x | 2.54 GB | 29.92 dB | 0.0533 |
| `lighttaew2_1` | 9.8 M | 23.8 | 23.8 | 504 | 31.5x | 0.14 s | 21.36x | 2.54 GB | 31.02 dB | 0.0489 |

Reading it:

* **The TAEHV pair (`taew2_1` / `lighttaew2_1`) are the pick, and they are
  effectively tied.** They share the architecture and are indistinguishable on
  speed (23.8 ms/chunk, 21.4x the Wan VAE). On these *real* zarr latents
  `lighttaew2_1`'s different training run wins both metrics (+1.1 dB PSNR,
  LPIPS 0.0489 vs 0.0533) — but that ordering does **not** survive the move to
  generated latents: in the engine at 4 steps, `taew2_1` scores 28.12 dB and
  `lighttaew2_1` 27.25 dB, reversing the gap. The two are within ~1 dB of each
  other in both directions, so treat them as equivalent and pick on other
  grounds; do not quote the standalone ranking as if it held for rollouts.
* **`lightvaew2_1` scores the highest PSNR (32.25 dB) yet the worst LPIPS
  (0.0847).** PSNR rewards its smoother, pruned-VAE output while LPIPS
  penalises the perceptual detail it loses; on this pair LPIPS is the metric
  to trust, and it is also 4.9x slower than the TAEHV pair for the privilege.
* The Wan VAE is only 1.5x realtime on its own, so at 16 fps playback it
  spends most of a frame budget decoding. Both TAEHV decoders are ~31x
  realtime, which moves the bottleneck entirely onto the DiT.

Engine-level confirmation (full rollout, `interactive/bench.py`, bf16,
compile=off, kv7/h6) — decode time per horizon collapses from 3.4 s (`wan`) to
0.19 s (`taew2_1`), and generation itself speeds up because the Wan VAE had
been competing for SMs on the overlapped decode stream.

This is what puts the engine over the real-time line: at 1 denoising step,
`lighttaew2_1` reaches **16.21** and `taew2_1` **16.12** end-to-end fps against
`wan`'s 10.72, and 16 fps is the playback rate. No `wan` configuration reaches
it at any step count. See `interactive/BENCH_REPORT.md` for the full
steps x decoder table.

## Engine integration

Not wired by WS-F — this is WS-A's call to make in `engine.py`. The exact
signature:

```python
def build_decoder(name: str,
                  device="cuda",
                  dtype: torch.dtype = torch.bfloat16,
                  wan_model_path: str = "/home/ashish/Wan2.1/",
                  weights: Optional[str] = None) -> LatentDecoder
```

* `name` ∈ `("wan", "lightvaew2_1", "taew2_1", "lighttaew2_1")` — feed it
  `cfg.decoder` straight.
* `wan_model_path` accepts either `/home/ashish/Wan2.1/` or
  `/home/ashish/Wan2.1/Wan2.1-T2V-1.3B` (it normalizes the trailing model dir
  itself, same rule as `engine._normalize_wan_path`), and is ignored by the
  three alternatives.
* `weights` overrides the alternative decoders' `.pth` path; leave it `None`
  for `interactive/decoder_weights/<name>.pth`.

What changes in the engine, minimally:

1. Build `self.decoder = build_decoder(cfg.decoder, device=self.device, dtype=self.dtype, wan_model_path=cfg.wan_model_path)` alongside the player.
2. `reset()`: replace `p.vae.model.clear_cache()` with `self.decoder.reset()`.
3. `_decode_gpu(chunk)`: replace the body with
   `return self.decoder.decode_chunk(chunk, half_res=self.cfg.decode_half_res)` —
   the return type (uint8 CUDA `[T, H, W, 3]`) is unchanged, so `_vae_scope`,
   `record_stream` and `_drain` are untouched.
4. Keep the whole thing on the session's single VAE stream. The
   `_vae_stream` invariant matters just as much for the alternatives: all three
   hold per-stream temporal state in tensors allocated inside `decode_chunk`,
   so every decode of a session must run on the same stream.
5. `cfg.decoder != "wan"` means `p.vae` is only needed for the seed *encode*
   path (if any); the alternatives never touch the Wan VAE, but building it
   costs ~0.5 GB, so a later cleanup could skip loading it entirely when a
   tiny decoder is selected. Not done here — `build_player` is WS-A's file.
