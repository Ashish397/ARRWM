# Interactive world model — bench report

Generated 2026-08-31T13:08:48+00:00 from `interactive/bench_results.csv` (24 rows: 17 real, 5 mock, 2 failed).

Tables and commentary below cover **real-checkpoint runs only**. Mock-engine rows and failed configs are listed at the end.

Reference clip: `phase1_step0001000_bf16_s4_coff_kv7_h6` (bf16, 4 steps, compile=off, 609 frames). PSNR/LPIPS below are against it, every 4th frame.

## Denoising steps vs speed and quality

| precision | steps | decoder | compile | e2e fps | gen fps | first-frame s | p95 chunk ms | PSNR vs ref | LPIPS vs ref |
|---|---|---|---|---|---|---|---|---|---|
| bf16 | 4 | lighttaew2_1 | off | 7.18 | 7.22 | 3.268 | 1697.4 | 27.249 | — |
| bf16 | 4 | lightvaew2_1 | off | 6.85 | 7.07 | 3.386 | 1739.8 | 28.993 | — |
| bf16 | 4 | taew2_1 | max-autotune-no-cudagraphs | 7.03 | 7.07 | 3.542 | 2258.4 | 20.456 | — |
| bf16 | 4 | taew2_1 | off | 7.17 | 7.21 | 3.275 | 1708.8 | 28.119 | — |
| bf16 | 4 | taew2_1 | off | 7.15 | 7.18 | 3.290 | 1738.2 | 28.119 | — |
| bf16 | 4 | wan | off | 6.19 | 7.01 | 3.509 | 1802.4 | — | — |
| bf16 | 3 | wan | off | 7.19 | 8.33 | 2.978 | 1533.0 | 19.699 | — |
| bf16 | 2 | lighttaew2_1 | off | 11.42 | 11.50 | 2.048 | 1071.6 | 14.906 | — |
| bf16 | 2 | lightvaew2_1 | off | 10.59 | 11.13 | 2.168 | 1113.4 | 15.0 | — |
| bf16 | 2 | taew2_1 | off | 11.43 | 11.52 | 2.045 | 1070.0 | 14.674 | — |
| bf16 | 2 | wan | off | 8.62 | 10.30 | 2.435 | 1245.3 | 18.796 | — |
| bf16 | 1 | lighttaew2_1 | off | 16.21 | 16.39 | 1.435 | 756.5 | 12.534 | — |
| bf16 | 1 | lightvaew2_1 | off | 14.59 | 15.64 | 1.555 | 798.9 | 12.525 | — |
| bf16 | 1 | taew2_1 | max-autotune-no-cudagraphs | 16.65 | 16.84 | 1.398 | 739.3 | 12.298 | — |
| bf16 | 1 | taew2_1 | off | 16.12 | 16.29 | 1.450 | 779.6 | 12.312 | — |
| bf16 | 1 | taew2_1 | off | 16.07 | 16.25 | 1.463 | 764.2 | 12.312 | — |
| bf16 | 1 | wan | off | 10.72 | 13.43 | 1.899 | 968.7 | 16.405 | — |

## All runs (sorted by end-to-end fps)

| timestamp | precision | denoising_steps | compile_mode | decoder | kv_cache_chunks | horizon_chunks | mean_gen_fps | mean_end_to_end_fps | mean_first_frame_latency_s | p95_chunk_gen_ms | peak_vram_gb | psnr_vs_ref | lpips_vs_ref | ckpt_sha256_first16 | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-08-31T13:08:14+00:00 | bf16 | 1 | max-autotune-no-cudagraphs | taew2_1 | 7 | 6 | 16.844 | 16.652 | 1.3979 | 739.3 | 20.144 | 12.298 | — | b68d99cd77b9154c | real-compile-retest;lpips-unavailable |
| 2026-08-31T11:13:57+00:00 | bf16 | 1 | off | lighttaew2_1 | 7 | 6 | 16.39 | 16.215 | 1.4354 | 756.55 | 20.159 | 12.534 | — | b68d99cd77b9154c | real-decoder;lpips-unavailable |
| 2026-08-31T11:04:32+00:00 | bf16 | 1 | off | taew2_1 | 7 | 6 | 16.289 | 16.117 | 1.4502 | 779.59 | 23.013 | 12.312 | — | b68d99cd77b9154c | real-decoder;lpips-unavailable |
| 2026-08-31T13:07:22+00:00 | bf16 | 1 | off | taew2_1 | 7 | 6 | 16.247 | 16.069 | 1.4631 | 764.24 | 23.025 | 12.312 | — | b68d99cd77b9154c | real-compile-retest;lpips-unavailable |
| 2026-08-31T11:06:16+00:00 | bf16 | 1 | off | lightvaew2_1 | 7 | 6 | 15.636 | 14.591 | 1.5548 | 798.9 | 20.45 | 12.525 | — | b68d99cd77b9154c | real-decoder;lpips-unavailable |
| 2026-08-31T11:00:52+00:00 | bf16 | 2 | off | taew2_1 | 7 | 6 | 11.515 | 11.429 | 2.0453 | 1070.03 | 20.159 | 14.674 | — | b68d99cd77b9154c | real-decoder;lpips-unavailable |
| 2026-08-31T11:12:16+00:00 | bf16 | 2 | off | lighttaew2_1 | 7 | 6 | 11.503 | 11.416 | 2.0482 | 1071.64 | 20.168 | 14.906 | — | b68d99cd77b9154c | real-decoder;lpips-unavailable |
| 2026-08-31T00:19:01+00:00 | bf16 | 1 | off | wan | 7 | 6 | 13.43 | 10.719 | 1.8993 | 968.71 | 21.851 | 16.405 | — | b68d99cd77b9154c | real-matrix-steps;lpips-unavailable |
| 2026-08-31T11:02:51+00:00 | bf16 | 2 | off | lightvaew2_1 | 7 | 6 | 11.127 | 10.586 | 2.1682 | 1113.4 | 20.434 | 15.0 | — | b68d99cd77b9154c | real-decoder;lpips-unavailable |
| 2026-08-31T00:16:45+00:00 | bf16 | 2 | off | wan | 7 | 6 | 10.298 | 8.622 | 2.4352 | 1245.33 | 21.874 | 18.796 | — | b68d99cd77b9154c | real-matrix-steps;lpips-unavailable |
| 2026-08-31T00:14:02+00:00 | bf16 | 3 | off | wan | 7 | 6 | 8.327 | 7.193 | 2.9777 | 1532.95 | 21.867 | 19.699 | — | b68d99cd77b9154c | real-matrix-steps;lpips-unavailable |
| 2026-08-31T11:10:20+00:00 | bf16 | 4 | off | lighttaew2_1 | 7 | 6 | 7.219 | 7.185 | 3.2685 | 1697.41 | 20.153 | 27.249 | — | b68d99cd77b9154c | real-decoder;lpips-unavailable |
| 2026-08-31T10:56:25+00:00 | bf16 | 4 | off | taew2_1 | 7 | 6 | 7.209 | 7.175 | 3.2748 | 1708.8 | 20.152 | 28.119 | — | b68d99cd77b9154c | real-decoder;lpips-unavailable |
| 2026-08-31T13:04:36+00:00 | bf16 | 4 | off | taew2_1 | 7 | 6 | 7.182 | 7.148 | 3.2897 | 1738.24 | 20.145 | 28.119 | — | b68d99cd77b9154c | real-compile-retest;lpips-unavailable |
| 2026-08-31T13:06:28+00:00 | bf16 | 4 | max-autotune-no-cudagraphs | taew2_1 | 7 | 6 | 7.066 | 7.032 | 3.5417 | 2258.39 | 20.16 | 20.456 | — | b68d99cd77b9154c | real-compile-retest;lpips-unavailable |
| 2026-08-31T10:58:56+00:00 | bf16 | 4 | off | lightvaew2_1 | 7 | 6 | 7.072 | 6.85 | 3.3858 | 1739.84 | 20.46 | 28.993 | — | b68d99cd77b9154c | real-decoder;lpips-unavailable |
| 2026-08-31T00:11:07+00:00 | bf16 | 4 | off | wan | 7 | 6 | 7.015 | 6.194 | 3.5086 | 1802.39 | 21.867 | — | — | b68d99cd77b9154c | real-ref-bf16-s4;reference |

## Commentary (auto-generated)

- Fastest overall: **bf16 / 1 steps / decoder=taew2_1 / compile=max-autotune-no-cudagraphs** at 16.65 end-to-end fps (1.398s to first frame).
- **Real-time capable (>= 16 e2e fps): 4 config(s).** Fastest: 1 steps/taew2_1 (16.65 fps), 1 steps/lighttaew2_1 (16.21 fps), 1 steps/taew2_1 (16.12 fps)
- Best fps at quality tier medium (25-30 dB): bf16 / 4 steps / decoder=lighttaew2_1 / compile=off — 7.18 fps, PSNR 27.249 dB
- Best fps at quality tier low (< 25 dB): bf16 / 1 steps / decoder=taew2_1 / compile=max-autotune-no-cudagraphs — 16.65 fps, PSNR 12.298 dB
- 1 steps: mean 15.06 e2e fps (2.17x vs 4 steps), mean PSNR 13.06 dB
- 2 steps: mean 10.51 e2e fps (1.52x vs 4 steps), mean PSNR 15.84 dB
- 3 steps: mean 7.19 e2e fps (1.04x vs 4 steps), mean PSNR 19.70 dB
- 4 steps: mean 6.93 e2e fps (1.00x vs 4 steps), mean PSNR 26.59 dB
- LPIPS is NaN/blank for some rows: the `lpips` package is not importable in this env, and the harness never installs into `flash`.

## Failed configs (excluded from the tables above)

- **bf16 / 4 steps / compile=reduce-overhead** (2026-08-31T00:26:03+00:00): real-matrix-compile;FAILED: torch.compile mode=reduce-overhead incompatible with the engine's persistent KV/cross-attn cache. Compilation succeeded; hard error at CUDA-graph record time (cudagraph_trees record_function -> _copy_inputs_and_remove_from_src): 'accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run' at causal_model.py:589 cross_attn_ffn, raised during engine.reset() chunk commit. Cudagraph trees assume the graph owns its output buffers; the rolling engine keeps cache tensors alive across invocations and mutates them in place. Not an OOM, not a graph break. Fix needs cudagraph_mark_step_begin() per invocation or cloning cache tensors out of the compiled region. Deferred; compile=off is the supported mode.
- **bf16 / 2 steps / compile=reduce-overhead** (2026-08-31T00:26:03+00:00): real-matrix-compile;FAILED: torch.compile mode=reduce-overhead incompatible with the engine's persistent KV/cross-attn cache. Compilation succeeded; hard error at CUDA-graph record time (cudagraph_trees record_function -> _copy_inputs_and_remove_from_src): 'accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run' at causal_model.py:589 cross_attn_ffn, raised during engine.reset() chunk commit. Cudagraph trees assume the graph owns its output buffers; the rolling engine keeps cache tensors alive across invocations and mutates them in place. Not an OOM, not a graph break. Fix needs cudagraph_mark_step_begin() per invocation or cloning cache tensors out of the compiled region. Deferred; compile=off is the supported mode.

## Mock-engine rows (not the real model)

The mock engine is a CPU/synthetic stand-in used to exercise the harness. Its timings are roughly an order of magnitude faster than the real model and its PSNR is a self-comparison, so it is never mixed into the numbers above.

| precision | steps | compile | e2e fps | gen fps | first-frame s |
|---|---|---|---|---|---|
| bf16 | 1 | off | 91.31 | 126.84 | 0.131 |
| bf16 | 2 | off | 63.93 | 88.81 | 0.188 |
| bf16 | 3 | off | 49.02 | 68.08 | 0.244 |
| bf16 | 4 | off | 39.96 | 55.51 | 0.300 |
| bf16 | 4 | off | 39.96 | 55.50 | 0.300 |
