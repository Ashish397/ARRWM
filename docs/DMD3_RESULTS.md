# DMD3 — what we established (2026-08-18 → 08-19)

All results from 91-step holder probes (2 nodes) and 200-step 8-node arms,
14e teacher (`v14e_pca8_raw/causal_lora_step0005000.pt`), `real_guidance_scale=0`.

## The recipe that won

| component | setting | evidence |
|---|---|---|
| **stat anchor** | `weight=1.0`, `mode=gt_window`, `M2_short=0.1`, `TV_short=0.1`, `M1=0` | the ONLY load-bearing DMD component; removing it collapses to black by ~step 61 |
| **rung sampling** | `dmd_sample_at_rungs=true` | visibly better samples; t drawn from the student's own ladder [1000,625,357,208] |
| **MAE gate** | `enabled=false` (weight always 1) | the gate is monotone in t and SELECTS FOR the oracle regime; W~0.03 at the rungs that form the image |
| **init** | **KL** (`run3_flip2_rollkl10k/step400`) | beats MSE by eye; band metrics disagree but are anti-correlated (see below) |
| **head** | **AR** (`ar_head=1.0, tf_head=0.0`), `commit=student` | better by eye; also 14% faster (42.1s vs 49.3s/gen-step) and 4 GB lighter (43.8 vs 48.0 GB) |
| **GAN** | **gt_transition** + nearest-match | gt_vs_fake lost in all 4 variants |
| **R1** | **`gamma=1e6`, `every_n=1`, `once_per_step=false`** | see below — the big finding |
| wavelet | on, `augment=true`; LL in/out **undecided** | `gtxn_r1e6_LL` running |

Best arm so far: `dmd10k_gtxn_r1e6_h6055957_123157` — KL + AR + gt_transition +
match + wavelet(HF-only) + R1 1e6.

## R1: it works, and gamma had to be ~1e5x bigger than anyone tried

`ladd_r1_normalize_tokens=true` divides the finite-difference by T ~ 4.7e4, so
`grad_sq` carries 1/T^2. gamma=10 was therefore an EFFECTIVE 4.6e-9. Combined
with a 1-in-10 duty cycle (`every_n=10` x `once_per_step=true` x
`dfake_gen_update_ratio=5`) the dose was ~4.6e-11 of nominal.

Measured, gt_transition, KL+AR:

| | gamma=10, 1-in-10 | **gamma=1e6, 5-of-5** |
|---|---|---|
| `r1_grad_sq` | 1.910, 1.063, 0.153 | **0.015, 0.002, 0.007** |
| disc gap | 2.31, 2.03, 2.11, 1.54, 2.79 | **0.64, 0.91, 0.77, 0.57, 1.68** |
| `d_loss` | 0.170, 0.191, 0.211, 0.281 | **0.437, 0.369, 0.421, 0.488** |

R1 drove its own objective down ~100x and narrowed the disc gap from ~2.2 to
~0.8 with `d_loss` rising toward chance — textbook regularisation, a disc that
no longer overpowers the generator. This is the best-looking arm.

NOTE a code trace predicted R1 was structurally BLOCKED (every disc conv is
`spectral_norm`-wrapped, which makes `<dL/dW, W> == 0` and should annihilate
R1's mechanism). **That prediction was falsified by measurement.** Spectral norm
does not fully block it. Trust the measurement.

Also fixed en route: `last_r1_fired` was a plain assignment overwritten by disc
updates 1-4, so it always logged 0; and the modulo cadence could be missed
forever when `every_n` and the gen ratio were not commensurate (now debt-based:
fires when >= n steps have elapsed since the last ACTUAL firing).

## GAN mode: gt_transition beats gt_vs_fake, and the difference is structural

| gt_vs_fake variant | change | outcome |
|---|---|---|
| original | positional pairs, gamma=10 | DEAD disc (gap 0.02, `d_loss` = log2 = chance) |
| `nor1` | R1 off | alive, **pixelates** |
| `v2` | + nearest-match retrieval | alive, **greys out** |
| `v3` | + m1 per-channel magnitude norm | still bad |
| `v4` | + gamma=1e-2 | (running) |

Each fix worked mechanically — R1 rescaling revived the disc, retrieval widened
the gap to 2.9-4.2 (WIDER than gt_transition's 2.3-2.8), m1 removed the
brightness cue — and none produced better samples. The load-bearing difference
is the one thing deliberately kept different: **transition structure**.
gt_transition judges `(chunk_n, chunk_{n+1})` i.e. DYNAMICS; gt_vs_fake judges a
single chunk's APPEARANCE. For a world model the former is the useful question.

Corollary: a WIDER disc gap is not better. v2 had the widest gap of any arm and
worse samples — a wide gap means the disc is winning, not that the generator is
learning.

## Ablated and found INERT (removed from the recipe)

| component | verdict |
|---|---|
| trajectory distillation (`dmd_real_traj_*`) | improved the TEACHER (`m_real` 0.2727 -> 0.2576) but the student was unchanged (`m_fake` 0.3733 vs 0.3711) at **+48% wall clock**. REVERTED. |
| band-local eq.8 normalizer | no difference |
| critic timestep shift (`aux_teacher_timestep_shift`) | no difference; also moot under rung sampling |
| `timestep_shift` / `min_score_timestep` | dead under rung sampling (sampler returns before applying them) |
| forward geometry 1/2/3 chunks (G1/G2/G5) | all identical — BUT the gate was throttling them to W~0.03, so retested with the gate off as `cleanfwd3_*` |

Pattern worth remembering: six components in a row read null while only the
stat anchor changed an outcome. `sa_M2only` / `sa_TVonly` decompose it.

## Metrics warning

**`gen/dmd_mae_gate_m_fake` is ANTI-CORRELATED with sample quality here.** It
ranked mse > kl > ar_kl; the eye ranks the reverse. It measures band MAE under
GT context, and MAE is minimised by regressing to the conditional mean — so an
arm that mean-seeks harder scores BETTER while looking worse. Do not use it to
rank arms. Use rollout measures (`teacher_match@70f`, per-latent std vs GT).

Also: the MAE gate is monotone in `t` and `t` is redrawn every step, so ALWAYS
regress out `gen/timestep` before reading any m_real/m_fake trend. The 35-step
"collapse" that started this investigation was a timestep artefact.

## Saved recipes

- `sbatch/train_dmd3.sbatch` / `_ar.sbatch` — stationary TF / AR
- `sbatch/train_dmd3_flashgan.sbatch` — + full GAN stack
- `sbatch/train_dmd3_futureGT.sbatch` — clean half 3 chunks ahead, RoPE coupled
- `sbatch/train_dmd3_roll.sbatch` — rolling + GAN (this doc's recipe)
