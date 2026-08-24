# A/B/C texture reference scale — all six checkpoints + strict03 baseline

Assembled 2026-08-23 from `eval/texture_abc_<arm>/REPORT.md` (reference wave on
holders 6106319/6106320, complete; logs `logs/texabc_*_p1.log`/`_p2.log`, no
errors). Same ride (20240115085313), same battery, same A/B references in every
run — cross-arm comparison is like-for-like.

## C_late (~25 s) — the depth the decision rule reads

Targets: **HF ratio → B band 0.86–0.89** (from above), **fft anisotropy → ~1.10
(A)**, **angular entropy → 0.978 (A)**, **luma kurtosis → −0.19 (A)**,
**Laplacian kurtosis → ~13.4 (A)**.

| arm | HF C_late/A | fft aniso fy/fx | angular entropy | luma kurt | Laplacian kurt | lap_var C_late/A |
|---|---|---|---|---|---|---|
| *A raw (target)* | *1.000* | *1.102* | *0.978* | *−0.19* | *13.4* | *1.000* |
| *B roundtrip (target band)* | *0.86–0.89* | *1.07–1.15* | *0.93–0.98* | *−0.16/+0.18* | *13.0–14.0* | *0.83–0.84* |
| nogan200 | 1.048 | 0.851 | **0.963** | **−0.915** | 18.7 | 1.30 |
| raw_t0 | **0.803** | **1.182** | 0.918 | −1.111 | **13.7** | **0.92** |
| wave01 | 2.835 | 0.256 | 0.841 | −0.703 | 18.2 | 2.23 |
| wave_ts | 3.067 | 0.208 | 0.804 | −0.772 | 18.1 | 2.23 |
| horizon_nogan90 | 1.075 | 0.987 | 0.957 | −1.184 | 21.7 | 1.41 |
| horizon_wave90 | 0.431 | 0.216 | 0.839 | −1.548 | **66.7** | 0.79 |
| strict03 (baseline pathology) | 2.205 | 0.363 | 0.843 | −1.209 | 13.1 | 2.05 |

Depth amplification (HF C_late/C_early): nogan200 0.98 · raw_t0 0.97 ·
wave01 2.88 · wave_ts 3.04 · horizon_nogan90 1.06 · horizon_wave90 0.34 ·
strict03 1.49.

## Noise floor (added 2026-08-23, 4 noise seeds per reference checkpoint)

`--seed` is a no-op under `ODE_FLOW_REC` (chunk noise comes from
`ODE_FLOW_SEED`, `utils/eval_causal_AR.py:976-981`); the pipeline is otherwise
bit-deterministic. Varying `ODE_FLOW_SEED` (1234/43/44/45) on the same ride:

| metric (C_late) | nogan200 mean±sd | raw_t0 mean±sd | gap in σ |
|---|---|---|---|
| hf_power_frac | 0.0175 ± 0.0020 | 0.0115 ± 0.0023 | **2.8** |
| laplacian_kurtosis | 18.8 ± 0.9 | 14.9 ± 1.4 | **3.4** |
| fft_aniso fy/fx | 0.94 ± 0.24 | 1.22 ± 0.34 | 1.0 |
| angular_entropy | 0.968 ± 0.011 | 0.952 ± 0.024 | 0.9 |
| luma_kurtosis | −0.88 ± 0.25 | −1.13 ± 0.09 | 1.5 |

(C_early is tighter: HF separates at 6.2σ.) Consequences:

1. **HF power and Laplacian kurtosis are the reliable axes** — they separate
   the two references at ≥2.8σ; raw_t0's lower HF is a REAL effect, not seed
   luck.
2. **C_late anisotropy and angular entropy are noise-dominated at this
   granularity** (seed sd up to ±0.34 on anisotropy) — single-seed readings
   of these axes cannot rank close arms. The pathological arms (aniso
   0.21–0.36, HF 2–3×) remain far outside even this noise.
3. **Protocol for judging V/new arms**: run the diagnostic at ≥3
   `ODE_FLOW_SEED`s, compare means, and treat differences under ~2× the
   pooled sd above as unresolved.

## ganfix_poolrich (TF head, run h6106490_094745) — multi-seed, added 2026-08-23

4 noise seeds (1234/43/44/45), C_late mean±sd, alongside the references:

| metric (C_late) | nogan200 | raw_t0 | **poolrich** | target |
|---|---|---|---|---|
| hf_power_frac | 0.0175±0.0020 | 0.0115±0.0023 | **0.0241±0.0037** | 0.0159–0.0165 (B) |
| laplacian_var | 0.0157±0.0021 | 0.0109±0.0008 | **0.0206±0.0019** | ~0.0097 (B) |
| patch8_std (local contrast) | 0.0629±0.0095 | 0.0733±0.0038 | **0.0559±0.0048** | ~0.052 (B) |
| fft_aniso fy/fx | 0.94±0.24 | 1.22±0.34 | **1.10±0.17** | 1.10 (A) |
| angular_entropy | 0.968±0.011 | 0.952±0.024 | **0.938±0.014** | 0.978 (A) |
| laplacian_kurtosis | 18.8±0.9 | 14.9±1.4 | **16.0±1.4** | ~13.4 (A) |
| luma_kurtosis | −0.88±0.25 | −1.13±0.09 | **−0.91±0.16** | −0.19 (A) |

**Verdict.** Poolrich is the opposite failure of the wavelet arms' cousin, not
a texture win: it fabricates the MOST high-frequency energy of the three (HF
1.30× A at C_late, ~2.3σ above nogan200, ~4σ above raw_t0; Laplacian variance
highest) while having the LOWEST local contrast (patch8_std). That
quantitative pair — high fine-grain energy, low local contrast — is
"grainy-but-soft", matching the researcher's perceived sharpness loss. Its
anisotropy mean (1.10) is the closest to natural but that axis is
noise-dominated; angular entropy is the worst of the three (~2σ below
nogan200). raw_t0's signature is the inverse: lowest HF, highest local
contrast — "clean and sharp".

## Readings

1. **A11 instrument validation: PASSED.** The battery reproduces the known
   human ranking `horizon_nogan90 > horizon_wave90` that the old CLIP/isotropic
   instruments inverted. wave90's signature is unmistakable: anisotropy 0.216,
   angular entropy 0.839, Laplacian kurtosis 66.7 (HF concentrated in sparse
   structured lines = banding), HF power collapsed to 0.43× (texture death).
   The instrument is fit to judge the V arms.
2. **The two decision-rule references SPLIT the battery.** raw_t0 wins HF
   (0.803), anisotropy (1.182 vs A 1.102), Laplacian kurtosis (13.7 ≈ A) and
   lap_var; nogan200 wins angular entropy (0.963 vs 0.918) and luma kurtosis
   (−0.915 vs −1.111). Neither dominates the other. See critique item 1.
3. **raw_t0 overshoots the HF target downward**: 0.803 is *below* the B band
   (0.86–0.89), already at C_early (0.825). It is mildly over-smoothed, not
   "toward B from above" — the only arm on that side.
4. **nogan200 posts near-natural HF (1.05) at 25 s despite melting at ~8 s.**
   Melting reads as *benign* on most of this texture battery (only luma/Laplacian
   kurtosis react). The battery measures texture statistics, not structural
   survival. See critique item 5.
5. **Wavelet arms confirmed catastrophic and depth-amplifying** (HF 2.8–3.1×,
   aniso 0.21–0.26, Cl/Ce ≈ 3), both worse than strict03. wave_ts worst overall
   — consistent with "wavelet lost at BOTH timesteps".
6. **Long horizon amplifies kurtosis pathology** (horizon_nogan90 lap kurt 21.7
   vs nogan200 18.7) at similar HF — horizon training trades texture statistics
   slightly differently, but the no-GAN arms are the two best on angular
   entropy.
