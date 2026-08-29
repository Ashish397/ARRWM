# Definitive CARN + Flash GAN + decoder pullback knowledge transfer

Date: 28 August 2026.

This document explains the mechanism and evidence behind the current
definitive stationary Phase 3. It is deliberately more architectural than the
live-run onboarding. Read `docs/PHASE3_STATIONARY_DEFINITIVE_2808.md` for the
exact launch recipe and current W&B IDs.

## Executive summary

The current system is strong because each component has one narrow job:

- DMD/Flash produces the action-conditioned generative trajectory.
- CARN models and corrects autoregressive drift with aligned forward and
  reverse operators.
- The harmony bridge teaches the GAN a normalized one-step correction pattern
  without doubling its generator field.
- The VGG GAN judges texture on decoded Flash frames.
- The Q1~0.99 structural decoder pullback transports the current pixel
  cotangent back to the generator without retaining the full decoder graph.
- `raw_grad_hp` blocks broad brightness-patch gradients while allowing the
  forward image mean to evolve.
- The stat anchor owns global distribution statistics.
- A frozen, previously trained action critic at `.3` supplies a small stable
  action-retention force without learning alongside the GAN.

The integrated 200-step stationary recipe is being trialled now. Individual
mechanisms have evidence; the complete integration remains provisional until
its step-100/200 checkpoints and videos are reviewed.

## System flow

```text
Phase-2 KL-ODE step 400
          |
          v
stationary seven-chunk AR generator ---- frozen action critic (.3)
          |                                  |
          | Flash t=60 graph-live slab       | latent action guidance
          v                                  v
decoded frames -> frozen VGG statistics -> trained GAN head
                                          |
                                          | current pixel cotangent
                                          v
                         Q1~0.99 graph-free decoder pullback
                                          |
                                          v
                                  generator latent field

CARN in parallel:
R1 --F(+drift)--> R2 --G(-drift)--> R1
 |                     |
 |                     +--> aux-minus target
 +--------------------------> staged committed-memory correction
 F(former) -> G(F(latter)) --> normalized harmony transition
```

## CARN constituents

### The state domains

Use `R1` for the less-drifted rollout state and `R2` for the next, more-drifted
state. The coherent implementation trains two distinct maps on aligned rollout
pairs:

```text
F : R1 -> R2
G : R2 -> R1
```

`F` is the forward noiser: it learns one increment of autoregressive texture
drift. `G` is the reverse noiser: it learns the correction back toward the
less-drifted domain. The important redesign is that G receives direct paired
`R2 -> R1` grounding; it is not trained only through the synthetic cycle
`G(F(R1)) -> R1`.

### Pair and cycle training

The forward pair objective grounds F on real rollout progression. The cycle
objective keeps F and G mutually consistent. During the reverse part of the
cycle, F is frozen so the system cannot minimize the objective by moving both
maps together in an arbitrary direction.

Both operators preserve the consumer input's per-channel spatial mean and its
centred mean-absolute contrast. This shared moment contract is essential: the
old consumers used different moment rules and could fight the stat anchor even
when their texture directions agreed.

### Aux-minus/internalisation

Aux-minus uses the shared reverse map as a target:

```text
target = G(raw rollout)
reverse_noiser_internalize_weight = 0.25
```

This teaches the generator to internalize the correction without replacing
the score, Flash slab or recurrent state directly. In the matched standalone
study, W&B `334wiw34` was visually the cleanest CARN treatment and preserved
detail better than standalone plus-former or literal minus-latter.

### Staged committed-memory correction

The same G is used on committed autoregressive memory:

```text
committed = blend(raw commit, G(raw commit), alpha)
```

The definitive schedule is deliberately staged:

```text
step <= 100: alpha = 0
step 150:    alpha = 0.125
step >= 200: alpha = 0.25
```

Aux-minus is active from the beginning; recurrent correction is introduced
only after the generator has adapted. This avoids the historical immediate
full-strength `commit_aux` collision. Commit also uses the same absolute
rollout level as the streaming slab rather than a fixed level-one assumption.

## The harmony bridge

The bridge constructs a coherent real transition:

```text
F(clean former) -> G(F(clean latter))
```

Semantically this is `+1 -> 0`: the former demonstrates one forward drift
step, while the latter demonstrates the result after one valid correction.
G never consumes clean ground truth; it consumes an F-produced R2-domain
state, which matches its training domain.

The bridge shares a fixed unit GAN generator budget:

```text
gt_vs_fake_weight = 0.5
gt_transition_weight = 0.5
```

The coherent no-transition control uses `gt_vs_fake_weight=1.0`. This split is
not cosmetic. Earlier stacks enabled two full GAN objectives and accidentally
doubled the generator field, so their apparent synergy was confounded by
scale. The harmony calibration normalized that budget and grounded all
negative consumers on the same G and moment contract.

Weight-zero calibration W&B `udw86qaq` proved the bridge route, paired reverse
contract, staged commit and bounded displacement. The active mechanism screen
`ylahxo9v` was visually accepted over its coherent control. The integrated
DC-off/stationary treatment is the current W&B `kv9axmpu` trial.

## Why standalone “plus former” is not used

There is a crucial distinction:

- `former_plus` is the standalone historical treatment that applies F to a
  clean former while using an otherwise literal transition target.
- The harmony bridge also uses F on the former, but it simultaneously maps
  the latter through `G(F(latter))`, respects G's input domain, shares the
  moment contract and splits a fixed GAN budget.

Therefore a bridge log mentioning `[CARN-FORMER]` does not mean the standalone
`CARNMODE=former_plus` treatment has been promoted.

Standalone plus-former W&B `p68q87lv` completed and proved its route, but its
late outputs accumulated a fine lattice and soft chromatic cast; final detail
and high-frequency retention were materially below aux-minus. Earlier mixed
stacks also queried G outside its trained domain or encoded an approximate
two-step target. Plus-former remains in the launcher only as a reproducible
ablation. It is not ready to be part of the definitive recipe.

Literal minus-latter is also not promoted: it accumulated directional
texture/haze and adds a clean-GT extrapolation caveat without winning visually.

## The replacement Flash GAN

The GAN is not the old repository GAN with a new label. The production route
is explicitly aligned on both sides:

```text
flash_dmd_enabled=true
flash_dmd_gan_t=60
ladd_fake_sample_source=flash
pix_finish_grad_enabled=false
pix_flash_grad_select_enabled=true
```

The discriminator trains on decoded frames from the Flash t=60 slab. The
generator query selects the latest contiguous graph-live frames from that same
slab. The frozen pretrained VGG feature maps are summarized through pooled
texture statistics; the VGG encoder stays frozen while its adapter/head
trains. The selected operating point uses D LR `1e-3`, one discriminator
update per step and R1 `1`.

### Two live-GAN bugs that must remain fixed

1. The detached pixel cache once keyed fresh micro-group tensors by
   `data_ptr`. Allocator address reuse produced false hits and stale real,
   fake and R1 pixels. Production now keeps
   `ladd_pixel_decode_cache=false`; every micro-group decode must be genuine.
2. The Q1 field once queried D before a deferred current-batch update. The
   definitive order is inline D first, proof that the update counter advanced,
   then query the new head and pull back its pixel cotangent:

```text
ladd_defer_disc_update=false
surrogate_decoder_fresh_disc_order=true
surrogate_decoder_disc_updated_before_field=1
```

Any run missing either correction is not parameter-selection evidence.

## How the Q1~0.99 surrogate works

The original decoder-shaped pullback approximated too many sensitive residual
stages and reached only about Q1~0.6. A blockwise/full-chain audit showed that
small local angular errors were amplified by later decoder stages. The final
design stopped approximating those sensitive residual VJPs.

The immutable bundle uses:

- exact fixed stages `4,8,12,16`;
- exact tied/manual residual VJPs `1--3,5--7,9--11,13--15`;
- the exact public latent prefix;
- only the frozen learned stage-0 low core.

Measured full-chain results:

| gate | worst-seed Q1 |
|---|---:|
| paired, 81 examples | `0.990999` |
| Cartesian, 108 unrelated z/v pairs | `0.992351` |

The all-residual path is present on the first field and requires zero fitting
or reconvergence updates when the GAN head changes. It transports the current
cotangent because the residual VJPs are analytic/tied to the frozen WAN
decoder. Only changing the frozen VAE invalidates that structural claim.

Measured combined capture/reverse cost is `188.29 ms / 4.09 GB`, versus
`208.59 ms / 11.67 GB` for differentiable forward plus exact autograd VJP.

Artifact:

`/scratch/u6ex/as1748.u6ex/ARRWM_data/gan_aligned_discrimination/surrogate_field_2708/local_vjp_granular_audit_r0_h6158256/exact_residual_ladder_2808/decoder_shaped_all_residual_seed0.pt`

SHA-256:

`de82ec3211ec408f34cca883be53c15d835e19d5a3860a0b7fe8840dad450357`

## Stat anchor and brightness control

### What the stat anchor owns

The stat anchor is retained at weight `1.0`. It stabilizes the desired global
statistics and provides a reference for the image's mean/contrast trajectory.
Removing it was a useful ablation: the GAN alone did not reliably preserve
those statistics. It is not redundant with the texture GAN.

The anchor uses clean-match offset handling (`stat_anchor_use_clean_match_offset=true`,
`stat_anchor_match_k=2`) in the harmony recipe. All CARN consumers use the
same moment-preserving contract so they do not tug against it.

### Why global DC rejection was insufficient

An image-wide zero-mean cotangent can still contain a broad positive patch
balanced by a weak negative field elsewhere. That produces the observed local
bright/milky deltas even though the global DC number looks perfect.

The accepted solution is `raw_grad_hp`:

- discriminator forward view: raw RGB, so the scene mean may evolve;
- generator cotangent: fixed B3 high-pass followed by exact zero-mean
  projection;
- SWT contribution: zero.

Synthetic checks rejected more than 89% of a broad patch gradient while
retaining more than 90% of checkerboard detail. The cache-off/fresh-order
DC-off screen `v7k43yrb` was visually accepted and chosen over the also-good
DC-on `ne3hjgsd` because raw forward evidence permits natural mean evolution.

This is called “DC off” because D's forward input is not DC-centred. The
generator-side brightness gradient is still high-pass/zero-mean protected.
Keep the stat anchor on. Do not substitute the raw nearest matcher, and do not
reintroduce SWT/wavelets: the user explicitly discarded that route.

The remaining delayed milkiness followed a motion stall when the seed context
left the rolling window, then resolved as motion resumed. Its timing points to
attention/curriculum rather than a direct brightness-gradient failure. This
motivated the current 200-step stationary first-seven-chunk phase before any
later rolling phase.

## Frozen action critic at 0.3

The VGG GAN is deliberately action-blind. A pooled VGG action-conditioning
experiment received the correct shuffled actions and learned strong real/fake
texture margins, yet correct-minus-wrong-action margins stayed essentially
zero and the mismatch loss stayed at `log(2)`. It is not the action solution.

Action retention is supplied separately by the earlier checkpoint-trained
action critic:

```text
action_critic_aux_enabled=true
action_critic_freeze=true
action_teacher_mode=off
generator_action_z_guidance_weight=0.3
```

The critic loads strictly, has no optimizer and stays frozen while gradients
still reach the generator latent. This isolates a stable action direction from
the moving texture discriminator.

Two stronger alternatives were screened: frozen guidance doubled to `.6`
(`ea2g8x49`) and an online 59.3M-parameter critic with two updates per step
(`pstbyie5`). The researcher's visual verdict is that stronger and online
action treatments appear to break trajectory/model behaviour. Treat that as a
qualitative selection rather than a universal quantitative theorem, but do
not promote either. The definitive value is the earlier frozen critic at `.3`.

## Current definitive trial and causal control

The integrated GAN treatment is W&B `kv9axmpu` on child `6170183.15`. It starts
from KL-ODE step 400, trains a fixed 21-frame/seven-chunk stationary window for
200 steps, samples every 15, and saves full checkpoints at 100 and 200.

The matched no-GAN control is W&B `yi4iwouc` on child `6170182.6`. It disables
the discriminator, Q1 pullback and every LADD pair objective with last-wins
runtime settings while retaining Flash DMD, stat anchor, CARN F/G cycle,
aux/internalisation, staged commit, action critic and stationary geometry.
Because the harmony bridge itself is discriminator-side, it is expected to be
inert in the no-GAN control.

This pair answers whether the promoted adversarial texture route adds value
over the otherwise matched stationary DMD/CARN/action system. It does not
re-open the choice of wavelet, plus-former or online action critic.

## Evidence discipline for the next agent

- Call the Q1 numbers measured.
- Call harmony and brightness mechanism screens visually accepted/calibrated.
- Call the complete stationary recipe a live definitive trial until step 200.
- Compare matched checkpoint steps, not unmatched final frames.
- Inspect textured regions and motion/action behaviour, not only sky exposure.
- Prove flags from resolved config and runtime counters.
- Preserve negative controls and report uncertainty rather than manufacturing
  synergy from interacting loss values.

