# No-critic ablation

`generator_action_z_guidance_weight: 0.3 -> 0.0`; everything else matches the
Default recipe. Evaluated over the standard directional benchmark: 32 held-out
low-egomotion contexts x 8 compass commands = 256 rollouts, scored with the same
frozen instruments as every other model.

## Result

Removing the critic does not weaken control. It removes it.

| axis | no-critic | Default | population |
|---|---|---|---|
| geometric corruption | 1.6% | 16.8% | all 256 |
| scene relocation | 10.9% | 10.9% | active |
| style shift | 0.0% | 2.5% | active |
| high-frequency degradation | 16.3% | 27.2% | active |
| conjuration | 0.4% | 1.2% | all 256 |
| **control failure** | **77.3%** | **0.4%** | all 256 |
| **legitimate** | **17.2%** | **73%** | feature-valid |
| active population | **92** | 239 | of 256 |

It wins or ties four of the five quality axes and fails control almost entirely.
The active-population figure is the explanation: 64% of its rollouts are
near-static, against 0.4% for Default. A camera that does not move cannot melt
geometry, drift in style, lose sharpness or conjure anything, so the quality
columns measure the absence of events rather than the presence of quality.

Realised throttle spread across the eight commands is 0.072, against 0.652 for
Default. Steering is deader still: -0.003 on L and +0.003 on R, against -0.685
and +0.674. Median image translation is 26.5 against 310.6.

This is a different failure from No-AdaLN, which moves confidently and always
forward (spread 0.287, all positive). No-critic barely moves and responds to
nothing.

Two things it does *not* break. Scene relocation is identical to Default at
10.9%, so when it moves it is exactly as well-behaved. And under a no-op command
it holds still about as well as Default (camera motion 1.838 against 1.498, with
the real reference at 1.064) — unsurprising for a model that ignores its actions.

## Figures

`wedges/wedge_g0.png` is the one to look at: the No Critic panel has no signed
structure at all, next to Default's blue forward and red reverse lobes, and next
to No AdaLN's large all-forward lobes. `wedge_g1.png` shows the same for
steering. The `wedge_all_*` variants place the fleet baselines alongside.

Generated with the ablation panel set, substituting No Critic for the duplicate
Default slot that the published figure carries in that position.

## Files

| file | contents |
|---|---|
| `results/headtohead_nocritic.csv` | CoTracker->PCA readout, `g0`..`g7` per rollout |
| `results/vlm_nocritic.csv` | VLM probes: `p_uncanny`, `p_style`, `p_novel` |
| `results/consensus_nocritic.csv` | sibling-consensus relocation inliers |
| `results/style_nocritic.csv` | DINOv2 style drift |
| `results/hf_nocritic.csv` | sibling-relative sharpness loss |
| `results/popin_nocritic.csv` | conjuration detector output |
| `results/static_nocritic.csv` | near-static mask defining the active population |

Thresholds are the deployed ones, identical to the published table:
`p_uncanny > 0.5`, `consensus_inl < 50`, `dino_drift > 0.72`, `B > 150`, pop-in
`flag`; control failure is near-static or realised motion more than 90 degrees
from the command.

## Not included

The stationary quality row. The no-op videos for this ablation are present and
its camera-drift row is reported above, but scoring the five quality axes over
the stationary set would mean re-running instruments across the other models,
which is out of scope — the published stationary table stands as it is.

The instrument-validation AUCs do not extend to this model: it is not in the
100-rollout human-labelled subset, so its scores use the deployed thresholds
without validation figures behind them for this model specifically.
