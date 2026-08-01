# No-critic ablation

`generator_action_z_guidance_weight: 0.3 -> 0.0`; everything else matches the
Default recipe. Evaluated over the standard directional benchmark: 32 held-out
low-egomotion contexts x 8 compass commands = 256 rollouts, scored with the same
frozen instruments as every other model.

## Result

Removing the critic does not weaken control. It removes it — and the one
quality axis it pays for is high-frequency detail.

| axis | no-critic | Default | population |
|---|---|---|---|
| geometric corruption | 1.6% | 16.8% | all 256 |
| scene relocation | 10.9% | 10.9% | active |
| style shift | 0.0% | 2.5% | active |
| high-frequency degradation | **16.3%** | **5.9%** | active |
| conjuration | 0.4% | 1.2% | all 256 |
| **control failure** | **77.3%** | **0.4%** | all 256 |
| **legitimate** | **17.2%** | **73%** | feature-valid |
| active population | **92** | 239 | of 256 |

(An earlier version of this table gave Default's HF as 27.2%; that cell was
wrong — the published figure is 6%, and `reference/fleet_hf.csv` over the
active population with `B > 150` gives 5.9%. Every other cell reproduced.)

The active-population figure is the frame for the quality columns: 64% of
no-critic's rollouts are near-static, against 0.4% for Default. A camera that
does not move cannot melt geometry, drift in style or conjure anything, so
several quality columns measure the absence of events rather than the presence
of quality. `like_for_like.txt` restricts both models to the 76 scenes where
both are active: the geometry advantage survives (5.3% vs 22.4%, p = 0.0022),
so it is a genuine effect rather than survivorship; relocation stops being
significant once populations are matched, so "indistinguishable on the scenes
where both move" is the supportable phrasing; and HF is significantly worse
there too (18.4% vs 5.3%, p = 0.012).

The clean statement: removing the critic severs command from output while
leaving geometric fidelity intact or better, at a measurable cost in
high-frequency detail.

Realised throttle spread across the eight commands is 0.072, against 0.652 for
Default. Steering is deader still: -0.003 on L and +0.003 on R, against -0.685
and +0.674. Median image translation is 26.5 against 310.6.

This is a different failure from No-AdaLN, which moves confidently and always
forward (spread 0.287, all positive). No-critic barely moves and responds to
nothing.

## Stationary (no-op) evaluation

Both models' 32 no-op rollouts, scored with the same instruments over the
current (30 July) generation of the stationary videos. Both columns come from
one fresh pass, so the pair is internally consistent; it is not expected to
reproduce the published stationary table, which was scored on the earlier
generation.

| axis | no-critic | Default |
|---|---|---|
| geometric corruption (`p_imp > 0.3`) | 0.0% | 0.0% |
| scene relocation (`drift_inl < 59`) | 3.1% | 3.1% |
| style shift (`dino_drift > 0.72`) | 0.0% | 0.0% |
| high-frequency degradation (`B > 150`) | **12.5%** | 0.0% |
| conjuration (pop-in `flag`) | 0.0% | 0.0% |
| camera drift (CoTracker; real ref 1.064) | 1.838 | 1.498 |

The single relocation flag is scene 21 for both models — that scene flags all
eight of our variants in the fresh pass, so it is a scene-level artefact. The
no-op row agrees with the directional one: indistinguishable from Default when
asked to hold still, worse only on high-frequency detail.

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
| `results/stationary_vlm_pair.csv` | no-op geometry/style probes, nocritic + Default |
| `results/stationary_cpu_all.csv` | no-op stillness/relocation, all 14 models |
| `results/stationary_popin_pair.csv` | no-op conjuration, nocritic + Default |
| `results/stationary_style_all.csv` | no-op DINOv2 drift, all models |
| `results/stationary_hf_all.csv` | no-op sharpness loss, all models |
| `like_for_like.txt` | quality axes restricted to the common active population |

Thresholds are the deployed ones, identical to the published table:
`p_uncanny > 0.5`, `consensus_inl < 50`, `dino_drift > 0.72`, `B > 150`, pop-in
`flag`; control failure is near-static or realised motion more than 90 degrees
from the command.

## Not included

The instrument-validation AUCs do not extend to this model: it is not in the
100-rollout human-labelled subset, so its scores use the deployed thresholds
without validation figures behind them for this model specifically.
