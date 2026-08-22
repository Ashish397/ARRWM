# Flow maps for the rolling-training runs (2026-08-21)

## Tool

`utils/flow_rolling_maps.py` (new, this analysis). The preferred latent-space
aligned+ellipse fans (`utils/flow_fan_all.py FF_STYLE=ellipse`) need 8-action
battery recordings plus a GPU/VAE, which training-time wandb clips do not
have, so this adapts the suite's existing **video** path — the Farneback
conventions of `utils/motion_check.py` / `utils/r08_robust_motion.py` — while
keeping the aligned+ellipse presentation: every panel shares grid, arrow gain
and |flow| color scale, and each grid cell carries a 1-sigma temporal
covariance ellipse (flow steadiness over the rollout) at its arrow tip.

Invocation (login node, no GPU):

    source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate arrwm
    python utils/flow_rolling_maps.py          # env: FR_RUNS/FR_OUT/FR_SKIP/FR_SCALE

Per clip: 3-panel PNG `analysis/flow_rolling/<run>_<step>.png`
(1 = flow map with ellipses over a faint mid-clip frame; 2 = radial-flow
profile split top/bottom half + pure-zoom fit `v_r = a*(p-c)`, free center;
3 = per-frame fwd-divergence / steer / |flow| time series, motion_check
formulas). Plus `flow_rolling_grid.png` (aligned runs-by-steps contact sheet)
and `flow_rolling_metrics.csv`. Flow at half resolution (416x240), first 14
frames (seed + transient) skipped for the aggregates.

## Clips mapped (wandb `pred_image_rollout` mp4s)

| run | wandb dir | steps |
|---|---|---|
| rollwarm_gan (baseline, no rolling) | run-20260820_125913-owqfk205 | 211 271 331 391 451 496 |
| rollcarn | run-20260820_223449-mpcnyduh | 211 271 331 391 421 |
| rollcombo2 (rollcarn + R1 every-5 gamma 1e3) | run-20260821_011922-yxhnh783 | 211 271 331 376 |
| rolllong (rollcombo2 recipe, 700 steps) | run-20260821_162525-4z2988ef | healthy 256 286 316 346; breakdown 466 481 511 526 541 |

## Observations

- **Healthy flow is a true 3-D egomotion fan in all four arms.** Every
  mid-training clip (all runs, steps 211-451) shows positive divergence
  (fwd +0.5 to +1.2) with strong depth structure: the ground half streams
  3-32x faster than the sky half (bottom/top slope ratio in the CSV), and the
  pure-zoom fit only reaches R2 0.53-0.76. Rolling training does not visibly
  change the *shape* of the flow field vs the baseline at matched steps.
- **rollcarn vs rollcombo2 (GAN-regularization effect): same flow geometry,
  different steadiness.** Mean maps and radial profiles are nearly identical
  (matched steps agree on fwd, b/t within noise), but rollcarn's flow is the
  steadiest of all arms — frame-to-frame jerk (mean |d steer| + |d fwd|)
  0.26-0.36 vs 0.71-0.83 for rollcombo2, and visibly smaller 1-sigma
  ellipses. The extra R1 (every 5 steps, gamma 1e3) buys nothing visible in
  flow structure and roughly doubles temporal jitter at these steps.
- **Baseline rollwarm_gan develops chunk-cadence seam jerk that rolling
  removes.** Its per-frame steer trace grows a sawtooth at the chunk cadence,
  jerk 0.79 (s331) -> 1.83 (s391) -> 2.14 (s451), with spikes to +-3.5
  px/frame at chunk boundaries by s496; rollcarn at the same steps sits at
  0.26-0.36. This is the seam artefact the rolling/CARN recipe
  (boundary_vae_roundtrip + carn_seam_affine + random rolling depth) was
  meant to fix, and the flow shows it fixed.
- **rolllong breakdown (s466-541) is NOT the 2-D zoom signature — divergence
  dies first, then motion freezes.** Healthy window (s256-346) looks like
  every other arm (fwd 0.5-1.2, b/t 10-32). At s466-481 forward divergence
  collapses to ~0 while |flow| *rises* (3.3 px/frame at s481): the field
  becomes a uniform lateral sweep/thrash (all arrows one direction, steer
  -2.2, jerk 4.4, ground/sky asymmetry gone, b/t 0.8-1.9, huge ellipses).
  The s481 zoom fit "R2=0.80" is degenerate: coefficient a=+0.001 with the
  variance carried by the translation terms — i.e. well fit by uniform PAN,
  not zoom. By s511-541 the clips are static texture: |flow| 0.33-0.45 of
  incoherent jitter, fwd = 0.00, zoom R2 0.17-0.22, b/t ~ -0.5. So the
  failure sequence is: egomotion loses its radial (depth) component ->
  uniform pan/thrash -> frozen world.
- **Chunk-cadence oscillation is visible everywhere** (fwd/steer sawtooth at
  a 3-4-frame period in panel 3) — this is per-chunk sampling cadence, not a
  defect unique to any arm; only its amplitude differs (see jerk numbers).

## Limitations

- **Viz sources differ across arms.** rollcarn/rollcombo2/rolllong pin
  `rollout_viz_source=finish` (finish-denoised, inference-parity). The
  rollwarm_gan baseline's code snapshot predates that flag (absent from its
  config.yaml): with its GAN on, the legacy implicit path visualized the
  flash slab / exit-rung tensor. Cross-arm texture/seam comparisons against
  the baseline are therefore confounded by source; the rolling arms compare
  cleanly with each other.
- Clips are single training-time rollouts on *different rides per step* —
  scene content and commanded actions differ, so per-step numbers mix policy
  and scene; only the structural signatures (divergence, depth asymmetry,
  seam jerk, ellipse size) are comparable, not exact magnitudes.
- Pixel-space Farneback flow at half-res, not the latent-space fan/ellipse
  battery; first 14 frames skipped as seed+transient (clip lengths vary,
  108-180 frames). No 8-direction action decomposition is possible from
  these single-action clips.
