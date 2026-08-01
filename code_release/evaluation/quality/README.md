# Reference-free quality evaluation

Action-forced rollouts have no paired ground-truth continuation, so these
instruments score failure modes that can be judged without a target video. Every
script here corresponds to a number in the paper.

All of these were developed and run on a workstation, not the cluster; the
rollout videos were generated on the cluster and scored here.

## Layout

`fleet_*` run an instrument over the full directional fleet — 256 rollouts
(32 contexts x 8 compass commands) for each of 13 models. `blind_*` run the same
or competing instruments over the 100-rollout human-labelled subset, which is
how each instrument was selected and validated. `stationary_*` and `noop_*` cover
the separate no-op evaluation.

| instrument | what it scores |
|---|---|
| `vlm_external` | geometric corruption, the deployed Qwen3-VL probe: `p_uncanny > 0.5` |
| `scene_consensus` | scene relocation: sibling-consensus ORB/RANSAC place identity, `consensus_inl < 50` |
| `fleet_style_6s` | style shift: DINOv2 drift to the six-second horizon, `dino_drift > 0.72` |
| `fleet_hf` | high-frequency degradation: sibling-relative sharpness loss, `B > 150` |
| `popin_fleet_all` (+ `popin_detect`, `popin_backends`) | conjuration: objects with no history that persist to the end |
| `fleet_static_inl` | the near-static mask defining the active population |
| `fleet_common` | fleet resolver shared by all of the above |

Validation, against the 100-rollout human-labelled subset. These produce the
appendix AUC table and the `*_validation_*` figures, not the main columns:

| instrument | what it validates |
|---|---|
| `blind_vlm_probes`, `blind_plausibility`, `fleet_reel_vlms` | the geometry probe across Qwen3-VL, InternVL3 and Cosmos |
| `blind_style_shift`, `style_shift` | style shift |
| `blind_scene_reloc`, `blind_scene_features` | relocation, and the ORB / SIFT / AKAZE choice |
| `blind_dino_drift` | DINOv2 drift as a relocation competitor |
| `blind_melt`, `melt_vlm_bench`, `melt_auc_table` | the melt probe: evaluated, reported, not deployed |
| `fleet_pal`, `pal_local` | PAL4VST artifact fraction: evaluated, reported, not deployed |
| `blind_temporal_novelty` | the conjuration probe across three VLMs |
| `blind_adjudicate`, `blind_cpu_metrics` | the competitor panels behind instrument selection |
| `hf_distribution`, `hf_fleet_distribution` | the HF ridgeline figures |
| `blind100_common` | shared loader for the blind set |

The no-op evaluation:

| instrument | what it scores |
|---|---|
| `noop_cpu`, `noop_vlm` | the stationary quality axes |
| `stationary_cotracker`, `stationary_signs`, `stationary_freeze` | camera drift and residual scene animation under a no-op |
| `stationary_wedges` | the no-op camera-drift wedge figure |

Superseded instruments have been removed rather than kept alongside: the earlier
conjuration probe and its detector bake-off (replaced by pop-in with RT-DETR),
`fleet_dino` (replaced by `fleet_style_6s`, whose window is capped to the
six-second horizon) and `fleet_novelty` (replaced by pop-in). Keeping two
generations of the same axis is how the wrong one gets used.

## Running

The rollout videos are far too large to ship, so every path comes from the
environment:

```bash
export AF_FLEET_DIR=/path/to/grids          # holds grids_A/ and baselines/
export AF_TILES_DIR=/path/to/tiles          # de-tiled ours-variant rollouts
export AF_BLIND_DIR=/path/to/blind100       # human-labelled subset
python scene_consensus.py fleet             # relocation over the full fleet
```

`pytest tests/test_eval_pipeline.py` checks the resolvers find their data and
that frame extraction is repeatable.

## Resume is stale-blind — recompute after re-rendering

Several scans append incrementally and skip any `(scene, model)` pair already
present in their output. That is a real hazard: if a rollout is re-rendered after
its row was written, the row is **skipped rather than recomputed**, and the file
silently ends up holding numbers from two different generations of the videos.

This has happened. In the reference `fleet_scene_reloc.csv`, all 256 yume
rollouts and matrixgame's BL/BR rollouts were re-rendered after their rows were
written, so those rows describe videos that no longer exist. Recomputed from
scratch, yume's relocation rate moves substantially.

**Delete the output file before a scan whenever any input video has changed.**
Do not rely on resume across a re-render.
