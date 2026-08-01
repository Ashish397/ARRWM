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
| `scene_consensus`, `blind_scene_reloc`, `blind_scene_features` | scene relocation: ORB + RANSAC place identity |
| `blind_style_shift`, `style_shift` | style shift: VGG Gram + CLIP against a real reference |
| `fleet_hf` | high-frequency degradation: Laplacian sharpness loss |
| `conjure_*`, `fleet_novelty`, `blind_temporal_novelty` | conjuration — objects appearing from nothing. Called "novelty" in the earlier naming; same axis. |
| `blind_vlm_probes`, `vlm_external`, `blind_plausibility` | geometric corruption, the deployed Qwen3-VL probe |
| `blind_melt`, `melt_vlm_bench`, `melt_auc_table` | the melt probe: evaluated, reported, not deployed |
| `fleet_pal`, `pal_local` | PAL4VST artifact fraction: evaluated, reported, not deployed |
| `fleet_dino`, `blind_dino_drift` | DINOv2 cosine drift, a relocation competitor |
| `blind_adjudicate`, `blind_cpu_metrics` | the instrument bake-off behind the validation appendix |
| `stationary_*`, `noop_*` | the no-op evaluation and its camera-drift wedges |

## Running

The rollout videos are far too large to ship, so every path comes from the
environment:

```bash
export AF_FLEET_DIR=/path/to/grids          # holds grids_A/ and baselines/
export AF_TILES_DIR=/path/to/tiles          # de-tiled ours-variant rollouts
export AF_BLIND_DIR=/path/to/blind100       # human-labelled subset
python scene_consensus.py                   # AF_SCENES=r00_B,r00_F limits the run
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
