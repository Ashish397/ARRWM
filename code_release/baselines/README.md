# Baseline control interfaces

The code that drives each external world model over the paper's action grid.
This is the material the appendix's baseline-control-interface section
describes: how a command like *forward-left* is expressed in each model's own
control convention, so that all thirteen models are asked for the same motion.

**The baseline repositories and weights are not redistributed here.** Each
runner expects its upstream repository already installed and importable; clone
it from the URL in the runner's docstring, then run the runner from inside that
checkout. Only the interface layer is ours and only that is included.

| file | model | what it does |
|---|---|---|
| `worldcam_runner.py` | WorldCam | camera-trajectory interface |
| `matrixgame_runner.py` | Matrix-Game 2 | keyboard/mouse action mapping |
| `minwm_runner.py` | minWM | action-vector interface |
| `vista_runner.py` | Vista | SVD-based driving-model interface |
| `decode_camera_controls_from_c2w_sequence.py` | Yume | camera-to-world sequence → Yume's camera controls |
| `worldplay_run.sh` | HY-WorldPlay | pose-sequence driver |

`shims/` holds compatibility patches, not method code. Astra and Yume were
released against older library versions and do not import under the versions
this project uses; each `sitecustomize.py` is auto-imported by Python at
interpreter start and restores the removed APIs. Copy the relevant one onto
`PYTHONPATH` when running that baseline. They are included because a reviewer
reproducing a baseline number will hit the same import failures we did.

## Paths

Runners resolve `AF_ROOT` (this release) and `WAN_MODELS` from the environment,
matching the top-level README. They were written for one cluster and are
included as the record of what was run, not as a turnkey harness — expect to
adjust output directories and any scheduler invocation for your own site.

## Directions

All runners emit the same eight-direction grid used throughout the paper —
`F, FR, R, BR, B, BL, L, FL` — one rollout per direction per scene, which is
what makes the 256-rollout fleet comparable across models.
