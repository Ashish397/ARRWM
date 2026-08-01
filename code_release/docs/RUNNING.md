# Running this code

## Hardware

Training was run on 8 nodes x 4 GH200 (96 GB) — global batch 32 at the default
config, about four hours to step 5000. The Batch16 and Batch64 variants use 4
and 16 nodes.

The released config does **not** fit on a single 32 GB card: LoRA rank 256 over
21 frames plus the frozen VAE, CoTracker teacher and critic exceeds memory on
the backward pass. To exercise the code on one GPU, reduce
`num_training_frames`, the LoRA `rank`, and `action_critic_base_channels` — that
is what `tests/training_smoke.py` does. It runs the real pipeline, but a
reduced-capacity version of it, so it verifies the code rather than reproducing
training.

Inference is far lighter: 48 denoising steps generate 7 chunks (~6 s of video)
in 86 s on a GH200, so 1.11 fps. At 20 steps, which costs little quality on a
flow model, that rises to 2.72 fps.

## Software

```bash
conda create -n actforcing python=3.10 && conda activate actforcing
pip install -r requirements.txt
```

`ffmpeg` must be on `PATH` (used to encode logged videos). `flash-attn` is
optional; the backbone falls back to PyTorch attention without it. CoTracker is
fetched through `torch.hub` on first use and cached, so the first run needs
network access.

Reference environment: Python 3.10, torch 2.8.0+cu128.

## What you need to supply

| artefact | where from |
|---|---|
| FrodoBots-2K video | public dataset, cited in the paper |
| Wan2.1-T2V-1.3B weights | public release; point `WAN_MODEL_PATH` at the directory *containing* the `Wan2.1-T2V-1.3B` folder |
| encoded latents, motion, captions | produced by `preprocessing/`, below |

Two environment variables locate everything:

```bash
export DATA_ROOT=/path/to/frodobots     # expects frodobots_{encoded,motion,captions}/
export WAN_MODEL_PATH=/path/to/wan
```

## Preparing the data

```bash
python preprocessing/pre_encode.py          # video      -> Wan VAE latents (zarr)
python preprocessing/pre_encode_motion.py   # video      -> CoTracker grid displacements
python preprocessing/fit_pca_basis.py       # displacements -> PCA basis
python preprocessing/score_all_windows.py   # per-window egomotion scores
python preprocessing/harvest_backward_windows.py
python preprocessing/build_balanced_pool.py # -> the training-window manifest
```

The split the paper used ships as `assets/train_windows.json` (2,577 rides,
63,792 windows: 60,000 forward plus 632 sustained-reverse oversampled 6x), so
the manifest-building steps are only needed to rebuild a split from scratch.

**`fit_pca_basis.py` does not reproduce the shipped basis.** The basis actually
used lives in `assets/pca_basis_ss_vae_8free.pt`; the script documents the
method. See `DECISIONS.md` §1 — the difference is not cosmetic, as the two put
the no-egomotion command in different places.

## Training

```bash
torchrun --nnodes=$N --nproc_per_node=4 \
  train.py --config_path configs/causal_lora_diffusion_teacher_v14e.yaml --logdir logs/default
```

Add `--disable-wandb` to run without logging, or set `wandb_entity` and
`wandb_project` in the config. Checkpoints land in `--logdir`; the paper
evaluates `causal_lora_step0005000.pt`.

`auto_resume` picks up the newest checkpoint in the logdir, which is how the
four-hour wall-clock limit was handled.

## Evaluation

The action-forcing benchmark rolls 32 low-egomotion held-out contexts under
eight constant compass commands (F, FR, R, BR, B, BL, L, FL) at norm 0.5, giving
256 directional rollouts, plus one no-op continuation per context. The no-op
uses the calibrated `a_null = (-0.0234, -0.0013)` rather than the numerical PCA
origin.

Sampling is deterministic: block `b` draws noise with seed `1234 + b`, and the
same eight-block sequence is reused for every context and command.

`eval/` holds two families — the controllability figures (`wedge_plots.py`,
`following_*.py`, `response_*.py`) and the reference-free quality probes
(`blind_*.py`, `conjure_*.py`, `fleet_*.py`). Baseline rollouts require the six
baseline repositories, which are not redistributed here.

## Tests

```bash
pytest tests/          # ~1 s, no GPU, no data
```

Covers the action space (including the paper's published `a_null` and variance
split), the critic and conditioning modules, the training split, and equivalence
of every released config against the config that was actually run.

The end-to-end gate needs data and a GPU:

```bash
python tests/training_smoke.py --config <reduced-config> --logdir <dir> --check
```

Step 1 is asserted exactly; later steps carry a 5e-3 tolerance because the
backward pass is not deterministic (see README).
