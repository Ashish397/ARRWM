# Action Forcing

Code for *Action Forcing: Training World Models on Unsupervised Video by
Recovering Underlying Egomotion Bases*.

The method turns unlabelled driving video into action-supervised training data.
Pixel displacements are tracked on a fixed grid, principal components of those
displacements over a large corpus recover a signed, scalable, composable
egomotion basis, and a video DiT is adapted to that basis by an online latent
critic that distils a frozen decoder-tracker-PCA teacher without
backpropagating through the decoder or the tracker.

This release holds training, preprocessing, the evaluation instruments, and the
scripts that regenerate the paper's figures.

`validation/` holds the evidence that this release reproduces the reported
numbers, including what was checked by running it and what was not.

## Setup

Four environment variables locate everything. Nothing else is assumed about
your filesystem.

| variable | meaning |
|---|---|
| `AF_ROOT` | this directory |
| `DATA_ROOT` | the FrodoBots corpus (see *Data* below) |
| `WAN_MODELS` | directory containing the pretrained `Wan2.1-T2V-1.3B/` |
| `HF_HOME` | a writable HuggingFace cache |

```bash
export AF_ROOT=$PWD
export DATA_ROOT=/path/to/frodobots
export WAN_MODELS=/path/to/wan_models
export HF_HOME=/path/to/hf_cache
export PYTHONPATH=$AF_ROOT

pip install -r requirements.txt      # install PyTorch first, matched to your CUDA
```

Python 3.10 and PyTorch 2.10 are what the reported runs used.

The SLURM launchers additionally need `CONDA_ROOT` (a conda install providing an
environment named `arrwm`), and some sites set `SCRATCH`. Individual evaluation
scripts take their own optional overrides — `AF_FLEET_DIR`, `AF_BLIND_DIR`,
`PCA_BASIS`, `WG_OUT`, `RC_OUT` and similar — documented where they are used.

**Run scripts from this directory.** Several resolve `configs/`,
`preprocessing/checkpoints/` and `assets/` by relative path.

## Data

The corpus is [FrodoBots-2K](https://huggingface.co/datasets/frodobots/FrodoBots-2K),
public and not redistributed here. Under `DATA_ROOT` the pipeline expects:

```
frodobots_data/output_rides_*/ride_*/recordings/*.m3u8   raw video
frodobots_encoded_*/<ride_ts>.zarr                       Wan-VAE latents
frodobots_motion/output_rides_*/ride_*/motion.npy        CoTracker displacements
frodobots_captions/train/output_rides_*/ride_*/*.json    captions + T5 embeddings
```

The first is the input; the other three are produced by `preprocessing/`.

## Reproducing

### 1. Preprocess

```bash
python preprocessing/pre_encode_direct.py --rides_csv <rides>.csv \
    --output_root $DATA_ROOT/frodobots_encoded     # video -> Wan VAE latents
python preprocessing/pre_encode_motion.py          # CoTracker 10x10 grid -> motion.npy
python preprocessing/ride_level_caption.py --ride_dir <ride> --phase caption
python preprocessing/pre_encode_text.py            # captions -> Wan T5 embeddings
```

The frozen PCA action basis ships as `preprocessing/checkpoints/pca_basis.pt`
and **is the artifact of record — use it as-is.** `preprocessing/fit_pca_basis.py`
is included for provenance: re-fitting reproduces the component directions to
|cos| ≥ 0.998 but shifts the origin of the throttle axis, because the shipped
basis was fitted on a stationary-weighted sample. See `validation/README.md`.

### 2. Train

```bash
python train.py --config_path configs/causal_lora_diffusion_teacher_v14e.yaml \
    --logdir logs/v14e
sbatch sbatch/train_v14e.sbatch          # or, on SLURM
```

One config and one launcher per reported run. `..._v14e.yaml` is the Default
model; `_16node` and `_4node` are the batch-size ablations (global batch 64 and
16 against Default's 32); `_pca2`, `_pca4`, `_noatok`, `_noadaln` and
`_nocritic` are the conditioning ablations. Each is self-contained and trains
from step 0 to 5000.

`ARRWM_ACTION_ENCODER=pca_raw` must match `teacher_action_encoder` in the
config; the launchers set it.

**The first run is slow.** The dataset scans every zarr under `encoded_root` to
build a ride manifest — tens of minutes on the full corpus. It is cached at
`<logdir>/.ride_manifest.pt`; point a symlink at an existing one to skip it.

### 3. Evaluate and plot

```bash
bash sbatch/launch_inject.sh A       # action-injection eval, all 8 runs
python evaluation/chunk_metrics.py   # per-chunk commanded vs realised
python figures/wedge_plots.py        # and the other scripts in figures/
```

`evaluation/quality/` holds the instruments behind the paper's quality tables;
see its own README. **Several read reference CSVs that ship beside them — pass
an explicit output path, or a re-run will overwrite an artifact backing a
published number.**

## Layout

```
train.py  inference.py    entry points
trainer/                  training loop
model/  utils/  pipeline/ the method: action critic, adaLN-Zero, state probe, rollout
wan/                      vendored Wan2.1 backbone (DiT, VAE, T5)
preprocessing/            raw video -> latents, motion, captions, PCA basis
selection/                window scoring and train/eval pool selection
evaluation/               action-injection eval; quality/ = the paper's instruments
figures/                  regenerates the paper's figures
configs/  sbatch/         one config and one launcher per reported run
assets/                   training-window manifest, ride attributes
tests/                    goldens and paper-table assertions (`pytest tests`)
validation/               evidence, including what is and is not verified
```

## Determinism

`seed` is set in every config and offset per rank, so data order and
initialisation are deterministic. Training is **not** bit-reproducible past the
first step: the backward pass uses non-deterministic kernels. Run-to-run loss
agreement is ~1e-3, against a loss that moves by ~0.09 between adjacent steps.

## Known limitations

- Six figures in the appendix are produced by fleet-reel scripts that are not
  part of this release; `tools`-side gating names them explicitly.
- `inference.py` expects keys (`generator_ckpt`, `data_path`, `output_folder`)
  that no shipped config defines; supply your own, or use the rollout path
  exercised by `evaluation/inject_eval.py`.
- Captioning needs `OpenGVLab/InternVL3-8B`, which is downloaded on first use.

## Licence

CC-BY-NC-SA-4.0, inherited through LongLive and Self-Forcing from Wan2.1.
See `LICENSE` and `NOTICE`.
