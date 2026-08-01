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
public and not redistributed here. It is a gated dataset: accept the terms on
the dataset page while logged in, then authenticate before downloading.

```bash
pip install "huggingface_hub[cli]"
hf auth login                                   # or: export HF_TOKEN=hf_...

hf download frodobots/FrodoBots-2K --repo-type dataset \
    --local-dir $DATA_ROOT/frodobots_data
```

It is roughly 2 TB in total. To develop against a slice, restrict the download
to a few ride groups — each `output_rides_N` is a few dozen rides, and the
pipeline is happy with any subset:

```bash
hf download frodobots/FrodoBots-2K --repo-type dataset \
    --include "output_rides_0/*" --local-dir $DATA_ROOT/frodobots_data
```

Extract any archives in place, so that rides sit at
`frodobots_data/output_rides_*/ride_<id>_<timestamp>/`. Each ride holds
`recordings/` plus the control, GPS and IMU logs. **`recordings/` contains three
streams**: the front camera (`uid_s_1000`), the rear camera and the audio (both
`uid_s_1001`). Only the front camera is used; every script selects it by name,
so do not flatten or rename the recordings directory.

Under `DATA_ROOT` the pipeline expects:

```
frodobots_data/output_rides_*/ride_*/recordings/*.m3u8   raw video (downloaded)
frodobots_encoded_*/<ride_ts>.zarr                       Wan-VAE latents
frodobots_motion/output_rides_*/ride_*/motion.npy        CoTracker displacements
frodobots_captions/train/output_rides_*/ride_*/*.json    captions + T5 embeddings
```

The first is the input; the other three are produced by `preprocessing/`.

You also need the Wan2.1 backbone, which is ungated:

```bash
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir $WAN_MODELS/Wan2.1-T2V-1.3B
```

## Reproducing

### 1. Preprocess

Four stages, in this order. Each is independent per ride and safe to shard
across nodes; each skips rides whose output already exists, so an interrupted
run can simply be restarted.

```bash
# 0. index the rides (front-camera stream only) -> rides.csv
python preprocessing/build_rides_csv.py \
    --data_root $DATA_ROOT/frodobots_data --out rides.csv

# 1. video -> Wan VAE latents, one zarr per ride            [GPU, the slow one]
python preprocessing/pre_encode_direct.py --rides_csv rides.csv \
    --output_root $DATA_ROOT/frodobots_encoded \
    --vae_path $WAN_MODELS/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth

# 2. video -> CoTracker 10x10 grid displacements, motion.npy          [GPU]
#    reads DATA_ROOT/frodobots_data, writes DATA_ROOT/frodobots_motion
python preprocessing/pre_encode_motion.py

# 3. video -> ride captions (InternVL3-8B, downloaded on first use)   [GPU]
python preprocessing/ride_level_caption.py \
    --output_rides_dir $DATA_ROOT/frodobots_data/output_rides_0 --phase both

# 4. captions -> Wan T5 embeddings, written beside each caption JSON  [GPU]
python preprocessing/pre_encode_text.py --workers-per-device 2
```

Stage 1 dominates: it decodes and VAE-encodes every frame. Stages 1 and 2 both
read the raw video and are the two you would shard first. Stage 3 is the only
one that needs network access, to fetch the captioning model.

To check a single ride before committing to the corpus, point stage 3 at one
ride with `--ride_dir <ride>` and pass a one-row `rides.csv` to stage 1.

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
