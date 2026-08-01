# Action Forcing

Code for *Action Forcing: Training World Models on Unsupervised Video by
Recovering Underlying Egomotion Bases*.

The method turns unlabelled driving video into action-supervised training data.
Pixel displacements are tracked on a fixed grid, principal components of those
displacements over a large corpus recover a signed, scalable, composable
egomotion basis, and a video DiT is adapted to that basis by an online latent
critic that distils a frozen decoder-tracker-PCA teacher without
backpropagating through the decoder or the tracker.

## Layout

```
train.py            entry point
configs/            one self-contained config per reported variant
trainer/            the training loop
model/              action critic, action conditioning, ss_vae, DiT patches
utils/              dataset, Wan wrapper, scheduler, distributed helpers
wan/                vendored Wan2.1 (upstream, unmodified)
preprocessing/         video -> latents, tracking, PCA fit, window selection
eval/               controllability figures and the reference-free quality probes
assets/             action basis, training-window manifest, held-out ride list
tests/              behaviour tests and the end-to-end training gate
docs/               RUNNING.md, DECISIONS.md, CHECKLIST.md
```

## Configs

Seven variants, matching the paper's tables:

| config | paper name | differs by |
|---|---|---|
| `default.yaml` | Ours (Default) | — |
| `batch64.yaml` | Ours (Batch64) | global batch 64 |
| `batch16.yaml` | Ours (Batch16) | global batch 16 |
| `pca4.yaml` | Ours (pca4) | critic supervises 4 components |
| `pca2.yaml` | Ours (pca2) | critic supervises 2 components |
| `no_adaln.yaml` | Ours (No AdaLN) | adaLN-Zero pathway off |
| `no_action_tokens.yaml` | Ours (No Act. Tok.) | action-token pathway off |

Each is self-contained: every hyperparameter that produced a reported number is
readable in one file, with no inheritance to chase.

## Quick start

```bash
pip install -r requirements.txt
export DATA_ROOT=/path/to/frodobots WAN_MODEL_PATH=/path/to/wan-weights

pytest tests/                                    # ~1s, no GPU or data needed
torchrun --nproc_per_node=4 train.py --config_path configs/causal_lora_diffusion_teacher_v14e.yaml
```

`docs/RUNNING.md` covers data preparation, hardware, and the evaluation
pipeline. `docs/DECISIONS.md` records the judgement calls made in preparing this
release. `docs/CHECKLIST.md` maps the AAAI reproducibility checklist to the
files that support each answer.

## The action space

Actions are the leading principal components of tracked grid displacement,
tanh-squashed at 2.5 sigma per component. The first captures forward-backward
motion (58% of tracked-motion variance) and the second yaw (20%); only these two
are exposed to the DiT, while the critic supervises the leading eight.

The basis ships as `assets/pca_basis_ss_vae_8free.pt`. Note it lives inside that
checkpoint rather than in a standalone file — `preprocessing/fit_pca_basis.py`
documents the fitting method but does **not** reproduce the shipped basis. See
`docs/DECISIONS.md` §1; this matters because the two differ, and only the
shipped one puts the no-egomotion command at the paper's
`a_null = (-0.0234, -0.0013)`.

## Reproducibility notes

**Training is not bit-reproducible.** Seeds are set and the first optimiser step
reproduces exactly, but the backward pass uses non-deterministic CUDA kernels,
so from step 2 onward two runs of identical code diverge by roughly 5e-4 in
loss. Measured on an RTX 5090 with torch 2.8 / CUDA 12.8. This is ordinary for
GPU training and does not affect any reported conclusion, but it means loss
curves will not overlay exactly.

**What is not included.** The FrodoBots videos (public, cited in the paper), the
Wan2.1 weights (public), trained checkpoints, and the six baseline model
repositories. `docs/RUNNING.md` says where each comes from.

## License

See `LICENSE`. Vendored `wan/` retains its upstream license; `utils/scheduler.py`
and `utils/wan_wrapper.py` derive from Self-Forcing under CC-BY-NC-SA-4.0, as
their headers record.
