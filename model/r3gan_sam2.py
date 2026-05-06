"""SAM2-pixel R3GAN discriminator with ADM 2D heads.

Migration of the Flash-DMD paper's pixel-GAN strategy onto our
action-forcing video pipeline. The discriminator stack is:

  * Frozen SAM2 Hiera image encoder (multi-scale FPN features).
  * Trainable per-scale heads following the **ADM 2D head spec**
    (Lu et al. 2025, "Adversarial Diffusion Postprocessing",
    arXiv:2507.18569 §B.1, Fig. 6) — the SDXL-Lightning-style
    ConvNet:
        3× [Conv2d 4×4 stride 2 → GroupNorm(32) → SiLU]
        1× [Conv2d 3×3 stride 1 → GroupNorm(32) → SiLU]
        AdaptiveAvgPool2d → Linear(256, 1)
    All convs project to the same 256-channel hidden dim so
    GroupNorm(32) always divides cleanly. No ResBlocks, no
    AttentionBlocks, no zero-init on the conv weights (the linear
    head is zero-init, matching the latent disc convention so D
    starts as a constant-zero classifier — paper-silent, harmless).

Architecture
------------
Input:  pixel video ``[B, F, 3, H_pix, W_pix]`` in ``[-1, 1]``.
Steps:
  1. Reshape to ``[B*F, 3, H_pix, W_pix]``.
  2. ``[-1, 1] → [0, 1]`` then resize to the encoder's expected
     resolution (default 512×512 for Hiera-B+).
  3. ImageNet normalize: ``(x - mean) / std`` with the standard
     ``[0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]`` stats.
  4. Frozen SAM2 image encoder forward → list of multi-scale feature
     maps from the Hiera FPN.
  5. One ADM 2D head per scale, each producing a per-frame scalar
     logit.
  6. Aggregate logits across scales (mean) and across frames
     (configurable: mean / max / topk_mean) → ``[B]`` scalar.

Divergences from ADM
--------------------
* **Backbone**: ADM uses raw SAM ViT-H and hooks layers 3, 6, 9, 12
  — four features at the SAME spatial resolution (ViT doesn't
  downsample). We use SAM2 Hiera-B+ which is hierarchical
  (multi-scale FPN); the closest analog to ADM's "evenly-spaced
  layer hooks" is the FPN's stage outputs (4 features at DIFFERENT
  spatial resolutions). Structural; we accept and document.
* **2D vs 3D head**: ADM has a 3D head variant for video (§B.2,
  3×3×3 Conv3d + AvgPool1d-time). We use ADM's 2D head per-frame
  and pool across frames at the logit level. Cheaper; matches our
  existing R3GAN pooling convention. Punt the 3D head to a future
  ablation if the per-frame disc isn't temporally coherent enough.
* **Loss family**: ADM uses hinge (Eq. 8). We use R3GAN+R1+R2 for
  its local-convergence proof; the disc architecture is paper-
  aligned, only the loss-shape differs.
* **Real input**: ADM feeds raw pixel reals directly. Our dataset
  is preprocessed to latent so real = V(GT_latent) goes through
  the VAE decode roundtrip. Mild distribution shift (V-decode
  manifold vs natural images); auditor-acknowledged.

Memory + compute notes
----------------------
* SAM2 image encoder frozen (``requires_grad=False`` on every
  param). We do NOT wrap the forward in ``torch.no_grad()`` because
  the G-update needs gradient through the input ``pred_pixel`` (←
  VAE-decoded ``pred_image`` ← generator). With params frozen, the
  autograd graph still records the input-side path, but encoder
  weight gradients are not allocated.
* All-fp32 path (encoder + heads + buffers) — R1/R2 second-order
  gradient penalties are numerically unstable under bf16. Memory
  cost ~80 MB params + ~2× activation vs bf16; comfortably within
  budget on Hiera-B+.

Lazy import
-----------
The ``sam2`` package is imported inside the class init so that
unrelated configs (``gan_backbone: latent_3d_conv``) don't require
SAM2 to be installed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Frozen SAM2 image encoder loader.
# ---------------------------------------------------------------------------


def _load_sam2_image_encoder(
    checkpoint_path: str,
    config_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    """Build the full SAM2 model and return ONLY the image encoder
    (Hiera). All params frozen, ``eval()`` mode set.

    SAM2's ``build_sam2`` calls Hydra's ``compose(config_name=...)``
    which expects a config NAME relative to a search path (typically
    ``pkg://sam2``), NOT a filesystem path. Hydra explodes on absolute
    paths because the slashes confuse its config-tree resolver.

    To accept both forms, we:
      * If ``config_path`` is an absolute path that resolves (via
        symlinks) under the installed ``sam2`` package, convert it to
        its package-relative form (e.g.
        ``/.../sam2/configs/sam2.1/sam2.1_hiera_b+.yaml`` →
        ``configs/sam2.1/sam2.1_hiera_b+.yaml``).
      * Otherwise pass through as-is and let Hydra try its search
        path. Filesystem-existence is checked when the input is an
        absolute path; relative names defer to Hydra's lookup.
    """
    try:
        from sam2.build_sam import build_sam2
    except ImportError as exc:
        raise ImportError(
            "The 'sam2' package is required for "
            "R3GANDiscriminatorSAM2Pixel. Install via\n"
            "  pip install git+https://github.com/facebookresearch/sam2.git\n"
            "and download the Hiera checkpoint from\n"
            "  https://github.com/facebookresearch/sam2#download-checkpoints"
        ) from exc
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"sam2_checkpoint_path does not exist: {checkpoint_path}"
        )
    cp = Path(config_path)
    if cp.is_absolute():
        if not cp.exists():
            raise FileNotFoundError(
                f"sam2_config_path does not exist: {config_path}"
            )
        # Resolve symlinks so the path lands inside the sam2 package
        # (the v8 sbatch's PREREQUISITES instruct symlinking the
        # package's configs/sam2.1/sam2.1_hiera_b+.yaml into the
        # workspace's sam2_checkpoints/ dir).
        try:
            import sam2 as _sam2_pkg
            sam2_dir = Path(_sam2_pkg.__file__).resolve().parent
            resolved = cp.resolve()
            rel = resolved.relative_to(sam2_dir)
            config_name = str(rel)
        except (ValueError, ImportError):
            # Resolved path isn't under the sam2 package — fall back
            # to passing the absolute path. Hydra will likely fail,
            # but the error message gives the user a clear pointer.
            config_name = str(cp)
    else:
        # Relative name: trust the caller; Hydra resolves via search path.
        config_name = str(cp)
    sam2_model = build_sam2(config_name, checkpoint_path, device=str(device))
    encoder = sam2_model.image_encoder
    encoder.to(device=device, dtype=dtype)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


def _probe_feature_channels(
    encoder: nn.Module,
    image_resolution: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Run a dummy forward to discover the FPN's channel + spatial
    dims. Returns ``(channels_per_scale, spatial_per_scale)``.
    """
    encoder_was_training = encoder.training
    encoder.eval()
    with torch.no_grad():
        dummy = torch.zeros(
            1, 3, image_resolution, image_resolution,
            device=device, dtype=dtype,
        )
        out = encoder(dummy)
    if encoder_was_training:
        encoder.train()
    # SAM2's image_encoder returns a dict with keys
    # ``vision_features``, ``vision_pos_enc``, ``backbone_fpn``. The
    # FPN is what we want for multi-scale heads.
    if isinstance(out, dict) and "backbone_fpn" in out:
        feats = list(out["backbone_fpn"])
    elif isinstance(out, (list, tuple)):
        feats = list(out)
    else:
        feats = [out]
    chans = [int(f.shape[1]) for f in feats]
    spat = [(int(f.shape[2]), int(f.shape[3])) for f in feats]
    return chans, spat


# ---------------------------------------------------------------------------
# ADM 2D discriminator head (Lu et al. 2025, arXiv:2507.18569 §B.1, Fig. 6).
#
# Exactly as specified in the paper: the SDXL-Lightning-style ConvNet.
# Three downsampling 4×4-stride-2 conv blocks (each followed by GN(32) +
# SiLU), one 3×3-stride-1 refinement conv block, AdaptiveAvgPool2d, then
# a Linear(256, 1) projecting to a scalar logit.
#
# All convs project to a fixed 256-channel hidden dim so GroupNorm(32)
# always divides cleanly (no fallback ``min(32, C)`` — the paper's spec
# never needs it because every conv's output is 256).
#
# Linear head is zero-init (paper-silent; we keep the latent-disc
# convention so D starts as a constant-zero classifier).
# ---------------------------------------------------------------------------


class _ADM2DHead(nn.Module):
    """ADM 2D head (arXiv:2507.18569 §B.1, Fig. 6).

    Forward path (input ``[B*F, C_in, H, W]``):
        Conv2d 4×4 stride 2 (C_in → 256) → GN(32) → SiLU
        Conv2d 4×4 stride 2 (256 → 256)  → GN(32) → SiLU
        Conv2d 4×4 stride 2 (256 → 256)  → GN(32) → SiLU
        Conv2d 3×3 stride 1 (256 → 256)  → GN(32) → SiLU
        AdaptiveAvgPool2d(1) → Flatten → Linear(256, 1, zero-init)
    """

    HIDDEN: int = 256
    GN_GROUPS: int = 32

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        c_in = int(in_channels)
        layers: List[nn.Module] = []
        for _ in range(3):
            layers += [
                nn.Conv2d(c_in, self.HIDDEN, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(self.GN_GROUPS, self.HIDDEN),
                nn.SiLU(),
            ]
            c_in = self.HIDDEN
        layers += [
            nn.Conv2d(self.HIDDEN, self.HIDDEN, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(self.GN_GROUPS, self.HIDDEN),
            nn.SiLU(),
        ]
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(self.HIDDEN, 1)
        # Zero-init the linear head so D starts as a constant-zero
        # classifier (paper-silent; matches the latent disc convention).
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        h = self.body(feat)
        h = F.adaptive_avg_pool2d(h, 1).flatten(1)  # [B*F, 256]
        return self.head(h).squeeze(-1)             # [B*F]

    def forward_dense(self, feat: torch.Tensor) -> torch.Tensor:
        """Per-position logit map (skip GAP).

        Returns ``[B*F, H_post, W_post]`` per-spatial-position logit.
        Reuses the same ``Linear`` head's weights as a 1×1 Conv2d so
        no extra params; both the scalar and dense forward share the
        same trained classifier — they differ only in WHERE the GAP
        is applied (scalar = before head, dense = after head).
        """
        h = self.body(feat)  # [B*F, 256, H_post, W_post]
        # Linear weight [1, 256] reshapes to Conv1×1 weight [1, 256, 1, 1].
        w = self.head.weight.unsqueeze(-1).unsqueeze(-1)
        b = self.head.bias  # [1]
        out = F.conv2d(h, w, b)  # [B*F, 1, H_post, W_post]
        return out.squeeze(1)


# Backwards-compat alias so external imports of the legacy class name
# still resolve (architecturally a drop-in replacement).
_SAM2DiscHead = _ADM2DHead


# ---------------------------------------------------------------------------
# Heads + frame-pool aggregation as a standalone Module.
#
# Split off from ``R3GANDiscriminatorSAM2Pixel`` so the trainer can
# DDP-wrap ONLY the trainable params. The frozen SAM2 encoder has
# ``requires_grad=False`` everywhere, which is fundamentally
# incompatible with ``DDP(disc_full, find_unused_parameters=False)``
# (DDP errors), and only sort-of works with
# ``find_unused_parameters=True`` (the reducer relies on forward-time
# tracking through the wrapper, which is bypassed if the caller
# routes a forward through the un-wrapped disc to drive a
# heads-only path — silently breaks rank synchronization).
#
# The clean fix is to split: keep the encoder un-wrapped (no DDP
# needed; no trainable params), and DDP-wrap this Heads module
# (every param is trainable; ``find_unused_parameters=False`` works
# and forward-tracking fires correctly).
# ---------------------------------------------------------------------------


class _R3GANDiscHeads(nn.Module):
    """Trainable heads + per-scale mean + frame-pool. Operates on a
    list of FPN feature maps and returns a per-clip scalar logit.

    DDP-wrap ME, not the full disc.
    """

    def __init__(
        self,
        feat_channels: List[int],
        frame_pool: str = "mean",
        frame_pool_topk: int = 4,
    ) -> None:
        super().__init__()
        if frame_pool not in ("mean", "max", "topk_mean", "none"):
            raise ValueError(
                f"frame_pool must be 'mean' | 'max' | 'topk_mean' | "
                f"'none'; got {frame_pool!r}."
            )
        self.heads = nn.ModuleList([
            _ADM2DHead(c) for c in feat_channels
        ])
        self.frame_pool = frame_pool
        self.frame_pool_topk = max(1, int(frame_pool_topk))

    def forward(
        self,
        features: List[torch.Tensor],
        batch_size: int,
        num_frames: int,
    ) -> torch.Tensor:
        per_scale: List[torch.Tensor] = []
        for feat, head in zip(features, self.heads):
            per_scale.append(head(feat))  # [B*F]
        scale_logits = torch.stack(per_scale, dim=1)  # [B*F, n_scales]
        per_frame = scale_logits.mean(dim=1).view(batch_size, num_frames)
        if self.frame_pool == "mean":
            per_sample = per_frame.mean(dim=1)
        elif self.frame_pool == "max":
            per_sample = per_frame.gather(
                1,
                per_frame.abs().argmax(dim=1, keepdim=True),
            ).squeeze(1)
        elif self.frame_pool == "topk_mean":
            k = min(self.frame_pool_topk, num_frames)
            _, idx = per_frame.abs().topk(k, dim=1)
            per_sample = per_frame.gather(1, idx).mean(dim=1)
        else:  # "none" — per-frame logits, no temporal aggregation.
            # Returns ``[B*F]`` so consumers can compute per-frame
            # adversarial losses (forces disc to enforce sharpness on
            # every frame independently rather than averaging out
            # late-frame degradation).
            return per_frame.reshape(batch_size * num_frames).float()
        return per_sample.float()

    def forward_dense(
        self,
        features: List[torch.Tensor],
        batch_size: int,
        num_frames: int,
        target_h: Optional[int] = None,
        target_w: Optional[int] = None,
    ) -> torch.Tensor:
        """Per-frame-per-spatial-token dense logit map.

        For each FPN scale, applies the head's ``forward_dense`` to
        get ``[B*F, H_s, W_s]`` per-position logits. Bilinear-resizes
        each scale to a common grid (smallest scale by default, or
        the supplied ``(target_h, target_w)``), averages across scales,
        reshapes to ``[B, F, target_h, target_w]``.

        Used as the training target for ``gan_d_approx`` —
        the approx model predicts this dense logit map from
        ``(gen_lat, gt_lat)`` without backprop through V+SAM2.
        """
        per_scale: List[torch.Tensor] = []
        for feat, head in zip(features, self.heads):
            per_scale.append(head.forward_dense(feat))  # [B*F, H_s, W_s]
        if target_h is None or target_w is None:
            target_h = min(p.shape[1] for p in per_scale)
            target_w = min(p.shape[2] for p in per_scale)
        aligned: List[torch.Tensor] = []
        for p in per_scale:
            if p.shape[1] != target_h or p.shape[2] != target_w:
                p = F.interpolate(
                    p.unsqueeze(1).float(),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            aligned.append(p)
        avg = torch.stack(aligned, dim=0).mean(dim=0)  # [B*F, t_h, t_w]
        return avg.reshape(
            batch_size, num_frames, target_h, target_w,
        ).float()


# ---------------------------------------------------------------------------
# Top-level discriminator.
# ---------------------------------------------------------------------------


class R3GANDiscriminatorSAM2Pixel(nn.Module):
    """R3GAN discriminator with frozen SAM2 image encoder + ADM 2D heads.
    Operates on pixel video.

    Forward signature: ``forward(pixel_video) -> [B]`` scalar logits.
    Input ``pixel_video`` is ``[B, F, 3, H, W]`` in ``[-1, 1]`` (the
    WAN VAE's decode_to_pixel output range).

    Constructor args:
      sam2_checkpoint_path: path to the SAM2 .pt weights.
      sam2_config_path: path to the matching SAM2 yaml config.
      image_resolution: square input edge for the encoder
        (default 512). Smaller = faster + less memory; coarser
        feature maps. Hiera-B+ is happiest at 1024 but 512 still
        produces strong features.
      device, dtype: where + what precision to load the encoder in.
        Default fp32 for the all-fp32 path needed by R3GAN's R1/R2
        second-order penalties.
      preserve_aspect: scale by the longer edge instead of forcing
        a square input. Lets road geometry stay un-distorted.
      frame_pool: ``mean | max | topk_mean`` — how to aggregate
        per-frame logits to a per-clip scalar.
      frame_pool_topk: only consulted when ``frame_pool='topk_mean'``.
    """

    def __init__(
        self,
        sam2_checkpoint_path: str,
        sam2_config_path: str,
        *,
        image_resolution: int = 512,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        preserve_aspect: bool = True,
        pad_to_square: bool = False,
        frame_pool: str = "mean",
        frame_pool_topk: int = 4,
        encoder_chunk_size: int = 0,
    ) -> None:
        super().__init__()
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.image_resolution = int(image_resolution)
        self._encoder_dtype = dtype
        self.preserve_aspect = bool(preserve_aspect)
        # ``pad_to_square=True`` zero-pads the SHORTER edge of the
        # decoded pixel video so the input becomes square at the
        # video's longer edge (e.g. 480x832 → 832x832), then resizes
        # to the encoder's square ``image_resolution`` (e.g. 512x512).
        # This matches the probe's square shape (so SAM2 Hiera's
        # pos_embed has the right size) AND preserves the road-edge
        # geometry that ``preserve_aspect=False`` (squash) destroys.
        # When True, supersedes ``preserve_aspect``.
        self.pad_to_square = bool(pad_to_square)
        if frame_pool not in ("mean", "max", "topk_mean", "none"):
            raise ValueError(
                f"frame_pool must be 'mean' | 'max' | 'topk_mean' | "
                f"'none'; got {frame_pool!r}."
            )
        self.frame_pool = frame_pool
        self.frame_pool_topk = max(1, int(frame_pool_topk))
        # ``encoder_chunk_size``: when > 0 and the per-call batch ``B*F``
        # exceeds this value, run the SAM2 encoder forward in chunks of
        # this many frames and concat the per-FPN-scale outputs along
        # the batch axis. Cuts the SAM2 encoder peak transient memory
        # ~B*F/chunk_size× at no semantic cost (encoder is purely
        # per-frame; no cross-frame attention). Default 0 = no chunking
        # (legacy behavior).
        self.encoder_chunk_size = int(encoder_chunk_size)
        self.image_encoder = _load_sam2_image_encoder(
            sam2_checkpoint_path, sam2_config_path, device, dtype,
        )
        feat_channels, feat_spatial = _probe_feature_channels(
            self.image_encoder, self.image_resolution, device, dtype,
        )
        self._feat_channels = feat_channels
        self._feat_spatial = feat_spatial
        # ADM 2D heads + frame-pool aggregation (arXiv:2507.18569 §B.1).
        # Encapsulated in ``_R3GANDiscHeads`` so DDP can wrap ONLY the
        # trainable submodule (the frozen SAM2 encoder is left
        # un-wrapped — see the class docstring for details).
        self.heads_module = _R3GANDiscHeads(
            feat_channels=feat_channels,
            frame_pool=self.frame_pool,
            frame_pool_topk=self.frame_pool_topk,
        )
        self.heads_module.to(device=device, dtype=dtype)
        # ImageNet normalization buffers — placed at the encoder's
        # device + dtype at registration time so the first forward
        # doesn't device-mismatch (registering CPU buffers and then
        # relying on a later disc.to(device) call is fragile because
        # the trainer assigns the disc to ``self.r3gan_disc`` without
        # an explicit ``.to`` afterward).
        self.register_buffer(
            "_imgnet_mean",
            torch.tensor(
                [0.485, 0.456, 0.406], device=device, dtype=dtype,
            ).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_imgnet_std",
            torch.tensor(
                [0.229, 0.224, 0.225], device=device, dtype=dtype,
            ).view(1, 3, 1, 1),
            persistent=False,
        )
        if logging.getLogger().isEnabledFor(logging.INFO):
            head0 = next(self.heads[0].parameters())
            enc0 = next(self.image_encoder.parameters())
            logging.info(
                "[R3GANDiscriminatorSAM2Pixel] built (ADM 2D heads): "
                "resolution=%d num_scales=%d feat_channels=%s "
                "feat_spatial=%s preserve_aspect=%s pad_to_square=%s "
                "frame_pool=%s encoder_device=%s encoder_dtype=%s "
                "head_device=%s head_dtype=%s",
                self.image_resolution, len(feat_channels),
                feat_channels, feat_spatial,
                self.preserve_aspect, self.pad_to_square,
                self.frame_pool,
                enc0.device, enc0.dtype, head0.device, head0.dtype,
            )

    def train(self, mode: bool = True):
        """Override ``Module.train`` so the FROZEN SAM2 image encoder
        stays in ``eval`` mode regardless of the discriminator's mode.
        Hiera uses stochastic depth (DropPath) which is active only in
        ``train`` mode — without this override, ``disc.train()`` in
        the trainer would re-enable DropPath on the "frozen" encoder
        and produce different features each forward, polluting the
        discriminator's real-vs-fake signal.
        """
        super().train(mode)
        # Heads inherit ``mode``; encoder is pinned to eval.
        self.image_encoder.eval()
        return self

    @property
    def final_channels(self) -> int:
        # Compatibility with the latent discriminator's interface.
        return sum(self._feat_channels)

    def _encode_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Frozen SAM2 image encoder forward; returns the FPN feature
        list. Encoder params have ``requires_grad=False`` so input
        gradient flows through but no encoder weight gradients are
        allocated.

        Optionally chunked along the batched-frame axis (dim 0) when
        ``self.encoder_chunk_size > 0`` and ``x.shape[0]`` exceeds it.
        Chunking is semantically transparent (the SAM2 encoder is
        purely per-frame; no cross-frame attention) and trims the
        peak transient activation memory by the chunking factor.
        """
        def _to_fpn_list(o: object) -> List[torch.Tensor]:
            if isinstance(o, dict) and "backbone_fpn" in o:
                return list(o["backbone_fpn"])
            if isinstance(o, (list, tuple)):
                return list(o)
            return [o]  # type: ignore[list-item]

        chunk = int(getattr(self, "encoder_chunk_size", 0))
        if chunk <= 0 or x.shape[0] <= chunk:
            return _to_fpn_list(self.image_encoder(x))

        per_scale: List[List[torch.Tensor]] = []
        for s in range(0, x.shape[0], chunk):
            e = min(s + chunk, x.shape[0])
            out = _to_fpn_list(self.image_encoder(x[s:e]))
            if not per_scale:
                per_scale = [[t] for t in out]
            else:
                if len(out) != len(per_scale):
                    raise RuntimeError(
                        "SAM2 encoder returned different FPN-scale count "
                        f"({len(out)} vs {len(per_scale)}) across chunks; "
                        "chunked forward assumes a stable feature layout."
                    )
                for i, t in enumerate(out):
                    per_scale[i].append(t)
        return [torch.cat(scale_chunks, dim=0) for scale_chunks in per_scale]

    def forward_features(self, pixel_video: torch.Tensor) -> List[torch.Tensor]:
        """Run the resize → ImageNet-norm → frozen-SAM2 forward and
        return the FPN feature list. The downstream caller can either:

          * pass the features straight to ``forward_heads`` (= the
            classic ``forward(pixel)`` flow), or
          * stash them as detached tensors with
            ``requires_grad_(True)`` and run R1/R2 on the
            **feature manifold** (Option A in the distilled-critic
            design) — much cheaper than running R1/R2 through the
            full V + SAM2 + heads stack.

        Returns: ``List[Tensor]``, one feature map per FPN scale,
        shape ``[B*F, C_s, H_s, W_s]``.
        """
        if pixel_video.dim() != 5 or pixel_video.shape[2] != 3:
            raise ValueError(
                "R3GANDiscriminatorSAM2Pixel expects [B, F, 3, H, W] "
                f"in [-1, 1]; got shape {tuple(pixel_video.shape)}."
            )
        B, F_, _, H_, W_ = pixel_video.shape
        x = pixel_video.to(dtype=self._encoder_dtype)
        x = x.flatten(0, 1)  # [B*F, 3, H, W]
        x = (x + 1.0) * 0.5
        # Resize. Three modes:
        #   * pad_to_square=True (preferred) — zero-pad the SHORTER
        #     edge so the input is square at the video's longer edge
        #     (e.g. 480x832 → 832x832 with 176px black pad top+bottom),
        #     then resize the square to ``(R, R)``. Geometry preserved
        #     AND Hiera's pos_embed (sized at the square probe) gets
        #     the right shape.
        #   * preserve_aspect=True (legacy) — scale so the LONGER
        #     edge equals ``image_resolution`` and let Hiera process
        #     the resulting non-square input. Geometry preserved but
        #     Hiera's pos_embed mismatches at non-square inputs at
        #     low resolutions (Hiera ``_get_pos_embed`` shape error).
        #   * preserve_aspect=False — squash to a square ``(R, R)``.
        #     Cheapest, but distorts the aspect ratio (driving frames
        #     are typically ~1.7:1, so a forced square crushes road
        #     geometry by 40%+).
        if self.pad_to_square:
            longer = max(H_, W_)
            pad_h = longer - H_
            pad_w = longer - W_
            if pad_h > 0 or pad_w > 0:
                # F.pad expects (pad_w_left, pad_w_right, pad_h_top,
                # pad_h_bottom) for 4D tensors. Symmetric padding so
                # the (visually-meaningful) center of the frame stays
                # centered after the resize. Padded with 0.0 (= black,
                # matches the post-[-1,1]→[0,1] background convention).
                pad = (
                    pad_w // 2, pad_w - pad_w // 2,
                    pad_h // 2, pad_h - pad_h // 2,
                )
                x = F.pad(x, pad, mode="constant", value=0.0)
            target_h = self.image_resolution
            target_w = self.image_resolution
        elif self.preserve_aspect:
            longer = max(H_, W_)
            if longer != self.image_resolution:
                scale = self.image_resolution / float(longer)
                target_h = max(1, int(round(H_ * scale)))
                target_w = max(1, int(round(W_ * scale)))
            else:
                target_h, target_w = H_, W_
        else:
            target_h = self.image_resolution
            target_w = self.image_resolution
        cur_h, cur_w = x.shape[-2], x.shape[-1]
        if target_h != cur_h or target_w != cur_w:
            x = F.interpolate(
                x,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
        mean = self._imgnet_mean.to(dtype=x.dtype)
        std = self._imgnet_std.to(dtype=x.dtype)
        x = (x - mean) / std
        # Frozen encoder forward; pinned to eval() via the train()
        # override so DropPath / dropout don't fire.
        return self._encode_features(x)

    @property
    def heads(self) -> nn.ModuleList:
        """Back-compat: expose the heads ModuleList directly. Tests +
        legacy code may iterate ``disc.heads``; new code in the
        distilled-critic flow should DDP-wrap ``disc.heads_module``
        and call its ``forward(features, B, F)`` instead.
        """
        return self.heads_module.heads

    def forward_heads(
        self,
        features: List[torch.Tensor],
        batch_size: int,
        num_frames: int,
    ) -> torch.Tensor:
        """Run the trainable ADM 2D heads + per-scale mean + frame-pool
        on a list of FPN features. Delegates to ``self.heads_module``.

        ``features`` is the output of ``forward_features`` (or a
        detached + ``requires_grad_(True)`` clone for R1/R2 on the
        feature manifold). Each tensor is ``[B*F, C_s, H_s, W_s]``.

        Returns ``[B]`` scalar logits.

        NOTE for the distilled-critic D-update path: do NOT call this
        method on the un-wrapped disc; route the heads forward through
        the DDP-wrapped ``disc.heads_module`` instead so DDP's forward-
        tracking fires correctly. The legacy R3GAN path (where the
        full disc is DDP-wrapped) still uses ``forward_heads`` from
        within ``forward(pixel_video)`` and that path is fine.
        """
        return self.heads_module(features, batch_size, num_frames)

    def forward_dense_heads(
        self,
        features: List[torch.Tensor],
        batch_size: int,
        num_frames: int,
        target_h: Optional[int] = None,
        target_w: Optional[int] = None,
    ) -> torch.Tensor:
        """Per-frame-per-spatial-token DENSE logit map.

        Used as the training target for ``gan_d_approx`` (the
        PerceptualApprox-based replacement for LatentSAM2Critic).
        Delegates to ``self.heads_module.forward_dense``.

        Returns ``[B, F, target_h, target_w]`` (defaults to the
        smallest scale's grid when ``target_h/w`` are None).
        """
        return self.heads_module.forward_dense(
            features, batch_size, num_frames, target_h, target_w,
        )

    def forward(self, pixel_video: torch.Tensor) -> torch.Tensor:
        """Forward over a pixel video. Equivalent to
        ``forward_heads(forward_features(pixel_video), B, F)``.

        Args:
            pixel_video: ``[B, F, 3, H, W]`` in ``[-1, 1]`` (e.g. from
                ``self.model.vae.decode_to_pixel(latent)``).

        Returns:
            ``[B]`` scalar logits. The classic R3GAN R1/R2 penalties
            take the second-order gradient w.r.t. ``pixel_video``;
            for the distilled-critic D-update path the caller should
            instead use ``forward_features`` (no_grad) +
            ``forward_heads`` on detached features so R1/R2 run on
            the **feature manifold** without holding V+SAM2 graph.
        """
        if pixel_video.dim() != 5 or pixel_video.shape[2] != 3:
            raise ValueError(
                "R3GANDiscriminatorSAM2Pixel expects [B, F, 3, H, W] "
                f"in [-1, 1]; got shape {tuple(pixel_video.shape)}."
            )
        B, F_, _, _, _ = pixel_video.shape
        features = self.forward_features(pixel_video)
        return self.forward_heads(features, batch_size=B, num_frames=F_)
