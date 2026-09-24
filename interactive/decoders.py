"""Latent->pixel decoder zoo for the interactive world model (WS-F).

One interface, four backends, all obeying the SAME streaming contract the
engine already uses for the Wan VAE:

    dec = build_decoder("lightvaew2_1", device="cuda", dtype=torch.bfloat16,
                        wan_model_path="/home/ashish/Wan2.1/")
    dec.reset()                                   # once per ride/session
    for chunk in chunks:                          # chunk = [1, 3, 16, 60, 104]
        frames = dec.decode_chunk(chunk)          # uint8 [T, 480, 832, 3] on GPU

``reset()`` clears all temporal state.  ``decode_chunk`` is causal: it sees
only the chunk it is handed plus the state left by previous chunks, never
future context.  Frame counts follow the Wan VAE exactly — the FIRST chunk
after a reset yields 9 pixel frames (the VAE's "special first" latent frame
decodes to 1 pixel frame, the other two to 4 each), every later chunk yields
12.  Callers that need a fixed 12 must pad the first chunk themselves; the
engine already handles the ragged first chunk.

Backends
--------
wan            utils.wan_wrapper.WanVAEWrapper — the training/eval reference.
lightvaew2_1   LightX2V's pruned (dim=96, pruning_rate=0.75) causal-Conv3D
               Wan VAE.  Same cached_decode/clear_cache streaming contract.
taew2_1        madebyollin's TAEHV for Wan 2.1 (MIT).  Conv2D + MemBlock
               temporal memory; streamed chunk-at-a-time exactly (see
               _TaehvChunkStream).
lighttaew2_1   LightX2V's TAE for Wan 2.1 — byte-compatible with the TAEHV
               base architecture, different training run.

Latent scaling (this is the easy thing to get wrong)
----------------------------------------------------
The repo's latents (zarr, generator output) live in the NORMALIZED Wan latent
space, i.e. ``(raw_vae_latent - mean) / std``.  ``WanVAEWrapper.decode_to_pixel``
undoes that internally by passing ``scale=[mean, 1/std]`` down to the VAE,
which computes ``z / scale[1] + scale[0]``.

  * wan / lightvaew2_1 : both take ``scale=[mean, 1/std]`` and de-normalize
    internally.  Feed them our latents verbatim.
  * taew2_1 : TAEHV's README is explicit — "TAEHV does not use any latent
    scales / shifts (TAEHV encodes / decodes exactly what diffusion models
    use)".  Feed it our latents verbatim; de-normalizing first costs ~14 dB.
  * lighttaew2_1 : the OPPOSITE, despite sharing the architecture.  LightX2V
    trained their TAE in the RAW Wan VAE latent space, so it needs
    ``z * std + mean`` first — this is their ``WanVAE_tiny(need_scaled=True)``
    branch.  Measured on seed-zarr latents against the Wan reference:
    verbatim 16.4 dB vs de-normalized 31.8 dB.  Getting this backwards is
    exactly the washed-out/checkerboard failure it looks like.

Nothing here needs the DiT; ``interactive/decoder_bench.py`` exercises all of
it from zarr latents alone.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

log = logging.getLogger("decoders")

DECODER_WEIGHTS_DIR = Path(__file__).resolve().parent / "decoder_weights"

DECODER_NAMES = ("wan", "lightvaew2_1", "taew2_1", "lighttaew2_1")

# Wan 2.1 latent normalisation (identical in utils/wan_wrapper.py, LightX2V and
# diffusers).  Kept here so the alternative decoders never have to import the
# Wan wrapper just to read two constants.
WAN_LATENT_MEAN = (
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
)
WAN_LATENT_STD = (
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
)


# ---------------------------------------------------------------------------
# interface
# ---------------------------------------------------------------------------
class LatentDecoder:
    """Streaming latent->pixel decoder.

    Subclasses implement ``reset`` and ``_decode`` (which returns pixels as
    a float tensor ``[T, 3, H, W]`` in [0, 1] on the compute device).
    """

    name: str = "?"
    #: how many pixel frames the first post-reset chunk drops relative to 4*T
    first_chunk_trim: int = 3

    def __init__(self, device, dtype: torch.dtype):
        self.device = torch.device(device)
        self.dtype = dtype

    # -- public API ---------------------------------------------------------
    def reset(self) -> None:
        """Clear all temporal state.  Call once per ride, before chunk 0."""
        raise NotImplementedError

    @torch.no_grad()
    def decode_chunk(self, latents: torch.Tensor, *,
                     half_res: bool = False) -> torch.Tensor:
        """Decode one chunk of latents.

        Args:
            latents: ``[1, F_lat, 16, 60, 104]`` (F_lat is normally 3),
                normalized Wan latent space, any float dtype.
            half_res: area-downsample the output 2x (engine's decode_half_res).

        Returns:
            uint8 RGB ``[F_pix, H, W, 3]`` on ``self.device``.  F_pix is
            ``4*F_lat - 3`` for the first chunk after ``reset()`` and
            ``4*F_lat`` afterwards.
        """
        if latents.dim() != 5:
            raise ValueError(f"expected [B, F, C, H, W] latents, got {tuple(latents.shape)}")
        if latents.shape[0] != 1:
            raise ValueError("streaming decode requires batch size 1")
        vid = self._decode(latents.to(device=self.device, dtype=self.dtype))
        if half_res:
            vid = torch.nn.functional.interpolate(vid, scale_factor=0.5, mode="area")
        return (vid.permute(0, 2, 3, 1) * 255).round_().clamp_(0, 255).to(torch.uint8)

    # -- to implement -------------------------------------------------------
    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        """[1, F, C, H, W] -> float [T, 3, H, W] in [0, 1]."""
        raise NotImplementedError

    # -- misc ---------------------------------------------------------------
    def parameter_count(self) -> int:
        return 0

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} name={self.name} dtype={self.dtype}>"


# ---------------------------------------------------------------------------
# 1. Wan reference
# ---------------------------------------------------------------------------
class WanDecoder(LatentDecoder):
    """The training/eval reference: utils.wan_wrapper.WanVAEWrapper.

    Uses the repo's approved streaming contract verbatim — one
    ``model.clear_cache()`` per reset, then ``decode_to_pixel(..., use_cache=True)``
    per chunk.  Never loops plain ``decode_to_pixel``.
    """

    name = "wan"

    def __init__(self, device, dtype, wan_model_path: str,
                 model_name: str = "Wan2.1-T2V-1.3B"):
        super().__init__(device, dtype)
        from utils.wan_wrapper import WanVAEWrapper
        root = Path(str(wan_model_path).rstrip("/"))
        if root.name.startswith("Wan2.1-T2V"):
            model_name, root = root.name, root.parent
        self.vae = WanVAEWrapper(model_name, model_root=root / model_name)
        self.vae = self.vae.to(device=self.device, dtype=self.dtype).eval()
        self.vae.requires_grad_(False)

    def reset(self) -> None:
        self.vae.model.clear_cache()

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        px = self.vae.decode_to_pixel(latents, use_cache=True)   # [1, T, 3, H, W] in [-1, 1]
        return (0.5 * (px[0].float() + 1.0)).clamp_(0, 1)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.vae.model.decoder.parameters())


# ---------------------------------------------------------------------------
# 2. LightX2V pruned causal-Conv3D Wan VAE
# ---------------------------------------------------------------------------
class LightVaeDecoder(LatentDecoder):
    """LightX2V ``lightvaew2_1``: Wan's causal Conv3D VAE, pruned to 25 % width.

    Architecturally the closest thing to the Wan VAE here, and it exposes the
    identical ``clear_cache()`` + ``cached_decode(z, scale)`` streaming
    interface, so the temporal state is exact (no overlap window, no
    look-ahead).  Note the architecture is NOT the repo's ``wan/modules/vae.py``
    — LightX2V's Decoder3d halves ``in_dim`` at three upsample stages, so the
    checkpoint only loads into the vendored LightX2V module.
    """

    name = "lightvaew2_1"

    def __init__(self, device, dtype, weights: Optional[str] = None):
        super().__init__(device, dtype)
        from interactive.vendor.lightx2v_wan_vae import WanVAE_

        path = Path(weights) if weights else DECODER_WEIGHTS_DIR / "lightvaew2_1.pth"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing — fetch it with\n"
                "  curl -L -o interactive/decoder_weights/lightvaew2_1.pth \\\n"
                "    https://huggingface.co/lightx2v/Autoencoders/resolve/main/lightvaew2_1.pth")
        sd = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        # Config from LightX2V's `_video_vae` with use_lightvae=True: dim 96,
        # 4x pruning, and (unlike upstream Wan) temperal_downsample
        # [False, True, True].  Verified by a strict load of the checkpoint.
        self.model = WanVAE_(
            dim=96, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2,
            attn_scales=[], temperal_downsample=[False, True, True],
            dropout=0.0, pruning_rate=0.75,
        )
        self.model.load_state_dict(sd, strict=True)
        # The encoder is dead weight for a decode-only engine.
        self.model.encoder = None
        self.model = self.model.to(device=self.device, dtype=self.dtype).eval()
        self.model.requires_grad_(False)
        self._scale = [
            torch.tensor(WAN_LATENT_MEAN, device=self.device, dtype=self.dtype),
            1.0 / torch.tensor(WAN_LATENT_STD, device=self.device, dtype=self.dtype),
        ]

    def reset(self) -> None:
        # clear_cache() would also touch the (deleted) encoder cache.
        self.model.clear_decode_cache()

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        z = latents.permute(0, 2, 1, 3, 4)                       # [1, C, F, H, W]
        out = self.model.cached_decode(z, self._scale)           # [1, 3, T, H, W] in [-1, 1]
        out = out.float().clamp_(-1, 1)[0].permute(1, 0, 2, 3)   # [T, 3, H, W]
        return out.mul_(0.5).add_(0.5)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.decoder.parameters())


# ---------------------------------------------------------------------------
# 3/4. TAEHV family (taew2_1, lighttaew2_1)
# ---------------------------------------------------------------------------
class _TaehvChunkStream:
    """Chunk-granular streaming decode for TAEHV, exact and GPU-parallel.

    ``StreamingTAEHV`` in the vendored taehv.py is exact but strictly
    frame-at-a-time: it pushes ONE frame through the whole 23-block decoder per
    call, which on a 5090 at 480x832 is dominated by kernel launch overhead.
    ``TAEHV.decode_video(parallel=True)`` is the fast path but zero-pads each
    MemBlock's temporal memory at t=0, which is only right for a clip that
    starts at t=0 — used per chunk it re-starts the memory every chunk and the
    seams show.

    This class is the fast path made causal: identical to
    ``apply_model_with_memblocks_parallel`` except that each MemBlock's memory
    at the chunk's first frame is the previous chunk's last input *at that
    block* (carried in ``self._carry``) instead of zeros.  That is exactly what
    the sequential/streaming path feeds, so the output is bit-comparable with
    ``StreamingTAEHV`` while keeping all of a chunk's frames in one batch.

    Startup trim: TAEHV's temporal upscale is 4 and ``frames_to_trim`` is 3, so
    the first 3 raw decoder frames are dropped globally after a reset — which
    makes the first chunk 9 frames and the rest 12, matching the Wan VAE.
    """

    def __init__(self, taehv):
        from interactive.vendor.taehv import MemBlock, SuperMemBlock
        self._memtypes = (MemBlock, SuperMemBlock)
        self.taehv = taehv
        self.reset()

    def reset(self) -> None:
        self._carry = [None] * len(self.taehv.decoder)
        self._n_raw = 0

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, T, C, H, W] latents -> [N, T', 3, H*8, W*8] in [0, 1]."""
        n = x.shape[0]
        h = x.flatten(0, 1)
        for i, b in enumerate(self.taehv.decoder):
            if isinstance(b, self._memtypes):
                nt, c, hh, ww = h.shape
                t = nt // n
                _h = h.view(n, t, c, hh, ww)
                prev = self._carry[i]
                if prev is None:
                    prev = _h.new_zeros((n, 1, c, hh, ww))
                mem = torch.cat([prev, _h[:, : t - 1]], dim=1).reshape(h.shape)
                self._carry[i] = _h[:, -1:].clone()
                h = b(h, mem)
            else:
                h = b(h)
        nt, c, hh, ww = h.shape
        out = self.taehv.postprocess_output_frames(h.view(n, nt // n, c, hh, ww))
        trim = self.taehv.frames_to_trim
        start = max(0, min(out.shape[1], trim - self._n_raw))
        self._n_raw += out.shape[1]
        return out[:, start:]


class TaehvDecoder(LatentDecoder):
    """TAEHV-family tiny decoder (``taew2_1`` or LightX2V's ``lighttaew2_1``).

    Both checkpoints share the TAEHV base architecture (23-block Conv2D
    decoder, 9.8 M params, 4x temporal / 8x spatial), so one class serves both.

    They do NOT share a latent convention, which is the one trap here:

      * ``taew2_1`` is trained on the diffusion model's own (normalized) latent
        space and applies no mean/std — feed it our latents verbatim.
      * ``lighttaew2_1`` is trained on the RAW Wan VAE latent space, so it needs
        ``z * std + mean`` applied first (LightX2V's ``need_scaled=True``).

    Measured against the Wan reference on seed-zarr latents, using the wrong
    one costs ~14 dB and shows up as a colour cast plus checkerboard texture.
    Output is [0, 1] RGB in both cases.
    """

    #: checkpoints trained in the RAW (un-normalized) Wan latent space
    _DENORMALIZE = {"lighttaew2_1"}

    def __init__(self, name: str, device, dtype, weights: Optional[str] = None):
        super().__init__(device, dtype)
        from interactive.vendor.taehv import TAEHV

        self.name = name
        self.denormalize = name in self._DENORMALIZE
        path = Path(weights) if weights else DECODER_WEIGHTS_DIR / f"{name}.pth"
        if not path.exists():
            url = ("https://raw.githubusercontent.com/madebyollin/taehv/main/taew2_1.pth"
                   if name == "taew2_1" else
                   f"https://huggingface.co/lightx2v/Autoencoders/resolve/main/{name}.pth")
            raise FileNotFoundError(
                f"{path} missing — fetch it with\n"
                f"  curl -L -o interactive/decoder_weights/{name}.pth {url}")
        # TAEHV sniffs variant flags out of the filename; both of ours are the
        # plain base variant (patch_size 1, 16 latent channels), and the
        # "lighttae..." prefix trips none of the sniffs.
        taehv = TAEHV(checkpoint_path=str(path))
        taehv.encoder = None                     # decode-only
        taehv = taehv.to(device=self.device, dtype=self.dtype).eval()
        taehv.requires_grad_(False)
        self.taehv = taehv
        self.stream = _TaehvChunkStream(taehv)
        if self.denormalize:
            self._mean = torch.tensor(
                WAN_LATENT_MEAN, device=self.device, dtype=self.dtype).view(1, 1, 16, 1, 1)
            self._std = torch.tensor(
                WAN_LATENT_STD, device=self.device, dtype=self.dtype).view(1, 1, 16, 1, 1)

    def reset(self) -> None:
        self.stream.reset()

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        if self.denormalize:
            latents = latents * self._std + self._mean
        out = self.stream.decode(latents)        # [1, T, 3, H, W] in [0, 1]
        return out[0].float()

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.taehv.decoder.parameters())


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------
def build_decoder(name: str,
                  device="cuda",
                  dtype: torch.dtype = torch.bfloat16,
                  wan_model_path: str = "/home/ashish/Wan2.1/",
                  weights: Optional[str] = None) -> LatentDecoder:
    """Build one of ``DECODER_NAMES``.

    Args:
        name: wan | lightvaew2_1 | taew2_1 | lighttaew2_1.
        device / dtype: compute device and weight dtype (bf16 recommended).
        wan_model_path: parent dir holding ``Wan2.1-T2V-1.3B/`` (only used by
            ``name="wan"``; the alternatives read
            ``interactive/decoder_weights/<name>.pth``).
        weights: override the alternative decoders' checkpoint path.
    """
    name = str(name).lower()
    if name == "wan":
        return WanDecoder(device, dtype, wan_model_path)
    if name == "lightvaew2_1":
        return LightVaeDecoder(device, dtype, weights)
    if name in ("taew2_1", "lighttaew2_1"):
        return TaehvDecoder(name, device, dtype, weights)
    raise ValueError(f"unknown decoder {name!r}; expected one of {DECODER_NAMES}")
