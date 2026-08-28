"""Graph-free, decoder-shaped approximation to WAN's pixel-to-latent VJP.

The expensive discriminator supplies a detached pixel cotangent.  A detached
WAN forward supplies the actual multiscale decoder state.  The reverse path
then composes exactly-cotangent-linear local operators, retaining the frozen
RGB head, resamplers, public-latent prefix, and selected difficult residual
blocks exactly.  No graph connects the decode to the generator latent; the
returned field is consumed through a linear synthetic-gradient loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn

from model.tied_residual_vjp import StructuredResidualVJP

__all__ = [
    "DecoderForwardState", "DecoderShapedPullback",
    "generator_decoder_pullback_loss", "public_pixel_output_vjp",
]


RESIDUAL_STAGE_TO_UPSAMPLE = {
    1: 0, 2: 1, 3: 2,
    5: 4, 6: 5, 7: 6,
    9: 8, 10: 9, 11: 10,
    13: 12, 14: 13, 15: 14,
}


def _combine_batched(
    calls: Sequence[torch.Tensor], *, batch: int, calls_per_sample: int,
) -> torch.Tensor:
    if len(calls) != int(batch) * int(calls_per_sample):
        raise RuntimeError(
            f"expected {batch}x{calls_per_sample} decoder calls, got {len(calls)}"
        )
    samples = []
    for sample in range(int(batch)):
        start = sample * int(calls_per_sample)
        samples.append(torch.cat(
            list(calls[start:start + int(calls_per_sample)]), dim=2,
        ))
    return torch.cat(samples, dim=0)


def _special_phase_pack(
    cotangent: torch.Tensor, *, input_frames: int, input_height: int,
    input_width: int,
) -> torch.Tensor:
    batch, channels, frames, height, width = cotangent.shape
    if height == input_height and width == input_width:
        spatial = cotangent
    elif height == 2 * input_height and width == 2 * input_width:
        spatial = cotangent.reshape(
            batch, channels, frames, input_height, 2, input_width, 2,
        ).permute(0, 1, 4, 6, 2, 3, 5).reshape(
            batch, channels * 4, frames, input_height, input_width,
        )
    else:
        raise ValueError(
            f"unsupported spatial phase {height}x{width} -> "
            f"{input_height}x{input_width}"
        )
    if frames == input_frames:
        return spatial.contiguous()
    if frames != 2 * input_frames - 1:
        raise ValueError(f"unsupported temporal phase {frames} -> {input_frames}")
    result = spatial.new_zeros(
        batch, spatial.shape[1] * 2, input_frames, input_height, input_width,
    )
    result[:, :spatial.shape[1], 0] = spatial[:, :, 0]
    if input_frames > 1:
        result[:, :, 1:] = spatial[:, :, 1:].reshape(
            batch, spatial.shape[1], input_frames - 1, 2,
            input_height, input_width,
        ).permute(0, 1, 3, 2, 4, 5).reshape(
            batch, spatial.shape[1] * 2, input_frames - 1,
            input_height, input_width,
        )
    return result.contiguous()


def public_pixel_output_vjp(
    decoder_output: torch.Tensor, pixel_cotangent: torch.Tensor,
) -> torch.Tensor:
    """Exact VJP of WAN wrapper's clamp and self-seed dummy-frame trim."""
    if decoder_output.dim() != 5 or pixel_cotangent.dim() != 5:
        raise ValueError("decoder output and pixel cotangent must be rank five")
    incoming = pixel_cotangent.detach().float().permute(0, 2, 1, 3, 4)
    if (
        decoder_output.shape[0] != incoming.shape[0]
        or decoder_output.shape[1] != incoming.shape[1]
        or decoder_output.shape[2] != incoming.shape[2] + 1
        or decoder_output.shape[-2:] != incoming.shape[-2:]
    ):
        raise ValueError(
            f"public pixel geometry {tuple(pixel_cotangent.shape)} is not the "
            f"dummy-trimmed form of {tuple(decoder_output.shape)}"
        )
    result = torch.zeros_like(decoder_output, dtype=torch.float32)
    result[:, :, 1:] = incoming
    # The hook's tensor has already been clamped in-place by WanVAEWrapper.
    # Values strictly inside the interval are precisely the unsaturated set;
    # equality is measure-zero and choosing zero is the safe clamp subgradient.
    return result * (decoder_output.detach().float().abs() < 1.0)


@dataclass(frozen=True)
class DecoderForwardState:
    latent: torch.Tensor
    pixels: torch.Tensor
    activations: Tuple[torch.Tensor, ...]


class DecoderShapedPullback(nn.Module):
    """Compose frozen/local WAN backward operators without a decoder graph."""

    _boundary_module_indices = (
        ("middle", 2),
        ("upsamples", 0), ("upsamples", 1), ("upsamples", 2),
        ("upsamples", 3), ("upsamples", 4), ("upsamples", 5),
        ("upsamples", 6), ("upsamples", 7), ("upsamples", 8),
        ("upsamples", 9), ("upsamples", 10), ("upsamples", 11),
        ("upsamples", 12), ("upsamples", 13), ("upsamples", 14),
        ("head", 2),
    )

    def __init__(
        self, vae: nn.Module, local_models: Sequence[nn.Module], *,
        exact_residual_stages: Sequence[int] = (6, 7),
        exact_fixed_stages: bool = True,
    ) -> None:
        super().__init__()
        if len(local_models) != 17:
            raise ValueError(f"expected 17 local operators, got {len(local_models)}")
        self.vae = vae.eval().requires_grad_(False)
        self.local_models = nn.ModuleList(local_models).eval().requires_grad_(False)
        self.exact_fixed_stages = bool(exact_fixed_stages)
        decoder = self.vae.model.decoder
        stages = sorted(set(int(value) for value in exact_residual_stages))
        invalid = set(stages) - set(RESIDUAL_STAGE_TO_UPSAMPLE)
        if invalid:
            raise ValueError(f"non-residual exact stages {sorted(invalid)}")
        self.exact_residual_stages = tuple(stages)
        self.exact_residual_ops = nn.ModuleDict({
            str(stage): StructuredResidualVJP(
                decoder.upsamples[RESIDUAL_STAGE_TO_UPSAMPLE[stage]],
            )
            for stage in stages
        }).eval().requires_grad_(False)

    @classmethod
    def from_bundle(
        cls, vae: nn.Module, bundle_path: str, *,
        map_location: str | torch.device = "cpu",
    ) -> "DecoderShapedPullback":
        """Load one immutable, audit-selected local-operator bundle."""
        bundle = torch.load(
            bundle_path, map_location=map_location, weights_only=False,
        )
        if bundle.get("kind") != "wan_decoder_shaped_pullback_bundle":
            raise RuntimeError(f"invalid decoder pullback bundle {bundle_path}")
        records = bundle.get("models", [])
        if len(records) != 17:
            raise RuntimeError(f"bundle has {len(records)} local operators, expected 17")
        from model.local_vjp_surrogate import (
            IdentityAnchoredLocalVJP, LocalLinearVJPSurrogate,
        )
        models = []
        for stage_index, record in enumerate(records):
            if int(record["stage_index"]) != stage_index:
                raise RuntimeError("bundle local operators are out of order")
            if bool(record["identity_anchor"]):
                model = IdentityAnchoredLocalVJP(
                    state_channels=int(record["state_channels"]),
                    width=int(record["width"]), blocks=int(record["blocks"]),
                )
            else:
                model = LocalLinearVJPSurrogate(
                    state_channels=int(record["state_channels"]),
                    packed_cotangent_channels=int(record["packed_channels"]),
                    output_channels=int(record["output_channels"]),
                    width=int(record["width"]), blocks=int(record["blocks"]),
                )
            model.load_state_dict(record["state_dict"], strict=True)
            models.append(model)
        return cls(
            vae, models,
            exact_residual_stages=tuple(bundle["exact_residual_stages"]),
            exact_fixed_stages=bool(bundle.get("exact_fixed_stages", True)),
        )

    def capture(self, latent: torch.Tensor) -> DecoderForwardState:
        """Run one detached WAN forward and preserve every reverse boundary."""
        if latent.dim() != 5:
            raise ValueError(f"latent must be [B,F,C,H,W], got {tuple(latent.shape)}")
        decoder = self.vae.model.decoder
        boundaries = [[] for _ in range(18)]
        handles = []

        def prehook(_module, inputs):
            boundaries[0].append(inputs[0])

        handles.append(decoder.conv1.register_forward_pre_hook(prehook))
        for boundary_index, (owner, index) in enumerate(
            self._boundary_module_indices, start=1,
        ):
            module = getattr(decoder, owner)[index]

            def hook(_module, _inputs, output, *, target=boundary_index):
                boundaries[target].append(output)

            handles.append(module.register_forward_hook(hook))
        source = latent.detach().float()
        try:
            with torch.no_grad():
                pixels = self.vae.decode_to_pixel(source, seed_first=True).float()
        finally:
            for handle in handles:
                handle.remove()
        calls_per_sample = int(source.shape[1]) + 1
        activations = tuple(
            _combine_batched(
                calls, batch=int(source.shape[0]),
                calls_per_sample=calls_per_sample,
            ).detach().float()
            for calls in boundaries
        )
        return DecoderForwardState(
            latent=source, pixels=pixels.detach(), activations=activations,
        )

    def _fixed_stage_vjp(
        self, stage_index: int, state: torch.Tensor, incoming: torch.Tensor,
    ) -> torch.Tensor:
        decoder = self.vae.model.decoder
        with torch.enable_grad():
            x = state.detach().requires_grad_(True)
            if stage_index == 4:
                module = decoder.upsamples[3]
                chunks = (1,) * int(x.shape[2])
            elif stage_index == 8:
                module = decoder.upsamples[7]
                if int(x.shape[2]) % 2 != 1:
                    raise RuntimeError(
                        "WAN second temporal resampler input must be odd"
                    )
                chunks = (1,) + (2,) * ((int(x.shape[2]) - 1) // 2)
            elif stage_index == 12:
                module, chunks = decoder.upsamples[11], None
            elif stage_index == 16:
                y = x
                for module in decoder.head:
                    y = module(y)
                return torch.autograd.grad(y, x, grad_outputs=incoming)[0].detach()
            else:
                raise ValueError(stage_index)
            if chunks is None:
                y = module(x)
            else:
                outputs, cursor, cache = [], 0, [None]
                for length in chunks:
                    feature_index = [0]
                    outputs.append(module(
                        x[:, :, cursor:cursor + length],
                        feat_cache=cache, feat_idx=feature_index,
                    ))
                    cursor += length
                if cursor != int(x.shape[2]):
                    raise RuntimeError("fixed resampler chunk geometry changed")
                y = torch.cat(outputs, dim=2)
            return torch.autograd.grad(y, x, grad_outputs=incoming)[0].detach()

    def _prefix_vjp(
        self, latent: torch.Tensor, boundary_cotangent: torch.Tensor,
    ) -> torch.Tensor:
        with torch.enable_grad():
            source = latent.detach().float().requires_grad_(True)
            seeded = torch.cat([source[:, 0:1], source], dim=1)
            channel_first = seeded.permute(0, 2, 1, 3, 4)
            mean = self.vae.mean.to(source).view(1, -1, 1, 1, 1)
            inverse_std = (1.0 / self.vae.std.to(source)).view(
                1, -1, 1, 1, 1,
            )
            boundary = self.vae.model.conv2(channel_first / inverse_std + mean)
            return torch.autograd.grad(
                boundary, source, grad_outputs=boundary_cotangent.to(boundary),
            )[0].detach()

    def pullback(
        self, captured: DecoderForwardState, pixel_cotangent: torch.Tensor,
    ) -> torch.Tensor:
        """Approximate ``J_decode(z)^T v`` from detached state and ``v``."""
        activations = captured.activations
        value = public_pixel_output_vjp(activations[-1], pixel_cotangent)
        for stage_index in reversed(range(17)):
            state = activations[stage_index]
            key = str(stage_index)
            if key in self.exact_residual_ops:
                value = self.exact_residual_ops[key](state, value)
            elif self.exact_fixed_stages and stage_index in (4, 8, 12, 16):
                value = self._fixed_stage_vjp(stage_index, state, value)
            else:
                packed = _special_phase_pack(
                    value, input_frames=int(state.shape[2]),
                    input_height=int(state.shape[3]),
                    input_width=int(state.shape[4]),
                )
                with torch.no_grad():
                    value = self.local_models[stage_index](state, packed)
        return self._prefix_vjp(captured.latent, value).float().detach()

    def forward(
        self, latent: torch.Tensor, pixel_cotangent: torch.Tensor,
    ) -> torch.Tensor:
        return self.pullback(self.capture(latent), pixel_cotangent)


def generator_decoder_pullback_loss(
    latent: torch.Tensor, field: torch.Tensor, *, weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Linear loss whose derivative with respect to ``latent`` is ``-field``."""
    if latent.shape != field.shape:
        raise ValueError(f"latent {tuple(latent.shape)} != field {tuple(field.shape)}")
    detached = field.detach().to(device=latent.device, dtype=torch.float32)
    main = -(latent.float() * detached).flatten(1).sum(dim=1).mean()
    return float(weight) * main, {
        "train/surrogate_g_main": float(main.detach()),
        "train/surrogate_g_weighted": float(weight) * float(main.detach()),
        "train/surrogate_g_weight": float(weight),
        "train/surrogate_g_field_rms": float(detached.square().mean().sqrt()),
        "train/surrogate_decoder_shaped": 1.0,
    }
