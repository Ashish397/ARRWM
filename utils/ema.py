"""DDP-friendly EMA wrapper for the generator's trainable params.

Mirrors the surface of ``Causal-Forcing/utils/distributed.EMA_FSDP`` but
without the ``FSDP.summon_full_params`` dance — we use plain DDP, so each
rank already holds the full parameter shard locally and we can read /
write directly from ``module.named_parameters()``.

API:
    ema = GeneratorEMA(module, decay=0.99, include_buffers=False)
    ema.update(module)            # called after optimizer.step()
    sd  = ema.state_dict()        # picklable {name: cpu_fp32_tensor}
    ema.load_state_dict(sd)       # restore
    ema.copy_to(module)           # write EMA weights into the live module
                                  # (used at eval / checkpoint export time)

Storage
-------
The shadow is kept in **fp32 on CPU** — same as Causal-Forcing's
``EMA_FSDP``. CPU storage keeps GPU memory free for the actual training
forward/backward; the only GPU cost is a single .float() copy per
``update`` call, which is negligible compared to the per-step DiT work.

Trainable-only filter
---------------------
EMA tracks only ``requires_grad=True`` parameters. Frozen weights (e.g.
the merged v14-LoRA portion of the DiT, action heads gated off via
``train_action_projection=false``, etc.) are NOT shadowed — there's no
gradient signal moving them so EMA(p) == p trivially.

DDP module unwrap
-----------------
If the caller passes a ``DistributedDataParallel`` instance, we unwrap
to ``module.module`` automatically; this keeps the shadow keyed by the
inner DiT's parameter names, which makes checkpoints portable across
DDP / no-DDP / FSDP-summoned environments.
"""
from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn

try:
    from torch.nn.parallel import DistributedDataParallel as _DDP
except Exception:  # pragma: no cover
    _DDP = None  # type: ignore


def _unwrap(module: nn.Module) -> nn.Module:
    if _DDP is not None and isinstance(module, _DDP):
        return module.module
    return module


class GeneratorEMA:
    """Exponential moving average of a module's trainable parameters."""

    def __init__(
        self,
        module: nn.Module,
        *,
        decay: float = 0.99,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        trainable_only: bool = True,
    ) -> None:
        if not (0.0 <= decay <= 1.0):
            raise ValueError(f"EMA decay must be in [0, 1]; got {decay}")
        self.decay = float(decay)
        self.device = torch.device(device)
        self.dtype = dtype
        self.trainable_only = bool(trainable_only)
        self.shadow: Dict[str, torch.Tensor] = {}
        self._init_shadow(module)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _named_params(self, module: nn.Module) -> Iterable:
        inner = _unwrap(module)
        for n, p in inner.named_parameters():
            if self.trainable_only and not p.requires_grad:
                continue
            yield n, p

    @torch.no_grad()
    def _init_shadow(self, module: nn.Module) -> None:
        for n, p in self._named_params(module):
            self.shadow[n] = p.detach().to(device=self.device, dtype=self.dtype).clone()

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        d = self.decay
        seen = 0
        skipped = 0
        for n, p in self._named_params(module):
            if n not in self.shadow:
                self.shadow[n] = p.detach().to(
                    device=self.device, dtype=self.dtype
                ).clone()
                continue
            seen += 1
            shadow = self.shadow[n]
            live = p.detach().to(device=self.device, dtype=self.dtype)
            if shadow.shape != live.shape:
                self.shadow[n] = live.clone()
                skipped += 1
                continue
            shadow.mul_(d).add_(live, alpha=1.0 - d)
        if skipped > 0:
            logging.getLogger(__name__).warning(
                "EMA: replaced %d shadow entries due to shape drift", skipped,
            )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(self, sd: Dict[str, torch.Tensor], *, strict: bool = False) -> None:
        if not isinstance(sd, dict):
            raise TypeError(
                f"GeneratorEMA.load_state_dict expects a dict; got {type(sd)}"
            )
        loaded = 0
        missing_in_ckpt = []
        for n in list(self.shadow.keys()):
            if n in sd:
                loaded += 1
                self.shadow[n] = sd[n].to(
                    device=self.device, dtype=self.dtype
                ).clone()
            else:
                missing_in_ckpt.append(n)
        if strict and missing_in_ckpt:
            raise RuntimeError(
                f"GeneratorEMA.load_state_dict missing {len(missing_in_ckpt)} "
                f"shadow entries in checkpoint: {missing_in_ckpt[:5]}..."
            )
        logging.getLogger(__name__).info(
            "EMA: loaded %d/%d shadow entries (missing %d).",
            loaded, len(self.shadow), len(missing_in_ckpt),
        )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    @torch.no_grad()
    def copy_to(self, module: nn.Module, *, strict: bool = False) -> None:
        """Write the EMA shadow into the live module's parameters.

        Useful right before checkpoint export or eval. Note: this is
        destructive — the live module's training state is overwritten.
        Snapshot first if you intend to continue training afterwards.
        """
        inner = _unwrap(module)
        copied = 0
        for n, p in inner.named_parameters():
            if self.trainable_only and not p.requires_grad:
                continue
            if n not in self.shadow:
                if strict:
                    raise RuntimeError(
                        f"GeneratorEMA.copy_to: parameter {n!r} not in shadow."
                    )
                continue
            shadow = self.shadow[n]
            if shadow.shape != p.shape:
                if strict:
                    raise RuntimeError(
                        f"GeneratorEMA.copy_to: shape drift on {n!r} "
                        f"(shadow={tuple(shadow.shape)} live={tuple(p.shape)})"
                    )
                continue
            p.data.copy_(shadow.to(device=p.device, dtype=p.dtype))
            copied += 1
        logging.getLogger(__name__).info("EMA: copied %d parameters into live module.", copied)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def num_params(self) -> int:
        return len(self.shadow)


__all__ = ["GeneratorEMA"]
