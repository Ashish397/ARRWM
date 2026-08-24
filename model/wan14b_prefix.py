"""Prefix-only loader for the Wan2.1-T2V-14B DiT (WP-14B / section B2).

Why this exists
---------------
The LADD discriminator projects onto FROZEN teacher features
(``model/ladd_disc.py::WanFeatureProjector``). Until now those features
came from ``real_score`` — the v14e-distilled 1.3B teacher the student is
itself being distilled from. ``docs/GAN_REDESIGN.md`` Point 6 argues that
tap is redundant with the DMD term: the critic and the DMD teacher read
the same representation, so the GAN gradient carries little information
the distillation loss does not already carry. The replacement is an
INDEPENDENT frozen backbone — stock Wan 2.1-T2V-14B — whose features the
student has never been trained against.

The blocker was cost: the full 14B DiT is ~28 GB in bf16, which does not
fit alongside generator + fake_score + real_score + optimizer state. The
fix is that the discriminator only ever reads the SHALLOW taps
(``ladd_feature_blocks: [0, 2, 4, 8]`` — texture lives early), so blocks
above the deepest tap are never executed and never need to exist:

    num_layers = max(taps) + 1        # 9 blocks for taps [0, 2, 4, 8]

A 9-block prefix plus the four embedding stacks is ~3.4 B params ≈ 6.8 GB
in bf16, and the weights for blocks 0-8 live entirely in shards 1-2 of the
6-shard checkpoint (``head.*`` is in shard 6; ``safetensors`` reads only
the requested tensors, so pulling the 3 tiny head tensors out of shard 6
costs a seek, not 5 GB).

Contract with ``model/ladd_disc.py``
------------------------------------
* The returned module is a RAW ``WanModel``, not a ``WanDiffusionWrapper``.
  The wrapper consumes the model's return value
  (``utils/wan_wrapper.py:749-753`` unpacks ``flow_pred`` and converts it
  to x0), which a truncated backbone cannot produce meaningfully; the
  projector only wants hooked block outputs and discards the return value
  (``model/ladd_disc.py``'s ``_run_teacher``).
* It is ``requires_grad_(False).eval()``: frozen. The input-to-feature
  path stays differentiable, which is what the disc needs.
* ``head`` is loaded when available purely so the discarded forward
  output is not garbage; when ``load_head=False`` the head is
  ZERO-filled (never left as uninitialised ``to_empty`` memory, which
  would produce NaNs and poison the loss through nothing at all).

Memory note: the module is constructed on the ``meta`` device, cast to
the target dtype while still meta (free), then materialised straight on
the target device and filled tensor-by-tensor from the shards. Peak host
RAM is one tensor, not one model — a plain fp32 CPU construction would
cost ~13.6 GB per rank BEFORE the cast, which with 8 ranks/node is an
OOM.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch


__all__ = [
    "WAN14B_DEFAULT_PATH",
    "load_wan14b_prefix",
    "load_wan_prefix",
    "prefix_num_layers_for_taps",
]


# Default on-cluster location (symlinks into the shared Wan-AI mirror).
WAN14B_DEFAULT_PATH = "/scratch/u6ex/as1748.u6ex/frodobots/Wan2.1-T2V-14B"

# Config keys we hard-require from the checkpoint's ``config.json``. The
# values are NOT hardcoded here — they are read from the checkpoint so a
# different Wan variant loads correctly or fails loudly, never silently
# with a mis-shaped model.
_REQUIRED_CFG_KEYS = (
    "dim", "ffn_dim", "num_heads", "num_layers", "in_dim", "out_dim",
)

# Reference values for Wan2.1-T2V-14B — used only for the log line and
# for a sanity warning when a checkpoint disagrees.
_WAN14B_REFERENCE = {
    "dim": 5120, "ffn_dim": 13824, "num_heads": 40, "num_layers": 40,
    "in_dim": 16, "out_dim": 16,
}


def prefix_num_layers_for_taps(block_indices: Sequence[int]) -> int:
    """Number of transformer blocks a tap list requires: ``max(taps) + 1``."""
    taps = [int(i) for i in block_indices]
    if not taps:
        raise ValueError(
            "prefix_num_layers_for_taps: empty tap list. The 14B backbone "
            "requires EXPLICIT ``ladd_feature_blocks`` — the auto-default "
            "for a 40-block teacher is [8, 16, 24, 32, 39], which would "
            "load the full 28 GB model."
        )
    if min(taps) < 0:
        raise ValueError(f"negative block index in {taps}")
    return max(taps) + 1


def _read_config(model_path: str) -> Dict[str, int]:
    cfg_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(
            f"load_wan14b_prefix: no config.json under {model_path!r}. "
            "Point ``ladd_disc_backbone_model_name`` at a Wan diffusers "
            "checkpoint directory (the one holding config.json + "
            "diffusion_pytorch_model*.safetensors)."
        )
    with open(cfg_path, "r") as fh:
        cfg = json.load(fh)
    missing = [k for k in _REQUIRED_CFG_KEYS if k not in cfg]
    if missing:
        raise KeyError(
            f"load_wan14b_prefix: {cfg_path} is missing required keys "
            f"{missing}. Got keys {sorted(cfg.keys())}."
        )
    return cfg


def _build_weight_map(model_path: str) -> Dict[str, str]:
    """``{tensor_key: absolute shard path}`` for the checkpoint."""
    index_path = os.path.join(
        model_path, "diffusion_pytorch_model.safetensors.index.json"
    )
    if os.path.isfile(index_path):
        with open(index_path, "r") as fh:
            weight_map = json.load(fh)["weight_map"]
        return {
            k: os.path.join(model_path, v) for k, v in weight_map.items()
        }
    # Single-file checkpoint (no shard index).
    shards = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(
            "load_wan14b_prefix: no *.safetensors and no shard index under "
            f"{model_path!r}."
        )
    from safetensors import safe_open
    weight_map: Dict[str, str] = {}
    for shard in shards:
        with safe_open(shard, framework="pt") as fh:
            for key in fh.keys():
                weight_map[key] = shard
    return weight_map


def _wanted_keys(state_keys: Sequence[str], num_layers: int) -> List[str]:
    """State-dict keys that belong to the truncated prefix model."""
    wanted = []
    for key in state_keys:
        if key.startswith("blocks."):
            idx = int(key.split(".")[1])
            if idx >= num_layers:
                continue
        wanted.append(key)
    return wanted


def load_wan14b_prefix(
    model_path: str = WAN14B_DEFAULT_PATH,
    *,
    max_block: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    load_head: bool = True,
    low_cpu_mem: bool = True,
    log: Optional[bool] = None,
) -> torch.nn.Module:
    """Load blocks ``0..max_block`` of a Wan DiT as a frozen raw ``WanModel``.

    Args:
        model_path: diffusers-format Wan checkpoint directory.
        max_block: deepest transformer block the caller taps. The model
            is built with ``max_block + 1`` blocks; everything deeper is
            never constructed and never read off disk.
        device: target device for the materialised weights.
        dtype: target dtype (checkpoint is fp32 on disk; bf16 halves it).
        load_head: load ``head.*`` from the checkpoint. When False the
            head is zero-filled — the projector discards the model
            output either way, but the parameters must not be left as
            uninitialised memory.
        low_cpu_mem: construct on ``meta`` and materialise straight onto
            ``device`` (peak host RAM = one tensor). Set False to fall
            back to a plain CPU construction (needs ~2x the model size
            in host RAM, but does not depend on ``to_empty``).
        log: emit the one-line summary at INFO. ``None`` (default) means
            rank 0 only — otherwise all 8 ranks print it.

    Returns:
        A frozen (``requires_grad_(False).eval()``) ``WanModel`` with
        ``max_block + 1`` transformer blocks, on ``device``/``dtype``.
    """
    from safetensors import safe_open
    from wan.modules.model import WanModel, rope_params

    if log is None:
        try:
            import torch.distributed as _dist
            log = (not _dist.is_available() or not _dist.is_initialized()
                   or _dist.get_rank() == 0)
        except Exception:
            log = True

    max_block = int(max_block)
    if max_block < 0:
        raise ValueError(f"max_block must be >= 0; got {max_block}")

    cfg = _read_config(model_path)
    _mtype = str(cfg.get("model_type", "t2v"))
    if _mtype != "t2v":
        # An i2v backbone builds img_emb + i2v cross-attention and then
        # requires ``clip_fea``/``y`` at forward time, which the feature
        # projector has no way to supply: it dies at the FIRST disc
        # forward on a bare ``AssertionError`` (wan/modules/model.py),
        # after ~7 GB has been loaded on every rank. Refuse here instead.
        raise ValueError(
            f"load_wan14b_prefix: model_type={_mtype!r} at {model_path!r}. "
            "Only 't2v' is supported — the projector cannot supply the "
            "clip_fea/y conditioning an i2v backbone requires."
        )
    ckpt_layers = int(cfg["num_layers"])
    # HARD ASSERT (GAN_REDESIGN B2 step (d)): a tap deeper than the
    # checkpoint has no weights to load and would silently run on
    # uninitialised blocks.
    if max_block >= ckpt_layers:
        raise ValueError(
            f"load_wan14b_prefix: deepest requested tap {max_block} is out "
            f"of range for a {ckpt_layers}-block checkpoint at "
            f"{model_path!r}. ``ladd_feature_blocks`` must satisfy "
            f"max(taps) < {ckpt_layers}."
        )
    num_layers = max_block + 1

    # Geometry is whatever the checkpoint says. The 14B reference is only a
    # readback aid: this loader is deliberately model-agnostic (GAN_REDESIGN
    # TWO Option B points it at the stock 1.3B, where every one of these keys
    # legitimately differs), so a mismatch is INFO, never a warning.
    _diffs = {
        k: (cfg[k], ref) for k, ref in _WAN14B_REFERENCE.items()
        if int(cfg[k]) != ref
    }
    if _diffs and log:
        logging.info(
            "[wan14b_prefix] non-14B geometry (fine — dim_teacher/patch are "
            "read from the checkpoint): %s",
            {k: f"{got} (14B: {ref})" for k, (got, ref) in _diffs.items()},
        )

    device = torch.device(device)
    build_kwargs = dict(
        model_type=str(cfg.get("model_type", "t2v")),
        patch_size=tuple(cfg.get("patch_size", (1, 2, 2))),
        text_len=int(cfg.get("text_len", 512)),
        in_dim=int(cfg["in_dim"]),
        dim=int(cfg["dim"]),
        ffn_dim=int(cfg["ffn_dim"]),
        freq_dim=int(cfg.get("freq_dim", 256)),
        text_dim=int(cfg.get("text_dim", 4096)),
        out_dim=int(cfg["out_dim"]),
        num_heads=int(cfg["num_heads"]),
        num_layers=num_layers,
        eps=float(cfg.get("eps", 1e-6)),
        # Architecture flags: forward them when the checkpoint declares
        # them. Ignoring these built a different architecture and then
        # blamed the CHECKPOINT for the resulting missing tensors —
        # and ``window_size`` carries no weights at all, so a local-
        # attention checkpoint would silently run global attention.
        # Neither Wan2.1 config on disk sets them; the defaults match.
        qk_norm=bool(cfg.get("qk_norm", True)),
        cross_attn_norm=bool(cfg.get("cross_attn_norm", True)),
        window_size=tuple(cfg.get("window_size", (-1, -1))),
    )

    if low_cpu_mem:
        # Construct on meta: ``init_weights``' xavier draws are no-ops on
        # meta tensors, so the ~3.4 B random fp32 numbers are never
        # generated. Cast while meta (free), then allocate on-device.
        with torch.device("meta"):
            model = WanModel(**build_kwargs)
        model.to(dtype)
        model.to_empty(device=device)
    else:
        model = WanModel(**build_kwargs)
        model.to(device=device, dtype=dtype)

    # ``self.freqs`` is a PLAIN attribute (not a buffer — see the comment
    # in WanModel.__init__), so neither ``.to()`` nor ``to_empty()``
    # touches it; under the meta build it is a meta tensor and the
    # forward's ``self.freqs.to(device)`` would raise. Recompute it
    # verbatim (same formula, same theta, same table size) on the target
    # device.
    head_dim = build_kwargs["dim"] // build_kwargs["num_heads"]
    rope_len = int(getattr(model, "rope_max_seq_len", 10000))
    model.freqs = torch.cat(
        [
            rope_params(rope_len, head_dim - 4 * (head_dim // 6)),
            rope_params(rope_len, 2 * (head_dim // 6)),
            rope_params(rope_len, 2 * (head_dim // 6)),
        ],
        dim=1,
    ).to(device)

    # ---- fill every parameter/buffer from the shards -----------------
    weight_map = _build_weight_map(model_path)
    target_sd = model.state_dict()
    wanted = _wanted_keys(list(target_sd.keys()), num_layers)

    zero_filled: List[str] = []
    missing: List[str] = []
    by_shard: Dict[str, List[str]] = {}
    for key in wanted:
        if key.startswith("head.") and not load_head:
            zero_filled.append(key)
            continue
        shard = weight_map.get(key)
        if shard is None:
            missing.append(key)
            continue
        by_shard.setdefault(shard, []).append(key)

    if missing:
        # ``head.*`` absent from the checkpoint is survivable (the output
        # is discarded); anything else is a real mismatch.
        head_missing = [k for k in missing if k.startswith("head.")]
        other_missing = [k for k in missing if not k.startswith("head.")]
        if other_missing:
            raise KeyError(
                "load_wan14b_prefix: checkpoint is missing "
                f"{len(other_missing)} required tensors, e.g. "
                f"{sorted(other_missing)[:5]}. Refusing to run on "
                "uninitialised weights."
            )
        logging.warning(
            "[wan14b_prefix] head tensors %s absent from the checkpoint "
            "— zero-filling (the projector discards the model output).",
            sorted(head_missing),
        )
        zero_filled.extend(head_missing)

    loaded = 0
    n_elem = 0
    with torch.no_grad():
        for key in zero_filled:
            target_sd[key].zero_()
            n_elem += target_sd[key].numel()
        for shard in sorted(by_shard):
            with safe_open(shard, framework="pt") as fh:
                for key in by_shard[shard]:
                    src = fh.get_tensor(key)
                    dst = target_sd[key]
                    if tuple(src.shape) != tuple(dst.shape):
                        raise ValueError(
                            f"load_wan14b_prefix: shape mismatch for {key!r}: "
                            f"checkpoint {tuple(src.shape)} vs model "
                            f"{tuple(dst.shape)}."
                        )
                    dst.copy_(src.to(dtype=dst.dtype))
                    loaded += 1
                    n_elem += dst.numel()

    covered = loaded + len(zero_filled)
    if covered != len(wanted):
        raise RuntimeError(
            f"load_wan14b_prefix: filled {covered}/{len(wanted)} tensors — "
            "some parameters would be uninitialised. This is a loader bug."
        )

    model.requires_grad_(False)
    model.eval()
    # Provenance for the trainer's log line / the disc-side assert.
    model._wan14b_prefix_source = str(model_path)
    model._wan14b_prefix_num_layers = int(num_layers)
    model._wan14b_prefix_ckpt_num_layers = int(ckpt_layers)

    if log:
        logging.info(
            "[wan14b_prefix] loaded %d/%d blocks from %s: dim=%d ffn=%d "
            "heads=%d in_dim=%d patch=%s params=%.2fB (%.2f GB @ %s) "
            "shards=%d head=%s device=%s",
            num_layers, ckpt_layers, model_path, build_kwargs["dim"],
            build_kwargs["ffn_dim"], build_kwargs["num_heads"],
            build_kwargs["in_dim"], build_kwargs["patch_size"],
            n_elem / 1e9,
            n_elem * torch.empty((), dtype=dtype).element_size() / 1e9,
            dtype, len(by_shard),
            "zeroed" if any(k.startswith("head.") for k in zero_filled)
            else "loaded",
            device,
        )
    return model


# Nothing in this module is 14B-specific — geometry comes from the
# checkpoint's config.json, and the trainer reads dim_teacher/patch_size back
# off the returned model. ``load_wan_prefix`` is the name to use when the
# target is not the 14B (e.g. GAN_REDESIGN TWO Option B: the stock 1.3B).
load_wan_prefix = load_wan14b_prefix
