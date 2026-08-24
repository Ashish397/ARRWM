"""Tests for WP-14B: the Wan2.1-T2V-14B prefix backbone + LADD projector.

Covers
------
* ``model/wan14b_prefix.py``: prefix truncation, sharded-index load,
  exact weight match vs the source model, dtype/device/frozen state,
  head zero-fill, and the two fail-loud asserts (tap beyond the
  checkpoint, missing tensors).
* ``model/ladd_disc.py`` raw-backbone mode: the projector calls the raw
  ``WanModel`` (never ``WanDiffusionWrapper``) at the chunk's ACTUAL
  token count, taps the right blocks, stays differentiable w.r.t. the
  input, and refuses the three silent-corruption paths (action tokens,
  a tap deeper than the loaded prefix, ``clean_x``/``seq_len`` that the
  raw model cannot honour).
* Byte-identical OFF: with ``backbone=None`` the projector calls
  ``real_score`` with exactly the legacy kwargs.

CPU-only by default. ``wan`` touches ``torch.cuda`` at import, and WAN's
attention asserts CUDA when flash-attn is installed, so both are stubbed
on a CPU-only host — test scaffolding only, nothing in the shipped path
changes, and on a GPU host the real flash-attn kernel is exercised.

Additionally, ``WAN14B_PREFIX_SMOKE=1`` runs the REAL load-and-forward
smoke against the on-disk 14B checkpoint (~6.8 GB, taps [0, 2, 4, 8]);
it is skipped by default.

Run:
    python testing/test_wan14b_prefix.py
    WAN14B_PREFIX_SMOKE=1 python testing/test_wan14b_prefix.py
or  python -m pytest testing/test_wan14b_prefix.py -q
"""
import os
import sys
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- CPU scaffolding -------------------------------------------------
if not torch.cuda.is_available():
    # ``wan/modules/t5.py`` evaluates ``torch.cuda.current_device()`` in a
    # class body at import time.
    torch.cuda.current_device = lambda: 0

import wan.modules.model as wan_model  # noqa: E402
from model.ladd_disc import (  # noqa: E402
    WanFeatureProjector,
    build_ladd_disc,
)
from model.wan14b_prefix import (  # noqa: E402
    WAN14B_DEFAULT_PATH,
    load_wan14b_prefix,
    prefix_num_layers_for_taps,
)


def _sdpa_attention(q, k, v, k_lens=None, window_size=(-1, -1), **_kw):
    """CPU stand-in for WAN's flash-attn call (uniform lengths only)."""
    assert window_size == (-1, -1)
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    )
    return out.transpose(1, 2).contiguous()


# The tiny fixtures below run on CPU, where WAN's flash-attn path asserts
# CUDA, so the kernel is stubbed globally. The real-weights smoke restores
# the genuine kernel when it runs on a GPU (that is what training uses).
_REAL_FLASH_ATTENTION = wan_model.flash_attention
wan_model.flash_attention = _sdpa_attention

TINY = dict(
    model_type="t2v", patch_size=(1, 2, 2), text_len=8, in_dim=16, dim=32,
    ffn_dim=64, freq_dim=256, text_dim=64, out_dim=16, num_heads=4,
)
CKPT_LAYERS = 4


def _write_tiny_ckpt(tmpdir, *, drop_key=None, sharded=True):
    """Write a 4-block tiny Wan checkpoint; returns (path, source model)."""
    import json
    from safetensors.torch import save_file

    src = wan_model.WanModel(num_layers=CKPT_LAYERS, **TINY)
    sd = {k: v.contiguous().float() for k, v in src.state_dict().items()}
    if drop_key is not None:
        sd.pop(drop_key)
    cfg = dict(TINY)
    cfg["num_layers"] = CKPT_LAYERS
    cfg["patch_size"] = list(cfg["patch_size"])
    with open(os.path.join(tmpdir, "config.json"), "w") as fh:
        json.dump(cfg, fh)

    if not sharded:
        save_file(sd, os.path.join(tmpdir, "diffusion_pytorch_model.safetensors"))
        return tmpdir, src

    # Mimic the real layout: embeddings + shallow blocks in shard 1,
    # deep blocks + head in the last shard.
    def _shard_of(key):
        if key.startswith("blocks."):
            return 0 if int(key.split(".")[1]) < 2 else 1
        return 1 if key.startswith("head.") else 0

    names = [
        "diffusion_pytorch_model-00001-of-00002.safetensors",
        "diffusion_pytorch_model-00002-of-00002.safetensors",
    ]
    weight_map = {k: names[_shard_of(k)] for k in sd}
    for i, name in enumerate(names):
        save_file(
            {k: v for k, v in sd.items() if _shard_of(k) == i},
            os.path.join(tmpdir, name),
        )
    with open(
        os.path.join(
            tmpdir, "diffusion_pytorch_model.safetensors.index.json"
        ), "w",
    ) as fh:
        json.dump({"metadata": {}, "weight_map": weight_map}, fh)
    return tmpdir, src


# =====================================================================
# prefix loader
# =====================================================================


def test_prefix_num_layers_for_taps():
    assert prefix_num_layers_for_taps([0, 2, 4, 8]) == 9
    assert prefix_num_layers_for_taps([0]) == 1
    for bad, want in (([], "EXPLICIT"), ([-1, 3], "negative")):
        try:
            prefix_num_layers_for_taps(bad)
        except ValueError as exc:
            # Message-checked: with the empty-list guard deleted,
            # `max([])` raises ValueError too and a bare except passes.
            assert want in str(exc), (want, str(exc))
            continue
        raise AssertionError(f"expected ValueError for {bad}")


def _assert_prefix_matches(model, src, num_layers, dtype):
    assert len(model.blocks) == num_layers
    assert not model.training
    got = model.state_dict()
    ref = src.state_dict()
    for key, val in got.items():
        assert val.dtype == dtype, f"{key}: {val.dtype}"
        exp = ref[key].to(dtype)
        assert torch.equal(val, exp), f"weight mismatch at {key}"
    # blocks past the prefix were never constructed
    assert not any(
        k.startswith(f"blocks.{i}.") for k in got for i in range(
            num_layers, CKPT_LAYERS
        )
    )
    assert all(not p.requires_grad for p in model.parameters())
    assert model.freqs.device.type == "cpu"
    assert model.freqs.is_complex()


def test_prefix_load_sharded_matches_source():
    with tempfile.TemporaryDirectory() as td:
        path, src = _write_tiny_ckpt(td)
        for low_cpu_mem in (True, False):
            model = load_wan14b_prefix(
                path, max_block=2, device="cpu", dtype=torch.float32,
                low_cpu_mem=low_cpu_mem, log=False,
            )
            _assert_prefix_matches(model, src, 3, torch.float32)
            assert model._wan14b_prefix_num_layers == 3
            assert model._wan14b_prefix_ckpt_num_layers == CKPT_LAYERS


def test_prefix_load_single_file_and_bf16():
    with tempfile.TemporaryDirectory() as td:
        path, src = _write_tiny_ckpt(td, sharded=False)
        model = load_wan14b_prefix(
            path, max_block=0, device="cpu", dtype=torch.bfloat16, log=False,
        )
        _assert_prefix_matches(model, src, 1, torch.bfloat16)


def test_tap_beyond_checkpoint_raises():
    with tempfile.TemporaryDirectory() as td:
        path, _ = _write_tiny_ckpt(td)
        try:
            load_wan14b_prefix(path, max_block=CKPT_LAYERS, device="cpu",
                               log=False)
        except ValueError as exc:
            assert "out of range" in str(exc)
            return
        raise AssertionError("expected ValueError for a tap past the ckpt")


def test_missing_tensor_is_fail_loud():
    with tempfile.TemporaryDirectory() as td:
        path, _ = _write_tiny_ckpt(td, drop_key="blocks.1.self_attn.q.weight")
        try:
            load_wan14b_prefix(path, max_block=2, device="cpu",
                               dtype=torch.float32, log=False)
        except KeyError as exc:
            assert "missing" in str(exc)
            return
        raise AssertionError("expected KeyError for a missing block tensor")


def test_load_head_false_zero_fills():
    with tempfile.TemporaryDirectory() as td:
        path, _ = _write_tiny_ckpt(td)
        model = load_wan14b_prefix(
            path, max_block=1, device="cpu", dtype=torch.float32,
            load_head=False, log=False,
        )
        for name, param in model.head.named_parameters():
            # Never uninitialised ``to_empty`` memory.
            assert torch.count_nonzero(param) == 0, name
            assert torch.isfinite(param).all(), name


# =====================================================================
# LADD projector — raw-backbone mode
# =====================================================================


def _tiny_backbone(tmpdir, max_block=2):
    path, _ = _write_tiny_ckpt(tmpdir)
    return load_wan14b_prefix(
        path, max_block=max_block, device="cpu", dtype=torch.float32,
        log=False,
    )


def _disc_inputs(backbone, b=2, f=2, h=8, w=16, grad=True):
    # NON-SQUARE by default (h != w): with square fixtures an H/W
    # transposition in the seq_len arithmetic or in the [B,F,C,H,W] ->
    # [B,C,F,H,W] permute is arithmetically invisible.
    x = torch.randn(b, f, TINY["in_dim"], h, w, requires_grad=grad)
    # DISTINCT per-frame timesteps: a uniform t cannot distinguish real
    # per-frame handling from broadcasting one column.
    t = torch.tensor([[500, 300][:f] * 1 for _ in range(b)],
                     dtype=torch.long)[:, :f]
    pe = torch.randn(b, TINY["text_len"], TINY["text_dim"])
    return x, t, pe


def test_projector_runs_raw_backbone_at_actual_token_count():
    with tempfile.TemporaryDirectory() as td:
        backbone = _tiny_backbone(td)
        seen = {}
        raw_forward = backbone.forward

        def _spy(x, t, context, seq_len, **kw):
            seen["seq_len"] = seq_len
            seen["x_shape"] = tuple(x.shape)
            seen["t_shape"] = tuple(t.shape)
            return raw_forward(x, t, context, seq_len, **kw)

        backbone.forward = _spy
        proj = WanFeatureProjector(
            real_score=None, block_indices=[0, 2], backbone=backbone,
        )
        x, t, pe = _disc_inputs(backbone)
        feats = proj(x_noisy=x, timestep=t, prompt_embeds=pe)

        n_tok = 2 * (8 // 2) * (16 // 2)  # T' * H' * W' = 64
        assert seen["seq_len"] == n_tok, seen
        # NOT the wrapper's padded budget, and the model was handed
        # [B, C, F, H, W] with H and W the right way round.
        assert seen["x_shape"] == (2, TINY["in_dim"], 2, 8, 16)
        assert seen["t_shape"] == (2, 2)
        assert sorted(feats) == [0, 2]
        for idx, feat in feats.items():
            assert feat.shape == (2, n_tok, TINY["dim"]), (idx, feat.shape)
        # differentiable w.r.t. the disc input
        sum(f.float().sum() for f in feats.values()).backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()


def test_projector_taps_the_requested_blocks():
    with tempfile.TemporaryDirectory() as td:
        backbone = _tiny_backbone(td)
        proj = WanFeatureProjector(
            real_score=None, block_indices=[0, 2], backbone=backbone,
        )
        x, t, pe = _disc_inputs(backbone, grad=False)
        feats = proj(x_noisy=x, timestep=t, prompt_embeds=pe)
        # Block 2's output must differ from block 0's (real depth, not
        # the same hook firing twice), and both must be finite.
        assert not torch.allclose(feats[0], feats[2])
        assert torch.isfinite(feats[0]).all()
        assert torch.isfinite(feats[2]).all()
        # No persistent hooks left behind.
        for blk in backbone.blocks:
            assert not blk._forward_hooks


def test_projector_rejects_wrapper_only_arguments():
    with tempfile.TemporaryDirectory() as td:
        backbone = _tiny_backbone(td)
        proj = WanFeatureProjector(
            real_score=None, block_indices=[0], backbone=backbone,
        )
        x, t, pe = _disc_inputs(backbone, grad=False)
        for kwargs, want in (
            ({"clean_x": torch.zeros_like(x)}, "clean_x"),
            ({"aug_t": t}, "aug_t"),
            ({"seq_len": 18721}, "actual token count"),
        ):
            try:
                proj(x_noisy=x, timestep=t, prompt_embeds=pe, **kwargs)
            except ValueError as exc:
                # Assert on the MESSAGE: a bare `except ValueError` is
                # satisfied by any coincidental ValueError, so the guard
                # could be deleted and the test would still pass.
                assert want in str(exc), (want, str(exc))
                continue
            raise AssertionError(f"expected ValueError for {list(kwargs)}")


def test_build_ladd_disc_backbone_guards():
    with tempfile.TemporaryDirectory() as td:
        backbone = _tiny_backbone(td, max_block=2)
        common = dict(
            real_score=None, dim_teacher=TINY["dim"], dim_proj=8,
            use_csm=False, patch_size=(1, 2, 2), backbone=backbone,
        )
        # tap deeper than the loaded prefix
        try:
            build_ladd_disc(block_indices=[0, 3], **common)
            raise AssertionError("expected ValueError for a deep tap")
        except ValueError as exc:
            assert "num_layers_loaded" in str(exc)
        # action tokens harvested off real_score
        try:
            build_ladd_disc(
                block_indices=[0], action_tokens_per_frame=1, **common
            )
            raise AssertionError("expected ValueError for a_per_f=1")
        except ValueError as exc:
            assert "action tokens" in str(exc)
        # dim_teacher mismatch
        bad = dict(common)
        bad["dim_teacher"] = 1536
        try:
            build_ladd_disc(block_indices=[0], **bad)
            raise AssertionError("expected ValueError for dim mismatch")
        except ValueError as exc:
            assert "hidden dim" in str(exc)


def test_disc_forward_through_raw_backbone():
    with tempfile.TemporaryDirectory() as td:
        backbone = _tiny_backbone(td)
        disc = build_ladd_disc(
            real_score=None, block_indices=[0, 2], dim_teacher=TINY["dim"],
            dim_proj=8, use_csm=True, cmap_dim=0, patch_size=(1, 2, 2),
            action_tokens_per_frame=0, backbone=backbone,
        )
        disc.to(dtype=torch.float32)
        x, t, pe = _disc_inputs(backbone)
        logits = disc(x_noisy=x, timestep=t, prompt_embeds=pe)
        # per-token logits over both taps: 2 taps * B * T' * H' * W'
        assert logits.shape[0] == 2
        assert logits.shape[1] == 2 * 2 * 4 * 8
        assert torch.isfinite(logits).all()
        logits.sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        # the frozen backbone accumulated no gradient and is not a
        # disc parameter (so DDP never sees it, and neither does the
        # optimizer or the checkpoint)
        assert all(p.grad is None for p in backbone.parameters())
        disc_params = {id(p) for p in disc.parameters()}
        assert not any(id(p) in disc_params for p in backbone.parameters())


# =====================================================================
# gap-closing tests (added after a mutation audit found the originals
# could not distinguish which block was tapped, never checked the RoPE
# table, and never exercised the production bf16/fp32 pairing)
# =====================================================================


def test_projector_tap_identity_is_exact():
    """Each tap must be THAT block's OUTPUT — not a neighbour, not its input.

    The original assertion was only ``feats[0] != feats[2]``, which any two
    distinct tensors satisfy: a tap shifted by one block, or a hook capturing
    block INPUT, both passed. A silent tap shift changes the feature
    distribution the GAN gradient is computed against, with no crash.
    """
    with tempfile.TemporaryDirectory() as td:
        backbone = _tiny_backbone(td)
        x, t, pe = _disc_inputs(backbone, grad=False)
        proj = WanFeatureProjector(
            real_score=None, block_indices=[0, 2], backbone=backbone,
        )
        feats = proj(x_noisy=x, timestep=t, prompt_embeds=pe)

        # Independent reference: hook EVERY block, replay the same call.
        ref_out, ref_in = {}, {}

        def _mk(k):
            def _h(_m, inp, out):
                ref_out[k] = (
                    out[0] if isinstance(out, tuple) else out
                ).detach().clone()
                ref_in[k] = inp[0].detach().clone()
            return _h

        handles = [
            b.register_forward_hook(_mk(k))
            for k, b in enumerate(backbone.blocks)
        ]
        try:
            with torch.no_grad():
                    # Token count derived from the fixture's own shape, NOT a
                # literal: a hardcoded 2*4*8 silently decouples from
                # ``_disc_inputs`` the moment its defaults change, and the
                # reference forward would then compare against a differently
                # padded run.
                _b, _f, _c, _h, _w = x.shape
                _pt, _ph, _pw = backbone.patch_size
                backbone(
                    x.permute(0, 2, 1, 3, 4), t=t, context=pe,
                    seq_len=(_f // _pt) * (_h // _ph) * (_w // _pw),
                    max_block=2,
                )
        finally:
            for h in handles:
                h.remove()

        for idx in (0, 2):
            assert torch.equal(feats[idx], ref_out[idx]), (
                f"tap {idx} is not block {idx}'s output")
            assert not torch.equal(feats[idx], ref_in[idx]), (
                f"tap {idx} returned block {idx}'s INPUT")


def test_prefix_freqs_match_a_reference_model():
    """The hand-recomputed RoPE table must equal the stock WanModel's.

    ``freqs`` is a plain attribute, so it is absent from state_dict() and
    escapes the exact-weight comparison entirely. A wrong split or theta
    gives a silently mis-positioned attention geometry.
    """
    with tempfile.TemporaryDirectory() as td:
        path, src = _write_tiny_ckpt(td)
        model = load_wan14b_prefix(
            path, max_block=2, device="cpu", dtype=torch.float32, log=False,
        )
        assert model.freqs.shape == src.freqs.shape
        assert torch.equal(model.freqs, src.freqs.to(model.freqs.device)), (
            "recomputed RoPE table differs from the stock WanModel's")


def test_projector_casts_fp32_input_for_a_bf16_backbone():
    """The production pairing: bf16 backbone, fp32 disc tensors.

    Every other projector test builds the backbone in fp32, so the cast in
    ``_cast_if_float`` could be deleted with the whole suite still green
    while production died on "Input type (float) and bias type
    (c10::BFloat16) should be the same".
    """
    with tempfile.TemporaryDirectory() as td:
        path, _ = _write_tiny_ckpt(td)
        backbone = load_wan14b_prefix(
            path, max_block=2, device="cpu", dtype=torch.bfloat16, log=False,
        )
        proj = WanFeatureProjector(
            real_score=None, block_indices=[0, 2], backbone=backbone,
        )
        x, t, pe = _disc_inputs(backbone, grad=True)
        feats = proj(x_noisy=x, timestep=t, prompt_embeds=pe)
        assert feats[0].dtype == torch.bfloat16, feats[0].dtype
        sum(f.float().sum() for f in feats.values()).backward()
        assert x.grad is not None and x.grad.dtype == torch.float32
        assert torch.isfinite(x.grad).all()


def test_head_missing_from_ckpt_is_zero_filled():
    """head.* absent from the checkpoint -> zeros, never to_empty garbage."""
    with tempfile.TemporaryDirectory() as td:
        path, _ = _write_tiny_ckpt(td, drop_key="head.head.weight")
        model = load_wan14b_prefix(
            path, max_block=1, device="cpu", dtype=torch.float32, log=False,
        )
        w = model.head.head.weight
        assert torch.count_nonzero(w) == 0, "head.head.weight not zero-filled"
        assert torch.isfinite(w).all()


def test_every_param_is_finite_after_load():
    """Global net against ANY unfilled ``to_empty`` tensor.

    ``to_empty`` hands back uninitialised memory; the coverage check is the
    primary defence, but it is one assertion in one test. This is a cheap
    second net that does not care WHICH tensor was missed.
    """
    with tempfile.TemporaryDirectory() as td:
        backbone = _tiny_backbone(td)
        for n, p in backbone.named_parameters():
            assert torch.isfinite(p).all(), f"non-finite parameter {n}"
            assert p.abs().max() < 1e4, f"implausible magnitude in {n}"


def test_loader_rejects_i2v_checkpoints():
    """i2v needs clip_fea/y the projector cannot supply — refuse at LOAD.

    Without this the load succeeds, ~7 GB lands on every rank, and the run
    dies at the first disc forward on a bare AssertionError.
    """
    import json
    with tempfile.TemporaryDirectory() as td:
        path, _ = _write_tiny_ckpt(td)
        cfg = json.load(open(os.path.join(path, "config.json")))
        cfg["model_type"] = "i2v"
        with open(os.path.join(path, "config.json"), "w") as fh:
            json.dump(cfg, fh)
        try:
            load_wan14b_prefix(path, max_block=1, device="cpu", log=False)
        except ValueError as exc:
            assert "i2v" in str(exc) or "model_type" in str(exc), str(exc)
            return
        raise AssertionError("expected ValueError for an i2v checkpoint")


def test_build_ladd_disc_rejects_patch_size_mismatch():
    """A COARSER disc patch than the backbone's is otherwise SILENT.

    The backbone emits more tokens than the heads expect, the disc slices
    the first ``real_tokens`` and reshapes them into a wrong grid, and the
    run continues on a bogus spatial layout.
    """
    with tempfile.TemporaryDirectory() as td:
        backbone = _tiny_backbone(td)
        try:
            build_ladd_disc(
                real_score=None, block_indices=[0], dim_teacher=TINY["dim"],
                dim_proj=8, use_csm=False, patch_size=(1, 4, 4),
                action_tokens_per_frame=0, backbone=backbone,
            )
        except ValueError as exc:
            assert "patch_size" in str(exc), str(exc)
            return
        raise AssertionError("expected ValueError for a patch_size mismatch")


# =====================================================================
# byte-identical OFF: backbone=None still drives real_score
# =====================================================================


class _StubWrapper(nn.Module):
    """Minimal ``WanDiffusionWrapper`` stand-in that records its kwargs."""

    def __init__(self, n_blocks=12, dim=8):
        super().__init__()
        self.blocks = nn.ModuleList(
            [nn.Linear(dim, dim) for _ in range(n_blocks)]
        )
        self.dim = dim
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append({
            k: v for k, v in kwargs.items() if k != "noisy_image_or_video"
        })
        x = kwargs["noisy_image_or_video"]
        b, f, _c, h, w = x.shape
        tok = x.mean(dim=2).reshape(b, f * h * w, 1).expand(-1, -1, self.dim)
        for blk in self.blocks:
            tok = blk(tok)
        return tok


def test_backbone_none_keeps_the_legacy_wrapper_call():
    stub = _StubWrapper()
    proj = WanFeatureProjector(real_score=stub, block_indices=[0, 5])
    x = torch.randn(1, 2, 16, 4, 4, requires_grad=True)
    t = torch.zeros(1, 2, dtype=torch.long)
    pe = torch.randn(1, 3, 8)
    feats = proj(x_noisy=x, timestep=t, prompt_embeds=pe)

    assert sorted(feats) == [0, 5]
    assert len(stub.calls) == 1
    call = stub.calls[0]
    # exactly the legacy kwargs — no seq_len, no raw-model arguments
    assert set(call) == {"conditional_dict", "timestep"}
    assert set(call["conditional_dict"]) == {"prompt_embeds"}
    assert torch.equal(call["timestep"], t)
    feats[0].sum().backward()
    assert x.grad is not None


# =====================================================================
# opt-in real-weights smoke
# =====================================================================


def test_real_14b_prefix_load_and_forward():
    if os.environ.get("WAN14B_PREFIX_SMOKE", "") != "1":
        # A print+return here would be counted by pytest as a PASS, so
        # the suite would report "13 passed" while 12 tests ran.
        msg = "real 14B smoke needs WAN14B_PREFIX_SMOKE=1 (and a GPU)"
        try:
            import pytest
            pytest.skip(msg)
        except ImportError:
            print(f"[skip] {msg}")
            return
    path = os.environ.get("WAN14B_PATH", WAN14B_DEFAULT_PATH)
    taps = [0, 2, 4, 8]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        # exercise the real kernel, not the CPU stub
        wan_model.flash_attention = _REAL_FLASH_ATTENTION
    backbone = load_wan14b_prefix(
        path, max_block=max(taps), device=device, dtype=torch.bfloat16,
    )
    assert len(backbone.blocks) == prefix_num_layers_for_taps(taps) == 9
    assert backbone.dim == 5120 and backbone.in_dim == 16
    assert tuple(backbone.patch_size) == (1, 2, 2)
    n_param = sum(p.numel() for p in backbone.parameters())
    print(f"[smoke] prefix params = {n_param / 1e9:.2f}B "
          f"({n_param * 2 / 1e9:.2f} GB bf16)")

    disc = build_ladd_disc(
        real_score=None, block_indices=taps, dim_teacher=int(backbone.dim),
        dim_proj=64, use_csm=True, cmap_dim=0, patch_size=(1, 2, 2),
        action_tokens_per_frame=0, backbone=backbone,
    ).to(device=device, dtype=torch.float32)

    # Small spatial grid keeps the CPU variant of this smoke tractable;
    # the token-count logic is resolution-independent.
    hw = (60, 104) if device == "cuda" else (16, 16)
    x = torch.randn(1, 3, 16, *hw, device=device, requires_grad=True)
    t = torch.full((1, 3), 500, dtype=torch.long, device=device)
    pe = torch.randn(1, 512, 4096, device=device)
    logits = disc(x_noisy=x, timestep=t, prompt_embeds=pe)
    n_tok = 3 * (hw[0] // 2) * (hw[1] // 2)
    assert logits.shape == (1, len(taps) * n_tok), logits.shape
    assert torch.isfinite(logits).all()
    logits.mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    print(f"[smoke] logits {tuple(logits.shape)} over {n_tok} tokens/tap; "
          f"grad_norm={x.grad.norm().item():.4f}")
    if device == "cuda":
        print(f"[smoke] attention kernel = "
              f"{'flash-attn' if wan_model.flash_attention is _REAL_FLASH_ATTENTION else 'sdpa-stub'}; "
              f"peak alloc = "
              f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, "
              f"peak reserved = "
              f"{torch.cuda.max_memory_reserved() / 2**30:.2f} GiB")


if __name__ == "__main__":
    # Both entry points must work: `python testing/test_wan14b_prefix.py`
    # (the usage in this module's docstring) and `pytest`. The GPU smoke
    # raises pytest's Skipped, which is a BaseException and would abort the
    # plain-script run with a traceback — catch it and report it AS a skip,
    # so the summary line never counts an unexecuted test as a pass.
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    n_pass = n_skip = 0
    for fn in fns:
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001 - Skipped is a BaseException
            if type(exc).__name__ == "Skipped":
                print(f"SKIP {fn.__name__}: {exc}")
                n_skip += 1
                continue
            raise
        print(f"PASS {fn.__name__}")
        n_pass += 1
    print(f"\n{n_pass} passed, {n_skip} skipped")
