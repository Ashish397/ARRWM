"""Coverage for ``LADDRegisterReadout`` and ``ladd_readout='register'``.

``docs/ONE_FORCING_PORT.md`` divergence 2: One-Forcing pools each tap with
a LEARNED register token that cross-attends over that tap's token
sequence; ours mean-pools/convolves projected features. The port added
``model/ladd_disc.py::LADDRegisterReadout`` plus the ``readout`` switch on
``LADDDiscriminator`` / ``build_ladd_disc``, with three deliberate
refusals. None of it had a test.

Pinned here:
  * CONSTRUCTION in register mode replaces CCM/CSM/heads/cmapper (they are
    None) -- building-and-not-using them would leave them ungradiented and
    trip the disc's ``find_unused_parameters=False`` DDP reducer;
  * output is ``[B, 1]`` -- the shape the RpGAN/relativistic reductions
    and the R1 finite-difference estimator already assume;
  * the backward reaches EVERY register-head parameter (register tokens,
    both cross-attn stacks, the MLP head) -- again the DDP reducer
    contract, and the thing that silently fails if a tap is dropped;
  * ``readout='ladd'`` is BIT-IDENTICAL to the pre-existing head: same
    parameters from the same seed as the historical call (which passed no
    ``readout`` at all), same forward output, and no ``register_readout.*``
    entry in the state dict;
  * the three refusal paths: unknown readout name, ``cmap_dim > 0``, and
    ``freeze_projector_mixing=true``;
  * ``LADDRegisterReadout``'s own guards (empty taps, missing tap
    features, wrong feature rank/dim, ``real_tokens`` longer than the
    captured sequence) and the padding trim.

CPU-only, fp32, no GPU. ``utils.wan_wrapper`` (pulled in lazily by
``LADDRegisterReadout.__init__`` for ``build_cls_pred_branch``) drags
``wan.modules.t5``, which evaluates ``torch.cuda.current_device()`` at
import, so it is pre-imported here under a patch.

Run:
    python -m pytest testing/test_ladd_readout_register.py -q
or
    python testing/test_ladd_readout_register.py
"""
import os
import sys
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Pre-import the module whose T5 import touches CUDA at module scope. The
# register head's lazy ``from utils.wan_wrapper import build_cls_pred_branch``
# then hits a warm sys.modules entry.
with patch.object(torch.cuda, "current_device", return_value=0):
    import utils.wan_wrapper  # noqa: F401,E402

from model.ladd_disc import (  # noqa: E402
    LADDDiscriminator, LADDRegisterReadout, build_ladd_disc,
)

DIM_T = 16          # dim_teacher; must be divisible by the head count
NHEADS = 4
TAPS = [0, 2]
B, F_, C, H, W = 2, 2, 4, 4, 4
SEED = 1234

REG_KW = dict(
    register_blocks_per_token=1,
    register_block_ffn_dim=32,
    register_block_num_heads=NHEADS,
    register_head_hidden_dim=16,
    register_head_num_layers=1,
    register_head_dropout=0.0,
)


class _Projector:
    """Stand-in for ``WanFeatureProjector``: returns one ``[B, L, dim]``
    token sequence per tap, zero-padded past the real-token region exactly
    the way the WAN wrapper's fixed ``seq_len`` forward does.

    Tokens are the raw 2x2 patch contents (C*ph*pw == ``DIM_T``), so
    distinct tokens really are distinct vectors -- a rank-1 / broadcast
    fixture would make the register's attention uniform and hide a dead
    k/v path."""

    def __init__(self, taps=TAPS, dim=DIM_T, pad=5):
        self.taps = list(taps)
        self.dim = dim
        self.pad = pad
        self.calls = 0

    def __call__(self, x_noisy, **_kw):
        self.calls += 1
        b, f, c, h, w = x_noisy.shape
        patches = F.unfold(
            x_noisy.reshape(b * f, c, h, w), kernel_size=2, stride=2,
        )                                      # [b*f, c*4, (h/2)*(w/2)]
        tok = patches.transpose(1, 2).reshape(b, f * (h // 2) * (w // 2), -1)
        assert tok.shape[-1] == self.dim, tok.shape
        out = {}
        for i in self.taps:
            t = tok * (1.0 + i) + float(i)     # each tap sees its own view
            if self.pad:
                t = torch.cat(
                    [t, t.new_zeros(b, self.pad, self.dim)], dim=1)
            out[i] = t
        return out


def _build(readout="register", *, seed=SEED, **over):
    kw = dict(
        projector=_Projector(),
        block_indices=list(TAPS),
        dim_teacher=DIM_T,
        dim_proj=8,
        use_csm=False,
        cmap_dim=0,
        patch_size=(1, 2, 2),
    )
    if readout is not None:
        kw["readout"] = readout
    if readout == "register":
        kw.update(REG_KW)
    kw.update(over)
    torch.manual_seed(seed)
    return LADDDiscriminator(**kw)


def _x(requires_grad=True, seed=7):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, F_, C, H, W, generator=g)
    x.requires_grad_(requires_grad)
    return x


def _fwd(disc, x):
    return disc(
        x,
        timestep=torch.zeros(B, F_, dtype=torch.long),
        prompt_embeds=torch.zeros(B, 1, 8),
    )


# ===========================================================================
# construction
# ===========================================================================
def test_register_mode_replaces_ccm_csm_heads_and_cmapper():
    d = _build("register")
    assert d.readout == "register"
    assert d.register_readout is not None
    assert isinstance(d.register_readout, LADDRegisterReadout)
    assert d.ccm is None and d.csm is None
    assert d.heads is None and d.cmapper is None


def test_register_head_is_a_submodule_so_it_lands_in_disc_parameters():
    """Parameter ownership is the stated reason the class exists: the head
    must be stepped by ``r3gan_optimizer`` (built from ``disc.parameters()``),
    not by the teacher's or the fake critic's optimizer."""
    d = _build("register")
    own = {id(p) for p in d.parameters()}
    head = {id(p) for p in d.register_readout.parameters()}
    assert head and head <= own
    keys = set(d.state_dict())
    assert any(k.startswith("register_readout.") for k in keys)
    assert d.num_params == sum(p.numel() for p in d.parameters())


def test_register_head_builds_one_register_token_and_stack_per_tap():
    d = _build("register")
    r = d.register_readout
    assert r.block_indices == sorted(TAPS)
    assert r.register_tokens.register_tokens.shape == (len(TAPS), DIM_T)
    assert len(r.gan_ca_blocks) == len(TAPS)
    assert all(len(s) == REG_KW["register_blocks_per_token"]
               for s in r.gan_ca_blocks)
    assert "taps=" in r.describe() and "params=" in r.describe()


def test_block_indices_are_deduped_and_sorted():
    d = _build("register", block_indices=[2, 0, 2])
    assert d.register_readout.block_indices == [0, 2]


class _FakeTeacher(torch.nn.Module):
    """Minimal shape the real ``WanFeatureProjector`` hook-search accepts:
    something with a ``blocks`` ModuleList of >= 10 entries."""

    def __init__(self, n=12):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [torch.nn.Identity() for _ in range(n)])


def test_builder_forwards_the_readout_switch():
    """``build_ladd_disc`` is the only construction site the trainer uses."""
    torch.manual_seed(SEED)
    d = build_ladd_disc(
        real_score=_FakeTeacher(),
        block_indices=list(TAPS),
        dim_teacher=DIM_T,
        dim_proj=8,
        use_csm=False,
        readout="register",
        **REG_KW,
    )
    assert d.readout == "register"
    assert d.register_readout is not None and d.ccm is None


# ===========================================================================
# forward + gradient
# ===========================================================================
def test_output_is_one_logit_per_row():
    d = _build("register").eval()
    out = _fwd(d, _x(requires_grad=False))
    assert out.shape == (B, 1)
    assert torch.isfinite(out).all()


def test_gradient_reaches_every_register_head_parameter():
    """A tap whose stack never runs, or a register token never used, would
    arrive here with ``grad is None`` -- and would take the disc's
    ``find_unused_parameters=False`` DDP reducer down at runtime."""
    d = _build("register").eval()
    x = _x()
    out = _fwd(d, x)
    out.sum().backward()

    named = list(d.register_readout.named_parameters())
    assert named
    missing = [n for n, p in named if p.requires_grad and p.grad is None]
    assert not missing, f"no gradient reached: {missing}"
    dead = [n for n, p in named
            if p.requires_grad and float(p.grad.abs().sum()) == 0.0]
    assert not dead, f"zero gradient reached: {dead}"
    # every structural group is represented
    groups = {n.split(".")[0] for n, _ in named}
    assert {"register_tokens", "gan_ca_blocks", "cls_pred_branch"} <= groups
    # ... and the input keeps its gradient path
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_both_taps_influence_the_logit():
    """Drop one tap's contribution and the logit must move."""
    d = _build("register").eval()
    proj = d.projector
    x = _x(requires_grad=False)
    base = _fwd(d, x).clone()

    orig = proj.__class__.__call__

    def _zero_second(self, x_noisy, **kw):
        feats = orig(self, x_noisy, **kw)
        feats[TAPS[1]] = torch.zeros_like(feats[TAPS[1]])
        return feats

    proj.__class__.__call__ = _zero_second
    try:
        moved = _fwd(d, x)
    finally:
        proj.__class__.__call__ = orig
    assert not torch.allclose(base, moved)


def test_padding_is_trimmed_before_the_register_attends():
    """``real_tokens`` is computed from the latent geometry; the zero pad
    the wrapper appends must not reach the register token."""
    d = _build("register").eval()
    x = _x(requires_grad=False)
    a = _fwd(d, x).clone()
    d.projector.pad = 41           # a lot more padding, same real content
    b = _fwd(d, x)
    assert torch.allclose(a, b, atol=1e-6), (
        "changing only the zero-padding length changed the logit -- the "
        "readout is attending over padding")


def test_register_head_is_deterministic_at_dropout_zero():
    d = _build("register").eval()
    x = _x(requires_grad=False)
    assert torch.equal(_fwd(d, x), _fwd(d, x))


# ===========================================================================
# readout='ladd' is bit-identical to the pre-existing head
# ===========================================================================
def test_ladd_readout_matches_the_historical_no_kwarg_construction():
    """The historical call passed no ``readout``. Same seed -> same
    parameters, byte for byte, and the same forward output. If the switch
    had reordered or added any module construction, the RNG stream would
    diverge and this fails."""
    legacy = _build(None)              # no readout kwarg at all
    explicit = _build("ladd")
    assert legacy.readout == "ladd"

    ls, es = legacy.state_dict(), explicit.state_dict()
    assert set(ls) == set(es)
    for k in ls:
        assert torch.equal(ls[k], es[k]), f"param {k} differs"

    x = _x(requires_grad=False)
    legacy.eval(); explicit.eval()
    assert torch.equal(_fwd(legacy, x), _fwd(explicit, x))


def test_ladd_readout_builds_no_register_parameters():
    d = _build("ladd")
    assert d.register_readout is None
    assert d.ccm is not None and d.heads is not None
    assert not any(k.startswith("register_readout.")
                   for k in d.state_dict())


def test_register_and_ladd_readouts_are_different_critics():
    a = _build("ladd").eval()
    b = _build("register").eval()
    x = _x(requires_grad=False)
    oa, ob = _fwd(a, x), _fwd(b, x)
    assert oa.shape[0] == ob.shape[0] == B
    assert ob.shape == (B, 1)
    # the ladd head emits per-token logits (no scalar_output here)
    assert oa.shape[1] > 1


# ===========================================================================
# refusals
# ===========================================================================
def test_unknown_readout_name_is_refused():
    for bad in ("regsiter", "REGISTER", "one_forcing", ""):
        with pytest.raises(ValueError) as ei:
            _build(bad)
        assert "ladd_readout must be 'ladd' or 'register'" in str(ei.value)
        assert repr(bad) in str(ei.value)


def test_register_refuses_cmap_dim():
    with pytest.raises(ValueError) as ei:
        _build("register", cmap_dim=4, prompt_embed_dim=8)
    msg = str(ei.value)
    assert "does not " in msg and "cmap" in msg
    assert "ladd_cmap_dim=0" in msg


def test_register_refuses_freeze_projector_mixing():
    with pytest.raises(ValueError) as ei:
        _build("register", freeze_projector_mixing=True)
    msg = str(ei.value)
    assert "ladd_freeze_projector_mixing=true" in msg
    assert "inert" in msg


def test_those_two_options_are_still_legal_on_the_ladd_readout():
    """The refusals must be scoped to register mode, not global."""
    d = _build("ladd", cmap_dim=4, prompt_embed_dim=8)
    assert d.cmapper is not None
    d2 = _build("ladd", freeze_projector_mixing=True)
    assert not any(p.requires_grad for p in d2.ccm.parameters())


# ===========================================================================
# LADDRegisterReadout's own guards
# ===========================================================================
def _readout(taps=TAPS, dim=DIM_T):
    torch.manual_seed(SEED)
    return LADDRegisterReadout(
        block_indices=list(taps),
        dim_teacher=dim,
        blocks_per_token=1,
        block_ffn_dim=32,
        block_num_heads=NHEADS,
        head_hidden_dim=16,
        head_num_layers=1,
        head_dropout=0.0,
    )


def test_empty_tap_list_is_refused():
    with pytest.raises(ValueError) as ei:
        _readout(taps=[])
    assert "block_indices is empty" in str(ei.value)


def test_missing_tap_features_raise():
    r = _readout()
    feats = {TAPS[0]: torch.randn(B, 6, DIM_T)}
    with pytest.raises(RuntimeError) as ei:
        r(feats)
    assert "no features for block indices" in str(ei.value)


def test_wrong_feature_rank_raises():
    r = _readout()
    feats = {i: torch.randn(B, 6) for i in TAPS}
    with pytest.raises(ValueError) as ei:
        r(feats)
    assert "expected tap features" in str(ei.value)


def test_wrong_feature_dim_raises():
    r = _readout()
    feats = {i: torch.randn(B, 6, DIM_T + 1) for i in TAPS}
    with pytest.raises(ValueError) as ei:
        r(feats)
    assert "!= dim_teacher" in str(ei.value)


def test_real_tokens_longer_than_the_sequence_raises():
    r = _readout()
    feats = {i: torch.randn(B, 6, DIM_T) for i in TAPS}
    with pytest.raises(RuntimeError) as ei:
        r(feats, real_tokens=99)
    assert "< expected real_tokens" in str(ei.value)


def test_bf16_tap_features_are_cast_at_the_head_boundary():
    """Tap features arrive in the backbone dtype; the head is fp32."""
    r = _readout().eval()
    feats = {i: torch.randn(B, 6, DIM_T, dtype=torch.bfloat16) for i in TAPS}
    out = r(feats)
    assert out.shape == (B, 1)
    assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
def main():
    g = dict(globals())
    names = [n for n in g if n.startswith("test_")]
    names.sort(key=lambda n: g[n].__code__.co_firstlineno)
    for n in names:
        g[n]()
        print(f"  ok  {n}")
    print(f"\nALL {len(names)} TESTS PASSED")


if __name__ == "__main__":
    main()
