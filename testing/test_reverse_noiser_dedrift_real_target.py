"""Coverage for ``reverse_noiser_dedrift_apply_to_real_target`` (2026-08-26).

WHAT THE FLAG DOES.  ``_compute_kl_grad`` forms the DMD gradient as
``grad = pred_fake_image - pred_real_image``, where ``pred_real_image`` is
the TEACHER's x0 estimate computed from the SAME noisy tensor the student's
fake_score denoises -- so it is not clean GT and inherits part of the
student rollout's drift signature.  With the new flag the de-drift operator
(``_dedrift_with_reverse_noiser``) is applied to that target, exactly as
``reverse_noiser_dedrift_apply_to_flash`` already applies it to the
student's flash slab.

WHAT IS PINNED HERE.

  1. DEFAULT-OFF BYTE IDENTITY, against the REAL SOURCE.  The new block is
     lifted out of the shipped ``_compute_kl_grad`` AST (from the
     ``apply_to_real_target`` guard through ``grad = torch.nan_to_num(grad)``)
     and EXECUTED here -- not re-implemented -- so the test cannot drift from
     the code.  Flag off => ``pred_real_image`` is the SAME OBJECT afterwards
     (identity, stronger than bitwise equality: nothing downstream can
     observe a difference), and ``grad`` is bitwise the pre-change value.
     Each identity assertion carries a MUTATION CONTROL that flips exactly
     one flag and shows the assertion then fails.

  2. THE DOUBLE GATE.  Both ``reverse_noiser_dedrift_enabled`` AND
     ``reverse_noiser_dedrift_apply_to_real_target`` must be true.  Either
     one alone is a no-op.  (This project's endemic failure is the silent
     half-enabled path, so both halves are pinned.)

  3. ON-PATH.  With both flags on and a non-identity reverse noiser, the
     value flowing into ``grad``, into the ``self._latest_pred_real_image``
     stash and into the normalizer is EXACTLY what a direct call to
     ``_dedrift_with_reverse_noiser`` returns -- and is NOT the raw teacher
     output.  Both network-selection modes are covered (cycle/``reverse_
     noiser`` and ``fn_pair_mode='rollout_to_gt'``/``forward_noiser``).

  4. GRADIENT-FLOW GUARD.  This is the regression test for the
     "boundary_vae_roundtrip" class of bug (a ``no_grad`` placed one
     statement too wide silently severed the DMD gradient past roll 1).
     The new call site wraps ONLY the de-drift in ``torch.no_grad()``; the
     statements after it must still build a graph.  Pinned by making
     ``pred_fake_image`` graph-connected to a leaf and asserting the leaf
     receives a nonzero finite gradient through ``grad``.  Also pinned: the
     de-drifted target itself stays a CONSTANT (``requires_grad False``),
     matching the pre-change behaviour (the real caller runs the whole
     method under ``torch.no_grad()``, and ``grad`` is consumed detached by
     ``_dmd_loss_with_fp_gate``).

  5. The noiser's ``requires_grad`` flags are restored by the helper.

  6. The model registers the flag and it defaults to False.

CPU-only, no CUDA, no dataset.  ``wan/modules/t5.py`` evaluates
``torch.cuda.current_device()`` at import, so the import is patched exactly
as ``testing/test_dmd_manifold_gate.py`` does.

Run:
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" \
        python -m pytest testing/test_reverse_noiser_dedrift_real_target.py -q
"""
import ast
import inspect
import os
import sys
import textwrap
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with patch.object(torch.cuda, "current_device", return_value=0):
    from model.dmd_action_forcing import ActionForcingDMD
    from model.forward_noiser import ForwardNoiser

FLAG = "reverse_noiser_dedrift_apply_to_real_target"
SHAPE = (1, 2, 16, 8, 8)


# ===========================================================================
# Lift the shipped block out of the real source
# ===========================================================================
def _kl_grad_ast():
    src = textwrap.dedent(inspect.getsource(ActionForcingDMD._compute_kl_grad))
    return ast.parse(src).body[0]


def _real_target_region_code():
    """Compile the shipped region: the de-drift guard through nan_to_num.

    Start = the ``if`` whose test names the new flag.  End = the
    ``grad = torch.nan_to_num(grad)`` statement.  Everything in between --
    the ``_latest_pred_real_image`` stash, the fingerprint-probe install,
    ``grad = pred_fake_image - pred_real_image`` and the whole normalization
    block -- is the REAL code, so the ordering constraint (de-drift before
    the stash / before the grad) is pinned by construction rather than by a
    paraphrase that could go stale.
    """
    fn = _kl_grad_ast()
    start = None
    for i, st in enumerate(fn.body):
        if not isinstance(st, ast.If):
            continue
        consts = {
            n.value for n in ast.walk(st.test)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        }
        if FLAG in consts:
            start = i
            break
    assert start is not None, (
        f"no top-level `if` in _compute_kl_grad gates on {FLAG!r} -- the "
        "real-target de-drift block moved or was removed."
    )
    end = None
    for j in range(start + 1, len(fn.body)):
        st = fn.body[j]
        if (isinstance(st, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "grad"
                        for t in st.targets)
                and isinstance(st.value, ast.Call)
                and getattr(st.value.func, "attr", None) == "nan_to_num"):
            end = j
            break
    assert end is not None, (
        "`grad = torch.nan_to_num(grad)` not found after the de-drift block."
    )
    body = fn.body[start:end + 1]

    # The region must still contain, IN THIS ORDER: the de-drift call, the
    # stash, and the grad formation.  If someone reorders them the lift
    # would silently keep passing, so assert the order here.
    seen = []
    for k, st in enumerate(body):
        src_st = ast.dump(st)
        if "_dedrift_with_reverse_noiser" in src_st:
            seen.append(("dedrift", k))
        if "_latest_pred_real_image" in src_st:
            seen.append(("stash", k))
        if (isinstance(st, ast.Assign) and isinstance(st.value, ast.BinOp)
                and isinstance(st.value.op, ast.Sub)):
            seen.append(("grad", k))
    order = [n for n, _ in seen]
    assert order[:3] == ["dedrift", "stash", "grad"], (
        f"unexpected statement order in the lifted region: {order}. The "
        "de-drift MUST precede both the stash and the grad formation."
    )

    mod = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(mod)
    return compile(mod, filename="<kl_grad_real_target_region>", mode="exec")


REGION = _real_target_region_code()


# ===========================================================================
# Minimal stub `self` -- only what the lifted region reads
# ===========================================================================
class _Stub:
    """Everything the region touches, and nothing else."""

    # The two methods the region calls are the REAL ones (de-drift) or an
    # inert stand-in (the fingerprint-probe installer, which is orthogonal).
    _dedrift_with_reverse_noiser = (
        ActionForcingDMD._dedrift_with_reverse_noiser
    )

    def _install_dmd_fp_denoise_fn(self, *a, **kw):
        self._fp_installed = True

    def __init__(self, **kw):
        # de-drift knobs (defaults mirror ActionForcingDMD.__init__)
        setattr(self, FLAG, False)
        self.reverse_noiser_dedrift_enabled = False
        self.reverse_noiser_dedrift_level = 1
        self.reverse_noiser_dedrift_min_level = 1
        self.reverse_noiser_dedrift_steps = 1
        self.reverse_noiser_dedrift_alpha0 = 1.0
        self.reverse_noiser_dedrift_alpha_decay = 0.5
        self.fn_pair_mode = "r1_vs_r2"
        self.forward_noiser_cycle_enabled = False
        self.forward_noiser = None
        self.reverse_noiser = None
        # normalization knobs
        self.dmd_normalization_enabled = True
        self.dmd_normalization_band_local = False
        self.dmd_normalization_denom_floor = 0.05
        # stash
        self._latest_pred_real_image = None
        self._fp_installed = False
        for k, v in kw.items():
            setattr(self, k, v)


def _noiser(seed: int = 0, scale: float = 0.05) -> ForwardNoiser:
    """A NON-identity noiser.

    ``ForwardNoiser`` zero-inits ``out_proj`` so a freshly built one returns
    delta == 0 and the de-drift is a no-op -- which would make every
    "on-path" assertion below vacuously true.  Perturb ``out_proj`` so the
    operator actually moves the tensor.
    """
    torch.manual_seed(seed)
    n = ForwardNoiser(
        latent_channels=SHAPE[2], hidden_dim=16, num_blocks=1,
        max_carn_step=4,
    )
    with torch.no_grad():
        n.out_proj.weight.normal_(0.0, scale)
        n.out_proj.bias.normal_(0.0, scale)
    return n.eval()


def _tensors(requires_grad: bool = False):
    torch.manual_seed(1234)
    pred_real = torch.randn(*SHAPE)
    leaf = torch.randn(*SHAPE, requires_grad=requires_grad)
    # pred_fake is graph-connected to `leaf` when requires_grad is set.
    pred_fake = leaf * 2.0
    est_clean = torch.randn(*SHAPE)
    return pred_real, pred_fake, est_clean, leaf


def _run(stub, pred_real, pred_fake, est_clean, normalization=True):
    """Execute the shipped region with the stub as ``self``."""
    g = {"torch": torch}
    loc = {
        "self": stub,
        "pred_real_image": pred_real,
        "pred_fake_image": pred_fake,
        "estimated_clean_image_or_video": est_clean,
        "timestep": torch.zeros(SHAPE[0], SHAPE[1], dtype=torch.long),
        "conditional_dict": {},
        "tf_kwargs_real": {},
        "normalization": normalization,
        "gradient_mask": torch.ones(*SHAPE, dtype=torch.bool),
    }
    exec(REGION, g, loc)
    return loc


# ===========================================================================
# 1. Registration + default
# ===========================================================================
def test_flag_is_registered_and_defaults_false():
    src = inspect.getsource(ActionForcingDMD.__init__)
    tree = ast.parse(textwrap.dedent(src)).body[0]
    found = None
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign)
                and any(getattr(t, "attr", None) == FLAG
                        for t in node.targets)):
            continue
        found = node
    assert found is not None, f"{FLAG} is not registered in __init__."
    consts = [
        n.value for n in ast.walk(found.value) if isinstance(n, ast.Constant)
    ]
    assert FLAG in consts, f"{FLAG} must be read off `args` by name."
    assert False in consts, f"{FLAG} must default to False (byte-identical)."


# ===========================================================================
# 2. Default-off byte identity (+ mutation controls)
# ===========================================================================
def test_off_leaves_pred_real_the_same_object():
    pr, pf, ec, _ = _tensors()
    out = _run(_Stub(), pr, pf, ec)
    assert out["pred_real_image"] is pr, (
        "flag off must leave pred_real_image untouched (same object)."
    )


def test_off_grad_is_bitwise_the_pre_change_value():
    pr, pf, ec, _ = _tensors()
    out = _run(_Stub(), pr, pf, ec)
    # The pre-change value, recomputed independently.
    ref = (pf - pr) / (ec - pr).abs().mean(
        dim=[1, 2, 3, 4], keepdim=True,
    ).clamp_min(0.05)
    ref = torch.nan_to_num(ref)
    assert torch.equal(out["grad"], ref)


def test_apply_flag_on_but_dedrift_disabled_is_identity():
    """The helper's OWN gate. Half-enabled must be a no-op, not a partial."""
    pr, pf, ec, _ = _tensors()
    stub = _Stub(**{FLAG: True}, reverse_noiser_dedrift_enabled=False,
                 forward_noiser_cycle_enabled=True, reverse_noiser=_noiser())
    out = _run(stub, pr, pf, ec)
    assert out["pred_real_image"] is pr


def test_dedrift_enabled_but_apply_flag_off_is_identity():
    """The call-site gate. Other de-drift consumers (flash / train_chunk)
    must be able to run WITHOUT dragging the real target along."""
    pr, pf, ec, _ = _tensors()
    stub = _Stub(reverse_noiser_dedrift_enabled=True,
                 forward_noiser_cycle_enabled=True, reverse_noiser=_noiser())
    out = _run(stub, pr, pf, ec)
    assert out["pred_real_image"] is pr


def test_mutation_control_both_flags_on_breaks_identity():
    """The identity assertions above are only meaningful if flipping the
    flags actually changes the outcome."""
    pr, pf, ec, _ = _tensors()
    stub = _Stub(**{FLAG: True}, reverse_noiser_dedrift_enabled=True,
                 forward_noiser_cycle_enabled=True, reverse_noiser=_noiser())
    out = _run(stub, pr, pf, ec)
    assert out["pred_real_image"] is not pr
    assert not torch.equal(out["pred_real_image"], pr)


def test_missing_network_is_still_identity():
    """Both flags on but no reverse noiser built -> passthrough, not crash."""
    pr, pf, ec, _ = _tensors()
    stub = _Stub(**{FLAG: True}, reverse_noiser_dedrift_enabled=True,
                 forward_noiser_cycle_enabled=True, reverse_noiser=None)
    out = _run(stub, pr, pf, ec)
    assert out["pred_real_image"] is pr


def test_below_min_level_is_identity():
    pr, pf, ec, _ = _tensors()
    stub = _Stub(**{FLAG: True}, reverse_noiser_dedrift_enabled=True,
                 forward_noiser_cycle_enabled=True, reverse_noiser=_noiser(),
                 reverse_noiser_dedrift_level=0,
                 reverse_noiser_dedrift_min_level=1)
    out = _run(stub, pr, pf, ec)
    assert out["pred_real_image"] is pr


# ===========================================================================
# 3. On-path: the value used downstream IS the de-drifted one
# ===========================================================================
def _on_stub(**kw):
    return _Stub(**{FLAG: True}, reverse_noiser_dedrift_enabled=True,
                 forward_noiser_cycle_enabled=True, reverse_noiser=_noiser(),
                 **kw)


def test_on_path_equals_direct_helper_call_cycle_mode():
    pr, pf, ec, _ = _tensors()
    stub = _on_stub()
    with torch.no_grad():
        expect = ActionForcingDMD._dedrift_with_reverse_noiser(stub, pr, 1)
    out = _run(stub, pr, pf, ec)
    assert torch.equal(out["pred_real_image"], expect)
    assert not torch.equal(out["pred_real_image"], pr)


def test_on_path_equals_direct_helper_call_rollout_to_gt_mode():
    """``fn_pair_mode='rollout_to_gt'`` selects the FORWARD noiser as the
    corrector; the call site must not care which net the helper picks."""
    pr, pf, ec, _ = _tensors()
    stub = _Stub(**{FLAG: True}, reverse_noiser_dedrift_enabled=True,
                 fn_pair_mode="rollout_to_gt", forward_noiser=_noiser(seed=7))
    ref = _Stub(**{FLAG: True}, reverse_noiser_dedrift_enabled=True,
                fn_pair_mode="rollout_to_gt",
                forward_noiser=stub.forward_noiser)
    with torch.no_grad():
        expect = ActionForcingDMD._dedrift_with_reverse_noiser(ref, pr, 1)
    out = _run(stub, pr, pf, ec)
    assert torch.equal(out["pred_real_image"], expect)
    assert not torch.equal(out["pred_real_image"], pr)


def test_level_comes_from_the_shared_flat_knob():
    """Both call sites (flash + real target) read the SAME flat
    ``reverse_noiser_dedrift_level``; nothing is derived per sample here."""
    pr, pf, ec, _ = _tensors()
    stub = _on_stub(reverse_noiser_dedrift_level=3,
                    reverse_noiser_dedrift_steps=2)
    with torch.no_grad():
        expect3 = ActionForcingDMD._dedrift_with_reverse_noiser(stub, pr, 3)
        expect1 = ActionForcingDMD._dedrift_with_reverse_noiser(stub, pr, 1)
    out = _run(stub, pr, pf, ec)
    assert torch.equal(out["pred_real_image"], expect3)
    assert not torch.equal(expect3, expect1), (
        "the fixture's noiser is level-insensitive, so this test proves "
        "nothing -- pick a different seed/scale."
    )


def test_stash_holds_the_dedrifted_value_not_the_raw_teacher_output():
    """SIGN-OFF ITEM: ``_latest_pred_real_image`` (and the
    ``pred_real_image_detached`` returned to the caller, which feeds the MAE
    gate / manifold gate / real_score_mae_vs_gt) now carries the DE-DRIFTED
    target when the flag is on -- i.e. the value the gradient is actually
    formed from."""
    pr, pf, ec, _ = _tensors()
    stub = _on_stub()
    out = _run(stub, pr, pf, ec)
    assert stub._latest_pred_real_image is not None
    assert torch.equal(stub._latest_pred_real_image, out["pred_real_image"])
    assert not torch.equal(stub._latest_pred_real_image, pr)
    assert not stub._latest_pred_real_image.requires_grad


def test_grad_and_normalizer_both_use_the_dedrifted_target():
    pr, pf, ec, _ = _tensors()
    stub = _on_stub()
    out = _run(stub, pr, pf, ec)
    dd = out["pred_real_image"]
    expect = pf - dd
    expect = expect / (ec - dd).abs().mean(
        dim=[1, 2, 3, 4], keepdim=True,
    ).clamp_min(0.05)
    expect = torch.nan_to_num(expect)
    assert torch.equal(out["grad"], expect)
    # ...and the normalizer really did move (otherwise the assertion above
    # would also hold for a raw-target normalizer).
    n_dd = (ec - dd).abs().mean()
    n_raw = (ec - pr).abs().mean()
    assert not torch.equal(n_dd, n_raw)


def test_band_local_normalizer_also_uses_the_dedrifted_target():
    pr, pf, ec, _ = _tensors()
    stub = _on_stub(dmd_normalization_band_local=True)
    out = _run(stub, pr, pf, ec)
    dd = out["pred_real_image"]
    m = torch.ones(*SHAPE)
    num = ((ec - dd).abs() * m).sum(dim=[1, 2, 3, 4], keepdim=True)
    den = m.sum(dim=[1, 2, 3, 4], keepdim=True).clamp_min(1.0)
    expect = torch.nan_to_num((pf - dd) / (num / den).clamp_min(0.05))
    assert torch.equal(out["grad"], expect)


# ===========================================================================
# 4. Gradient-flow guard (boundary_vae_roundtrip regression)
# ===========================================================================
def test_generator_gradient_path_survives_the_new_call_site():
    """The bug class: a ``no_grad`` one statement too wide silently severs
    the DMD gradient. The de-drift is wrapped in ``no_grad``; the grad
    formation after it must still build a graph back to the student."""
    pr, pf, ec, leaf = _tensors(requires_grad=True)
    stub = _on_stub()
    out = _run(stub, pr, pf, ec)
    grad = out["grad"]
    assert grad.requires_grad, (
        "grad lost its autograd connection -- the new no_grad block is too "
        "wide (this is the boundary_vae_roundtrip failure mode)."
    )
    grad.sum().backward()
    assert leaf.grad is not None
    assert torch.isfinite(leaf.grad).all()
    assert float(leaf.grad.abs().sum()) > 0.0


def test_gradient_path_is_identical_with_the_flag_off():
    """Control: the same backward with the flag off must also reach the leaf
    (so the assertion above is about the flag, not about the harness)."""
    pr, pf, ec, leaf = _tensors(requires_grad=True)
    out = _run(_Stub(), pr, pf, ec)
    assert out["grad"].requires_grad
    out["grad"].sum().backward()
    assert float(leaf.grad.abs().sum()) > 0.0


def test_dedrifted_target_stays_a_constant():
    """``pred_real_image`` is a TARGET, never a gradient carrier: the real
    caller runs this whole method under ``torch.no_grad()`` and
    ``_dmd_loss_with_fp_gate`` consumes ``grad`` detached. The de-drift must
    not start a graph on the target side (it would keep a ~30M-param G
    forward alive for nothing)."""
    pr, pf, ec, _ = _tensors(requires_grad=True)
    stub = _on_stub()
    out = _run(stub, pr, pf, ec)
    assert not out["pred_real_image"].requires_grad
    assert out["pred_real_image"].grad_fn is None


def test_works_under_the_callers_outer_no_grad():
    """Production shape: ``compute_distribution_matching_loss`` wraps the
    whole ``_compute_kl_grad`` call in ``torch.no_grad()``."""
    pr, pf, ec, _ = _tensors()
    stub = _on_stub()
    with torch.no_grad():
        out = _run(stub, pr, pf, ec)
    assert not out["grad"].requires_grad
    assert not torch.equal(out["pred_real_image"], pr)
    assert torch.isfinite(out["grad"]).all()


def test_noiser_requires_grad_flags_are_restored():
    pr, pf, ec, _ = _tensors()
    net = _noiser()
    for p in net.parameters():
        p.requires_grad_(True)
    stub = _on_stub()
    stub.reverse_noiser = net
    _run(stub, pr, pf, ec)
    assert all(p.requires_grad for p in net.parameters()), (
        "the de-drift left the noiser's params frozen -- its optimizer would "
        "silently stop training."
    )
