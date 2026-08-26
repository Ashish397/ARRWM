"""Coverage for the 2026-08-25 train/infer CONTRACT-ALIGNMENT fixes in
``pipeline/action_forcing_training.py``.

Both flags DEFAULT TO TRUE, i.e. they change what the already-queued
8-node jobs do, which is precisely why they need coverage:

  FIX 1 ``flash_dmd_commit_ladder_endpoint``  -- the KV cache is committed
        from the LADDER ENDPOINT (post-finish-denoise ``cache_pred``,
        t~208), what inference commits, instead of the t=60 flash
        prediction. Legacy = flag False.
  FIX 3 ``carn_seam_affine_apply_to_output`` -- the CARN seam correction
        is applied to the EMITTED ``output`` and the ``clean_chunk``
        stash as well as to the KV commit. Legacy = flag False.

What is pinned:

  * ``_contract_flag`` resolution order (attribute > env var > default),
    every falsy env spelling, and that the default really is the ALIGNED
    value for both names;
  * ``_carn_seam_correct`` is a PROVABLE no-op with every CARN knob at
    its default -- it returns the SAME tensor object, so ``grad_fn`` and
    identity both survive -- and specifically at ``carn_seam_affine_
    lambda == 0`` even with a target published;
  * the seam maths itself (temperature, drift, affine at lambda=1) and
    the 2026-08-25 ``carn_seam_affine_match_mean`` default flip, with the
    legacy value reproducing the mean+std blend;
  * seam telemetry publishes on the ``record=True`` (KV-commit) site only
    and can be switched off;
  * the two FIX sites, in BOTH rollout methods, are pinned against the
    REAL SOURCE: the flag's conditional expression and the ``if
    _carn_out:`` guard are lifted out of the parsed AST and EXECUTED
    here, so the legacy flag value is shown to reproduce the pre-fix
    expression (``cache_pred``/untouched ``output_pred``) using the
    shipped code rather than a copy of it. The ``ladder_endpoint_pred =
    cache_pred`` stash is pinned to occur BEFORE the flash reassignment
    that overwrites ``cache_pred`` -- if it moved after, FIX 1 would
    silently become a no-op.

CPU-only. ``pipeline.action_forcing_training`` pulls in ``wan.modules.t5``
(which evaluates ``torch.cuda.current_device()`` at import), so the
import runs under the usual patch. No model is constructed: the two
methods under test are called UNBOUND-style on an instance made with
``__new__``.

Run:
    python -m pytest testing/test_contract_flag_defaults.py -q
or
    python testing/test_contract_flag_defaults.py
"""
import ast
import copy
import os
import sys
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with patch.object(torch.cuda, "current_device", return_value=0):
    import pipeline.action_forcing_training as AFT  # noqa: E402

PIPE = AFT.ActionForcingTrainingPipeline
SRC_PATH = AFT.__file__.replace(".pyc", ".py")
with open(SRC_PATH) as _f:
    SRC = _f.read()
TREE = ast.parse(SRC)

FLAG_LADDER = ("flash_dmd_commit_ladder_endpoint",
               "FLASH_DMD_COMMIT_LADDER_ENDPOINT")
FLAG_CARN = ("carn_seam_affine_apply_to_output",
             "CARN_SEAM_AFFINE_APPLY_TO_OUTPUT")
ROLLOUTS = ("inference_with_trajectory", "generate_chunk_with_cache")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _pipe(**attrs):
    """A pipeline instance without ``__init__`` -- class attributes
    (``_FALSEY``) and the real methods, none of the 1.3B machinery."""
    p = PIPE.__new__(PIPE)
    for k, v in attrs.items():
        setattr(p, k, v)
    return p


class _Env:
    """Context manager that sets/clears env vars and always restores."""

    def __init__(self, **kv):
        self.kv = kv
        self.old = {}

    def __enter__(self):
        for k, v in self.kv.items():
            self.old[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return self

    def __exit__(self, *a):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return False


def _method(name):
    cls = next(
        n for n in TREE.body
        if isinstance(n, ast.ClassDef) and n.name == PIPE.__name__
    )
    fns = [n for n in cls.body
           if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
           and n.name == name]
    assert len(fns) == 1, f"{name}: found {len(fns)} definitions"
    return fns[0]


def _assigns_to(fn, target):
    return [
        n for n in ast.walk(fn)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == target
                for t in n.targets)
    ]


def _eval_node(node, ns):
    e = ast.Expression(body=copy.deepcopy(node))
    ast.fix_missing_locations(e)
    return eval(compile(e, "<pipeline-expr>", "eval"), {}, ns)


def _exec_node(node, ns):
    m = ast.Module(body=[copy.deepcopy(node)], type_ignores=[])
    ast.fix_missing_locations(m)
    exec(compile(m, "<pipeline-stmt>", "exec"), ns)
    return ns


def _is_contract_flag_call(node, names):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_contract_flag"
        and [getattr(a, "value", None) for a in node.args] == list(names)
    )


# ===========================================================================
# _contract_flag resolution
# ===========================================================================
def test_both_flags_default_to_the_aligned_value():
    """No attribute, no env var -> TRUE for both. This is the behaviour
    the queued jobs will actually run."""
    p = _pipe()
    with _Env(**{FLAG_LADDER[1]: None, FLAG_CARN[1]: None}):
        assert p._contract_flag(*FLAG_LADDER) is True
        assert p._contract_flag(*FLAG_CARN) is True


def test_env_var_falsey_spellings_restore_legacy():
    p = _pipe()
    for v in ("0", "false", "FALSE", "no", "off", "", "  Off  "):
        with _Env(**{FLAG_LADDER[1]: v}):
            assert p._contract_flag(*FLAG_LADDER) is False, repr(v)


def test_env_var_truthy_spellings_keep_the_fix():
    p = _pipe()
    for v in ("1", "true", "yes", "ON", "anything"):
        with _Env(**{FLAG_CARN[1]: v}):
            assert p._contract_flag(*FLAG_CARN) is True, repr(v)


def test_attribute_beats_the_env_var_both_ways():
    with _Env(**{FLAG_LADDER[1]: "0"}):
        assert _pipe(**{FLAG_LADDER[0]: True})._contract_flag(
            *FLAG_LADDER) is True
    with _Env(**{FLAG_LADDER[1]: "1"}):
        assert _pipe(**{FLAG_LADDER[0]: False})._contract_flag(
            *FLAG_LADDER) is False


def test_attribute_none_falls_through_to_env_then_default():
    p = _pipe(**{FLAG_CARN[0]: None})
    with _Env(**{FLAG_CARN[1]: "0"}):
        assert p._contract_flag(*FLAG_CARN) is False
    with _Env(**{FLAG_CARN[1]: None}):
        assert p._contract_flag(*FLAG_CARN) is True


def test_explicit_default_is_honoured():
    p = _pipe()
    with _Env(NOT_A_REAL_FLAG_FOR_TESTS=None):
        assert p._contract_flag("nope", "NOT_A_REAL_FLAG_FOR_TESTS",
                                default=False) is False
        assert p._contract_flag("nope", "NOT_A_REAL_FLAG_FOR_TESTS") is True


def test_falsey_table_is_what_the_docstring_claims():
    assert set(PIPE._FALSEY) == {"0", "false", "no", "off", ""}


# ===========================================================================
# _carn_seam_correct -- the provable no-op
# ===========================================================================
def _x(seed=0, requires_grad=True):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(2, 3, 4, 5, 5, generator=g)
    if requires_grad:
        x = (x * 1.0).requires_grad_(True)
        x = x * 2.0            # give it a real grad_fn
    return x


def test_seam_is_identity_with_every_knob_at_its_default():
    p = _pipe()
    x = _x()
    out = p._carn_seam_correct(x)
    assert out is x, "the default path must not even copy the tensor"
    assert out.grad_fn is x.grad_fn
    assert torch.equal(out, x)


def test_seam_is_identity_at_lambda_zero_even_with_a_target():
    """lambda == 0 is the documented "byte-identical" setting. A target is
    published but must not be consulted."""
    p = _pipe(
        carn_seam_affine_lambda=0.0,
        _carn_seam_target=(torch.full((4,), 9.0), torch.full((4,), 9.0)),
    )
    x = _x()
    out = p._carn_seam_correct(x)
    assert out is x
    assert out.grad_fn is x.grad_fn
    assert torch.equal(out, x)


def test_seam_is_identity_when_the_target_is_missing():
    p = _pipe(carn_seam_affine_lambda=0.7, _carn_seam_target=None)
    x = _x()
    assert p._carn_seam_correct(x) is x


def test_seam_none_knobs_read_as_off():
    """``float(getattr(...) or default)`` -- YAML nulls must not crash."""
    p = _pipe(
        carn_seam_temp=None,
        carn_seam_drift_lambda=None,
        carn_seam_affine_lambda=None,
    )
    x = _x()
    assert p._carn_seam_correct(x) is x


# ===========================================================================
# _carn_seam_correct -- the maths, when it IS live
# ===========================================================================
def test_temperature_reinflates_deviations_exactly():
    T = 1.09
    p = _pipe(carn_seam_temp=T)
    x = _x(requires_grad=False)
    out = p._carn_seam_correct(x)
    mu = x.float().mean(dim=(0, 1, 3, 4), keepdim=True)
    assert torch.allclose(out, (mu + T * (x.float() - mu)), atol=1e-6)
    assert out.dtype == x.dtype
    # variance really went UP (that is the point of the knob)
    assert float(out.std()) > float(x.std())


def test_drift_subtracts_lambda_times_the_channel_vector():
    vec = torch.tensor([0.1, -0.2, 0.3, 0.0])
    p = _pipe(carn_seam_drift_lambda=0.5, _carn_seam_drift_vec=vec)
    x = _x(requires_grad=False)
    out = p._carn_seam_correct(x)
    want = x.float() - 0.5 * vec.view(1, 1, -1, 1, 1)
    assert torch.allclose(out, want, atol=1e-6)


def test_drift_is_off_without_a_published_vector():
    p = _pipe(carn_seam_drift_lambda=0.5, _carn_seam_drift_vec=None)
    x = _x()
    assert p._carn_seam_correct(x) is x


def test_affine_lambda_one_pins_the_std_and_keeps_the_mean():
    """Default (post-2026-08-25) behaviour: SCALE re-anchored, each chunk
    KEEPS ITS OWN per-channel mean."""
    tgt_std = torch.tensor([0.5, 1.5, 2.5, 3.5])
    tgt_mean = torch.tensor([9.0, -9.0, 4.0, -4.0])
    p = _pipe(
        carn_seam_affine_lambda=1.0,
        _carn_seam_target=(tgt_mean, tgt_std),
        _carn_seam_affine_logged=True,
    )
    x = _x(requires_grad=False)
    out = p._carn_seam_correct(x)
    got_std = out.float().std(dim=(0, 1, 3, 4), unbiased=True)
    got_mean = out.float().mean(dim=(0, 1, 3, 4))
    src_mean = x.float().mean(dim=(0, 1, 3, 4))
    assert torch.allclose(got_std, tgt_std, atol=1e-3)
    assert torch.allclose(got_mean, src_mean, atol=1e-4), (
        "mean matching is OFF by default -- the chunk's own mean must "
        "survive (forward motion is what moves it)")
    assert not torch.allclose(got_mean, tgt_mean, atol=1e-2)


def test_legacy_match_mean_true_reproduces_the_mean_plus_std_blend():
    tgt_std = torch.tensor([0.5, 1.5, 2.5, 3.5])
    tgt_mean = torch.tensor([9.0, -9.0, 4.0, -4.0])
    p = _pipe(
        carn_seam_affine_lambda=1.0,
        carn_seam_affine_match_mean=True,
        _carn_seam_target=(tgt_mean, tgt_std),
        _carn_seam_affine_logged=True,
    )
    x = _x(requires_grad=False)
    out = p._carn_seam_correct(x)
    assert torch.allclose(
        out.float().mean(dim=(0, 1, 3, 4)), tgt_mean, atol=1e-4)
    assert torch.allclose(
        out.float().std(dim=(0, 1, 3, 4), unbiased=True), tgt_std, atol=1e-3)


def test_the_two_match_mean_settings_actually_differ():
    tgt = (torch.tensor([9.0, -9.0, 4.0, -4.0]),
           torch.tensor([0.5, 1.5, 2.5, 3.5]))
    x = _x(requires_grad=False)
    a = _pipe(carn_seam_affine_lambda=0.5, _carn_seam_target=tgt,
              _carn_seam_affine_logged=True)._carn_seam_correct(x)
    b = _pipe(carn_seam_affine_lambda=0.5, _carn_seam_target=tgt,
              carn_seam_affine_match_mean=True,
              _carn_seam_affine_logged=True)._carn_seam_correct(x)
    assert not torch.allclose(a, b)


def test_seam_preserves_dtype_and_the_autograd_graph():
    p = _pipe(
        carn_seam_temp=1.05,
        carn_seam_affine_lambda=0.5,
        _carn_seam_target=(torch.zeros(4), torch.ones(4)),
        _carn_seam_affine_logged=True,
    )
    x = _x()
    out = p._carn_seam_correct(x)
    assert out.dtype == x.dtype
    assert out.grad_fn is not None
    out.sum().backward()
    assert torch.isfinite(x.grad).all() if x.grad is not None else True


def test_seam_handles_bf16_without_promoting():
    p = _pipe(carn_seam_temp=1.2)
    x = _x(requires_grad=False).to(torch.bfloat16)
    out = p._carn_seam_correct(x)
    assert out.dtype == torch.bfloat16


# ===========================================================================
# seam telemetry
# ===========================================================================
def test_record_publishes_seam_telemetry_on_the_commit_site_only():
    tgt = (torch.zeros(4), torch.ones(4) * 2.0)
    metrics = {}
    p = _pipe(
        carn_seam_affine_lambda=0.5, _carn_seam_target=tgt,
        _carn_seam_affine_logged=True, _last_extension_metrics=metrics,
    )
    x = _x(requires_grad=False)
    p._carn_seam_correct(x)                     # record=False (output site)
    assert metrics == {}, "the emit site must not publish seam telemetry"
    p._carn_seam_correct(x, record=True)        # record=True (KV commit)
    for k in ("carn_seam_gain", "carn_seam_mu_shift",
              "carn_commit_std", "carn_pred_std", "carn_seam_blocks"):
        assert k in metrics, k
    assert metrics["carn_seam_blocks"] == 1.0


def test_telemetry_averages_over_blocks():
    tgt = (torch.zeros(4), torch.ones(4) * 2.0)
    metrics = {}
    p = _pipe(
        carn_seam_affine_lambda=0.5, _carn_seam_target=tgt,
        _carn_seam_affine_logged=True, _last_extension_metrics=metrics,
    )
    for i in range(3):
        p._carn_seam_correct(_x(seed=i, requires_grad=False), record=True)
    assert metrics["carn_seam_blocks"] == 3.0
    assert 0.0 < metrics["carn_pred_std"] < 10.0


def test_telemetry_can_be_switched_off():
    tgt = (torch.zeros(4), torch.ones(4) * 2.0)
    metrics = {}
    p = _pipe(
        carn_seam_affine_lambda=0.5, _carn_seam_target=tgt,
        _carn_seam_affine_logged=True, _last_extension_metrics=metrics,
        carn_seam_telemetry=False,
    )
    p._carn_seam_correct(_x(requires_grad=False), record=True)
    assert metrics == {}


def test_telemetry_never_takes_a_step_down():
    """``_last_extension_metrics`` absent / wrong type must be swallowed."""
    tgt = (torch.zeros(4), torch.ones(4) * 2.0)
    p = _pipe(carn_seam_affine_lambda=0.5, _carn_seam_target=tgt,
              _carn_seam_affine_logged=True)
    p._carn_seam_correct(_x(requires_grad=False), record=True)   # no attr
    p2 = _pipe(carn_seam_affine_lambda=0.5, _carn_seam_target=tgt,
               _carn_seam_affine_logged=True,
               _last_extension_metrics=object())
    p2._carn_seam_correct(_x(requires_grad=False), record=True)


# ===========================================================================
# FIX 1 -- the commit source, executed from the real source tree
# ===========================================================================
def _commit_src_ifexp(method):
    """The ``_commit_src = <ladder> if <flag> else <cache_pred>`` node,
    plus the extra bindings its test needs.

    The flag is allowed to be read INLINE (``... if self._contract_flag(...)
    else ...``) or hoisted into a local first (``_use_ladder =
    self._contract_flag(...)``) -- both spellings appear in this tree's
    history and both are correct as long as the hoisted value is exactly
    the same ``_contract_flag`` call. Anything else fails."""
    fn = _method(method)
    assigns = _assigns_to(fn, "_commit_src")
    assert len(assigns) == 1, f"{method}: {len(assigns)} _commit_src assigns"
    node = assigns[0].value
    assert isinstance(node, ast.IfExp), (
        f"{method}: _commit_src is no longer a conditional expression")

    if _is_contract_flag_call(node.test, FLAG_LADDER):
        return node, {}
    assert isinstance(node.test, ast.Name), (
        f"{method}: _commit_src's test is neither the _contract_flag call "
        f"nor a local bound to it ({ast.dump(node.test)[:80]})")
    src = [
        a for a in _assigns_to(fn, node.test.id)
        if _is_contract_flag_call(a.value, FLAG_LADDER)
    ]
    assert len(src) == 1, (
        f"{method}: local {node.test.id!r} gating the commit source is not "
        f"bound to self._contract_flag{FLAG_LADDER}")
    return node, {node.test.id: src[0].value}


@pytest.mark.parametrize("method", ROLLOUTS)
def test_fix1_expression_selects_ladder_endpoint_by_default(method):
    """The REAL conditional expression is lifted out of the shipped source
    and evaluated. Default (flag unset) must pick the ladder endpoint;
    the legacy flag value must reproduce the pre-fix ``cache_pred``."""
    node, extra = _commit_src_ifexp(method)
    assert isinstance(node.body, ast.Name)
    assert isinstance(node.orelse, ast.Name)
    assert (node.body.id, node.orelse.id) == (
        "ladder_endpoint_pred", "cache_pred"), (
        f"{method}: branches are ({node.body.id}, {node.orelse.id}) -- "
        "the aligned tensor must be the TRUE branch")

    ladder = torch.full((1,), 208.0)
    flash = torch.full((1,), 60.0)

    def _run(pipe):
        ns = dict(ladder_endpoint_pred=ladder, cache_pred=flash, self=pipe)
        for name, val in extra.items():          # evaluate the real hoist
            ns[name] = _eval_node(val, dict(ns))
        return _eval_node(node, ns)

    with _Env(**{FLAG_LADDER[1]: None}):
        assert _run(_pipe()) is ladder, "default must commit the LADDER ENDPOINT"
        assert _run(_pipe(**{FLAG_LADDER[0]: False})) is flash, (
            "flag False must reproduce the pre-fix expression (cache_pred, "
            "i.e. the t=60 flash prediction)")

    with _Env(**{FLAG_LADDER[1]: "0"}):
        assert _run(_pipe()) is flash


@pytest.mark.parametrize("method", ROLLOUTS)
def test_fix1_stash_happens_before_the_flash_reassignment(method):
    """``ladder_endpoint_pred = cache_pred`` must be taken BEFORE
    ``cache_pred = flash_dmd_pred.detach()``. If it moved after, FIX 1
    would alias the flash tensor and become a silent no-op."""
    fn = _method(method)
    stash = _assigns_to(fn, "ladder_endpoint_pred")
    assert len(stash) == 1, f"{method}: {len(stash)} stash assignments"
    assert isinstance(stash[0].value, ast.Name)
    assert stash[0].value.id == "cache_pred"

    reassign = [
        a for a in _assigns_to(fn, "cache_pred")
        if isinstance(a.value, ast.Call)
        and isinstance(a.value.func, ast.Attribute)
        and a.value.func.attr == "detach"
        and isinstance(a.value.func.value, ast.Name)
        and a.value.func.value.id == "flash_dmd_pred"
    ]
    assert len(reassign) == 1, (
        f"{method}: expected exactly one "
        f"``cache_pred = flash_dmd_pred.detach()``, found {len(reassign)}")
    assert stash[0].lineno < reassign[0].lineno, (
        f"{method}: the ladder-endpoint stash (line {stash[0].lineno}) is "
        f"NOT before the flash reassignment (line {reassign[0].lineno})")


@pytest.mark.parametrize("method", ROLLOUTS)
def test_fix1_commit_input_is_the_detached_selected_source(method):
    fn = _method(method)
    a = _assigns_to(fn, "commit_input_clean")
    srcs = [
        n for n in a
        if isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Attribute)
        and n.value.func.attr == "detach"
        and isinstance(n.value.func.value, ast.Name)
        and n.value.func.value.id == "_commit_src"
    ]
    assert len(srcs) == 1, (
        f"{method}: commit_input_clean no longer reads _commit_src.detach()")


# ===========================================================================
# FIX 3 -- the seam-on-output guard, executed from the real source tree
# ===========================================================================
def _carn_out_guards(method):
    fn = _method(method)
    flag = [
        n for n in _assigns_to(fn, "_carn_out")
        if _is_contract_flag_call(n.value, FLAG_CARN)
    ]
    assert len(flag) == 1, (
        f"{method}: expected one ``_carn_out = self._contract_flag"
        f"{FLAG_CARN}``, found {len(flag)}")
    guards = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Name) and n.test.id == "_carn_out"
    ]
    assert guards, f"{method}: no ``if _carn_out:`` guard"
    return flag[0], guards


@pytest.mark.parametrize("method", ROLLOUTS)
def test_fix3_output_guard_executes_both_ways(method):
    """Execute the shipped ``if _carn_out:`` guard. Legacy (False) must
    leave the emitted tensor untouched -- the same object, so the pre-fix
    behaviour is bit-identical by construction."""
    _flag, guards = _carn_out_guards(method)
    out_guards = [
        g for g in guards
        if any(isinstance(t, ast.Name) and t.id == "output_pred"
               for s in g.body if isinstance(s, ast.Assign)
               for t in s.targets)
    ]
    assert len(out_guards) == 1, (
        f"{method}: expected one output_pred seam guard")
    g = out_guards[0]

    seam_pipe = _pipe(
        carn_seam_temp=1.25, _carn_seam_affine_logged=True)
    x = _x(requires_grad=False)

    ns_off = {"_carn_out": False, "output_pred": x, "self": seam_pipe}
    _exec_node(g, ns_off)
    assert ns_off["output_pred"] is x, (
        "legacy flag value must leave the emitted output untouched")

    ns_on = {"_carn_out": True, "output_pred": x, "self": seam_pipe}
    _exec_node(g, ns_on)
    assert ns_on["output_pred"] is not x
    assert torch.allclose(
        ns_on["output_pred"], seam_pipe._carn_seam_correct(x))


def test_fix3_also_guards_the_clean_chunk_stash():
    """``inference_with_trajectory`` stashes the aux teacher's clean half;
    the aligned behaviour corrects that too."""
    _flag, guards = _carn_out_guards("inference_with_trajectory")
    cc = [
        g for g in guards
        if any(isinstance(t, ast.Name) and t.id == "_cc"
               for s in g.body if isinstance(s, ast.Assign)
               for t in s.targets)
    ]
    assert len(cc) == 1, "the clean_chunk stash lost its seam correction"

    seam_pipe = _pipe(carn_seam_temp=1.25, _carn_seam_affine_logged=True)
    x = _x(requires_grad=False)
    ns = {"_carn_out": False, "_cc": x, "self": seam_pipe}
    _exec_node(cc[0], ns)
    assert ns["_cc"] is x
    ns = {"_carn_out": True, "_cc": x, "self": seam_pipe}
    _exec_node(cc[0], ns)
    assert ns["_cc"] is not x


@pytest.mark.parametrize("method", ROLLOUTS)
def test_the_kv_commit_site_always_gets_the_seam_with_record(method):
    """FIX 3 changed WHO ELSE gets the correction; the KV commit must
    still get it, and it is still the only ``record=True`` caller."""
    fn = _method(method)
    calls = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "_carn_seam_correct"
    ]
    rec = [c for c in calls
           if any(k.arg == "record" for k in c.keywords)]
    assert len(rec) == 1, (
        f"{method}: expected exactly one record=True seam call, got "
        f"{len(rec)}")


# ---------------------------------------------------------------------------
def main():
    g = dict(globals())
    names = [n for n in g if n.startswith("test_")]
    names.sort(key=lambda n: g[n].__code__.co_firstlineno)
    n_run = 0
    for n in names:
        fn = g[n]
        marks = getattr(fn, "pytestmark", [])
        params = [m for m in marks if getattr(m, "name", "") == "parametrize"]
        if params:
            argnames = [a.strip() for a in params[0].args[0].split(",")]
            for vals in params[0].args[1]:
                vals = vals if isinstance(vals, tuple) else (vals,)
                fn(**dict(zip(argnames, vals)))
                n_run += 1
            print(f"  ok  {n}  x{len(params[0].args[1])}")
        else:
            fn()
            n_run += 1
            print(f"  ok  {n}")
    print(f"\nALL {n_run} TEST CASES PASSED")


if __name__ == "__main__":
    main()
