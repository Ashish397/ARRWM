"""The 2026-08-26 ruling: gt_vs_fake positives are CLEAN GT, by default.

THE RULING (researcher, verbatim, binding):

    "gtclean should be the default, we should not be noising the clean data
     that goes into disc ... the gan needs to be separated - the transition
     side of things goes to the CARN model and the gt vs fake is yours.
     Leave the transition gt to be done whatever with but for gt vs fake we
     should never be noising."

So:
  * ``pair_mode == "gt_vs_fake"``   -> the real/positive rows are NEVER put
    through the CARN forward noiser. This is the DEFAULT; no flag required.
  * ``pair_mode == "gt_transition"`` -> UNCHANGED, bit for bit.

WHY IT MATTERS.  The forward-noiser application to the real rows is
moment-preserving (per-channel DC mean + per-sample mean magnitude of the
ORIGINAL are restored), so the ONLY thing it alters is texture STRUCTURE --
exactly the axis ``gt_vs_fake`` exists to judge.  Noising the positives
teaches the disc that CARN-drifted texture IS real footage.

WHAT IS PINNED HERE, and how.  Every assertion is made against code
EXECUTED out of the real source, never against a paraphrase:

  1. ``lift_current`` compiles the shipped apply regions out of the live
     ``_ladd_run_pair_mode`` AST.
  2. ``lift_pre_change`` compiles the SAME regions out of
     ``git show HEAD:trainer/causal_action_forcing_train.py``.  HEAD is a
     valid pre-change reference: the two regions were verified byte-
     identical between HEAD and the pre-edit working tree (there are other
     uncommitted changes in the file, but none inside these regions), and
     ``test_pre_change_golden_is_really_the_old_code`` re-checks that the
     golden lacks the new gate before trusting it.

Then:
  (i)   default gt_vs_fake positives are BIT-IDENTICAL to clean GT;
  (ii)  gt_transition (cpp==2) output is BIT-IDENTICAL to the pre-change
        code on identical inputs -- the exact-match test;
  (iii) ``forward_noiser_allow_gt_vs_fake=true`` reproduces the pre-change
        gt_vs_fake tensor BIT-for-BIT;
  (iv)  ``test_pre_change_code_fails_the_ruling`` runs the OLD code and
        asserts it VIOLATES the ruling -- i.e. this suite genuinely fails
        on the pre-change trainer;
  (v)   the monotone counters and the banner contract.

CPU-only, no CUDA, no dataset.  ``wan/modules/t5.py`` evaluates
``torch.cuda.current_device()`` at import, so imports are patched exactly
as ``testing/test_fn_apply_decoupled.py`` does.

Run:
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" \
        python -m pytest testing/test_fn_gtvf_clean.py -q
"""
import ast
import inspect
import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

with patch.object(torch.cuda, "current_device", return_value=0):
    from trainer import causal_action_forcing_train as CAFT

Trainer = CAFT.ActionForcingDMDTrainer

_TRAINER_REL = "trainer/causal_action_forcing_train.py"


# ===========================================================================
# Source lifting: current tree and the pre-change golden
# ===========================================================================
def _current_source():
    return textwrap.dedent(inspect.getsource(Trainer._ladd_run_pair_mode))


def _pre_change_source():
    """``_ladd_run_pair_mode`` as it stood BEFORE the ruling landed."""
    try:
        blob = subprocess.run(
            ["git", "show", "HEAD:" + _TRAINER_REL],
            cwd=_REPO, capture_output=True, text=True, timeout=120,
        )
    except Exception as exc:                                # pragma: no cover
        pytest.skip("git unavailable for the pre-change golden: %r" % (exc,))
    if blob.returncode != 0:                                # pragma: no cover
        pytest.skip("git show failed: %s" % (blob.stderr.strip(),))
    tree = ast.parse(blob.stdout)
    for node in ast.walk(tree):
        if (isinstance(node, ast.FunctionDef)
                and node.name == "_ladd_run_pair_mode"):
            lines = blob.stdout.split("\n")
            return textwrap.dedent(
                "\n".join(lines[node.lineno - 1:node.end_lineno]))
    pytest.skip("HEAD has no _ladd_run_pair_mode")          # pragma: no cover


def _fn_ast(src):
    return ast.parse(src).body[0]


def _stmt_names(stmt):
    return {n.id for n in ast.walk(stmt) if isinstance(n, ast.Name)}


def _parent_body_containing(fn, pred):
    """Find the statement list holding the first statement matching pred."""
    stack = [fn.body]
    while stack:
        body = stack.pop(0)
        for i, st in enumerate(body):
            if pred(st):
                return body, i
            for field in ("body", "orelse", "finalbody"):
                sub = getattr(st, field, None)
                if isinstance(sub, list) and sub and isinstance(
                        sub[0], ast.stmt):
                    stack.append(sub)
    return None, None


def _compile_slice(body, lo, hi, tag):
    mod = ast.Module(body=body[lo:hi + 1], type_ignores=[])
    ast.fix_missing_locations(mod)
    return compile(mod, "<%s>" % tag, "exec")


def _decoupled_region(src, tag):
    """``_fn_decoupled = ...`` through the statement that applies the FN."""
    fn = _fn_ast(src)
    body, idx = _parent_body_containing(
        fn,
        lambda st: (isinstance(st, ast.Assign)
                    and any(getattr(t, "id", None) == "_fn_decoupled"
                            for t in st.targets)),
    )
    assert idx is not None, "the decoupled region is gone from %s" % tag
    end = None
    for j in range(idx + 1, len(body)):
        st = body[j]
        if isinstance(st, ast.If) and "_dmode" in _stmt_names(st):
            end = j
            break
    assert end is not None, "the decoupled apply body is gone from %s" % tag
    return _compile_slice(body, idx, end, "decoupled_" + tag)


def _matched_region(src, tag):
    """``_gt_both = ...`` through the statement holding ``_chain_app``."""
    fn = _fn_ast(src)
    body, idx = _parent_body_containing(
        fn,
        lambda st: (isinstance(st, ast.Assign)
                    and any(getattr(t, "id", None) == "_gt_both"
                            for t in st.targets)),
    )
    assert idx is not None, "the matched region is gone from %s" % tag
    end = None
    for j in range(idx, len(body)):
        if "_chain_app" in _stmt_names(body[j]):
            end = j
            break
    assert end is not None, "the matched apply body is gone from %s" % tag
    return _compile_slice(body, idx, end, "matched_" + tag)


_CUR_SRC = _current_source()
_DEC_CUR = _decoupled_region(_CUR_SRC, "cur")
_MAT_CUR = _matched_region(_CUR_SRC, "cur")


def _pre_regions():
    src = _pre_change_source()
    return _decoupled_region(src, "pre"), _matched_region(src, "pre")


# ===========================================================================
# Deterministic stand-in for the trained forward noiser (CARN)
# ===========================================================================
class _FN(torch.nn.Module):
    """Adds a fixed zero-mean texture pattern scaled by the cond level.

    Zero-mean and non-constant so it SURVIVES the call site's moment-
    preserving restore -- a pure DC shift would be undone and a real
    application would look like a no-op.

    NON-COMMUTATIVE by construction.  The ``sigmoid(x)`` factor makes
    ``f_a(f_b(x)) != f_b(f_a(x))``, so a change to the ORDER in which the
    chained apply feeds conditioning levels (``range(1, _max_t + 1)``)
    shows up in the OUTPUT TENSOR, not only in the recorded call list.
    An affine stand-in (``x + k*lvl*pat``) is order-BLIND -- the
    increments simply sum -- which is exactly why the first version of
    this suite could not catch a reversed chain loop.
    ``test_stand_in_noiser_is_order_sensitive`` pins this property.
    """

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, x, carn_step, residual=True):
        self.calls.append(
            (tuple(x.shape), carn_step.detach().cpu().tolist(), bool(residual))
        )
        pat = torch.sin(
            torch.arange(x.numel(), dtype=x.dtype).reshape(x.shape) * 0.7)
        lvl = carn_step.to(x.dtype).view(-1, *([1] * (x.dim() - 1)))
        return x + 0.25 * lvl * pat * torch.sigmoid(x)


def _model(*, allow_gtvf=False, decoupled=True, gt_former=True,
           gt_both=False, chain=True, npb=3, seed_clean=3,
           former_mode="weak"):
    """``former_mode=None`` leaves the attribute UNSET, so the shipped
    ``getattr(..., "forward_noiser_former_mode", "weak")`` DEFAULT is the
    thing under test.  Without this, no assertion in the suite can see a
    change to that default."""
    m = SimpleNamespace(
        forward_noiser_allow_gt_vs_fake=bool(allow_gtvf),
        forward_noiser_apply_decoupled=bool(decoupled),
        forward_noiser_apply_gt_former=bool(gt_former),
        forward_noiser_apply_gt_both=bool(gt_both),
        forward_noiser_chain_levels=bool(chain),
        forward_noiser_former_mode=str(former_mode),
        forward_noiser_step_unconditioned=False,
        forward_noiser_apply_gt_level=1,
        forward_noiser_apply_gt_level_max=0,
        forward_noiser_apply_gt_drift_cap=False,
        num_frame_per_block=npb,
        dmd_context_clean_frames=seed_clean * npb,
        forward_noiser=_FN(),
    )
    if former_mode is None:
        delattr(m, "forward_noiser_former_mode")
    return m


def _base_ns(slf, **extra):
    ns = {
        "torch": torch,
        "self": slf,
        "getattr": getattr,
        "bool": bool, "int": int, "str": str, "float": float,
        "max": max, "min": min, "range": range, "len": len,
        "__builtins__": __builtins__,
    }
    ns.update(extra)
    return ns


# --------------------------------------------------------------------------
# DECOUPLED (positional, chunks_per_pair == 1) harness
# --------------------------------------------------------------------------
def _run_decoupled(code, *, pair_mode="gt_vs_fake", allow_gtvf=False,
                   n_pairs=8, npb=3, seed=1234):
    m = _model(allow_gtvf=allow_gtvf, npb=npb)
    slf = SimpleNamespace(model=m, config=SimpleNamespace(),
                          is_main_process=True)
    g = torch.Generator().manual_seed(seed)
    real = torch.randn(n_pairs, npb, 4, 6, 8, generator=g)
    ns = _base_ns(
        slf,
        real_chunks_det=real,
        chunks_per_pair=1,
        n_pairs=n_pairs,
        pairs=[(i, i) for i in range(n_pairs)],
        pair_mode=pair_mode,
    )
    exec(code, ns, ns)
    return ns["real_chunks_det"], real, slf, m.forward_noiser


# --------------------------------------------------------------------------
# MATCHED-POOL harness (this is the site gt_transition actually uses)
# --------------------------------------------------------------------------
def _run_matched(code, *, pair_mode, chunks_per_pair, allow_gtvf=False,
                 n_pairs=4, B=1, Kk=2, n_uniq=6, npb=3, seed=99,
                 former_mode="weak", seed_clean=3):
    """Execute a lifted matched-pool region.

    ``former_mode`` and ``seed_clean`` are threaded all the way into the
    model the region actually reads (``_model`` builds a FRESH namespace
    every call, so setting them on a returned model afterwards is a
    no-op -- that bug made the weak/strong parametrization dead).
    ``seed_clean`` sets ``dmd_context_clean_frames``, which drives
    ``_seed_r1`` and therefore how DEEP the per-row target levels go:
    ``seed_clean=1`` is what reaches ``_max_t >= 3`` and so actually
    exercises the chain ORDERING loop.
    """
    m = _model(allow_gtvf=allow_gtvf, npb=npb, former_mode=former_mode,
               seed_clean=seed_clean)
    slf = SimpleNamespace(model=m, config=SimpleNamespace(),
                          is_main_process=True)
    g = torch.Generator().manual_seed(seed)
    ru = torch.randn(n_uniq, chunks_per_pair * npb, 4, 6, 8, generator=g)
    gm = torch.arange(n_pairs * B * Kk).reshape(n_pairs * B, Kk) % n_uniq
    ns = _base_ns(
        slf,
        ru=ru,
        npb=npb,
        B=B,
        Kk=Kk,
        gm=gm,
        n_pairs=n_pairs,
        pairs=[(i, i + 1) for i in range(n_pairs)],
        rows_bu=[(0, r) for r in range(n_uniq)],
        chunks_per_pair=chunks_per_pair,
        pair_mode=pair_mode,
        _fn_decoupled=True,
        # Computed in the ENCLOSING scope of the real method (outside the
        # lifted matched region), so the harness must supply it. Recomputed
        # here with the SAME expression the source uses -- and
        # ``test_the_gate_predicate_is_pair_mode_keyed_and_default_clean``
        # separately evaluates the shipped expression itself, so a drift
        # between the two would be caught there.
        _fn_block_gtvf=((pair_mode == "gt_vs_fake") and not allow_gtvf),
    )
    exec(code, ns, ns)
    return ns["ru"], ru, slf, m.forward_noiser


# ===========================================================================
# 0. The golden really is the old code
# ===========================================================================
def test_pre_change_golden_is_really_the_old_code():
    src = _pre_change_source()
    assert "_fn_block_gtvf" not in src, (
        "the HEAD golden already carries the ruling gate, so it cannot "
        "serve as a pre-change reference -- re-point _pre_change_source()"
    )
    assert "forward_noiser_allow_gt_vs_fake" not in src
    assert "_fn_decoupled" in src, (
        "the HEAD golden predates the decoupled apply entirely; it is the "
        "wrong reference for this comparison"
    )
    # and the CURRENT source does carry the gate
    assert "_fn_block_gtvf" in _CUR_SRC


# ===========================================================================
# (i) DEFAULT: gt_vs_fake positives are bit-identical to clean GT
# ===========================================================================
def test_default_gt_vs_fake_positives_are_bit_identical_to_clean_gt():
    out, orig, slf, fn = _run_decoupled(_DEC_CUR)
    assert out is orig, (
        "RULING VIOLATED: with NO flag set, the gt_vs_fake real rows are "
        "not even the same object as the clean GT"
    )
    assert torch.equal(out, orig)
    assert fn.calls == [], (
        "RULING VIOLATED: the CARN forward noiser was CALLED on the "
        "gt_vs_fake positives"
    )


def test_default_matched_pool_gt_vs_fake_positives_are_clean():
    """The SECOND apply site (matched pool, cpp==1) obeys the ruling too."""
    out, orig, slf, fn = _run_matched(
        _MAT_CUR, pair_mode="gt_vs_fake", chunks_per_pair=1)
    assert out is orig and torch.equal(out, orig), (
        "RULING VIOLATED at the matched-pool apply site"
    )
    assert fn.calls == []


# ===========================================================================
# (iv) THE PRE-CHANGE CODE FAILS THIS SUITE
# ===========================================================================
def test_pre_change_code_fails_the_ruling():
    """Run the OLD code and show it violates the ruling at BOTH sites."""
    dec_pre, mat_pre = _pre_regions()

    out, orig, _slf, fn = _run_decoupled(dec_pre)
    assert out is not orig and not torch.equal(out, orig), (
        "the pre-change decoupled block did NOT noise the gt_vs_fake "
        "positives -- the premise of this whole change is wrong"
    )
    assert fn.calls, "pre-change decoupled block never called the FN"

    out_m, orig_m, _slf_m, fn_m = _run_matched(
        mat_pre, pair_mode="gt_vs_fake", chunks_per_pair=1)
    assert not torch.equal(out_m, orig_m), (
        "the pre-change matched-pool block did NOT noise the gt_vs_fake "
        "positives"
    )
    assert fn_m.calls


# ===========================================================================
# (ii) EXACT MATCH: gt_transition is byte-identical to the pre-change code
# ===========================================================================
def test_gt_transition_matched_apply_is_bit_identical_to_pre_change():
    """cpp==2 gt_transition: current output == pre-change output, exactly."""
    _dec_pre, mat_pre = _pre_regions()
    out_cur, orig_cur, _s1, fn_cur = _run_matched(
        _MAT_CUR, pair_mode="gt_transition", chunks_per_pair=2)
    out_pre, orig_pre, _s2, fn_pre = _run_matched(
        mat_pre, pair_mode="gt_transition", chunks_per_pair=2)

    assert torch.equal(orig_cur, orig_pre), "harness inputs diverged"
    assert fn_cur.calls, "gt_transition must still call the CARN noiser"
    assert fn_cur.calls == fn_pre.calls, (
        "the CARN call sequence changed for gt_transition:\n cur=%r\n pre=%r"
        % (fn_cur.calls, fn_pre.calls)
    )
    assert torch.equal(out_cur, out_pre), (
        "gt_transition output is NOT bit-identical to the pre-change code"
    )
    # not vacuous: the transform really did something
    assert not torch.equal(out_cur, orig_cur), (
        "mutation control: gt_transition was a no-op in this harness, so "
        "the exact-match assertion proves nothing"
    )


def _max_cond(fn):
    """Deepest conditioning level the chained apply actually reached."""
    return max([max(c) for (_s, c, _r) in fn.calls], default=0)


def _cond_order(fn):
    """The ORDER the chained apply fed conditioning levels, deduplicated."""
    out = []
    for (_s, conds, _r) in fn.calls:
        c = max(conds)
        if not out or out[-1] != c:
            out.append(c)
    return out


# ``seed_clean=1`` + one real row per fake (Kk=1) removes the Req-1 cap,
# so the per-row targets run 0,0,1,1,2,2,3,3,4,4 (weak) / 0,0,1,2,...,8
# (strong) and the chain loop genuinely composes multiple levels in order.
_DEEP = dict(n_pairs=10, B=1, Kk=1, n_uniq=10, seed_clean=1)


def test_stand_in_noiser_is_order_sensitive():
    """The harness can SEE a reordered chain. Without this it cannot."""
    fn = _FN()
    x = torch.randn(2, 3, 4, 6, 8, generator=torch.Generator().manual_seed(3))
    c1 = torch.ones(2, dtype=torch.long)
    c3 = torch.full((2,), 3, dtype=torch.long)
    a = fn(fn(x, c1), c3)
    b = fn(fn(x, c3), c1)
    assert not torch.equal(a, b), (
        "the stand-in noiser is COMMUTATIVE, so no assertion in this file "
        "can detect a reordered chain loop -- the ordering coverage would "
        "be fake"
    )


def test_deep_config_really_reaches_max_t_at_least_three():
    """Guard the guard: if the depth regresses, the ordering cover dies."""
    for mode, want_min in (("weak", 3), ("strong", 3)):
        _out, _orig, _slf, fn = _run_matched(
            _MAT_CUR, pair_mode="gt_transition", chunks_per_pair=2,
            former_mode=mode, seed=7, **_DEEP)
        assert _max_cond(fn) >= want_min, (
            "%s: chained apply only reached _max_t=%d; at _max_t<=1 the "
            "loop `range(1, _max_t + 1)` is indistinguishable from its "
            "reverse, so chain ORDER is not being tested"
            % (mode, _max_cond(fn))
        )
        assert len(_cond_order(fn)) >= 3, _cond_order(fn)


def test_shipped_default_former_mode_is_weak():
    """Pin the DEFAULT, not just the two explicit values.

    The harness normally SETS ``forward_noiser_former_mode``, so the
    shipped ``getattr(..., "weak")`` fallback is never exercised and a
    change to it is invisible.  Here the attribute is absent, so the
    default is what runs -- and it must still behave as "weak".
    """
    out_def, _o, _s, fn_def = _run_matched(
        _MAT_CUR, pair_mode="gt_transition", chunks_per_pair=2,
        former_mode=None, seed=7, **_DEEP)
    out_weak, _o2, _s2, fn_weak = _run_matched(
        _MAT_CUR, pair_mode="gt_transition", chunks_per_pair=2,
        former_mode="weak", seed=7, **_DEEP)
    out_strong, _o3, _s3, fn_strong = _run_matched(
        _MAT_CUR, pair_mode="gt_transition", chunks_per_pair=2,
        former_mode="strong", seed=7, **_DEEP)
    assert fn_def.calls == fn_weak.calls and torch.equal(out_def, out_weak), (
        "the shipped forward_noiser_former_mode default no longer behaves "
        "as 'weak'"
    )
    assert not torch.equal(out_def, out_strong), (
        "mutation control: weak and strong are indistinguishable here, so "
        "the default assertion is vacuous"
    )


def test_weak_and_strong_are_genuinely_different():
    """The parametrization is LIVE: the two cells run different code.

    This is the assertion whose absence made the old 2x2 grid a 1x2 --
    ``_run_matched`` rebuilt the model and silently reverted to "weak".
    """
    out_w, _o, _s, fn_w = _run_matched(
        _MAT_CUR, pair_mode="gt_transition", chunks_per_pair=2,
        former_mode="weak", seed=7, **_DEEP)
    out_s, _o2, _s2, fn_s = _run_matched(
        _MAT_CUR, pair_mode="gt_transition", chunks_per_pair=2,
        former_mode="strong", seed=7, **_DEEP)
    assert _max_cond(fn_s) > _max_cond(fn_w), (
        "strong must reach deeper levels than weak (%d vs %d)"
        % (_max_cond(fn_s), _max_cond(fn_w))
    )
    assert not torch.equal(out_w, out_s), (
        "weak and strong produced IDENTICAL tensors -- the mode is not "
        "reaching the shipped code"
    )


@pytest.mark.parametrize("mode", ["weak", "strong"])
@pytest.mark.parametrize("depth", ["shallow", "deep"])
def test_gt_transition_bit_identical_across_configs(mode, depth):
    """Exact match over a REAL grid: mode x chain depth.

    The deep cell drives ``_max_t >= 3``, so the chained apply composes
    several conditioning levels and the comparison covers chain ORDER as
    well as chain membership -- both through the recorded call sequence
    and, because ``_FN`` is non-commutative, through the output tensor.
    """
    _dec_pre, mat_pre = _pre_regions()
    kw = dict(_DEEP) if depth == "deep" else dict(n_uniq=9)

    def _go(code):
        return _run_matched(
            code, pair_mode="gt_transition", chunks_per_pair=2,
            former_mode=mode, seed=7, **kw)

    o_cur, orig_c, _s1, f_cur = _go(_MAT_CUR)
    o_pre, orig_p, _s2, f_pre = _go(mat_pre)

    assert torch.equal(orig_c, orig_p), "harness inputs diverged"
    assert f_cur.calls, "gt_transition must still call the CARN noiser"
    assert f_cur.calls == f_pre.calls, (
        "the CARN call sequence changed for gt_transition (%s/%s):\n"
        " cur=%r\n pre=%r" % (mode, depth, _cond_order(f_cur),
                              _cond_order(f_pre))
    )
    assert torch.equal(o_cur, o_pre), (
        "gt_transition output is NOT bit-identical to the pre-change code "
        "(%s/%s)" % (mode, depth)
    )
    assert not torch.equal(o_cur, orig_c), (
        "mutation control: the transform was a no-op in this cell, so the "
        "exact-match assertion proves nothing"
    )
    if depth == "deep":
        assert _max_cond(f_cur) >= 3, _max_cond(f_cur)


def test_gt_transition_still_prints_its_banner(capsys):
    _out, _orig, _slf, _fn = _run_matched(
        _MAT_CUR, pair_mode="gt_transition", chunks_per_pair=2)
    err = capsys.readouterr().err
    assert "[FN-CHAIN-APP]" in err, (
        "gt_transition lost its proof-of-fire banner"
    )
    assert "pair_mode=gt_transition" in err
    assert "[FN-GTVF-CLEAN]" not in err, (
        "the gt_vs_fake bypass marker printed on a gt_transition call"
    )


# ===========================================================================
# (iii) THE ESCAPE HATCH restores the old behaviour, bit for bit
# ===========================================================================
def test_escape_hatch_reproduces_pre_change_gt_vs_fake_exactly():
    dec_pre, mat_pre = _pre_regions()

    out_cur, orig_cur, _s1, fn_cur = _run_decoupled(_DEC_CUR, allow_gtvf=True)
    out_pre, orig_pre, _s2, fn_pre = _run_decoupled(dec_pre)
    assert torch.equal(orig_cur, orig_pre)
    assert fn_cur.calls == fn_pre.calls
    assert torch.equal(out_cur, out_pre), (
        "forward_noiser_allow_gt_vs_fake=true did not reproduce the "
        "pre-change gt_vs_fake tensor"
    )
    assert not torch.equal(out_cur, orig_cur)

    out_m, _o, _s3, fn_m = _run_matched(
        _MAT_CUR, pair_mode="gt_vs_fake", chunks_per_pair=1, allow_gtvf=True)
    out_mp, _op, _s4, fn_mp = _run_matched(
        mat_pre, pair_mode="gt_vs_fake", chunks_per_pair=1)
    assert fn_m.calls == fn_mp.calls
    assert torch.equal(out_m, out_mp)


def test_escape_hatch_default_is_false_on_the_model():
    import model.dmd_action_forcing as MDA
    src = inspect.getsource(MDA)
    assert (
        'self.forward_noiser_allow_gt_vs_fake = bool(\n'
        '            getattr(args, "forward_noiser_allow_gt_vs_fake", False)\n'
        '        )'
    ) in src, (
        "the escape hatch is not registered on the model with default False"
    )


def test_escape_hatch_is_readable_off_the_config_object():
    """sbatch overrides land on the config, not always on the model."""
    m = _model(allow_gtvf=False)
    delattr(m, "forward_noiser_allow_gt_vs_fake")
    slf = SimpleNamespace(
        model=m,
        config=SimpleNamespace(forward_noiser_allow_gt_vs_fake=True),
        is_main_process=True)
    g = torch.Generator().manual_seed(1234)
    real = torch.randn(8, 3, 4, 6, 8, generator=g)
    ns = _base_ns(slf, real_chunks_det=real, chunks_per_pair=1, n_pairs=8,
                  pairs=[(i, i) for i in range(8)], pair_mode="gt_vs_fake")
    exec(_DEC_CUR, ns, ns)
    assert ns["real_chunks_det"] is not real, (
        "the escape hatch is not honoured off self.config"
    )


def test_missing_flag_everywhere_still_yields_clean_positives():
    """A model/config built before the flag existed must still be CLEAN."""
    m = _model(allow_gtvf=False)
    delattr(m, "forward_noiser_allow_gt_vs_fake")
    slf = SimpleNamespace(model=m, config=SimpleNamespace(),
                          is_main_process=True)
    g = torch.Generator().manual_seed(1234)
    real = torch.randn(8, 3, 4, 6, 8, generator=g)
    ns = _base_ns(slf, real_chunks_det=real, chunks_per_pair=1, n_pairs=8,
                  pairs=[(i, i) for i in range(8)], pair_mode="gt_vs_fake")
    exec(_DEC_CUR, ns, ns)
    assert ns["real_chunks_det"] is real


# ===========================================================================
# (v) COUNTERS and BANNERS -- proof from a counter, never from the patch
# ===========================================================================
def test_bypass_counter_rises_and_applied_counter_stays_zero():
    m = _model(allow_gtvf=False)
    slf = SimpleNamespace(model=m, config=SimpleNamespace(),
                          is_main_process=True)
    g = torch.Generator().manual_seed(5)
    for i in range(3):
        real = torch.randn(6, 3, 4, 6, 8, generator=g)
        ns = _base_ns(slf, real_chunks_det=real, chunks_per_pair=1,
                      n_pairs=6, pairs=[(j, j) for j in range(6)],
                      pair_mode="gt_vs_fake")
        exec(_DEC_CUR, ns, ns)
        assert ns["real_chunks_det"] is real
        assert int(getattr(slf, "_fn_gtvf_noise_skipped", 0)) == i + 1, (
            "the bypass counter is not monotone"
        )
    assert int(getattr(slf, "_fn_gtvf_noise_applied", 0)) == 0


def test_applied_counter_rises_only_under_the_escape_hatch():
    _out, _orig, slf, _fn = _run_decoupled(_DEC_CUR, allow_gtvf=True)
    assert int(getattr(slf, "_fn_gtvf_noise_applied", 0)) == 1
    assert int(getattr(slf, "_fn_gtvf_noise_skipped", 0)) == 0


def test_counters_are_emitted_by_both_logs_dicts():
    """A counter nobody logs is not a counter."""
    src = _CUR_SRC
    # 5 emission sites = matched-branch logs dict + positional logs dict
    # + the THREE early returns (F_total < npb, n_chunks < 2 x2). With the
    # early returns covered, "always emitted" is literally true: there is
    # no return path out of _ladd_run_pair_mode that omits the counters,
    # so an ABSENT key means a broken emit, never a zero.
    for key in ("train/fn_gtvf_noise_skipped", "train/fn_gtvf_noise_applied"):
        assert src.count('"%s": float(' % key) == 5, (
            "%s must be emitted by the matched-branch dict, the positional "
            "dict AND all three early returns (found %d)"
            % (key, src.count('"%s": float(' % key))
        )
    # No `return` inside the method may omit them.
    import re as _re
    _returns = _re.findall(r"return zero, \{[^}]*\}", src, flags=_re.S)
    for _r in _returns:
        assert "fn_gtvf_noise_skipped" in _r, (
            "an early return omits the ruling counters:\n%s" % _r)
    # emitted via getattr default, so a step before the first bypass still
    # reports 0.0 rather than nothing.
    assert 'getattr(self, "_fn_gtvf_noise_skipped", 0)' in src
    assert 'getattr(self, "_fn_gtvf_noise_applied", 0)' in src


def test_decoupled_app_banner_never_prints_for_gt_vs_fake(capsys):
    _run_decoupled(_DEC_CUR)
    err = capsys.readouterr().err
    assert "[FN-DECOUPLED-APP] ACTIVE" not in err, (
        "the string the ruling says must never appear again printed with "
        "pair_mode=gt_vs_fake"
    )
    assert "[FN-GTVF-CLEAN] BYPASS site=decoupled" in err
    assert "pair_mode=gt_vs_fake" in err


def test_decoupled_app_banner_still_prints_under_the_escape_hatch(capsys):
    _run_decoupled(_DEC_CUR, allow_gtvf=True)
    err = capsys.readouterr().err
    assert "[FN-DECOUPLED-APP] ACTIVE" in err
    assert "pair_mode=gt_vs_fake" in err


def test_matched_bypass_banner(capsys):
    _run_matched(_MAT_CUR, pair_mode="gt_vs_fake", chunks_per_pair=1)
    err = capsys.readouterr().err
    assert "[FN-GTVF-CLEAN] BYPASS site=matched_pool" in err
    assert "[FN-CHAIN-APP]" not in err


# ===========================================================================
# Scope: the ruling names gt_vs_fake ONLY
# ===========================================================================
def test_adjacent_chunks_is_untouched_by_the_ruling():
    """adjacent_chunks' "real" is the source latent, not GT."""
    dec_pre, _mat_pre = _pre_regions()
    out_cur, orig, _s, fn_cur = _run_decoupled(
        _DEC_CUR, pair_mode="adjacent_chunks")
    out_pre, orig_p, _s2, fn_pre = _run_decoupled(
        dec_pre, pair_mode="adjacent_chunks")
    assert torch.equal(orig, orig_p)
    assert out_cur is not orig, "the gate leaked past gt_vs_fake"
    assert fn_cur.calls == fn_pre.calls
    assert torch.equal(out_cur, out_pre)


def test_the_gate_predicate_is_pair_mode_keyed_and_default_clean():
    """The gate expression itself, lifted from source and evaluated."""
    fn = _fn_ast(_CUR_SRC)
    body, idx = _parent_body_containing(
        fn,
        lambda st: (isinstance(st, ast.Assign)
                    and any(getattr(t, "id", None) == "_fn_block_gtvf"
                            for t in st.targets)),
    )
    assert idx is not None, "_fn_block_gtvf assignment not found"
    expr = compile(ast.Expression(body=body[idx].value), "<gate>", "eval")
    for mode, allow, want in [
        ("gt_vs_fake", False, True),
        ("gt_vs_fake", True, False),
        ("gt_transition", False, False),
        ("gt_transition", True, False),
        ("adjacent_chunks", False, False),
    ]:
        env = {"pair_mode": mode, "_fn_allow_gtvf": allow}
        assert bool(eval(expr, env, env)) is want, (mode, allow)


def test_no_other_forward_noiser_call_site_can_reach_gt_vs_fake():
    """Census of every FN application to the disc's REAL rows.

    Each site must be provably out of reach for gt_vs_fake, either by an
    explicit ``pair_mode == "gt_transition"`` test, by a ``chunks_per_pair
    == 2`` gate, or by ``_fn_block_gtvf``.
    """
    src = _CUR_SRC
    _flat = " ".join(src.split())
    # 1. CARN-FORMER: explicit pair_mode test.
    assert 'pair_mode == "gt_transition" and _carn_knob' in _flat, (
        "the [CARN-FORMER] real-former apply lost its gt_transition gate"
    )
    assert 'pair_mode == "gt_transition" and _carn_latter_knob' in _flat, (
        "the [CARN-LATTER-REVERSE] real-latter apply lost its "
        "gt_transition gate"
    )
    # 2. CARN-MATCH-POOL: cpp==2 gate.
    assert ("if carn and chunks_per_pair == 2 and bool(getattr( self.model, "
            '"ladd_gt_transition_carn_match_pool", False)):') in _flat, (
        "the [CARN-MATCH-POOL] real-former apply lost its cpp==2 gate"
    )
    # 3 + 4. The two ruling-gated sites.
    assert src.count(
        "if _fn_decoupled_would_apply and not _fn_block_gtvf:") == 1
    assert src.count(
        "if _fn_matched_would_apply and not _fn_block_gtvf:") == 1
    # No fifth: every direct ``self.model.forward_noiser(`` application and
    # every call through the sign-aware ``_carn_transform`` helper is
    # accounted for by the routes above.
    # 6 direct = decoupled uncond/chained/single (3, _fn_block_gtvf-gated)
    #   + [CARN-MATCH-POOL] (1, cpp==2-gated)
    #   + matched chained (1) + matched legacy (1), both _fn_block_gtvf-gated.
    # 2 generic = CARN single-call + recursive; the wrappers are invoked only
    # by the explicitly gt_transition-gated former/latter routes above.
    assert src.count("self.model.forward_noiser(") == 6, (
        "the number of forward-noiser call sites inside _ladd_run_pair_mode "
        "changed (%d) -- re-run the audit, a new site may be able to reach "
        "the gt_vs_fake positives" % src.count("self.model.forward_noiser(")
    )
    assert src.count("x = noiser(") == 2, (
        "the number of sign-aware CARN helper call sites changed (%d) -- "
        "re-run the gt_vs_fake reachability audit"
        % src.count("x = noiser(")
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
