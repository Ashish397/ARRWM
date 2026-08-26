"""Coverage for ``forward_noiser_apply_decoupled`` (2026-08-26).

THE BUG IT FIXES.  Before this flag the ONLY code path that applied the
learned CARN forward-noiser (FN) to the GT/real disc input lived inside
``_ladd_run_pair_mode``'s MATCHED-POOL branch and was gated on
``chunks_per_pair == 2``.  That made the FN application reachable only
with

    ladd_gt_transition_enabled=true  AND  ladd_gt_transition_match=true

so an arm running ``ladd_gt_transition_enabled=false`` +
``ladd_gt_vs_fake_enabled=true`` (the current gansig_gtvf_* recipe) that
set ``forward_noiser_apply_gt_former=true`` got a SILENT NO-OP: the FN
trained but never touched a tensor.

WHAT IS PINNED HERE.

  1. DEFAULT-OFF BYTE-IDENTITY, against the REAL SOURCE.  The decoupled
     block is lifted out of the shipped ``_ladd_run_pair_mode`` AST and
     EXECUTED here -- not re-implemented -- so the test cannot drift from
     the code.  With the flag off the block must leave ``real_chunks_det``
     as the SAME OBJECT (identity, not just equality), which is a stronger
     statement than bitwise equality: nothing downstream can observe a
     difference.  Each identity assertion carries an explicit MUTATION
     CONTROL that flips exactly one input and shows the assertion fails.

  2. POSITIVE: flag ON with ``pair_mode="gt_vs_fake"`` /
     ``chunks_per_pair == 1`` -- the FN really is called, the realized
     per-row levels are NON-ZERO, level-0 rows keep their clean bytes, the
     moment-preserving restore holds, and the ``[FN-DECOUPLED-APP]``
     banner is emitted (silent no-ops are this project's endemic failure).

  3. The COUPLED path is unchanged: the matched-branch guard expression
     and the ``_tgt`` schedule are lifted from source and shown to reduce
     to their pre-change form whenever ``chunks_per_pair == 2``, for BOTH
     values of the new flag.

  4. The model registers the flag and it defaults to False.

CPU-only, no CUDA, no dataset.  ``wan/modules/t5.py`` evaluates
``torch.cuda.current_device()`` at import, so imports are patched exactly
as ``testing/test_gan_code_fixes.py`` does.

Run:
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" \
        python -m pytest testing/test_fn_apply_decoupled.py -q
"""
import ast
import inspect
import os
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with patch.object(torch.cuda, "current_device", return_value=0):
    from trainer import causal_action_forcing_train as CAFT

Trainer = CAFT.ActionForcingDMDTrainer


# ===========================================================================
# Lift the shipped block out of the real source
# ===========================================================================
def _pair_mode_fn_ast():
    """AST of the REAL ``_ladd_run_pair_mode``."""
    src = textwrap.dedent(inspect.getsource(Trainer._ladd_run_pair_mode))
    return ast.parse(src).body[0]


def _decoupled_block_code():
    """Compile ``_fn_decoupled = ...`` + the ``if`` that consumes it.

    Both are direct statements of the method body, so this is the exact
    shipped code -- no copy, no paraphrase.
    """
    fn = _pair_mode_fn_ast()
    idx = None
    for i, st in enumerate(fn.body):
        if (isinstance(st, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "_fn_decoupled"
                        for t in st.targets)):
            idx = i
            break
    assert idx is not None, (
        "`_fn_decoupled` assignment not found at the top level of "
        "_ladd_run_pair_mode -- the decoupled block moved or was removed."
    )
    guard = fn.body[idx + 1]
    assert isinstance(guard, ast.If), (
        "the statement after the `_fn_decoupled` assignment is no longer "
        "the guarded apply block."
    )
    names = {n.id for n in ast.walk(guard.test) if isinstance(n, ast.Name)}
    assert "_fn_decoupled" in names, "the apply block is not flag-gated."
    mod = ast.Module(body=[fn.body[idx], guard], type_ignores=[])
    ast.fix_missing_locations(mod)
    return compile(mod, "<fn_decoupled_block>", "exec")


_BLOCK = _decoupled_block_code()


# ===========================================================================
# Fixtures
# ===========================================================================
class _RecordingFN(torch.nn.Module):
    """Deterministic stand-in for the trained ForwardNoiser.

    Adds a FIXED zero-mean texture pattern scaled by the conditioning
    level, so (a) the output survives the moment-preserving restore (a
    pure DC shift would be undone, hiding a real application), and (b)
    every call is recorded for chain-composition assertions.
    """

    def __init__(self, seed=0):
        super().__init__()
        self.calls = []
        g = torch.Generator().manual_seed(seed)
        self._g = g

    def forward(self, x, carn_step, residual=True):
        self.calls.append(
            (tuple(x.shape), carn_step.detach().cpu().tolist(), bool(residual))
        )
        pat = torch.sin(
            torch.arange(x.numel(), dtype=x.dtype).reshape(x.shape) * 0.7
        )
        lvl = carn_step.to(x.dtype).view(-1, *([1] * (x.dim() - 1)))
        return x + 0.25 * lvl * pat


def make_ctx(
    *,
    decoupled,
    gt_former=True,
    gt_both=False,
    chunks_per_pair=1,
    n_pairs=6,
    pair_mode="gt_vs_fake",
    former_mode="weak",
    chain_levels=True,
    step_uncond=False,
    apply_gt_level=1,
    B=1,
    npb=3,
    have_fn=True,
    seed=1234,
):
    """Namespace the lifted block executes in."""
    fn = _RecordingFN() if have_fn else None
    model = SimpleNamespace(
        forward_noiser_apply_decoupled=bool(decoupled),
        forward_noiser_apply_gt_former=bool(gt_former),
        forward_noiser_apply_gt_both=bool(gt_both),
        forward_noiser_former_mode=former_mode,
        forward_noiser_chain_levels=bool(chain_levels),
        forward_noiser_step_unconditioned=bool(step_uncond),
        forward_noiser_apply_gt_level=int(apply_gt_level),
        forward_noiser=fn,
    )
    slf = SimpleNamespace(
        model=model,
        config=SimpleNamespace(),
        is_main_process=True,
    )
    g = torch.Generator().manual_seed(seed)
    real = torch.randn(
        n_pairs * B, chunks_per_pair * npb, 4, 6, 8, generator=g
    )
    ns = {
        "torch": torch,
        "self": slf,
        "real_chunks_det": real,
        "chunks_per_pair": int(chunks_per_pair),
        "n_pairs": int(n_pairs),
        "pairs": [(i, i) for i in range(n_pairs)],
        "pair_mode": pair_mode,
        "getattr": getattr,
        "bool": bool,
        "int": int,
        "str": str,
        "max": max,
        "min": min,
        "range": range,
        "__builtins__": __builtins__,
    }
    ns["_orig_real"] = real
    ns["_fn_module"] = fn
    return ns


def run_block(ns):
    exec(_BLOCK, ns, ns)
    return ns["real_chunks_det"]


# ===========================================================================
# 1. DEFAULT-OFF byte-identity (+ mutation controls)
# ===========================================================================
def test_default_off_is_object_identity():
    """Flag off => the block does not touch ``real_chunks_det`` at all."""
    ns = make_ctx(decoupled=False)
    out = run_block(ns)
    assert out is ns["_orig_real"], (
        "flag OFF must leave real_chunks_det as the SAME OBJECT"
    )
    assert ns["_fn_module"].calls == [], "flag OFF called the forward noiser"


def test_default_off_mutation_control():
    """MUTATION CONTROL: the identity assertion is not vacuous."""
    ns = make_ctx(decoupled=True)
    out = run_block(ns)
    assert out is not ns["_orig_real"], (
        "mutation control failed: turning the flag ON did not change the "
        "tensor, so the default-off identity test proves nothing"
    )
    assert not torch.equal(out, ns["_orig_real"])
    assert ns["_fn_module"].calls, "flag ON did not call the forward noiser"


def test_default_off_via_config_only_object():
    """The flag is also honoured off ``self.config`` (sbatch overrides)."""
    ns = make_ctx(decoupled=False)
    delattr(ns["self"].model, "forward_noiser_apply_decoupled")
    out = run_block(ns)
    assert out is ns["_orig_real"]
    # mutation control: same namespace shape, flag set on config instead
    ns2 = make_ctx(decoupled=False)
    delattr(ns2["self"].model, "forward_noiser_apply_decoupled")
    ns2["self"].config.forward_noiser_apply_decoupled = True
    assert run_block(ns2) is not ns2["_orig_real"]


@pytest.mark.parametrize(
    "kw, why",
    [
        (dict(gt_former=False, gt_both=False), "no apply_gt_* flag set"),
        (dict(chunks_per_pair=2), "gt_transition (cpp==2) is the coupled path"),
        (dict(have_fn=False), "no forward noiser constructed"),
        (dict(n_pairs=0), "no pairs"),
    ],
)
def test_block_is_inert_without_its_preconditions(kw, why):
    ns = make_ctx(decoupled=True, **kw)
    if kw.get("n_pairs") == 0:
        ns["real_chunks_det"] = torch.zeros(0, 3, 4, 6, 8)
        ns["_orig_real"] = ns["real_chunks_det"]
    out = run_block(ns)
    assert out is ns["_orig_real"], f"block should be inert: {why}"


# ===========================================================================
# 2. POSITIVE: gt_transition OFF, gt_vs_fake ON -> the FN really fires
# ===========================================================================
def _weak(j):
    return max(0, min((j + 1) // 2, j))


def _strong(j):
    return max(0, min(max(0, j), j))


def test_positive_gtvf_only_fires_with_nonzero_levels(capsys):
    """The exact configuration the arms run: no gt_transition, no match."""
    ns = make_ctx(decoupled=True, pair_mode="gt_vs_fake",
                  chunks_per_pair=1, n_pairs=8, former_mode="weak")
    out = run_block(ns)
    fn = ns["_fn_module"]

    assert out is not ns["_orig_real"]
    assert fn.calls, "the FN was never called -- SILENT NO-OP"

    banner = capsys.readouterr().err
    assert "[FN-DECOUPLED-APP]" in banner, "no banner: the run cannot prove it"
    assert "pair_mode=gt_vs_fake" in banner
    # levels must not be all-zero (the all-zero-levels bug: the FN output
    # would be discarded row by row and the apply would be a no-op)
    lvl = banner.split("levels=")[1].split("]")[0] + "]"
    levels = ast.literal_eval(lvl)
    assert levels == [_weak(j) for j in range(8)], levels
    assert max(levels) > 0, "all-zero levels => FN output discarded"
    assert "maxlvl=%d" % max(levels) in banner


def _banner_levels(err):
    lvl = err.split("levels=")[1].split("]")[0] + "]"
    return ast.literal_eval(lvl)


def test_positive_levels_weak_and_strong_schedules(capsys):
    """The realized levels follow the documented schedule.

    Cross-checked two ways: the banner's ``levels=`` (what an operator
    reads off a log) AND the per-cond row COUNTS the FN actually saw
    (what the tensors experienced) -- so a truthful-banner/lying-tensor
    split would fail.
    """
    for mode, ref in (("weak", _weak), ("strong", _strong)):
        ns = make_ctx(decoupled=True, n_pairs=7, former_mode=mode)
        run_block(ns)
        want = [ref(j) for j in range(7)]
        got = _banner_levels(capsys.readouterr().err)
        assert got == want, f"{mode} banner: {got} != {want}"
        # chain rule: row t is fed cond=c iff t >= c and t % 2 == c % 2
        seen = {}
        for _shape, conds, _res in ns["_fn_module"].calls:
            assert len(set(conds)) == 1, "one cond per call"
            seen[conds[0]] = seen.get(conds[0], 0) + len(conds)
        for c in range(1, max(want) + 1):
            expect = sum(1 for t in want if t >= c and t % 2 == c % 2)
            assert seen.get(c, 0) == expect, (mode, c, seen)


def test_positive_level_zero_rows_keep_clean_bytes():
    ns = make_ctx(decoupled=True, n_pairs=6, former_mode="weak")
    orig = ns["_orig_real"].clone()
    out = run_block(ns)
    # weak: row j target = (j+1)//2 capped at j -> row 0 is level 0
    assert _weak(0) == 0
    assert torch.equal(out[0], orig[0]), (
        "a level-0 row must keep the ORIGINAL clean GT bytes (the FN is "
        "undefined at level 0)"
    )
    assert not torch.equal(out[3], orig[3]), "row 3 should have been noised"


def test_positive_moment_preserving_restore():
    """Only texture STRUCTURE changes -- no brightness/colour shift.

    The shipped restore is (i) per-channel DC re-centre then (ii) a
    per-row SCALAR rescale to the original mean magnitude.  So the exact
    invariants are: mean magnitude preserved, and the per-channel DC
    vector preserved UP TO ONE SCALAR PER ROW (a scalar rescale cannot
    introduce a colour shift, which is what _mean_equalize_pair keys on).
    """
    ns = make_ctx(decoupled=True, n_pairs=6, former_mode="strong")
    orig = ns["_orig_real"].clone()
    out = run_block(ns)

    a_in = orig.abs().mean(dim=[1, 2, 3, 4])
    a_out = out.abs().mean(dim=[1, 2, 3, 4])
    assert torch.allclose(a_in, a_out, rtol=1e-5, atol=1e-6), (
        "mean magnitude drifted -- the brightness cue the disc latches on"
    )

    m_in = orig.mean(dim=[1, 3, 4])          # [rows, C]
    m_out = out.mean(dim=[1, 3, 4])
    for r in range(m_in.shape[0]):
        s = (m_out[r] @ m_in[r]) / (m_in[r] @ m_in[r])
        assert torch.allclose(m_out[r], s * m_in[r], atol=1e-6), (
            f"row {r}: per-channel DC is not a pure scalar rescale of the "
            f"original -> a colour shift leaked in"
        )
        # s = a_in / a_out(post-recentre); how far it sits from 1 is a
        # property of the STAND-IN noiser's amplitude, not of the code --
        # only bounded here to catch a runaway rescale.
        assert 0.5 < float(s) < 2.0, f"row {r}: DC scale {float(s)}"

    # MUTATION CONTROL: without the restore (raw FN output) the magnitude
    # invariant would be violated for the noised rows.
    raw = orig.clone()
    raw[3] = raw[3] + 0.25 * 3 * torch.sin(
        torch.arange(raw[3].numel(), dtype=raw.dtype).reshape(raw[3].shape)
        * 0.7)
    assert not torch.allclose(
        raw.abs().mean(dim=[1, 2, 3, 4]), a_in, rtol=1e-5, atol=1e-6), (
        "mutation control failed: the magnitude assertion is vacuous"
    )


def test_positive_req1_cap_never_exceeds_student_drift():
    """Req-1: the real is noised strictly LESS than the fake it faces."""
    for mode in ("weak", "strong"):
        for j in range(10):
            t = _weak(j) if mode == "weak" else _strong(j)
            # student chunk j has drift level >= j+1
            assert t <= j, (mode, j, t)
            assert t < j + 1


def test_positive_step_unconditioned_uses_cond_zero():
    ns = make_ctx(decoupled=True, n_pairs=6, step_uncond=True)
    out = run_block(ns)
    fn = ns["_fn_module"]
    assert fn.calls, "step-unconditioned FN was never applied"
    assert all(set(c) == {0} for (_s, c, _r) in fn.calls), (
        "a step-unconditioned FN must be applied at carn_step=0 only"
    )
    assert torch.equal(out[0], ns["_orig_real"][0]), "level-0 row must be clean"


def test_positive_non_chain_fixed_level_is_req1_capped():
    ns = make_ctx(decoupled=True, n_pairs=5, chain_levels=False,
                  apply_gt_level=3)
    run_block(ns)
    fn = ns["_fn_module"]
    assert len(fn.calls) == 1, "non-chain mode must be a SINGLE call"
    conds = fn.calls[0][1]
    # row j gets min(3, j); row 0 -> 0 -> clamped to 1 for the call but
    # discarded afterwards.
    assert conds == [max(1, min(3, j)) for j in range(5)], conds


# ===========================================================================
# 3. The COUPLED (gt_transition) path is unchanged
# ===========================================================================
def _find_matched_apply_guard():
    """The matched-pool apply guard, lifted from source."""
    fn = _pair_mode_fn_ast()
    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
        if {"_gt_both", "_gt_former", "chunks_per_pair"} <= names:
            return node.test
    raise AssertionError("matched-pool FN apply guard not found")


def test_coupled_guard_reduces_to_legacy_when_flag_off():
    """(cpp==2 or (flag and cpp==1)) == (cpp==2) for every flag-off input."""
    test = _find_matched_apply_guard()
    expr = compile(ast.Expression(body=test), "<guard>", "eval")

    class _M:
        forward_noiser = object()
        forward_noiser_apply_gt_both = False
        forward_noiser_apply_gt_former = True

    for cpp in (1, 2, 3):
        env = {
            "self": SimpleNamespace(model=_M()),
            "_gt_both": False,
            "_gt_former": True,
            "chunks_per_pair": cpp,
            "_fn_decoupled": False,
            "getattr": getattr,
        }
        assert bool(eval(expr, env, env)) == (cpp == 2), (
            f"flag OFF changed the coupled guard at cpp={cpp}"
        )
    # MUTATION CONTROL: with the flag ON, cpp==1 must now pass.
    env = {
        "self": SimpleNamespace(model=_M()),
        "_gt_both": False,
        "_gt_former": True,
        "chunks_per_pair": 1,
        "_fn_decoupled": True,
        "getattr": getattr,
    }
    assert bool(eval(expr, env, env)) is True, (
        "mutation control failed: flag ON did not open cpp==1"
    )


def test_coupled_target_schedule_unchanged_at_cpp2():
    """The matched ``_tgt`` schedule at cpp==2 is the pre-change formula."""
    src = textwrap.dedent(inspect.getsource(Trainer._ladd_run_pair_mode))
    assert "_L = max(0, (_u + 1) - (_seed_r1 - 1))" in src, (
        "the coupled (cpp==2) latter-level formula changed"
    )
    assert "_t = (max(0, _L - 2) if _mode == \"strong\"" in src
    assert "else max(0, (_L - 1) // 2))" in src
    # and the decoupled cpp==1 arm exists alongside it
    assert "_Lf = max(0, _u - (_seed_r1 - 1))" in src, (
        "the decoupled (cpp==1) matched-pool schedule is missing"
    )
    # Equivalence of the two forms at cpp==2 (the noised chunk is the
    # FORMER at _u, own level = L-1):
    for u in range(12):
        for seed_r1 in (1, 2, 6):
            L = max(0, (u + 1) - (seed_r1 - 1))
            Lf = max(0, u - (seed_r1 - 1))
            assert max(0, (L - 1) // 2) == Lf // 2, (u, seed_r1)
            assert max(0, L - 2) == max(0, Lf - 1), (u, seed_r1)


def test_half_offsets_are_legacy_at_cpp2():
    src = textwrap.dedent(inspect.getsource(Trainer._ladd_run_pair_mode))
    assert "_half_offsets = ((0, npb) if chunks_per_pair == 2" in src
    assert "for _h0 in _half_offsets:" in src
    assert "for _h0 in (0, npb):" not in src, (
        "the legacy half loop is still hard-coded somewhere -- an empty "
        "slice would be fed to the FN at cpp==1"
    )


# ===========================================================================
# 4. Model registration + default
# ===========================================================================
def test_model_registers_the_flag_default_false():
    import model.dmd_action_forcing as MDA
    src = inspect.getsource(MDA)
    assert (
        'self.forward_noiser_apply_decoupled = bool(\n'
        '            getattr(args, "forward_noiser_apply_decoupled", False)\n'
        '        )'
    ) in src, "the flag is not registered on the model with default False"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
