"""Plumbing test for ``carn_seam_affine_match_mean``.

The consumer (``pipeline/action_forcing_training.py``, in
``_carn_seam_correct``) reads the flag off the PIPELINE object::

    _mm = bool(getattr(self, "carn_seam_affine_match_mean", False))

but until 2026-08-25 nothing registered it as a config key on
``ActionForcingDMD`` and nothing stamped it onto the pipeline, so setting
it in a job config was SILENTLY IGNORED -- the run always took the
default (mean matching OFF) no matter what the YAML said. This pins the
two-line plumbing:

  1. REGISTRATION -- ``ActionForcingDMD.__init__`` reads
     ``args.carn_seam_affine_match_mean`` into ``self`` (bool, default
     False);
  2. PUBLISH -- ``generate_next_chunk`` stamps it onto ``pipe`` every
     roll, right beside ``pipe.carn_seam_affine_lambda``, and
     UNCONDITIONALLY (not inside the ``lambda > 0`` block, which only
     publishes the target).

Both statements are lifted out of the REAL parsed source and EXECUTED
here against stand-in ``args``/``self``/``pipe`` namespaces, so this
tracks the shipped code rather than a copy of it -- ``ActionForcingDMD``
itself cannot be constructed on CPU.

DEFAULT-FALSE is pinned on both sides: an ``args``/``self`` with the
attribute absent must yield ``False``, i.e. every run that does not set
the flag is byte-identical to before this change.

CPU-only, no model constructed, no CUDA.

Run:
    python -m pytest testing/test_carn_seam_match_mean_plumbing.py -q
or
    python testing/test_carn_seam_match_mean_plumbing.py
"""
import ast
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FLAG = "carn_seam_affine_match_mean"
SIBLING = "carn_seam_affine_lambda"

SRC_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "model", "dmd_action_forcing.py")
with open(SRC_PATH) as _f:
    SRC = _f.read()
TREE = ast.parse(SRC)


# ---------------------------------------------------------------------------
# helpers -- pull the real statements out of the real source
# ---------------------------------------------------------------------------
def _method(cls_name, fn_name):
    for c in ast.walk(TREE):
        if isinstance(c, ast.ClassDef) and c.name == cls_name:
            for f in c.body:
                if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                        and f.name == fn_name:
                    return f
    raise AssertionError(f"{cls_name}.{fn_name} not found in {SRC_PATH}")


def _assignments(fn, obj, attr):
    """Every ``<obj>.<attr> = ...`` statement inside ``fn``."""
    out = []
    for n in ast.walk(fn):
        if not isinstance(n, ast.Assign):
            continue
        for t in n.targets:
            if (isinstance(t, ast.Attribute) and t.attr == attr
                    and isinstance(t.value, ast.Name) and t.value.id == obj):
                out.append(n)
    return out


def _exec(node, env):
    mod = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(mod)
    exec(compile(mod, SRC_PATH, "exec"), env)  # noqa: S102
    return env


# ---------------------------------------------------------------------------
# 1. registration: ActionForcingDMD.__init__ reads it off ``args``
# ---------------------------------------------------------------------------
def test_flag_is_registered_in_init():
    init = _method("ActionForcingDMD", "__init__")
    stmts = _assignments(init, "self", FLAG)
    assert len(stmts) == 1, (
        f"expected exactly one `self.{FLAG} = ...` in "
        f"ActionForcingDMD.__init__, got {len(stmts)}")


def test_registration_sits_next_to_its_sibling_flag():
    """It must be registered with the other carn_seam_affine knob so the
    two can never drift apart (one registered, one not -- the bug)."""
    init = _method("ActionForcingDMD", "__init__")
    lam = _assignments(init, "self", SIBLING)
    mm = _assignments(init, "self", FLAG)
    assert lam and mm
    assert abs(mm[0].lineno - lam[0].end_lineno) <= 6, (
        f"self.{FLAG} (line {mm[0].lineno}) drifted away from "
        f"self.{SIBLING} (ends line {lam[0].end_lineno})")


@pytest.mark.parametrize("val,want", [
    (True, True), (False, False), (1, True), (0, False),
])
def test_registration_reads_args_value(val, want):
    init = _method("ActionForcingDMD", "__init__")
    stmt = _assignments(init, "self", FLAG)[0]
    env = {"self": SimpleNamespace(), "args": SimpleNamespace(**{FLAG: val})}
    _exec(stmt, env)
    got = getattr(env["self"], FLAG)
    assert got is want, f"args.{FLAG}={val!r} -> {got!r}, want {want!r}"


def test_registration_default_is_false_when_args_lacks_the_key():
    """A config that never mentions the flag must be byte-identical to
    the pre-change behaviour."""
    init = _method("ActionForcingDMD", "__init__")
    stmt = _assignments(init, "self", FLAG)[0]
    env = {"self": SimpleNamespace(), "args": SimpleNamespace()}
    _exec(stmt, env)
    assert getattr(env["self"], FLAG) is False


# ---------------------------------------------------------------------------
# 2. publish: generate_next_chunk stamps it onto ``pipe``
# ---------------------------------------------------------------------------
def test_flag_is_published_to_the_pipeline():
    fn = _method("ActionForcingDMD", "generate_next_chunk")
    stmts = _assignments(fn, "pipe", FLAG)
    assert len(stmts) == 1, (
        f"expected exactly one `pipe.{FLAG} = ...` in generate_next_chunk, "
        f"got {len(stmts)}")


def test_publish_is_unconditional():
    """It must sit at the same statement level as
    ``pipe.carn_seam_affine_lambda`` -- NOT inside the ``lambda > 0``
    block, or a run with the affine off would leave a stale value from a
    previous pipeline owner."""
    fn = _method("ActionForcingDMD", "generate_next_chunk")
    mm = _assignments(fn, "pipe", FLAG)[0]
    lam = _assignments(fn, "pipe", SIBLING)[0]
    holder = None
    for n in ast.walk(fn):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(n, field, None)
            if isinstance(block, list) and any(s is lam for s in block):
                holder = block
    assert holder is not None, "could not locate the publisher block"
    assert any(s is mm for s in holder), (
        f"pipe.{FLAG} is not in the same block as pipe.{SIBLING} -- it "
        "must be published unconditionally, every roll")


@pytest.mark.parametrize("val,want", [(True, True), (False, False)])
def test_publish_forwards_the_registered_value(val, want):
    fn = _method("ActionForcingDMD", "generate_next_chunk")
    stmt = _assignments(fn, "pipe", FLAG)[0]
    pipe = SimpleNamespace()
    env = {"self": SimpleNamespace(**{FLAG: val}), "pipe": pipe}
    _exec(stmt, env)
    assert getattr(pipe, FLAG) is want


def test_publish_default_is_false_when_unset_on_self():
    """Old checkpoints / objects built before the flag existed must
    publish False, not raise."""
    fn = _method("ActionForcingDMD", "generate_next_chunk")
    stmt = _assignments(fn, "pipe", FLAG)[0]
    pipe = SimpleNamespace()
    env = {"self": SimpleNamespace(), "pipe": pipe}
    _exec(stmt, env)
    assert getattr(pipe, FLAG) is False


# ---------------------------------------------------------------------------
# 3. end-to-end: config value -> self -> pipe -> what the consumer reads
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cfg,want", [
    ({FLAG: True}, True),
    ({FLAG: False}, False),
    ({}, False),                       # unset -> legacy-safe default
])
def test_config_value_reaches_the_consumer(cfg, want):
    reg = _assignments(_method("ActionForcingDMD", "__init__"), "self", FLAG)[0]
    pub = _assignments(
        _method("ActionForcingDMD", "generate_next_chunk"), "pipe", FLAG)[0]
    model, pipe = SimpleNamespace(), SimpleNamespace()
    _exec(reg, {"self": model, "args": SimpleNamespace(**cfg)})
    _exec(pub, {"self": model, "pipe": pipe})
    # exactly what pipeline/action_forcing_training.py's
    # ``_carn_seam_correct`` evaluates:
    assert bool(getattr(pipe, FLAG, False)) is want


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
