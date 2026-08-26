"""Coverage for ``model/one_forcing_gan.py::split_of_head_param_groups``.

The function builds the two ``fake_optimizer`` param groups (backbone @
``fake_lr``, One-Forcing disc head @ ``gan_of_head_lr``). Its own
docstring states why it is a function at all: each of the three ways the
split can go wrong -- a dropped parameter, a duplicated parameter, an
empty head group -- is SILENT at runtime, and a dropped parameter is the
very bf16-ulp starvation bug the flag exists to remove, in a new dress.
It shipped untested.

Pinned here:
  * the split is by OBJECT IDENTITY, not by name -- a DDP-prefixed or
    duplicated NAME must not change the outcome, and two parameters that
    happen to hold equal values must not be conflated;
  * group ORDER (backbone first, so ``param_groups[0]`` keeps meaning
    "the fake_lr group") and the two learning rates;
  * exact partition: every head param in the head group, everything else
    in the backbone group, nothing in both, nothing lost;
  * frozen (``requires_grad=False``) head params are DROPPED rather than
    smuggled into the optimizer;
  * all four ``RuntimeError`` paths: no trainable head param, a duplicate
    in the head group, head params absent from the optimizer's list, and
    an empty backbone group;
  * the result is accepted by a real ``torch.optim.AdamW`` and each group
    really steps at its own LR;
  * the disc-head MODULE NAMES agree across the three places they are
    spelled (``model/dmd_action_forcing._OF_HEAD_MODULE_NAMES`` and both
    trainers) -- a drift there produces an EMPTY head group, which is the
    exact silent failure this function raises about.

CPU-only, no CUDA, torch only.

Run:
    python -m pytest testing/test_of_head_param_groups.py -q
or
    python testing/test_of_head_param_groups.py
"""
import ast
import os
import re
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.one_forcing_gan import split_of_head_param_groups  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAKE_LR = 3e-6
HEAD_LR = 2e-4


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(3, 3)
        self.b = nn.Linear(3, 2)


class _Head(nn.Module):
    def __init__(self):
        super().__init__()
        self._cls_pred_branch = nn.Linear(2, 1)
        self._register_tokens = nn.Parameter(torch.randn(2, 2))


def _model():
    bb, head = _Backbone(), _Head()
    head_params = list(head.parameters())
    all_params = list(bb.parameters()) + head_params
    return bb, head, all_params, head_params


def _ids(seq):
    return [id(p) for p in seq]


# ---------------------------------------------------------------------------
# the happy path
# ---------------------------------------------------------------------------
def test_split_is_exact_ordered_and_lr_tagged():
    bb, head, all_params, head_params = _model()
    groups = split_of_head_param_groups(
        all_params, head_params, fake_lr=FAKE_LR, head_lr=HEAD_LR)

    assert len(groups) == 2
    backbone, headg = groups
    assert backbone["lr"] == FAKE_LR
    assert headg["lr"] == HEAD_LR
    # backbone FIRST -- param_groups[0] must keep meaning "the fake_lr group"
    assert _ids(backbone["params"]) == _ids(bb.parameters())
    assert set(_ids(headg["params"])) == set(_ids(head_params))
    # partition: disjoint, complete, nothing lost
    assert not set(_ids(backbone["params"])) & set(_ids(headg["params"]))
    assert (len(backbone["params"]) + len(headg["params"])
            == len(all_params))
    assert set(_ids(backbone["params"]) + _ids(headg["params"])) == set(
        _ids(all_params))


def test_lrs_are_floats_even_from_string_config_values():
    _bb, _h, all_params, head_params = _model()
    groups = split_of_head_param_groups(
        all_params, head_params, fake_lr="1e-6", head_lr="2e-4")
    assert groups[0]["lr"] == pytest.approx(1e-6)
    assert groups[1]["lr"] == pytest.approx(2e-4)
    assert isinstance(groups[0]["lr"], float)
    assert isinstance(groups[1]["lr"], float)


def test_backbone_order_is_preserved():
    """Optimizer state is positional in checkpoints; the backbone group
    must keep ``all_params`` order."""
    _bb, _h, all_params, head_params = _model()
    groups = split_of_head_param_groups(
        all_params, head_params, fake_lr=FAKE_LR, head_lr=HEAD_LR)
    expect = [p for p in all_params if id(p) not in set(_ids(head_params))]
    assert _ids(groups[0]["params"]) == _ids(expect)


# ---------------------------------------------------------------------------
# identity, not name
# ---------------------------------------------------------------------------
def test_matching_is_by_identity_not_by_name():
    """``fake_optimizer`` is built from a possibly DDP-wrapped module whose
    ``named_parameters`` carry a ``module.`` prefix, so a name-based match
    would silently produce an EMPTY head group. Give the head params names
    that share nothing with the backbone's and vice-versa: the split must
    still be right."""
    bb, head, all_params, head_params = _model()

    class _DDPish(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.module = inner

    wrapped_head = _DDPish(head)
    names = [n for n, _ in wrapped_head.named_parameters()]
    assert all(n.startswith("module.") for n in names)

    groups = split_of_head_param_groups(
        all_params, [p for _, p in wrapped_head.named_parameters()],
        fake_lr=FAKE_LR, head_lr=HEAD_LR)
    assert set(_ids(groups[1]["params"])) == set(_ids(head_params))


def test_value_equal_but_distinct_parameters_are_not_conflated():
    twin_a = nn.Parameter(torch.ones(2, 2))
    twin_b = nn.Parameter(torch.ones(2, 2))
    assert torch.equal(twin_a, twin_b) and twin_a is not twin_b
    all_params = [twin_a, twin_b]
    groups = split_of_head_param_groups(
        all_params, [twin_b], fake_lr=FAKE_LR, head_lr=HEAD_LR)
    assert _ids(groups[0]["params"]) == [id(twin_a)]
    assert _ids(groups[1]["params"]) == [id(twin_b)]


# ---------------------------------------------------------------------------
# frozen head params
# ---------------------------------------------------------------------------
def test_frozen_head_params_are_dropped_not_smuggled_in():
    bb, head, all_params, head_params = _model()
    frozen = head_params[0]
    frozen.requires_grad_(False)
    # ``all_params`` is already requires_grad-filtered upstream
    live_all = [p for p in all_params if p.requires_grad]
    groups = split_of_head_param_groups(
        live_all, head_params, fake_lr=FAKE_LR, head_lr=HEAD_LR)
    assert id(frozen) not in _ids(groups[0]["params"])
    assert id(frozen) not in _ids(groups[1]["params"])
    assert set(_ids(groups[1]["params"])) == {
        id(p) for p in head_params if p.requires_grad}


def test_all_head_params_frozen_raises():
    bb, head, all_params, head_params = _model()
    for p in head_params:
        p.requires_grad_(False)
    live_all = [p for p in all_params if p.requires_grad]
    with pytest.raises(RuntimeError) as ei:
        split_of_head_param_groups(
            live_all, head_params, fake_lr=FAKE_LR, head_lr=HEAD_LR)
    assert "no trainable discriminator-head" in str(ei.value)


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------
def test_empty_head_list_raises():
    _bb, _h, all_params, _hp = _model()
    with pytest.raises(RuntimeError) as ei:
        split_of_head_param_groups(
            all_params, [], fake_lr=FAKE_LR, head_lr=HEAD_LR)
    msg = str(ei.value)
    assert "gan_of_head_lr > 0" in msg
    assert "fake_lr" in msg          # names the silent failure it prevents


def test_head_params_absent_from_the_optimizer_list_raises():
    """``adding_cls_branch(attach_to_model=True)`` must run BEFORE
    ``_build_optimizer``; if it did not, the head is never stepped."""
    bb, head, _all, head_params = _model()
    backbone_only = list(bb.parameters())
    with pytest.raises(RuntimeError) as ei:
        split_of_head_param_groups(
            backbone_only, head_params, fake_lr=FAKE_LR, head_lr=HEAD_LR)
    msg = str(ei.value)
    assert "absent from the fake optimizer" in msg
    assert "attach_to_model=True" in msg
    assert f"{len(head_params)} disc-head" in msg


def test_partially_present_head_params_raise():
    bb, head, all_params, head_params = _model()
    dropped = head_params[-1]
    partial = [p for p in all_params if p is not dropped]
    with pytest.raises(RuntimeError) as ei:
        split_of_head_param_groups(
            partial, head_params, fake_lr=FAKE_LR, head_lr=HEAD_LR)
    assert "1 of " in str(ei.value)


def test_duplicate_head_param_in_the_optimizer_list_raises():
    """A parameter listed twice would be stepped twice per iteration."""
    bb, head, all_params, head_params = _model()
    dup = all_params + [head_params[0]]
    with pytest.raises(RuntimeError) as ei:
        split_of_head_param_groups(
            dup, head_params, fake_lr=FAKE_LR, head_lr=HEAD_LR)
    assert "duplicate" in str(ei.value)
    assert "stepped twice" in str(ei.value)


def test_empty_backbone_group_raises():
    """Every fake_score parameter matching the head means the DMD critic's
    own recipe would move onto the head LR."""
    _bb, _h, _all, head_params = _model()
    with pytest.raises(RuntimeError) as ei:
        split_of_head_param_groups(
            list(head_params), head_params,
            fake_lr=FAKE_LR, head_lr=HEAD_LR)
    assert "backbone param group is EMPTY" in str(ei.value)


# ---------------------------------------------------------------------------
# the groups are usable by a real optimizer
# ---------------------------------------------------------------------------
def test_groups_drive_a_real_adamw_at_two_different_rates():
    torch.manual_seed(0)
    bb, head, all_params, head_params = _model()
    groups = split_of_head_param_groups(
        all_params, head_params, fake_lr=1e-3, head_lr=1e-1)
    opt = torch.optim.AdamW(groups, lr=1e-3)
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["lr"] == 1e-3
    assert opt.param_groups[1]["lr"] == 1e-1

    before = [p.detach().clone() for p in all_params]
    for p in all_params:
        p.grad = torch.ones_like(p)
    opt.step()
    deltas = [
        float((p.detach() - b).abs().max())
        for p, b in zip(all_params, before)
    ]
    head_ids = set(_ids(head_params))
    bb_deltas = [d for p, d in zip(all_params, deltas)
                 if id(p) not in head_ids]
    hd_deltas = [d for p, d in zip(all_params, deltas) if id(p) in head_ids]
    assert min(hd_deltas) > max(bb_deltas) * 10, (
        f"head steps {hd_deltas} are not clearly larger than backbone "
        f"steps {bb_deltas} -- the LRs did not reach the parameters")


def test_every_parameter_is_stepped_exactly_once():
    """The dropped-parameter bug, restated as a measurement: give every
    parameter the same gradient and confirm none stayed put and none moved
    twice as far as its group-mate."""
    torch.manual_seed(0)
    _bb, _h, all_params, head_params = _model()
    groups = split_of_head_param_groups(
        all_params, head_params, fake_lr=1e-2, head_lr=1e-2)
    opt = torch.optim.SGD(groups, lr=1e-2)
    before = [p.detach().clone() for p in all_params]
    for p in all_params:
        p.grad = torch.ones_like(p)
    opt.step()
    moves = [float((p.detach() - b).abs().max())
             for p, b in zip(all_params, before)]
    assert all(m == pytest.approx(1e-2) for m in moves), moves


# ---------------------------------------------------------------------------
# the head module names must agree everywhere they are spelled
# ---------------------------------------------------------------------------
_EXPECTED_HEAD_MODULES = ("_cls_pred_branch", "_register_tokens",
                          "_gan_ca_blocks")


def _module_name_tuples(path):
    """Every tuple/list literal of >=2 strings that mentions
    ``_cls_pred_branch``, as spelled in ``path``."""
    with open(os.path.join(ROOT, path)) as f:
        tree = ast.parse(f.read())
    out = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.Tuple, ast.List)):
            vals = [e.value for e in n.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)]
            if "_cls_pred_branch" in vals:
                out.append(tuple(vals))
    return out


def test_of_head_module_names_agree_across_the_tree():
    """``_OF_HEAD_MODULE_NAMES`` is documented as "the ONE place they are
    spelled for the model"; the trainers spell their own copies. A drift
    makes the head group EMPTY or PARTIAL -- silently, at 8-node scale."""
    seen = {}
    for path in (
        "model/dmd_action_forcing.py",
        "trainer/causal_action_forcing_train.py",
        "trainer/causal_rolling_staircase_train.py",
    ):
        tuples = _module_name_tuples(path)
        assert tuples, f"{path}: no disc-head module-name tuple found"
        seen[path] = tuples
        for t in tuples:
            assert set(t) == set(_EXPECTED_HEAD_MODULES), (
                f"{path}: head module names {t} != {_EXPECTED_HEAD_MODULES}")

    # the af trainer also matches by NAME PREFIX; those must be the same
    # three names with a trailing dot.
    with open(os.path.join(
            ROOT, "trainer/causal_action_forcing_train.py")) as f:
        src = f.read()
    for m in _EXPECTED_HEAD_MODULES:
        assert f'"{m}."' in src, (
            f"the af trainer's _of_head_param_names prefix for {m} is gone")


def test_dmd_module_constant_matches():
    import importlib
    from unittest.mock import patch
    with patch.object(torch.cuda, "current_device", return_value=0):
        mod = importlib.import_module("model.dmd_action_forcing")
    assert set(mod._OF_HEAD_MODULE_NAMES) == set(_EXPECTED_HEAD_MODULES)


def test_the_rolling_trainer_still_calls_the_shared_splitter():
    """Both trainers were meant to funnel through this function; if a call
    site inlines its own split again, the asserted contract is lost."""
    with open(os.path.join(
            ROOT, "trainer/causal_rolling_staircase_train.py")) as f:
        src = f.read()
    assert re.search(r"split_of_head_param_groups\s*\(", src), (
        "causal_rolling_staircase_train no longer calls "
        "split_of_head_param_groups")
    assert "fake_lr=fake_lr" in src and "head_lr=" in src


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
