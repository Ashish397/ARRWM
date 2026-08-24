"""WP-PIXGAN (B1) / T3-A — trainer wiring: gate, construction, optimizer,
save + FAIL-LOUD resume, and the A18-D3 ``pix_*`` override registration.

Scope of the chunk under test (nothing else):
  * gate ``gan_pixel_texture_enabled`` (default False) — NOT a new
    ``gan_backbone`` value, and independent of ``gan_enabled``;
  * ``self.pixel_texture_disc`` (frozen contract 1) + its DDP attr;
  * ``self.pix_optimizer`` = ``torch.optim.Adam``, betas (0.0, 0.9),
    lr = ``pix_gan_lr`` (1e-5);
  * ``[mem-inventory]`` registration;
  * the checkpoint save block in
    ``trainer/causal_rolling_staircase_train.py`` under ITS OWN gate, and
    ``ActionForcingDMDTrainer._maybe_resume``, which must RAISE — never warn
    — on a missing or unclean pixel-critic restore (docs/WP_PIXGAN.md §3,
    "the known trap");
  * ``_maybe_resume``'s early-return admitting the pixel-critic case.

CPU-ONLY.  There is no GPU on the build node, so the two ``__init__`` blocks
are exercised by extracting their shipped source text and ``exec``-ing it
against a stub ``self`` (``world_size=1`` so the DDP branch is not taken —
that branch is verified structurally, not executed).  Everything else calls
the real trainer methods.

Run (the thread caps are NOT optional — nproc=144 here and an uncapped run
looks hung for >30 min):

    cd /scratch/u6ex/as1748.u6ex/ARRWM
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONPATH=. \
      /scratch/u6ex/as1748.u6ex/miniforge3/envs/arrwm/bin/python \
      -m pytest -q testing/test_pixgan_trainer_wiring.py
"""
import ast
import os
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.pixel_texture_disc import (  # noqa: E402
    PIX_DEFAULTS, PIX_GAN_BETAS, PIX_GAN_LR, PixelTextureDisc,
)
# Reuse T2's comment/docstring stripper verbatim rather than writing a second
# implementation of the same idea (testing/test_pixel_texture_disc.py:77).
from testing.test_pixel_texture_disc import code_only  # noqa: E402

# Wan's T5 wrapper evaluates ``torch.cuda.current_device()`` at import time
# even though this test never constructs a model (idiom borrowed from
# testing/test_r1_cadence_and_override_guard.py).
with patch.object(torch.cuda, "current_device", return_value=0):
    from trainer import causal_action_forcing_train as CAFT
    from trainer import causal_rolling_staircase_train as CRST

Trainer = CAFT.ActionForcingDMDTrainer
Parent = CRST.RollingStaircaseDMDTrainer

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_AF_SRC_PATH = os.path.join(_ROOT, "trainer", "causal_action_forcing_train.py")
_RS_SRC_PATH = os.path.join(
    _ROOT, "trainer", "causal_rolling_staircase_train.py")


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _af_source():
    return _read(_AF_SRC_PATH)


def _rs_source():
    return _read(_RS_SRC_PATH)


# ---------------------------------------------------------------------------
# Extracting the two ``__init__`` blocks so they can be executed on CPU.
#
# We slice the SHIPPED source between greppable anchors (never line numbers:
# the trainer is ~12.6k lines and under concurrent edit) and exec it.  If an
# anchor ever moves, these tests fail loudly rather than silently testing a
# stale copy.
# ---------------------------------------------------------------------------
_CTOR_START = '        self.gan_pixel_texture_enabled = bool(\n'
_CTOR_END = ('        # ------------------------------------------------'
             '------------------\n        # SC-DMD (Salt)')
_OPT_START = ('        self.pix_optimizer: Optional[torch.optim.Optimizer]'
              ' = None\n')
_OPT_END = '        # GAN warmup / disc-start schedule (used by the LADD path).'


def _slice(src, start, end):
    i = src.index(start)
    j = src.index(end, i)
    return textwrap.dedent(src[i:j])


def _ctor_block():
    return _slice(_af_source(), _CTOR_START, _CTOR_END)


def _opt_block():
    return _slice(_af_source(), _OPT_START, _OPT_END)


class _RecordingLogging:
    """Stand-in for the ``logging`` module that records every emitted line."""

    def __init__(self):
        self.records = []

    def info(self, msg, *args):
        self.records.append(("info", msg, args))

    def warning(self, msg, *args):
        self.records.append(("warning", msg, args))

    def error(self, msg, *args):
        self.records.append(("error", msg, args))


class _CtorStub:
    """The ``self`` the two shipped ``__init__`` blocks are exec'd against.

    The WP-PIXGAN members are DERIVED off the real class, never
    hand-maintained -- the same rule (and for the same reason) as
    ``testing/test_pixgan_trainer_supply.py``'s ``StubTrainer``.  A
    hand-picked list is two things that must agree with nothing enforcing
    it: this file used to pass a bare ``SimpleNamespace`` here, so when the
    A21 fix added ONE method (``_pix_resolve_cfg``) and the ctor started
    calling it at construction time, 8 tests died with an
    ``AttributeError`` that read like a product regression and was not.

    Everything WP-PIXGAN owns is named ``_pix_*``, ``PIX_*`` or
    ``*pixel_texture*`` (frozen contract 1 pins the last one), so bind by
    that rule and let the next added member arrive for free.  Anything the
    blocks read that does NOT match the rule is caught by
    ``test_ctor_stub_supplies_every_self_attribute_the_blocks_read``, which
    fails with a message that says what to add rather than an
    ``AttributeError`` from inside an ``exec``.
    """

    for _n in dir(Trainer):
        if (_n.startswith("_pix_") or _n.startswith("PIX_")
                or "pixel_texture" in _n):
            locals()[_n] = getattr(Trainer, _n)
    del _n

    def __init__(self, cfg):
        self.config = cfg
        self.device = torch.device("cpu")
        self.world_size = 1     # DDP branch not taken (no GPU on this node)
        self.local_rank = 0
        self.is_main_process = True
        self.gan_enabled = False  # pixel arm runs with the transition GAN OFF


# The ctor now RESOLVES + VALIDATES the whole ``pix_*`` block up front
# (``_pix_resolve_cfg``), so the stub config has to be one production would
# actually accept.  Two keys are load-bearing:
#
#   * ``pix_r1_gamma`` -- the module default 1.0 is MEASURED INERT and
#     ``_pix_resolve_r1_gamma`` refuses it by design.  A stub carrying it
#     would be a stub proving the trainer runs in a configuration that must
#     never launch (same reasoning as the supply suite's stub).
#   * ``pix_real_pool_windows`` -- the shipped 4096 (configs/
#     action_forcing_phase3_dmd.yaml) clears the A21 margin, so the resolve
#     does not fire its pool warning through the real ``logging``.
#
# Neither is a gate: the gate-off tests never reach the resolve, and the
# gate itself is still absent unless a test passes it.
_STUB_CFG = {
    "pix_r1_gamma": 200.0,
    "pix_real_pool_windows": 4096,
}


def _run_init_blocks(**cfg_kwargs):
    """Execute the shipped construction + optimizer blocks on a stub self.

    Returns ``(self_stub, recording_logging)``.
    """
    base = dict(_STUB_CFG)
    base.update(cfg_kwargs)
    cfg = SimpleNamespace(**base)
    stub = _CtorStub(cfg)
    log = _RecordingLogging()
    ns = {
        "self": stub,
        "cfg": cfg,
        "torch": torch,
        "logging": log,
        "Optional": CAFT.Optional if hasattr(CAFT, "Optional") else None,
        "DDP": CAFT.DDP,
    }
    if ns["Optional"] is None:
        import typing
        ns["Optional"] = typing.Optional
    exec(compile(_ctor_block(), "<pixgan-ctor>", "exec"), ns)
    exec(compile(_opt_block(), "<pixgan-opt>", "exec"), ns)
    return stub, log


# ---------------------------------------------------------------------------
# 1. Gate: default off, byte-identical off
# ---------------------------------------------------------------------------
def test_gate_defaults_to_false_when_config_is_silent():
    stub, _ = _run_init_blocks()
    assert stub.gan_pixel_texture_enabled is False


def test_gate_off_builds_no_module_no_optimizer():
    stub, log = _run_init_blocks(gan_pixel_texture_enabled=False)
    assert stub.pixel_texture_disc is None
    assert stub.pixel_texture_disc_ddp is None
    assert stub.pix_optimizer is None


def test_gate_off_emits_no_log_line():
    """Byte-identical off: not one new log key / line."""
    _, log = _run_init_blocks(gan_pixel_texture_enabled=False)
    assert log.records == []


def test_gate_off_consumes_no_rng():
    """Byte-identical off: the global torch RNG stream is untouched.

    Constructing the critic draws (conv init + the spectral-norm power
    vectors), so an accidentally-ungated build would shift every downstream
    draw in the run.
    """
    torch.manual_seed(1234)
    before = torch.get_rng_state().clone()
    _run_init_blocks(gan_pixel_texture_enabled=False)
    assert torch.equal(torch.get_rng_state(), before)


def test_gate_on_does_consume_rng_so_the_off_test_has_teeth():
    torch.manual_seed(1234)
    before = torch.get_rng_state().clone()
    _run_init_blocks(gan_pixel_texture_enabled=True)
    assert not torch.equal(torch.get_rng_state(), before)


def test_gate_is_not_a_gan_backbone_value():
    """``gan_backbone`` must still reject anything but ladd_teacher_feat."""
    src = _af_source()
    assert "gan_backbone must be 'ladd_teacher_feat'" in src
    # No pixel value was smuggled into the backbone selector.
    assert 'gan_backbone == "pixel' not in src
    assert 'gan_backbone", "pixel' not in src


def test_pixel_block_does_not_read_gan_enabled_as_a_gate():
    """The two critics are independent: the build must not require the
    transition GAN to be on."""
    block = _ctor_block() + _opt_block()
    for line in block.split("\n"):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert not stripped.startswith("if self.gan_enabled"), line


# ---------------------------------------------------------------------------
# 2. Construction + optimizer, gate on
# ---------------------------------------------------------------------------
def test_gate_on_builds_the_critic_under_the_frozen_attr_name():
    stub, _ = _run_init_blocks(gan_pixel_texture_enabled=True)
    assert isinstance(stub.pixel_texture_disc, PixelTextureDisc)
    # frozen contract 1: model/disc_holdout_probe.py looks up this exact name
    assert hasattr(stub, "pixel_texture_disc")
    # single-process: no DDP wrapper
    assert stub.pixel_texture_disc_ddp is None


def test_critic_is_fp32_and_in_train_mode():
    stub, _ = _run_init_blocks(gan_pixel_texture_enabled=True)
    d = stub.pixel_texture_disc
    assert d.training is True
    for p in d.parameters():
        assert p.dtype is torch.float32
        assert p.device.type == "cpu"


def test_optimizer_is_plain_adam_with_spec_betas_and_lr():
    stub, _ = _run_init_blocks(gan_pixel_texture_enabled=True)
    opt = stub.pix_optimizer
    assert type(opt) is torch.optim.Adam          # NOT AdamW (§5.2)
    assert not isinstance(opt, torch.optim.AdamW)
    g = opt.param_groups[0]
    assert tuple(g["betas"]) == tuple(PIX_GAN_BETAS) == (0.0, 0.9)
    assert g["lr"] == PIX_GAN_LR == 1e-5


def test_optimizer_lr_and_betas_are_config_overridable():
    stub, _ = _run_init_blocks(
        gan_pixel_texture_enabled=True, pix_gan_lr=3e-5,
        pix_gan_betas=[0.5, 0.99],
    )
    g = stub.pix_optimizer.param_groups[0]
    assert g["lr"] == pytest.approx(3e-5)
    assert tuple(g["betas"]) == (0.5, 0.99)


def test_optimizer_covers_every_trainable_parameter():
    stub, _ = _run_init_blocks(gan_pixel_texture_enabled=True)
    owned = {id(p) for grp in stub.pix_optimizer.param_groups
             for p in grp["params"]}
    trainable = {id(p) for p in stub.pixel_texture_disc.parameters()
                 if p.requires_grad}
    assert owned == trainable and owned


def test_no_trainable_params_raises_like_the_ladd_path():
    """Mirror of the LADD build's RuntimeError guard."""
    stub, _ = _run_init_blocks(gan_pixel_texture_enabled=True)
    for p in stub.pixel_texture_disc.parameters():
        p.requires_grad_(False)
    stub.pix_optimizer = None
    ns = {"self": stub, "cfg": stub.config, "torch": torch,
          "logging": _RecordingLogging(), "DDP": CAFT.DDP}
    import typing
    ns["Optional"] = typing.Optional
    with pytest.raises(RuntimeError, match="no trainable parameters"):
        exec(compile(_opt_block(), "<pixgan-opt>", "exec"), ns)


def test_gate_on_logs_a_main_process_param_count_line():
    _, log = _run_init_blocks(gan_pixel_texture_enabled=True)
    msgs = " ".join(r[1] for r in log.records)
    assert "Pixel-texture discriminator built" in msgs
    assert "Pixel-texture optimizer built" in msgs


def _self_attrs_read_by_the_blocks():
    """Every ``self.X`` the two shipped blocks READ without first setting."""
    tree = ast.parse(_ctor_block() + "\n" + _opt_block())
    stored, loaded = set(), []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"):
            if isinstance(node.ctx, ast.Store):
                stored.add(node.attr)
            else:
                loaded.append(node.attr)
    # ``getattr(self, "x", default)`` reads are tolerant by construction.
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) == 3
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "self"
                and isinstance(node.args[1], ast.Constant)):
            stored.add(node.args[1].value)
    return sorted({a for a in loaded if a not in stored})


def test_ctor_stub_supplies_every_self_attribute_the_blocks_read():
    """The stub cannot silently fall behind the shipped ctor.

    This is the guard for the failure this suite already suffered: a new
    trainer member (``_pix_resolve_cfg``) appeared in the ctor, the stub
    did not have it, and 8 tests failed with an ``AttributeError`` raised
    from inside an ``exec`` -- indistinguishable at a glance from a real
    product regression.  ``_CtorStub`` derives every ``_pix_*`` / ``PIX_*``
    / ``*pixel_texture*`` member off the real class, which covers
    everything WP-PIXGAN owns; this test covers the rest, and fails with a
    name instead of a stack trace.
    """
    stub = _CtorStub(SimpleNamespace(**_STUB_CFG))
    missing = [a for a in _self_attrs_read_by_the_blocks()
               if not hasattr(stub, a)]
    assert not missing, (
        "the shipped __init__ blocks read self.%s, which _CtorStub does not "
        "supply. If the name is WP-PIXGAN's it should already be derived "
        "(_pix_*/PIX_*/*pixel_texture*) -- check the naming; otherwise add "
        "it to _CtorStub.__init__." % ", self.".join(missing)
    )


def test_the_pix_members_are_derived_not_listed():
    """The bind-by-rule loop is the point; a literal list must not creep in.

    Every ``_pix_*`` / ``PIX_*`` / ``*pixel_texture*`` member of the real
    trainer must be reachable on the stub, with nothing to update by hand
    when one is added.
    """
    owned = [n for n in dir(Trainer)
             if n.startswith("_pix_") or n.startswith("PIX_")
             or "pixel_texture" in n]
    assert "_pix_resolve_cfg" in owned          # the member that broke this
    stub = _CtorStub(SimpleNamespace(**_STUB_CFG))
    missing = [n for n in owned if not hasattr(stub, n)]
    assert not missing, missing


def test_ddp_wrapper_branch_is_present_and_named_consistently():
    """The DDP path cannot execute here (no GPU); pin it structurally."""
    block = _ctor_block()
    assert "self.pixel_texture_disc_ddp = DDP(" in block
    assert "find_unused_parameters=False" in block
    assert "broadcast_buffers=False" in block
    assert "if self.world_size > 1:" in block


# ---------------------------------------------------------------------------
# 3. [mem-inventory] registration
# ---------------------------------------------------------------------------
def test_mem_inventory_lists_the_pixel_attrs_and_the_gate():
    src = _af_source()
    inv = src.split("names_trainer = [")[1].split("]")[0]
    for name in ("pixel_texture_disc", "pixel_texture_disc_ddp",
                 "pix_optimizer"):
        assert f'"{name}"' in inv, name
    flags = src.split("flags = [")[1].split("]")[0]
    assert '"gan_pixel_texture_enabled"' in flags


# ---------------------------------------------------------------------------
# 4. Checkpoint save — under its OWN gate
# ---------------------------------------------------------------------------
class _TinyGen(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2, 2)


def _trainer_stub(tmp_path, disc=None, opt=None, pix_gate=False, **flags):
    """A real ``ActionForcingDMDTrainer`` instance with only the attributes
    the code under test touches.

    Deliberately a real instance, not a SimpleNamespace:
      * ``_maybe_resume`` uses zero-arg ``super()``, which requires
        ``isinstance(self, ActionForcingDMDTrainer)``;
      * ``log_dir`` is a read-only PROPERTY derived from ``config.log_dir`` /
        ``config.run_name`` -- it cannot be assigned, so the checkpoint
        directory is steered through the config, and the real
        ``_checkpoint_path`` naming is exercised rather than faked.
    """
    t = Trainer.__new__(Trainer)
    t.config = SimpleNamespace(
        log_dir=str(tmp_path),
        run_name="run",
        keep_last_n_checkpoints=0,      # no pruning during the test
        auto_resume=True,
        gan_pixel_texture_enabled=pix_gate,
    )
    gen = _TinyGen()
    t.is_main_process = True
    t.step = 1
    t.generator_ddp = None
    t.optimizer = torch.optim.SGD(gen.parameters(), lr=0.1)
    t.config_path = "cfg.yaml"
    t.fake_score_updates_enabled = False
    t.generator_ema = None
    t.model = SimpleNamespace(
        generator=SimpleNamespace(model=gen),
        action_projection=None,
    )
    # gate flags read by the _maybe_resume early return
    t.action_critic_loss_active = flags.get("action_critic_loss_active", False)
    t.gan_enabled = flags.get("gan_enabled", False)
    t.real_teacher_train_online = flags.get("real_teacher_train_online", False)
    t.state_probe_aux_active = flags.get("state_probe_aux_active", False)
    t.gan_pixel_texture_enabled = pix_gate
    # LADD critic absent throughout: the pixel arm runs with the transition
    # GAN OFF, which is exactly the configuration the trap hides in.
    t.r3gan_disc = None
    t.r3gan_disc_ddp = None
    t.r3gan_optimizer = None
    t.pixel_texture_disc = disc
    t.pixel_texture_disc_ddp = None
    t.pix_optimizer = opt
    return t


def _ckpt_path(t):
    return t._checkpoint_path(t.step)


def _trained_critic():
    """A critic + Adam with non-trivial (non-zero) optimizer state."""
    torch.manual_seed(7)
    disc = PixelTextureDisc()
    opt = torch.optim.Adam(disc.parameters(), lr=PIX_GAN_LR,
                           betas=PIX_GAN_BETAS)
    x = torch.randn(1, 3, 32, 32)
    disc(x).mean().backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    return disc, opt


def test_save_writes_nothing_when_the_gate_is_off(tmp_path):
    disc, opt = _trained_critic()
    t = _trainer_stub(tmp_path, disc, opt, pix_gate=False)
    Parent._save_checkpoint(t)
    state = torch.load(_ckpt_path(t), map_location="cpu", weights_only=False)
    assert "pixel_texture_disc" not in state
    assert "pix_optimizer" not in state


def test_save_writes_both_keys_under_its_own_gate_with_gan_enabled_false(
        tmp_path):
    """The whole point: ``gan_enabled`` is False and the critic still saves."""
    disc, opt = _trained_critic()
    t = _trainer_stub(tmp_path, disc, opt, pix_gate=True)
    assert t.gan_enabled is False
    Parent._save_checkpoint(t)
    state = torch.load(_ckpt_path(t), map_location="cpu", weights_only=False)
    assert "pixel_texture_disc" in state
    assert "pix_optimizer" in state
    assert "r3gan_discriminator" not in state


def test_save_and_restore_key_names_agree_on_both_sides():
    save_block = _rs_source().split(
        "# WP-PIXGAN — pixel-texture critic + its Adam, under ITS OWN GATE.")[1]
    save_block = save_block.split("# ForwardNoiser")[0]
    saved = {k for k in ("pixel_texture_disc", "pix_optimizer")
             if f'state["{k}"]' in save_block}
    resume_block = _af_source().split(
        "# WP-PIXGAN — pixel-texture critic + its Adam. FAIL LOUD.")[1]
    resume_block = resume_block.split("# ForwardNoiser")[0]
    restored = {k for k in ("pixel_texture_disc", "pix_optimizer")
                if f'"{k}"' in resume_block}
    assert saved == restored == {"pixel_texture_disc", "pix_optimizer"}


# ---------------------------------------------------------------------------
# 5. Resume — FAIL LOUD
# ---------------------------------------------------------------------------
def _write_ckpt(t, state):
    path = _ckpt_path(t)
    torch.save(state, path)
    return path


def _run_resume(t):
    """Call the REAL ActionForcingDMDTrainer._maybe_resume.

    The parent's ``_maybe_resume`` (reached via zero-arg ``super()``) is
    stubbed out because it rebuilds the generator/optimizer; everything from
    the early-return onward is the shipped code.
    """
    with patch.object(Parent, "_maybe_resume", lambda self: None):
        Trainer._maybe_resume(t)


def test_resume_round_trips_critic_and_optimizer_faithfully(tmp_path):
    """Round-trip is checked against what is ON DISK, not against the live
    saved module.

    TRAP, learned the hard way: ``PixelTextureDisc`` is spectral-normalised,
    and merely *reading* ``disc.conv1.weight`` runs the parametrization,
    which in train mode does a power iteration and mutates the ``_u`` / ``_v``
    buffers IN PLACE. Comparing the restored critic against a still-live
    reference therefore fails for reasons that have nothing to do with the
    checkpoint. ``state_dict()`` itself is safe (it exposes
    ``parametrizations.weight.original``, never the composed ``weight``).
    """
    saved_disc, saved_opt = _trained_critic()
    saver = _trainer_stub(tmp_path, saved_disc, saved_opt, pix_gate=True)
    Parent._save_checkpoint(saver)
    on_disk = torch.load(_ckpt_path(saver), map_location="cpu",
                         weights_only=False)
    ckpt_sd = on_disk["pixel_texture_disc"]
    ckpt_opt = on_disk["pix_optimizer"]

    # A fresh, differently-initialised critic + a pristine Adam.
    torch.manual_seed(99)
    live_disc = PixelTextureDisc()
    live_opt = torch.optim.Adam(live_disc.parameters(), lr=PIX_GAN_LR,
                                betas=PIX_GAN_BETAS)
    orig_key = next(k for k in ckpt_sd if k.endswith("weight.original")
                    and "conv1" in k)
    assert not torch.allclose(live_disc.state_dict()[orig_key],
                              ckpt_sd[orig_key])
    assert live_opt.state_dict()["state"] == {}

    t = _trainer_stub(tmp_path, live_disc, live_opt, pix_gate=True)
    _run_resume(t)

    live_sd = live_disc.state_dict()
    assert set(live_sd) == set(ckpt_sd) and live_sd
    for k in ckpt_sd:
        assert torch.equal(live_sd[k], ckpt_sd[k]), k

    live_state = live_opt.state_dict()["state"]
    assert set(live_state) == set(ckpt_opt["state"]) and live_state
    for k in ckpt_opt["state"]:
        for term in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(live_state[k][term],
                               ckpt_opt["state"][k][term]), (k, term)
    assert live_opt.state_dict()["param_groups"] == ckpt_opt["param_groups"]


def test_resume_raises_when_the_critic_key_is_missing(tmp_path):
    """The known trap: a pre-pixel checkpoint must NOT silently re-init."""
    disc, opt = _trained_critic()
    t = _trainer_stub(tmp_path, disc, opt, pix_gate=True)
    _write_ckpt(t, {"step": 1, "generator": {}})
    with pytest.raises(RuntimeError) as ei:
        _run_resume(t)
    msg = str(ei.value)
    assert "pixel_texture_disc" in msg
    assert "re-initialise" in msg or "reinitialise" in msg
    assert "warmup" in msg


def test_resume_raises_when_the_optimizer_key_is_missing(tmp_path):
    disc, opt = _trained_critic()
    t = _trainer_stub(tmp_path, disc, opt, pix_gate=True)
    _write_ckpt(t, {"step": 1, "pixel_texture_disc": disc.state_dict()})
    with pytest.raises(RuntimeError, match="pix_optimizer"):
        _run_resume(t)


def test_resume_raises_on_a_shape_mismatched_state_dict(tmp_path):
    """NOT the LADD path's tolerant shape-drop: §4 geometry is frozen."""
    disc, opt = _trained_critic()
    bad = {k: v.clone() for k, v in disc.state_dict().items()}
    key = next(k for k in sorted(bad) if "conv1" in k and bad[k].dim() == 4)
    bad[key] = torch.zeros(1, 1, 1, 1)
    t = _trainer_stub(tmp_path, disc, opt, pix_gate=True)
    _write_ckpt(t, {"step": 1, "pixel_texture_disc": bad,
                    "pix_optimizer": opt.state_dict()})
    with pytest.raises(RuntimeError) as ei:
        _run_resume(t)
    assert "did not load cleanly" in str(ei.value)


def test_resume_raises_on_a_missing_tensor_inside_the_state_dict(tmp_path):
    disc, opt = _trained_critic()
    bad = dict(disc.state_dict())
    bad.pop(sorted(bad)[0])
    t = _trainer_stub(tmp_path, disc, opt, pix_gate=True)
    _write_ckpt(t, {"step": 1, "pixel_texture_disc": bad,
                    "pix_optimizer": opt.state_dict()})
    with pytest.raises(RuntimeError):
        _run_resume(t)


def test_resume_raises_on_an_unexpected_key_in_the_state_dict(tmp_path):
    disc, opt = _trained_critic()
    bad = dict(disc.state_dict())
    bad["conv9.weight"] = torch.zeros(1)
    t = _trainer_stub(tmp_path, disc, opt, pix_gate=True)
    _write_ckpt(t, {"step": 1, "pixel_texture_disc": bad,
                    "pix_optimizer": opt.state_dict()})
    with pytest.raises(RuntimeError):
        _run_resume(t)


def test_resume_raises_on_an_unloadable_optimizer_state(tmp_path):
    disc, opt = _trained_critic()
    t = _trainer_stub(tmp_path, disc, opt, pix_gate=True)
    _write_ckpt(t, {
        "step": 1,
        "pixel_texture_disc": disc.state_dict(),
        "pix_optimizer": {"state": {}, "param_groups": []},   # wrong arity
    })
    with pytest.raises(RuntimeError, match="pix_optimizer"):
        _run_resume(t)


def test_resume_does_not_merely_warn(tmp_path, caplog):
    """A warn-and-continue would leave a fresh critic silently in place.

    This is the behaviour the LADD path above deliberately has and the pixel
    path deliberately does NOT.
    """
    disc, opt = _trained_critic()
    t = _trainer_stub(tmp_path, disc, opt, pix_gate=True)
    _write_ckpt(t, {"step": 1})
    with pytest.raises(RuntimeError):
        _run_resume(t)
    assert not [r for r in caplog.records
                if r.levelname in ("WARNING", "ERROR")
                and "pixel_texture_disc" in r.getMessage()]


def test_ladd_path_is_still_tolerant_so_we_only_changed_our_own(tmp_path):
    """Guard against 'fixing' the LADD restore into fail-loud by accident."""
    fn = _maybe_resume_source()
    resume_src = fn.split("if self.gan_enabled and self.r3gan_disc")[1]
    resume_src = resume_src.split("# WP-PIXGAN")[0]
    assert "strict=False" in resume_src
    assert "shape_dropped" in resume_src
    assert "raise" not in resume_src


# ---------------------------------------------------------------------------
# 6. The early-return must admit the pixel-critic case
# ---------------------------------------------------------------------------
def test_early_return_admits_the_pixel_case(tmp_path):
    """With ONLY the pixel gate on, the restore code must actually run.

    Proven behaviourally: a checkpoint with no pixel key must raise.  Before
    the early-return fix this returned silently and the trap reappeared
    through the back door.
    """
    disc, opt = _trained_critic()
    t = _trainer_stub(tmp_path, disc, opt, pix_gate=True)
    _write_ckpt(t, {"step": 1})
    assert t.gan_enabled is False
    assert t.action_critic_loss_active is False
    assert t.real_teacher_train_online is False
    assert t.state_probe_aux_active is False
    with pytest.raises(RuntimeError):
        _run_resume(t)


def test_early_return_still_fires_when_everything_is_off(tmp_path):
    disc, opt = _trained_critic()
    t = _trainer_stub(tmp_path, disc, opt, pix_gate=False)
    _write_ckpt(t, {"step": 1})
    assert t.gan_pixel_texture_enabled is False
    _run_resume(t)      # returns silently — nothing to restore
    # the critic was left exactly as constructed
    assert t.pixel_texture_disc is disc


def _maybe_resume_source():
    """Source text of ``ActionForcingDMDTrainer._maybe_resume`` only.

    By AST, so it cannot accidentally slice the ``__init__`` optimizer-build
    site, which shares the ``if self.gan_enabled and self.r3gan_disc ...``
    spelling.
    """
    src = _af_source()
    tree = ast.parse(src)
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name == "ActionForcingDMDTrainer":
            for n in cls.body:
                if isinstance(n, ast.FunctionDef) and n.name == "_maybe_resume":
                    return ast.get_source_segment(src, n)
    raise AssertionError("ActionForcingDMDTrainer._maybe_resume not found")


def _maybe_resume_early_return_test():
    """Return the AST of the guard expression in ``_maybe_resume``'s
    early-return ``if not (...)``.

    Located by AST, never by string slicing: the trainer is ~12.6k lines and
    the condition is a multi-line ``not (a or b or ...)`` with comments
    interleaved, which no naive ``.split()`` survives.
    """
    tree = ast.parse(_af_source())
    fn = None
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name == "ActionForcingDMDTrainer":
            for n in cls.body:
                if isinstance(n, ast.FunctionDef) and n.name == "_maybe_resume":
                    fn = n
    assert fn is not None, "ActionForcingDMDTrainer._maybe_resume not found"
    for node in fn.body:
        if (isinstance(node, ast.If)
                and isinstance(node.test, ast.UnaryOp)
                and isinstance(node.test.op, ast.Not)
                and any(isinstance(s, ast.Return) for s in node.body)):
            return node.test
    raise AssertionError("no `if not (...): return` early-return found")


def test_early_return_condition_names_the_gate_in_source():
    cond = ast.unparse(_maybe_resume_early_return_test())
    assert "gan_pixel_texture_enabled" in cond, cond
    # and the pre-existing disjuncts are still there (nothing was replaced)
    for other in ("action_critic_loss_active", "gan_enabled",
                  "real_teacher_train_online", "state_probe_aux_active"):
        assert other in cond, other


# ---------------------------------------------------------------------------
# 7. A18-D3 — every ``pix_*`` key is sourced by the override guard
# ---------------------------------------------------------------------------
def _guard_ast_scan(src_to_scan):
    """Run the trainer's own A18-D3 AST scan in isolation."""
    tree = ast.parse(_af_source())
    keep = []
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == "_ast_config_sourced_keys":
            keep.append(n)
        elif isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id.startswith("_OVERRIDE_GUARD")
                for t in n.targets):
            keep.append(n)
    ns = {"ast": ast}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "<guard>", "exec"), ns)
    return ns, ns["_ast_config_sourced_keys"](
        ns["_OVERRIDE_GUARD_PREFIXES"], src_to_scan)


def test_pix_defaults_are_NOT_auto_harvested_from_the_critic_module():
    """Measured, not assumed — this is why explicit registration exists.

    The DEFAULTS harvest only fires for a module that itself reads config
    with a non-literal key; model/pixel_texture_disc.py reads config not at
    all (by design), so its ``PIX_DEFAULTS`` is never harvested.
    """
    _, found = _guard_ast_scan(_read(
        os.path.join(_ROOT, "model", "pixel_texture_disc.py")))
    assert {k for k in found if k.startswith("pix_")} == set()


def test_every_pix_defaults_key_is_sourced_from_the_trainer():
    _, found = _guard_ast_scan(_af_source())
    missing = sorted(set(PIX_DEFAULTS) - found)
    assert missing == [], missing


def test_the_gate_key_itself_is_sourced():
    _, found = _guard_ast_scan(_af_source())
    assert "gan_pixel_texture_enabled" in found


def test_no_pix_r2_symbol_anywhere_in_the_wiring():
    """R2 is deleted and must not be resurrected (WP_PIXGAN §2).

    Scans CODE ONLY. The wiring comments *correctly* state that R2 does not
    exist and must never be added back -- that prose is doing real work, and
    a raw-source grep would force it to be deleted to keep the tripwire
    green. Exactly the false positive T2 already hit and solved, so this
    reuses T2's ``code_only`` helper rather than inventing a second one.
    """
    joined = code_only(_af_source()) + code_only(_rs_source())
    assert "pix_r2" not in joined.lower()
    # ... and the prose really does discuss R2, else this test is vacuous.
    assert "pix_r2" in (_af_source() + _rs_source()).lower()


def test_the_r2_tripwire_actually_fires_on_a_planted_violation():
    """A guard nobody has seen trip is not a guard."""
    planted = (
        '"""Docstring mentioning pix_r2_sigma in prose only."""\n'
        "# a comment naming pix_r2_every_n\n"
        "pix_r2_gamma = 0.0\n"
    )
    code = code_only(planted)
    assert "pix_r2_gamma" in code            # the real assignment survives
    assert "pix_r2_sigma" not in code        # docstring stripped
    assert "pix_r2_every_n" not in code      # comment stripped
    # A banned name smuggled through a string literal must still be caught.
    smuggled = 'def h(o):\n    return getattr(o, "pix_r2_gamma")\n'
    assert "pix_r2_gamma" in code_only(smuggled)


def test_pix_gan_weight_has_no_default_and_is_not_hardcoded():
    """§5.5 — no default, and absence must stay LOUD.

    T3-C UPDATE. This test used to assert that NO ``getattr(cfg,
    "pix_gan_weight", ...)`` site existed at all, which was true while
    nothing read the knob. T3-C must read it (the G-term cannot be weighted
    otherwise), so the blanket ban is replaced by the invariant it was
    standing in for, and which is the thing §5.5 actually asks for:

      * every read passes ``None`` as the getattr default -- never a
        number, so no constant can be inherited by accident;
      * the value is turned into a float ONLY through
        ``resolve_gan_weight(..., strict=True)``, which raises on ``None``;
      * neither the withdrawn 0.03 nor §24's LADD-arm weights (0.65-0.80,
        0.054 -- different domain, reduction, loss form and critic) appear
        as a default anywhere.
    """
    import re

    joined = _af_source() + _rs_source()
    assert "pix_gan_weight = 0.03" not in joined
    reads = re.findall(
        r'getattr\(\s*(?:cfg|self\.config)\s*,\s*"pix_gan_weight"\s*,'
        r'\s*([^)\n]*)\)',
        joined,
    )
    assert reads, "T3-C reads the knob; if that stopped, the G-term is dead"
    assert all(d.strip() == "None" for d in reads), reads
    assert "resolve_gan_weight" in joined
    for banned in ("0.03", "0.65", "0.80", "0.054"):
        assert f'"pix_gan_weight", {banned}' not in joined
        assert f"pix_gan_weight = {banned}" not in joined
