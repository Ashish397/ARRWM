"""WP-SURROGATE (B3) — the CONSUMPTION + DISTILLATION wiring, landed
2026-08-24 by MAIN under researcher order (docs/TASK_SURROGATE_CONSUMPTION.md).

Scope of the chunk under test (nothing else — the build step is
``testing/test_surrogate_trainer_wiring.py``, the module itself is
``testing/test_latent_texture_critic.py``, and B1's direct path is
``testing/test_pixgan_t3c_gterm.py``; this file touches NONE of those):

  * the surrogate branch inside ``_compute_pixel_texture_g_loss`` —
    alternatives-never-summed (returns before the decode path), the
    ``(weighted, raw, logs)`` contract (``raw`` UNWEIGHTED so the A7
    weight-probe protocol works unchanged), fail-loud on enabled+missing,
    applied-weight echoes read the runtime value;
  * ``_maybe_run_surrogate_distillation`` — gate-off byte-identical,
    regime flags (no-teacher / warmup / empty-pool) instead of silence,
    fake crops mask-selected THEN detached, real crops drawn READ-ONLY
    from ``_pix_real_pool`` (option (a): NEVER ``_pix_draw_reals``, whose
    ``_pix_recent_frames`` ring is A20 state owned by the D-loop), a
    distinct sync-generator salt (13; D=0, G=11), origins passed through;
  * the FAIL-LOUD save/resume blocks (save in
    ``trainer/causal_rolling_staircase_train.py``, restore in
    ``_maybe_resume``), exercised by anchor-extraction + exec — never by
    line number, the trainer is edited concurrently by other packages;
  * cadence, END TO END: the real distiller (tiny critic via
    ``build_from_config``) driven through the real wiring asserts
    ``n_teacher_refresh == ceil(steps / pix_teacher_refresh_every)`` — the
    checkpoint-invariant counter, never raw teacher-call counts (the
    module's own 2x-under-checkpointing finding).

Every "must not happen" guard here has a planted companion proving the
guard fires (WP_PIXGAN §14: a guard nobody has seen trip is not evidence).

CPU-ONLY. Reuses WP-PIXGAN's ``StubTrainer`` (read-only import — their
file is not edited) so the pix-side helpers under these tests are the
SHIPPED methods, not re-implementations. Thread caps are NOT optional
(nproc=144 here):

    cd /scratch/u6ex/as1748.u6ex/ARRWM
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONPATH=. \
      /scratch/u6ex/as1748.u6ex/miniforge3/envs/arrwm/bin/python \
      -m pytest -q testing/test_surrogate_consumption_wiring.py
"""
import math
import os
import random
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with patch.object(torch.cuda, "current_device", return_value=0):
    from trainer import causal_action_forcing_train as CAFT

from testing.test_pixgan_trainer_supply import (  # noqa: E402
    LAT_C, StubTrainer, method_code,
)

Trainer = CAFT.ActionForcingDMDTrainer

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_AF_SRC_PATH = os.path.join(_ROOT, "trainer", "causal_action_forcing_train.py")
_RS_SRC_PATH = os.path.join(
    _ROOT, "trainer", "causal_rolling_staircase_train.py")


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# Stub: WP-PIXGAN's StubTrainer (their shipped pix helpers) + the surrogate
# wiring under test bound on top. gan_warmup_steps=0 so ``_pix_gen_weight``
# resolves without ``_gan_warmup_shape_apply`` (a generator-schedule helper
# outside this file's scope); pix_gan_weight is required-by-design
# (resolve_gan_weight strict), so the stub supplies one explicitly.
# ---------------------------------------------------------------------------
class _SurStub(StubTrainer):
    _maybe_run_surrogate_distillation = (
        Trainer._maybe_run_surrogate_distillation
    )
    # Added 2026-08-26 alongside ``_RecDistiller.should_grad_check``.
    # ``_compute_pixel_texture_g_loss``'s surrogate branch takes its
    # critic through ``_surrogate_g_snapshot_critic`` (the guard against
    # the in-place-Adam version-counter hazard); that indirection landed
    # after this stub was written, so three consumption tests died on
    # AttributeError before reaching their assertions. Bind the SHIPPED
    # helper rather than stubbing it out, so the tests keep exercising
    # the real snapshot logic.
    _surrogate_g_snapshot_critic = Trainer._surrogate_g_snapshot_critic

    def __init__(self, **cfg):
        cfg.setdefault("pix_gan_weight", 0.1)
        super().__init__(**cfg)
        self.gan_warmup_steps = 0
        self.surrogate_critic_enabled = False
        self.latent_texture_critic = None
        self.latent_critic_optimizer = None
        self.latent_texture_distiller = None
        self._pix_real_pool = []


class _RecDistiller:
    """Records every ``step`` kwargs verbatim; returns a fixed log dict."""

    def __init__(self):
        self.calls = []

    def step(self, **kw):
        self.calls.append(kw)
        return {"train/surrogate_loss_total": 0.5}

    # Added 2026-08-26. The shipped trainer calls this right after
    # ``step`` to decide whether to run the periodic direct-vs-surrogate
    # gradient audit (``surrogate_grad_check_every``). That call site was
    # wired up after this stub was written, so every test in this file
    # died on AttributeError before reaching its own assertions -- 9
    # RED tests that were reporting a stale stub, not a real defect.
    def should_grad_check(self, current_step):
        return False


class _StubCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, z, latent_origin=None):
        return self.scale * z.float().mean(dim=2)


def _info(n_lat=6, requires_grad=False):
    z = torch.randn(1, n_lat, LAT_C, 60, 104)
    if requires_grad:
        z.requires_grad_(True)
    return {"flash_dmd_gan_x0": z}


def _pool_entry(gen, y0=3):
    return {
        "lat": torch.randn(2, LAT_C, 24, 32, generator=gen).to(torch.float16),
        "band": 0, "y0": int(y0), "x0": 1,
        "ride": "ride_a", "start": 0, "nf": 2,
        "uid": ("ride_a", 0, int(y0), 1),
    }


def _stub_with_distiller(**cfg):
    stub = _SurStub(**cfg)
    stub.surrogate_critic_enabled = True
    stub.latent_texture_critic = _StubCritic()
    stub.latent_texture_distiller = _RecDistiller()
    stub.latent_critic_optimizer = torch.optim.Adam(
        stub.latent_texture_critic.parameters(), lr=1e-4, betas=(0.0, 0.9))
    g = torch.Generator().manual_seed(0)
    stub._pix_real_pool = [_pool_entry(g, y0=i % 30) for i in range(8)]
    return stub


def _rng_fingerprint():
    return (
        torch.random.get_rng_state().clone(),
        random.getstate(),
        np.random.get_state()[1].copy(),
    )


def _same_rng(a, b):
    return (torch.equal(a[0], b[0]) and a[1] == b[1]
            and np.array_equal(a[2], b[2]))


# ---------------------------------------------------------------------------
# 1. Source-level commitments (the two PIXGAN tripwires + salt discipline).
# ---------------------------------------------------------------------------
def test_distillation_never_calls_pix_draw_reals():
    """Option (a) is a commitment, not a preference: ``_pix_draw_reals``
    mutates the ``_pix_recent_frames`` reuse ring, so a second caller
    corrupts A20's ``pix_real_reuse_frac`` telemetry."""
    code = method_code(Trainer._maybe_run_surrogate_distillation)
    assert "_pix_draw_reals" not in code
    assert "_pix_recent_frames" not in code
    assert "_pix_real_pool" in code  # the read-only supply it uses instead


def test_distillation_reads_no_pix_knob_outside_the_resolver():
    """PIXGAN tripwire 1: resolver-owned ``pix_*`` knobs are read ONLY via
    ``_pix_resolve_cfg()`` — a second read site with its own default is the
    exact seam class (§5b) both packages were bitten by."""
    code = method_code(Trainer._maybe_run_surrogate_distillation)
    for pat in ('getattr(cfg, "pix_', 'getattr(self.config, "pix_',
                'config, "pix_'):
        assert pat not in code, pat
    assert "_pix_resolve_cfg" in code


def test_planted_violation_would_be_caught():
    planted = 'x = getattr(self.config, "pix_crops_per_step", 4)'
    assert 'getattr(self.config, "pix_' in planted  # the guard has teeth


def test_three_salts_are_distinct():
    """D=0 (default), G=11, distiller=13 — three independent crop streams
    off one rank-synced generator family."""
    d = method_code(Trainer._maybe_run_surrogate_distillation)
    g = method_code(Trainer._compute_pixel_texture_g_loss)
    assert "salt=13" in d and "salt=11" not in d
    assert "salt=11" in g and "salt=13" not in g


def test_consumption_branch_returns_before_the_direct_path():
    """Alternatives, never summed: within the surrogate branch there is a
    return, and the direct decode (``_pix_decode_crops_grad``) appears only
    AFTER the branch."""
    # NOTE: scan post-``ast.unparse`` code — string quoting is normalised
    # (single quotes) and tuple returns gain parens, so match bare names.
    code = method_code(Trainer._compute_pixel_texture_g_loss)
    i = code.index("surrogate_critic_enabled")
    j = code.index("_pix_decode_crops_grad", i)
    branch = code[i:j]
    assert "generator_surrogate_loss" in branch
    assert "weighted_sur" in branch and "raw_sur" in branch
    assert "return" in branch
    assert "_pix_decode_crops_grad" not in branch


# ---------------------------------------------------------------------------
# 2. Distillation: gates and regime flags.
# ---------------------------------------------------------------------------
def test_gate_off_is_byte_identical():
    stub = _SurStub()  # surrogate_critic_enabled = False
    rec = _RecDistiller()
    stub.latent_texture_distiller = rec  # present but must not be touched
    out = {}
    info = _info()  # BEFORE the fingerprint — _info() itself draws randn
    before = _rng_fingerprint()
    Trainer._maybe_run_surrogate_distillation(
        stub, info, out, current_step=7)
    assert out == {}
    assert rec.calls == []
    assert _same_rng(before, _rng_fingerprint()), \
        "the OFF path drew from a global RNG"


def test_gate_off_test_has_teeth():
    """Companion: the same call with the gate ON does mutate ``out`` and
    does reach the distiller — so the OFF assertions can actually fail."""
    stub = _stub_with_distiller()
    out = {}
    Trainer._maybe_run_surrogate_distillation(
        stub, _info(), out, current_step=7)
    assert stub.latent_texture_distiller.calls
    assert out.get("train/surrogate_loss_total") == 0.5


def test_enabled_with_missing_attrs_raises():
    stub = _SurStub()
    stub.surrogate_critic_enabled = True  # gate says on, build never ran
    with pytest.raises(RuntimeError, match="build step did not run"):
        Trainer._maybe_run_surrogate_distillation(
            stub, _info(), {}, current_step=7)


def test_no_teacher_is_a_regime_flag_not_silence():
    stub = _stub_with_distiller()
    stub.pixel_texture_disc = None
    out = {}
    Trainer._maybe_run_surrogate_distillation(
        stub, _info(), out, current_step=7)
    assert out["train/surrogate_distill_no_teacher"] == 1.0
    assert stub.latent_texture_distiller.calls == []


def test_warmup_skip_is_a_regime_flag_not_silence():
    stub = _stub_with_distiller()
    stub.gan_disc_start_step = 100
    out = {}
    Trainer._maybe_run_surrogate_distillation(
        stub, _info(), out, current_step=5)
    assert out["train/surrogate_distill_warmup_skipped"] == 1.0
    assert stub.latent_texture_distiller.calls == []


def test_empty_pool_passes_none_real_and_flags_it():
    stub = _stub_with_distiller()
    stub._pix_real_pool = []
    out = {}
    Trainer._maybe_run_surrogate_distillation(
        stub, _info(), out, current_step=7)
    assert out["train/surrogate_distill_no_reals"] == 1.0
    (call,) = stub.latent_texture_distiller.calls
    assert call["z_real"] is None and call["origin_real"] is None
    assert call["z_fake"] is not None  # fake side still distils


# ---------------------------------------------------------------------------
# 3. Distillation: what actually reaches ``distiller.step``.
# ---------------------------------------------------------------------------
def test_full_path_call_contract():
    stub = _stub_with_distiller()
    out = {}
    Trainer._maybe_run_surrogate_distillation(
        stub, _info(requires_grad=True), out, current_step=7)
    (call,) = stub.latent_texture_distiller.calls
    rc = Trainer._pix_resolve_cfg(stub)
    # fake: [n_crops, lat_frames, C, crop_rows, crop_cols], DETACHED even
    # though the source latents carried grad (the distiller owns its graph)
    zf = call["z_fake"]
    assert tuple(zf.shape) == (
        rc["n_crops"], rc["lat_frames"], LAT_C,
        rc["crop_rows"], rc["crop_cols"])
    assert not zf.requires_grad and zf.grad_fn is None
    # real: same crop geometry, detached, straight from the pool
    zr = call["z_real"]
    assert tuple(zr.shape)[1:] == tuple(zf.shape)[1:]
    assert not zr.requires_grad
    # origins are the BATCH-SHARED (y, x) pair -- peer correction
    # 2026-08-24 11:xx: ``compute_teacher_targets`` reads origin as
    # ``(origin[0], origin[1])`` = (y0, x0), so a per-crop y-list was
    # being misread (crop 1's y as x0). x is 0 by construction (A24
    # does not band-match horizontal position).
    assert len(call["origin_fake"]) == 2
    assert all(isinstance(v, int) for v in call["origin_fake"])
    assert call["origin_fake"][1] == 0
    assert len(call["origin_real"]) == 2
    assert all(isinstance(v, int) for v in call["origin_real"])
    assert call["origin_real"][1] == 0
    # the trainer's optimizer and step are passed through verbatim
    assert call["optimizer"] is stub.latent_critic_optimizer
    assert call["current_step"] == 7
    assert callable(call["teacher_value_fn"])


def test_direct_field_receives_exact_origin_for_every_crop():
    stub = _stub_with_distiller()
    stub.latent_texture_critic.predicts_gradient = True
    stub.latent_texture_critic.pixel_condition_channels = 0
    Trainer._maybe_run_surrogate_distillation(
        stub, _info(requires_grad=True), {}, current_step=7,
    )
    (call,) = stub.latent_texture_distiller.calls
    assert len(call["origin_fake"]) == call["z_fake"].shape[0]
    assert len(call["origin_real"]) == call["z_real"].shape[0]
    assert all(len(origin) == 2 for origin in call["origin_fake"])
    assert all(len(origin) == 2 for origin in call["origin_real"])
    # Pool x0 is deliberately one in this fixture; the old mean-origin path
    # overwrote it with zero for every row.
    assert all(int(origin[1]) == 1 for origin in call["origin_real"])


def test_teacher_closure_decodes_trims_and_reshapes():
    """The §4.2 closure: graph-on decode -> border trim -> pixel disc ->
    ``[N, F, 1, h, w]``. Exercised through the recorded closure against
    the stub VAE and the real ``PixelTextureDisc``."""
    stub = _stub_with_distiller()
    Trainer._maybe_run_surrogate_distillation(
        stub, _info(), {}, current_step=7)
    (call,) = stub.latent_texture_distiller.calls
    rc = Trainer._pix_resolve_cfg(stub)
    z = call["z_fake"][:2].to(torch.float32).requires_grad_(True)
    v = call["teacher_value_fn"](z)
    n, f = int(z.shape[0]), 4 * int(z.shape[1])  # stub VAE: 4x temporal
    assert tuple(v.shape[:3]) == (n, f, 1)
    # border trim actually happened: pixel extent fed to the disc was
    # (8*rows - 2b, 8*cols - 2b); the disc downsamples by 8 (stride 2 x3)
    b = rc["border"]
    assert v.shape[-2] < (8 * rc["crop_rows"]) // 8 or b == 0
    # and the closure is graph-on end to end (the Sobolev target needs it)
    v.mean().backward()
    assert z.grad is not None and float(z.grad.abs().sum()) > 0.0


def test_distiller_step_fires_every_gated_step():
    """One ``step`` per trainer step, ``current_step`` passed through —
    cadence (refresh vs replay) is the distiller's own tested property;
    the wiring's job is only to never skip and never double-fire."""
    stub = _stub_with_distiller()
    for s in (3, 4, 5):
        Trainer._maybe_run_surrogate_distillation(
            stub, _info(), {}, current_step=s)
    assert [c["current_step"] for c in
            stub.latent_texture_distiller.calls] == [3, 4, 5]


def test_pool_is_read_only_for_the_distiller():
    stub = _stub_with_distiller()
    ids = [id(e) for e in stub._pix_real_pool]
    uids = [e["uid"] for e in stub._pix_real_pool]
    Trainer._maybe_run_surrogate_distillation(
        stub, _info(), {}, current_step=7)
    assert [id(e) for e in stub._pix_real_pool] == ids
    assert [e["uid"] for e in stub._pix_real_pool] == uids


# ---------------------------------------------------------------------------
# 4. Cadence, END TO END with the real distiller (tiny critic).
# ---------------------------------------------------------------------------
def _real_distiller_stub(refresh_every=3):
    stub = _SurStub()
    from model.latent_texture_critic import build_from_config
    cfg = SimpleNamespace(
        surrogate_critic_enabled=True,
        surrogate_critic_d_model=32,
        surrogate_critic_num_blocks=1,
        surrogate_critic_num_heads=2,
        pix_teacher_refresh_every=refresh_every,
    )
    (stub.latent_texture_critic,
     stub.latent_critic_optimizer,
     stub.latent_texture_distiller) = build_from_config(
        cfg, device=torch.device("cpu"))
    stub.surrogate_critic_enabled = stub.latent_texture_critic is not None
    assert stub.surrogate_critic_enabled
    g = torch.Generator().manual_seed(1)
    stub._pix_real_pool = [_pool_entry(g, y0=i % 30) for i in range(8)]
    return stub


def test_cadence_via_n_teacher_refresh_end_to_end():
    """``n_teacher_refresh == ceil(steps / N)`` after driving the REAL
    wiring + REAL distiller — the checkpoint-invariant counter, never raw
    teacher-call counts (which run 2x under checkpointing)."""
    n = 3
    stub = _real_distiller_stub(refresh_every=n)
    steps = 7
    out_last = {}
    p0 = [p.detach().clone()
          for p in stub.latent_texture_critic.parameters()]
    for s in range(steps):
        out_last = {}
        Trainer._maybe_run_surrogate_distillation(
            stub, _info(), out_last, current_step=s)
    want = math.ceil(steps / n)
    assert stub.latent_texture_distiller.n_teacher_refresh == want
    assert out_last["train/surrogate_n_teacher_refresh"] == float(want)
    # the optimizer really stepped: the critic moved
    moved = any(
        not torch.equal(a, b) for a, b in
        zip(p0, stub.latent_texture_critic.parameters()))
    assert moved, "7 distillation steps left the critic byte-identical"


# ---------------------------------------------------------------------------
# 5. The consumption branch.
# ---------------------------------------------------------------------------
def test_consumption_takes_the_surrogate_path_and_omits_decode_keys():
    stub = _stub_with_distiller()
    w, raw, logs = Trainer._compute_pixel_texture_g_loss(
        stub, _info(requires_grad=True), current_step=7)
    assert logs["train/surrogate_consumed"] == 1.0
    # the direct path never ran: no decode, no disc keys, no G-crop draw
    assert stub.n_decodes == 0
    for k in ("train/pix_g_loss", "train/pix_g_images",
              "train/pix_g_fake_logit_mean"):
        assert k not in logs, k
    # contract: raw is UNWEIGHTED (== the module's weight-1.0 main term),
    # weighted == w * raw
    assert float(raw.detach()) == pytest.approx(
        logs["train/surrogate_g_main"])
    weight = Trainer._pix_gen_weight(stub, 7)
    assert weight > 0.0
    assert float(w.detach()) == pytest.approx(weight * float(raw.detach()))
    # applied-weight echoes read the runtime value, not the probe call
    assert logs["train/surrogate_g_weight"] == pytest.approx(weight)
    assert logs["train/pix_g_weight"] == pytest.approx(weight)
    assert logs["train/surrogate_g_weighted"] == pytest.approx(
        weight * logs["train/surrogate_g_main"])
    # and the term is graph-bearing back to the generator's latents
    assert raw.requires_grad


def test_consumption_zero_weight_returns_none_weighted_raw_survives():
    """The §12 weight-free probe regime: nothing enters the loss, the
    unweighted tensor still exists for the A7 ratio telemetry."""
    stub = _stub_with_distiller(pix_gan_weight=0.0)
    w, raw, logs = Trainer._compute_pixel_texture_g_loss(
        stub, _info(requires_grad=True), current_step=7)
    assert w is None
    assert raw is not None and raw.requires_grad
    assert logs["train/surrogate_g_weight"] == 0.0


def test_consumption_enabled_with_missing_critic_raises():
    stub = _SurStub()
    stub.surrogate_critic_enabled = True  # critic stays None
    with pytest.raises(RuntimeError, match="Refusing to fall back"):
        Trainer._compute_pixel_texture_g_loss(
            stub, _info(requires_grad=True), current_step=7)


def test_consumption_off_leaves_the_direct_path_reachable():
    """With the gate off the method proceeds past the branch untouched —
    asserted at the cheapest observable point: the branch's own keys are
    absent and the direct path's crop draw ran (n_decodes > 0)."""
    stub = _stub_with_distiller()
    stub.surrogate_critic_enabled = False
    w, raw, logs = Trainer._compute_pixel_texture_g_loss(
        stub, _info(requires_grad=True), current_step=7)
    assert "train/surrogate_consumed" not in logs
    assert stub.n_decodes > 0
    assert "train/pix_g_loss" in logs


def test_consumption_does_not_freeze_the_critic_permanently():
    """``generator_surrogate_loss``'s try/finally, observed through the
    wiring: after a successful call the critic's params require grad
    again (a frozen critic would silently stop distillation)."""
    stub = _stub_with_distiller()
    Trainer._compute_pixel_texture_g_loss(
        stub, _info(requires_grad=True), current_step=7)
    assert all(p.requires_grad
               for p in stub.latent_texture_critic.parameters())


# ---------------------------------------------------------------------------
# 6. Save / resume, FAIL-LOUD — anchor extraction, never line numbers.
# ---------------------------------------------------------------------------
_SAVE_START = ("        # WP-SURROGATE (B3) — latent surrogate critic + its "
               "optimizer,")
_SAVE_END = "        # ForwardNoiser (CARN) + its optimizer — same rationale"
_RESUME_START = ("        # WP-SURROGATE (B3) — latent surrogate critic + "
                 "its optimizer.")
_RESUME_END = "        # ForwardNoiser (CARN) + its optimizer."


def _slice_block(src, start, end):
    i = src.index(start)
    # back up to the banner line above the start anchor if present
    j = src.index(end, i)
    return textwrap.dedent(src[i:j])


def _exec_block(block, self_obj, state, path="ckpt.pt"):
    ns = {"self": self_obj, "state": state, "path": path,
          "logging": __import__("logging"), "torch": torch}
    exec(compile(block, "<block>", "exec"), ns)
    return state


def test_anchors_are_unique_in_both_shipped_sources():
    rs, af = _read(_RS_SRC_PATH), _read(_AF_SRC_PATH)
    assert rs.count(_SAVE_START) == 1
    assert af.count(_RESUME_START) == 1


def test_save_block_writes_both_keys_when_attrs_exist():
    block = _slice_block(_read(_RS_SRC_PATH), _SAVE_START, _SAVE_END)
    critic = nn.Linear(4, 4)
    opt = torch.optim.Adam(critic.parameters(), lr=1e-4, betas=(0.0, 0.9))
    stub = SimpleNamespace(
        latent_texture_critic=critic, latent_critic_optimizer=opt)
    state = _exec_block(block, stub, {})
    assert set(state) == {"latent_texture_critic", "latent_critic_optimizer"}
    got = state["latent_texture_critic"]
    for k, v in critic.state_dict().items():
        assert torch.equal(got[k], v)


def test_save_block_is_silent_when_attrs_absent():
    block = _slice_block(_read(_RS_SRC_PATH), _SAVE_START, _SAVE_END)
    stub = SimpleNamespace(
        latent_texture_critic=None, latent_critic_optimizer=None)
    assert _exec_block(block, stub, {}) == {}


def test_save_block_never_gates_on_gan_enabled():
    """The B1(d) trap: in the surrogate arm the transition GAN is off, so
    a ``gan_enabled`` gate would silently never save the critic."""
    block = _slice_block(_read(_RS_SRC_PATH), _SAVE_START, _SAVE_END)
    assert "gan_enabled" not in method_code_like(block)


def method_code_like(block):
    """Strip comments so the assertion reads code, not prose."""
    return "\n".join(
        ln for ln in block.splitlines() if not ln.lstrip().startswith("#"))


def _resume_stub(enabled=True, critic=None, opt=None):
    return SimpleNamespace(
        surrogate_critic_enabled=enabled,
        latent_texture_critic=critic,
        latent_critic_optimizer=opt,
        is_main_process=True,
    )


def test_resume_raises_on_missing_critic_key_when_enabled():
    block = _slice_block(_read(_AF_SRC_PATH), _RESUME_START, _RESUME_END)
    stub = _resume_stub(critic=nn.Linear(4, 4))
    with pytest.raises(RuntimeError, match="no 'latent_texture_critic'"):
        _exec_block(block, stub, {})


def test_resume_raises_on_missing_optimizer_key_when_enabled():
    block = _slice_block(_read(_AF_SRC_PATH), _RESUME_START, _RESUME_END)
    critic = nn.Linear(4, 4)
    opt = torch.optim.Adam(critic.parameters(), lr=1e-4, betas=(0.0, 0.9))
    stub = _resume_stub(critic=critic, opt=opt)
    state = {"latent_texture_critic": critic.state_dict()}
    with pytest.raises(RuntimeError,
                       match="no 'latent_critic_optimizer'"):
        _exec_block(block, stub, state)


def test_resume_restores_cleanly_when_both_keys_present():
    block = _slice_block(_read(_AF_SRC_PATH), _RESUME_START, _RESUME_END)
    src_critic = nn.Linear(4, 4)
    dst_critic = nn.Linear(4, 4)
    opt = torch.optim.Adam(dst_critic.parameters(), lr=1e-4, betas=(0.0, 0.9))
    stub = _resume_stub(critic=dst_critic, opt=opt)
    state = {
        "latent_texture_critic": src_critic.state_dict(),
        "latent_critic_optimizer": torch.optim.Adam(
            src_critic.parameters(), lr=1e-4, betas=(0.0, 0.9)).state_dict(),
    }
    _exec_block(block, stub, state)
    for k, v in src_critic.state_dict().items():
        assert torch.equal(dst_critic.state_dict()[k], v)


def test_resume_raises_on_shape_mismatch():
    block = _slice_block(_read(_AF_SRC_PATH), _RESUME_START, _RESUME_END)
    stub = _resume_stub(critic=nn.Linear(4, 4))
    state = {"latent_texture_critic": nn.Linear(8, 8).state_dict(),
             "latent_critic_optimizer": {}}
    with pytest.raises(RuntimeError, match="did not load cleanly"):
        _exec_block(block, stub, state)


def test_resume_is_a_noop_when_disabled():
    block = _slice_block(_read(_AF_SRC_PATH), _RESUME_START, _RESUME_END)
    stub = _resume_stub(enabled=False, critic=None)
    _exec_block(block, stub, {})  # must not raise on the pre-surrogate ckpt
