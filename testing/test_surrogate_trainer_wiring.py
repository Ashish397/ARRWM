"""WP-SURROGATE (B3) — trainer wiring.

Two windows, both covered here:

1. **Build step** (landed 2026-08-24 morning, WP-SURROGATE): gate
   ``surrogate_critic_enabled`` (default False), DERIVED as
   ``self.latent_texture_critic is not None`` rather than independently
   read off config a second time (the seam-class fix this package's own
   module docstring describes); ``self.latent_texture_critic`` /
   ``self.latent_critic_optimizer`` / ``self.latent_texture_distiller``,
   built by the single seam-closing function
   ``model.latent_texture_critic.build_from_config``; the resolved-value
   echo log line (docs/WP_PIXGAN.md's rule: log what was ACTUALLY built
   off the live objects, never re-derived from ``self.config``); and a
   WARNING (found on review) when the surrogate is enabled without the
   pixel critic, which structurally means it will never train or be
   consumed for the whole run.
2. **Consumption wiring** (landed 2026-08-24 late morning, MAIN under
   researcher order, docs/TASK_SURROGATE_CONSUMPTION.md; reviewed and
   accepted by WP-SURROGATE as spec owner, two fixes applied on review —
   the warning above and the resume-gate fix in section 7 below): the
   per-step distillation call (``_maybe_run_surrogate_distillation``) and
   the generator's G-term branch (inside
   ``_compute_pixel_texture_g_loss``), plus FAIL-LOUD save/resume. The
   B1-owned crop/decode/disc helpers these call into are stubbed to
   small deterministic tensors here — their own correctness is B1's test
   surface (testing/test_pixgan_trainer_wiring.py and siblings), not
   re-tested in this file; what IS tested here is this package's own
   glue — gating, raises, regime flags, and that origins/tensors thread
   correctly into ``LatentSurrogateDistiller.step``.

CPU-ONLY, same pattern as ``testing/test_pixgan_trainer_wiring.py`` (whose
docstring and stub-generation idiom this file follows deliberately, for
the reason its own comment gives: hand-copying an attribute list is two
things that must agree with nothing enforcing it). There is no GPU on the
build node, so the ``__init__`` block is exercised by extracting its
shipped source text between greppable anchors and ``exec``-ing it against
a stub ``self`` — never line numbers, the trainer is edited concurrently
by other packages.

Run (thread caps are NOT optional — nproc=144 here):

    cd /scratch/u6ex/as1748.u6ex/ARRWM
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONPATH=. \
      /scratch/u6ex/as1748.u6ex/miniforge3/envs/arrwm/bin/python \
      -m pytest -q testing/test_surrogate_trainer_wiring.py
"""
import os
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.latent_texture_critic import (  # noqa: E402
    LatentSurrogateDistiller,
    LatentTextureCritic,
)
from testing.test_latent_texture_critic import CROP_SHAPE, TINY  # noqa: E402

# Wan's T5 wrapper evaluates torch.cuda.current_device() at import time even
# though this test never constructs a model (same idiom WP-PIXGAN's own
# trainer-wiring test uses, borrowed in turn from
# testing/test_r1_cadence_and_override_guard.py).
with patch.object(torch.cuda, "current_device", return_value=0):
    from trainer import causal_action_forcing_train as CAFT

Trainer = CAFT.ActionForcingDMDTrainer

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_AF_SRC_PATH = os.path.join(_ROOT, "trainer", "causal_action_forcing_train.py")


def _af_source():
    with open(_AF_SRC_PATH, encoding="utf-8") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# Extracting the __init__ block so it can run on CPU without the rest of the
# trainer's construction (VAE, DDP process group, WanModel, ...).
# ---------------------------------------------------------------------------
_BLOCK_START = (
    "        # WP-SURROGATE (B3) — latent surrogate critic. BUILD STEP "
    "ONLY this\n"
)
_BLOCK_END = (
    "        # ------------------------------------------------------"
    "------------\n        # SC-DMD (Salt)"
)


def _slice(src, start, end):
    i = src.index(start)
    j = src.index(end, i)
    return textwrap.dedent(src[i:j])


def _surrogate_block():
    return _slice(_af_source(), _BLOCK_START, _BLOCK_END)


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
    """The ``self`` the shipped block is exec'd against.

    ``config``, ``device``, ``is_main_process`` are what the BUILD call
    itself reads. ``gan_pixel_texture_enabled`` is also supplied here as
    a TRAINER ATTRIBUTE (not read off ``cfg``) because that is what the
    real trainer guarantees: the pixel-critic block earlier in
    ``__init__`` sets ``self.gan_pixel_texture_enabled`` before this
    block ever runs, and the new pixel-gate warning reads it the same
    way (``getattr(self, "gan_pixel_texture_enabled", ...)``, never off
    ``self.config``) -- matching that guarantee here is what lets a test
    pass ``gan_pixel_texture_enabled=True`` and have the warning actually
    see it. No hand-picked list of WP-SURROGATE members is needed beyond
    this (unlike WP-PIXGAN's stub) because this block calls no trainer
    method of its own — everything it does goes through the
    already-tested ``build_from_config``.
    """

    def __init__(self, cfg):
        self.config = cfg
        self.device = torch.device("cpu")
        self.is_main_process = True
        self.gan_pixel_texture_enabled = bool(
            getattr(cfg, "gan_pixel_texture_enabled", False)
        )


def _run_block(**cfg_kwargs):
    """Execute the shipped block on a stub self.

    Returns ``(self_stub, recording_logging)``.
    """
    cfg = SimpleNamespace(**cfg_kwargs)
    stub = _CtorStub(cfg)
    log = _RecordingLogging()
    ns = {"self": stub, "torch": torch, "logging": log}
    exec(compile(_surrogate_block(), "<surrogate-ctor>", "exec"), ns)
    return stub, log


# ---------------------------------------------------------------------------
# 1. Gate: default off, byte-identical off
# ---------------------------------------------------------------------------
def test_gate_defaults_to_false_when_config_is_silent():
    stub, _ = _run_block()
    assert stub.surrogate_critic_enabled is False


def test_gate_off_builds_no_module_no_optimizer_no_distiller():
    stub, log = _run_block(surrogate_critic_enabled=False)
    assert stub.latent_texture_critic is None
    assert stub.latent_critic_optimizer is None
    assert stub.latent_texture_distiller is None


def test_gate_off_emits_no_log_line():
    """Byte-identical off: not one new log key / line."""
    _, log = _run_block(surrogate_critic_enabled=False)
    assert log.records == []


def test_gate_off_consumes_no_rng():
    """Byte-identical off: the global torch RNG stream is untouched.

    Constructing the critic draws (the transformer blocks' weight init +
    the positional-embedding trunc_normal_ init), so an accidentally
    ungated build would shift every downstream draw in the run.
    """
    torch.manual_seed(1234)
    before = torch.get_rng_state().clone()
    _run_block(surrogate_critic_enabled=False)
    assert torch.equal(torch.get_rng_state(), before)


def test_gate_on_does_consume_rng_so_the_off_test_has_teeth():
    torch.manual_seed(1234)
    before = torch.get_rng_state().clone()
    _run_block(surrogate_critic_enabled=True)
    assert not torch.equal(torch.get_rng_state(), before)


def test_gate_absent_key_matches_explicit_false():
    """An old config predating this package's block must build identically
    to one that spells out ``surrogate_critic_enabled: false``."""
    a, log_a = _run_block()
    b, log_b = _run_block(surrogate_critic_enabled=False)
    assert a.latent_texture_critic is None and b.latent_texture_critic is None
    assert log_a.records == log_b.records == []


# ---------------------------------------------------------------------------
# 2. Gate: independent of the pixel critic's gate, not smuggled in as one
# ---------------------------------------------------------------------------
def test_block_does_not_read_gan_pixel_texture_enabled_as_its_own_gate():
    """The critic can be BUILT with no teacher connected — a real
    distillation step needs the pixel critic, but nothing in this block
    enforces that dependency (deliberately; see the block's own comment)."""
    stub, log = _run_block(
        surrogate_critic_enabled=True,
        # gan_pixel_texture_enabled deliberately absent from the stub cfg;
        # if the block secretly required it, this would AttributeError
        # inside build_from_config's getattr chain rather than silently
        # passing -- getattr's default makes absence safe either way, so
        # this also doubles as a "does not require the sibling gate to
        # even be configured" check.
    )
    assert stub.latent_texture_critic is not None


def test_surrogate_block_source_does_not_reference_pixel_gate_before_the_build():
    """The prose comment mentions the pixel gate for context (documentation,
    not a dependency -- fine). The pixel-gate WARNING (found on review, see
    section 6 below) legitimately reads it too, but ONLY as a diagnostic
    AFTER the critic already exists -- it must never gate the BUILD itself.
    Checked precisely: no CODE line (non-comment, non-blank) referencing
    the pixel gate may appear before ``build_from_config(`` in source
    order. The naive "no CODE line at all" version of this test (first
    draft) broke the moment the legitimate warning landed; distinguishing
    "diagnostic after" from "gate before" is the property that actually
    matters, and is exactly what the warning vs. the build call differ on."""
    block = _surrogate_block()
    build_call_pos = block.index("build_from_config(")
    before_build = block[:build_call_pos]
    code_lines = [
        ln for ln in before_build.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    offending = [ln for ln in code_lines if "gan_pixel_texture_enabled" in ln]
    assert not offending, (
        f"gate coupling found in code BEFORE the build call: {offending}"
    )
    # And the warning's own reference must exist -- otherwise this test
    # would vacuously pass if the warning were deleted entirely.
    after_build = block[build_call_pos:]
    assert "gan_pixel_texture_enabled" in after_build


# ---------------------------------------------------------------------------
# 3. Gate on: constructs the three objects, derived flag agrees
# ---------------------------------------------------------------------------
def test_gate_on_builds_critic_optimizer_distiller():
    stub, log = _run_block(surrogate_critic_enabled=True)
    assert stub.latent_texture_critic is not None
    assert stub.latent_critic_optimizer is not None
    assert stub.latent_texture_distiller is not None
    assert stub.latent_texture_distiller.critic is stub.latent_texture_critic
    assert stub.surrogate_critic_enabled is True


def test_surrogate_critic_enabled_is_derived_not_independently_settable():
    """The whole point of the derivation: setting the config key without
    the build actually succeeding cannot leave a stale True flag -- there
    is only one source of truth, ``latent_texture_critic is not None``."""
    src = _surrogate_block()
    # The gate is read exactly once, by build_from_config (inside the
    # imported function, not re-read here) -- this block itself must not
    # contain a second ``getattr(self.config, "surrogate_critic_enabled"``.
    assert src.count('getattr(self.config, "surrogate_critic_enabled"') == 0
    assert (
        "self.surrogate_critic_enabled = "
        "self.latent_texture_critic is not None"
    ) in src


# ---------------------------------------------------------------------------
# 4. The resolved-value echo reads LIVE OBJECTS, not the config a second
#    time — the specific property docs/WP_PIXGAN.md's rule requires.
# ---------------------------------------------------------------------------
def test_echo_reflects_the_resolved_cadence_not_the_requested_one():
    """Config asks for N=7; the echo must report what the DISTILLER
    actually holds, not a re-read of the config value -- these happen to
    agree here (there is no clamping on this particular knob), so the
    real assertion is on the SOURCE the number came from
    (test_echo_source_reads_the_live_distiller_attribute below). This test
    pins the end-to-end value as a sanity floor: format the exact record
    logging.info received and check the rendered line, not just the raw
    attribute -- a wrong %-token order would pass an attribute check but
    render nonsense."""
    # gan_pixel_texture_enabled=True isolates this test to the INFO echo
    # only -- leaving it unset would also fire the found-on-review WARNING
    # (surrogate built but the pixel gate is off, so nothing will ever
    # train or be consumed), which is correct behaviour but not what this
    # test is checking; see test_warning_fires_when_pixel_gate_is_off.
    stub, log = _run_block(
        surrogate_critic_enabled=True, pix_teacher_refresh_every=7,
        gan_pixel_texture_enabled=True,
    )
    assert stub.latent_texture_distiller.pix_teacher_refresh_every == 7
    assert len(log.records) == 1
    _, msg, args = log.records[0]
    rendered = msg % args
    assert "pix_teacher_refresh_every=7" in rendered, rendered


def test_echo_source_reads_the_live_distiller_attribute():
    """Static check that every echoed number's argument expression reads
    an attribute off ``self.latent_texture_distiller`` /
    ``self.latent_texture_critic`` -- never ``self.config`` a second time.
    This is the property the resolved-value-echo rule actually requires;
    the previous test only shows the numbers happen to agree, which a
    parallel-but-disconnected re-read could also produce."""
    src = _surrogate_block()
    log_call_start = src.index('"[ActionForcing] Latent surrogate critic built:')
    log_call = src[log_call_start:src.index(")", src.rindex("teacher_use_checkpoint,"))]
    assert "self.config" not in log_call, (
        "the echo re-reads self.config instead of the live built objects"
    )
    for attr in (
        "self.latent_texture_critic.num_params",
        "self.latent_texture_critic.d_model",
        "self.latent_texture_critic.num_blocks",
        "self.latent_texture_distiller.pix_teacher_refresh_every",
        "self.latent_texture_distiller.grad_loss_normalize",
        "self.latent_texture_distiller.grad_loss_weight",
        "self.latent_texture_distiller.value_loss_weight",
        "self.latent_texture_distiller.cache.capacity",
        "self.latent_texture_distiller.teacher_use_checkpoint",
    ):
        assert attr in log_call, f"echo does not read {attr}"


def test_echo_only_fires_on_main_process():
    stub = _CtorStub(SimpleNamespace(surrogate_critic_enabled=True))
    stub.is_main_process = False
    log = _RecordingLogging()
    ns = {"self": stub, "torch": torch, "logging": log}
    exec(compile(_surrogate_block(), "<surrogate-ctor>", "exec"), ns)
    assert stub.latent_texture_critic is not None  # still built
    assert log.records == []  # but not logged off the main process


# ---------------------------------------------------------------------------
# 5. Anchor stability — fails loudly, not silently, if the block moves
# ---------------------------------------------------------------------------
def test_anchors_are_unique_in_the_shipped_source():
    src = _af_source()
    assert src.count(_BLOCK_START) == 1
    assert src.count(_BLOCK_END) >= 1


def test_import_is_local_not_module_level():
    """Matches the codebase convention (every WP-* critic is imported
    inside its build ``if``, e.g. ``from model.pixel_texture_disc import
    PixelTextureDisc`` at the pixel block above) -- keeps an unused-import
    cost off every trainer construction where the gate is off."""
    src = _af_source()
    assert (
        "from model.latent_texture_critic import build_from_config" in src
    )
    assert (
        "\nfrom model.latent_texture_critic import build_from_config"
        not in src.split('class ActionForcingDMDTrainer')[0]
    ), "build_from_config must not be imported at module level"



# ---------------------------------------------------------------------------
# 6. The WARNING found on the 2026-08-24 consumption-wiring review: a
#    surrogate built without the pixel gate will never train or be
#    consumed, for the whole run, with only the INFO echo suggesting it is
#    live. Both structural sites confirm the claim; the trainer-side test
#    below confirms the warning that reports it.
# ---------------------------------------------------------------------------
def test_warning_fires_when_pixel_gate_is_off():
    stub, log = _run_block(
        surrogate_critic_enabled=True, gan_pixel_texture_enabled=False,
    )
    assert stub.latent_texture_critic is not None  # still built
    kinds = [k for k, _, _ in log.records]
    assert kinds == ["info", "warning"], log.records
    _, msg, args = log.records[1]
    rendered = msg % args
    assert "will NEVER train" in rendered and "will NEVER" in rendered


def test_warning_absent_key_matches_explicit_false():
    """gan_pixel_texture_enabled absent from cfg must behave like an
    explicit False -- same getattr-default convention as everywhere else
    in this block."""
    stub, log = _run_block(surrogate_critic_enabled=True)
    kinds = [k for k, _, _ in log.records]
    assert kinds == ["info", "warning"]


def test_warning_silent_when_pixel_gate_also_enabled():
    stub, log = _run_block(
        surrogate_critic_enabled=True, gan_pixel_texture_enabled=True,
    )
    kinds = [k for k, _, _ in log.records]
    assert kinds == ["info"], log.records


def test_warning_silent_off_main_process():
    stub = _CtorStub(SimpleNamespace(
        surrogate_critic_enabled=True, gan_pixel_texture_enabled=False,
    ))
    stub.is_main_process = False
    log = _RecordingLogging()
    ns = {"self": stub, "torch": torch, "logging": log}
    exec(compile(_surrogate_block(), "<surrogate-ctor>", "exec"), ns)
    assert log.records == []


# ---------------------------------------------------------------------------
# 7. The resume-gate fix, added on the same review: ``surrogate_critic_
#    enabled`` was absent from ``_maybe_resume``'s early-return OR-chain --
#    the exact trap WP-PIXGAN's own comment, one clause up, warns against.
#    Currently dead-safe (the consumption sites below never train the
#    critic without the pixel gate, which IS in the chain already), but
#    defense in depth should not depend on that invariant holding forever.
# ---------------------------------------------------------------------------
_RESUME_GATE_START = (
    "    def _maybe_resume(self) -> None:\n"
    "        super()._maybe_resume()\n"
    "        if not (\n"
)
# ``_slice`` returns everything up to but NOT INCLUDING the start of the
# END marker match -- so the marker must be the statement AFTER the
# closing "):\n    return\n", never text starting with "):\n" itself
# (a first draft used "        ):\n            return\n" as the marker,
# which excluded exactly the two lines needed, producing an if-statement
# with no closing paren: "SyntaxError: '(' was never closed"). Anchoring
# on the NEXT statement pulls the closing paren + return in naturally.
_RESUME_GATE_END = "        if not bool(getattr(self.config"


def test_resume_gate_anchor_is_unique_in_the_shipped_source():
    """The naive anchor (just the OR-chain's own opening lines) matched a
    DIFFERENT, unrelated ``if not (...)`` block elsewhere in this file on
    the first draft of this test -- the same condition shape recurs. This
    pins the fix: anchor on the enclosing method signature instead, and
    assert it occurs exactly once."""
    src = _af_source()
    assert src.count(_RESUME_GATE_START) == 1


def _resume_gate_source():
    """The bare OR-chain condition, as ``if not (\n ... \n):\n    return\n``
    -- the ``def``/``super()`` prefix used to make the anchor unique
    (see the anchor-uniqueness test above) is stripped back off, since
    downstream consumers want exactly the condition, at its ORIGINAL
    method-body indentation (unlike ``_slice``'s callers elsewhere in
    this file, this one deliberately does NOT dedent -- the functional
    tests below re-indent it themselves when wrapping it in ``_f``)."""
    full = _slice(_af_source(), _RESUME_GATE_START, _RESUME_GATE_END)
    # ``_slice`` dedents; re-derive the condition-only text by dropping
    # the two prefix lines the anchor needed for uniqueness.
    lines = full.splitlines(keepends=True)
    assert lines[0].strip().startswith("def _maybe_resume")
    assert "super()._maybe_resume()" in lines[1]
    return "".join(lines[2:])


def test_resume_gate_contains_surrogate_flag():
    src = _resume_gate_source()
    assert 'getattr(self, "surrogate_critic_enabled", False)' in src


def test_resume_gate_evaluates_true_for_surrogate_alone():
    """Functional, not just textual: exec the real condition with every
    OTHER flag in the OR-chain False and only surrogate_critic_enabled
    True -- must NOT early-return."""
    src = _resume_gate_source()
    stub = SimpleNamespace(
        action_critic_loss_active=False, gan_enabled=False,
        gan_pixel_texture_enabled=False, surrogate_critic_enabled=True,
        real_teacher_train_online=False, state_probe_aux_active=False,
    )
    ns = {"self": stub, "getattr": getattr}
    # ``src`` is ``if not (...):\n    return\n`` at method-body
    # indentation; wrap it in a function so ``return`` is legal and the
    # branch outcome is observable.
    fn_src = "def _f(self):\n" + src + "    return 'did-not-return'\n"
    exec(compile(fn_src, "<resume-gate>", "exec"), ns)
    assert ns["_f"](stub) == "did-not-return", (
        "the OR-chain early-returned with only surrogate_critic_enabled=True"
    )


def test_resume_gate_still_returns_when_everything_is_false():
    """Mutation-control-style: proves the previous test's stub is not
    vacuously passing because the condition never early-returns at all."""
    src = _resume_gate_source()
    stub = SimpleNamespace(
        action_critic_loss_active=False, gan_enabled=False,
        gan_pixel_texture_enabled=False, surrogate_critic_enabled=False,
        real_teacher_train_online=False, state_probe_aux_active=False,
    )
    ns = {"self": stub, "getattr": getattr}
    fn_src = "def _f(self):\n" + src + "    return 'did-not-return'\n"
    exec(compile(fn_src, "<resume-gate>", "exec"), ns)
    assert ns["_f"](stub) is None


# ---------------------------------------------------------------------------
# 8. Consumption wiring -- the per-step distillation call and the
#    generator's G-term branch. Landed 2026-08-24 (MAIN, researcher order,
#    docs/TASK_SURROGATE_CONSUMPTION.md), reviewed and two fixes applied
#    here (the resume gate above; the pixel-gate warning above). These
#    tests cover the part that is squarely this package's contract: the
#    gating, the FAIL-LOUD raises, the regime flags, and -- one functional
#    happy-path test -- that the glue code threads z_real/z_fake/origins
#    correctly into ``LatentSurrogateDistiller.step``. The B1-owned helpers
#    (``_pix_select_fake_latents``, ``_pix_take_crops_with_origins``, the
#    real VAE decode, the real pixel disc) are stubbed to deterministic
#    small tensors -- their OWN correctness is B1's test surface, not
#    re-tested here.
# ---------------------------------------------------------------------------
_DISTILL_START = "    def _maybe_run_surrogate_distillation(\n"
# NOTE: _surrogate_g_snapshot_critic was later inserted BETWEEN the
# distillation fn and the G-term, so anchoring the slice end on the
# G-term would swallow two functions (caught by _make_callable's
# "exactly one function" assertion -- which is exactly why that
# assertion is there rather than silently exec'ing both).
_DISTILL_END = "\n    def _surrogate_g_snapshot_critic(self)"
_GLOSS_START = "    def _compute_pixel_texture_g_loss(\n"
_GLOSS_END = "\n    def _maybe_run_pixel_texture_d_updates(\n"


def _make_callable(af_source, start, end, fname):
    """Extract a shipped method's FULL source (signature included) and
    exec it as a free function named ``fname``, callable as
    ``fname(stub, ...)``."""
    body = _slice(af_source, start, end)
    # The extracted text is ``    def name(\n        self,\n ...): ...``
    # at class-body indentation (4 spaces); dedent to top-level so it is a
    # valid module-level function definition.
    src = textwrap.dedent(body)
    ns = {
        "torch": torch, "logging": _RecordingLogging(), "Dict": dict,
        "Any": object, "Optional": __import__("typing").Optional,
        "Tuple": __import__("typing").Tuple, "List": list,
    }
    exec(compile(src, f"<{fname}>", "exec"), ns)
    # The def line's own name is whatever the source says; find it. Filter
    # by FunctionType specifically (not bare ``callable``) -- the seed
    # namespace's ``Dict``/``List`` bindings (dict, list) are themselves
    # callable and would otherwise match too, breaking the "exactly one"
    # assertion below.
    import types as _types
    fn = [v for v in ns.values() if isinstance(v, _types.FunctionType)]
    assert len(fn) == 1, (
        f"expected exactly one function in {fname}'s slice, found "
        f"{len(fn)}: {[f.__name__ for f in fn]}"
    )
    return fn[0]


class _PixStub:
    """Minimal stand-in providing exactly the B1-owned surface the
    surrogate call sites read. Deterministic, small, CPU-only."""

    def __init__(self, *, surrogate_critic_enabled, gan_pixel_texture_enabled=True):
        self.surrogate_critic_enabled = surrogate_critic_enabled
        self.gan_pixel_texture_enabled = gan_pixel_texture_enabled
        self.is_main_process = True
        self.gan_disc_start_step = 0
        # Dimension-agnostic on purpose: the teacher closure calls this
        # on a flattened [N*F, 3, h, w] input, but B1's own direct-path
        # decode helper (_pix_decode_crops_grad, stubbed below) is not
        # re-tested here and this stub does not assume its exact output
        # rank -- collapsing everything but the batch dim to a mean
        # works regardless (first draft assumed exactly 4-D and broke on
        # the direct path's 5-D stub output).
        self.pixel_texture_disc = (
            (lambda px: px.reshape(px.shape[0], -1).mean(dim=1)
             .reshape(-1, 1, 1, 1).expand(-1, 1, 2, 2))
            if gan_pixel_texture_enabled else None
        )
        if surrogate_critic_enabled:
            self.latent_texture_critic = LatentTextureCritic(**TINY)
            self.latent_critic_optimizer = torch.optim.Adam(
                self.latent_texture_critic.parameters(), lr=1e-3,
            )
            self.latent_texture_distiller = LatentSurrogateDistiller(
                self.latent_texture_critic, grad_loss_weight=1.0,
                pix_teacher_refresh_every=1,
            )
        else:
            self.latent_texture_critic = None
            self.latent_critic_optimizer = None
            self.latent_texture_distiller = None
        self._pix_real_pool = [
            {"lat": torch.randn(*CROP_SHAPE[1:]), "y0": 8 * i}
            for i in range(3)
        ]
        self._select_fake_calls = []
        self._crop_calls = []

    # Bound from the real class (same rule as WP-PIXGAN's stub): a
    # hand-written fake here would let the snapshot logic drift from the
    # shipped one without any test noticing.
    _surrogate_g_snapshot_critic = Trainer._surrogate_g_snapshot_critic

    def _pix_resolve_cfg(self):
        return {
            "n_crops": 2, "crop_rows": CROP_SHAPE[-2], "crop_cols": CROP_SHAPE[-1],
            "n_bands": 2, "border": 0, "crop_lat": (CROP_SHAPE[-2], CROP_SHAPE[-1]),
            "k_frames": CROP_SHAPE[1], "loss_form": "hinge", "decode_batch": 4,
        }

    def _pix_sync_generator(self, step, base, salt=0):
        return torch.Generator(device="cpu").manual_seed(1000 * step + salt)

    def _pix_gen_weight(self, step):
        return 1.0

    def _pix_select_fake_latents(self, info):
        self._select_fake_calls.append(True)
        lat = torch.randn(*CROP_SHAPE, requires_grad=True)
        return lat, {}

    def _pix_take_crops_with_origins(self, lat, *, n_crops, crop_rows,
                                      crop_cols, n_bands, gen):
        self._crop_calls.append(True)
        n = lat.shape[0]
        crops = lat[:n_crops] if n >= n_crops else lat.repeat(
            (n_crops + n - 1) // n, 1, 1, 1, 1)[:n_crops]
        ys = [4 * i for i in range(n_crops)]
        xs = [0] * n_crops
        bands = [0] * n_crops
        return crops, ys, xs, bands

    def _pix_vae_dtype(self):
        # Added by WP-PIXGAN's bf16-decode fix; the teacher closure calls
        # it, so the stub must supply it or the happy-path test dies on an
        # AttributeError that looks like a product bug and is not.
        return torch.float32

    def _vae_decode_grad(self, z):
        # Cheap stand-in: [N,F,C,h,w] -> [N,F,3,h,w], graph-carrying.
        return z[:, :, :3].contiguous() * 1.0

    def _pix_decode_crops_grad(self, crops, *, border, n_frames,
                                decode_batch, gen):
        return crops[:, :, :3].contiguous() * 1.0

    def _pix_g_snapshot_disc(self):
        return self.pixel_texture_disc

    def _pix_band_histogram(self, bands, n_bands, prefix=""):
        return {}


def test_distillation_gate_off_is_a_true_noop():
    fn = _make_callable(_af_source(), _DISTILL_START, _DISTILL_END,
                         "_maybe_run_surrogate_distillation")
    stub = _PixStub(surrogate_critic_enabled=False)
    out = {}
    fn(stub, {}, out, current_step=0)
    assert out == {}


def test_distillation_raises_on_missing_state_while_enabled():
    fn = _make_callable(_af_source(), _DISTILL_START, _DISTILL_END,
                         "_maybe_run_surrogate_distillation")
    stub = _PixStub(surrogate_critic_enabled=True)
    stub.latent_texture_distiller = None  # simulate a dropped resume
    with pytest.raises(RuntimeError, match="build step did not run"):
        fn(stub, {}, {}, current_step=0)


def test_distillation_no_teacher_is_a_regime_flag_not_a_raise():
    fn = _make_callable(_af_source(), _DISTILL_START, _DISTILL_END,
                         "_maybe_run_surrogate_distillation")
    stub = _PixStub(surrogate_critic_enabled=True, gan_pixel_texture_enabled=False)
    out = {}
    fn(stub, {}, out, current_step=0)
    # surrogate_distill_ran=0.0 accompanies the no-teacher flag by design:
    # "the distillation executed" must be a POSITIVE assertable fact, not
    # something inferred from the absence of a warning. That distinction is
    # what two NULL 60-step smokes cost before it existed (§4d).
    assert out == {
        "train/surrogate_distill_no_teacher": 1.0,
        "train/surrogate_distill_ran": 0.0,
    }
    assert not stub._select_fake_calls, (
        "no-teacher path must return before touching any crop machinery"
    )


def test_distillation_warmup_skip_is_a_regime_flag():
    fn = _make_callable(_af_source(), _DISTILL_START, _DISTILL_END,
                         "_maybe_run_surrogate_distillation")
    stub = _PixStub(surrogate_critic_enabled=True)
    stub.gan_disc_start_step = 100
    out = {}
    fn(stub, {}, out, current_step=0)
    assert out == {"train/surrogate_distill_warmup_skipped": 1.0}


def test_distillation_happy_path_threads_origins_and_calls_distiller_step():
    fn = _make_callable(_af_source(), _DISTILL_START, _DISTILL_END,
                         "_maybe_run_surrogate_distillation")
    stub = _PixStub(surrogate_critic_enabled=True)
    out = {}
    fn(stub, {}, out, current_step=0)
    assert stub._select_fake_calls and stub._crop_calls
    assert "train/critic_total_loss" in out  # the distiller's own telemetry
    fake_entry = stub.latent_texture_distiller.cache.latest("fake")
    assert fake_entry is not None
    # The trainer passes a single batch-shared origin -- mean y across
    # the stub's crops (0 and 4 for n_crops=2), x fixed at 0 (see the
    # trainer's own comment on why: A24 does not band-match horizontal
    # position, so there is no meaningful x to average). NOT the first
    # crop's raw origin -- that was the bug this fix corrected (a first
    # draft of both the trainer code and this test assumed the origin
    # list passed through un-aggregated).
    assert fake_entry.origin == (2, 0), fake_entry.origin


def test_distillation_masks_before_cropping_not_after():
    """Structural pin of the zero-gradient-trap ordering: the fake source
    is selected (mask-aware) BEFORE the crop draw, never the reverse."""
    src = _slice(_af_source(), _DISTILL_START, _DISTILL_END)
    i_select = src.index("_pix_select_fake_latents(info)")
    i_crop = src.index("_pix_take_crops_with_origins(")
    assert i_select < i_crop


def test_gterm_both_gates_off_returns_none_none_empty():
    """TRUE off: neither critic enabled -> nothing built, nothing logged."""
    fn = _make_callable(_af_source(), _GLOSS_START, _GLOSS_END,
                         "_compute_pixel_texture_g_loss")
    stub = _PixStub(surrogate_critic_enabled=False,
                    gan_pixel_texture_enabled=False)
    assert fn(stub, {}, current_step=0) == (None, None, {})


def test_gterm_surrogate_only_mode_serves_the_term(caplog=None):
    """Surrogate ON + pixel gate OFF is the SAM2/DINOv2/ConvNeXt branch's
    normal configuration -- the pretrained teacher replaces B1's
    from-scratch critic entirely, so the G-term must still be served.
    This deliberately supersedes the older assertion that the pixel gate
    being off meant no term at all; that was true only while the pixel
    critic was the surrogate's sole teacher."""
    fn = _make_callable(_af_source(), _GLOSS_START, _GLOSS_END,
                         "_compute_pixel_texture_g_loss")
    stub = _PixStub(surrogate_critic_enabled=True,
                    gan_pixel_texture_enabled=False)
    w, raw, logs = fn(stub, {}, current_step=0)
    assert raw is not None, "surrogate-only mode produced no G-term"
    assert logs.get("train/surrogate_consumed") == 1.0


def test_gterm_missing_critic_raises_never_falls_back_to_direct():
    fn = _make_callable(_af_source(), _GLOSS_START, _GLOSS_END,
                         "_compute_pixel_texture_g_loss")
    stub = _PixStub(surrogate_critic_enabled=True)
    stub.latent_texture_critic = None  # simulate a dropped resume
    with pytest.raises(RuntimeError, match="Refusing to fall back"):
        fn(stub, {}, current_step=0)


def test_gterm_surrogate_happy_path_never_reaches_direct_path_keys():
    """Alternatives-never-summed, proven by telemetry: the direct path's
    keys (pix_g_fake_logit_mean etc.) must be ABSENT when the surrogate
    served the term, not merely zero."""
    fn = _make_callable(_af_source(), _GLOSS_START, _GLOSS_END,
                         "_compute_pixel_texture_g_loss")
    stub = _PixStub(surrogate_critic_enabled=True)
    weighted, raw, logs = fn(stub, {}, current_step=0)
    assert raw is not None and weighted is not None
    assert logs.get("train/surrogate_consumed") == 1.0
    for k in ("train/pix_g_fake_logit_mean", "train/pix_g_patch_logit_spatial_var"):
        assert k not in logs, f"direct-path key {k} present in surrogate mode"


def test_gterm_surrogate_off_reaches_the_direct_path_instead():
    """Mutation-control for the previous test: with the gate off, the
    SAME stub (teacher present, gradient-carrying fake) must reach the
    direct decode->disc path, proving the branch is a real fork, not a
    dead ``if`` that always returns."""
    fn = _make_callable(_af_source(), _GLOSS_START, _GLOSS_END,
                         "_compute_pixel_texture_g_loss")
    stub = _PixStub(surrogate_critic_enabled=False)
    weighted, raw, logs = fn(stub, {}, current_step=0)
    assert "train/pix_g_fake_logit_mean" in logs
    assert "train/surrogate_consumed" not in logs


def test_gterm_passes_the_unwrapped_critic_never_a_module_attribute():
    """Structural pin of the no-DDP-wrapping requirement
    (LatentSurrogateDistiller's own docstring): the G-term branch must
    call generator_surrogate_loss with the UNWRAPPED critic, never a
    ``.module``-unwrapped copy the way the pixel critic and other
    DDP-wrapped modules are handled elsewhere in this file.

    The landed code passes through a local variable (``critic =
    self.latent_texture_critic`` a few lines above the call, matching
    the None-check-then-use idiom the distillation function also uses)
    rather than referencing the attribute inline at the call site --
    correct code, and a naive "does the call site's own text contain
    ``self.latent_texture_critic``" check (first draft of this test)
    misses it entirely by only looking at the wrong span. Traced
    properly: find whatever name the call passes as its first argument,
    then confirm THAT name was assigned from ``self.latent_texture_critic``
    with no ``.module`` anywhere between the assignment and the call.
    """
    src = _slice(_af_source(), _GLOSS_START, _GLOSS_END)
    call_start = src.index("generator_surrogate_loss(")
    call = src[call_start:src.index(")", call_start)]
    first_arg = call.split("(", 1)[1].split(",", 1)[0].strip()
    assert first_arg, f"could not parse the first argument out of: {call!r}"
    # The G-term must pass a SNAPSHOT, not the live critic: the
    # distillation calls latent_critic_optimizer.step() later in the same
    # iteration, which is in-place on the weights this graph saved, and
    # the generator's backward then raises "modified by an inplace
    # operation" (observed on all 8 ranks at step 21). So assert the
    # snapshot is what reaches the call...
    assign_pos = src.index(f"{first_arg} = (")
    assert "_surrogate_g_snapshot_critic()" in src[assign_pos:call_start], (
        "the G-term passes something other than the snapshot -- the "
        "in-place hazard is back"
    )
    # ...and that the snapshot itself takes the UNWRAPPED critic (DDP does
    # not support the double backward the Sobolev term needs).
    snap_src = _slice(
        _af_source(),
        "    def _surrogate_g_snapshot_critic(self)",
        "\n    def _compute_pixel_texture_g_loss(\n",
    )
    assert "critic = self.latent_texture_critic" in snap_src
    assert ".module" not in snap_src


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
