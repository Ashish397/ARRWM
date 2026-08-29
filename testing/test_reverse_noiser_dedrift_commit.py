"""Coverage for ``reverse_noiser_dedrift_apply_to_commit`` (2026-08-26).

WHAT THE FLAG DOES (``docs/CARN_FINAL_PLAN.md``).  Every pre-existing CARN
de-drift site corrects a loss-computation COPY (``train_chunk``, the flash
slab, the DMD real target).  None of them touch what the model actually
REMEMBERS.  This flag applies the same learned operator
(``ActionForcingDMD._dedrift_with_reverse_noiser``) to
``commit_input_clean`` -- the tensor fed to the Step-3.4 context-noise
commit forward -- in BOTH commit twins
(``ActionForcingTrainingPipeline.inference_with_trajectory`` and
``.generate_chunk_with_cache``), so the CORRECTED chunk is what every
subsequent chunk conditions on.

WHAT IS PINNED HERE.

  1. REGISTRATION + SEAM.  The flag is registered on ``ActionForcingDMD``
     and defaults False; the trainer installs the model as
     ``pipeline._carn_commit_dedrift_owner`` (the pipeline holds no model
     reference of its own); and the trainer does NOT park a second copy of
     the flag on the pipeline (single source of truth -> the half-enabled
     split-brain state is structurally impossible).

  2. CALL-SITE STRUCTURE, from the shipped AST.  Both twins call
     ``self._reverse_noiser_dedrift_commit`` on ``commit_input_clean``, and
     that call PRECEDES the ``_carn_seam_correct(..., record=True)`` call in
     the same method -- the ordering decision of Risk 5 (learned de-drift
     first, crude affine second) pinned structurally rather than by comment.

  3. DEFAULT-OFF BYTE IDENTITY, against the REAL SHIPPED PIPELINE (loaded
     from its file and run end-to-end with a stand-in generator, the
     ``testing/test_a23_finish_grad.py`` harness).  No owner, owner with the
     flag off, and owner with the flag on but ``reverse_noiser_dedrift_
     enabled`` off all give a bitwise-identical committed tensor, output,
     generator call trace and RNG state.  Each identity has a MUTATION
     CONTROL that flips one flag and shows it then fails.

  4. ON-PATH.  With both flags on and a NON-identity noiser, the tensor
     handed to the commit forward is exactly what a direct
     ``_dedrift_with_reverse_noiser`` call returns, and is NOT the raw
     ladder endpoint -- in both twins.

  5. SEMANTICS THE REVIEWER MUST SEE: the correction is COMMIT-ONLY.  The
     EMITTED chunk (the tensor the DMD/GAN losses score) is unchanged; only
     the KV memory is corrected.  Pinned so nobody mistakes this for the
     ``carn_seam_affine_apply_to_output`` behaviour.

  6. GRADIENT-FLOW GUARD.  ``commit_input_clean`` is ``_commit_src.detach()``
     and the commit forward runs under ``torch.no_grad()``, so the commit
     path was ALREADY graph-free.  Pinned from both sides: (a) the operator
     must not START a graph on the commit tensor (it would keep a ~30M-param
     forward alive for nothing), and (b) the generator's own parameters must
     still receive a nonzero finite gradient through the emitted chunk with
     the flag ON -- the ``boundary_vae_roundtrip`` regression class.

  7. ROLLOUT2-PROPAGATION PROOF.  ``_prebuild_rollout2_for_v24`` builds the
     rollout2 half of the forward/reverse noiser's training pairs by calling
     ``generate_chunk_with_cache`` repeatedly (asserted from the shipped
     AST).  Because the correction lands in the KV cache, chunk N+1 of such
     a sequence is generated from CORRECTED context.  Proved EXECUTABLY with
     a memory-dependent generator stand-in: chunk 2's output differs between
     flag-on and flag-off on the same seed, while chunk 1's does not.

  8. The noiser's ``requires_grad`` flags are restored by the helper.

CPU-only, no CUDA, no dataset.
Run:
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" \
        python -m pytest testing/test_reverse_noiser_dedrift_commit.py -q
"""
from __future__ import annotations

import ast
import importlib.util
import inspect
import math
import os
import sys
import textwrap
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

with patch.object(torch.cuda, "current_device", return_value=0):
    from model.dmd_action_forcing import ActionForcingDMD
    from model.forward_noiser import ForwardNoiser

FLAG = "reverse_noiser_dedrift_apply_to_commit"
OWNER_ATTR = "_carn_commit_dedrift_owner"
WRAPPER = "_reverse_noiser_dedrift_commit"


# ===========================================================================
# Load the REAL pipeline module from its file (t5 touches CUDA at import).
# Same loader as testing/test_a23_finish_grad.py.
# ===========================================================================
def _load_pipeline_module():
    name = "pipeline.action_forcing_training"
    if name in sys.modules:
        return sys.modules[name]
    if "utils.wan_wrapper" not in sys.modules:
        stub = types.ModuleType("utils.wan_wrapper")

        class WanDiffusionWrapper:  # noqa: D401 - type-hint stand-in only
            pass

        stub.WanDiffusionWrapper = WanDiffusionWrapper
        sys.modules["utils.wan_wrapper"] = stub
    if "pipeline" not in sys.modules:
        pkg = types.ModuleType("pipeline")
        pkg.__path__ = [f"{REPO}/pipeline"]
        sys.modules["pipeline"] = pkg
    spec = importlib.util.spec_from_file_location(
        name, f"{REPO}/pipeline/action_forcing_training.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


AFT = _load_pipeline_module()

DENOISING_STEPS = [1000, 750, 500, 250]
FLASH_T = 60
SEED = 1234
NPB = 3
LAT_C, LAT_H, LAT_W = 4, 6, 8


# ===========================================================================
# Fixtures: generator stand-ins, pipeline, owner
# ===========================================================================
class _Gen(torch.nn.Module):
    """Records every forward and makes the graph timestep-addressable
    (``grad(y, gen.w)`` is nonzero exactly at the timesteps in the graph)."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(1024))
        self.model = torch.nn.Module()
        self.calls = []

    def forward(self, *, noisy_image_or_video, conditional_dict, timestep,
                kv_cache, crossattn_cache, current_start):
        ti = int(timestep.reshape(-1)[0].item())
        x0 = noisy_image_or_video * self.w[ti] + 0.001 * float(ti)
        self.calls.append({
            "t": ti,
            "grad_enabled": torch.is_grad_enabled(),
            "in": noisy_image_or_video.detach().clone(),
            "in_requires_grad": bool(noisy_image_or_video.requires_grad),
        })
        return x0, x0


class _MemGen(_Gen):
    """Generator stand-in with an AR MEMORY: the t=0 commit forward writes
    the committed tensor's mean into ``kv_cache[0]['v']`` and every later
    forward adds it. This is the minimal faithful stand-in for "the KV cache
    is what the next chunk conditions on" -- without it no stub could ever
    show commit-site propagation."""

    def forward(self, *, noisy_image_or_video, conditional_dict, timestep,
                kv_cache, crossattn_cache, current_start):
        ti = int(timestep.reshape(-1)[0].item())
        mem = kv_cache[0].get("v")
        bias = 0.0 if mem is None else mem
        x0 = noisy_image_or_video * self.w[ti] + 0.001 * float(ti) + bias
        self.calls.append({
            "t": ti,
            "grad_enabled": torch.is_grad_enabled(),
            "in": noisy_image_or_video.detach().clone(),
            "in_requires_grad": bool(noisy_image_or_video.requires_grad),
        })
        if ti == 0:  # the Step-3.4 context-noise commit forward
            kv_cache[0]["v"] = noisy_image_or_video.detach().mean()
        return x0, x0


class _Sched:
    # ``inference_with_trajectory`` snaps its reported timestep range onto
    # the scheduler grid (``_round_to_grid``), so the grid must exist.
    timesteps = torch.arange(0, 1001, 50, dtype=torch.float32)

    def add_noise(self, x, noise, t):
        return x * 0.9 + noise * 0.1


def _build(gen_cls=_Gen, **attrs):
    gen = gen_cls()
    pipe = AFT.ActionForcingTrainingPipeline(
        denoising_step_list=list(DENOISING_STEPS),
        scheduler=_Sched(),
        generator=gen,
        num_frame_per_block=NPB,
        num_max_frames=NPB,
        rollout_frames=NPB,
        context_noise=0,
    )
    # ``inference_with_trajectory`` REALLOCATES the KV cache itself
    # (``_initialize_kv_cache``) at the production shape
    # [B, kv_cache_frames*1560, 12, 128] x 30 blocks x2 = ~1.7 GB per
    # pipeline, which OOMs a CPU test box. Shrink the sizing constants;
    # the generator stand-ins ignore the cache contents entirely.
    pipe.num_transformer_blocks = 1
    pipe._spatial_frame_seq_length = 4
    pipe.kv_cache1 = [{"k": None, "v": None}]
    pipe.crossattn_cache = [{"k": None, "v": None}]
    for k, v in attrs.items():
        setattr(pipe, k, v)
    return pipe, gen


def _noiser(seed: int = 0, scale: float = 0.05) -> ForwardNoiser:
    """A NON-identity noiser.

    ``ForwardNoiser`` zero-inits ``out_proj`` so a freshly built one returns
    delta == 0 and the de-drift is a numerical no-op -- which would make
    every "on-path" assertion below vacuously true. Perturb ``out_proj`` so
    the operator actually moves the tensor.
    """
    torch.manual_seed(seed)
    n = ForwardNoiser(
        latent_channels=LAT_C, hidden_dim=16, num_blocks=1, max_carn_step=4,
    )
    with torch.no_grad():
        n.out_proj.weight.normal_(0.0, scale)
        n.out_proj.bias.normal_(0.0, scale)
    return n.eval()


class _Owner:
    """Stand-in for ``ActionForcingDMD``: the REAL de-drift helper plus only
    the attributes it and the pipeline wrapper read."""

    _dedrift_with_reverse_noiser = (
        ActionForcingDMD._dedrift_with_reverse_noiser
    )

    def __init__(self, **kw):
        setattr(self, FLAG, False)
        self.reverse_noiser_dedrift_enabled = False
        self.reverse_noiser_dedrift_level = 1
        self.reverse_noiser_dedrift_min_level = 1
        self.reverse_noiser_dedrift_steps = 1
        self.reverse_noiser_dedrift_alpha0 = 1.0
        self.reverse_noiser_dedrift_alpha_decay = 0.5
        self.reverse_noiser_commit_alpha = 1.0
        self.reverse_noiser_commit_start_step = 0
        self.reverse_noiser_commit_ramp_steps = 0
        self._carn_commit_current_step = 0
        self.fn_pair_mode = "r1_vs_r2"
        self.forward_noiser_cycle_enabled = False
        self.forward_noiser = None
        self.reverse_noiser = None
        for k, v in kw.items():
            setattr(self, k, v)


def _armed_owner(net=None, **kw):
    """Both gates on, cycle mode, non-identity reverse noiser."""
    return _Owner(**{FLAG: True}, reverse_noiser_dedrift_enabled=True,
                  forward_noiser_cycle_enabled=True,
                  reverse_noiser=net if net is not None else _noiser(), **kw)


# ---- runners --------------------------------------------------------------
def _run_gcwc(pipe, *, frames=NPB, flash=False, requires_grad=False,
              start=0, seed=SEED):
    torch.manual_seed(seed)
    noise = torch.randn(1, frames, LAT_C, LAT_H, LAT_W)
    out, _tf, _tt = pipe.generate_chunk_with_cache(
        noise,
        current_start_frame=start,
        requires_grad=requires_grad,
        sync_exit_flags=False,
        force_exit_step=0,
        flash_dmd_enabled=flash,
        flash_dmd_gan_t=FLASH_T,
    )
    return out


def _run_iwt(pipe, *, frames=NPB, flash=False, requires_grad=False,
             seed=SEED):
    torch.manual_seed(seed)
    noise = torch.randn(1, frames, LAT_C, LAT_H, LAT_W)
    out, _tf, _tt = pipe.inference_with_trajectory(
        noise=noise,
        requires_grad=requires_grad,
        flash_dmd_enabled=flash,
        flash_dmd_gan_t=FLASH_T,
    )
    return out


RUNNERS = {"generate_chunk_with_cache": _run_gcwc,
           "inference_with_trajectory": _run_iwt}


def _commit_call(gen, context_noise=0):
    """The Step-3.4 context-noise commit forward (last t==0 call)."""
    hits = [c for c in gen.calls if c["t"] == context_noise]
    assert hits, "no context_noise commit forward recorded"
    return hits[-1]


def _trace(gen):
    return [(c["t"], c["grad_enabled"]) for c in gen.calls]


# ===========================================================================
# 1. Registration, default, and the model->pipeline seam
# ===========================================================================
def test_flag_is_registered_on_the_model_and_defaults_false():
    src = inspect.getsource(ActionForcingDMD.__init__)
    tree = ast.parse(textwrap.dedent(src)).body[0]
    found = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and any(getattr(t, "attr", None) == FLAG
                        for t in node.targets)):
            found = node
    assert found is not None, f"{FLAG} is not registered in __init__."
    consts = [
        n.value for n in ast.walk(found.value) if isinstance(n, ast.Constant)
    ]
    assert FLAG in consts, f"{FLAG} must be read off `args` by name."
    assert False in consts, f"{FLAG} must default to False (byte-identical)."


def _trainer_source():
    with open(f"{REPO}/trainer/causal_action_forcing_train.py") as fh:
        return fh.read()


_SEAM_ANCHOR = "        self.model.inference_pipeline = self.pipeline\n"
_SEAM_END = ('        if self.is_main_process:\n            logging.info(\n'
             '                "[ActionForcing] Pipeline: num_frame_per_block')


def _seam_block():
    src = _trainer_source()
    i = src.index(_SEAM_ANCHOR)
    j = src.index(_SEAM_END, i)
    return textwrap.dedent(src[i:j])


class _StubPipe:
    """A bare object, exactly like the real pipeline before the trainer
    touches it -- the attribute is ABSENT."""


def _run_seam(model=None, **cfg):
    import logging as _logging
    self_stub = SimpleNamespace(
        model=model if model is not None else SimpleNamespace(),
        pipeline=_StubPipe(),
        config=SimpleNamespace(**cfg),
        is_main_process=True,
    )
    ns = {"self": self_stub, "logging": _logging, "sys": sys}
    exec(compile(_seam_block(), "<seam>", "exec"), ns)
    return self_stub


def test_trainer_installs_the_owner_back_reference_on_the_pipeline():
    """The pipeline holds NO model reference of its own, so without this
    the wrapper can never reach ``_dedrift_with_reverse_noiser`` and the
    flag is a silent no-op (this codebase's endemic failure)."""
    model = SimpleNamespace(**{FLAG: False})
    st = _run_seam(model=model)
    assert getattr(st.pipeline, OWNER_ATTR, None) is model, (
        f"MODEL->PIPELINE SEAM BROKEN: {OWNER_ATTR} never reached the "
        "pipeline object."
    )


def test_owner_reference_is_installed_unconditionally():
    """Installed with the flag OFF too: an installed reference is
    distinguishable from 'nobody ever wired it'."""
    for val in (True, False):
        model = SimpleNamespace(
            **{FLAG: val}, reverse_noiser_dedrift_enabled=val,
            reverse_noiser_dedrift_level=1)
        st = _run_seam(model=model)
        assert getattr(st.pipeline, OWNER_ATTR, None) is model


def test_trainer_does_not_park_a_second_copy_of_the_flag_on_the_pipeline():
    """SINGLE SOURCE OF TRUTH. A pipeline-side copy could disagree with the
    model's and produce the half-enabled state silently."""
    model = SimpleNamespace(**{FLAG: True})
    st = _run_seam(model=model)
    assert not hasattr(st.pipeline, FLAG), (
        f"the trainer copied {FLAG} onto the pipeline; the gate must be "
        "read through the owner reference only."
    )


# ===========================================================================
# 2. Call-site structure, lifted from the shipped AST
# ===========================================================================
def _pipeline_method_ast(name):
    src = textwrap.dedent(inspect.getsource(
        getattr(AFT.ActionForcingTrainingPipeline, name)))
    return ast.parse(src).body[0]


def _calls_in(node):
    out = []
    for n in ast.walk(node):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "self"):
            out.append((n.func.attr, n))
    return out


@pytest.mark.parametrize("method", sorted(RUNNERS))
def test_both_twins_call_the_wrapper_on_the_commit_tensor(method):
    fn = _pipeline_method_ast(method)
    hits = [n for a, n in _calls_in(fn) if a == WRAPPER]
    assert len(hits) == 1, (
        f"{method} must call self.{WRAPPER} exactly once; found {len(hits)}."
    )
    args = [a for a in hits[0].args if isinstance(a, ast.Name)]
    assert [a.id for a in args] == ["commit_input_clean"], (
        f"{method}: self.{WRAPPER} must be applied to commit_input_clean "
        "(the tensor that feeds the KV commit forward), not to something "
        "else."
    )


@pytest.mark.parametrize("method", sorted(RUNNERS))
def test_dedrift_precedes_the_seam_affine_at_the_commit_site(method):
    """RISK-5 ORDERING DECISION, pinned structurally: learned de-drift
    FIRST, crude mean/std affine SECOND."""
    fn = _pipeline_method_ast(method)
    dedrift_line = seam_line = None
    for attr, n in _calls_in(fn):
        if attr == WRAPPER:
            dedrift_line = n.lineno
        if attr == "_carn_seam_correct" and any(
                kw.arg == "record" for kw in n.keywords):
            seam_line = n.lineno
    assert dedrift_line is not None and seam_line is not None
    assert dedrift_line < seam_line, (
        f"{method}: the learned de-drift must run BEFORE the seam affine "
        "(see _reverse_noiser_dedrift_commit for the rationale)."
    )


def test_the_reading_class_is_the_pipeline_not_the_dmd_model():
    assert hasattr(AFT.ActionForcingTrainingPipeline, WRAPPER)
    assert not hasattr(ActionForcingDMD, WRAPPER)


def test_prebuild_rollout2_goes_through_generate_chunk_with_cache():
    """The load-bearing call-graph fact behind the "falls out for free"
    claim: rollout2 (the second half of every FN training pair) is built by
    calling the corrected twin."""
    src = textwrap.dedent(
        inspect.getsource(ActionForcingDMD._prebuild_rollout2_for_v24))
    tree = ast.parse(src).body[0]
    names = {
        n.func.attr for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    assert "generate_chunk_with_cache" in names, (
        "_prebuild_rollout2_for_v24 no longer calls generate_chunk_with_"
        "cache -- the commit-site correction would NOT reach the FN "
        "training pairs and docs/CARN_FINAL_PLAN.md's central claim is "
        "stale."
    )


# ===========================================================================
# 3. Default-off byte identity (against the real shipped pipeline)
# ===========================================================================
@pytest.mark.parametrize("method", sorted(RUNNERS))
def test_no_owner_is_byte_identical(method):
    """A bare pipeline (tests, the eval harness) has no owner installed."""
    run = RUNNERS[method]
    pipe_a, gen_a = _build()
    assert not hasattr(pipe_a, OWNER_ATTR)
    out_a = run(pipe_a)
    rng_a = torch.get_rng_state().clone()

    pipe_b, gen_b = _build(**{OWNER_ATTR: _Owner()})
    out_b = run(pipe_b)
    rng_b = torch.get_rng_state().clone()

    assert torch.equal(out_a, out_b)
    assert torch.equal(_commit_call(gen_a)["in"], _commit_call(gen_b)["in"])
    assert _trace(gen_a) == _trace(gen_b)
    assert torch.equal(rng_a, rng_b)


@pytest.mark.parametrize("method", sorted(RUNNERS))
def test_apply_flag_on_but_dedrift_disabled_is_identity(method):
    """The helper's OWN gate: half-enabled must be a no-op, not a partial."""
    run = RUNNERS[method]
    pipe_a, gen_a = _build()
    out_a = run(pipe_a)
    owner = _Owner(**{FLAG: True}, reverse_noiser_dedrift_enabled=False,
                   forward_noiser_cycle_enabled=True, reverse_noiser=_noiser())
    pipe_b, gen_b = _build(**{OWNER_ATTR: owner})
    out_b = run(pipe_b)
    assert torch.equal(out_a, out_b)
    assert torch.equal(_commit_call(gen_a)["in"], _commit_call(gen_b)["in"])


@pytest.mark.parametrize("method", sorted(RUNNERS))
def test_dedrift_enabled_but_apply_flag_off_is_identity(method):
    """The call-site gate: the flash / train_chunk de-drift consumers must
    be able to run WITHOUT dragging the KV commit along."""
    run = RUNNERS[method]
    pipe_a, gen_a = _build()
    out_a = run(pipe_a)
    owner = _Owner(reverse_noiser_dedrift_enabled=True,
                   forward_noiser_cycle_enabled=True, reverse_noiser=_noiser())
    pipe_b, gen_b = _build(**{OWNER_ATTR: owner})
    out_b = run(pipe_b)
    assert torch.equal(out_a, out_b)
    assert torch.equal(_commit_call(gen_a)["in"], _commit_call(gen_b)["in"])


@pytest.mark.parametrize("method", sorted(RUNNERS))
def test_missing_network_and_below_min_level_are_identity(method):
    run = RUNNERS[method]
    pipe_a, gen_a = _build()
    out_a = run(pipe_a)
    for owner in (
        _Owner(**{FLAG: True}, reverse_noiser_dedrift_enabled=True,
               forward_noiser_cycle_enabled=True, reverse_noiser=None),
        _armed_owner(reverse_noiser_dedrift_level=0,
                     reverse_noiser_dedrift_min_level=1),
    ):
        pipe_b, gen_b = _build(**{OWNER_ATTR: owner})
        out_b = run(pipe_b)
        assert torch.equal(out_a, out_b)
        assert torch.equal(
            _commit_call(gen_a)["in"], _commit_call(gen_b)["in"])


@pytest.mark.parametrize("method", sorted(RUNNERS))
def test_mutation_control_both_flags_on_breaks_identity(method):
    """The identity assertions above are only meaningful if arming the flag
    actually changes the committed tensor."""
    run = RUNNERS[method]
    pipe_a, gen_a = _build()
    run(pipe_a)
    pipe_b, gen_b = _build(**{OWNER_ATTR: _armed_owner()})
    run(pipe_b)
    assert not torch.equal(
        _commit_call(gen_a)["in"], _commit_call(gen_b)["in"])


# ===========================================================================
# 4. On-path: the committed tensor IS the de-drifted one
# ===========================================================================
@pytest.mark.parametrize("method", sorted(RUNNERS))
@pytest.mark.parametrize("flash", [False, True])
def test_committed_tensor_equals_a_direct_helper_call(method, flash):
    run = RUNNERS[method]
    net = _noiser()
    pipe_off, gen_off = _build()
    run(pipe_off, flash=flash)
    raw = _commit_call(gen_off)["in"]

    pipe_on, gen_on = _build(**{OWNER_ATTR: _armed_owner(net)})
    run(pipe_on, flash=flash)
    got = _commit_call(gen_on)["in"]

    ref_owner = _armed_owner(net)
    with torch.no_grad():
        expect = ActionForcingDMD._dedrift_with_reverse_noiser(
            ref_owner, raw, 1)
    assert torch.equal(got, expect)
    assert not torch.equal(got, raw)


@pytest.mark.parametrize("method", sorted(RUNNERS))
def test_level_comes_from_the_shared_flat_knob(method):
    run = RUNNERS[method]
    net = _noiser()
    pipe_off, gen_off = _build()
    run(pipe_off)
    raw = _commit_call(gen_off)["in"]

    owner = _armed_owner(net, reverse_noiser_dedrift_level=3,
                         reverse_noiser_dedrift_steps=2)
    pipe_on, gen_on = _build(**{OWNER_ATTR: owner})
    run(pipe_on)
    with torch.no_grad():
        expect3 = ActionForcingDMD._dedrift_with_reverse_noiser(owner, raw, 3)
        expect1 = ActionForcingDMD._dedrift_with_reverse_noiser(
            _armed_owner(net), raw, 1)
    assert torch.equal(_commit_call(gen_on)["in"], expect3)
    assert not torch.equal(expect3, expect1), (
        "the fixture's noiser is level-insensitive, so this test proves "
        "nothing -- pick a different seed/scale."
    )


def test_absolute_commit_level_matches_streaming_slab_position_rule():
    net = _noiser()
    owner = _armed_owner(
        net,
        reverse_noiser_commit_use_absolute_level=True,
        num_frame_per_block=NPB,
        dmd_context_clean_frames=3 * NPB,
        forward_noiser_max_carn_step=4,
    )
    pipe, _ = _build(**{OWNER_ATTR: owner})
    x = torch.randn(1, NPB, LAT_C, LAT_H, LAT_W)
    # With three clean seed chunks, frame 15 is chunk index 5 and therefore
    # CARN level 5-(3-1)=3, exactly the streaming-slab convention.
    with torch.no_grad():
        got = pipe._reverse_noiser_dedrift_commit(x, frame_start=5 * NPB)
        expect = owner._dedrift_with_reverse_noiser(x, 3)
        wrong = owner._dedrift_with_reverse_noiser(x, 1)
    assert torch.equal(got, expect)
    assert not torch.equal(got, wrong)
    assert pipe._last_extension_metrics["carn_commit_dedrift_level"] == 3.0


def test_absolute_commit_level_requires_an_explicit_aligned_position():
    owner = _armed_owner(
        reverse_noiser_commit_use_absolute_level=True,
        num_frame_per_block=NPB,
        dmd_context_clean_frames=3 * NPB,
        forward_noiser_max_carn_step=4,
    )
    pipe, _ = _build(**{OWNER_ATTR: owner})
    x = torch.randn(1, NPB, LAT_C, LAT_H, LAT_W)
    with pytest.raises(RuntimeError, match="requires frame_start"):
        pipe._reverse_noiser_dedrift_commit(x)
    with pytest.raises(RuntimeError, match="npb-aligned"):
        pipe._reverse_noiser_dedrift_commit(x, frame_start=NPB + 1)


@pytest.mark.parametrize("method", sorted(RUNNERS))
def test_the_correction_is_commit_only_the_emitted_chunk_is_unchanged(method):
    """DOCUMENTED SEMANTICS, not an accident. Only the AR memory is
    corrected; the tensor the DMD/GAN losses score is byte-identical. If a
    future change makes the emitted chunk corrected too, this test must be
    updated DELIBERATELY (it is the difference between "the model remembers
    a corrected chunk" and "the losses score a corrected chunk")."""
    run = RUNNERS[method]
    pipe_off, gen_off = _build()
    out_off = run(pipe_off)
    pipe_on, gen_on = _build(**{OWNER_ATTR: _armed_owner()})
    out_on = run(pipe_on)
    assert torch.equal(out_off, out_on)
    assert not torch.equal(
        _commit_call(gen_off)["in"], _commit_call(gen_on)["in"])


def test_telemetry_counter_proves_the_flag_fired():
    """"Prove a flag fired from a counter, never from the patch."""
    pipe_on, _ = _build(**{OWNER_ATTR: _armed_owner()})
    _run_gcwc(pipe_on)
    m = pipe_on._last_extension_metrics
    assert m.get("carn_commit_dedrift_applied", 0.0) >= 1.0
    assert "carn_commit_dedrift_noop" not in m

    pipe_off, _ = _build(**{OWNER_ATTR: _Owner()})
    _run_gcwc(pipe_off)
    assert not any(
        k.startswith("carn_commit_dedrift")
        for k in pipe_off._last_extension_metrics
    )


def test_declined_correction_is_recorded_as_a_noop_not_silently():
    """Flag armed but the helper's own gate declines (level < min_level):
    the config believes it is correcting and is not. That must be visible."""
    owner = _armed_owner(reverse_noiser_dedrift_level=0,
                         reverse_noiser_dedrift_min_level=1)
    pipe, _ = _build(**{OWNER_ATTR: owner})
    _run_gcwc(pipe)
    m = pipe._last_extension_metrics
    assert m.get("carn_commit_dedrift_noop", 0.0) >= 1.0
    assert "carn_commit_dedrift_applied" not in m


def test_commit_only_alpha_does_not_weaken_the_aux_corrector():
    """The new dose blends only the graph-free KV commit output."""
    net = _noiser()
    full_owner = _armed_owner(net)
    weak_owner = _armed_owner(net, reverse_noiser_commit_alpha=0.25)
    pipe_full, gen_full = _build(**{OWNER_ATTR: full_owner})
    pipe_weak, gen_weak = _build(**{OWNER_ATTR: weak_owner})
    _run_gcwc(pipe_full)
    _run_gcwc(pipe_weak)
    raw = _commit_call(gen_full)["in"]
    weak = _commit_call(gen_weak)["in"]
    # Re-run the owner helper on the same pre-commit source by recovering it
    # from a flag-off control; the weak commit must be the exact 0.25 blend.
    pipe_off, gen_off = _build()
    _run_gcwc(pipe_off)
    x = _commit_call(gen_off)["in"]
    expect_full = full_owner._dedrift_with_reverse_noiser(x, 1)
    assert torch.allclose(raw, expect_full)
    assert torch.allclose(weak, x + 0.25 * (expect_full - x))
    assert weak_owner.reverse_noiser_dedrift_alpha0 == 1.0


def test_commit_warmup_and_ramp_are_step_resolved():
    net = _noiser()
    owner = _armed_owner(
        net,
        reverse_noiser_commit_alpha=0.25,
        reverse_noiser_commit_start_step=100,
        reverse_noiser_commit_ramp_steps=100,
        _carn_commit_current_step=100,
    )
    pipe, gen = _build(**{OWNER_ATTR: owner})
    _run_gcwc(pipe)
    pipe_off, gen_off = _build()
    _run_gcwc(pipe_off)
    assert torch.equal(_commit_call(gen)["in"], _commit_call(gen_off)["in"])
    assert pipe._last_extension_metrics["carn_commit_dedrift_alpha"] == 0.0

    owner._carn_commit_current_step = 150
    pipe2, gen2 = _build(**{OWNER_ATTR: owner})
    _run_gcwc(pipe2)
    x = _commit_call(gen_off)["in"]
    full = owner._dedrift_with_reverse_noiser(x, 1)
    assert torch.allclose(_commit_call(gen2)["in"], x + 0.125 * (full - x))
    assert pipe2._last_extension_metrics["carn_commit_dedrift_alpha"] == 0.125

    owner._carn_commit_current_step = 200
    pipe3, gen3 = _build(**{OWNER_ATTR: owner})
    _run_gcwc(pipe3)
    assert torch.allclose(_commit_call(gen3)["in"], x + 0.25 * (full - x))
    assert pipe3._last_extension_metrics["carn_commit_dedrift_alpha"] == 0.25


def test_commit_displacement_is_published_only_on_existing_sync_milestones():
    owner = _armed_owner(reverse_noiser_commit_alpha=0.25)
    pipe, _ = _build(**{OWNER_ATTR: owner})
    _run_gcwc(pipe)
    metrics = pipe._last_extension_metrics
    assert metrics["carn_commit_dedrift_rel_step"] == 0.0
    assert math.isfinite(metrics["carn_commit_dedrift_rel"])
    assert metrics["carn_commit_dedrift_rel"] >= 0.0


# ===========================================================================
# 5. Gradient-flow guard
# ===========================================================================
@pytest.mark.parametrize("method", sorted(RUNNERS))
def test_commit_tensor_stays_graph_free_with_the_flag_on(method):
    """``commit_input_clean`` is ``_commit_src.detach()`` and the commit
    forward runs under ``no_grad``. The operator must not start a graph
    there -- that would keep a ~30M-param forward alive for nothing."""
    run = RUNNERS[method]
    pipe, gen = _build(**{OWNER_ATTR: _armed_owner()})
    run(pipe, requires_grad=True)
    c = _commit_call(gen)
    assert c["in_requires_grad"] is False
    assert c["grad_enabled"] is False


@pytest.mark.parametrize("method", sorted(RUNNERS))
def test_generator_parameters_still_receive_gradient_with_the_flag_on(method):
    """The ``boundary_vae_roundtrip`` regression class: new plumbing on the
    commit path silently severing the student's gradient. The emitted chunk
    must still reach the generator's own parameters."""
    run = RUNNERS[method]
    pipe, gen = _build(**{OWNER_ATTR: _armed_owner()})
    out = run(pipe, requires_grad=True)
    assert out.requires_grad, (
        "the emitted chunk lost its autograd connection -- the commit "
        "de-drift severed the DMD gradient."
    )
    g, = torch.autograd.grad(out.sum(), gen.w)
    assert torch.isfinite(g).all()
    assert float(g.abs().sum()) > 0.0


@pytest.mark.parametrize("method", sorted(RUNNERS))
def test_gradient_control_flag_off_reaches_the_same_parameters(method):
    """Control: the assertion above must be about the flag, not the
    harness -- and the gradient must be the SAME (the commit path was
    already graph-free, so R cannot change it)."""
    run = RUNNERS[method]
    pipe_off, gen_off = _build()
    out_off = run(pipe_off, requires_grad=True)
    g_off, = torch.autograd.grad(out_off.sum(), gen_off.w)
    pipe_on, gen_on = _build(**{OWNER_ATTR: _armed_owner()})
    out_on = run(pipe_on, requires_grad=True)
    g_on, = torch.autograd.grad(out_on.sum(), gen_on.w)
    assert float(g_off.abs().sum()) > 0.0
    assert torch.equal(g_off, g_on), (
        "the commit de-drift changed the generator's gradient -- it must "
        "not, the commit path is detached + no_grad by construction."
    )


def test_noiser_requires_grad_flags_are_restored():
    net = _noiser()
    for p in net.parameters():
        p.requires_grad_(True)
    pipe, _ = _build(**{OWNER_ATTR: _armed_owner(net)})
    _run_gcwc(pipe)
    assert all(p.requires_grad for p in net.parameters()), (
        "the de-drift left the noiser's params frozen -- its optimizer "
        "would silently stop training."
    )


# ===========================================================================
# 6. ROLLOUT2-PROPAGATION PROOF (executable)
# ===========================================================================
def _two_chunk_rollout(owner=None, net=None):
    """Roll two consecutive chunks through the SHIPPED
    ``generate_chunk_with_cache`` against a MEMORY-DEPENDENT generator, the
    way ``_prebuild_rollout2_for_v24`` rolls rollout2."""
    attrs = {} if owner is None else {OWNER_ATTR: owner}
    pipe, gen = _build(gen_cls=_MemGen, **attrs)
    c1 = _run_gcwc(pipe, start=0, seed=SEED)
    c2 = _run_gcwc(pipe, start=NPB, seed=SEED + 1)
    return c1, c2, gen


def test_commit_correction_propagates_to_the_next_chunk():
    """THE CENTRAL CLAIM of docs/CARN_FINAL_PLAN.md, proved rather than
    trusted: correcting the KV commit changes what the NEXT chunk is
    generated from. Chunk 1 is identical (the correction lands after its
    output is emitted); chunk 2 is NOT."""
    net = _noiser()
    c1_off, c2_off, gen_off = _two_chunk_rollout()
    c1_on, c2_on, gen_on = _two_chunk_rollout(owner=_armed_owner(net))

    assert torch.equal(c1_off, c1_on), (
        "chunk 1's EMITTED output should be unchanged (commit-only "
        "correction)."
    )
    assert not torch.equal(c2_off, c2_on), (
        "PROPAGATION FAILED: chunk 2 was generated identically with and "
        "without the commit correction, so the corrected chunk is NOT what "
        "the model conditions on and the plan's central claim is false."
    )
    # ...and the mechanism really is the KV memory, not some other channel.
    assert not torch.equal(gen_off.calls[0]["in"], gen_off.calls[0]["in"] * 0)
    off_first_of_chunk2 = [c for c in gen_off.calls if c["t"] == 1000][1]
    on_first_of_chunk2 = [c for c in gen_on.calls if c["t"] == 1000][1]
    assert torch.equal(off_first_of_chunk2["in"], on_first_of_chunk2["in"]), (
        "chunk 2's INPUT NOISE differed -- the harness is not controlled, "
        "so the propagation assertion above proves nothing."
    )


def test_propagation_control_flag_off_gives_an_identical_second_chunk():
    """Control for the test above: two flag-off rollouts must agree
    exactly, so the difference there is attributable to the flag."""
    a1, a2, _ = _two_chunk_rollout()
    b1, b2, _ = _two_chunk_rollout(owner=_Owner())
    assert torch.equal(a1, b1)
    assert torch.equal(a2, b2)


def test_every_commit_in_a_multi_chunk_rollout_is_corrected():
    """Not just the first: R runs at EVERY chunk commit (Risk 4, cost)."""
    pipe, gen = _build(gen_cls=_MemGen, **{OWNER_ATTR: _armed_owner()})
    _run_gcwc(pipe, start=0, seed=SEED)
    _run_gcwc(pipe, start=NPB, seed=SEED + 1)
    assert pipe._last_extension_metrics.get(
        "carn_commit_dedrift_applied", 0.0) >= 1.0
    assert float(getattr(pipe, "_carn_commit_dedrift_calls", 0.0)) == 2.0
