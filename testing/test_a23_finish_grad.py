"""A23 / WP-PIXGAN — grad-on FINAL finish rung + inference-parity fake.

Locks the behaviour of ``pix_finish_grad_enabled`` in
``pipeline/action_forcing_training.py::generate_chunk_with_cache``:

  1. gate absent == gate False == byte-identical (same outputs, same
     generator call trace, same RNG consumption, no new info key);
  2. gate on  -> ``_clean_chunk_grad`` requires grad and
     ``torch.autograd.grad`` reaches the generator's own parameters;
  3. EXACTLY ONE rung is attached (earlier finish rungs and the exit
     rung contribute nothing to that buffer's graph);
  4. the KV commit tensor and the original ``clean_chunk`` buffer stay
     DETACHED even with the gate on (paper 3.3 cross-timestep
     decoupling);
  5. with flash ON, the grad buffer holds the PRE-FLASH ladder
     endpoint, not the t=flash_dmd_gan_t flash output;
  6. the frames that could NOT attach a grad rung are left at the
     buffer's ZEROS init -- they are not backfilled with anything,
     and in particular not with the flash tensor. Section 6 below
     covers flash-ON *with* a detached block, which is the
     combination the earlier tests never exercised together.

The pipeline package cannot be imported normally on a CPU node
(``pipeline/__init__.py`` -> ``utils.wan_wrapper`` -> ``wan.modules.t5``
touches ``torch.cuda.current_device()`` at import time), so the module
is loaded from its file path with the heavy wrapper stubbed. Nothing in
the code under test uses ``WanDiffusionWrapper`` beyond a type hint.

The generator stand-in indexes a 1024-wide parameter vector BY THE
INTEGER TIMESTEP of the forward. That makes the autograd graph
self-reporting: ``grad(buffer.sum(), gen.w)`` is non-zero exactly at the
timesteps whose forwards are in the graph, which is how tests 3 and 5
are decided rather than by inspecting ``grad_fn`` names.

Run: python -m pytest testing/test_a23_finish_grad.py -q
"""
from __future__ import annotations

import importlib.util
import sys
import types

import torch

REPO = "/scratch/u6ex/as1748.u6ex/ARRWM"


def _load_pipeline_module():
    name = "pipeline.action_forcing_training"
    if name in sys.modules:
        return sys.modules[name]
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
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


class _Gen(torch.nn.Module):
    """Records every forward and makes the graph timestep-addressable."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(1024))
        # ``_inner_model`` reads ``generator.model``; a bare Module has
        # no ``set_adapter``/``peft_config`` so phase-LoRA dispatch is a
        # no-op, and no ``local_attn_size`` so the window is "unknown".
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
            "out": x0,
        })
        return x0, x0


class _Sched:
    def add_noise(self, x, noise, t):
        return x * 0.9 + noise * 0.1


def _build(**attrs):
    gen = _Gen()
    pipe = AFT.ActionForcingTrainingPipeline(
        denoising_step_list=list(DENOISING_STEPS),
        scheduler=_Sched(),
        generator=gen,
        num_frame_per_block=3,
        num_max_frames=3,
        rollout_frames=3,
        context_noise=0,
    )
    pipe.kv_cache1 = [{"k": None, "v": None}]
    pipe.crossattn_cache = [{"k": None, "v": None}]
    for k, v in attrs.items():
        setattr(pipe, k, v)
    return pipe, gen


def _run(pipe, *, frames=3, flash=False, requires_grad=True,
         force_exit_step=0):
    torch.manual_seed(SEED)
    noise = torch.randn(1, frames, 4, 6, 8)
    out, t_from, t_to = pipe.generate_chunk_with_cache(
        noise,
        current_start_frame=0,
        requires_grad=requires_grad,
        sync_exit_flags=False,
        force_exit_step=force_exit_step,
        flash_dmd_enabled=flash,
        flash_dmd_gan_t=FLASH_T,
    )
    return out


def _trace(gen):
    return [(c["t"], c["grad_enabled"]) for c in gen.calls]


def _commit_call(gen, context_noise=0):
    """The Step-3.4 context-noise commit forward (last t==0 call)."""
    hits = [c for c in gen.calls if c["t"] == context_noise]
    assert hits, "no context_noise commit forward recorded"
    return hits[-1]


# ---------------------------------------------------------------------
# 1. gate off == byte-identical
# ---------------------------------------------------------------------

def test_gate_absent_equals_gate_false_bitwise():
    pipe_a, gen_a = _build()                      # attribute ABSENT
    assert not hasattr(pipe_a, "pix_finish_grad_enabled")
    out_a = _run(pipe_a)
    rng_a = torch.get_rng_state().clone()

    pipe_b, gen_b = _build(pix_finish_grad_enabled=False)
    out_b = _run(pipe_b)
    rng_b = torch.get_rng_state().clone()

    assert torch.equal(out_a, out_b)
    assert torch.equal(pipe_a._clean_chunk, pipe_b._clean_chunk)
    assert torch.equal(
        _commit_call(gen_a)["in"], _commit_call(gen_b)["in"])
    # Same forwards, same order, same grad state, same RNG consumption.
    assert _trace(gen_a) == _trace(gen_b)
    assert torch.equal(rng_a, rng_b)


def test_gate_off_publishes_nothing():
    for attrs in ({}, {"pix_finish_grad_enabled": False}):
        pipe, _gen = _build(**attrs)
        _run(pipe)
        assert getattr(pipe, "_clean_chunk_grad", None) is None
        assert getattr(pipe, "_clean_chunk_grad_mask", None) is None
        # no A23 telemetry leaks into the per-call metrics dict either
        assert not any(
            k.startswith("pix_finish_grad")
            for k in pipe._last_extension_metrics
        )


def test_gate_on_does_not_change_values_or_rng_order():
    """The gate adds a graph, never a different number, and never a
    different RNG draw — the committed/returned tensors are unchanged."""
    pipe_off, gen_off = _build(pix_finish_grad_enabled=False)
    out_off = _run(pipe_off)
    rng_off = torch.get_rng_state().clone()

    pipe_on, gen_on = _build(pix_finish_grad_enabled=True)
    out_on = _run(pipe_on)
    rng_on = torch.get_rng_state().clone()

    assert torch.equal(out_off, out_on)
    assert torch.equal(pipe_off._clean_chunk, pipe_on._clean_chunk)
    assert torch.equal(
        _commit_call(gen_off)["in"], _commit_call(gen_on)["in"])
    assert torch.equal(rng_off, rng_on)
    # identical forward sequence; only the FINAL finish rung flips to
    # grad-enabled.
    assert [c["t"] for c in gen_off.calls] == [c["t"] for c in gen_on.calls]
    diffs = [
        (a["t"], a["grad_enabled"], b["grad_enabled"])
        for a, b in zip(gen_off.calls, gen_on.calls)
        if a["grad_enabled"] != b["grad_enabled"]
    ]
    assert diffs == [(DENOISING_STEPS[-1], False, True)], diffs


# ---------------------------------------------------------------------
# 2. gate on -> real graph to generator weights
# ---------------------------------------------------------------------

def test_gate_on_buffer_requires_grad_and_reaches_weights():
    pipe, gen = _build(pix_finish_grad_enabled=True)
    _run(pipe)
    buf = pipe._clean_chunk_grad
    assert buf is not None
    assert buf.requires_grad
    assert buf.grad_fn is not None
    g = torch.autograd.grad(buf.sum(), gen.w, allow_unused=True)[0]
    assert g is not None
    assert torch.count_nonzero(g) > 0
    assert torch.isfinite(g).all()


def test_in_place_slice_write_into_zeros_buffer_carries_grad():
    """The buffer is a preallocated ``torch.zeros_like`` written by
    in-place slice assignment — verify (not assume) that autograd
    survives that, on a SINGLE-block call where every frame is live."""
    pipe, gen = _build(pix_finish_grad_enabled=True)
    _run(pipe, frames=3)
    buf = pipe._clean_chunk_grad
    assert buf is not None and buf.requires_grad
    assert buf.grad_fn is not None          # CopySlices, not a leaf
    g = torch.autograd.grad(buf.sum(), gen.w, allow_unused=True)[0]
    assert torch.count_nonzero(g) > 0


def test_multi_block_call_attaches_every_block_EXCEPT_the_trailing_one():
    """DOCUMENTED behaviour, not a defect. ``finish_grad_active``
    mirrors ``flash_grad_active``: on a MULTI-block call the trailing
    block is deliberately left no_grad, implementing the researcher's
    "6 chunks instead of 7" memory intent for the heavy iter-1
    rollout. Single-block calls (iters k>=2, ~99% of streaming calls)
    keep their only block live — see the separate test above.

    3 blocks of npb=3: blocks 0 and 1 live, block 2 DETACHED.
    """
    pipe, gen = _build(pix_finish_grad_enabled=True)
    _run(pipe, frames=9)                    # 3 blocks of npb=3
    buf = pipe._clean_chunk_grad
    assert buf is not None and buf.requires_grad

    def _live(sl):
        g = torch.autograd.grad(
            buf[:, sl].sum(), gen.w,
            allow_unused=True, retain_graph=True)[0]
        return g is not None and bool(torch.count_nonzero(g) > 0)

    assert _live(slice(0, 3)), "block 0 must be live"
    assert _live(slice(3, 6)), "block 1 must be live"
    assert not _live(slice(6, 9)), (
        "block 2 is the TRAILING block of a multi-block call and must "
        "be detached (documented memory intent)"
    )


def test_flash_buffer_publishes_the_same_partial_liveness_mask():
    pipe, gen = _build()
    _run(pipe, frames=9, flash=True)
    buf = pipe._flash_dmd_gan_output
    mask = pipe._flash_dmd_gan_grad_mask
    assert buf is not None and buf.requires_grad
    assert mask.tolist() == [True] * 6 + [False] * 3

    def _live(sl):
        g = torch.autograd.grad(
            buf[:, sl].sum(), gen.w,
            allow_unused=True, retain_graph=True,
        )[0]
        return g is not None and bool(torch.count_nonzero(g) > 0)

    assert _live(slice(0, 6))
    assert not _live(slice(6, 9))


def test_s7_aux_gate_keeps_trailing_exit_and_flash_blocks_graph_live():
    """Phase-3 S7 experiment: auxiliaries may train the trailing block
    without changing the independent DMD mask.  Prove both graph sources,
    rather than trusting the boolean or whole-buffer ``requires_grad``."""
    pipe, gen = _build()
    pipe.aux_train_trailing_chunk = True
    out = _run(pipe, frames=9, flash=True)
    flash = pipe._flash_dmd_gan_output
    mask = pipe._flash_dmd_gan_grad_mask
    assert flash is not None and flash.requires_grad
    assert mask.tolist() == [True] * 9

    exit_g = torch.autograd.grad(
        out[:, 6:9].sum(), gen.w,
        allow_unused=True, retain_graph=True,
    )[0]
    flash_g = torch.autograd.grad(
        flash[:, 6:9].sum(), gen.w,
        allow_unused=True, retain_graph=True,
    )[0]
    assert exit_g is not None and torch.count_nonzero(exit_g) > 0
    assert flash_g is not None and torch.count_nonzero(flash_g) > 0


def test_s7_aux_gate_extends_optional_finish_graph_to_trailing_block():
    pipe, gen = _build(pix_finish_grad_enabled=True)
    pipe.aux_train_trailing_chunk = True
    _run(pipe, frames=9)
    buf = pipe._clean_chunk_grad
    assert buf is not None and buf.requires_grad
    g = torch.autograd.grad(
        buf[:, 6:9].sum(), gen.w, allow_unused=True,
    )[0]
    assert g is not None and torch.count_nonzero(g) > 0
    assert pipe._clean_chunk_grad_mask.tolist() == [True] * 9


def test_telemetry_counts_attached_blocks_excluding_trailing_block():
    """``pix_finish_grad_blocks`` counts blocks that actually attached
    a grad rung, which on a multi-block call is n_blocks - 1."""
    pipe, _gen = _build(pix_finish_grad_enabled=True)
    _run(pipe, frames=9)                    # 3 blocks
    st = pipe._pix_finish_grad_stats
    assert st["pix_finish_grad_blocks"] == 2.0          # 3 - 1
    assert st["pix_finish_grad_no_rung"] == 0.0
    assert st["pix_finish_grad_nograd"] == 0.0
    assert st["pix_finish_grad_published"] == 1.0
    # surfaced through the existing per-call metrics channel
    assert pipe._last_extension_metrics["pix_finish_grad_blocks"] == 2.0


def test_single_block_call_attaches_its_only_block():
    pipe, _gen = _build(pix_finish_grad_enabled=True)
    _run(pipe, frames=3)
    st = pipe._pix_finish_grad_stats
    assert st["pix_finish_grad_blocks"] == 1.0
    assert st["pix_finish_grad_published"] == 1.0


# ---------------------------------------------------------------------
# 3b. frame-level attachment mask — the anti-silent-dilution guard
# ---------------------------------------------------------------------

def test_mask_is_all_true_on_a_single_block_call():
    pipe, _gen = _build(pix_finish_grad_enabled=True)
    _run(pipe, frames=3)
    m = pipe._clean_chunk_grad_mask
    assert m is not None
    assert m.dtype == torch.bool
    assert m.shape == (3,)
    assert bool(m.all())


def test_mask_marks_only_the_trailing_block_false():
    pipe, _gen = _build(pix_finish_grad_enabled=True)
    _run(pipe, frames=9)                    # 3 blocks of npb=3
    m = pipe._clean_chunk_grad_mask
    assert m is not None
    assert m.tolist() == [True] * 6 + [False] * 3, m.tolist()


def test_mask_sum_equals_the_logged_frame_count():
    for frames in (3, 6, 9):
        pipe, _gen = _build(pix_finish_grad_enabled=True)
        _run(pipe, frames=frames)
        st = pipe._pix_finish_grad_stats
        m = pipe._clean_chunk_grad_mask
        assert float(int(m.sum())) == st["pix_finish_grad_frames"], frames
        assert st["pix_finish_grad_frames_total"] == float(frames), frames
        # the dilution a mask-ignoring consumer would silently eat
        dilution = 1.0 - (st["pix_finish_grad_frames"]
                          / st["pix_finish_grad_frames_total"])
        assert 0.0 <= dilution < 1.0


def test_no_rung_frames_read_false_in_the_mask():
    """A block whose random exit rung WAS the last rung has no finish
    rung to attach to. Those frames are written DETACHED and must read
    False in the mask, exactly like the trailing-block frames."""
    pipe, gen = _build(pix_finish_grad_enabled=True)
    pipe.same_step_across_blocks = False
    # block 0 exits at the LAST rung -> empty finish loop -> no_rung;
    # block 1 exits at rung 0 -> attaches; block 2 is the trailing
    # block of a multi-block call -> deliberately detached.
    pipe.generate_and_sync_list = (
        lambda *a, **k: [len(DENOISING_STEPS) - 1, 0, 0])
    _run(pipe, frames=9, force_exit_step=None)

    m = pipe._clean_chunk_grad_mask
    assert m is not None
    assert m.tolist() == [False] * 3 + [True] * 3 + [False] * 3, m.tolist()

    st = pipe._pix_finish_grad_stats
    assert st["pix_finish_grad_no_rung"] == 1.0
    assert st["pix_finish_grad_blocks"] == 1.0
    assert st["pix_finish_grad_frames"] == 3.0
    assert st["pix_finish_grad_frames_total"] == 9.0

    # and the graph agrees with the mask
    buf = pipe._clean_chunk_grad
    g0 = torch.autograd.grad(
        buf[:, :3].sum(), gen.w, allow_unused=True, retain_graph=True)[0]
    g1 = torch.autograd.grad(
        buf[:, 3:6].sum(), gen.w, allow_unused=True)[0]
    assert g0 is None or torch.count_nonzero(g0) == 0
    assert g1 is not None and torch.count_nonzero(g1) > 0


def test_mask_is_none_whenever_the_buffer_is_none():
    # gate off
    pipe, _gen = _build()
    _run(pipe)
    assert getattr(pipe, "_clean_chunk_grad_mask", None) is None
    # gate on but nothing attached (every block exits at the last rung)
    pipe, _gen = _build(pix_finish_grad_enabled=True)
    _run(pipe, force_exit_step=len(DENOISING_STEPS) - 1)
    assert pipe._clean_chunk_grad is None
    assert pipe._clean_chunk_grad_mask is None
    assert pipe._pix_finish_grad_stats["pix_finish_grad_frames"] == 0.0


def test_no_finish_rung_is_counted_and_not_silently_gradless():
    """Exit at the LAST rung => the post-exit loop body never runs.
    The buffer must not be published as a grad-carrying fake."""
    pipe, _gen = _build(pix_finish_grad_enabled=True)
    _run(pipe, force_exit_step=len(DENOISING_STEPS) - 1)
    st = pipe._pix_finish_grad_stats
    assert st["pix_finish_grad_no_rung"] == 1.0
    assert st["pix_finish_grad_blocks"] == 0.0
    assert st["pix_finish_grad_published"] == 0.0
    assert pipe._clean_chunk_grad is None


def test_requires_grad_false_caller_gets_no_graph():
    pipe, _gen = _build(pix_finish_grad_enabled=True)
    _run(pipe, requires_grad=False)
    assert pipe._clean_chunk_grad is None


# ---------------------------------------------------------------------
# 3. exactly ONE rung attached
# ---------------------------------------------------------------------

def test_exactly_one_rung_is_in_the_graph():
    pipe, gen = _build(pix_finish_grad_enabled=True)
    _run(pipe, force_exit_step=0)
    buf = pipe._clean_chunk_grad
    g = torch.autograd.grad(buf.sum(), gen.w, allow_unused=True)[0]
    live = torch.nonzero(g).flatten().tolist()
    # ``gen.w`` is indexed by the forward's integer timestep, so a
    # non-zero entry at index t means "the t-rung forward is in this
    # buffer's graph". Exactly one entry may be live: the FINAL rung.
    assert live == [DENOISING_STEPS[-1]], live
    # explicitly: exit rung (1000) and the intermediate finish rungs
    # (750, 500) contribute nothing.
    for t in (DENOISING_STEPS[0], DENOISING_STEPS[1], DENOISING_STEPS[2]):
        assert g[t].item() == 0.0, t


def test_grad_rung_input_is_detached():
    """One-rung bound is structural: the grad rung's INPUT carries no
    graph, so nothing upstream of it can be reached."""
    pipe, gen = _build(pix_finish_grad_enabled=True)
    _run(pipe, force_exit_step=0)
    grad_calls = [
        c for c in gen.calls
        if c["t"] == DENOISING_STEPS[-1] and c["grad_enabled"]
    ]
    assert len(grad_calls) == 1
    assert grad_calls[0]["in_requires_grad"] is False


# ---------------------------------------------------------------------
# 4. commit + clean_chunk stay detached
# ---------------------------------------------------------------------

def test_commit_and_clean_chunk_stay_detached_with_gate_on():
    for flash in (False, True):
        pipe, gen = _build(pix_finish_grad_enabled=True)
        _run(pipe, flash=flash)
        commit = _commit_call(gen)
        assert commit["in_requires_grad"] is False, flash
        assert commit["grad_enabled"] is False, flash
        assert pipe._clean_chunk.requires_grad is False, flash
        assert pipe._clean_chunk.grad_fn is None, flash


# ---------------------------------------------------------------------
# 5. flash on + gate on -> grad buffer is the PRE-FLASH ladder endpoint
# ---------------------------------------------------------------------

def test_flash_on_grad_buffer_is_ladder_endpoint_not_flash():
    pipe, gen = _build(pix_finish_grad_enabled=True)
    _run(pipe, flash=True, force_exit_step=0)

    buf = pipe._clean_chunk_grad
    assert buf is not None and buf.requires_grad

    # The flash forward (t=60) must NOT be in the grad buffer's graph.
    g = torch.autograd.grad(buf.sum(), gen.w, allow_unused=True)[0]
    live = torch.nonzero(g).flatten().tolist()
    assert live == [DENOISING_STEPS[-1]], live
    assert g[FLASH_T].item() == 0.0

    # Value check: the buffer equals the last finish-rung output, and
    # NOT the flash output (which is what ``clean_chunk`` holds).
    ladder = [c for c in gen.calls if c["t"] == DENOISING_STEPS[-1]][-1]["out"]
    flash = [c for c in gen.calls if c["t"] == FLASH_T][-1]["out"]
    assert torch.equal(buf.detach(), ladder.detach())
    assert torch.equal(pipe._clean_chunk, flash.detach())
    assert not torch.equal(buf.detach(), flash.detach())


def test_flash_off_grad_buffer_matches_clean_chunk_values():
    """With flash OFF the two buffers agree in VALUE (same tensor,
    one detached, one not) — the divergence in the test above is
    entirely the flash pass.

    Single-block call, so every frame is mask-True and the whole
    buffer is written; see section 6 for what the mask-FALSE frames
    hold."""
    pipe, _gen = _build(pix_finish_grad_enabled=True)
    _run(pipe, flash=False, force_exit_step=0)
    assert bool(pipe._clean_chunk_grad_mask.all())
    assert torch.equal(
        pipe._clean_chunk_grad.detach(), pipe._clean_chunk)


# ---------------------------------------------------------------------
# 6. the FALLBACK: mask-False frames are ZEROS, never the flash tensor
#
# The defect this section pins. The fallback branch used to write
# ``_grad_slice = cache_pred.detach()`` while its own comment said the
# slice is written with ``finish_grad_pred`` "never ``cache_pred``".
# By that point in the loop ``cache_pred = flash_dmd_pred.detach()``,
# so with flash ON the detached frames held the t=60 FLASH tensor --
# bit-identical, measured max|diff| = 0.0 on a 9-frame / 3-block call.
# The guarding assert lived in the OTHER branch. The mask protected a
# compliant consumer, but anything decoding the whole buffer saw
# precisely the tensor A23 exists to exclude, labelled as the ladder
# endpoint.
#
# Flash ON *together with* a detached block was untested until now:
# ``test_flash_on_...`` ran a single block (all frames live) and
# ``test_multi_block_...`` ran flash OFF.
# ---------------------------------------------------------------------

def _detached_frame_report(buf, mask, gen):
    """``(nonzero_frames, flash_contaminated_frames)`` over the mask-FALSE
    frames of an A23 grad buffer.

    Takes the buffer explicitly rather than reading it off the pipe so the
    planted-violation companion can feed it a deliberately corrupted one.
    """
    b = buf.detach()
    flash_frames = []
    for c in gen.calls:
        if c["t"] == FLASH_T:
            o = c["out"].detach()
            flash_frames.extend(o[:, i] for i in range(o.shape[1]))
    nonzero, contaminated = [], []
    for f in range(b.shape[1]):
        if bool(mask[f]):
            continue
        sl = b[:, f]
        if float(sl.abs().sum()) != 0.0:
            nonzero.append(f)
        if any(torch.equal(sl, ff) for ff in flash_frames):
            contaminated.append(f)
    return nonzero, contaminated


def test_flash_ON_with_a_detached_block_leaves_zeros_not_the_flash_tensor():
    """THE REGRESSION TEST. 3 blocks of npb=3 with flash ON: blocks 0
    and 1 attach, block 2 is the trailing block and must not."""
    pipe, gen = _build(pix_finish_grad_enabled=True)
    _run(pipe, frames=9, flash=True, force_exit_step=0)

    buf, mask = pipe._clean_chunk_grad, pipe._clean_chunk_grad_mask
    assert buf is not None and mask is not None
    assert mask.tolist() == [True] * 6 + [False] * 3, mask.tolist()

    nonzero, contaminated = _detached_frame_report(buf, mask, gen)
    assert contaminated == [], (
        f"frames {contaminated} of the A23 grad buffer hold the FLASH "
        "tensor -- exactly what A23 exists to exclude")
    assert nonzero == [], (
        f"frames {nonzero} were backfilled; the fallback must leave the "
        "zeros init so nothing can be mistaken for a ladder endpoint")

    # and the flash forward really did produce a different, non-zero
    # tensor for that block -- i.e. the zeros are a choice, not an
    # accident of a flash pass that never ran.
    flash_out = [c for c in gen.calls if c["t"] == FLASH_T][-1]["out"].detach()
    assert float(flash_out.abs().sum()) > 0.0
    assert torch.equal(pipe._clean_chunk[:, 6:9], flash_out)
    assert not torch.equal(buf.detach()[:, 6:9], flash_out)


def test_the_zeros_rule_holds_with_flash_OFF_too():
    """One invariant, no flash-dependence: mask-False => exactly zeros."""
    pipe, gen = _build(pix_finish_grad_enabled=True)
    _run(pipe, frames=9, flash=False, force_exit_step=0)
    nonzero, contaminated = _detached_frame_report(
        pipe._clean_chunk_grad, pipe._clean_chunk_grad_mask, gen)
    assert contaminated == [] and nonzero == []


def test_no_rung_frames_are_zeros_too():
    """The other way a block fails to attach: its exit rung IS the last
    rung, so the finish loop is empty. Same rule."""
    pipe, gen = _build(pix_finish_grad_enabled=True)
    pipe.same_step_across_blocks = False
    pipe.generate_and_sync_list = (
        lambda *a, **k: [len(DENOISING_STEPS) - 1, 0, 0])
    _run(pipe, frames=9, flash=True, force_exit_step=None)
    mask = pipe._clean_chunk_grad_mask
    assert mask.tolist() == [False] * 3 + [True] * 3 + [False] * 3
    nonzero, contaminated = _detached_frame_report(
        pipe._clean_chunk_grad, mask, gen)
    assert contaminated == [] and nonzero == []


def test_the_zeros_guard_FIRES_on_a_planted_flash_backfill():
    """PLANTED VIOLATION: re-create the old fallback by hand and prove the
    guard above trips. A guard nobody has seen fire is not evidence."""
    pipe, gen = _build(pix_finish_grad_enabled=True)
    _run(pipe, frames=9, flash=True, force_exit_step=0)
    mask = pipe._clean_chunk_grad_mask

    # exactly what the old ``_grad_slice = cache_pred.detach()`` did
    planted = pipe._clean_chunk_grad.detach().clone()
    flash_out = [c for c in gen.calls if c["t"] == FLASH_T][-1]["out"].detach()
    planted[:, 6:9] = flash_out

    nonzero, contaminated = _detached_frame_report(planted, mask, gen)
    assert contaminated == [6, 7, 8], contaminated
    assert nonzero == [6, 7, 8], nonzero

    # a non-flash backfill is still caught by the zeros half of the rule
    planted2 = pipe._clean_chunk_grad.detach().clone()
    planted2[:, 6:9] = 1.5
    nz2, cont2 = _detached_frame_report(planted2, mask, gen)
    assert nz2 == [6, 7, 8] and cont2 == []


# =====================================================================
# 7. A24 / WP-PIXGAN -- the ``pix_kv_commit_check_every`` KV tripwire
#
# ``_exit_fn`` / ``_finish_fn`` / ``_flash_fn`` are checkpointed with
# ``use_reentrant=False`` and every one of them WRITES the block's K/V
# slots. In FORWARD order Step 3.4's context-noise commit writes last,
# and that graph-free state is what paper 3.3 requires. At BACKWARD each
# checkpoint is RECOMPUTED and re-writes the SAME slots, and nothing
# re-runs the commit afterwards. The failure is silent: finite, well
# scaled, simply wrong K/V.
#
# The stubs below add the one thing the earlier stubs lacked -- a
# generator that actually writes K/V -- so the hazard is reproducible on
# CPU with no model.
#
# NAMING, because two different things are called "block":
#   ``pix_kv_commit_first_mismatch_block`` = the ROLLOUT block index
#       (``block_index`` of the ``all_num_frames`` loop);
#   ``pix_kv_commit_first_mismatch_layer`` = the TRANSFORMER block index
#       (position in ``kv_cache1``).
# =====================================================================

KV_LAYERS = 3
KV_TOKENS = 8


class _KVGen(_Gen):
    """``_Gen`` plus the side effect that matters: every forward WRITES
    the K/V slots, exactly like the real DiT's causal attention does.

    The written value is a function of (input tensor, timestep), so two
    forwards at different rungs leave DIFFERENT K/V behind -- which is
    the whole reason a backward-time recompute is destructive.
    """

    def __init__(self, layers=KV_LAYERS, tokens=KV_TOKENS):
        super().__init__()
        self.layers = layers
        self.tokens = tokens
        self.kv_writes = []

    def forward(self, *, noisy_image_or_video, conditional_dict, timestep,
                kv_cache, crossattn_cache, current_start):
        ti = int(timestep.reshape(-1)[0].item())
        # NOTE the ``tanh``. ``_Gen``'s pure ``x * w[t]`` needs no INTERNAL
        # activation for its backward, so a non-reentrant checkpoint never
        # has to unpack anything and never recomputes -- measured. A real
        # transformer block does need its internals, so the stub grows one
        # genuine intermediate; otherwise this suite would "prove" the
        # hazard absent for a reason that has nothing to do with the
        # pipeline.
        hidden = torch.tanh(noisy_image_or_video * self.w[ti])
        x0 = hidden * self.w[ti] + 0.001 * float(ti)
        self.calls.append({
            "t": ti,
            "grad_enabled": torch.is_grad_enabled(),
            "in": noisy_image_or_video.detach().clone(),
            "in_requires_grad": bool(noisy_image_or_video.requires_grad),
            "out": x0,
        })
        out = (x0, x0)
        with torch.no_grad():
            sig = (
                float(noisy_image_or_video.detach().float().mean())
                + 0.01 * float(ti)
            )
            for li, blk in enumerate(kv_cache):
                for name in ("k", "v"):
                    t = blk.get(name)
                    if t is None:
                        continue
                    ramp = torch.arange(
                        t.numel(), dtype=torch.float32,
                    ).reshape(t.shape)
                    t.copy_(
                        sig * (1.0 + 0.001 * ramp)
                        + 0.1 * li
                        + (0.5 if name == "v" else 0.0)
                    )
        self.kv_writes.append((ti, bool(torch.is_grad_enabled())))
        return out


def _build_kv(**attrs):
    """``_build`` with REAL K/V buffers and a K/V-writing generator."""
    gen = _KVGen()
    pipe = AFT.ActionForcingTrainingPipeline(
        denoising_step_list=list(DENOISING_STEPS),
        scheduler=_Sched(),
        generator=gen,
        num_frame_per_block=3,
        num_max_frames=3,
        rollout_frames=3,
        context_noise=0,
    )
    pipe.kv_cache1 = [
        {
            "k": torch.zeros(1, KV_TOKENS, 2, 4),
            "v": torch.zeros(1, KV_TOKENS, 2, 4),
        }
        for _ in range(KV_LAYERS)
    ]
    pipe.crossattn_cache = [
        {"k": None, "v": None} for _ in range(KV_LAYERS)
    ]
    for k, v in attrs.items():
        setattr(pipe, k, v)
    return pipe, gen


def _run_kv(pipe, *, frames=3, flash=False, requires_grad=True,
            force_exit_step=0):
    torch.manual_seed(SEED)
    noise = torch.randn(1, frames, 4, 6, 8)
    out, _, _ = pipe.generate_chunk_with_cache(
        noise,
        current_start_frame=0,
        requires_grad=requires_grad,
        sync_exit_flags=False,
        force_exit_step=force_exit_step,
        flash_dmd_enabled=flash,
        flash_dmd_gan_t=FLASH_T,
    )
    return out


def _kv_snapshot(pipe):
    return [
        (b["k"].detach().clone(), b["v"].detach().clone())
        for b in pipe.kv_cache1
    ]


def _armed_run(**extra):
    """One armed rollout, stopped just before the post-backward hash."""
    pipe, gen = _build_kv(
        pix_finish_grad_enabled=True,
        pix_kv_commit_check_every=1,
        **extra,
    )
    _run_kv(pipe)
    return pipe, gen


# ---------------------------------------------------------------------
# 7a. OFF is byte-identical, including RNG state
# ---------------------------------------------------------------------
def test_kv_tripwire_absent_equals_zero_equals_byte_identical():
    """Flag absent == flag 0 == pre-A24: same output, same K/V, same RNG
    state, no new attribute, no new key."""
    pa, _ = _build_kv(pix_finish_grad_enabled=True)
    out_a = _run_kv(pa)
    rng_a = torch.get_rng_state().clone()
    kv_a = _kv_snapshot(pa)

    pb, _ = _build_kv(
        pix_finish_grad_enabled=True, pix_kv_commit_check_every=0)
    out_b = _run_kv(pb)
    rng_b = torch.get_rng_state().clone()
    kv_b = _kv_snapshot(pb)

    assert torch.equal(out_a, out_b)
    assert torch.equal(rng_a, rng_b), "OFF changed RNG consumption"
    for (ka, va), (kb, vb) in zip(kv_a, kv_b):
        assert torch.equal(ka, kb) and torch.equal(va, vb)

    # no bookkeeping state is even created when the flag is off
    for p in (pa, pb):
        assert not hasattr(p, "_pix_kv_commit_pending")
        assert not hasattr(p, "_pix_kv_commit_calls")
        assert not hasattr(p, "_pix_kv_commit_stats")
        assert not hasattr(p, "_pix_kv_commit_cache_tag")
        # the public hook is inert and emits nothing
        assert p.pix_kv_commit_verify() == {}
        assert not [
            k for k in p._last_extension_metrics if k.startswith("pix_kv_")
        ]


def test_kv_tripwire_on_changes_neither_values_nor_rng():
    """ON must be observationally identical too -- the fingerprint is
    RNG-free (arange/sin/cos only) and read-only."""
    pa, _ = _build_kv(pix_finish_grad_enabled=True)
    out_a = _run_kv(pa)
    rng_a = torch.get_rng_state().clone()
    kv_a = _kv_snapshot(pa)

    pb, _ = _build_kv(
        pix_finish_grad_enabled=True, pix_kv_commit_check_every=1)
    out_b = _run_kv(pb)
    rng_b = torch.get_rng_state().clone()
    kv_b = _kv_snapshot(pb)

    assert torch.equal(out_a, out_b)
    assert torch.equal(rng_a, rng_b), "arming the tripwire moved the RNG"
    for (ka, va), (kb, vb) in zip(kv_a, kv_b):
        assert torch.equal(ka, kb) and torch.equal(va, vb)


def test_kv_tripwire_trace_is_identical_on_and_off():
    """Same generator call trace: the tripwire adds NO forward."""
    pa, ga = _build_kv(pix_finish_grad_enabled=True)
    _run_kv(pa)
    pb, gb = _build_kv(
        pix_finish_grad_enabled=True, pix_kv_commit_check_every=1)
    _run_kv(pb)
    assert _trace(ga) == _trace(gb)


# ---------------------------------------------------------------------
# 7b. ON: the check runs and reports MATCH on a clean path
# ---------------------------------------------------------------------
def test_kv_tripwire_arms_and_records_one_entry_per_rollout_block():
    pipe, _ = _armed_run()
    recs = pipe._pix_kv_commit_pending
    assert recs is not None and len(recs) == 1
    rec = recs[0]
    assert rec["block_index"] == 0
    assert rec["tok_a"] == 0
    assert rec["tok_b"] == 3 * pipe.frame_seq_length
    # the commit fingerprint and the two ckpt fingerprints all exist
    assert rec["commit"] is not None
    assert rec["exit"] is not None, "exit-rung ckpt was not fingerprinted"
    assert rec["finish"] is not None, "A23 finish ckpt not fingerprinted"
    assert rec["commit"].shape == (KV_LAYERS, 2, 5)


def test_kv_tripwire_reports_match_on_a_clean_path():
    pipe, _ = _armed_run()
    rep = pipe.pix_kv_commit_verify()
    assert rep["pix_kv_commit_match"] == 1.0
    assert rep["pix_kv_commit_checked_blocks"] == 1.0
    assert rep["pix_kv_commit_mismatch_blocks"] == 0.0
    assert rep["pix_kv_commit_site_code"] == 1.0
    assert rep["pix_kv_commit_max_rel_delta"] == 0.0
    assert "pix_kv_commit_blame" not in rep
    assert "pix_kv_commit_first_mismatch_block" not in rep


def test_kv_tripwire_record_is_one_shot():
    """A second verify with nothing pending emits nothing -- it must not
    re-report a stale MATCH."""
    pipe, _ = _armed_run()
    assert pipe.pix_kv_commit_verify()["pix_kv_commit_match"] == 1.0
    assert pipe.pix_kv_commit_verify() == {}


def test_kv_tripwire_cadence_arms_every_nth_call():
    pipe, _ = _build_kv(
        pix_finish_grad_enabled=True, pix_kv_commit_check_every=3)
    armed = []
    for _ in range(6):
        _run_kv(pipe)
        armed.append(pipe._pix_kv_commit_pending is not None)
        pipe.pix_kv_commit_verify()
    assert armed == [True, False, False, True, False, False], armed


# ---------------------------------------------------------------------
# 7c. PLANTED VIOLATION -- the tripwire must FIRE
#
# Positive AND negative control from an identical setup, so the assertion
# cannot be satisfied by an identity/tautology: the ONLY difference
# between the two branches is the mutation.
# ---------------------------------------------------------------------
def test_kv_tripwire_FIRES_on_a_planted_mutation_between_the_two_hashes():
    # negative control -- same construction, nothing planted
    control, _ = _armed_run()
    control_rep = control.pix_kv_commit_verify()
    assert control_rep["pix_kv_commit_match"] == 1.0

    # positive control -- one element of one layer's V is disturbed
    pipe, _ = _armed_run()
    before = pipe.kv_cache1[1]["v"].detach().clone()
    pipe.kv_cache1[1]["v"][0, 0, 0, 0] += 1e-2
    assert not torch.equal(before, pipe.kv_cache1[1]["v"]), "plant no-op'd"

    rep = pipe.pix_kv_commit_verify()
    assert rep["pix_kv_commit_match"] == 0.0, rep
    assert rep["pix_kv_commit_mismatch_blocks"] == 1.0
    assert rep["pix_kv_commit_checked_blocks"] == 1.0
    # rollout block 0, transformer layer 1 -- the one that was mutated
    assert rep["pix_kv_commit_first_mismatch_block"] == 0.0
    assert rep["pix_kv_commit_first_mismatch_layer"] == 1.0, rep
    assert rep["pix_kv_commit_max_rel_delta"] > 1e-5

    # and the two reports differ ONLY because of the plant
    assert control_rep["pix_kv_commit_match"] != rep["pix_kv_commit_match"]


def test_kv_tripwire_FIRES_on_a_mutation_of_every_single_slot():
    """Sweep: mutating ANY (layer, k/v) slot is caught, and localised to
    that layer. Guards against a fingerprint that only looks at layer 0
    or only at ``k``."""
    seen = []
    for layer in range(KV_LAYERS):
        for name in ("k", "v"):
            pipe, _ = _armed_run()
            pipe.kv_cache1[layer][name][0, 2, 1, 3] += 5e-2
            rep = pipe.pix_kv_commit_verify()
            assert rep["pix_kv_commit_match"] == 0.0, (layer, name, rep)
            assert rep["pix_kv_commit_first_mismatch_layer"] == float(layer)
            seen.append((layer, name))
    assert len(seen) == KV_LAYERS * 2


def test_kv_tripwire_is_sensitive_to_a_small_relative_perturbation():
    """Not a gross-change-only detector: a ~1e-4 relative nudge fires."""
    pipe, _ = _armed_run()
    v = pipe.kv_cache1[0]["k"]
    v[0, 1, 0, 0] += 1e-4 * float(v[0, 1, 0, 0].abs().clamp_min(1.0))
    rep = pipe.pix_kv_commit_verify()
    assert rep["pix_kv_commit_match"] == 0.0, rep


def test_kv_tripwire_cannot_be_fooled_by_a_zeroed_slot():
    """Wiping a slot to zeros is a mismatch, not a 'match' by symmetry."""
    pipe, _ = _armed_run()
    pipe.kv_cache1[2]["k"].zero_()
    rep = pipe.pix_kv_commit_verify()
    assert rep["pix_kv_commit_match"] == 0.0, rep
    assert rep["pix_kv_commit_first_mismatch_layer"] == 2.0


# ---------------------------------------------------------------------
# 7d. ATTRIBUTION -- which recompute is to blame
#
# MEASURED, and stated here so nobody over-reads this suite: on CPU with
# these stubs ``.backward()`` does NOT re-enter the checkpointed
# functions. ``use_reentrant=False`` recomputes lazily, on the first
# UNPACK of a saved tensor, and autograd never needs one from the frame
# for arithmetic this small. So this file CANNOT prove the hazard occurs
# in production -- that is the GPU smoke's job.
#
# What it CAN prove, and does below, is that the instrument works: it
# fires on the exact K/V state a recompute would leave behind, names the
# right recompute, and its verdict agrees with ground truth after a real
# backward whichever way that backward goes.
# ---------------------------------------------------------------------
def _replay(pipe, gen, t_value, frames=3):
    """Re-run ONE recorded forward against the live cache. This is
    precisely what a backward-time recompute does: same function, same
    inputs, same K/V write -- so the resulting cache state is the state a
    recompute would leave."""
    rec = [c for c in gen.calls if c["t"] == t_value][-1]
    with torch.no_grad():
        gen(
            noisy_image_or_video=rec["in"],
            conditional_dict={},
            timestep=torch.full([1, frames], t_value, dtype=torch.int64),
            kv_cache=pipe.kv_cache1,
            crossattn_cache=pipe.crossattn_cache,
            current_start=0,
        )


def test_kv_tripwire_blames_the_FINISH_rung_recompute():
    """Replay the A23 ``_finish_fn`` forward on top of Step 3.4's commit
    -- the exact corruption this package is hunting."""
    pipe, gen = _armed_run()
    committed = _kv_snapshot(pipe)
    _replay(pipe, gen, DENOISING_STEPS[-1])
    after = _kv_snapshot(pipe)
    assert any(
        not torch.equal(a, b) for (a, _), (b, _) in zip(committed, after)
    ), "replay left the cache unchanged -- the test is not testing"

    rep = pipe.pix_kv_commit_verify()
    assert rep["pix_kv_commit_match"] == 0.0, rep
    assert rep["pix_kv_commit_blame"] == 2.0, (
        "expected blame=2 (finish-rung recompute), got %r" % (rep,))
    assert rep["pix_kv_commit_first_mismatch_block"] == 0.0


def test_kv_tripwire_blames_the_EXIT_rung_recompute():
    """Same instrument, different culprit. Proves ``blame`` discriminates
    instead of always returning one constant."""
    pipe, gen = _armed_run()
    _replay(pipe, gen, DENOISING_STEPS[0])
    rep = pipe.pix_kv_commit_verify()
    assert rep["pix_kv_commit_match"] == 0.0, rep
    assert rep["pix_kv_commit_blame"] == 1.0, (
        "expected blame=1 (exit-rung recompute), got %r" % (rep,))


def test_kv_tripwire_blames_the_FLASH_recompute():
    pipe, gen = _build_kv(
        pix_finish_grad_enabled=True, pix_kv_commit_check_every=1)
    _run_kv(pipe, flash=True)
    assert pipe._pix_kv_commit_pending[0]["flash"] is not None
    _replay(pipe, gen, FLASH_T)
    rep = pipe.pix_kv_commit_verify()
    assert rep["pix_kv_commit_match"] == 0.0, rep
    assert rep["pix_kv_commit_blame"] == 3.0, (
        "expected blame=3 (flash recompute), got %r" % (rep,))


def test_kv_tripwire_blames_UNKNOWN_for_a_state_no_recompute_produces():
    """``blame`` must not guess. A cache state that matches none of the
    three recorded forwards reads -1, not a plausible-looking 1/2/3."""
    pipe, _ = _armed_run()
    for blk in pipe.kv_cache1:
        blk["k"].add_(3.7)
    rep = pipe.pix_kv_commit_verify()
    assert rep["pix_kv_commit_match"] == 0.0
    assert rep["pix_kv_commit_blame"] == -1.0, rep


def test_kv_tripwire_verdict_agrees_with_ground_truth_after_a_real_backward():
    """The one assertion that survives whatever ``use_reentrant=False``
    decides to do: run the rollout, back-propagate through the A23 grad
    buffer for real, then require the tripwire's verdict to EQUAL a
    direct tensor-by-tensor comparison of the cache. If a future torch
    starts recomputing here, this test keeps holding and
    ``test_..._blames_the_FINISH_rung_recompute`` above says what the
    verdict will be."""
    pipe, _ = _armed_run()
    buf = pipe._clean_chunk_grad
    assert buf is not None and buf.requires_grad

    committed = _kv_snapshot(pipe)
    buf.sum().backward()
    after = _kv_snapshot(pipe)
    unchanged = all(
        torch.equal(a, b) and torch.equal(c, d)
        for (a, c), (b, d) in zip(committed, after)
    )

    rep = pipe.pix_kv_commit_verify()
    assert "pix_kv_commit_match" in rep, rep
    assert (rep["pix_kv_commit_match"] == 1.0) is unchanged, (
        "tripwire verdict %r disagrees with ground truth unchanged=%r"
        % (rep, unchanged))


# ---------------------------------------------------------------------
# 7e. OMIT-NEVER-FAKE
# ---------------------------------------------------------------------
def test_no_match_key_when_the_cache_was_reset_between_the_hashes():
    pipe, _ = _armed_run()
    pipe.reset_cache_state()          # what setup_sequence does per ride
    rep = pipe.pix_kv_commit_verify()
    assert "pix_kv_commit_match" not in rep, rep
    assert rep["pix_kv_commit_skipped"] == 1.0


def test_no_match_key_when_the_cache_was_reallocated_between_the_hashes():
    pipe, _ = _armed_run()
    pipe.kv_cache1 = [
        {"k": torch.zeros(1, KV_TOKENS, 2, 4),
         "v": torch.zeros(1, KV_TOKENS, 2, 4)}
        for _ in range(KV_LAYERS)
    ]
    rep = pipe.pix_kv_commit_verify()
    assert "pix_kv_commit_match" not in rep, rep
    assert rep["pix_kv_commit_skipped"] == 1.0


def test_no_match_key_when_there_is_nothing_to_fingerprint():
    """The legacy stub cache (k/v = None) must produce NO match key --
    'nothing was checked' may never read as 'checked and matched'."""
    pipe, _ = _build(
        pix_finish_grad_enabled=True, pix_kv_commit_check_every=1)
    _run(pipe)
    rep = pipe.pix_kv_commit_verify()
    assert "pix_kv_commit_match" not in rep, rep


def test_no_match_key_without_an_armed_run():
    pipe, _ = _build_kv(
        pix_finish_grad_enabled=True, pix_kv_commit_check_every=1)
    assert pipe.pix_kv_commit_verify() == {}


# ---------------------------------------------------------------------
# 7f. deferred (fallback) site -- labelled, never conflated
# ---------------------------------------------------------------------
def test_deferred_site_reports_under_its_own_site_code():
    pipe, _ = _armed_run()
    # no explicit verify; the NEXT rollout picks the record up
    _run_kv(pipe)
    m = pipe._last_extension_metrics
    assert m["pix_kv_commit_site_code"] == 2.0, m
    assert "pix_kv_commit_match" in m
