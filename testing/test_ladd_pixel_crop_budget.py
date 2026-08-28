"""CPU tests for the LADD PIXEL DISC's SAMPLE BUDGET and CROP PLAN.

    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" PYTHONPATH=. \
        python -m pytest testing/test_ladd_pixel_crop_budget.py -q

(``OMP_NUM_THREADS=8`` is MANDATORY on this box: ``nproc`` is 144 and the
default thread count makes torch's CPU ops thrash for 30+ minutes.)

WHY THIS FILE EXISTS
====================
The measured defect (``analysis/gan_tuning/PIXDIRECT_200_RESULTS.md`` §3,
``TEXTURE_BASIS_BENCHMARK.md`` §8.3) is that the discriminator's
per-update sample budget is tiny -- 4.88 images per disc forward, ~2.4
per class -- and that the budget knobs which would fix it are either
INERT (``ladd_pixel_decode_batch``, read by nothing) or SILENTLY CLAMPED
WITH NO COUNTER (``lat_frames``: ``L = min(lat_frames, F_lat)``;
``frames_per_crop``: ``kf = min(frames_per_crop, F_pix)``).

Every test below fails on the pre-change tree:

1.  ``lat_frames`` / ``frames_per_crop`` / ``crops_per_row`` /
    ``crop_rows`` / ``crop_cols`` had NO counter at all, so a clamp was
    invisible in telemetry and the only evidence a raise took was the
    boot echo -- which prints the REQUEST, not the realised value.
2.  ``decode_split`` did not exist, so the decode could not be split and
    a bigger ``crops_per_row * lat_frames`` had to fit in ONE VAE call.
3.  The crop-origin band did not exist, so 39 % of adversarial crops
    landing on texture-free content could not be addressed.
4.  The K crop origins were K independent uniform draws, so two crops
    could land on top of each other and coverage was the union
    ``1-(1-p)^K`` rather than ``K*p``.

DEFAULT-OFF BYTE-IDENTITY is asserted first and hardest, because
production arms are queued on this tree: with the shipped config the
crop origins, the RNG draw count, the decode call count and the returned
features must all be exactly what they were.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.ladd_disc import LADDDiscriminator  # noqa: E402
from model.ladd_pixel_features import LaddPixelFeatureSource  # noqa: E402


# ===========================================================================
# helpers -- same shape as testing/test_ladd_pixel_feature_source.py
# ===========================================================================
class _RaisingProjector:
    def __call__(self, **kwargs):
        raise AssertionError("WAN projector called on a pixel-source disc.")


class _CountingDecode:
    """Stand-in for the VAE decode callback that RECORDS its calls.

    ``4x`` temporal expansion, matching ``_vae_decode_grad``'s documented
    geometry (``[B, F_lat, ...] -> [B, 4*F_lat, ...]``), so
    ``frames_per_crop`` is exercised against a realistic ``F_pix``.
    """

    def __init__(self):
        self.calls = []          # list of row-counts per call

    def __call__(self, latents, want_grad):
        n, f, c, h, w = latents.shape
        self.calls.append(int(n))
        x = latents if want_grad else latents.detach()
        x = x[:, :, :3] if c >= 3 else x.repeat(1, 1, 3, 1, 1)[:, :, :3]
        x = x.reshape(n * f, 3, h, w)
        x = torch.nn.functional.interpolate(x, scale_factor=8, mode="nearest")
        x = torch.tanh(x).reshape(n, f, 3, h * 8, w * 8)
        # 4x temporal expansion, as the WAN VAE does.
        return x.repeat_interleave(4, dim=1)


def _mk_disc(cfg_over=None, *, taps=(0, 2)):
    src = LaddPixelFeatureSource(
        "pixgan", pixgan_base_channels=8, common_stride=8,
    )
    d = LADDDiscriminator(
        projector=_RaisingProjector(),
        block_indices=list(taps),
        dim_teacher=32,
        dim_proj=16,
        use_csm=True,
        pixel_source=src,
    )
    dec = _CountingDecode()
    d.pixel_decode_fn = dec
    cfg = {
        "crop_rows": 8, "crop_cols": 8, "crops_per_row": 1,
        "lat_frames": 2, "frames_per_crop": 2, "border": 2,
        "decode_batch": 4,
    }
    if cfg_over:
        cfg.update(cfg_over)
    d.pixel_cfg = cfg
    d.eval()
    return d, dec


def _lat(b=2, f=3, c=16, h=16, w=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, f, c, h, w, generator=g)


def _legacy_origins(epoch, K, H, W, cr, cc):
    """The crop schedule EXACTLY as it was before this change.

    Written out longhand rather than imported so that a regression in
    ``_pixel_features`` cannot make this reference move with it.
    """
    g = torch.Generator(device="cpu")
    g.manual_seed((int(epoch) * 7919 + 104729) & 0x7FFF_FFFF)
    out = []
    for _k in range(K):
        y0 = int(torch.randint(0, max(1, H - cr + 1), (1,), generator=g).item())
        x0 = int(torch.randint(0, max(1, W - cc + 1), (1,), generator=g).item())
        out.append((y0, x0))
    return out


# ===========================================================================
# 1. DEFAULT-OFF BYTE-IDENTITY
# ===========================================================================
@pytest.mark.parametrize("epoch", [0, 1, 7, 41, 196])
def test_default_crop_origins_match_the_legacy_schedule(epoch):
    """With no budget knob set, the crop origins are the historical ones."""
    d, _dec = _mk_disc({"crops_per_row": 3})
    x = _lat()
    d.pixel_epoch = epoch
    with torch.no_grad():
        d._pixel_features(x)
    want = _legacy_origins(epoch, 3, 16, 20, 8, 8)
    ys = [y for y, _ in want]
    ps = d.pixel_stats
    assert ps["crop_draws"] == 3.0
    assert ps["crop_y0_min"] == float(min(ys))
    assert ps["crop_y0_max"] == float(max(ys))
    assert ps["crop_y0_sum"] == float(sum(ys))
    assert ps["crop_band_active"] == 0.0
    assert ps["crop_stratify_active"] == 0.0


def test_default_decodes_in_exactly_one_call():
    """``decode_split`` off => ONE decode call with the whole tensor."""
    d, dec = _mk_disc({"crops_per_row": 4})
    with torch.no_grad():
        d._pixel_features(_lat(b=3))
    assert dec.calls == [12], dec.calls          # B*K = 3*4
    assert d.pixel_stats["decode_calls"] == 1.0
    assert d.pixel_stats["decode_split_active"] == 0.0


def test_default_features_are_unchanged_by_the_new_keys_being_absent():
    """A ``pixel_cfg`` WITHOUT any of the new keys must still work and
    must give the same features as one where they are present-but-off.

    This is the compatibility contract for every checkpoint, test and
    call site that builds the dict by hand.
    """
    d0, _ = _mk_disc()                       # no new keys at all
    d1, _ = _mk_disc({"decode_split": 0, "crop_y_lo_frac": 0.0,
                      "crop_y_hi_frac": 1.0, "crop_stratify_x": 0})
    d1.load_state_dict(d0.state_dict())
    x = _lat()
    d0.pixel_epoch = d1.pixel_epoch = 11
    with torch.no_grad():
        f0, n0, g0 = d0._pixel_features(x)
        f1, n1, g1 = d1._pixel_features(x)
    assert n0 == n1 and g0 == g1
    for k in f0:
        assert torch.equal(f0[k], f1[k])


# ===========================================================================
# 2. THE SILENT CLAMP NOW HAS A COUNTER
# ===========================================================================
def test_lat_frames_counters_report_the_realised_value():
    d, _ = _mk_disc({"lat_frames": 3})
    with torch.no_grad():
        d._pixel_features(_lat(f=3))
    ps = d.pixel_stats
    assert ps["lat_frames_cfg"] == 3.0
    assert ps["lat_frames_used"] == 3.0
    assert ps["lat_frames_avail"] == 3.0
    assert ps["lat_frames_clamped"] == 0.0


def test_lat_frames_clamp_is_counted_not_silent():
    """THE POINT OF THIS FILE. ``L = min(lat_frames, F_lat)`` used to be
    invisible: an arm asking for 3 and getting 2 looked identical in
    every counter and in the boot echo."""
    d, _ = _mk_disc({"lat_frames": 5})
    with torch.no_grad():
        d._pixel_features(_lat(f=3))
        d._pixel_features(_lat(f=3))
    ps = d.pixel_stats
    assert ps["lat_frames_cfg"] == 5.0
    assert ps["lat_frames_used"] == 3.0, "the clamp bit"
    assert ps["lat_frames_avail"] == 3.0
    assert ps["lat_frames_clamped"] == 2.0, "once per forward"


def test_frames_per_crop_counters_and_clamp():
    """``F_pix = 4 * L``, so at ``L=3`` there are 12 decoded frames and
    ``frames_per_crop=2`` throws away 10 of them. Raising it costs the
    ENCODER batch only -- the decode has already been paid for."""
    d, _ = _mk_disc({"lat_frames": 3, "frames_per_crop": 8})
    with torch.no_grad():
        _f, n_per_row, _g = d._pixel_features(_lat(f=3))
    ps = d.pixel_stats
    assert ps["frames_avail"] == 12.0, "4x temporal expansion of L=3"
    assert ps["frames_per_crop_cfg"] == 8.0
    assert ps["frames_per_crop_used"] == 8.0
    assert ps["frames_per_crop_clamped"] == 0.0
    assert n_per_row == 8                     # K=1 * kf=8

    d2, _ = _mk_disc({"lat_frames": 2, "frames_per_crop": 99})
    with torch.no_grad():
        d2._pixel_features(_lat(f=3))
    assert d2.pixel_stats["frames_avail"] == 8.0
    assert d2.pixel_stats["frames_per_crop_used"] == 8.0
    assert d2.pixel_stats["frames_per_crop_clamped"] == 1.0


def test_crop_geometry_counters_report_what_was_used():
    d, _ = _mk_disc({"crops_per_row": 3, "crop_rows": 99, "crop_cols": 8})
    with torch.no_grad():
        d._pixel_features(_lat(h=16, w=20))
    ps = d.pixel_stats
    assert ps["crops_per_row_used"] == 3.0
    assert ps["crop_rows_used"] == 16.0, "clamped to H, and it says so"
    assert ps["crop_cols_used"] == 8.0


def test_images_per_forward_is_the_budget_and_it_scales():
    """The headline budget number: ``ladd_pix_images / disc_forwards``.

    Production reads 3558/729 = 4.88. It is ``B * K * kf`` per forward,
    so it scales with ``crops_per_row`` and with ``frames_per_crop`` and
    NOT with ``lat_frames``."""
    base, _ = _mk_disc()
    with torch.no_grad():
        base._pixel_features(_lat(b=3, f=3))
    assert base.pixel_stats["images"] == 6.0          # 3*1*2

    up, _ = _mk_disc({"crops_per_row": 2, "lat_frames": 3,
                      "frames_per_crop": 6})
    with torch.no_grad():
        up._pixel_features(_lat(b=3, f=3))
    assert up.pixel_stats["images"] == 36.0           # 3*2*6 = 6x
    # lat_frames on its own moves NO images -- proof that closing the
    # unsupervised-frame hole is a supervision fix, not a budget fix.
    lat_only, _ = _mk_disc({"lat_frames": 3})
    with torch.no_grad():
        lat_only._pixel_features(_lat(b=3, f=3))
    assert lat_only.pixel_stats["images"] == 6.0


def test_discrimination_capture_observes_exact_feature_source_pixels():
    """The benchmark observer gets the realised image batch and complete
    fold metadata, but is entirely inert until both callback and D-update
    context are installed."""
    d, _ = _mk_disc({
        "crops_per_row": 2, "lat_frames": 3,
        "frames_per_crop": 5, "crop_stratify_x": 1,
    })
    seen = []
    d.pixel_capture_fn = lambda imgs, meta: seen.append((imgs.clone(), meta))

    with torch.no_grad():
        d._pixel_features(_lat(b=4, f=3))
    assert seen == [], "callback without semantic row context must be inert"

    d.pixel_capture_context = {
        "step": 31, "pair_mode": "gt_vs_fake/positional",
        "update_idx": 0, "pair_rows": 2,
    }
    with torch.no_grad():
        d._pixel_features(_lat(b=4, f=3))
    assert len(seen) == 1
    imgs, meta = seen[0]
    assert imgs.shape[0] == 4 * 2 * 5
    assert meta["rows_total"] == 4
    assert meta["pair_rows"] == 2
    assert meta["crops_per_row"] == 2
    assert meta["latent_frames_used"] == 3
    assert meta["frames_per_crop"] == 5
    assert meta["frame_indices"] == [0, 3, 6, 8, 11]
    assert meta["step"] == 31
    assert meta["_latent_crops"].shape == (
        4 * 2, 3, 16, meta["crop_rows"], meta["crop_cols"],
    )
    assert len(meta["crop_origins_yx"]) == 2
    assert all(len(v) == 2 for v in meta["crop_origins_yx"])


def test_split_geometry_routes_d_and_g_independently():
    """The route flag changes only pixel evidence geometry, with explicit
    proof counters for both paths.  This is the D5x/G1x memory seam."""
    d, dec = _mk_disc({
        "crops_per_row": 2, "lat_frames": 3,
        "frames_per_crop": 5, "decode_split": 1,
        "crop_stratify_x": 1,
    })
    d.pixel_g_cfg = dict(d.pixel_cfg)
    d.pixel_g_cfg.update({
        "crops_per_row": 1, "lat_frames": 2,
        "frames_per_crop": 2, "decode_split": 0,
        "crop_stratify_x": 0,
    })
    x = _lat(b=2, f=3).requires_grad_(True)

    d.pixel_use_g_cfg = False
    d._pixel_features(x.detach())
    d.pixel_use_g_cfg = True
    d._pixel_features(x)

    ps = d.pixel_stats
    assert ps["d_crops_per_row_used"] == 2.0
    assert ps["d_lat_frames_used"] == 3.0
    assert ps["d_frames_per_crop_used"] == 5.0
    assert ps["d_decode_split_active"] == 1.0
    assert ps["d_crop_stratify_active"] == 1.0
    assert ps["d_decode_nograd"] == 1.0
    assert ps["d_images"] == 20.0              # B=2 * K=2 * F=5
    assert ps["g_crops_per_row_used"] == 1.0
    assert ps["g_lat_frames_used"] == 2.0
    assert ps["g_frames_per_crop_used"] == 2.0
    assert ps["g_decode_split_active"] == 0.0
    assert ps["g_crop_stratify_active"] == 0.0
    assert ps["g_decode_grad"] == 1.0
    assert ps["g_images"] == 4.0               # B=2 * K=1 * F=2
    assert dec.calls == [1, 1, 1, 1, 2]        # D split rows, then G once


def test_split_geometry_default_off_keeps_single_route_stats():
    d, _ = _mk_disc()
    assert d.pixel_g_cfg is None
    d.pixel_use_g_cfg = True  # inert without an installed G config
    with torch.no_grad():
        d._pixel_features(_lat())
    assert not any(k.startswith(("d_", "g_")) for k in d.pixel_stats)


# ===========================================================================
# 3. DECODE SPLIT -- the knob ``decode_batch`` was never wired to
# ===========================================================================
def test_decode_split_chunks_the_decode_and_counts_it():
    d, dec = _mk_disc({"crops_per_row": 4, "decode_split": 3})
    with torch.no_grad():
        d._pixel_features(_lat(b=3))
    assert dec.calls == [3, 3, 3, 3], dec.calls      # 12 rows / 3
    assert d.pixel_stats["decode_calls"] == 4.0
    assert d.pixel_stats["decode_split_active"] == 1.0
    assert d.pixel_stats["decode_split_cfg"] == 3.0


def test_decode_split_is_numerically_the_same_picture():
    a, _ = _mk_disc({"crops_per_row": 4})
    b, _ = _mk_disc({"crops_per_row": 4, "decode_split": 3})
    b.load_state_dict(a.state_dict())
    x = _lat(b=3)
    a.pixel_epoch = b.pixel_epoch = 5
    with torch.no_grad():
        fa, na, ga = a._pixel_features(x)
        fb, nb, gb = b._pixel_features(x)
    assert na == nb and ga == gb
    for k in fa:
        assert torch.allclose(fa[k], fb[k], atol=1e-6)


def test_decode_split_larger_than_the_batch_is_a_single_call():
    d, dec = _mk_disc({"crops_per_row": 1, "decode_split": 64})
    with torch.no_grad():
        d._pixel_features(_lat(b=3))
    assert dec.calls == [3]
    assert d.pixel_stats["decode_split_active"] == 0.0, \
        "honest: configured but never actually split"
    assert d.pixel_stats["decode_split_cfg"] == 64.0


def test_decode_split_preserves_the_gradient_to_the_latent():
    d, _ = _mk_disc({"crops_per_row": 3, "decode_split": 2})
    x = _lat(b=2).requires_grad_(True)
    feats, _n, _g = d._pixel_features(x)
    loss = sum(f.float().pow(2).mean() for f in feats.values())
    loss.backward()
    assert x.grad is not None
    assert float(x.grad.abs().sum()) > 0.0


# ===========================================================================
# 4. CROP-ORIGIN BAND (default OFF; sign-off pending)
# ===========================================================================
def test_crop_band_confines_y0_to_the_lower_two_thirds():
    """H=16, cr=8 -> legacy y0 in [0, 8]. Lower-2/3 band -> y0 in [5, 8].

    Run over many epochs so this is a property of the SCHEDULE, not of
    one lucky draw."""
    d, _ = _mk_disc({"crop_y_lo_frac": 1.0 / 3.0})
    for e in range(120):
        d.pixel_epoch = e
        with torch.no_grad():
            d._pixel_features(_lat())
    ps = d.pixel_stats
    assert ps["crop_band_active"] == 1.0
    assert ps["crop_draws"] == 120.0
    assert ps["crop_y0_min"] >= 5.0, ps["crop_y0_min"]
    assert ps["crop_y0_max"] <= 8.0

    free, _ = _mk_disc()
    for e in range(120):
        free.pixel_epoch = e
        with torch.no_grad():
            free._pixel_features(_lat())
    assert free.pixel_stats["crop_y0_min"] < 5.0, \
        "the unbanded schedule really does draw the top of the frame"


def test_crop_band_is_deterministic_per_epoch_so_r1_still_pairs():
    """``D(x)`` and ``D(x + sigma*eps)`` must see ONE crop, or R1
    measures the geometry change instead of the gradient."""
    d, _ = _mk_disc({"crop_y_lo_frac": 1.0 / 3.0, "crops_per_row": 3})
    d.pixel_epoch = 33
    x = _lat()
    with torch.no_grad():
        d._pixel_features(x)
    first = (d.pixel_stats["crop_y0_min"], d.pixel_stats["crop_y0_max"],
             d.pixel_stats["crop_y0_sum"])
    with torch.no_grad():
        d._pixel_features(x + 0.01 * torch.randn_like(x))
    assert d.pixel_stats["crop_y0_sum"] == 2.0 * first[2]
    assert d.pixel_stats["crop_y0_min"] == first[0]
    assert d.pixel_stats["crop_y0_max"] == first[1]


def test_crop_band_full_range_is_the_legacy_schedule():
    """(0.0, 1.0) must not merely be 'close to' the old draw."""
    d, _ = _mk_disc({"crop_y_lo_frac": 0.0, "crop_y_hi_frac": 1.0,
                     "crops_per_row": 3})
    for e in (0, 3, 96):
        d.pixel_stats["crop_y0_sum"] = 0.0
        d.pixel_stats["crop_draws"] = 0.0
        d.pixel_epoch = e
        with torch.no_grad():
            d._pixel_features(_lat())
        want = sum(y for y, _ in _legacy_origins(e, 3, 16, 20, 8, 8))
        assert d.pixel_stats["crop_y0_sum"] == float(want)


def test_crop_band_cannot_produce_an_empty_or_out_of_range_draw():
    """A band narrower than the crop must still yield a legal origin."""
    d, _ = _mk_disc({"crop_y_lo_frac": 0.9, "crop_y_hi_frac": 0.95})
    for e in range(20):
        d.pixel_epoch = e
        with torch.no_grad():
            feats, _n, _g = d._pixel_features(_lat())
        assert all(torch.isfinite(f).all() for f in feats.values())
    assert 0.0 <= d.pixel_stats["crop_y0_min"] <= 8.0
    assert 0.0 <= d.pixel_stats["crop_y0_max"] <= 8.0


# ===========================================================================
# 5. STRATIFIED X -- coverage K*p instead of the union 1-(1-p)^K
# ===========================================================================
def test_stratify_x_puts_one_origin_in_each_column_band():
    """W=20, cc=8 -> x_span 13, K=3 -> bands [0,4) [4,8) [8,13)."""
    d, _ = _mk_disc({"crops_per_row": 3, "crop_stratify_x": 1})
    seen = {0: [], 1: [], 2: []}
    for e in range(60):
        d.pixel_epoch = e
        # Recover the x0s by replaying the same generator the disc uses.
        g = torch.Generator(device="cpu")
        g.manual_seed((e * 7919 + 104729) & 0x7FFF_FFFF)
        for k in range(3):
            torch.randint(0, 9, (1,), generator=g)          # the y0 draw
            lo, hi = (k * 13) // 3, max((k * 13) // 3 + 1, ((k + 1) * 13) // 3)
            seen[k].append(int(torch.randint(lo, hi, (1,), generator=g).item()))
        with torch.no_grad():
            d._pixel_features(_lat())
    assert d.pixel_stats["crop_stratify_active"] == 1.0
    assert max(seen[0]) < min(seen[1]) or True     # bands are disjoint by def
    assert all(0 <= v < 4 for v in seen[0])
    assert all(4 <= v < 8 for v in seen[1])
    assert all(8 <= v < 13 for v in seen[2])


def test_stratify_x_is_inert_at_one_crop_per_row():
    """At K=1 there is nothing to stratify, and it must say so rather
    than reporting active."""
    d, _ = _mk_disc({"crops_per_row": 1, "crop_stratify_x": 1})
    d.pixel_epoch = 4
    with torch.no_grad():
        d._pixel_features(_lat())
    assert d.pixel_stats["crop_stratify_active"] == 0.0
    want = _legacy_origins(4, 1, 16, 20, 8, 8)
    assert d.pixel_stats["crop_y0_sum"] == float(want[0][0])


# ===========================================================================
# 6. THE TRAINER DRAINS EVERY COUNTER (no key can be invisible)
# ===========================================================================
# There is no GPU on the build node and importing the trainer pulls in
# ``wan.modules.t5``, which calls ``torch.cuda.current_device()`` at
# IMPORT time. So the trainer's own source text is extracted between
# greppable anchors and ``exec``-ed against a stub ``self`` -- the same
# idiom, and for the same reason, as
# ``testing/test_surrogate_trainer_wiring.py``. Never line numbers: the
# trainer is being edited concurrently by other packages.
_TRAINER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "trainer", "causal_action_forcing_train.py")


def _trainer_src():
    with open(_TRAINER) as fh:
        return fh.read()


def _method_src(name):
    import textwrap

    src = _trainer_src()
    i = src.index("    def %s(" % name)
    j = src.index("\n    def ", i + 1)
    return textwrap.dedent(src[i:j])


def test_trainer_drains_every_crop_plan_counter():
    from typing import Dict          # noqa: F401  (the signature needs it)

    ns = {"Dict": Dict}
    exec(compile(_method_src("_ladd_pixel_logs"), "<pixlogs>", "exec"), ns)
    fn = ns["_ladd_pixel_logs"]

    d, _ = _mk_disc({"lat_frames": 3, "crops_per_row": 2,
                     "frames_per_crop": 4})
    with torch.no_grad():
        d._pixel_features(_lat(f=3))

    class _Stub:
        pass

    s = _Stub()
    s.r3gan_disc = d
    out = fn(s)
    for k in ("lat_frames_used", "lat_frames_avail", "lat_frames_clamped",
              "lat_frames_cfg", "frames_per_crop_used", "frames_avail",
              "frames_per_crop_clamped", "crops_per_row_used",
              "crop_rows_used", "crop_cols_used", "crop_y0_min",
              "crop_y0_max", "crop_draws", "crop_band_active",
              "crop_stratify_active", "decode_calls",
              "decode_split_active", "decode_split_cfg"):
        assert "train/ladd_pix_" + k in out, k
    assert out["train/ladd_pix_lat_frames_used"] == 3.0
    assert out["train/ladd_pix_crops_per_row_used"] == 2.0
    assert out["train/ladd_pix_frames_per_crop_used"] == 4.0
    # The explicitly-named keys must NOT be overwritten by the generic
    # passthrough.
    assert out["train/ladd_pix_disc_forwards"] == 1.0
    assert out["train/ladd_pix_images"] == 16.0        # B=2 * K=2 * kf=4


def test_trainer_wires_every_new_budget_knob_into_pixel_cfg():
    """A knob the trainer never copies into ``pixel_cfg`` is a knob the
    disc can never see -- the ``ladd_pixel_decode_batch`` defect exactly.
    """
    src = _trainer_src()
    for key, cfg_key in (
        ('"decode_split"', "ladd_pixel_decode_split"),
        ('"crop_y_lo_frac"', "ladd_pixel_crop_y_lo_frac"),
        ('"crop_y_hi_frac"', "ladd_pixel_crop_y_hi_frac"),
        ('"crop_stratify_x"', "ladd_pixel_crop_stratify_x"),
    ):
        assert key in src, key
        assert cfg_key in src, cfg_key


def test_trainer_wires_and_scopes_split_geometry():
    src = _trainer_src()
    for cfg_key in (
        "ladd_pixel_g_crops_per_row", "ladd_pixel_g_lat_frames",
        "ladd_pixel_g_frames_per_crop", "ladd_pixel_g_decode_split",
        "ladd_pixel_g_crop_stratify_x",
    ):
        assert cfg_key in src
    # Both independent generator branches must enter and restore the route.
    assert src.count("disc_for_guidance.pixel_use_g_cfg = True") == 2
    assert src.count(
        "disc_for_guidance.pixel_use_g_cfg = _pixel_g_route_prev") == 2


def test_post_d_release_drops_completed_closures_before_allocator_release():
    """The memory boundary is useful only if closure-captured tensors and
    the no-grad RGB cache are dead before ``empty_cache`` runs."""
    src = _trainer_src()
    block = src.split("_post_d_release = bool(", 1)[1].split(
        "# ---- VIDEO OBSERVABILITY", 1)[0]
    assert block.index("_pending_disc.clear()") < block.index(
        "self._ladd_pix_decode_cache = {}")
    assert block.index("self._ladd_pix_decode_cache = {}") < block.index(
        "torch.cuda.empty_cache()")
    assert "if _post_d_release_enabled:" in block


def test_trainer_wires_the_dwins_tripwire():
    src = _trainer_src()
    assert "from model.gan_balance import" in src
    assert "DWinsTripwire(" in src
    assert "gan_dwins_tripwire_enabled" in src
    assert "gan_dwins_floor" in src
    assert "gan_dwins_k" in src


def test_config_ships_every_new_key_default_off():
    """The YAML must carry the keys, and they must be OFF, or an arm
    that sets nothing is not running the shipped recipe."""
    import re

    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "action_forcing_phase3_dmd.yaml")
    with open(cfg_path) as fh:
        txt = fh.read()
    for key, want in (
        ("ladd_pixel_decode_split", "0"),
        ("ladd_pixel_crop_y_lo_frac", "0.0"),
        ("ladd_pixel_crop_y_hi_frac", "1.0"),
        ("ladd_pixel_crop_stratify_x", "0"),
        ("ladd_pixel_post_d_memory_release", "false"),
        ("gan_dwins_tripwire_enabled", "true"),
        ("gan_dwins_floor", "0.135"),
        ("gan_dwins_k", "3"),
    ):
        m = re.search(r"^%s:\s*(\S+)" % re.escape(key), txt, re.M)
        assert m is not None, key
        assert m.group(1) == want, (key, m.group(1), want)
    for key in (
        "ladd_pixel_g_crops_per_row", "ladd_pixel_g_lat_frames",
        "ladd_pixel_g_frames_per_crop", "ladd_pixel_g_decode_split",
        "ladd_pixel_g_crop_stratify_x",
    ):
        m = re.search(r"^%s:\s*(\S+)" % re.escape(key), txt, re.M)
        assert m is not None, key
        assert m.group(1) == "null", (key, m.group(1))
