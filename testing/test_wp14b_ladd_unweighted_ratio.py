"""WP-14B — the LADD unweighted-ratio call site (``ladd_unweighted_ratio``).

Scope of the chunk under test (nothing else):
  * ``ladd_unweighted_ratio`` — the five guards (pixel-folded, multi-mode,
    stat-sideband active, weight unavailable/non-positive, no-grad) each
    OMIT rather than zero-fill on failure, with a distinct reason key;
  * the division identity the whole approach depends on:
    ``‖∇(total_weight * g_rp)‖ / total_weight == ‖∇g_rp‖`` to float32
    precision, verified against REAL autograd on a small network, not
    asserted from algebra alone;
  * the two stash-and-reset sites in ``_ladd_run_pair_mode`` /
    ``_compute_ladd_losses`` are exercised only implicitly (this test
    calls the pure function directly, matching how PIXGAN's own T3-C
    suite tests ``grad_at`` — see ``testing/test_pixgan_t3c_gterm.py``);
    the stash sites themselves are read-through-``getattr`` one-liners
    with no branching of their own, so a source-text check below pins
    their shape instead of re-running the 2900-line function they sit in.

CPU-ONLY. Imports the real trainer module (not a replica) using the same
CUDA-import stub ``testing/test_pixgan_t3c_gterm.py`` already established
for this exact file.

Run:
    OMP_NUM_THREADS=4 python testing/test_wp14b_ladd_unweighted_ratio.py
or  OMP_NUM_THREADS=4 python -m pytest testing/test_wp14b_ladd_unweighted_ratio.py -q
"""
import os
import sys
from unittest.mock import patch

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Wan's T5 wrapper evaluates ``torch.cuda.current_device()`` at import time
# (idiom shared with testing/test_pixgan_t3c_gterm.py).
with patch.object(torch.cuda, "current_device", return_value=0):
    from trainer import causal_action_forcing_train as CAFT

grad_at = CAFT.grad_at
ladd_unweighted_ratio = CAFT.ladd_unweighted_ratio

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_AF_SRC_PATH = os.path.join(_ROOT, "trainer", "causal_action_forcing_train.py")


def _af_source():
    with open(_AF_SRC_PATH, encoding="utf-8") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# 1. The division identity itself, on real autograd, not algebra alone.
# ---------------------------------------------------------------------------
def _toy_grad_pair(total_weight, seed=0):
    """A tiny network producing (weighted_loss, raw_loss, dmd_loss, param).

    ``weighted_loss`` is literally ``total_weight * raw_loss`` — mirroring
    ``generator_gan_loss = gen_gan_weight * g_rp`` in the real code — so
    the test measures the SAME linearity the trainer leans on, not a
    hand-picked example that happens to be linear.
    """
    g = torch.Generator().manual_seed(seed)
    p = torch.randn(6, 5, generator=g, requires_grad=True)
    x = torch.randn(8, 6, generator=g)
    y = torch.randn(8, 6, generator=g)
    raw_loss = (x @ p).pow(2).sum()
    weighted_loss = total_weight * raw_loss
    dmd_loss = (y @ p).abs().sum()  # independent, non-GAN "generator_loss"
    return weighted_loss, raw_loss, dmd_loss, p


def test_division_identity_matches_independent_autograd():
    """‖∇(w*raw)‖ / w == ‖∇raw‖ to float32 precision, for several w."""
    for w in (1.0, 0.03, 7.5, 1e-4):
        weighted_loss, raw_loss, _dmd, p = _toy_grad_pair(w, seed=1)
        gg = grad_at(weighted_loss, p, retain_graph=True)
        ref = grad_at(raw_loss, p, retain_graph=True)
        assert gg is not None and ref is not None
        recovered = float(gg.norm()) / w
        assert abs(recovered - float(ref.norm())) <= 1e-4 * max(
            1.0, float(ref.norm())
        ), (w, recovered, float(ref.norm()))


def test_full_call_site_recovers_the_unweighted_norm_and_ratio():
    """End-to-end through ``ladd_unweighted_ratio`` itself, all guards open."""
    w = 0.03
    weighted_loss, raw_loss, dmd_loss, p = _toy_grad_pair(w, seed=2)
    gg = grad_at(weighted_loss, p, retain_graph=True)
    gr = grad_at(dmd_loss, p, retain_graph=True)
    ref_raw_norm = float(grad_at(raw_loss, p, retain_graph=True).norm())

    out = ladd_unweighted_ratio(
        gg=gg, gr=gr, mode_count=1, total_weight=w, stat_value=0.0,
        pix_folded=False,
    )
    assert "train/ladd_gan_grad_unweighted_unavailable" not in out
    assert abs(out["train/ladd_gan_grad_norm_unweighted"] - ref_raw_norm) <= (
        1e-4 * max(1.0, ref_raw_norm)
    )
    expected_ratio = ref_raw_norm / float(gr.norm())
    assert abs(
        out["train/ladd_gan_grad_ratio_unweighted"] - expected_ratio
    ) <= 1e-4 * max(1.0, expected_ratio)


def test_cosine_is_scale_invariant_so_no_separate_unweighted_cosine_needed():
    """The claim the docstring makes: cos(gg, gr) == cos(raw, gr) for w>0.

    This is the justification for NOT computing a second cosine inside
    ``ladd_unweighted_ratio`` — verified here, not just asserted in prose.
    """
    w = 4.2
    weighted_loss, raw_loss, dmd_loss, p = _toy_grad_pair(w, seed=3)
    gg = grad_at(weighted_loss, p, retain_graph=True)
    raw = grad_at(raw_loss, p, retain_graph=True)
    gr = grad_at(dmd_loss, p, retain_graph=True)
    cos_weighted = float(
        torch.dot(gg, gr) / (gg.norm() * gr.norm())
    )
    cos_raw = float(
        torch.dot(raw, gr) / (raw.norm() * gr.norm())
    )
    assert abs(cos_weighted - cos_raw) < 1e-5


# ---------------------------------------------------------------------------
# 2. Every guard OMITS (never zero-fills) on failure, with a distinct key.
# ---------------------------------------------------------------------------
def _valid_inputs():
    w = 1.0
    weighted_loss, _raw, dmd_loss, p = _toy_grad_pair(w, seed=4)
    gg = grad_at(weighted_loss, p, retain_graph=True)
    gr = grad_at(dmd_loss, p, retain_graph=True)
    return dict(gg=gg, gr=gr, mode_count=1, total_weight=w, stat_value=0.0,
                pix_folded=False)


def test_guard_pix_folded():
    kw = _valid_inputs(); kw["pix_folded"] = True
    out = ladd_unweighted_ratio(**kw)
    assert out["train/ladd_gan_grad_unweighted_unavailable"] == 1.0
    assert out["train/ladd_gan_grad_unweighted_reason_pix_folded"] == 1.0
    assert "train/ladd_gan_grad_ratio_unweighted" not in out
    assert "train/ladd_gan_grad_norm_unweighted" not in out


def test_guard_multimode():
    for count in (0, 2, 3, None):
        kw = _valid_inputs(); kw["mode_count"] = count
        out = ladd_unweighted_ratio(**kw)
        assert out["train/ladd_gan_grad_unweighted_unavailable"] == 1.0
        assert out["train/ladd_gan_grad_unweighted_reason_multimode"] == float(
            count or 0)
        assert "train/ladd_gan_grad_ratio_unweighted" not in out


def test_guard_stat_active_including_none():
    for bad_stat in (0.5, -0.001, None):
        kw = _valid_inputs(); kw["stat_value"] = bad_stat
        out = ladd_unweighted_ratio(**kw)
        assert out["train/ladd_gan_grad_unweighted_unavailable"] == 1.0
        assert out["train/ladd_gan_grad_unweighted_reason_stat_active"] == 1.0
        assert "train/ladd_gan_grad_ratio_unweighted" not in out
    # Exactly 0.0 must NOT trip this guard.
    kw = _valid_inputs(); kw["stat_value"] = 0.0
    out = ladd_unweighted_ratio(**kw)
    assert "train/ladd_gan_grad_unweighted_reason_stat_active" not in out


def test_guard_weight_unavailable_or_nonpositive():
    for bad_w in (None, 0.0, -1.0):
        kw = _valid_inputs(); kw["total_weight"] = bad_w
        out = ladd_unweighted_ratio(**kw)
        assert out["train/ladd_gan_grad_unweighted_unavailable"] == 1.0
        assert out[
            "train/ladd_gan_grad_unweighted_reason_weight_unavailable"
        ] == 1.0
        assert "train/ladd_gan_grad_ratio_unweighted" not in out


def test_guard_no_grad():
    for missing in ("gg", "gr"):
        kw = _valid_inputs(); kw[missing] = None
        out = ladd_unweighted_ratio(**kw)
        assert out["train/ladd_gan_grad_unweighted_unavailable"] == 1.0
        assert out["train/ladd_gan_grad_unweighted_reason_no_grad"] == 1.0
        assert "train/ladd_gan_grad_ratio_unweighted" not in out


def test_guard_dmd_denom_zero_is_omitted_not_zero_filled():
    """The forgeable-zero trap this whole discipline exists to avoid:
    a zero DMD denominator must OMIT the ratio, never emit ratio=0.0
    (which would read as "the LADD term contributes nothing")."""
    w = 1.0
    weighted_loss, _raw, _dmd, p = _toy_grad_pair(w, seed=5)
    gg = grad_at(weighted_loss, p, retain_graph=True)
    gr = torch.zeros_like(gg)  # a genuine zero DMD gradient
    out = ladd_unweighted_ratio(
        gg=gg, gr=gr, mode_count=1, total_weight=w, stat_value=0.0,
        pix_folded=False,
    )
    # The unweighted NORM is still knowable (it doesn't need gr) --
    # only the RATIO (which divides by ‖gr‖) must be omitted.
    assert "train/ladd_gan_grad_norm_unweighted" in out
    assert "train/ladd_gan_grad_ratio_unweighted" not in out
    assert out["train/ladd_gan_grad_unweighted_unavailable"] == 1.0
    assert out["train/ladd_gan_grad_unweighted_reason_dmd_denom_zero"] == 1.0


# ---------------------------------------------------------------------------
# 3. Source-text pins on the two stash sites (no branching to unit-test,
#    so a text check is the appropriate level -- mirrors how
#    test_pixgan_t3c_gterm.py pins config-block shape by source text).
# ---------------------------------------------------------------------------
def test_stash_reset_precedes_the_mode_loop_in_compute_ladd_losses():
    src = _af_source()
    reset_at = src.index("self._ladd_last_gen_gan_weight = None")
    loop_at = src.index('for mode_name, real_src_t, suffix in enabled_modes:')
    assert reset_at < loop_at, (
        "the per-call reset must run BEFORE the mode loop, or a mode that "
        "never reaches the weighted branch would read a stale prior value"
    )


def test_both_ladd_run_pair_mode_return_sites_stash_before_returning():
    src = _af_source()
    n_stash = src.count("self._ladd_last_gen_gan_weight = float(gen_gan_weight)")
    assert n_stash == 2, (
        "expected exactly one stash at each of _ladd_run_pair_mode's two "
        f"return sites (matched + positional); found {n_stash}"
    )


def test_call_site_reads_getattr_never_touches_config_directly():
    """The whole point of this design: the call site must read the LIVE
    stash via getattr, never re-derive total_weight/stat/mode_count from
    ``self.config``/``self.model`` -- that re-derivation is exactly the
    class of bug (two echoes, one reading config, one reading runtime,
    disagreeing) this session's incidents warned against."""
    src = _af_source()
    call_at = src.index("out.update(ladd_unweighted_ratio(")
    window = src[call_at:call_at + 700]
    for attr in (
        "_ladd_last_mode_count", "_ladd_last_total_weight",
        "_ladd_last_gen_gan_stat_value",
    ):
        assert f'getattr(\n                                self, "{attr}"' in window \
            or f'getattr(self, "{attr}"' in window, attr


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
