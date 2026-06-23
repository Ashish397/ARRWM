"""Regression test for the ``seed_first`` self-seeding VAE decode
(utils/wan_wrapper.py:WanVAEWrapper.decode_to_pixel).

Bug it guards: ``use_cache=True`` routes to the WAN VAE ``cached_decode``,
which never clears the temporal-conv ``feat_map``, so an INDEPENDENT clip's
first frame is convolved with the previous (unrelated) decode's leftover
cache -> a ghost of another frame in frame 0. ``seed_first`` must instead use
the cache-CLEARING ``decode`` path AND prepend a replica of the clip's own
first latent (consumed as the WAN special-first 1-pixel frame, then sliced
off), so frame 0 is faithful.

These tests mock the VAE model with the real WAN temporal geometry
(t latents, cleared cache -> 1 + 4*(t-1) pixel frames; first latent special)
so the prepend+slice frame accounting is locked in, and assert the
clearing ``decode`` path is used (not the leaky ``cached_decode``).

Run: python -m pytest testing/test_vae_seed_first_decode.py -q
"""
import torch

try:
    # NOTE: importing the wrapper pulls in wan.modules.t5, which calls
    # torch.cuda.current_device() at class-definition time -> requires a GPU.
    # On a CPU/login node this raises; skip there (run on a GPU node / CI).
    from utils.wan_wrapper import WanVAEWrapper
    _IMPORT_OK = True
except Exception as _exc:  # pragma: no cover - environment dependent
    _IMPORT_OK = False
    _IMPORT_ERR = _exc

import pytest

pytestmark = pytest.mark.skipif(
    not _IMPORT_OK,
    reason="WanVAEWrapper import needs a GPU (wan.modules.t5 CUDA-at-import)",
)


class _FakeVAEModel:
    """Mimics the WAN VAE's per-call decode geometry + cache bookkeeping.

    decode():        clears the cache (start), emits 1 + 4*(t-1) pixel frames
                     (the first latent is the "special first" -> 1 frame),
                     marks ``cleared``.
    cached_decode(): does NOT clear; emits 4*t (warm cache -> first latent is
                     no longer special). Records that it was used.
    """

    def __init__(self):
        self.cleared = False
        self.decode_calls = 0
        self.cached_calls = 0

    def clear_cache(self):
        self.cleared = True

    def _emit(self, z, n_pix):
        # z: [1, C, t, h, w] -> [1, 3, n_pix, h, w]
        b, _c, _t, h, w = z.shape
        return torch.arange(n_pix, dtype=torch.float32).view(1, 1, n_pix, 1, 1).expand(
            b, 3, n_pix, h, w
        ).clone()

    def decode(self, z, scale):
        self.clear_cache()           # plain decode brackets with clear_cache
        self.decode_calls += 1
        t = z.shape[2]
        return self._emit(z, 1 + 4 * (t - 1))

    def cached_decode(self, z, scale):
        self.cached_calls += 1       # NO clear -> leaky
        t = z.shape[2]
        return self._emit(z, 4 * t)


def _make_wrapper():
    w = WanVAEWrapper.__new__(WanVAEWrapper)  # skip heavy __init__ (weights)
    w.model = _FakeVAEModel()
    w.mean = torch.zeros(16)
    w.std = torch.ones(16)
    return w


def _latent(F_lat, C=16, H=8, W=8):
    # decode_to_pixel expects [B, F_lat, C, H, W]
    return torch.randn(1, F_lat, C, H, W)


def test_seed_first_uses_clearing_decode_not_cached():
    w = _make_wrapper()
    _ = w.decode_to_pixel(_latent(5), seed_first=True)
    assert w.model.decode_calls == 1, "seed_first must use the clearing decode()"
    assert w.model.cached_calls == 0, "seed_first must NOT use leaky cached_decode()"
    assert w.model.cleared is True, "seed_first must clear the temporal cache"


def test_seed_first_frame_count_is_uniform_4x():
    # With the dummy prepended (t -> t+1), decode emits 1 + 4*t; slicing the
    # dummy's 1 special-first frame leaves exactly 4*F_lat pixel frames.
    w = _make_wrapper()
    for F_lat in (1, 3, 7, 21):
        out = w.decode_to_pixel(_latent(F_lat), seed_first=True)
        assert out.shape[1] == 4 * F_lat, (
            f"F_lat={F_lat}: expected {4*F_lat} pixel frames, got {out.shape[1]}"
        )
        assert out.shape[2] == 3 and out.shape[0] == 1


def test_seed_first_overrides_use_cache():
    w = _make_wrapper()
    _ = w.decode_to_pixel(_latent(4), use_cache=True, seed_first=True)
    assert w.model.cached_calls == 0, "seed_first must override use_cache"
    assert w.model.decode_calls == 1


def test_plain_use_cache_still_routes_to_cached_decode():
    # Guard the other direction: genuine streaming continuation still uses the
    # warm cache path (unchanged behavior).
    w = _make_wrapper()
    _ = w.decode_to_pixel(_latent(4), use_cache=True)
    assert w.model.cached_calls == 1
    assert w.model.decode_calls == 0


if __name__ == "__main__":
    if not _IMPORT_OK:
        raise SystemExit(
            f"SKIP: cannot import WanVAEWrapper on this node ({_IMPORT_ERR}). "
            "Run on a GPU node."
        )
    test_seed_first_uses_clearing_decode_not_cached()
    test_seed_first_frame_count_is_uniform_4x()
    test_seed_first_overrides_use_cache()
    test_plain_use_cache_still_routes_to_cached_decode()
    print("All seed_first decode tests passed.")
