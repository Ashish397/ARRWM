import numpy as np

from grids.eval import panel32_seed_bundles as bundles


def test_seed_spans_are_causal_disjoint_and_end_at_boundary():
    assert bundles.SEED_SPANS == ((0, 9), (9, 21), (21, 33))
    assert bundles.SEED_SPANS[0][0] == 0
    assert bundles.SEED_SPANS[-1][1] == 33
    assert all(a[1] == b[0] for a, b in zip(bundles.SEED_SPANS, bundles.SEED_SPANS[1:]))
    assert sum(stop - start for start, stop in bundles.SEED_SPANS) == 33


def test_wan_temporal_contract():
    assert bundles.PIXEL_FRAMES == 1 + 4 * (bundles.LATENT_FRAMES - 1)
    assert bundles.LATENT_FRAMES == 3 * len(bundles.SEED_SPANS)


def test_physical_zero_is_not_assumed_numeric_zero():
    # The extractor always applies the fitted PCA centring before its squash.
    # Therefore a zero pixel-flow field generally maps away from (0, 0).
    mean = np.ones(200)
    comp_t = np.zeros((200, 16))
    comp_t[:, 0] = 1
    raw = (np.zeros((1, 200)) - mean) @ comp_t
    assert raw[0, 0] != 0
