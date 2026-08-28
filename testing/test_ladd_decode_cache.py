"""Regression tests for allocator-safe LADD detached-decode caching."""

from pathlib import Path

import torch

from model.ladd_decode_cache import lookup_decode, store_decode


def test_cache_hits_only_the_exact_unmodified_tensor_object():
    cache = {}
    source = torch.zeros(2, 3)
    decoded = torch.tensor([10.0])
    store_decode(cache, source, 7, decoded)

    assert lookup_decode(cache, source, 7) is decoded
    assert lookup_decode(cache, source, 8) is None
    assert lookup_decode(cache, source.clone(), 7) is None


def test_in_place_mutation_cannot_return_a_stale_decode():
    cache = {}
    source = torch.zeros(2, 3)
    store_decode(cache, source, 4, torch.tensor([0.0]))

    source.fill_(1.0)
    assert lookup_decode(cache, source, 4) is None


def test_cache_strongly_retains_source_identity_against_allocator_reuse():
    cache = {}
    first = torch.zeros(4)
    store_decode(cache, first, 1, torch.tensor([3.0]))

    # A distinct temporary is always a miss even when geometry and epoch are
    # identical. Keeping ``first`` in the cache entry prevents its Python id
    # from being recycled while the entry is live.
    second = torch.ones(4)
    assert lookup_decode(cache, second, 1) is None
    assert next(iter(cache.values()))[0] is first


def test_trainer_exposes_unambiguous_cache_enabled_telemetry():
    trainer = (
        Path(__file__).parents[1] / "trainer" /
        "causal_action_forcing_train.py"
    ).read_text()
    assert 'getattr(cfg, "ladd_pixel_decode_cache", False)' in trainer
    assert '"train/ladd_pix_decode_cache_enabled"' in trainer
