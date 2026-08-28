import numpy as np

from analysis.gan_tuning.paired_decoder_pullback_benchmark import (
    _architecture,
    _paired_items,
)


class Record:
    def __init__(self, ride, rank, step):
        self.meta = {"ride": ride, "rank": rank, "step": step}


def test_paired_items_keep_roles_and_rows_without_crossing():
    records = [Record("train-a", 2, 1), Record("test-a", 5, 1)]
    split = {"surrogate_train": ["train-a"], "surrogate_test": ["test-a"]}
    items = _paired_items(records, split, (0, 1, 2), crop=0)
    assert [item[0] for item in items] == list(range(6))
    assert [item[2] for item in items] == [0, 1, 2, 0, 1, 2]
    assert [item[3] for item in items] == ["train"] * 3 + ["test"] * 3


def test_high_bandwidth_stage_capacity_contract():
    standard, _ = _architecture("standard96")
    high, z_width = _architecture("highbandwidth384")
    assert standard == (24, 48, 96)
    assert high == (24, 192, 384)
    assert z_width == 192
    # Relative element capacity after strides 8, 64 and 256.
    np.testing.assert_allclose([high[0] / 24, high[1] / 192, high[2] / 768], [1, 1, .5])
