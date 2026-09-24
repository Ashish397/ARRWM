from grids.eval.panel32_dispatch_eval import remaining_seconds


def test_remaining_seconds_parses_slurm_formats():
    assert remaining_seconds("02:03") == 123
    assert remaining_seconds("1:02:03") == 3723
    assert remaining_seconds("2-01:02:03") == 176523


def test_remaining_seconds_rejects_non_times():
    assert remaining_seconds("N/A") == 0
    assert remaining_seconds("") == 0
