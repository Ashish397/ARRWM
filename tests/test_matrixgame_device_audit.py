from copy import deepcopy

from grids.eval.matrixgame_device_audit import ACTIONS, SAMPLE_SECONDS, compare


def _fleet() -> dict:
    rows = []
    for action in ACTIONS:
        samples = [
            {"second": second, "black_screen": False, "grey_mean": 80.0}
            for second in SAMPLE_SECONDS
        ]
        rows.append({
            "context_id": "scene", "action": action, "samples": samples,
        })
    return {"status": "pass", "contexts": ["scene"], "rows": rows}


def test_compare_requires_all_per_second_black_labels_to_match() -> None:
    left = _fleet()
    right = deepcopy(left)
    result = compare(left, right)
    assert result["status"] == "pass"
    assert result["agreement_fraction"] == 1.0
    assert result["mismatches"] == []

    right["rows"][0]["samples"][6]["black_screen"] = True
    right["rows"][0]["samples"][6]["grey_mean"] = 0.0
    result = compare(left, right)
    assert result["status"] == "fail"
    assert result["agreement_fraction"] < 1.0
    assert result["mismatches"] == [{
        "context_id": "scene",
        "action": "F",
        "second": 6,
        "left_black": False,
        "right_black": True,
        "left_mean": 80.0,
        "right_mean": 0.0,
    }]

