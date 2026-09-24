import pandas as pd

from grids.eval.panel32_render_six_axis_overall import (
    BAND_METRICS,
    ENDPOINTS,
    summarize_group,
)


def test_six_axis_summary_persists_event_onsets_and_clean_union():
    assert BAND_METRICS == (
        "control",
        "style",
        "geometry",
        "hf",
        "conjuration",
        "relocation",
    )
    actions = ("F", "FR", "R", "BR", "B", "BL", "L", "FL", "N")
    rollouts = [
        (f"context-{index:03d}", action)
        for index in range(32)
        for action in actions
    ]
    rows = []
    for context_id, action in rollouts:
        for metric in ("control", "style", "geometry", "hf"):
            for endpoint in ENDPOINTS:
                rows.append({
                    "model_id": "model",
                    "context_id": context_id,
                    "action": action,
                    "metric": metric,
                    "window_end_s": endpoint,
                    "failed": 0,
                })
        rows.extend([
            {
                "model_id": "model",
                "context_id": context_id,
                "action": action,
                "metric": "conjuration",
                "window_end_s": 6,
                "failed": int((context_id, action) == ("context-000", "F")),
            },
            {
                "model_id": "model",
                "context_id": context_id,
                "action": action,
                "metric": "relocation",
                "window_end_s": 6,
                "failed": int((context_id, action) == ("context-000", "FR")),
            },
        ])

    result = summarize_group(
        "main",
        pd.DataFrame(rows),
        [{"model_id": "model", "label": "Model"}],
        {"model": [("context-000", "BR", 20.0)]},
        {"model": [("context-000", "R", 10.0)]},
    ).set_index("endpoint_s")

    one = 100.0 / 288
    assert result.loc[6, "conjuration_rate_pct"] == one
    assert result.loc[12, "conjuration_rate_pct"] == 2 * one
    assert result.loc[18, "relocation_rate_pct"] == one
    assert result.loc[24, "relocation_rate_pct"] == 2 * one
    assert result.loc[6, "clean_all_six_count"] == 286
    assert result.loc[12, "clean_all_six_count"] == 285
    assert result.loc[24, "clean_all_six_count"] == 284
