"""The paper's quality columns, recomputed from the shipped result files.

Each test states the exact recipe — source file, rule, population — and asserts
it reproduces the published column. That pins the provenance: if a future edit
swaps in a different artefact or threshold, these fail rather than quietly
shifting a reported number.

Two of the five columns are pinned so far. The rest are still being traced; see
docs/DECISIONS.md.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

QUALITY = Path(__file__).resolve().parents[1] / "evaluation" / "quality"

# Directory names carry the run key; the paper uses display names.
ALIAS = {"ours_pca8": "pca8", "ours_16node": "16node", "ours_4node": "4node",
         "ours_pca4": "pca4", "ours_pca2": "pca2",
         "ours_noadaln": "noadaln", "ours_noatok": "noatok"}


def _csv(name):
    path = QUALITY / name
    if not path.exists():
        pytest.skip(f"{name} not shipped")
    df = pd.read_csv(path)
    if "model" in df:
        df["m"] = df.model.replace(ALIAS)
    return df


@pytest.fixture(scope="module")
def active():
    """Feature-valid, non-static rollouts: the population three columns report on."""
    mask = _csv("canonical_static_mask.csv")
    return mask[(mask.feature_valid == True) & (mask.active == 1)][["m", "scene"]]  # noqa: E712


def test_geometric_corruption_matches_the_paper():
    """Geometry % = fraction with p_uncanny > 0.5, over all 256 directional rollouts.

    Reported over the full fleet rather than the active population, because the
    probe needs no feature-rich reference.
    """
    published = {"worldplay": 80, "matrixgame": 71, "worldcam": 68, "astra": 52,
                 "yume": 9, "minwm": 1, "pca8": 17, "16node": 26}
    vlm = _csv("results_external_vlm.csv")
    for model, want in published.items():
        rows = vlm[vlm.m == model]
        assert len(rows) == 256, f"{model}: {len(rows)} rollouts, expected 256"
        got = round(100 * (rows.p_uncanny > 0.5).mean())
        assert got == want, f"geometry {model}: recomputed {got}%, paper says {want}%"


def test_scene_relocation_matches_the_paper(active):
    """Relocation % = fraction with consensus_inl < 50, over the active population.

    Sibling consensus rather than a single reference: each rollout's score is its
    best inlier count against any other model's end frame plus the real reference,
    so a scene everybody leaves is not scored as everybody relocating.

    The cut of 50 is the deployed threshold, carried in the reel-generation code
    ("deployed relocated threshold"), not a value chosen to fit this table.
    """
    published = {"worldplay": 77, "matrixgame": 95, "worldcam": 88, "astra": 67,
                 "yume": 58, "minwm": 4, "pca8": 11, "16node": 9}
    joined = _csv("fleet_scene_consensus.csv").merge(active, on=["m", "scene"])
    for model, want in published.items():
        rows = joined[joined.m == model]
        assert len(rows), f"{model}: no rows in the active population"
        got = round(100 * (rows.consensus_inl < 50).mean())
        assert got == want, f"relocation {model}: recomputed {got}%, paper says {want}%"


def test_the_shipped_artefacts_are_one_generation():
    """All four result files must describe the same fleet.

    The superseded relocation file held rows computed against videos that had
    since been re-rendered. Coverage is the cheap check that catches a mismatched
    vintage being reintroduced.
    """
    for name, expected in [("results_external_vlm.csv", 3328),
                           ("fleet_scene_consensus.csv", 3328)]:
        assert len(_csv(name)) == expected, f"{name} does not cover the full 256x13 fleet"


def test_active_population_is_smaller_than_the_fleet(active):
    """Near-static and wet-lens rollouts are excluded, so active < 3328."""
    assert 2500 < len(active) < 3328


def test_control_failure_matches_the_paper(active):
    """Control fail % = near-static OR realised motion >90 deg from the command.

    Commands are unit-norm 0.5 in (throttle, yaw); a rollout fails when the dot
    product of commanded and realised motion is non-positive, or when the static
    mask marks it near-static. Reported over the feature-valid population.

    Freezing and steering the wrong way are both control failures, which is why
    minWM scores 22% despite following directions well when it moves at all.
    """
    import glob
    import numpy as np

    published = {"worldplay": 1, "matrixgame": 0, "worldcam": 13, "astra": 19,
                 "yume": 5, "minwm": 22, "pca8_8node": 0, "16node": 1}
    s = 0.5
    r = s / np.sqrt(2)
    commands = {"F": (s, 0), "B": (-s, 0), "R": (0, s), "L": (0, -s),
                "FR": (r, r), "FL": (r, -r), "BR": (-r, r), "BL": (-r, -r)}

    import os
    hh = os.environ.get("AF_HEADTOHEAD_DIR",
                        str(QUALITY.parent.parent.parent / "grids" / "eval" / "headtohead"))
    motion_files = sorted(glob.glob(os.path.join(hh, "headtohead_*.csv")))
    if not motion_files:
        pytest.skip("head-to-head motion readouts not present")
    motion = pd.concat([pd.read_csv(f) for f in motion_files], ignore_index=True)

    mask = _csv("canonical_static_mask.csv")
    key = lambda m, sc: (m.replace("ours_", ""), sc)  # noqa: E731
    static = {key(m, sc): bool(v) for m, sc, v in
              zip(mask.model, mask.scene, mask.static)}
    valid = {key(m, sc): bool(v) for m, sc, v in
             zip(mask.model, mask.scene, mask.feature_valid)}

    for model, want in published.items():
        rows = motion[motion.model == model]
        if not len(rows):
            continue
        name = "pca8" if model == "pca8_8node" else model
        failed = total = 0
        for t in rows.itertuples():
            scene = f"r{int(t.window):02d}_{t.dir}"
            if not valid.get(key(name, scene), True):
                continue
            total += 1
            cx, cy = commands[t.dir]
            if static.get(key(name, scene), False) or cx * t.g0 + cy * t.g1 <= 0:
                failed += 1
        got = round(100 * failed / max(total, 1))
        assert got == want, f"control fail {model}: recomputed {got}%, paper says {want}%"
