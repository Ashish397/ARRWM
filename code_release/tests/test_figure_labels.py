"""The figure legends must match the submitted figures, verbatim.

A released figure script has one job: reproduce the figure in the paper. These
strings were read off the submitted PNGs, so consolidating the wording — which
is tempting, the paper is inconsistent — breaks that job silently. If the
figures are re-rendered for camera-ready, update this file and the paper in the
same change.
"""
from __future__ import annotations

from figures.figure_labels import label


def test_wedge_legends():
    """wedge_all_g*.png: "Ours (...)" for our runs, cased names for baselines."""
    for run, want in [("pca8", "Ours (Default)"), ("16node", "Ours (batch 64)")]:
        assert label(run, ours=True) == want
    for run, want in [("minwm", "minWM"), ("matrixgame", "Matrix-Game"),
                      ("worldcam", "WorldCam"), ("yume", "Yume"),
                      ("worldplay", "WorldPlay"), ("astra", "Astra")]:
        assert label(run) == want


def test_response_curve_legends():
    """response_curves_eval_*.png: lowercase, "ours X" with no brackets."""
    want = {"pca8": "ours Default", "16node": "ours batch size 64",
            "minwm": "minwm", "matrixgame": "matrixgame", "worldcam": "worldcam",
            "yume": "yume", "worldplay": "worldplay", "astra": "astra"}
    for run, s in want.items():
        assert label(run, style="curves") == s


def test_following_family_legends():
    """following_FAMILY_*.png: each family names its runs by what it varies."""
    assert label("4node", style="following_nodes") == "batch size 16"
    assert label("pca8", style="following_nodes") == "batch size 32"
    assert label("16node", style="following_nodes") == "batch size 64"

    assert label("pca8", style="following_encoders") == "pca8"

    assert label("pca8", style="following_injection") == "pca8 (batch size 32, adaln+tokens)"
    assert label("noatok", style="following_injection") == "no action tokens (adaln-only)"
    assert label("noadaln", style="following_injection") == "no AdaLN (tokens-only)"


def test_the_default_run_has_three_names_and_that_is_intended():
    """The same run is "Default", "batch size 32" and "pca8" across families.

    Pinned deliberately: it is the paper's inconsistency, and a future reader
    who "fixes" it here would change three published figures.
    """
    names = {label("pca8", ours=True), label("pca8", style="following_nodes"),
             label("pca8", style="following_encoders")}
    assert names == {"Ours (Default)", "batch size 32", "pca8"}


def test_unknown_runs_fall_through_to_their_key():
    assert label("some_new_ablation") == "some_new_ablation"
    assert label("nocritic", style="curves") == "No Critic"  # style falls back
