"""Display names for the trained runs, shared by every figure script.

Each figure script grew its own label dict, so the same ablation could appear as
"Default", "Ours (Default)", "ours Default" and "batch size 32" across four
figures in one paper. This module collects them in one place.

**It does not unify them.** The strings below are exactly what the submitted
figures render, inconsistencies included, because a released figure script has
one job: reproduce the figure in the paper. Consolidating the wording is a
camera-ready decision — if the figures are re-rendered for camera-ready, change
``STYLES`` here and the paper together, not one without the other.

Keys cover the directory-name variants the scripts inherited: the default model
is written ``pca8``, ``pca8_8node`` and ``8node8pca`` in different places, and
all three denote the same run.
"""

# run key -> display name, in the "Ours (...)" style the wedge figures use.
# Batch sizes are the effective global batch (nodes x 4).
NAMES = {
    "pca8": "Default",
    "pca8_8node": "Default",
    "8node8pca": "Default",
    "8node": "Default",
    "pca4": "PCA4",
    "pca2": "PCA2",
    "16node": "batch 64",
    "4node": "batch 16",
    "noatok": "No Action Tokens",
    "noadaln": "No AdaLN",
    "nocritic": "No Critic",
}

# External baselines, for the figures that place us alongside them.
BASELINES = {
    "minwm": "minWM",
    "matrixgame": "Matrix-Game",
    "worldcam": "WorldCam",
    "yume": "Yume",
    "worldplay": "WorldPlay",
    "astra": "Astra",
    "real": "Real",
    "freeze": "Freeze",
}

# Per-figure overrides. Each entry reproduces one submitted figure's legend
# verbatim; a run absent from a style falls back to NAMES/BASELINES above.
STYLES = {
    # response_curves_eval.py: lowercase throughout, "ours X" without brackets.
    "curves": {
        # every alias of the default run, since this script keys on pca8_8node
        "pca8": "ours Default", "pca8_8node": "ours Default",
        "8node8pca": "ours Default", "8node": "ours Default",
        "16node": "ours batch size 64",
        "minwm": "minwm", "matrixgame": "matrixgame", "worldcam": "worldcam",
        "yume": "yume", "worldplay": "worldplay", "astra": "astra",
    },
    # following_family_realdir.py, one style per family. The batch family names
    # all three runs by batch size, so the default model is "batch size 32"
    # there and "pca8" in the encoder family.
    "following_nodes": {
        "4node": "batch size 16", "pca8": "batch size 32", "16node": "batch size 64",
    },
    "following_encoders": {"pca8": "pca8", "pca4": "pca4", "pca2": "pca2"},
    "following_injection": {
        "pca8": "pca8 (batch size 32, adaln+tokens)",
        "noatok": "no action tokens (adaln-only)",
        "noadaln": "no AdaLN (tokens-only)",
    },
}


def label(run, ours=False, style=None):
    """Display name for a run key.

    style names an entry in STYLES, for figures whose submitted legend does not
    follow the default wording. ours=True prefixes "Ours (...)" for figures that
    mix our models with external baselines and need the provenance explicit; it
    is ignored when a style already spells the prefix out.
    """
    if style:
        override = STYLES[style].get(run)
        if override is not None:
            return override
    name = NAMES.get(run) or BASELINES.get(run) or run
    return f"Ours ({name})" if ours and run in NAMES else name
