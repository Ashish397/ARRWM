"""Canonical display names for the trained runs, shared by every figure script.

Each figure script grew its own label dict, so the same ablation could appear as
"Default", "Ours (Default)", "ours Default" and "batch size 32 (top8)" across
four figures in one paper. This module is the single source of truth.

Keys cover the directory-name variants the scripts inherited: the default model
is written ``pca8``, ``pca8_8node`` and ``8node8pca`` in different places, and
all three denote the same run.
"""

# run key -> display name. Batch sizes are the effective global batch (nodes x 4).
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


def label(run, ours=False):
    """Display name for a run key.

    ours=True prefixes "Ours (...)" for figures that mix our models with
    external baselines and need the provenance made explicit.
    """
    name = NAMES.get(run) or BASELINES.get(run) or run
    return f"Ours ({name})" if ours and run in NAMES else name
