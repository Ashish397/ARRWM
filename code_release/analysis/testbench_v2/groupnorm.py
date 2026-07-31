"""Sibling-group normalization + dirty-source detection.

Conventions:
  - All videos generated from the same scene form a sibling group; every
    metric is reported relative to the group median. This cancels scene
    content changes and source defects. For external models the group is all
    models' rollouts of the same scene.
  - Dirty sources (water/smudge on lens, ~12% of clips): Laplacian variance
    at t=1s below the fleet's ~12th percentile -> exclude those scenes from
    degradation claims. Dirtiness is a property of the source clip, so the
    flag is computed per scene (median over siblings), not per video.
"""
import numpy as np

DIRTY_PCTL = 12.0
MIN_GROUP = 3


def add_group_relative(df, cols, group_key="scene"):
    """Add rel_{col} = col - median(col within sibling group).

    Rows in groups smaller than MIN_GROUP get NaN (no meaningful median).
    Real refs (scene < 0) are left NaN unless mapped to scenes.
    """
    df = df.copy()
    grouped = df[df[group_key] >= 0].groupby(group_key)
    for c in cols:
        med = grouped[c].transform("median")
        n = grouped[c].transform("count")
        rel = df.loc[df[group_key] >= 0, c] - med
        rel[n < MIN_GROUP] = np.nan
        df["rel_" + c] = np.nan
        df.loc[df[group_key] >= 0, "rel_" + c] = rel
    return df


def add_dirty_flags(df, lapvar_col="lapvar_base", pctl=DIRTY_PCTL):
    """Scene-level dirty flag from the fleet distribution of t=1s Laplacian
    variance. Returns df with 'scene_lapvar' and boolean 'dirty_scene'."""
    df = df.copy()
    scene_lv = (df[df["scene"] >= 0].groupby("scene")[lapvar_col]
                .median().rename("scene_lapvar"))
    thresh = np.percentile(scene_lv.values, pctl)
    df = df.merge(scene_lv, on="scene", how="left")
    df["dirty_scene"] = df["scene_lapvar"] < thresh
    df["dirty_scene"] = df["dirty_scene"].fillna(False)
    df.attrs["dirty_threshold"] = float(thresh)
    return df


def zscore_fleet(df, cols, ref_mask=None):
    """Fleet-wide z-scoring (robust: median/1.4826*MAD). If ref_mask is given,
    the location/scale are estimated on that subset only (e.g. clean scenes)."""
    df = df.copy()
    base = df[ref_mask] if ref_mask is not None else df
    for c in cols:
        v = base[c].dropna().values
        loc = np.median(v)
        scale = 1.4826 * np.median(np.abs(v - loc)) + 1e-9
        df["z_" + c] = (df[c] - loc) / scale
    return df
