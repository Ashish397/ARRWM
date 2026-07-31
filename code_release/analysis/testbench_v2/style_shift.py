"""Style shift — the primary discriminator for external models. Three
instruments, reported SEPARATELY (never blended):

  (a) zsum:  MUSIQ drift (ctx->end) + VGG-Gram distance + MS-SWD color
      distance, fleet-z-scored per component and summed. [V1: AUC 0.86 on
      subtle drift — context, not a target]
  (b) csd:   CSD style-embedding drift ctx->end (fleet-z-scored). Should
      fire hard on photoreal->game restyling.
  (c) vlm:   the VLM first-vs-last-second style question. Blind on subtle
      drift in V1 (AUC 0.50) — untrusted until validated.

Anchoring (sibling-group design): the real continuation of a scene is the
per-scene anchor — each rollout's drift is reported as EXCESS over its
scene's real-ref drift, which cancels scene-specific innocent drift exactly.
Scenes without a mapped real sibling fall back to the group median as
anchor, marked anchor_source='group_median'.

Flagging threshold: recalibrated on this fleet — the p95 of the CENTERED
real-ref instrument values (the spread of innocent real drift), applied to
the excess. Never reuse V1 population stats.

Validation gates before the flags are used:
  - known-restyled positive control (matrixgame) vs real refs;
  - human style_shift labels from labels/human_tiers.csv (subtle drift).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import groupnorm

NULL_PCTL = 95.0
INSTRUMENT_COLS = dict(zsum="style_zsum", csd="z_csd_drift_ctx_end",
                       vlm="vlm_style_pyes")
COMPONENTS = ["musiq_ctx_end_drift", "gram_ctx_end", "msswd_ctx_end",
              "csd_drift_ctx_end"]


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if not len(pos) or not len(neg):
        return np.nan
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def build(df, vlm_style_csv=None):
    """Adds instrument columns + per-scene anchored excess_* columns."""
    present = [c for c in COMPONENTS if c in df]
    df = groupnorm.zscore_fleet(df, present)   # unit normalization
    zsum_parts = [c for c in ("z_musiq_ctx_end_drift", "z_gram_ctx_end",
                              "z_msswd_ctx_end") if c in df]
    df["style_zsum"] = df[zsum_parts].sum(axis=1, min_count=len(zsum_parts))
    if vlm_style_csv and os.path.exists(vlm_style_csv):
        v = pd.read_csv(vlm_style_csv)[["vid", "vlm_style_pyes"]]
        df = df.merge(v, on="vid", how="left")
    else:
        df["vlm_style_pyes"] = np.nan

    # per-scene anchor: the real sibling if present, else group median
    for name, col in INSTRUMENT_COLS.items():
        if col not in df:
            continue
        real_anchor = (df[df.model == "real"].groupby("scene")[col].median()
                       .rename("_ra"))
        med_anchor = (df[(df.scene >= 0) & (df.model != "real")]
                      .groupby("scene")[col].median().rename("_ma"))
        df = df.merge(real_anchor, on="scene", how="left")
        df = df.merge(med_anchor, on="scene", how="left")
        df[f"excess_{name}"] = df[col] - df["_ra"].fillna(df["_ma"])
        df[f"anchor_source_{name}"] = np.where(df["_ra"].notna(),
                                               "real", "group_median")
        df = df.drop(columns=["_ra", "_ma"])
    return df


def null_thresholds(df):
    """p95 of the CENTERED real-ref instrument values = the spread of
    innocent real first-vs-last drift, applied to the anchored excess.
    Fallback (no real refs): clean-fleet p90, flagged *_fallback."""
    real = df[df.model == "real"]
    th = {}
    for name, col in INSTRUMENT_COLS.items():
        vals = real[col].dropna().values if col in real else np.array([])
        if len(vals) >= 8:
            th[name] = float(np.percentile(vals - np.median(vals), NULL_PCTL))
        else:
            mask = ~df["dirty_scene"] if "dirty_scene" in df else slice(None)
            pool = df.loc[mask, f"excess_{name}"].dropna().values \
                if f"excess_{name}" in df else np.array([])
            th[name] = float(np.percentile(pool, 90.0)) if len(pool) else np.nan
            th[name + "_fallback"] = True
    return th


def rates(df, th):
    """Per-model style-shift rate per instrument (excess > threshold),
    non-dirty scenes only. anchor_frac_real records how often the anchor
    was the real sibling rather than the group-median fallback."""
    d = df[(~df["dirty_scene"]) & (df.model != "real") & (df.scene >= 0)]
    rows = []
    for model, g in d.groupby("model"):
        row = dict(model=model, n=len(g))
        for name in INSTRUMENT_COLS:
            v = g[f"excess_{name}"].dropna() if f"excess_{name}" in g else pd.Series(dtype=float)
            row[f"style_rate_{name}"] = float((v > th[name]).mean()) if len(v) else np.nan
            row[f"style_n_{name}"] = int(len(v))
            src = g.get(f"anchor_source_{name}")
            row[f"anchor_frac_real_{name}"] = \
                float((src == "real").mean()) if src is not None else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("model")


def validate(df, labels_csv=None, restyled_model="matrixgame"):
    """Both validation gates. Prints AUCs; instrument (c) — and any
    instrument that fails — must not be used for flagging."""
    if restyled_model in set(df.model):
        print(f"positive control '{restyled_model}' vs real refs:")
        pos, neg = df[df.model == restyled_model], df[df.model == "real"]
        for name, col in INSTRUMENT_COLS.items():
            a = auc(pos[col].values if col in pos else [],
                    neg[col].values if col in neg else [])
            msg = "n/a (missing data)" if np.isnan(a) else \
                f"{a:.3f}" + ("" if a >= 0.70 else "  <-- DO NOT TRUST")
            print(f"  {name:5s}: AUC = {msg}")
    else:
        print(f"positive control '{restyled_model}' not in this feature set "
              "— restyle gate PENDING (required before fleet flags are used)")

    if labels_csv and os.path.exists(labels_csv):
        lab = pd.read_csv(labels_csv)
        m = lab.merge(df, on=["model", "scene", "direction"], how="inner")
        if len(m):
            print(f"human style_shift labels (subtle drift, n={len(m)}, "
                  f"{int(m.style_shift.sum())} positive):")
            for name in INSTRUMENT_COLS:
                col = f"excess_{name}"
                if col not in m:
                    continue
                a = auc(m.loc[m.style_shift == 1, col].values,
                        m.loc[m.style_shift == 0, col].values)
                msg = "n/a" if np.isnan(a) else f"{a:.3f}"
                print(f"  {name:5s} (excess): AUC = {msg}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.path.join(here, "out", "features_grouped.csv"))
    ap.add_argument("--vlm-style", default=os.path.join(here, "out", "vlm_style.csv"))
    ap.add_argument("--labels", default=os.path.join(here, "labels", "human_tiers.csv"))
    ap.add_argument("--restyled", default="matrixgame")
    ap.add_argument("--out", default=os.path.join(here, "out", "style_shift.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.features)
    df = build(df, args.vlm_style)
    th = null_thresholds(df)
    print("recalibrated thresholds (centered real-ref p95):",
          {k: (round(v, 3) if isinstance(v, float) else v) for k, v in th.items()})
    validate(df, args.labels, args.restyled)
    r = rates(df, th)
    r.to_csv(args.out, index=False)
    df.to_csv(args.out.replace(".csv", "_pervideo.csv"), index=False)
    print(r.to_string(index=False))


if __name__ == "__main__":
    main()
