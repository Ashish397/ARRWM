"""Quality composite: ridge regression over group-relative features,
evaluated against labels/human_tiers.csv — the sole ground truth.

Label schema: scene_key (e.g. r04_BL), scene (int), direction, model,
tier (float, HIGHER = better), style_shift (0/1), action_follow (0/1), note.

Model-selection protocol:
  - Any metric/weight/blend is selected by leave-one-scene-out CV (folds by
    scene int, so r04_BL and r04_BR never straddle a fold boundary).
  - Pairwise accuracy counts cross-tier pairs WITHIN a scene_key only
    (human tiers were assigned within one scene+direction context);
    ties = 1/2 credit; macro-averaged over scenes.
  - Weights are fitted on training folds only; evaluation pairs are held
    out of the fit.

The composite score is calibrated to tier, so HIGHER = better everywhere
downstream (scorecard, Bradley-Terry prescreen).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import groupnorm

# group-relative features feeding the composite (V1 feature set)
FEATURES = [
    "rel_csd_drift_base_end",     # style-embedding drift start->end
    "rel_dino_consistency",       # frame consistency (adjacent native frames)
    "rel_niqe_end", "rel_unique_end", "rel_liqe_end",
    "rel_haze_dark_rise", "rel_dark_frac_drift",
    "rel_raft_warp_err",
    "rel_csd_patch_spread",       # worst-region CSD-drift spread
]


def _ridge_fit(X, y, alpha=1.0):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    n, d = Xs.shape
    w = np.linalg.solve(Xs.T @ Xs + alpha * np.eye(d), Xs.T @ (y - y.mean()))
    return dict(mu=mu, sd=sd, w=w, b=float(y.mean()))


def _ridge_predict(m, X):
    return ((np.asarray(X) - m["mu"]) / m["sd"]) @ m["w"] + m["b"]


def pairwise_accuracy(pred, tier):
    """Cross-tier pairwise accuracy, ties = 1/2 credit. pred and tier share
    the same direction convention (higher = better)."""
    correct, total = 0.0, 0
    n = len(pred)
    for i in range(n):
        for j in range(i + 1, n):
            if tier[i] == tier[j]:
                continue
            total += 1
            d_pred, d_tier = pred[i] - pred[j], tier[i] - tier[j]
            if d_pred == 0:
                correct += 0.5
            elif np.sign(d_pred) == np.sign(d_tier):
                correct += 1
    return (correct / total, total) if total else (np.nan, 0)


def build_feature_table(cpu_csv, gpu_csv):
    cpu = pd.read_csv(cpu_csv)
    gpu = pd.read_csv(gpu_csv)
    drop = [c for c in ("model", "scene", "direction", "native_fps")
            if c in gpu.columns]
    df = cpu.merge(gpu.drop(columns=drop), on="vid", how="inner")
    raw_cols = [f[len("rel_"):] for f in FEATURES]
    df = groupnorm.add_group_relative(df, [c for c in raw_cols if c in df], "scene")
    df = groupnorm.add_dirty_flags(df)
    return df


def _merge_labels(df, labels, features):
    lab = labels.merge(
        df[["model", "scene", "direction"] + features + ["dirty_scene"]],
        on=["model", "scene", "direction"], how="left")
    missing = lab[lab[features[0]].isna()]
    if len(missing):
        print(f"WARNING: {len(missing)} labeled rows lack features: "
              + ", ".join(missing.scene_key + "/" + missing.model))
    return lab.dropna(subset=features)


def loso_cv(df, labels, alpha=1.0, features=None):
    """Leave-one-scene-out CV (folds by scene int, pairs within scene_key).
    Returns (per-scene_key df with predictions, macro acc, final model)."""
    features = features or [f for f in FEATURES if f in df.columns]
    lab = _merge_labels(df, labels, features)
    rows, preds = [], []
    for held in sorted(lab["scene"].unique()):
        tr, te = lab[lab.scene != held], lab[lab.scene == held]
        if len(tr) < len(features) + 2 or len(te) < 2:
            continue
        m = _ridge_fit(tr[features].values, tr["tier"].values.astype(float), alpha)
        te = te.copy()
        te["composite_pred"] = _ridge_predict(m, te[features].values)
        preds.append(te)
        for key, g in te.groupby("scene_key"):
            acc, npairs = pairwise_accuracy(g["composite_pred"].values,
                                            g["tier"].values)
            rows.append(dict(scene=held, scene_key=key, acc=acc,
                             n_pairs=npairs,
                             dirty=bool(g["dirty_scene"].iloc[0])))
    per_key = pd.DataFrame(rows)
    # macro over scenes: average scene_key accs within a scene first
    macro = float(per_key.groupby("scene")["acc"].mean().mean()) \
        if len(per_key) else np.nan
    final = _ridge_fit(lab[features].values, lab["tier"].values.astype(float),
                       alpha)
    final["features"] = features
    final["alpha"] = alpha
    preds = pd.concat(preds) if preds else pd.DataFrame()
    return per_key, macro, final, preds


def score_fleet(df, model):
    feats = model["features"]
    ok = df[feats].notna().all(axis=1)
    df = df.copy()
    df["composite"] = np.nan
    df.loc[ok, "composite"] = _ridge_predict(
        {k: model[k] for k in ("mu", "sd", "w", "b")}, df.loc[ok, feats].values)
    return df


def gate_report(preds, per_key, macro, alpha, out_path):
    """Gate (b) artifact: per-scene tables for human go/no-go review."""
    with open(out_path, "w") as f:
        f.write("# Gate (b): composite vs human tiers\n\n")
        f.write(f"LOSO macro cross-tier pairwise accuracy (ties=1/2, "
                f"macro over scenes): **{macro:.3f}** (alpha={alpha}; "
                f"V1 context: ~0.78-0.82)\n\n")
        f.write("## Per scene_key accuracy\n\n")
        f.write(per_key.to_markdown(index=False, floatfmt=".3f"))
        f.write("\n\n## Held-out predictions vs human tiers\n")
        for key, g in preds.groupby("scene_key"):
            g = g.sort_values("tier", ascending=False)
            f.write(f"\n### {key}\n\n")
            cols = ["model", "tier", "composite_pred", "style_shift", "note"]
            f.write(g[cols].to_markdown(index=False, floatfmt=".2f"))
            f.write("\n")
    print(f"gate (b) report -> {out_path}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    cpu_csv = os.environ.get("TB2_CPU_OUT", os.path.join(here, "out", "cpu_features.csv"))
    gpu_csv = os.environ.get("TB2_GPU_OUT", os.path.join(here, "out", "gpu_features.csv"))
    labels_csv = os.environ.get("TB2_LABELS", os.path.join(here, "labels", "human_tiers.csv"))
    out_dir = os.path.join(here, "out")
    os.makedirs(out_dir, exist_ok=True)

    df = build_feature_table(cpu_csv, gpu_csv)
    df.to_csv(os.path.join(out_dir, "features_grouped.csv"), index=False)

    if not os.path.exists(labels_csv):
        print(f"no labels at {labels_csv} — wrote grouped features only")
        return
    labels = pd.read_csv(labels_csv)
    best = None
    for alpha in (0.1, 1.0, 10.0):   # selected by LOSO, honest
        per_key, macro, final, preds = loso_cv(df, labels, alpha)
        print(f"alpha={alpha}: LOSO macro cross-tier pairwise acc = {macro:.3f}")
        if best is None or macro > best[1]:
            best = (alpha, macro, per_key, final, preds)
    alpha, macro, per_key, final, preds = best
    per_key.to_csv(os.path.join(out_dir, "composite_loso.csv"), index=False)
    json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v)
               for k, v in final.items()},
              open(os.path.join(out_dir, "composite_model.json"), "w"), indent=1)
    score_fleet(df, final).to_csv(os.path.join(out_dir, "features_scored.csv"),
                                  index=False)
    gate_report(preds, per_key, macro, alpha,
                os.path.join(out_dir, "gate_b_report.md"))


if __name__ == "__main__":
    main()
