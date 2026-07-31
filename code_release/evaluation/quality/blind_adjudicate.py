"""Master adjudication for the blind100 metric bakeoff.

Loads every available out/blind_*.csv, and for each axis with blind100 ground truth
prints a competitor table (AUC vs the relevant human tag), ranked. Metrics not yet
computed are skipped. Geometry components are also validated against the reference CSV.
"""
import os, re
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
REF = os.path.expanduser("~/blind100_labels_and_scores.csv")


def auc(score, pos, higher_bad=True):
    s = np.asarray(score, float) * (1 if higher_bad else -1)
    y = np.asarray(pos, int); ok = ~np.isnan(s); s, y = s[ok], y[ok]
    p, n = s[y == 1], s[y == 0]
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    a = np.concatenate([p, n]); r = pd.Series(a).rank().values
    return (r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))


def load(name):
    p = os.path.join(OUT, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def main():
    lab = pd.read_csv(REF)
    def tag(t): return lab.other.astype(str).str.contains(t, case=False, na=False).astype(int)
    lab["t_style"] = tag("style"); lab["t_scene"] = tag("scene")
    lab["t_implaus"] = tag(r"implausible|uncanny"); lab["t_shimmer"] = tag("shimmer")
    lab["t_static"] = tag("static")

    d = lab.copy()
    for f in ["blind_cpu_metrics.csv", "blind_style_shift.csv", "blind_scene_reloc.csv",
              "blind_plausibility.csv", "blind_dino_drift.csv",
              "blind_melt_qwen25vl7b.csv", "blind_melt_cosmos_reason1_7b.csv",
              "blind_pal.csv", "blind_depth.csv"]:
        t = load(f)
        if t is None:
            continue
        keep = [c for c in t.columns if c not in ("vid", "model", "scene")]
        if f == "blind_melt_qwen25vl7b.csv":
            t = t.rename(columns={"melt_pyes": "melt_qwen"}); keep = ["blind_id", "melt_qwen"]
        if f == "blind_melt_cosmos_reason1_7b.csv":
            t = t.rename(columns={"melt_pyes": "melt_cosmos"}); keep = ["blind_id", "melt_cosmos"]
        d = d.merge(t[keep], on="blind_id", how="left", suffixes=("", "_dup"))

    # axis -> (target, [(col, higher_bad, label)])
    AX = {
        "GEOMETRY mangle (n=%d)": ("human_geom_mangle", [
            ("melt_cosmos", True, "melt Cosmos"), ("melt_qwen", True, "melt Qwen2.5"),
            ("qwen_melt_pyes", True, "melt Qwen (ref CSV)"),
            ("depth_rough_base", True, "depth-curvature"), ("pal_max", True, "PAL4VST (rerun)"),
            ("pal4vst_max", True, "PAL4VST (ref CSV)"), ("p_uncanny", True, "VLM uncanny"),
            ("hf_lap_drift", True, "laplacian drift")]),
        "STYLE shift (n=%d)": ("t_style", [
            ("ss_clip_dist", True, "CLIP drift"), ("dino_drift", True, "DINOv2 drift"),
            ("p_style", True, "VLM p_style"), ("ss_gram_dist", True, "VGG-Gram")]),
        "SCENE relocation (n=%d)": ("t_scene", [
            ("orb_inliers_ransac", False, "ORB+RANSAC inliers"),
            ("inliers", False, "ORB inliers (scene_reloc)"),
            ("orb_raw_matches", False, "ORB raw (no RANSAC)"),
            ("dino_drift", True, "DINOv2 drift"), ("ss_clip_dist", True, "CLIP drift")]),
        "PLAUSIBILITY implausible (n=%d)": ("t_implaus", [
            ("p_novel", True, "VLM p_novel"), ("p_uncanny", True, "VLM p_uncanny")]),
        "HAZE/high-freq ~shimmer proxy (n=%d)": ("t_shimmer", [
            ("hf_lap_drift", True, "laplacian sharpness loss"),
            ("hf_darkchan_drift", True, "dark-channel veiling"),
            ("hf_contrast_drift", True, "contrast loss")]),
        "STATIC/stillness (n=%d)": ("t_static", [
            ("static_ncc", True, "adjacent NCC (high=static)"),
            ("orb_inliers_ransac", True, "ORB inliers (high=static)"),
            ("static_absdiff", False, "mean abs-diff (low=static)")]),
    }
    for title, (target, cols) in AX.items():
        y = d[target]
        print("\n" + title % int(y.sum()) + f"   [target={target}]")
        res = []
        for col, hb, label in cols:
            if col not in d.columns:
                continue
            res.append((label, auc(d[col], y, hb), col))
        for label, a, col in sorted(res, key=lambda x: (-x[1] if x[1] == x[1] else 0)):
            print(f"    {label:26s} AUC={a:.3f}")

    # geometry validation vs reference CSV
    print("\n=== geometry rerun vs reference CSV (Spearman) ===")
    for rerun, ref in [("melt_qwen", "qwen_melt_pyes"), ("pal_max", "pal4vst_max"),
                       ("depth_rough_base", "depth_rough_base")]:
        if rerun in d.columns and ref in lab.columns:
            a = d[rerun]; b = lab.set_index("blind_id").loc[d.blind_id, ref].values
            m = ~(np.isnan(a) | np.isnan(b))
            if m.sum() > 3:
                rho = pd.Series(a[m].values).corr(pd.Series(b[m]), method="spearman")
                print(f"    {rerun:18s} vs {ref:18s} rho={rho:.3f} (n={m.sum()})")


if __name__ == "__main__":
    main()
