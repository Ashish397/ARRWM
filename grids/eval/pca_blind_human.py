"""PCA + usefulness analysis of ALL blind100 candidate metrics against the human
error dimensions. 'Useful' = aligns with a human-labelled error type."""
import os
import pandas as pd, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
lab = pd.read_csv(os.path.expanduser("~/blind100_labels_and_scores.csv"))
tag = lambda t: lab.other.astype(str).str.contains(t, case=False, na=False).astype(int)
# human error dimensions (each error type)
H = pd.DataFrame({"blind_id": lab.blind_id})
H["h_geom"] = (lab.geom == "mangle").astype(int)
H["h_style"] = tag("style")
H["h_scene"] = tag("scene")
H["h_shimmer"] = tag("shimmer")
H["h_implaus"] = tag(r"implausible|uncanny")
HD = ["h_geom", "h_style", "h_scene", "h_shimmer", "h_implaus"]


def L(f, ren=None, cols=None):
    p = os.path.join(HERE, "out", f)
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    if ren:
        d = d.rename(columns=ren)
    keep = ["blind_id"] + (cols if cols else [c for c in d.columns if c not in
            ("blind_id", "vid", "model", "scene", "direction", "note", "tier")])
    return d[[c for c in keep if c in d.columns]]


d = H.copy()
for v in ["qwen3vl8b", "cosmos_reason1_7b", "internvl3_8b"]:
    s = v.split("_")[0][:4]
    t = L(f"blind_vlmprobes_{v}.csv", {"p_style": f"pstyle_{s}", "p_uncanny": f"punc_{s}", "p_novel": f"pnov_{s}"})
    d = d.merge(t[["blind_id", f"pstyle_{s}", f"punc_{s}", f"pnov_{s}"]], on="blind_id")
    tn = L(f"blind_temporal_{v}.csv", {"novel_sudden": f"nsud_{s}"})
    d = d.merge(tn[["blind_id", f"nsud_{s}"]], on="blind_id")
d = d.merge(L("blind_style_shift.csv", cols=["ss_gram_dist", "ss_clip_dist"]), on="blind_id")
d = d.merge(L("blind_dino_drift.csv", cols=["dino_drift"]), on="blind_id")
d = d.merge(L("blind_scene_reloc.csv", cols=["inliers"]), on="blind_id")
d = d.merge(L("blind_scene_features.csv", cols=["orb_inliers", "sift_inliers", "akaze_inliers"]), on="blind_id")
d = d.merge(L("blind_pal.csv", cols=["pal_max"]), on="blind_id")
d = d.merge(L("blind_depth.csv", cols=["depth_rough_base", "depth_tinstab"]), on="blind_id")
d = d.merge(L("blind_melt_qwen25vl7b.csv", {"melt_pyes": "melt_qwen"}, ["melt_qwen"]), on="blind_id")
d = d.merge(L("blind_melt_cosmos_reason1_7b.csv", {"melt_pyes": "melt_cosmos"}, ["melt_cosmos"]), on="blind_id")
d = d.merge(L("blind_cpu_metrics.csv", cols=["hf_lap_drift", "hf_darkchan_drift", "hf_contrast_drift"]), on="blind_id")
d = d.merge(L("blind_hf.csv", cols=["B"]), on="blind_id")

# relocation metrics: LOW inliers = relocated -> negate so higher = worse
for c in ["inliers", "orb_inliers", "sift_inliers", "akaze_inliers"]:
    d[c] = -d[c]
METRICS = [c for c in d.columns if c not in HD + ["blind_id"]]
print(f"blind100: {len(d)} rollouts, {len(METRICS)} candidate metrics, {len(HD)} human dims\n")


def auc(score, lab_):
    s = np.asarray(score, float); y = np.asarray(lab_, int); ok = ~np.isnan(s); s, y = s[ok], y[ok]
    p, n = s[y == 1], s[y == 0]
    if len(p) == 0 or len(n) == 0:
        return np.nan
    a = np.concatenate([p, n]); r = pd.Series(a).rank().values
    return (r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))


# ---- usefulness table: AUC of each metric vs each human dimension ----
print("=== USEFULNESS: AUC(metric, human error dim). Best per column in []. ===")
print(f"{'metric':16s}" + "".join(f"{h.replace('h_',''):>9s}" for h in HD) + "   bestAUC")
res = {}
for m in METRICS:
    aucs = [auc(d[m], d[h]) for h in HD]
    res[m] = aucs
best = {h: max(res, key=lambda m: res[m][i] if res[m][i] == res[m][i] else 0) for i, h in enumerate(HD)}
for m in METRICS:
    aucs = res[m]
    row = "".join((f"[{a:.2f}]" if best[HD[i]] == m else f" {a:.2f} ") for i, a in enumerate(aucs))
    print(f"{m:16s}" + row + f"   {max(a for a in aucs if a==a):.2f}")
print("\nBest metric per human error dimension:")
for i, h in enumerate(HD):
    print(f"  {h:12s} -> {best[h]:16s} (AUC {res[best[h]][i]:.2f})")
# useless metrics: max AUC < 0.6 across all human dims
useless = [m for m in METRICS if max((a for a in res[m] if a == a), default=0) < 0.6]
print(f"\nMetrics useful for NO human dimension (max AUC<0.6): {useless}")

# ---- PCA on metrics + human dims ----
allc = METRICS + HD
X = d[allc].astype(float).dropna()
Z = (X - X.mean()) / X.std()
U, S, Vt = np.linalg.svd(Z.values, full_matrices=False)
var = S**2 / (S**2).sum()
print(f"\n=== PCA on {len(METRICS)} metrics + {len(HD)} human dims ===")
print("var:", " ".join(f"PC{i+1}={v*100:.0f}%" for i, v in enumerate(var[:6])), f" (cum6={np.cumsum(var)[5]*100:.0f}%)")
print("\nFor each human dim: which PC it loads on, and the metrics sharing that PC:")
for h in HD:
    j = allc.index(h)
    pc = int(np.argmax(np.abs(Vt[:, j])))            # dominant PC for this human dim
    load_h = Vt[pc, j]
    aligned = sorted([(allc[k], Vt[pc, k]) for k in range(len(METRICS))],
                     key=lambda x: -abs(x[1]))[:5]
    sign = np.sign(load_h)
    print(f"  {h:12s} loads PC{pc+1} ({load_h:+.2f}); top metrics on PC{pc+1}: " +
          ", ".join(f"{n}({v:+.2f})" for n, v in aligned))
