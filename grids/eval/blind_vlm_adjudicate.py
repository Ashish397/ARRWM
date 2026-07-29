"""VLM bakeoff adjudication: p_uncanny (geometry) and p_novel (plausibility/actors)
across Qwen3-VL-8B / Cosmos-Reason1-7B / InternVL3-8B on blind100."""
import os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
MODELS = ["qwen3vl8b", "cosmos_reason1_7b", "internvl3_8b"]
ACTORS = {"V069": "robot", "V009": "car(approach)", "V034": "car(side)", "V028": "skier"}


def auc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int); ok = ~np.isnan(s); s, y = s[ok], y[ok]
    p, n = s[y == 1], s[y == 0]
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    a = np.concatenate([p, n]); r = pd.Series(a).rank().values
    return (r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))


def main():
    lab = pd.read_csv(os.path.expanduser("~/blind100_labels_and_scores.csv"))
    lab["t_implaus"] = lab.other.astype(str).str.contains(r"implausible|uncanny", case=False, na=False).astype(int)
    probes = {}
    for m in MODELS:
        f = os.path.join(OUT, f"blind_vlmprobes_{m}.csv")
        if os.path.exists(f):
            probes[m] = pd.read_csv(f)

    print("=== GEOMETRY: p_uncanny vs human_geom_mangle (n_pos=%d) ===" % int(lab.human_geom_mangle.sum()))
    for m, p in probes.items():
        d = lab.merge(p[["blind_id", "p_uncanny"]], on="blind_id")
        print(f"    {m:20s} AUC={auc(d.p_uncanny, d.human_geom_mangle):.3f}")

    print("\n=== PLAUSIBILITY: p_novel vs implausible tag (n_pos=%d) ===" % int(lab.t_implaus.sum()))
    for m, p in probes.items():
        d = lab.merge(p[["blind_id", "p_novel"]], on="blind_id")
        print(f"    {m:20s} AUC={auc(d.p_novel, d.t_implaus):.3f}")

    print("\n=== ACTOR firing: p_novel per VLM (should be high) ===")
    hdr = "    " + "video/actor".ljust(20) + "".join(m[:10].ljust(12) for m in MODELS)
    print(hdr)
    for vid, name in ACTORS.items():
        cells = ""
        for m in MODELS:
            v = probes[m].set_index("blind_id").p_novel.get(vid, np.nan) if m in probes else np.nan
            cells += f"{v:<12.3f}" if v == v else "--          "
        print(f"    {(vid+' '+name):20s}{cells}")

    print("\n=== STYLE (bonus): p_style vs style tag ===")
    lab["t_style"] = lab.other.astype(str).str.contains("style", case=False, na=False).astype(int)
    for m, p in probes.items():
        d = lab.merge(p[["blind_id", "p_style"]], on="blind_id")
        print(f"    {m:20s} AUC={auc(d.p_style, d.t_style):.3f}")


if __name__ == "__main__":
    main()
