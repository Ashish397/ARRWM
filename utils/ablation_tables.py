"""Build the 3 deliverable tables from the full-eval metric suite:
(a) 7-model x 4-axis master table, (b) direction-level, (c) colour-level.
Writes analysis/eval_final/ABLATION_METRICS.md + suite_all.csv (merged).
"""
import glob
import numpy as np
import pandas as pd

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
RUNS = ["pca8_8node", "pca4", "pca2", "16node", "4node", "noatok", "noadaln"]

# merge pass A
a = pd.concat([pd.read_csv(f) for f in glob.glob(f"{ARR}/analysis/eval_final/suiteA_*.csv")])
# merge vlisa
vl = []
for run in RUNS:
    df = pd.read_csv(f"{ARR}/analysis/eval_final/vlisa_all_{run}.csv")
    df["run"] = run
    # window col like r00_F -> split
    df[["window", "dir"]] = df["window"].str.extract(r"(r\d+)_(\w+)")
    vl.append(df[["run", "window", "dir", "upper_mean"]])
vl = pd.concat(vl).rename(columns={"upper_mean": "vlisa"})
m = a.merge(vl, on=["run", "window", "dir"], how="outer")
m = m.rename(columns={"pal_mean": "pal", "qwen_top4": "qwen"})
cols = pd.read_csv(f"{ARR}/analysis/eval_final/window_colours.csv")
m = m.merge(cols[["window", "colour", "green_frac"]], on="window", how="left")
m["green_terc"] = pd.qcut(m.green_frac, 3, labels=["low-green", "mid-green", "high-green"])
m.to_csv(f"{ARR}/analysis/eval_final/suite_all.csv", index=False)
AXES = ["vlisa", "pal", "qwen", "center_nov", "spawn_jump"]

lines = ["# Ablation metric suite — full phase-A eval (7 models x 32 windows x 8 dirs)",
         "", f"Videos scored: {len(m)} (VideoLISA melt | PAL warp | Qwen surface | DINO spawn: center_nov + jump)",
         "", "## (a) Master table — mean per model (parentheses: p90 worst-tail)", ""]
rows = []
for run in RUNS:
    s = m[m.run == run]
    row = {"model": run}
    for ax in AXES:
        row[ax] = f"{s[ax].mean():.3f} ({s[ax].quantile(0.9):.3f})"
    rows.append(row)
t = pd.DataFrame(rows).set_index("model")
lines += [t.to_markdown(), ""]

lines += ["## (b) Direction-level (mean per model x dir)", ""]
for ax in AXES:
    piv = m.pivot_table(index="run", columns="dir", values=ax, aggfunc="mean")
    piv = piv.reindex(RUNS)[["F", "FR", "R", "BR", "B", "BL", "L", "FL"]]
    lines += [f"### {ax}", piv.round(3).to_markdown(), ""]

lines += ["## (c) Colour-level (seed-scene palette; all models pooled + per-model vlisa/pal)", ""]
pooled = m.groupby("colour")[AXES].mean().round(3)
pooled["n_videos"] = m.groupby("colour").size()
lines += ["### by colour family (pooled over models)", pooled.to_markdown(), ""]
terc = m.groupby("green_terc", observed=True)[AXES].mean().round(3)
terc["n_videos"] = m.groupby("green_terc", observed=True).size()
lines += ["### by vegetation ('shade of green') tercile (pooled)", terc.to_markdown(), ""]
for ax in ("vlisa", "pal"):
    piv = m.pivot_table(index="run", columns="colour", values=ax, aggfunc="mean").reindex(RUNS)
    lines += [f"### {ax} by model x colour", piv.round(3).to_markdown(), ""]

open(f"{ARR}/analysis/eval_final/ABLATION_METRICS.md", "w").write("\n".join(lines))
print("\n".join(lines[:40]))
print("... written to ABLATION_METRICS.md")
