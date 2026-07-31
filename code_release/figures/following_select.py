"""Select-direction version of following_8node8pca_by_REALdir.png — only
Forward, Right, Forward-Right, Backward, in a single row. Reuses build() and
the paper styling from following_by_realdir.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figures.following_by_realdir import build, DNAME

CDIR = "logs/v14e_pca8_raw/control_test"
KEY = "8node8pca"
SEL = ["F", "R", "FR", "B"]
CHUNK_SEC = 0.75   # one chunk = 3 latent frames = 12 px @16fps = 0.75s of video


def plot(df, out, value, ylabel, ylim, show_commanded):
    fig, ax = plt.subplots(2, 2, figsize=(11, 9.5))
    for i, D in enumerate(SEL):
        a = ax[i // 2][i % 2]
        sub = df[df["dir"] == D]
        if show_commanded:
            cref = sub.groupby("step")["commanded"].mean().sort_index()
            if len(cref):
                a.plot(cref.index,
                       cref.rolling(9, center=True, min_periods=3).mean().values,
                       ":", color="gray", lw=2, label="commanded |·| (target)")
        for br, col, ls in [("gt", "C0", "-"), ("flip", "crimson", "-")]:
            g = sub[sub.branch == br].groupby("step")[value].mean().sort_index()
            if len(g):
                a.plot(g.index,
                       g.rolling(9, center=True, min_periods=3).mean().values,
                       ls, color=col, lw=2.5, label=br)
        a.axhline(0, color="gray", lw=.6)
        a.set_ylim(*ylim)
        mins = len(sub) * CHUNK_SEC / 60.0
        a.set_title(f"{DNAME[D]} ({mins:.0f} min)", fontsize=17)
        if i // 2 == 1:
            a.set_xlabel("step", fontsize=16)
        if i % 2 == 0:
            a.set_ylabel(ylabel, fontsize=16)
        if i == 0:
            a.legend(fontsize=12)
        a.tick_params(labelsize=13)
        a.grid(alpha=.3)
    fig.tight_layout(pad=0.6)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    df = build(CDIR)
    plot(df, f"analysis/following_{KEY}_by_REALdir_select.png",
         "cos", "direction agreement (cosine)", (-1.05, 1.05), False)
    plot(df, f"analysis/following_{KEY}_by_REALdir_select_strength.png",
         "realized", "strength of followed action", (-0.5, 1.15), True)
    print(f"saved following_{KEY}_by_REALdir_select (+_strength) | rows={len(df)}")
