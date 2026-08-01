"""Flip-following by REAL ride direction, per v14e run. Produces TWO views:

  A) DIRECTION-ONLY (the original, cleaner view): cosine agreement between the
     commanded action and the teacher-read action, per real
     direction, GT solid / FLIP dashed. -> following_{run}_by_REALdir.png
  B) DIRECTION+STRENGTH: realized magnitude along the command (c.t/|c|, sign-
     preserved) with the commanded magnitude as the gain=1 target line. Weak-but-
     correct now scores low. -> following_{run}_by_REALdir_strength.png
"""
import glob, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from figures.figure_labels import label

RUNS = [(f"logs/v14e_{d}/control_test", label(k), k) for d, k in [
    ("16node", "16node"), ("pca8_raw", "8node8pca"), ("4node", "4node"),
    ("pca2", "pca2"), ("pca4", "pca4"), ("noatok", "noatok"),
    ("noadaln", "noadaln"), ("nocritic", "nocritic"),
]]
DIRS = ["F", "FR", "R", "BR", "B", "BL", "L", "FL"]
DNAME = {"F": "Forward", "FR": "Forward-Right", "R": "Right", "BR": "Backward-Right",
         "B": "Backward", "BL": "Backward-Left", "L": "Left", "FL": "Forward-Left"}
TH = 0.1
CHUNK_SEC = 0.75  # one chunk = 3 latent frames = 12 px @16fps = 0.75s of video


def build(cdir):
    rows = []
    for f in glob.glob(f"{cdir}/metrics_r*.jsonl"):
        for ln in open(f):
            try:
                d = json.loads(ln)
            except Exception:
                continue
            step = d.get("step")
            # Keep only the first 4 offset windows (levels 0-3, offset<=81=3*tot_f).
            # The live eval later added deeper windows (108-189) whose action stats
            # differ; restricting to the common 4 makes every step comparable.
            if (d.get("offset") or 0) > 3 * 27:
                continue
            for br in ("gt", "flip"):
                b = d.get(br, {})
                cz2, cz7 = b.get("cz2"), b.get("cz7")
                tz2, tz7 = b.get("tz2"), b.get("tz7")
                if not cz2:
                    continue
                for k in range(len(cz2)):
                    c2, c7 = cz2[k], cz7[k]
                    t2, t7 = tz2[k], tz7[k]
                    r2, r7 = (c2, c7) if br == "gt" else (-c2, -c7)
                    fb = "F" if r2 > TH else ("B" if r2 < -TH else "")
                    lr = "R" if r7 > TH else ("L" if r7 < -TH else "")
                    dn = fb + lr
                    if not dn:
                        continue
                    c = np.array([c2, c7]); t = np.array([t2, t7])
                    cn = np.linalg.norm(c); tn = np.linalg.norm(t)
                    cos = float(c @ t / (cn * tn)) if cn * tn > 1e-6 else np.nan
                    proj = float(c @ t / cn) if cn > 1e-6 else np.nan
                    rows.append((step, br, dn, cos, proj, float(cn)))
    return pd.DataFrame(rows, columns=["step", "branch", "dir", "cos", "realized", "commanded"]).dropna()


def plot_direction(df, title, out):
    """Original direction-only view (cosine agreement)."""
    fig, ax = plt.subplots(2, 4, figsize=(22, 10))
    for i, D in enumerate(DIRS):
        a = ax[i // 4][i % 4]; sub = df[df["dir"] == D]
        for br, col, ls in [("gt", "C0", "-"), ("flip", "crimson", "-")]:
            g = sub[sub.branch == br].groupby("step")["cos"].mean().sort_index()
            if len(g):
                a.plot(g.index, g.rolling(9, center=True, min_periods=3).mean().values, ls, color=col, lw=2.5, label=br)
        a.axhline(0, color="gray", lw=.6); a.set_ylim(-1.05, 1.05)
        a.set_title(f"{DNAME[D]} ({len(sub) * CHUNK_SEC / 60:.0f} min)", fontsize=17)
        a.set_xlabel("step", fontsize=16)
        if i % 4 == 0:
            a.set_ylabel("direction agreement (cosine)", fontsize=16)
        a.tick_params(labelsize=14)
        a.grid(alpha=.3)
        if i == 0:
            a.legend(fontsize=13)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def plot_strength(df, title, out):
    """Direction + strength view (realized magnitude vs commanded target).
    Paper styling: no suptitle, big fonts, labeled y axis."""
    fig, ax = plt.subplots(2, 4, figsize=(22, 10))
    for i, D in enumerate(DIRS):
        a = ax[i // 4][i % 4]; sub = df[df["dir"] == D]
        cref = sub.groupby("step")["commanded"].mean().sort_index()
        if len(cref):
            a.plot(cref.index, cref.rolling(9, center=True, min_periods=3).mean().values,
                   ":", color="gray", lw=2, label="commanded |·| (target)")
        for br, col, ls in [("gt", "C0", "-"), ("flip", "crimson", "-")]:
            g = sub[sub.branch == br].groupby("step")["realized"].mean().sort_index()
            if len(g):
                a.plot(g.index, g.rolling(9, center=True, min_periods=3).mean().values, ls, color=col,
                       lw=2.5, label=br)
        a.axhline(0, color="gray", lw=.6); a.set_ylim(-0.5, 1.15)
        a.set_title(f"{DNAME[D]} ({len(sub) * CHUNK_SEC / 60:.0f} min)", fontsize=17)
        a.set_xlabel("step", fontsize=16)
        if i % 4 == 0:
            a.set_ylabel("strength of followed action", fontsize=16)
        a.tick_params(labelsize=14)
        a.grid(alpha=.3)
        if i == 0:
            a.legend(fontsize=13)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


if __name__ == "__main__":
    for cdir, title, key in RUNS:
        df = build(cdir)
        if not len(df):
            print(f"[skip] {cdir}: no metrics"); continue
        plot_direction(df, title, f"analysis/following_{key}_by_REALdir.png")
        plot_strength(df, title, f"analysis/following_{key}_by_REALdir_strength.png")
        print(f"saved following_{key} (direction + strength) | rows={len(df)} steps {int(df.step.min())}..{int(df.step.max())}")
