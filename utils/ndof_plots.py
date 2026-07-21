"""Uncommanded-DOF (roll=PC6, pitch=PC7) plots from analysis/chunk_metrics.csv.

Aggregated over ALL chunks per (run,step,branch) -- the per-direction split was
too small-sample and hid the trend. Two views:
  A) per-run: roll & pitch, each shown as NET (mean, vs the real-ride reference)
     and MAGNITUDE (mean |.|), for GT and FLIP branches.
  B) cross-run comparison: |roll| and |pitch| magnitude (FLIP) vs step, all runs.
Also keeps the commanded (PC0/PC1) following-by-direction graph.
"""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

CSV = "analysis/chunk_metrics.csv"
RUNS = {"16node": "16node b64", "pca8_8node": "8node b32", "4node": "4node b16",
        "pca2": "pca2 top2", "pca4": "pca4 top4", "noatok": "noatok tok-off"}
ROLL, PITCH = 6, 7


def _agg(df):
    """(run,step,branch) -> mean & mean-abs of gen roll/pitch + mean real roll/pitch."""
    g = df.groupby(["run", "step", "branch"]).agg(
        roll=("g6", "mean"), pitch=("g7", "mean"),
        roll_mag=("g6", lambda x: np.mean(np.abs(x))), pitch_mag=("g7", lambda x: np.mean(np.abs(x))),
        real_roll=("r6", "mean"), real_pitch=("r7", "mean")).reset_index()
    return g


def per_run(g):
    for run, lab in RUNS.items():
        d = g[g.run == run]
        if not len(d):
            continue
        fig, ax = plt.subplots(2, 2, figsize=(17, 10))
        for col, (dim, name) in enumerate([("roll", "ROLL (PC6)"), ("pitch", "PITCH (PC7)")]):
            realcol = f"real_{dim}"
            # top: NET (mean) gen for GT & FLIP + real reference
            a = ax[0][col]
            for br, c in [("gt", "C0"), ("flip", "crimson")]:
                s = d[d.branch == br].sort_values("step")
                if len(s):
                    a.plot(s.step, s[dim].rolling(3, center=True, min_periods=1).mean(), color=c, lw=2, label=f"{br} gen")
            rr = d[d.branch == "gt"].sort_values("step")
            if len(rr):
                a.axhline(rr[realcol].mean(), color="gray", ls="--", lw=1.5, label="real ride")
            a.axhline(0, color="k", lw=.5)
            a.set_title(f"{name} — NET (mean)"); a.set_xlabel("step"); a.grid(alpha=.3); a.legend(fontsize=9)
            # bottom: MAGNITUDE mean|.|
            a = ax[1][col]
            for br, c in [("gt", "C0"), ("flip", "crimson")]:
                s = d[d.branch == br].sort_values("step")
                if len(s):
                    a.plot(s.step, s[f"{dim}_mag"].rolling(3, center=True, min_periods=1).mean(), color=c, lw=2, label=f"{br}")
            a.set_title(f"{name} — MAGNITUDE mean|.| (higher = rolls/pitches more)")
            a.set_xlabel("step"); a.grid(alpha=.3); a.legend(fontsize=9)
        fig.suptitle(f"{lab} — uncommanded roll/pitch vs step (aggregated over all chunks)", fontsize=14)
        fig.tight_layout(); fig.savefig(f"analysis/ndof_rollpitch_{run}.png", dpi=115); plt.close(fig)
        print(f"saved analysis/ndof_rollpitch_{run}.png")


def cross_run(g):
    for br in ("gt", "flip"):
        fig, ax = plt.subplots(1, 2, figsize=(18, 6))
        for col, dim in enumerate(["roll", "pitch"]):
            a = ax[col]
            for run, lab in RUNS.items():
                s = g[(g.run == run) & (g.branch == br)].sort_values("step")
                if len(s) > 2:
                    a.plot(s.step, s[f"{dim}_mag"].rolling(3, center=True, min_periods=1).mean(), lw=2, marker="o", ms=2, label=lab)
            a.set_title(f"{dim.upper()} magnitude mean|.| ({br.upper()})")
            a.set_xlabel("step"); a.set_ylabel("mean |value|"); a.grid(alpha=.3); a.legend(fontsize=9)
        fig.suptitle(f"Uncommanded roll/pitch MAGNITUDE across runs — {br.upper()} (higher = more spurious roll/pitch)", fontsize=13)
        fig.tight_layout(); fig.savefig(f"analysis/ndof_rollpitch_compare_{br}.png", dpi=120); plt.close(fig)
        print(f"saved analysis/ndof_rollpitch_compare_{br}.png")


def main():
    df = pd.read_csv(CSV)
    for c in [f"r{d}" for d in range(8)]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    g = _agg(df.dropna(subset=["r6", "r7"]))
    per_run(g); cross_run(g)


if __name__ == "__main__":
    main()
