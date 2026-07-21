"""Response gain: realized vs commanded action magnitude, per axis, SPLIT BY BRANCH.

Pooling GT+FLIP is unfair: GT commands are almost all forward (inflate the forward
gain) and FLIP commands are almost all backward (deflate the backward gain), so
"direction" gets confounded with "branch". We therefore make separate figures:
  GT            : response_gain_gt.png            (real commands ~ forward half)
  FLIP          : response_gain_flip.png          (counterfactual ~ backward half)
  FLIP settled  : response_gain_flip_settled.png  (drop chunks 0,1 -- the model has to
                  decelerate then reverse after a flip, so the first 2 chunks are a
                  transient, not steady-state following)

Slope of realized-on-commanded = gain (1 = faithful, <1 = under-response).
Window: last LASTN steps per run.  Source: chunk_metrics.csv (fallback ndof_following.csv).
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

CSV = "analysis/chunk_metrics.csv" if os.path.exists("analysis/chunk_metrics.csv") else "analysis/ndof_following.csv"
RUNS = {
    "pca8_8node": ("batch size 32 (tokens ON)", "C1"), "noatok": ("no action tokens (tokens OFF)", "C5"),
    "16node": ("batch size 64", "C0"), "pca4": ("pca4 (top4)", "C3"),
    "pca2": ("pca2 (top2)", "C4"), "4node": ("batch size 16", "C2"),
}
AXES = [(0, "throttle / fwd-back (PC0)"), (1, "steer / yaw (PC1)")]
LASTN = 1000
MIN_RUN_N = 60
MIN_BIN_N = 12


def num(s):
    return pd.to_numeric(s, errors="coerce")


def gain_slope(x, y):
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    if len(x) < MIN_RUN_N:
        return np.nan
    A = np.vstack([x, np.ones_like(x)]).T
    return float(np.linalg.lstsq(A, y, rcond=None)[0][0])


def binned(x, y, lo, hi, nb=13):
    bins = np.linspace(lo, hi, nb); idx = np.digitize(x, bins)
    bx, by, be = [], [], []
    for bi in range(1, len(bins)):
        m = idx == bi
        if m.sum() >= MIN_BIN_N:
            bx.append(x[m].mean()); by.append(y[m].mean()); be.append(y[m].std() / np.sqrt(m.sum()))
    return np.array(bx), np.array(by), np.array(be)


def converged(df):
    parts = []
    for run in df.run.unique():
        d = df[df.run == run]
        if len(d):
            parts.append(d[d.step >= d.step.max() - LASTN])
    return pd.concat(parts) if parts else df


def make_figure(sub, title, out):
    gains = {}
    fig, ax = plt.subplots(1, 2, figsize=(19, 7.5))
    for ai, (d, name) in enumerate(AXES):
        a = ax[ai]; cc, gc = f"cmd{d}", f"g{d}"
        allc = num(sub[cc])
        lo, hi = np.nanquantile(allc, 0.02), np.nanquantile(allc, 0.98)
        a.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="y=x (gain 1)")
        a.axhline(0, color="gray", lw=.5); a.axvline(0, color="gray", lw=.5)
        if d == 0:   # dataset action range on the throttle axis
            a.axhline(0.5, color="#555", ls="-.", lw=1.3)
            a.axhline(-0.3, color="#555", ls="-.", lw=1.3)
            a.text(0.99, 0.5, " dataset max forward (+0.5)", transform=a.get_yaxis_transform(),
                   ha="right", va="bottom", fontsize=9, color="#555")
            a.text(0.99, -0.3, " dataset max backward (−0.3)", transform=a.get_yaxis_transform(),
                   ha="right", va="top", fontsize=9, color="#555")
        if d == 1:   # dataset steer range (real turns saturate the tanh squash)
            a.axhline(0.98, color="#555", ls="-.", lw=1.3)
            a.axhline(-1.0, color="#555", ls="-.", lw=1.3)
            a.text(0.99, 0.98, " dataset max right (+0.98)", transform=a.get_yaxis_transform(),
                   ha="right", va="bottom", fontsize=9, color="#555")
            a.text(0.99, -1.0, " dataset max left (−1.0)", transform=a.get_yaxis_transform(),
                   ha="right", va="top", fontsize=9, color="#555")
        for run, (lab, col) in RUNS.items():
            r = sub[sub.run == run]
            x, y = num(r[cc]).values, num(r[gc]).values
            g = gain_slope(x, y); gains[(run, d)] = g
            bx, by, be = binned(x[np.isfinite(x) & np.isfinite(y)], y[np.isfinite(x) & np.isfinite(y)], lo, hi)
            if len(bx):
                lw = 2.6 if run in ("pca8_8node", "noatok") else 1.4
                a.errorbar(bx, by, yerr=be, marker="o", ms=4, lw=lw, capsize=2, color=col,
                           label=f"{lab}: gain={g:.2f}" if np.isfinite(g) else lab)
        a.set_xlabel(f"commanded {name}"); a.set_ylabel("realized (teacher-read)")
        a.set_title(name); a.grid(alpha=.3); a.legend(fontsize=8)
    fig.suptitle(f"{title} — response gain (slope), last {LASTN} steps", fontsize=13)
    fig.tight_layout(); fig.savefig(out, dpi=125); plt.close(fig)
    print(f"saved {out}")
    return gains


def main():
    df = pd.read_csv(CSV)
    # keep only the first 4 offset windows (levels 0-3) so every step/run is comparable
    if "offset" in df.columns:
        df = df[num(df["offset"]) <= 3 * 27]
    df = converged(df)
    df["chunk"] = num(df["chunk"])
    gt = df[df.branch == "gt"]
    fl = df[df.branch == "flip"]
    fl_settled = fl[fl.chunk >= 2]
    g_gt = make_figure(gt, "GT (real commands)", "analysis/response_gain_gt.png")
    g_fl = make_figure(fl, "FLIP (counterfactual, all chunks)", "analysis/response_gain_flip.png")
    g_fs = make_figure(fl_settled, "FLIP settled (chunks 2+; drop decel/reverse transient)",
                       "analysis/response_gain_flip_settled.png")
    rows = []
    for run, (lab, _) in RUNS.items():
        rows.append(dict(run=run, label=lab,
                         gt_throttle=round(g_gt.get((run, 0), np.nan), 3), gt_steer=round(g_gt.get((run, 1), np.nan), 3),
                         flip_throttle=round(g_fl.get((run, 0), np.nan), 3), flip_steer=round(g_fl.get((run, 1), np.nan), 3),
                         flipset_throttle=round(g_fs.get((run, 0), np.nan), 3), flipset_steer=round(g_fs.get((run, 1), np.nan), 3)))
    tbl = pd.DataFrame(rows); tbl.to_csv("analysis/response_gain_table.csv", index=False)
    print("\n=== RESPONSE GAIN (last %d steps) — gt / flip / flip-settled ===" % LASTN)
    print(tbl.to_string(index=False))


if __name__ == "__main__":
    main()
