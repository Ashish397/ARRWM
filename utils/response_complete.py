"""COMPLETE throttle response figure: counterfactual on both halves of the axis.

Negative half: FLIP branch of the training-time control tests (real ride
actions sign-flipped; rides are ~all forward, so flips are ~all backward).
Positive half: phase-S stationary-seed forward sweep (commanded throttle
0.1..0.8; the true continuation is "stay still", so following is counter-
factual, not continuation mimicry). Both halves drop the first 2 generated
chunks (decel / accel transient) — "settled" convention.

Sources: analysis/chunk_metrics.csv (flip) + logs/eval_final/S/<run>/
control_test/metrics_r*.jsonl (sweep). Writes analysis/response_complete.png
+ response_complete_table.csv.
"""
import os, glob, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARR = os.environ.get("ARR_ROOT", "/scratch/u6ex/as1748.u6ex/ARRWM")
RUNS = {"pca8_8node": ("Default", "C1"), "16node": ("batch size 64", "C0"),
        "pca4": ("pca4", "C3"), "pca2": ("pca2", "C4"),
        "4node": ("batch size 16", "C2"), "noatok": ("no action tokens", "C5"),
        "noadaln": ("no AdaLN", "C6")}
LASTN = 1000
SETTLE = 2                      # drop first N generated chunks on both halves
MIN_BIN_N = 12


def flip_binned(run, d=0, extra=None):
    df = pd.read_csv(f"{ARR}/analysis/chunk_metrics.csv")
    df = df[(df.run == run) & (df.branch == "flip")]
    if "offset" in df.columns:
        df = df[pd.to_numeric(df["offset"], errors="coerce") <= 3 * 27]
    if not len(df):
        return np.array([]), np.array([]), np.array([])
    df = df[df.step >= df.step.max() - LASTN]
    df = df[pd.to_numeric(df["chunk"]) >= SETTLE]
    x = pd.to_numeric(df[f"cmd{d}"], errors="coerce").values
    y = pd.to_numeric(df[f"g{d}"], errors="coerce").values
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    if extra is not None and len(extra[0]):   # pooled extra (cmd, realized) points
        x = np.concatenate([x, extra[0]]); y = np.concatenate([y, extra[1]])
    bins = np.linspace(np.quantile(x, 0.02), np.quantile(x, 0.98), 13)
    idx = np.digitize(x, bins)
    bx, by, be = [], [], []
    for bi in range(1, len(bins)):
        m = idx == bi
        if m.sum() >= MIN_BIN_N:
            bx.append(x[m].mean()); by.append(y[m].mean()); be.append(y[m].std() / np.sqrt(m.sum()))
    return np.array(bx), np.array(by), np.array(be)


def fl_points(run):
    """Phase-FL flip-eval (mined moderate-left held-out windows): raw settled
    (commanded, realized) steer chunk pairs. Fills the sparse +0.35..+0.8 band."""
    xs, ys = [], []
    for f in glob.glob(f"{ARR}/logs/eval_final/FL/{run}/control_test/metrics_r*.jsonl"):
        for ln in open(f):
            try:
                d = json.loads(ln)
            except Exception:
                continue
            v = d.get("FL")
            if isinstance(v, dict):
                xs += v["cz7"][SETTLE:]
                ys += v["tz7"][SETTLE:]
    return np.array(xs, float), np.array(ys, float)


def sweep_points(run):
    rows = []
    for f in glob.glob(f"{ARR}/logs/eval_final/S/{run}/control_test/metrics_r*.jsonl"):
        for ln in open(f):
            try:
                d = json.loads(ln)
            except Exception:
                continue
            for k, v in d.items():
                if not (isinstance(v, dict) and k.startswith("F")):
                    continue
                m = int(k[1:]) / 100.0
                rows += [(m, tz) for tz in v["tz2"][SETTLE:]]
    if not rows:
        return np.array([]), np.array([]), np.array([])
    df = pd.DataFrame(rows, columns=["cmd", "g0"])
    # cap samples/point at the largest FLIP bin (132) so both halves of the
    # figure carry comparable sample sizes; fixed seed -> reproducible
    rng = np.random.default_rng(0)
    df = df.groupby("cmd", group_keys=False).apply(
        lambda g: g if len(g) <= 132 else g.sample(132, random_state=0))
    g = df.groupby("cmd").g0.agg(["mean", "sem", "count"])
    print("[resp] sweep samples/point:", dict(g["count"]))
    return g.index.values, g["mean"].values, g["sem"].values


def main():
    fig, ax = plt.subplots(figsize=(11, 7.5))
    lim = 0.85
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.2, label="y=x (gain 1)")
    ax.axhline(0, color="black", lw=1.2); ax.axvline(0, color="gray", lw=0.5)
    ax.axvspan(0.5, lim, color="gray", alpha=0.08)
    # dataset-observed action range (dotted horizontals, same as response_gain)
    ax.axhline(0.5, color="#555", ls=":", lw=1.4)
    ax.axhline(-0.3, color="#555", ls=":", lw=1.4)
    ax.text(0.01, 0.5, " dataset max forward (+0.5)", transform=ax.get_yaxis_transform(),
            ha="left", va="bottom", fontsize=9, color="#555")
    ax.text(0.01, -0.3, " dataset max backward (−0.3)", transform=ax.get_yaxis_transform(),
            ha="left", va="top", fontsize=9, color="#555")
    tab = []
    for run, (lab, col) in RUNS.items():
        fx, fy, fe = flip_binned(run)
        sx, sy, se = sweep_points(run)
        lw = 2.6 if run == "pca8_8node" else 1.5
        # one continuous solid line per model: concatenate flip + sweep halves
        x = np.concatenate([fx, sx]); y = np.concatenate([fy, sy]); e = np.concatenate([fe, se])
        o = np.argsort(x)
        if len(x):
            ax.plot(x[o], y[o], marker="o", ms=4, lw=lw, color=col, ls="-", label=lab)
            ax.fill_between(x[o], y[o] - e[o], y[o] + e[o], color=col, alpha=0.3, lw=0)
        if len(sx):
            tab.append(dict(run=run, **{f"m{m:.1f}": round(v, 3) for m, v in zip(sx, sy)}))
    ax.set_xlabel("commanded throttle")
    ax.set_ylabel("realized throttle (frozen CoTracker→PCA teacher)")
    ax.set_xlim(-0.75, lim); ax.set_ylim(-0.75, 0.75)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.legend(fontsize=14); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{ARR}/analysis/response_complete.png", dpi=130)
    if tab:
        pd.DataFrame(tab).to_csv(f"{ARR}/analysis/response_complete_table.csv", index=False)
        print(pd.DataFrame(tab).to_string(index=False))
    print(f"[resp] saved {ARR}/analysis/response_complete.png")

    # steer: FLIP alone covers both signs (real rides steer both ways), no sweep
    fig, ax = plt.subplots(figsize=(11, 7.5))
    slim = 1.05
    ax.plot([-slim, slim], [-slim, slim], "k--", lw=1.2, label="y=x (gain 1)")
    ax.axhline(0, color="black", lw=1.2); ax.axvline(0, color="gray", lw=0.5)
    ax.axhline(0.98, color="#555", ls=":", lw=1.4)
    ax.axhline(-1.0, color="#555", ls=":", lw=1.4)
    ax.text(0.01, 0.98, " dataset max right (+0.98)", transform=ax.get_yaxis_transform(),
            ha="left", va="bottom", fontsize=9, color="#555")
    ax.text(0.01, -1.0, " dataset max left (−1.0)", transform=ax.get_yaxis_transform(),
            ha="left", va="top", fontsize=9, color="#555")
    for run, (lab, col) in RUNS.items():
        fx, fy, fe = flip_binned(run, d=1, extra=fl_points(run))
        lw = 2.6 if run == "pca8_8node" else 1.5
        if len(fx):
            ax.plot(fx, fy, marker="o", ms=4, lw=lw, color=col, ls="-", label=lab)
            ax.fill_between(fx, fy - fe, fy + fe, color=col, alpha=0.3, lw=0)
    ax.set_xlabel("commanded steer")
    ax.set_ylabel("realized steer (frozen CoTracker→PCA teacher)")
    ax.set_xlim(-slim, slim); ax.set_ylim(-slim, slim)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.legend(fontsize=14); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{ARR}/analysis/response_complete_steer.png", dpi=130)
    print(f"[resp] saved {ARR}/analysis/response_complete_steer.png")


if __name__ == "__main__":
    main()
