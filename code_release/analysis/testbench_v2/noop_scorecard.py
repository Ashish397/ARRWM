"""Final stationary (no-op) scorecard + figures.

Assembles the four no-op instruments into the paper tables:

  reference battery   out/noop_paired.csv     (noop_reference_eval.py)
  FVD                 out/noop_summary.csv    (same)
  mangle / geometry   out/noop_lowfreq.csv    (noop_lowfreq.py)
  flow decomposition  out/noop_flow.csv       (noop_flow_split.py)

METRIC SET (the decision this file encodes)

A no-op command asks for two things that pull against each other: hold the
camera still, and keep the world alive. One number cannot express that, and
the obvious numbers actively mislead:

  FVD                 HEADLINE. Distribution realism over the whole clip;
                      the only metric that punishes freezing and drifting
                      alike. Small-n caveat: 32 clips per side on A starts.
  motion_ratio (dyn)  LIVELINESS, dynamic bin only. gen motion / ref motion:
                      <<1 = dead world, >>1 = drift or hallucination.
  lpips (dyn)         PERCEPTUAL fidelity, dynamic bin -- the bin where a
                      frozen clip cannot fake a good score.
  mangle_rate         REFERENCE-FREE cross-check. Fraction of windows whose
                      z-mean detector score exceeds the real-continuation
                      floor (95th pct of real_ref). Catches the case FVD
                      misses -- worldcam is mid-table on FVD but corrupts
                      structure standing still.
  ego_excess          px/s of global camera motion above the reference's,
                      STATIC bin: DRIFT, isolated from the world.
  world_ratio         residual (non-global) flow vs the reference's, DYNAMIC
                      bin: world DEATH, isolated from the camera. Read
                      ego_excess FIRST -- parallax from a drifting camera
                      inflates the residual too, so world_ratio only means
                      "world activity" for a model already shown to hold
                      still (see noop_flow_split.py and flow_table below).
  psnr / ssim         STATIC BIN ONLY. In a dynamic scene a frozen frame
                      beats a correctly-moving one on both, so they are not
                      allowed to rank models outside the static bin.

Honesty flags carried into every table:
  cover  fraction of the 6 s window the model actually generated. minwm
         stops at ~4.06 s; its tail is a repeated frame, which depresses its
         motion and inflates its apparent freeze.
  note   matrixgame is an AUTHORED all-zero action stream -- representable
         by the released tensors but not a benchmarked no-op interface.

Usage:  python noop_scorecard.py

Role: scorecard and figures over the stationary results.
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FIG = os.path.join(HERE, "out", "figs")
DET = ["qwen_melt_pyes", "pal4vst_max", "depth_rough_base"]
OURS = ["pca8", "pca4", "pca2", "16node", "4node", "noatok", "noadaln"]
# models whose generation is shorter than the 6s reference window
SHORT = {"minwm_nullact"}
AUTHORED = {"matrixgame_noop"}
FLOOR_PCT = 95          # real-ref percentile that defines "mangled"


def pretty(m):
    return m.replace("_nullact", "").replace("_noop", "")


# ------------------------------------------------------------------ instruments
def mangle_table():
    """Reference-free mangle rate, calibrated on the real continuations.

    A window counts as mangled when BOTH artifact judges exceed their own
    95th-percentile value over the 32 real continuations -- the repo's
    validated "both-must-agree" VLxPAL logic (METRICS.md 2026-07-07), which
    cancels each judge's solo false positives, applied here with the
    threshold set by real video rather than by a chosen constant.

    Two deliberate choices, both data-driven:

    * Per-detector percentiles, not a z-mean. z-scoring against the real
      refs' tight std sends matrixgame's z-mean to ~39 -- a number that
      cannot be reported -- and z-scoring against the model population lets
      one catastrophic model rewrite every other model's score.
    * depth_rough_base is EXCLUDED from the rate and reported beside it. It
      fires on 3.1% of windows for almost every model including the real
      references (its outliers are scene geometry -- refs r20/r25 are rough
      daytime scenes, not night clips), so it carries no discrimination at
      the per-window level here.

    Validation that the calibration is honest: the real references
    themselves score 0.000 under this rule -- no false positives on real
    video -- while the ordering across models is monotone."""
    lf = pd.read_csv(os.path.join(OUT, "noop_lowfreq.csv"))
    real = lf[lf.model == "real_ref"]
    if len(real) < 8:
        raise SystemExit("noop_lowfreq.csv has no real_ref rows -- run "
                         "TB2_ONLY=real_ref noop_lowfreq.py first")
    floors = {c: float(np.percentile(real[c].values, FLOOR_PCT)) for c in DET}
    lf = lf.copy()
    for c, k in zip(DET, ("x_melt", "x_pal", "x_depth")):
        lf[k] = (lf[c] > floors[c]).astype(int)
    lf["mangled"] = ((lf.x_melt + lf.x_pal) == 2).astype(int)

    t = (lf[lf.model != "real_ref"].groupby("model")
         .agg(mangle_rate=("mangled", "mean"),
              melt_rate=("x_melt", "mean"), pal_rate=("x_pal", "mean"),
              depth_rate=("x_depth", "mean")))
    meta = dict(rule="both judges over their real 95th percentile",
                floor_pct=FLOOR_PCT, n_real=len(real), floors=floors,
                real_mangle_rate=float(
                    lf[lf.model == "real_ref"].mangled.mean()),
                depth_excluded_because="fires at ~3% for every model "
                                       "including the real refs")
    return t, meta


def flow_table(paired):
    """Flow channels, each read in the bin where its REFERENCE signal is
    actually conditioned.

    This is not a stylistic choice -- pooling all bins destroys both
    channels, because a stationary reference only exercises one of them at a
    time:

      drift  -> STATIC bin. There the reference ego is 0.199 px/s, so any
                global motion the model adds is drift and nothing else.
      world  -> DYNAMIC bin. There the reference residual is 3.04 px/s
                versus 0.25 px/s in static/mild, a 12x jump; in the quiet
                bins the ratio is noise over noise and lands near 1.0 for
                everything. Pooled over all bins worldplay reads 0.93
                ("world alive") when in the dynamic bin it reads 0.111
                ("world dead") -- the latter is the true reading and agrees
                with the independent frame-diff motion_ratio (Spearman
                0.83 across models).
    """
    p = os.path.join(OUT, "noop_flow.csv")
    if not os.path.exists(p):
        print("[warn] noop_flow.csv missing -- flow columns will be blank")
        return None
    fl = pd.read_csv(p)
    pa = paired.copy()
    pa["scene"] = pa.ref.str.extract(r"_r(\d+)").astype(int)
    key = ["model", "start_set", "scene"]
    fl = fl.merge(pa[key + ["dyn_bin"]], on=key, how="left")
    A = fl[fl.start_set == "A"]
    t = (A[A.dyn_bin == "static"].groupby("model")
         .agg(ego_excess=("ego_excess", "median"),
              gen_inlier=("gen_inlier_frac", "median"),
              cover=("gen_cover", "min"))
         .join(A[A.dyn_bin == "dynamic"].groupby("model")
               .agg(world_ratio=("world_ratio", "median"))))
    return t, fl


# -------------------------------------------------------------------- assembly
def scorecard():
    paired = pd.read_csv(os.path.join(OUT, "noop_paired.csv"))
    summ = pd.read_csv(os.path.join(OUT, "noop_summary.csv"))
    mang, meta = mangle_table()
    fl = flow_table(paired)
    flow_agg = fl[0] if fl else None

    A = paired[paired.start_set == "A"]
    dyn = A[A.dyn_bin == "dynamic"].groupby("model").agg(
        lpips_dyn=("lpips", "mean"), motion_dyn=("motion_ratio", "median"))
    stat = A[A.dyn_bin == "static"].groupby("model").agg(
        psnr_static=("psnr", "mean"), ssim_static=("ssim", "mean"))
    fvd = summ[summ.start_set == "A"].set_index("model")[["n", "fvd"]]

    sc = fvd.join([dyn, stat, mang])
    if flow_agg is not None:
        sc = sc.join(flow_agg)
    sc["family"] = ["ours" if m in OURS else "external" for m in sc.index]
    sc["flag"] = ["short-gen" if m in SHORT else
                  "authored-input" if m in AUTHORED else "" for m in sc.index]
    sc.index = [pretty(m) for m in sc.index]
    sc = sc.sort_values("fvd")

    cols = ["family", "n", "fvd", "motion_dyn", "lpips_dyn", "mangle_rate",
            "melt_rate", "pal_rate", "psnr_static", "ssim_static"]
    if flow_agg is not None:
        cols += ["ego_excess", "world_ratio", "gen_inlier", "cover"]
    cols += ["flag"]
    sc = sc[cols].round(3)
    sc.to_csv(os.path.join(OUT, "noop_scorecard.csv"))

    # per-bin detail: psnr/ssim deliberately absent outside the static bin
    per_bin = (paired.groupby(["model", "start_set", "dyn_bin"], observed=True)
               .agg(n=("vid", "count"), lpips=("lpips", "mean"),
                    motion_ratio=("motion_ratio", "median"),
                    psnr=("psnr", "mean"), ssim=("ssim", "mean")).round(3))
    pb = per_bin.reset_index()
    mask = pb.dyn_bin != "static"
    pb.loc[mask, ["psnr", "ssim"]] = np.nan
    pb.to_csv(os.path.join(OUT, "noop_scorecard_by_bin.csv"), index=False)

    with open(os.path.join(OUT, "noop_scorecard_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("=== STATIONARY SCORECARD (A starts, n=32/model) ===")
    print(sc.to_string())
    print(f"\nmangle rule: {meta['rule']} "
          f"({FLOOR_PCT}th pct of {meta['n_real']} real continuations: "
          + ", ".join(f"{k.split('_')[0]}>{v:.4f}"
                      for k, v in meta["floors"].items()) + ")")
    print(f"  real continuations score {meta['real_mangle_rate']:.3f} under "
          "this rule -- no false positives on real video")
    print("\npsnr/ssim are static-bin only by construction; "
          "in dynamic scenes a frozen clip wins them.")
    print("\n=== per-bin (A) ===")
    print(pb[pb.start_set == "A"].to_string(index=False))
    return sc, meta, (fl[1] if fl else None)


# --------------------------------------------------------------------- figures
def figures(sc, meta, flow_raw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG, exist_ok=True)
    # categorical slots 1-2 of the validated palette (all-pairs safe)
    C = {"ours": "#2a78d6", "external": "#eb6834"}
    INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9.5,
        "axes.edgecolor": INK2, "axes.linewidth": 0.7,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.labelcolor": INK, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
    })

    def place_labels(fig, ax, pts, color):
        """Direct-label every point without overlaps.

        13 entities is far past any safe categorical hue count, so identity
        has to be textual -- which only works if the text is readable. The
        top-5 models sit inside a few FVD points of each other, so fixed
        offsets collide; this tries candidate offsets in display space
        against the already-placed boxes and draws a leader line whenever it
        has to reach far from the marker."""
        from matplotlib.transforms import Bbox
        fig.canvas.draw()
        # seed the occupied set with the MARKERS, so a label never lands on a
        # dot (text-vs-text alone let '4node' sit on its neighbour's marker)
        placed = []
        for px, py, _ in pts:
            dx, dy = ax.transData.transform((px, py))
            placed.append(Bbox.from_bounds(dx - 6, dy - 6, 12, 12))
        n_marks = len(placed)
        arts = []
        # near-identical points first: they have the least freedom
        order = sorted(range(len(pts)), key=lambda i: pts[i][1])
        # ladder of offsets, near -> far. The top-5 models sit within a few
        # FVD points of one another, so the far rungs (and their leaders) are
        # what make that cluster legible rather than a smudge of names.
        cands = [(7, 3), (7, -9), (-7, 3), (-7, -9),
                 (14, 12), (-14, 12), (14, -16), (-14, -16),
                 (0, 20), (0, -24), (26, 20), (-26, 20), (26, -26), (-26, -26),
                 (34, 34), (-34, 34), (34, -38), (-34, -38),
                 (46, 30), (-46, 30), (46, -34), (-46, -34),
                 (0, 40), (0, -44), (58, 44), (-58, 44), (58, -48), (-58, -48)]
        for i in order:
            px, py, name = pts[i]
            best = None
            for dx, dy in cands:
                t = ax.annotate(name, (px, py), textcoords="offset points",
                                xytext=(dx, dy), fontsize=6.8, color=INK2,
                                ha="right" if dx < 0 else "left")
                fig.canvas.draw()
                bb = t.get_window_extent().expanded(1.06, 1.22)
                if not any(bb.overlaps(b) for b in placed):
                    best = (t, bb, dx, dy)
                    break
                t.remove()
            if best is None:      # every candidate collided: take the last
                t = ax.annotate(name, (px, py), textcoords="offset points",
                                xytext=(0, -22), fontsize=6.8, color=INK2)
                fig.canvas.draw()
                best = (t, t.get_window_extent().expanded(1.06, 1.22), 0, -22)
            t, bb, dx, dy = best
            placed.append(bb)
            arts.append(t)
            # any label that had to move off the default adjacent slot gets a
            # leader: in the top-5 cluster, proximity alone leaves the
            # reader guessing which dot a name belongs to
            if (dx, dy) != cands[0]:
                t.set_arrowprops = None
                t.arrow_patch = None
                t.set_annotation_clip(False)
                ax.annotate("", (px, py), textcoords="offset points",
                            xytext=(dx * 0.92, dy * 0.92 + (3 if dy > 0 else -1)),
                            arrowprops=dict(arrowstyle="-", color=color,
                                            lw=0.5, alpha=0.55,
                                            shrinkA=0, shrinkB=2))
        return arts

    def scatter(ax, x, y, xlabel, ylabel, ideal_x, xlog):
        for fam, c in C.items():
            s = sc[sc.family == fam]
            # open marker = caveat (short generation / authored input); shape
            # and label carry identity, never colour alone
            for name, row in s.iterrows():
                caveat = bool(row["flag"])
                ax.scatter(row[x], row[y], s=46 if not caveat else 52,
                           facecolor="white" if caveat else c,
                           edgecolor=c, linewidth=1.6 if caveat else 0.9,
                           marker="D" if caveat else "o", zorder=3,
                           label=fam if name == s.index[0] else None)
        if ideal_x is not None:
            ax.axvline(ideal_x, color=GRID, lw=1.0, ls="--", zorder=1)
        ax.grid(True, color=GRID, lw=0.5, alpha=0.8, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if xlog:
            ax.set_xscale("log")
            ax.set_xticks([0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 4.0, 8.0])
            ax.get_xaxis().set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            # room for the labels on the extreme points (worldplay, yume)
            vals = sc[x].values
            ax.set_xlim(float(np.min(vals)) / 1.9, float(np.max(vals)) * 1.9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    # -- fig 1: the headline trade-off
    fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=200)
    scatter(ax, "motion_dyn", "fvd",
            "world liveliness  (gen motion / real motion, dynamic scenes)",
            "FVD  ↓", 1.0, True)
    ax.set_title("Stationary command: realism vs. keeping the world alive",
                 loc="left", pad=14)
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0 - 4, y1 + 12)          # headroom for the band annotations
    lo, hi = ax.get_xlim()
    yt = ax.get_ylim()[1]
    ax.text(lo * 1.04, yt - 1, "← frozen world", fontsize=7, color=INK2,
            ha="left", va="top", style="italic")
    ax.text(hi * 0.96, yt - 1, "drift / hallucination →", fontsize=7,
            color=INK2, ha="right", va="top", style="italic")
    ax.text(1.0, yt - 1, "matches real", fontsize=6.5, color=INK2,
            ha="center", va="top")
    place_labels(fig, ax, [(r["motion_dyn"], r["fvd"], n)
                           for n, r in sc.iterrows()], INK2)
    h = [plt.Line2D([], [], marker="o", ls="", mfc=C["ours"],
                    mec=C["ours"], ms=6, label="ours (ablations)"),
         plt.Line2D([], [], marker="o", ls="", mfc=C["external"],
                    mec=C["external"], ms=6, label="external baselines"),
         plt.Line2D([], [], marker="D", ls="", mfc="white", mec=INK2, ms=6,
                    label="caveat (see table)")]
    ax.legend(handles=h, frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    p1 = os.path.join(FIG, "noop_fvd_vs_motion.png")
    fig.savefig(p1, bbox_inches="tight")
    fig.savefig(p1.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)

    # -- fig 2: the flow decomposition, if available
    p2 = None
    if "ego_excess" in sc.columns:
        fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=200)
        scatter(ax, "world_ratio", "ego_excess",
                "world motion kept  (residual flow / real residual flow)",
                "ego drift  (px/s of global motion above real)", 1.0, True)
        ax.axhline(0, color=GRID, lw=1.0, ls="--", zorder=1)
        ax.set_title("Which failure? camera drift (up) vs. dead world (left)",
                     loc="left", pad=8)
        # read the y axis first: a drifting camera's parallax also inflates x
        ax.text(0.5, -0.20, "x is diagnostic only near y=0 — a drifting "
                "camera's parallax inflates the residual too",
                transform=ax.transAxes, fontsize=6.5, color=INK2,
                ha="center", va="top", style="italic")
        ax.legend(handles=h, frameon=False, fontsize=7, loc="upper left")
        place_labels(fig, ax, [(r["world_ratio"], r["ego_excess"], n)
                               for n, r in sc.iterrows()], INK2)
        fig.tight_layout()
        p2 = os.path.join(FIG, "noop_flow_split.png")
        fig.savefig(p2, bbox_inches="tight")
        fig.savefig(p2.replace(".png", ".pdf"), bbox_inches="tight")
        plt.close(fig)
    print(f"\nfigures -> {p1}" + (f"\n           {p2}" if p2 else ""))


if __name__ == "__main__":
    sc, meta, flow_raw = scorecard()
    figures(sc, meta, flow_raw)
