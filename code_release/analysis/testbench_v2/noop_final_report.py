"""Report stage for the final stationary evaluation: thresholds, bins,
paired scene-bootstrap, the behaviour-plane figure, tables and NOOP_FINAL.md.

Everything here is downstream of noop_final_eval.py's flow/paired stages; all
constants come from that module so the write-up cannot drift from the run.

Role: reporting stage over noop_final_eval's outputs.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import noop_final_eval as N  # noqa: E402

OUT, FIG = N.OUT, N.FIG
ARR = N.ARR
OURS = list(N.ABLATIONS.values())
PRETTY = {"real": "Real", "freeze": "Freeze", "minwm": "MinWM",
          "worldcam": "WorldCam", "worldplay": "WorldPlay", "yume": "Yume",
          "astra": "Astra", "matrixgame": "Matrix-Game$^{a}$"}
ORDER = ["real", "freeze"] + OURS + N.EXT
# supplementary drift-contamination bar (see the rule in main()); the
# specified Held<50 / medD>2*tau_D rule fires for nobody on this data
HELD_CONTAM_PCT = 90.0


def load_flow():
    parts = sorted(__import__("glob").glob(os.path.join(OUT, "noop_final_flow*.csv")))
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    return df.drop_duplicates(["system", "scene"])


# --------------------------------------------------------------- bootstrap
def boot_idx(n, rng):
    return rng.integers(0, n, size=(N.N_BOOT, n))


def ci(v):
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    os.makedirs(FIG, exist_ok=True)
    flow = load_flow()
    missing = flow[flow.missing.astype(bool)] if "missing" in flow else flow.iloc[:0]
    ok = flow[~flow.missing.astype(bool)].copy()

    # ---- tau_D from real continuations only (Phase-A + Phase-B, same gate)
    real_all = ok[ok.system.isin(["real", "real_B"])]
    tau_D = float(np.percentile(real_all.D_affine.values, N.TAU_PCT))

    # ---- activity bins from the 32 Phase-A real continuations only
    realA = ok[ok.system == "real"].set_index("scene").sort_index()
    M_real = realA[f"M{int(N.Q_MAIN * 100)}_affine"]
    t1, t2 = np.percentile(M_real.values, [100 / 3, 200 / 3])
    bins = pd.cut(M_real, [-np.inf, t1, t2, np.inf],
                  labels=["static", "mild", "dynamic"])
    dyn_scenes = sorted(bins[bins == "dynamic"].index.tolist())
    pd.DataFrame({"scene": M_real.index, "M_real_px_s": M_real.values,
                  "bin": bins.values,
                  "D_real_px_s": realA.D_affine.values}).to_csv(
        os.path.join(OUT, "noop_activity_bins.csv"), index=False)

    # ---- paired scene bootstrap: identical resamples for every system
    rng = np.random.default_rng(N.BOOT_SEED)
    bi_all = boot_idx(N.N_SCENES, rng)              # Held-Still, 32 scenes
    bi_dyn = boot_idx(len(dyn_scenes), rng)         # Ratio, dynamic scenes

    lp_path = os.path.join(OUT, "noop_final_lpips.csv")
    lp = pd.read_csv(lp_path) if os.path.exists(lp_path) else None
    fz = os.path.join(OUT, "noop_final_feats.npz")
    feats = np.load(fz) if os.path.exists(fz) else None

    def frechet(a, b):
        from scipy import linalg
        mu1, mu2 = a.mean(0), b.mean(0)
        s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
        cm = linalg.sqrtm(s1 @ s2)
        if np.iscomplexobj(cm):
            cm = cm.real
        return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * cm))

    Q = int(N.Q_MAIN * 100)
    rows, per_scene, sens = [], [], []
    for sysname in ORDER:
        s = ok[ok.system == sysname].set_index("scene").sort_index()
        if not len(s):
            continue
        scenes = s.index.values
        D = s.D_affine.reindex(range(N.N_SCENES)).values
        held = (D <= tau_D).astype(float)
        hs = 100.0 * np.nanmean(held)
        hs_b = 100.0 * np.nanmean(held[bi_all], axis=1)

        Mg = s[f"M{Q}_affine"].reindex(dyn_scenes).values
        Mr = M_real.reindex(dyn_scenes).values
        r = (Mg + N.EPSILON) / (Mr + N.EPSILON)
        ratio = float(np.exp(np.mean(np.log(r))))
        lr = np.log(r)
        ratio_b = np.exp(np.mean(lr[bi_dyn], axis=1))

        med_D = float(np.nanmedian(D))
        # Specified rule. On this data it fires for NOBODY: tau_D is the 95th
        # percentile of real D, so it is already a lax bar (real windows do
        # contain some genuine camera motion), and 2*tau_D = 50.5 px/s is
        # past every model's median. Recorded separately, not silently
        # widened.
        cont_spec = bool(hs < 50.0 or med_D > 2 * tau_D)
        # Supplementary bar, needed because the drifters must not be credited
        # with world liveliness (they sit at ratio 1.5-2.4): a model that
        # fails the real-calibrated stationarity test on more than 10% of
        # scenes, i.e. at least twice the ~5% failure rate the real
        # continuations show by construction. This separates cleanly here --
        # 59-75% for the drifters vs 96.9-100% for everything else.
        cont_held = bool(hs < HELD_CONTAM_PCT)
        contaminated = bool(cont_spec or cont_held)

        rec = dict(system=sysname, held_still=hs,
                   held_lo=ci(hs_b)[0], held_hi=ci(hs_b)[1],
                   ratio=ratio, ratio_lo=ci(ratio_b)[0], ratio_hi=ci(ratio_b)[1],
                   median_D=med_D, mean_D=float(np.nanmean(D)),
                   contaminated=contaminated,
                   contaminated_spec_rule=cont_spec,
                   contaminated_held_rule=cont_held,
                   n_scenes=int(np.isfinite(D).sum()),
                   inlier_affine=float(s.inl_affine.mean()),
                   snap_err_ms=float(s.snap_err.max() * 1000))

        # FVD (R3D-18) against this system's identically-cropped references
        if feats is not None and sysname in feats:
            rec["fvd"] = round(frechet(feats[sysname],
                                       feats[f"__ref__{sysname}"]), 1)
        # LPIPS on the reference-dynamic bin, bootstrapped by scene
        if lp is not None:
            l = lp[lp.system == sysname].set_index("scene").lpips.reindex(dyn_scenes).values
            rec["lpips_dyn"] = float(np.nanmean(l))
            lb = np.nanmean(l[bi_dyn], axis=1)
            rec["lpips_lo"], rec["lpips_hi"] = ci(lb)
        rows.append(rec)

        for sc in scenes:
            per_scene.append(dict(
                system=sysname, scene=int(sc), bin=str(bins.get(sc, "")),
                D_affine=float(s.D_affine[sc]), held=bool(s.D_affine[sc] <= tau_D),
                M_affine=float(s[f"M{Q}_affine"][sc]),
                M_real=float(M_real.get(sc, np.nan)),
                ratio=float((s[f"M{Q}_affine"][sc] + N.EPSILON)
                            / (M_real.get(sc, np.nan) + N.EPSILON)),
                inlier_affine=float(s.inl_affine[sc]),
                D_homog=float(s.D_homog[sc]), M90_homog=float(s.M90_homog[sc])))

        # ---- sensitivity: quantile choice, affine vs homography, robust stats
        srow = dict(system=sysname, n_dynamic=len(dyn_scenes),
                    ratio_geomean_Q90=ratio,
                    ratio_median=float(np.median(r)),
                    ratio_arithmetic_mean=float(np.mean(r)),
                    ratio_q25=float(np.percentile(r, 25)),
                    ratio_q75=float(np.percentile(r, 75)))
        for q in N.Q_SENS:
            mg = s[f"M{int(q * 100)}_affine"].reindex(dyn_scenes).values
            mr = ok[ok.system == "real"].set_index("scene")[
                f"M{int(q * 100)}_affine"].reindex(dyn_scenes).values
            srow[f"ratio_Q{int(q * 100)}"] = float(np.exp(np.mean(
                np.log((mg + N.EPSILON) / (mr + N.EPSILON)))))
        mgh = s.M90_homog.reindex(dyn_scenes).values
        mrh = ok[ok.system == "real"].set_index("scene").M90_homog.reindex(dyn_scenes).values
        srow["ratio_Q90_homography"] = float(np.exp(np.mean(
            np.log((mgh + N.EPSILON) / (mrh + N.EPSILON)))))
        srow["D_affine_median"] = med_D
        srow["D_homography_median"] = float(np.nanmedian(s.D_homog.values))
        sens.append(srow)

    res = pd.DataFrame(rows).set_index("system").reindex(
        [s for s in ORDER if s in set(r["system"] for r in rows)])
    pd.DataFrame(per_scene).to_csv(os.path.join(OUT, "noop_per_scene.csv"),
                                   index=False)
    pd.DataFrame(sens).to_csv(os.path.join(OUT, "noop_sensitivity.csv"),
                              index=False)

    meta = dict(tau_D=tau_D, n_real_for_tau=len(real_all),
                dyn_scenes=dyn_scenes, bin_edges=(float(t1), float(t2)),
                missing=missing)
    tables(res, meta)
    figure(res, meta)
    failure_cases(res, meta, per_scene)
    write_md(res, meta)
    print(res.round(3).to_string())
    return res, meta


# ------------------------------------------------------------------- tables
def tables(res, meta):
    cols = ["held_still", "held_lo", "held_hi", "ratio", "ratio_lo", "ratio_hi",
            "fvd", "lpips_dyn", "median_D", "contaminated"]
    t = res[[c for c in cols if c in res.columns]].copy()
    t.to_csv(os.path.join(OUT, "noop_main_table.csv"))

    def nm(s):
        p = PRETTY.get(s, s)
        return p + (r"$^{\dagger}$" if res.contaminated.get(s, False) else "")

    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Model & Held-Still (\%) $\uparrow$ & Residual ratio "
             r"($\leftrightarrow 1$) & FVD (R3D-18) $\downarrow$ & "
             r"LPIPS$_{\mathrm{dyn}}$ $\downarrow$ \\", r"\midrule"]
    for s, r in res.iterrows():
        fvd = "--" if s == "real" else (f"{r.fvd:.1f}" if "fvd" in r and
                                        np.isfinite(r.fvd) else "--")
        lp = f"{r.lpips_dyn:.3f}" if "lpips_dyn" in r and np.isfinite(r.lpips_dyn) else "--"
        if s == "real":
            lp = "--"
        # {\tiny ...} -- \tiny is a size DECLARATION, not a one-argument macro
        lines.append(
            f"{nm(s)} & {r.held_still:.1f} "
            f"{{\\tiny [{r.held_lo:.0f}, {r.held_hi:.0f}]}} & "
            f"{r.ratio:.2f} {{\\tiny [{r.ratio_lo:.2f}, {r.ratio_hi:.2f}]}} & "
            f"{fvd} & {lp} \\\\")
        if s == "freeze":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", "",
              r"\footnotesize $^{\dagger}$Residual ratio is \emph{not} "
              r"evidence of preserved independent world motion for these "
              r"models: camera parallax may enter the residual flow after "
              r"global-motion subtraction. Read it as apparent residual "
              r"motion only.",
              r"\footnotesize $^{a}$Matrix-Game is driven by an authored "
              r"all-zero action stream supported by the released tensors, "
              r"not an official benchmarked no-op interface.",
              r"\footnotesize Real is the reference set itself, so its FVD "
              r"and LPIPS are degenerate (0 by construction) and omitted."]
    open(os.path.join(OUT, "noop_main_table.tex"), "w").write("\n".join(lines))


# ------------------------------------------------------------------- figure
def _place(fig, ax, pts, ink):
    """Collision-aware direct labels with leaders (16 entities => textual
    identity; hue can never carry this many)."""
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    placed = []
    for px, py, _ in pts:
        dx, dy = ax.transData.transform((px, py))
        placed.append(Bbox.from_bounds(dx - 7, dy - 7, 14, 14))
    cands = [(8, 3), (8, -10), (-8, 3), (-8, -10), (0, 12), (0, -16),
             (16, 14), (-16, 14), (16, -18), (-16, -18), (0, 24), (0, -28),
             (30, 22), (-30, 22), (30, -26), (-30, -26), (44, 30), (-44, 30),
             (44, -34), (-44, -34), (0, 38), (0, -42)]
    for px, py, name in sorted(pts, key=lambda p: -p[1]):
        best = None
        for dx, dy in cands:
            t = ax.annotate(name, (px, py), textcoords="offset points",
                            xytext=(dx, dy), fontsize=7.2, color=ink,
                            ha="right" if dx < 0 else "left", zorder=6)
            fig.canvas.draw()
            bb = t.get_window_extent().expanded(1.05, 1.25)
            if not any(bb.overlaps(b) for b in placed):
                best = (t, bb, dx, dy)
                break
            t.remove()
        if best is None:
            t = ax.annotate(name, (px, py), textcoords="offset points",
                            xytext=(0, -42), fontsize=7.2, color=ink, zorder=6)
            fig.canvas.draw()
            best = (t, t.get_window_extent().expanded(1.05, 1.25), 0, -42)
        t, bb, dx, dy = best
        placed.append(bb)
        if (dx, dy) != cands[0]:
            ax.annotate("", (px, py), textcoords="offset points",
                        xytext=(dx * 0.9, dy * 0.9 + (3 if dy > 0 else -2)),
                        arrowprops=dict(arrowstyle="-", color=ink, lw=0.5,
                                        alpha=0.5, shrinkA=0, shrinkB=3),
                        zorder=2)


def figure(res, meta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # categorical slots 1-3 of the validated palette (all-pairs safe)
    C_OURS, C_EXT, C_REF = "#2a78d6", "#eb6834", "#1baf7a"
    INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 9.5, "axes.titlesize": 10.5,
        "axes.edgecolor": INK2, "axes.linewidth": 0.7, "xtick.color": INK2,
        "ytick.color": INK2, "text.color": INK, "axes.labelcolor": INK,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42})

    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=200)
    ax.axhline(1.0, color=INK2, lw=1.0, ls="--", alpha=0.65, zorder=1)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    pts = []
    for s, r in res.iterrows():
        if s in ("real", "freeze"):
            col, mk, ms = C_REF, ("*" if s == "real" else "s"), (300 if s == "real" else 80)
            # Real and WorldCam land almost on top of each other; the anchor
            # must stay visible or the plane loses its reference point
            zs = 7 if s == "real" else 5
        else:
            col = C_OURS if s in OURS else C_EXT
            mk, ms, zs = "o", 62, 4
        # drift-contaminated: reduced opacity + dagger, because parallax can
        # inflate their residual ratio
        cont = bool(r.contaminated)
        al = 0.42 if cont else 1.0
        ax.errorbar(r.held_still, r.ratio,
                    xerr=[[r.held_still - r.held_lo], [r.held_hi - r.held_still]],
                    yerr=[[r.ratio - r.ratio_lo], [r.ratio_hi - r.ratio]],
                    fmt="none", ecolor=col, elinewidth=0.9, capsize=1.8,
                    alpha=al * 0.75, zorder=3)
        ax.scatter(r.held_still, r.ratio, s=ms, marker=mk, color=col,
                   edgecolor="white", linewidth=0.9, alpha=al, zorder=zs)
        pts.append((r.held_still, r.ratio,
                    PRETTY.get(s, s).replace("$^{a}$", "") + ("†" if cont else "")))

    ax.set_yscale("log")
    ax.set_xlim(-4, 108)
    ax.axvline(100, color=GRID, lw=1.0, ls="--", zorder=1)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
    ax.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Held-Still rollouts (%)")
    ax.set_ylabel("Residual scene motion / real continuation\n(1 = real)")
    ax.set_title("No-op control: stopping the camera without stopping the world",
                 loc="left", pad=10)

    # read-the-plane guides
    y0, y1 = ax.get_ylim()
    # placed in the empty left half, away from the dense x=100 cluster; the
    # y-axis label already says "1 = real", so no inline "ideal" tag
    ax.text(26, 1.06, "real level", fontsize=7, color=INK2, ha="center",
            va="bottom", style="italic")
    ax.text(40, y0 * 1.35, "world frozen ↓", fontsize=7, color=INK2,
            ha="center", va="bottom", style="italic")
    ax.text(26, y1 * 0.80, "← camera drifts", fontsize=7, color=INK2,
            ha="center", va="top", style="italic")
    # the y=1 line and the x=100 rule already fix the target corner; a
    # leader arrow to it only crosses the data

    h = [plt.Line2D([], [], marker="o", ls="", color=C_OURS, ms=7,
                    label="ours (ablations)"),
         plt.Line2D([], [], marker="o", ls="", color=C_EXT, ms=7,
                    label="external baselines"),
         plt.Line2D([], [], marker="*", ls="", color=C_REF, ms=13,
                    label="Real continuation"),
         plt.Line2D([], [], marker="s", ls="", color=C_REF, ms=6.5,
                    label="Freeze baseline"),
         plt.Line2D([], [], marker="o", ls="", color=INK2, ms=7, alpha=0.42,
                    label="† drift-contaminated")]
    ax.legend(handles=h, frameon=False, fontsize=7.2, loc="lower left",
              handletextpad=0.5, borderaxespad=0.8)
    _place(fig, ax, pts, INK2)
    fig.text(0.5, -0.035,
             "† residual ratio is apparent motion only — camera parallax can "
             "survive global-motion subtraction.  95% paired scene-bootstrap "
             f"CIs ({N.N_BOOT} resamples).", ha="center", fontsize=6.8,
             color=INK2, style="italic")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"noop_behaviour_plane.{ext}"),
                    bbox_inches="tight")
    import matplotlib.pyplot as _p
    _p.close(fig)
    print(f"figure -> {os.path.join(FIG, 'noop_behaviour_plane.pdf')}")


# ----------------------------------------------------------- failure cases
def failure_cases(res, meta, per_scene):
    """Frame strips for the four regimes the plane is meant to separate,
    plus the worst global-motion fit in the run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    ps = pd.DataFrame(per_scene)
    dyn = set(meta["dyn_scenes"])
    cand = res.drop(index=[i for i in ("real", "freeze") if i in res.index])
    picks = []

    # "ratio near 1" means nearest to 1 in log space among the models that
    # actually hold still -- not merely the stillest model in a wide band
    live = cand[(~cand.contaminated) & (cand.held_still >= HELD_CONTAM_PCT)]
    if len(live):
        s = live.index[np.argmin(np.abs(np.log(live.ratio.values)))]
        picks.append(("held still, world alive (ratio≈1)", s))
    dead = cand[(~cand.contaminated)].sort_values("ratio")
    if len(dead):
        picks.append(("held still, world dead (ratio→0)", dead.index[0]))
    drift = cand[cand.contaminated].sort_values("held_still")
    if len(drift):
        picks.append(("camera drifts (ratio inflated by parallax)",
                      drift.index[0]))
    worst = ps[ps.system.isin(cand.index)].nsmallest(1, "inlier_affine")
    if len(worst):
        picks.append((f"global-motion fit failure "
                      f"(RANSAC inliers {float(worst.inlier_affine.iloc[0]):.2f})",
                      str(worst.system.iloc[0])))

    tshow = [0.75, 1.75, 2.75, 3.75, 4.75]
    with PdfPages(os.path.join(FIG, "noop_failure_cases.pdf")) as pdf:
        for title, sysname in picks:
            sub = ps[(ps.system == sysname) & (ps.scene.isin(dyn))]
            if not len(sub):
                sub = ps[ps.system == sysname]
            if title.startswith("global"):
                scene = int(worst.scene.iloc[0])
            elif "dead" in title:
                scene = int(sub.nsmallest(1, "ratio").scene.iloc[0])
            elif "drifts" in title:
                scene = int(sub.nlargest(1, "D_affine").scene.iloc[0])
            else:
                scene = int(sub.iloc[(sub.ratio - 1).abs().argsort()[:1]].scene.iloc[0])
            try:
                g, _, _ = N.load_clip(sysname, scene)
                r, _, _ = N.load_clip("real", scene)
            except Exception:  # noqa: BLE001
                continue
            fig, axes = plt.subplots(2, len(tshow),
                                     figsize=(2.6 * len(tshow), 3.3), dpi=150)
            for j, t in enumerate(tshow):
                k = int(round((t - N.T0) * N.GRID_FPS))
                k = min(max(k, 0), N.N_T - 1)
                for i, (fr, lab) in enumerate(((g, PRETTY.get(sysname, sysname)),
                                               (r, "real"))):
                    axes[i, j].imshow(fr[k])
                    axes[i, j].set_xticks([])
                    axes[i, j].set_yticks([])
                    if j == 0:
                        axes[i, j].set_ylabel(lab, fontsize=8)
                    if i == 0:
                        axes[i, j].set_title(f"t={t:g}s", fontsize=8)
            row = ps[(ps.system == sysname) & (ps.scene == scene)].iloc[0]
            fig.suptitle(f"{title}  —  {PRETTY.get(sysname, sysname)} r{scene:02d}"
                         f"   D={row.D_affine:.2f} px/s, ratio={row.ratio:.2f}, "
                         f"inliers={row.inlier_affine:.2f}", fontsize=9)
            fig.tight_layout(rect=[0, 0, 1, 0.94])
            pdf.savefig(fig, bbox_inches="tight")
            # also as PNG: the PDF is not viewable everywhere, and these
            # strips are the qualitative evidence behind the plane
            fig.savefig(os.path.join(
                FIG, "noop_failure_case_"
                     f"{title.split('(')[0].strip().replace(' ', '_').replace(',', '')}"
                     f"_{sysname}_r{scene:02d}.png"), bbox_inches="tight", dpi=110)
            plt.close(fig)
    print(f"failure cases -> {os.path.join(FIG, 'noop_failure_cases.pdf')}")


# ------------------------------------------------------------------ NOOP md
def write_md(res, meta):
    tau = meta["tau_D"]
    dyn = meta["dyn_scenes"]
    cont = [s for s in res.index if res.contaminated[s]]
    miss = meta["missing"]

    def row(s):
        r = res.loc[s]
        fvd = "—" if s == "real" or "fvd" not in r or not np.isfinite(r.fvd) \
            else f"{r.fvd:.1f}"
        lp = "—" if s == "real" or "lpips_dyn" not in r or not np.isfinite(r.lpips_dyn) \
            else f"{r.lpips_dyn:.3f}"
        d = "†" if r.contaminated else ""
        return (f"| {PRETTY.get(s, s).replace('$^{a}$','')}{d} | "
                f"{r.held_still:.1f} [{r.held_lo:.0f}, {r.held_hi:.0f}] | "
                f"{r.ratio:.2f} [{r.ratio_lo:.2f}, {r.ratio_hi:.2f}] | "
                f"{fvd} | {lp} | {r.median_D:.2f} |")

    md = f"""# Final stationary (no-op) evaluation

Main-paper result: the **behaviour plane** — Held-Still (%) against the
residual scene-motion ratio. A zero action command must stop the *camera*
without stopping the *world*, and those are separate axes, so the result is a
plane rather than a scalar.

Figure: `out/figs/noop_behaviour_plane.{{pdf,png}}`
Table: `out/noop_main_table.{{csv,tex}}` · per-scene `out/noop_per_scene.csv`
Bins `out/noop_activity_bins.csv` · sensitivity `out/noop_sensitivity.csv`
Failure cases `out/figs/noop_failure_cases.pdf`

## 1. Evaluation set and preprocessing

* **Contexts** — the {N.N_SCENES} Phase-A contexts available for every
  internal and external system, identical scene ids throughout, each scored
  against its own paired real continuation.
* **Horizon** — wall-clock **[{N.T0} s, {N.T1} s]**, exactly
  {N.T1 - N.T0:.1f} s of generation measured from the true generation
  boundary (frame 12 of the 16 fps references). **Nothing is padded,
  repeated or extrapolated.** Every system covers the closed interval;
  minwm covers it exactly (its final generated frame lands on t = 4.75 s,
  zero margin).
* **Temporal** — resampled to a common **{N.GRID_FPS} fps** grid,
  **{N.N_T} timestamps** t_k = {N.T0} + k/{N.GRID_FPS}
  (k = 0…{N.N_T - 1}), giving **{N.N_T - 1} adjacent pairs** with
  Δt = {N.DT:.4f} s. Sampling is **nearest source frame**, never blended —
  temporal interpolation ghosts and corrupts optical flow. Max temporal snap
  error is half a source frame interval: 0 ms for the 16 fps systems,
  {res.snap_err_ms.max():.1f} ms worst case overall. Because D and M are
  means over the {N.N_T - 1} pairs, this jitter averages out.
* **Spatial** — **isotropic** resize onto an **{N.CANVAS_W}×{N.CANVAS_H}**
  canvas with symmetric letterbox padding; never anisotropic stretch.
  Native 832×480 systems fill the canvas exactly (valid = 1.000); the
  1280×704 (Yume) and 640×352 (Matrix-Game) sources land at 832×458
  (valid = 0.954). **Padded pixels are excluded from every flow statistic**,
  and the valid mask is eroded by **{N.MASK_ERODE} px** first so letterbox
  edges cannot leak into the field. Paired metrics crop gen and ref to the
  identical valid box.

## 2. Optical-flow decomposition

RAFT-small, torchvision **`Raft_Small_Weights.DEFAULT`**. For each adjacent
pair: dense flow `F_t(x)`; a **global 2D affine** fitted with RANSAC
(`cv2.estimateAffine2D`, threshold **{N.RANSAC_THR} px**, max iters
**{N.RANSAC_ITERS}**, confidence **{N.RANSAC_CONF}**, correspondence grid
stride **{N.GRID} px**); its induced global flow `G_t(x)`; and the residual

    R_t(x) = F_t(x) − G_t(x)

**Affine is the primary result.** A homography is fitted to the *same* flow
field for the appendix sensitivity check — the main number never silently
switches to it. Excluded from all statistics: padded pixels, non-finite flow,
and pixels failing a **forward/backward consistency** check (both directions
computed; kept when ‖F_fwd + F_bwd∘F_fwd‖ ≤ **{N.FB_THRESH} px**). RANSAC
inlier fraction is recorded for every pair and scene
(`out/noop_final_pairs*.csv`).

## 3. Formulas

    D_i = mean_t median_x ‖G_t(x)‖₂ / Δt          [px/s]
    M_i = mean_t Q_{N.Q_MAIN:.2f},x ‖R_t(x)‖₂ / Δt          [px/s]

    tau_D  = {N.TAU_PCT}th percentile of D over real continuations
    held_i = 1[D_i ≤ tau_D]
    Held-Still (%) = 100 · mean_i held_i

    r_i    = (M_i^gen + ε) / (M_i^real + ε),   ε = {N.EPSILON} px/s
    Ratio  = exp( mean_i log r_i )     over reference-DYNAMIC scenes only

* **tau_D = {tau:.4f} px/s**, the {N.TAU_PCT}th percentile of D over
  **{meta['n_real_for_tau']}** real continuations (Phase-A + Phase-B, same
  stationary-context gate, identical preprocessing).
* **ε = {N.EPSILON} px/s**, fixed **a priori** (≈1/32 px of displacement per
  {N.DT:.4f} s pair, far below the RAFT noise floor) so it regularises a
  vanishing denominator without shifting real ratios. Declared in code before
  any ranking was inspected.
* **Activity bins** come from the real continuations of the {N.N_SCENES}
  Phase-A contexts **only**, as terciles of M^real
  (edges {meta['bin_edges'][0]:.3f} / {meta['bin_edges'][1]:.3f} px/s).
  The ratio uses the **reference-dynamic** bin only —
  **{len(dyn)} scenes**: {dyn}. Pooling static scenes would put a
  noise-dominated denominator under the ratio and make a frozen model look
  active.
* **Bootstrap** — {N.N_BOOT} **paired scene** resamples (seed
  {N.BOOT_SEED}); the same resampled scene indices are applied to every
  system so the intervals are comparable. Held-Still resamples the
  {N.N_SCENES} scenes; the ratio resamples the {len(dyn)} dynamic scenes.
  95% CIs are percentile intervals. Continuous D_i and per-scene ratios are
  saved in `noop_per_scene.csv` — the binary result is never the only output.

## 4. Baselines through the same pipeline

* **Real** — the reference continuations themselves. Its Held-Still is
  *computed*, not placed: {res.held_still['real']:.1f}%, which is the
  expected ≈{N.TAU_PCT}% since tau_D is that percentile of the (larger)
  A+B real pool. Its ratio is 1.00 by construction (numerator = denominator),
  so its FVD/LPIPS are degenerate and reported as —.
* **Freeze** — the final real **context** frame (t = {N.CTX_LAST_T} s) held
  for the whole {N.T1 - N.T0:.0f} s horizon, through the identical pipeline.
  It lands at Held-Still {res.held_still['freeze']:.1f}% and ratio
  {res.ratio['freeze']:.2f}: the bottom-right corner, exactly where a
  perfectly stationary but perfectly dead world belongs. Its nonzero residual
  ({res.ratio['freeze']:.2f} rather than 0) is the RAFT noise floor on
  identical frames, and is the instrument's effective zero.

## 5. Main table

| Model | Held-Still % ↑ | Residual ratio ↔1 | FVD (R3D-18) ↓ | LPIPS_dyn ↓ | median D (px/s) |
|---|---|---|---|---|---|
""" + "\n".join(row(s) for s in res.index) + f"""

† **drift-contaminated**: {', '.join(PRETTY.get(c, c).replace('$^{{a}}$','') for c in cont) if cont else 'none'}.
For these the residual ratio is **not** evidence of preserved independent
world motion — camera parallax survives global-motion subtraction. Describe
it only as *apparent residual motion*.

**How the flag is set, and a caveat about the specified rule.** The specified
criterion — Held-Still < 50% **or** median D_i > 2·tau_D — fires for
**no model on this data**, so used alone it would leave the drifters
unmarked and implicitly credited with world liveliness. The reason is that
tau_D is already a lax bar: it is the {N.TAU_PCT}th percentile of *real*
D, and the real stationary windows do contain some genuine camera motion, so
tau_D = {tau:.2f} px/s and 2·tau_D = {2 * tau:.2f} px/s sits above every
model's median D (largest is {res.median_D.max():.2f} px/s). Both flags are
stored separately in `noop_main_table.csv`
(`contaminated_spec_rule`, `contaminated_held_rule`).

The dagger in the figure and table therefore uses a supplementary bar:
**Held-Still < {HELD_CONTAM_PCT:.0f}%**, i.e. a model that fails the
real-calibrated stationarity test on more than 10% of scenes — at least twice
the ≈5% failure rate the real continuations show by construction. It
separates unambiguously here: {', '.join(f'{PRETTY.get(c, c)} {res.held_still[c]:.1f}%' for c in cont) if cont else '—'}
versus ≥{res[~res.contaminated].held_still.min():.1f}% for everything else.
Changing this bar changes which points carry a dagger and nothing else —
no metric depends on it.

FVD is computed with the repository's **Kinetics-pretrained R3D-18** feature
extractor on these same {N.N_SCENES} scenes and this same
{N.T1 - N.T0:.0f} s horizon. It is labelled **FVD (R3D-18)** and is **not**
CD-FVD, VideoMAE-FVD or standard I3D-FVD. LPIPS uses {N.LPIPS_N} uniformly
spaced paired timestamps over the horizon, reported on the
reference-dynamic bin, bootstrapped by scene; it is a supporting column, not
a ranking.

Matrix-Game is driven by an **authored all-zero action stream** — representable
by the released tensors, **not an official benchmarked no-op interface**.

## 6. Missing data

{"None — every system has all 32 scenes." if not len(miss) else miss[['system','scene','reason']].to_string(index=False)}

No output was retried, regenerated or discarded on quality grounds.

## 7. Interpretation

**Supported.**
* The plane separates two failure modes that a single motion number conflates:
  a frozen world (high Held-Still, ratio → 0) and camera drift (low
  Held-Still), and it does so against calibrated anchors — Real at ratio 1 and
  Freeze at the floor — rather than against an arbitrary scale.
* Freeze and Real land where they must, which is the pipeline's own sanity
  check: any model near Freeze is dead, any model near Real is behaving.
* **A single fidelity number cannot carry this axis.** The Freeze baseline —
  one real frame repeated for four seconds, with no world model at all —
  scores **FVD {res.fvd['freeze']:.1f}**, which is better than
  **{int((res.drop(index=['real', 'freeze']).fvd > res.fvd['freeze']).sum())} of the
  {len(res) - 2}** systems evaluated here. Any ranking that reads FVD alone
  therefore ranks a static image above most working world models on the
  stationary axis. The plane exists because the ratio axis is what
  separates them: Freeze sits at {res.ratio['freeze']:.2f}.
* Held-Still is thresholded **only** on real footage, and the activity bins
  are defined **only** from real continuations, so no generated output
  influences either.

**Requires qualification.**
* **The target corner is not ours.** On this plane the closest system to
  (100%, 1) is **WorldCam** ({res.held_still['worldcam']:.1f}%,
  {res.ratio['worldcam']:.2f}) — it holds the camera and keeps world motion
  at the real level better than any of our variants, which cluster at
  {res.loc[[o for o in OURS if not res.contaminated[o]]].ratio.min():.2f}–{res.loc[[o for o in OURS if not res.contaminated[o]]].ratio.max():.2f}
  and so under-animate the world by roughly a third. Our variants win on
  *fidelity* on the same clips (FVD
  {res.loc[[o for o in OURS if not res.contaminated[o]]].fvd.min():.1f}–{res.loc[[o for o in OURS if not res.contaminated[o]]].fvd.max():.1f}
  vs WorldCam {res.fvd['worldcam']:.1f}), and WorldCam carries a 0.25
  stationary corruption rate (`NOOP.md`) that this plane does not see. State
  both; do not present the behaviour plane as a win for our models.
* The residual ratio of any †-marked model. Parallax from a moving camera is
  not removed by a global affine (or homography) fit, so their residual is
  *apparent* motion.
* D_i is a per-frame-pair magnitude averaged over time, i.e. a rate of
  motion **path length**, not net displacement. A camera that jitters about a
  fixed pose scores a high D even with zero net drift. This is the intended
  reading of "not stationary", but it is not the same as "drifted away".
* FVD on {N.N_SCENES} clips per side is small-n; treat gaps of a few points
  as noise.
* Nearest-frame temporal resampling snaps by up to
  {res.snap_err_ms.max():.1f} ms for the highest-fps sources.

**Do not claim.**
* That a †-marked model preserves world motion.
* That this FVD is comparable to published CD-FVD / I3D-FVD numbers, or to
  the exploratory 6 s-horizon FVD in `NOOP.md` (different horizon, different
  preprocessing).
* That LPIPS or PSNR/SSIM rank models here — PSNR/SSIM in particular reward a
  frozen clip in a dynamic scene and are appendix-only.
* That the stationary corruption rate is the paper's geometry metric; it is a
  separate instrument with its own real-reference thresholding.

## 8. Appendix material

`NOOP.md` (exploratory diagnostics): FVD vs raw frame-difference motion ratio,
PSNR/SSIM, static/mild/dynamic breakdowns, the 64-context Phase-B ablation,
Matrix-Game collapse evidence, and the **stationary corruption rate** (the
real-reference-thresholded mangle instrument — reported under that name and
**not** a replacement for the paper's existing geometry metric).
Here: continuous drift distributions and RANSAC inlier distributions
(`noop_per_scene.csv`, `noop_final_pairs*.csv`), affine-vs-homography and
Q0.80/0.90/0.95 sensitivity (`noop_sensitivity.csv`), qualitative failure
cases (`noop_failure_cases.pdf`).

## 9. Reproducing

```bash
python analysis/testbench_v2/noop_final_eval.py flow     # TB2_SHARD/TB2_NSHARD
python analysis/testbench_v2/noop_final_eval.py paired
python analysis/testbench_v2/noop_final_eval.py report
```
"""
    p = os.path.join(ARR, "analysis", "eval_final", "NOOP_FINAL.md")
    open(p, "w").write(md)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
