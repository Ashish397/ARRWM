"""Quantitative world-grove analysis on the FULL 16-channel latents (no PCA).

Per model (from analysis/eval_final/flow_viz/world_lats_{run}.npz — one row per
latent frame, flattened [16,H/8,W/8] float16):

1) TANGLES — per window, the 28 pairwise distances between its 8 action
   trajectories over rollout time, in full latent space:
     monotonicity  = Spearman rho of dist vs time (1 = clean divergence)
     reapproach    = (max_sep - final_sep) / max_sep  (0 = never re-approach;
                     large = trajectories separated then crossed back = tangle)
     tangled_frac  = fraction of pairs with reapproach > 0.25

2) ACTION SEPARABILITY — residual endpoint vectors v(w,d) = end(w,d) -
   mean_d end(w,.) in full D:
     knn_acc   = leave-one-window-out 5-NN (cosine) action classification
                 accuracy (chance = 1/8)
     fisher    = between-action var / within-action var of the residuals
     silhouette= mean cosine silhouette by action label

3) ACTION SHARE OF SIGNAL SPACE — law of total variance at each rollout time:
     action_share(t)  = within-window (across-action) var / total var
     context_share(t) = between-window var / total var
   reported at the final frame + curve; plus the ACTION SUBSPACE effective
   dimensionality from PCA of the residual endpoints: participation ratio
   (sum(lam))^2 / sum(lam^2) and n90 (PCs for 90% var).

Writes analysis/eval_final/grove_metrics.csv, grove_action_share_curves.png,
grove_metrics_bars.png. Env: GT_RUNS colon list (default: all archives found).
"""
import os, glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
FV = f"{ARR}/analysis/eval_final/flow_viz"
EF = f"{ARR}/analysis/eval_final"
DNAMES = ["F", "FR", "R", "BR", "B", "BL", "L", "FL"]
ORDER = ["pca8_8node", "16node", "pca4", "pca2", "4node", "noatok", "noadaln",
         "minwm", "matrixgame", "worldcam", "yume", "worldplay", "astra"]


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean(); ry -= ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else 0.0


def load_run(run):
    f = f"{FV}/world_lats_{run}.npz"
    if not os.path.exists(f):
        return None
    z = np.load(f)
    lats = {}
    for k in z.files:
        wpart, d = k.rsplit("_", 1)
        lats[(int(wpart[1:]), d)] = z[k]
    Fl = min(v.shape[0] for v in lats.values())
    return {k: v[:Fl] for k, v in lats.items()}, Fl


def analyze(run):
    out = load_run(run)
    if out is None:
        return None
    lats, Fl = out
    wins = sorted({w for (w, _) in lats})
    wins = [w for w in wins if all((w, d) in lats for d in DNAMES)]

    # ---- 1) tangles: ALL pairwise trajectory distances over time (full-D) ----
    # every trajectory vs every other one — same-window pairs (shared context,
    # different action) AND cross-window pairs (different contexts).
    keys = [(w, d) for w in wins for d in DNAMES]
    N = len(keys)
    kw = np.array([w for w, _ in keys])
    X = np.stack([lats[k].astype(np.float32) for k in keys])              # [N, Fl, D]
    dists = np.empty((Fl, N, N), dtype=np.float32)
    for t in range(Fl):
        Xt = X[:, t]
        sq = np.sum(Xt ** 2, axis=1)
        d2 = sq[:, None] + sq[None, :] - 2.0 * (Xt @ Xt.T)
        dists[t] = np.sqrt(np.clip(d2, 0, None))
    iu, ju = np.triu_indices(N, 1)
    same_w = kw[iu] == kw[ju]
    D = dists[:, iu, ju]                                                  # [Fl, Npairs]
    t_idx = np.arange(Fl)
    # within-window pairs: shared start -> tangle = separate then re-approach
    monos = [spearman(t_idx, D[:, p]) for p in np.where(same_w)[0]]
    dmax_w = D[:, same_w].max(0); dfin_w = D[-1, same_w]
    reaps = (dmax_w - dfin_w) / np.maximum(dmax_w, 1e-9)
    mono = float(np.mean(monos)); reap = float(np.mean(reaps))
    tangled = float(np.mean(reaps > 0.25))
    # cross-window pairs: start far apart -> tangle = approach / close pass
    Dc = D[:, ~same_w]
    d0_c = Dc[0]; dmin_c = Dc.min(0)
    approach = (d0_c - dmin_c) / np.maximum(d0_c, 1e-9)
    cross_approach = float(np.mean(approach))
    # close pass = a cross-context pair dips below the typical FINAL separation
    # of same-window action pairs (i.e. two different worlds nearly meet)
    thresh = float(np.median(dfin_w))
    close_pass = float(np.mean(dmin_c < thresh))
    cross_end = float(np.mean(D[-1, ~same_w]))        # context scale at final frame

    # ---- ACTION SEPARABILITY (primary): distance between the SAME context's
    # outcomes under different actions, full latent space, final frame ----
    ka = np.array([DNAMES.index(d) for _, d in keys])
    a_i, a_j = ka[iu][same_w], ka[ju][same_w]
    sep_mat = np.zeros((8, 8)); cnt = np.zeros((8, 8))
    for p in range(len(a_i)):
        a, b = int(a_i[p]), int(a_j[p])
        sep_mat[a, b] += dfin_w[p]; sep_mat[b, a] += dfin_w[p]
        cnt[a, b] += 1; cnt[b, a] += 1
    sep_mat /= np.maximum(cnt, 1)
    sep_mean = float(np.mean(dfin_w))                 # mean same-context inter-action dist
    sep_norm = float(sep_mean / (cross_end + 1e-9))   # relative to context spread
    del dists, D, Dc, X

    # ---- ODE approximability: piecewise-linear segment complexity ----
    # Greedy RDP-style split in full-D: a trajectory that a few segments fit
    # is straight/smooth (a coarse ODE solve reproduces it); many segments =
    # curvy/jerky dynamics. Also chord/arc straightness (1 = perfectly straight).
    def n_segments(traj, tol):
        def max_dev(a, b):
            seg = traj[b] - traj[a]
            L2 = float(seg @ seg)
            if L2 == 0 or b - a < 2:
                return 0.0, a
            pts = traj[a + 1:b] - traj[a]
            proj = np.clip((pts @ seg) / L2, 0, 1)[:, None] * seg[None]
            dev = np.linalg.norm(pts - proj, axis=1)
            k = int(np.argmax(dev))
            return float(dev[k]), a + 1 + k
        stack, count = [(0, len(traj) - 1)], 0
        while stack:
            a, b = stack.pop()
            dev, k = max_dev(a, b)
            if dev > tol and k > a and k < b:
                stack += [(a, k), (k, b)]
            else:
                count += 1
        return count

    segs5, segs10, straights = [], [], []
    for w in wins:
        for d in DNAMES:
            traj = lats[(w, d)].astype(np.float32)
            step = np.linalg.norm(np.diff(traj, axis=0), axis=1)
            arc = float(step.sum())
            chord = float(np.linalg.norm(traj[-1] - traj[0]))
            if arc > 0:
                straights.append(chord / arc)
                segs5.append(n_segments(traj, 0.05 * arc))
                segs10.append(n_segments(traj, 0.10 * arc))
    seg5 = float(np.mean(segs5)); seg10 = float(np.mean(segs10))
    straight = float(np.mean(straights))

    # ---- 4 field-level dynamics metrics ----------------------------------
    # (a) RF straightness: mean_t ||v_t - chord_velocity||^2 / ||chord_velocity||^2
    #     (0 = perfectly straight constant-speed path; rectified-flow's target)
    rf_vals = []
    for w in wins:
        for d in DNAMES:
            traj = lats[(w, d)].astype(np.float32)
            v = np.diff(traj, axis=0)
            chord_v = (traj[-1] - traj[0]) / (len(traj) - 1)
            cn = float(chord_v @ chord_v)
            if cn > 0:
                rf_vals.append(float(np.mean(np.sum((v - chord_v) ** 2, 1)) / cn))
    rf_straight = float(np.mean(rf_vals))

    # (b) velocity-field consistency: at matched times, cosine of velocities
    #     across windows under the SAME action (vs cross-action baseline)
    Vel = {}                                          # (w,d) -> [Fl-1, D] normalized
    for w in wins:
        for d in DNAMES:
            v = np.diff(lats[(w, d)].astype(np.float32), axis=0)
            Vel[(w, d)] = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
    same_cos, cross_cos = [], []
    nw = len(wins)
    for t in range(Fl - 1):
        M = np.stack([Vel[(w, d)][t] for w in wins for d in DNAMES])   # [nw*8, D]
        G = M @ M.T
        aid = np.tile(np.arange(8), nw)
        wid = np.repeat(np.arange(nw), 8)
        iu2, ju2 = np.triu_indices(len(M), 1)
        dw = wid[iu2] != wid[ju2]
        sa = aid[iu2] == aid[ju2]
        same_cos.append(float(G[iu2, ju2][dw & sa].mean()))
        cross_cos.append(float(G[iu2, ju2][dw & ~sa].mean()))
    velcons_same = float(np.mean(same_cos)); velcons_cross = float(np.mean(cross_cos))

    # (c) shared-dynamics R^2 (transferability): one ridge model
    #     dx = A x + B onehot(a) + c in a 32-D PCA space, fit on half the
    #     windows, R^2 on held-out windows; vs each held-out window's own
    #     in-sample fit. transfer = shared R^2 / private R^2.
    allp = np.concatenate([lats[k].astype(np.float32) for k in Vel], 0)
    mu32 = allp.mean(0)
    _, _, Vv = torch.pca_lowrank(torch.from_numpy(allp - mu32), q=32, niter=4)
    P32 = Vv[:, :32].numpy(); del allp
    def design(ws):
        Xs, Ys = [], []
        for w in ws:
            for ai, d in enumerate(DNAMES):
                z = (lats[(w, d)].astype(np.float32) - mu32) @ P32     # [Fl, 32]
                oh = np.zeros((len(z) - 1, 8), np.float32); oh[:, ai] = 1
                Xs.append(np.concatenate([z[:-1], oh, np.ones((len(z) - 1, 1), np.float32)], 1))
                Ys.append(np.diff(z, axis=0))
        return np.concatenate(Xs), np.concatenate(Ys)
    tr_w, te_w = wins[0::2], wins[1::2]
    Xtr, Ytr = design(tr_w); Xte, Yte = design(te_w)
    xm = Xtr[:, :32].mean(0); xs = Xtr[:, :32].std(0) + 1e-6
    Xtr[:, :32] = (Xtr[:, :32] - xm) / xs
    Xte[:, :32] = (Xte[:, :32] - xm) / xs
    lam = float(len(Xtr)) * 1e-3
    W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ Ytr)
    ss_res = float(np.sum((Xte @ W - Yte) ** 2))
    ss_tot = float(np.sum((Yte - Ytr.mean(0)) ** 2))
    shared_r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    priv = []
    for w in te_w:
        Xw, Yw = design([w])
        Xw[:, :32] = (Xw[:, :32] - xm) / xs
        lam_w = float(len(Xw)) * 1e-3
        Ww = np.linalg.solve(Xw.T @ Xw + lam_w * np.eye(Xw.shape[1]), Xw.T @ Yw)
        priv.append(1.0 - float(np.sum((Xw @ Ww - Yw) ** 2)) /
                    (float(np.sum((Yw - Yw.mean(0)) ** 2)) + 1e-12))
    private_r2 = float(np.mean(priv))
    transfer = float(shared_r2 / (private_r2 + 1e-12))

    # (d) Levy-area signature ratio (rough-path level-2 vs level-1) on an
    #     8-D PCA projection: 0 for a straight line, grows with looping/area
    P8 = P32[:, :8]
    levy = []
    for w in wins:
        for d in DNAMES:
            z = (lats[(w, d)].astype(np.float32) - mu32) @ P8          # [Fl, 8]
            dz = np.diff(z, axis=0); zc = z[:-1] - z[0]
            A = 0.5 * (zc.T @ dz - dz.T @ zc)                          # antisym [8,8]
            s1 = z[-1] - z[0]
            n1 = float(s1 @ s1)
            if n1 > 0:
                levy.append(float(np.linalg.norm(A) / (0.5 * n1)))
    sig_levy = float(np.mean(levy))
    del Vel

    # ---- residual endpoint vectors ----
    V, labs, wlab = [], [], []
    for w in wins:
        E = np.stack([lats[(w, d)][-1].astype(np.float32) for d in DNAMES])
        E -= E.mean(0, keepdims=True)
        for i, d in enumerate(DNAMES):
            V.append(E[i]); labs.append(i); wlab.append(w)
    V = np.stack(V); labs = np.array(labs); wlab = np.array(wlab)
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)

    # ---- linear-composition test: diagonal commands are (c1+c2)/sqrt(2) of
    # cardinals, so a linear action residual predicts v(FR) ~ v(F)+v(R). ----
    DIAG = {"FR": ("F", "R"), "BR": ("B", "R"), "BL": ("B", "L"), "FL": ("F", "L")}
    fin_by = {}
    for w in wins:
        E = np.stack([lats[(w, d)][-1].astype(np.float32) for d in DNAMES])
        E -= E.mean(0, keepdims=True)
        for i, d in enumerate(DNAMES):
            fin_by[(w, d)] = E[i]
    comp_cos, comp_mag = [], []
    for w in wins:
        for dg, (c1, c2) in DIAG.items():
            pred = fin_by[(w, c1)] + fin_by[(w, c2)]
            act = fin_by[(w, dg)]
            na, np_ = np.linalg.norm(act), np.linalg.norm(pred)
            if na > 0 and np_ > 0:
                comp_cos.append(float(act @ pred / (na * np_)))
                comp_mag.append(float(na / (np_ / np.sqrt(2))))
    comp_cos_m = float(np.mean(comp_cos)); comp_mag_m = float(np.mean(comp_mag))

    # ---- 2) separability ----
    S = Vn @ Vn.T
    correct = 0
    for i in range(len(Vn)):
        mask = wlab != wlab[i]
        idx = np.argsort(-S[i][mask])[:5]
        votes = labs[mask][idx]
        pred = np.bincount(votes, minlength=8).argmax()
        correct += int(pred == labs[i])
    knn = correct / len(Vn)

    mu_all = V.mean(0)
    between = np.mean([np.sum((V[labs == a].mean(0) - mu_all) ** 2) for a in range(8)])
    within = np.mean([np.sum((V[labs == a] - V[labs == a].mean(0)) ** 2, axis=1).mean()
                      for a in range(8)])
    fisher = float(between / (within + 1e-12))

    sil = []
    for i in range(len(Vn)):
        own = S[i][(labs == labs[i]) & (np.arange(len(Vn)) != i)].mean()
        oth = max(S[i][labs == a].mean() for a in range(8) if a != labs[i])
        aa, bb = 1 - own, 1 - oth                       # cosine distance
        sil.append((bb - aa) / max(aa, bb) if max(aa, bb) > 0 else 0.0)
    sil = float(np.mean(sil))

    # ---- 3) variance decomposition over rollout time ----
    ashare, cshare = [], []
    for t in range(Fl):
        X = np.stack([lats[(w, d)][t].astype(np.float32) for w in wins for d in DNAMES])
        Xw = X.reshape(len(wins), 8, -1)
        tot = np.sum((X - X.mean(0)) ** 2, axis=1).mean()
        wmeans = Xw.mean(1)
        ctx = np.sum((wmeans - wmeans.mean(0)) ** 2, axis=1).mean()
        act = np.mean([np.sum((Xw[k] - Xw[k].mean(0)) ** 2, axis=1).mean()
                       for k in range(len(wins))])
        ashare.append(float(act / (tot + 1e-12)))
        cshare.append(float(ctx / (tot + 1e-12)))

    # effective dimensionality of the action subspace
    G = (V - V.mean(0)) @ (V - V.mean(0)).T / len(V)
    lam = np.linalg.eigvalsh(G); lam = np.clip(lam, 0, None)[::-1]
    pr = float(lam.sum() ** 2 / (np.sum(lam ** 2) + 1e-12))
    n90 = int(np.searchsorted(np.cumsum(lam) / lam.sum(), 0.9) + 1)

    print(f"[gt] {run}: mono={mono:.3f} reapproach={reap:.3f} tangled={tangled:.2f} "
          f"cross_approach={cross_approach:.3f} close_pass={close_pass:.4f} "
          f"sep_mean={sep_mean:.1f} sep_norm={sep_norm:.3f} "
          f"knn={knn:.3f} fisher={fisher:.3f} sil={sil:.3f} "
          f"action_share_end={ashare[-1]:.3f} ctx_share_end={cshare[-1]:.3f} "
          f"PR={pr:.1f} n90={n90}", flush=True)
    print(f"[gt] {run}: seg5={seg5:.1f} seg10={seg10:.1f} straight={straight:.3f} "
          f"rf={rf_straight:.2f} velcons={velcons_same:.3f}/{velcons_cross:.3f} "
          f"sharedR2={shared_r2:.3f} privR2={private_r2:.3f} transfer={transfer:.3f} "
          f"levy={sig_levy:.3f}", flush=True)
    return dict(run=run, mono=mono, reapproach=reap, tangled_frac=tangled,
                cross_approach=cross_approach, close_pass_frac=close_pass,
                sep_mean=sep_mean, sep_norm=sep_norm, sep_mat=sep_mat,
                seg5=seg5, seg10=seg10, straightness=straight,
                comp_cos=comp_cos_m, comp_mag=comp_mag_m,
                rf_straightness=rf_straight, velcons_same=velcons_same,
                velcons_cross=velcons_cross, shared_r2=shared_r2,
                private_r2=private_r2, transfer=transfer, sig_levy=sig_levy,
                knn_acc=knn, fisher=fisher, silhouette=sil,
                action_share_end=ashare[-1], context_share_end=cshare[-1],
                pr_dim=pr, n90=n90, ashare=ashare, cshare=cshare)


def main():
    runs = os.environ.get("GT_RUNS")
    if runs:
        runs = runs.split(":")
    else:
        runs = [os.path.basename(f)[len("world_lats_"):-4]
                for f in sorted(glob.glob(f"{FV}/world_lats_*.npz"))]
        runs = [r for r in ORDER if r in runs] + [r for r in runs if r not in ORDER]
    res = [r for r in (analyze(run) for run in runs) if r]

    import csv
    keys = ["run", "mono", "reapproach", "tangled_frac", "cross_approach",
            "close_pass_frac", "sep_mean", "sep_norm", "seg5", "seg10",
            "straightness", "comp_cos", "comp_mag",
            "rf_straightness", "velcons_same", "velcons_cross",
            "shared_r2", "private_r2", "transfer", "sig_levy",
            "knn_acc", "fisher", "silhouette",
            "action_share_end", "context_share_end", "pr_dim", "n90"]
    with open(f"{EF}/grove_metrics.csv", "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=keys)
        wcsv.writeheader()
        for r in res:
            wcsv.writerow({k: r[k] for k in keys})

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for r in res:
        ax.plot(range(len(r["ashare"])), r["ashare"], lw=2, label=r["run"])
    ax.set_xlabel("video time (latent frame)")
    ax.set_ylabel("action share of total latent variance")
    ax.set_title("How much of the latent signal the ACTION explains, over rollout time\n"
                 "(within-window across-action variance / total variance, full 16-ch latents)")
    ax.legend(ncol=3, fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{EF}/grove_action_share_curves.png", dpi=130)
    plt.close(fig)

    mets = [("sep_norm", "action separability\n(same-context inter-action dist\n/ cross-context dist)"),
            ("knn_acc", "action ID from displacement\n(LOWO 5-NN acc, chance 0.125)"),
            ("tangled_frac", "tangled same-context pairs\n(re-approach > 0.25)"),
            ("close_pass_frac", "cross-context close passes\n(dip below action separation)"),
            ("action_share_end", "action share of variance\n(final frame)"),
            ("pr_dim", "action subspace eff. dim\n(participation ratio)")]
    fig, axes = plt.subplots(1, 6, figsize=(30, 5.5))
    names = [r["run"] for r in res]
    for ax, (k, t) in zip(axes, mets):
        cols = ["#2166ac" if n in ("pca8_8node", "16node") else
                ("#92c5de" if n in ORDER[:7] else "#b2182b") for n in names]
        ax.bar(range(len(res)), [r[k] for r in res], color=cols, alpha=0.9)
        if k == "knn_acc":
            ax.axhline(0.125, color="gray", ls=":", lw=1.5)
        ax.set_xticks(range(len(res)), names, rotation=40, ha="right", fontsize=8)
        ax.set_title(t, fontsize=10); ax.grid(alpha=0.3, axis="y")
    fig.suptitle("World-grove metrics on full 16-channel latents "
                 "(dark blue = our best two, light blue = our ablations, red = external)")
    fig.tight_layout(); fig.savefig(f"{EF}/grove_metrics_bars.png", dpi=130)
    plt.close(fig)

    # 8x8 action-pair distance matrices (same context, different action)
    ncol = 5
    nrow = int(np.ceil(len(res) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.0 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(res):]:
        ax.set_visible(False)
    for ax, r in zip(axes, res):
        im = ax.imshow(r["sep_mat"], cmap="viridis")
        ax.set_xticks(range(8), DNAMES, fontsize=7)
        ax.set_yticks(range(8), DNAMES, fontsize=7)
        ax.set_title(f"{r['run']} (mean {r['sep_mean']:.0f})", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle("Same-context inter-action latent distance (final frame, full 16-ch): "
                 "how far apart different actions push the SAME seed latents")
    fig.tight_layout()
    fig.savefig(f"{EF}/grove_sep_matrices.png", dpi=120)
    plt.close(fig)
    print(f"[gt] saved grove_metrics.csv + grove_action_share_curves.png + "
          f"grove_metrics_bars.png + grove_sep_matrices.png")


if __name__ == "__main__":
    main()
