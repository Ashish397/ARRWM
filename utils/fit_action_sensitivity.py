"""Per-channel ACTION-SENSITIVITY weights for the serve-time EMD remap.

Lit-review-guided (2026-08-07): a blanket per-channel quantile transport
also removes legitimate action-dependent displacement, which is the
measured 62%->50% assignment cost of full remapping. Measure, from a
same-context action fan, how much each latent channel's committed
statistics actually MOVE with the commanded action:

    A_j = Var_over_dirs( mean_j(committed chunk) )   (+ std variant)

then emit a bounded correction weight that leaves action-carrying
channels alone and fully corrects action-inert ones:

    w_j = clamp( 2 * med(A) / (A_j + med(A)), 0, 1 )

so w=1 at/below the median sensitivity and falls toward 0 for the most
action-sensitive channels. Serve-side: ODE_EMDREMAP_ASENS=<npz> scales
the per-channel blend by w.

Env: AS_RUN (flow_{run} recording dir, default flow_pilot4_alldir8n2),
AS_OUT (npz path), AS_CHUNKS (committed chunks to pool, default all).
"""
import os
import numpy as np

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
FV = f"{ARR}/analysis/eval_final/flow_viz"
RUN = os.environ.get("AS_RUN", "pilot4_alldir8n2")
OUT = os.environ.get("AS_OUT", f"{FV}/action_sensitivity_{RUN}.npz")
DIRS = ["F", "FR", "R", "BR", "B", "BL", "L", "FL"]
NFB, C = 3, 16


def committed_stats(path):
    """-> (mean[C], std[C]) per committed chunk, list over chunks."""
    z = np.load(path)
    sdt = z["sdt"]
    out = {}
    for i, (c, r, t) in enumerate(sdt):
        if t > 0 and int(r) == 3:            # committed x0 of chunk c
            x = z[f"x{i}"].astype(np.float32).reshape(NFB, C, 60, 104)
            out[int(c)] = (x.mean(axis=(0, 2, 3)), x.std(axis=(0, 2, 3)))
    return out


def main():
    mus, sds = {}, {}                        # (dir, seed) -> {chunk: vec}
    for d in DIRS:
        for s in (0, 1):
            p = f"{FV}/flow_{RUN}/r08_{d}_s{s}/steps.npz"
            if not os.path.exists(p):
                continue
            st = committed_stats(p)
            for c, (m, sd) in st.items():
                mus.setdefault((c, s), {})[d] = m
                sds.setdefault((c, s), {})[d] = sd
    if not mus:
        raise SystemExit(f"no recordings under {FV}/flow_{RUN}")

    # Variance ACROSS DIRECTIONS, within (chunk, seed) — pooled.
    var_mu, var_sd = [], []
    for key, per in mus.items():
        if len(per) < 4:
            continue
        var_mu.append(np.stack([per[d] for d in sorted(per)]).var(axis=0))
        var_sd.append(np.stack([sds[key][d] for d in sorted(sds[key])])
                      .var(axis=0))
    A_mu = np.mean(var_mu, axis=0)
    A_sd = np.mean(var_sd, axis=0)
    # Normalize each family by its own median so they combine on equal
    # footing, then take the per-channel max (a channel is "protected"
    # if EITHER its mean or its spread tracks the action).
    A = np.maximum(A_mu / max(np.median(A_mu), 1e-12),
                   A_sd / max(np.median(A_sd), 1e-12))
    w = np.clip(2.0 / (A + 1.0), 0.0, 1.0).astype(np.float32)

    np.savez(OUT, w=w, A_mu=A_mu, A_sd=A_sd, A=A, dirs=np.array(DIRS))
    order = np.argsort(-A)
    print(f"[asens] {RUN}: pooled over {len(var_mu)} (chunk,seed) fans")
    print("[asens] most action-sensitive channels (protected):")
    for j in order[:5]:
        print(f"    ch{j:2d}  A={A[j]:7.3f}  w={w[j]:.3f}")
    print("[asens] least action-sensitive (fully corrected):")
    for j in order[-5:]:
        print(f"    ch{j:2d}  A={A[j]:7.3f}  w={w[j]:.3f}")
    print(f"[asens] w: min {w.min():.3f} med {np.median(w):.3f} "
          f"max {w.max():.3f} -> {OUT}")


if __name__ == "__main__":
    main()
