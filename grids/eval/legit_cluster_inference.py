"""Inference for the Default-vs-minWM legitimacy comparison under source-context clustering.

Data: out/final_flags_v2.csv (canonical per-rollout flags; feature-valid rollouts only).
Pairing: Default and minWM rollouts are paired by scene (context x command); context = the rNN prefix
(30 contexts x 8 commands = 240 pairs; r11/r21 are excluded upstream as wet-lens contexts).

Analyses (all reported in the 2026-09-08 answer to the reviewer-style questions):
  A  rollout-level McNemar (asymptotic and exact)             -- the test used in the manuscript
  B  within-context ICC of the paired difference, design effect
  C  Durkalski (2003) clustered McNemar; context-level paired t / Wilcoxon / sign tests
  D  pairs-cluster (case) bootstrap over contexts, percentile CI   <-- RANDOM; seed below
  E  GEE (binomial, exchangeable / independence, cluster-robust SE) and a cluster-robust paired mean

RANDOMNESS AND SEED
  The only stochastic step is the cluster bootstrap (D). It draws from numpy's default_rng with
  SEED = 0, i.e. numpy.random.default_rng(0). Reported intervals: 4000 reps -> [-4.6, +26.7] pp,
  20000 reps -> [-4.6, +26.2] pp. The rollout-level comparison bootstrap draws from the SAME
  generator immediately after the 4000-rep cluster bootstrap (it is therefore not independently
  seeded; re-run this file end-to-end to reproduce it exactly).

Usage: python legit_cluster_inference.py [reps]
"""
import os, sys
import numpy as np, pandas as pd
from math import sqrt
from scipy import stats

SEED = 0                       # numpy.random.default_rng(SEED); do not change without re-reporting
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
HERE = os.path.dirname(os.path.abspath(__file__))
FLAGS = os.path.join(HERE, "out", "final_flags_v2.csv")
A_MODEL, B_MODEL = "ours_pca8", "minwm"


def norm2(z):
    return 2 * (1 - stats.norm.cdf(abs(z)))


def icc1(x, g):
    df = pd.DataFrame({"x": x, "g": g}); m = df.groupby("g").size().mean()
    msb = df.groupby("g").x.mean().var(ddof=1) * m
    msw = df.groupby("g").x.apply(lambda v: v.var(ddof=1)).mean()
    return (msb - msw) / (msb + (m - 1) * msw)


def main():
    F = pd.read_csv(FLAGS); V = F[F.feature_valid.astype(bool)]
    D = V[V.model == A_MODEL].set_index("scene").legit; M = V[V.model == B_MODEL].set_index("scene").legit
    s = D.index.intersection(M.index); D, M = D[s].astype(int), M[s].astype(int)
    d = D - M; ctx = pd.Series(s.str.split("_").str[0], index=s); K = ctx.nunique(); n = len(d)
    print(f"paired rollouts n={n}, contexts K={K}, per context m={n // K}; point estimate {d.mean()*100:+.1f} pp")

    b = int(((D == 1) & (M == 0)).sum()); c = int(((D == 0) & (M == 1)).sum()); z = (b - c) / sqrt(b + c)
    print(f"[A] rollout McNemar: b={b} c={c} z={z:.2f} p_asym={norm2(z):.3f} exact p={stats.binomtest(b, b + c, 0.5).pvalue:.3f}")

    icc = icc1(d.values, ctx.values); deff = 1 + (n // K - 1) * icc
    Fst = stats.f_oneway(*[d[ctx == g].values for g in ctx.unique()])
    print(f"[B] ICC(d)={icc:.3f} design effect={deff:.2f} effective n≈{n / deff:.0f}; ANOVA of d across contexts F={Fst.statistic:.2f} p={Fst.pvalue:.4f}")

    e = (((D == 1) & (M == 0)).astype(int)).groupby(ctx).sum() - (((D == 0) & (M == 1)).astype(int)).groupby(ctx).sum()
    T = e.sum() / sqrt((e ** 2).sum()); cm = d.groupby(ctx).mean()
    print(f"[C] Durkalski clustered McNemar T={T:.2f} p={norm2(T):.3f} | context-level t p={stats.ttest_1samp(cm, 0).pvalue:.3f} "
          f"Wilcoxon p={stats.wilcoxon(cm[cm != 0]).pvalue:.3f} sign {int((cm > 0).sum())}>0 {int((cm < 0).sum())}<0 p={stats.binomtest(int((cm > 0).sum()), int((cm != 0).sum()), 0.5).pvalue:.3f}")

    rng = np.random.default_rng(SEED)                       # <-- the seed
    C = np.array(sorted(ctx.unique())); dv = d.values; cv = ctx.values

    def cluster_boot(reps):
        out = np.empty(reps)
        for i in range(reps):
            pick = rng.choice(C, K, replace=True); out[i] = np.concatenate([dv[cv == g] for g in pick]).mean()
        return out
    bs = cluster_boot(4000); lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"[D] cluster bootstrap seed={SEED} reps=4000: 95% percentile CI [{lo*100:+.1f}, {hi*100:+.1f}] pp, two-sided p≈{2*min((bs <= 0).mean(), (bs >= 0).mean()):.3f}")
    bs2 = cluster_boot(REPS); lo, hi = np.percentile(bs2, [2.5, 97.5])
    print(f"    reps={REPS}: CI [{lo*100:+.1f}, {hi*100:+.1f}] pp, p≈{2*min((bs2 <= 0).mean(), (bs2 >= 0).mean()):.3f}")
    rb = np.array([rng.choice(dv, n, replace=True).mean() for _ in range(4000)])
    print(f"    rollout-level bootstrap (same generator, after the cluster draws): CI [{np.percentile(rb, 2.5)*100:+.1f}, {np.percentile(rb, 97.5)*100:+.1f}] pp")

    try:
        import statsmodels.api as sm, statsmodels.formula.api as smf
        long = pd.DataFrame({"y": np.r_[D.values, M.values], "is_default": np.r_[np.ones(n), np.zeros(n)], "ctx": np.r_[cv, cv]})
        g1 = smf.gee("y ~ is_default", groups="ctx", data=long, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
        g2 = smf.gee("y ~ is_default", groups="ctx", data=long, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Independence()).fit()
        ols = smf.ols("d ~ 1", data=pd.DataFrame({"d": dv, "ctx": cv})).fit(cov_type="cluster", cov_kwds={"groups": cv})
        print(f"[E] GEE exchangeable: log-OR={g1.params['is_default']:.3f} z={g1.params['is_default']/g1.bse['is_default']:.2f} p={g1.pvalues['is_default']:.3f} α={g1.cov_struct.dep_params:.3f} | "
              f"GEE independence p={g2.pvalues['is_default']:.3f} | cluster-robust paired mean p={ols.pvalues['Intercept']:.3f}")
    except Exception as ex:
        print("[E] statsmodels unavailable:", ex)


if __name__ == "__main__":
    main()
