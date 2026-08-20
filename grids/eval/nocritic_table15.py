"""Table 15 (Movement / Animation) + Figure 28 data, WITH the No Critic variant.

Runs the two deployed stationary instruments unmodified except for the input
directory, over a working set that is the published stationary_evaluation plus
No Critic's 32 no-op rollouts:

  Movement  = mean over scenes of hypot(pc0 - pc0_real, pc1 - pc1_real)
              from stationary_cotracker_pca.py   (CoTracker->PCA departure
              from the real continuation, action units, real == 0 by construction)
  Animation = mean of life_pca from stationary_signs.py
              (localised coherent residual mover CELLS on an 8x8 grid, after
              removing the top-3 PCA modes of the displacement trajectories)

The seven published variants act as the validation anchor.
Writes out/stationary_cotracker_pca_nc.csv and out/stationary_signs_nc.csv.
"""
import os, sys, glob
import pandas as pd, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
WORK = f"{ARR}/stationary_eval_nc"
sys.path.insert(0, HERE)

PUB = {"pca8": (2.3, 1.4), "pca4": (2.1, 0.0), "pca2": (2.2, 0.0),
       "16node": (2.0, 0.6), "4node": (2.0, 0.0), "noatok": (2.1, 0.1),
       "noadaln": (32.2, 6.2), "real": (0.0, 1.1), "minwm": (3.0, 0.2),
       "matrixgame": (2.1, 0.0), "worldplay": (2.3, 0.0), "worldcam": (6.3, 2.8),
       "astra": (19.8, 11.2), "yume": (45.9, 0.8)}
DISP = {"pca8": "Ours (Default)", "pca4": "Ours (pca4)", "pca2": "Ours (pca2)",
        "16node": "Ours (Batch64)", "4node": "Ours (Batch16)",
        "noatok": "Ours (No Action Tokens)", "noadaln": "Ours (No AdaLN)",
        "nocritic": "Ours (No Critic)", "real": "REAL (reference)"}


def run(mod_name, out_name):
    out = os.path.join(HERE, "out", out_name)
    if os.path.exists(out):
        print(f"[skip] {out_name} exists")
        return out
    mod = __import__(mod_name)
    mod.DIR = WORK
    mod.OUT = out
    mod.main()
    return out


def main():
    pca_csv = run("stationary_cotracker_pca", "stationary_cotracker_pca_nc.csv")
    sgn_csv = run("stationary_signs", "stationary_signs_nc.csv")

    d = pd.read_csv(pca_csv)
    real = d[d.model == "real"].set_index("scene")[["pc0", "pc1"]]
    m = d[d.model != "real"].join(real, on="scene", rsuffix="_r")
    m["mag"] = np.hypot(m.pc1 - m.pc1_r, m.pc0 - m.pc0_r)
    movement = m.groupby("model").mag.mean()

    s = pd.read_csv(sgn_csv)
    animation = s.groupby("model").life_pca.mean()

    print("\n=== Table 15 reproduction (Movement = PCA departure; Animation = mean life_pca) ===")
    print(f"{'model':24s} {'n':>3s} {'Movement':>9s} {'Animation':>10s}   {'published':>14s}  match")
    order = ["real", "pca8", "pca4", "pca2", "16node", "4node", "noatok", "noadaln", "nocritic"]
    for k in order:
        if k == "real":
            mv, an = 0.0, animation.get(k, np.nan)
        else:
            mv, an = movement.get(k, np.nan), animation.get(k, np.nan)
        n = int((s.model == k).sum())
        pub = PUB.get(k)
        ok = ""
        if pub:
            ok = "OK" if (abs(mv - pub[0]) <= 0.15 and abs(an - pub[1]) <= 0.15) else "DIFF"
        print(f"{DISP.get(k,k):24s} {n:3d} {mv:9.1f} {an:10.1f}   "
              f"{str(pub) if pub else '(NEW)':>14s}  {ok}")


if __name__ == "__main__":
    main()
