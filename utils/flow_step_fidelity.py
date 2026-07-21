"""20-step vs 48-step teacher fidelity: trajs_14e8s20 vs trajs_14e8 (r08).

Matched noise + protocol; per block b (0..7), over 8 dirs x 4 seeds:
  end_rms   RMS(x20_final - x48_final)   pathwise endpoint delta
  sep20/48  inter-action separation ratio (action structure preserved?)
  d_mean/std committed-stat deltas
Accept 20-step if end_rms << inter-action separation and stats track.
"""
import numpy as np
FV = "/scratch/u6ex/as1748.u6ex/ARRWM/analysis/eval_final/flow_viz"
DIRS = ["F", "FR", "R", "BR", "B", "BL", "L", "FL"]
IU = np.triu_indices(8, 1)
a = np.load(f"{FV}/trajs_14e8_w8.npz")
b = np.load(f"{FV}/trajs_14e8s20_w8.npz")
print(f"{'blk':>3s} {'end_rms':>8s} {'sep48':>7s} {'sep20':>7s} {'d_mean':>8s} {'d_std':>7s}")
for blk in range(8):
    p = "" if blk == 0 else f"b{blk}_"
    rms, s48, s20, dm, ds = [], [], [], [], []
    for sd in range(4):
        E48 = np.stack([a[f"{p}{d}_{sd}"][-1].astype(np.float32) for d in DIRS])
        E20 = np.stack([b[f"{p}{d}_{sd}"][-1].astype(np.float32) for d in DIRS])
        rms.append(np.sqrt(((E48 - E20) ** 2).mean(1)).mean())
        for E, s in ((E48, s48), (E20, s20)):
            dd = np.linalg.norm(E[:, None] - E[None, :], axis=-1)
            s.append(dd[IU].mean())
        dm.append(E20.mean() - E48.mean()); ds.append(E20.std() - E48.std())
    D = E48.shape[1]
    print(f"{blk:3d} {np.mean(rms)*np.sqrt(D):8.1f} {np.mean(s48):7.1f} "
          f"{np.mean(s20):7.1f} {np.mean(dm):+8.4f} {np.mean(ds):+7.4f}")
print("(end_rms in same L2 units as sep; accept if end_rms well below sep48)")
