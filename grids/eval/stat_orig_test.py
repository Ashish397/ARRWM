import os, sys, glob
sys.path.insert(0, "/scratch/u6ex/as1748.u6ex/ARRWM/grids/eval")
import stationary_cotracker as S
S.DIR = "/scratch/u6ex/as1748.u6ex/ARRWM/stationary_evaluation"
S.OUT = "/scratch/u6ex/as1748.u6ex/ARRWM/grids/eval/out/stat_orig_test.csv"
import numpy as np, pandas as pd, torch, cv2
from scipy.spatial import cKDTree
# run the ORIGINAL main() but restricted to 3 variants for speed
_orig_glob = glob.glob
def patched(p):
    fs = _orig_glob(p)
    return [f for f in fs if os.path.basename(f).split("_r")[0] in ("pca8","noadaln","4node")]
S.glob.glob = patched
S.main()
d = pd.read_csv(S.OUT)
print("\n=== ORIGINAL script on the published stationary set ===")
print(f"{'model':10s} {'n':>3s} {'Movement':>9s} {'Animation':>10s}   published")
PUB={"pca8":(2.3,1.4),"noadaln":(32.2,6.2),"4node":(2.0,0.0)}
for m,s in d.groupby("model"):
    print(f"{m:10s} {len(s):3d} {s.camera_motion.median():9.1f} {s.signs_of_life.mean():10.1f}   {PUB.get(m)}")
