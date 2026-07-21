import sys
sys.path.insert(0, "/scratch/u6ex/as1748.u6ex/ARRWM")
import numpy as np, imageio
import utils.attitude_drift as A
p = "/scratch/u6ex/as1748.u6ex/ARRWM/analysis/eval_final/real_refs/real_20240201065845_144.mp4"
r = imageio.get_reader(p); fr = [np.asarray(f) for f in r]; r.close()
for i in (0, 20, 40, 60, 80):
    from PIL import Image
    d = np.asarray(A.depth_pipe(Image.fromarray(fr[i]))["predicted_depth"])
    print(f"frame {i}: depth shape={d.shape} min={d.min():.4f} max={d.max():.4f}")
    print("  attitude:", A.attitude(fr[i]))
