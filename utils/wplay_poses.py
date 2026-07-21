"""Generate WorldPlay pose JSONs for our 8 eval directions (simultaneous
translation+yaw per latent, matching our action semantics — pose strings can
only sequence actions). Convention (theirs): c2w per latent, w 0.08/latent,
yaw 3 deg/latent; K = fx=fy=969.697, cx=960, cy=540. 48 latents (189 frames).
Writes third_party/HY-WorldPlay/poses_ours/pose_{DIR}.json
"""
import json, os
import numpy as np

N_LAT = 48
STEP_T = 0.08
STEP_R = np.radians(3.0)
OUT = "/scratch/u6ex/as1748.u6ex/ARRWM/third_party/HY-WorldPlay/poses_ours"
K = [[969.697, 0.0, 960.0], [0.0, 969.697, 540.0], [0.0, 0.0, 1.0]]

DIRS = {
    "F": (STEP_T, 0.0), "B": (-STEP_T, 0.0),
    "L": (0.0, -STEP_R), "R": (0.0, STEP_R),
    "FL": (STEP_T, -STEP_R), "FR": (STEP_T, STEP_R),
    "BL": (-STEP_T, -STEP_R), "BR": (-STEP_T, STEP_R),
}


def main():
    os.makedirs(OUT, exist_ok=True)
    for d, (t, a) in DIRS.items():
        c2w = np.eye(4)
        entries = {}
        for i in range(N_LAT):
            entries[str(i)] = {"extrinsic": c2w.tolist(), "K": K}
            step = np.eye(4)
            step[0, 0] = np.cos(a); step[0, 2] = np.sin(a)
            step[2, 0] = -np.sin(a); step[2, 2] = np.cos(a)
            step[2, 3] = t
            c2w = c2w @ step
        with open(f"{OUT}/pose_{d}.json", "w") as f:
            json.dump(entries, f)
        print(f"pose_{d}.json: {N_LAT} latents, t={t:+.2f}/lat, yaw={np.degrees(a):+.1f}deg/lat")


if __name__ == "__main__":
    main()
