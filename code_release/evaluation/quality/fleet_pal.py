"""PAL4VST artifact fraction on the full 256x13 fleet (geometry candidate).
8 frames uniform over the generated span; per-frame = mean of L/R 512 tiles (top 15%
haze band dropped); pal_max = max over frames. Reuses pal_local.frame_score + TS.
Writes out/fleet_pal.csv (scene, model, pal_max, pal_mean)."""
import os
import numpy as np, pandas as pd
import fleet_common as fc
import pal_local as P

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "fleet_pal.csv")


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    idx = fc.fleet_index()
    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT); done = set(zip(prev.scene, prev.model))
        rows = prev.to_dict("records")
    else:
        rows = []
    for k, (scene, model) in enumerate(idx):
        if (scene, model) in done:
            continue
        try:
            n, fps = fc.meta(scene, model)
            ctx = fc.ctx_of(model)
            gidx = np.linspace(ctx, n - 1, 8).round().astype(int)
            frames = fc.frames_at(scene, model, gidx)
            if not frames:
                continue
            s = [P.frame_score(f) for f in frames]   # frame_score expects RGB (imageio)
            rows.append(dict(scene=scene, model=model,
                             pal_max=float(np.max(s)), pal_mean=float(np.mean(s))))
        except Exception as e:
            print(f"[fpal] {scene} {model} FAIL {str(e)[:70]}", flush=True); continue
        if (k + 1) % 50 == 0:
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(f"[fpal] {k+1}/{len(idx)}", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"[fpal] wrote {OUT} ({len(rows)})")


if __name__ == "__main__":
    main()
