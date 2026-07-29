"""High-frequency degradation on the full 256x13 fleet (CPU). Laplacian-variance
sharpness loss, resized to 832x448, base at 1s into generation vs end window;
sibling-relative B per scene. Writes out/fleet_hf.csv."""
import os
import cv2, numpy as np, pandas as pd
import fleet_common as fc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "fleet_hf.csv")
SIZE = (832, 448)


def lap_var(rgb):
    g = cv2.cvtColor(cv2.resize(rgb, SIZE), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []
    for k, (scene, model) in enumerate(fc.fleet_index()):
        try:
            n, fps = fc.meta(scene, model); ctx = fc.ctx_of(model)
            b0 = ctx + int(round(fps))
            end6 = min(n - 1, ctx + int(round(6.0 * fps)))          # 6s horizon cap
            base_idx = [i for i in range(b0, b0 + 4) if i < n]
            end_idx = list(range(max(ctx, end6 - 14), end6 + 1, 2))
            if not base_idx or not end_idx:
                continue
            fr = fc.frames_at(scene, model, base_idx + end_idx)
            base = np.mean([lap_var(f) for f in fr[:len(base_idx)]])
            end = np.mean([lap_var(f) for f in fr[len(base_idx):]])
            rows.append(dict(scene=scene, model=model, base_blur=round(base, 1),
                             end_blur=round(end, 1), d_blur=round(end - base, 1)))
        except Exception as e:
            print(f"[fhf] {scene} {model} FAIL {str(e)[:60]}", flush=True); continue
        if (k + 1) % 200 == 0:
            print(f"[fhf] {k+1}/3328", flush=True)
    df = pd.DataFrame(rows)
    df["x_blur"] = df.groupby("scene").d_blur.transform(lambda s: s - s.median())
    df["B"] = -df["x_blur"]
    df.to_csv(OUT, index=False)
    print(f"[fhf] wrote {OUT} ({len(df)})")


if __name__ == "__main__":
    main()
