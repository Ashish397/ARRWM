"""1-fps contact sheets for the same 90 random fleet videos the detector bake-off used
(seed 3). Raw frames, no detection overlay, so we can eyeball any conjuration the detectors
missed. One row per video (last real ctx frame + 1 frame/sec over the 6s horizon)."""
import os
import numpy as np, cv2
import fleet_common as fc

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, "out"); os.makedirs(OUTD, exist_ok=True)
SEED = 3; NVID = 90; PER_SHEET = 18; PANW = 200; MAXCOL = 8


def row(scene, mdl):
    n, fps = fc.meta(scene, mdl); ctx = fc.ctx_of(mdl)
    end = min(n - 1, ctx + int(round(6.0 * fps)))
    idx = [max(0, ctx - 1)] + list(range(ctx, end + 1, max(1, int(round(fps)))))
    idx = idx[:MAXCOL]
    frames = fc.frames_at(scene, mdl, idx)
    cells = []
    for j, f in enumerate(frames):
        im = cv2.cvtColor(cv2.resize(f, (PANW, int(PANW * f.shape[0] / f.shape[1]))), cv2.COLOR_RGB2BGR)
        lb = "real" if j == 0 else f"{j}s"
        h = np.full((15, im.shape[1], 3), 28, np.uint8)
        cv2.putText(h, lb, (3, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        cells.append(np.concatenate([h, im], 0))
    ch = cells[0].shape[0]
    while len(cells) < MAXCOL:
        cells.append(np.full((ch, PANW, 3), 28, np.uint8))
    body = np.concatenate(cells, 1)
    tag = np.full((body.shape[0], 150, 3), 18, np.uint8)
    cv2.putText(tag, mdl, (5, body.shape[0]//2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 220, 255), 1)
    cv2.putText(tag, scene, (5, body.shape[0]//2 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 220, 255), 1)
    return np.concatenate([tag, body], 1)


def main():
    order = list(fc.fleet_index())
    np.random.RandomState(SEED).shuffle(order)
    order = order[:NVID]
    rows = []
    for scene, mdl in order:
        try:
            rows.append(row(scene, mdl))
        except Exception as e:
            print("skip", mdl, scene, str(e)[:40])
    W = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, W - r.shape[1], cv2.BORDER_CONSTANT, value=(18, 18, 18)) for r in rows]
    for s in range(0, len(rows), PER_SHEET):
        chunk = rows[s:s + PER_SHEET]
        sep = [np.full((2, W, 3), 60, np.uint8)]
        stacked = []
        for r in chunk:
            stacked += [r] + sep
        img = np.concatenate(stacked, 0)
        p = os.path.join(OUTD, f"conjure_contact_{s//PER_SHEET + 1}.png")
        cv2.imwrite(p, img); print("wrote", p, f"({len(chunk)} videos)")


if __name__ == "__main__":
    main()
