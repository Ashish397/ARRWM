"""Conjuration by motion-compensated novelty, done right.

Sample 1 frame/sec (plus the last real context frame as the anchor). The CENTRAL QUARTER
is the region we care about - a spawn there is the failure. But to decide whether central
content is a spawn or a normal lateral reveal, we search the WHOLE previous frame, not just
its centre: an egomotion model legitimately brings things in from the left/right edges and
they migrate to the centre. So we take the central quarter of f_{n+1} as a template and slide
it over f_n out toward the edges (unshifted / left / right / up / down, several magnitudes).
  - if SOME shift finds the content in f_n (incl. its side regions) -> it panned in -> normal.
  - if NO shift finds it anywhere in f_n -> it appeared in the middle with no precedent -> CONJURED.
Score = largest connected conjured blob (fraction of the central quarter), MAX over pairs
(one spawn is all we need). CPU only. Usage: python conjure_shift.py"""
import os
import numpy as np, cv2
import fleet_common as fc

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, "out", "conjure_shift"); os.makedirs(OUTD, exist_ok=True)
WORK_W = 256        # full frame downscaled to this width (keeps the sides in view)
THR = 26            # grayscale diff (0-255) above which a template pixel is unmatched
BLUR = 3
# search offsets as fraction of FULL frame: reach from centre out to the edges (~0.25 = edge)
MAGS = [0.06, 0.12, 0.18, 0.25]

PROBE = [("r08_F", "minwm"), ("r03_BR", "yume"), ("r05_BL", "ours_pca4"),
         ("r00_FR", "worldcam"), ("r16_L", "ours_4node"), ("r03_FL", "minwm"),
         ("r09_F", "minwm")]


def gray(frame):
    H = int(WORK_W * frame.shape[0] / frame.shape[1])
    g = cv2.cvtColor(cv2.resize(frame, (WORK_W, H)), cv2.COLOR_RGB2GRAY)
    return cv2.GaussianBlur(g, (BLUR, BLUR), 0).astype(np.float32)


def offsets(W, H):
    o = [(0, 0)]
    for m in MAGS:
        dx, dy = int(round(m * W)), int(round(m * H))
        o += [(dx, 0), (-dx, 0), (0, dy), (0, -dy), (dx, dy), (-dx, dy), (dx, -dy), (-dx, -dy)]
    return o


BLOCK = 11          # neighbourhood that must align together for a match to count
THR = 20            # local mean abs-diff above which a block is unmatched


def conjured_mask(prev, now):
    """Template = central quarter of `now`. Search `prev` (whole frame) under offsets, but
    score matches at BLOCK level: a template pixel is explained by a shift only if its local
    neighbourhood aligns under that SAME shift (a real precedent moves as a coherent patch).
    Conjured = no single shift explains the block."""
    H, W = now.shape
    x0, x1 = int(0.25 * W), int(0.75 * W)
    y0, y1 = int(0.25 * H), int(0.75 * H)
    B = now[y0:y1, x0:x1]
    ch, cw = B.shape
    best = np.full((ch, cw), 1e9, np.float32)
    for (dx, dy) in offsets(W, H):
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        shifted = cv2.warpAffine(prev, M, (W, H), borderValue=-1000.0)
        A = shifted[y0:y1, x0:x1]
        valid = (A > -999).astype(np.float32)
        diff = np.where(A > -999, np.abs(B - A), 0.0).astype(np.float32)
        # local mean abs-diff over BLOCK, counting only valid (in-bounds) pixels
        num = cv2.boxFilter(diff, -1, (BLOCK, BLOCK), normalize=False)
        den = cv2.boxFilter(valid, -1, (BLOCK, BLOCK), normalize=False)
        local = np.where(den > 0.5 * BLOCK * BLOCK, num / np.maximum(den, 1), 1e9)
        best = np.minimum(best, local)
    mask = (best > THR).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask, (x0, y0, x1, y1)


def largest_blob_frac(mask):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return 0.0, None
    areas = stats[1:, cv2.CC_STAT_AREA]
    i = int(np.argmax(areas)) + 1
    return areas[i - 1] / mask.size, (lab == i).astype(np.uint8)


def main():
    for scene, mdl in PROBE:
        n, fps = fc.meta(scene, mdl); ctx = fc.ctx_of(mdl)
        end = min(n - 1, ctx + int(round(6.0 * fps)))
        idx = [max(0, ctx - 1)] + list(range(ctx, end + 1, max(1, int(round(fps)))))
        frames = fc.frames_at(scene, mdl, idx)
        gs = [gray(f) for f in frames]
        best_frac, best_k, best_mask, best_box = 0.0, -1, None, None
        series = []
        for k in range(len(gs) - 1):
            m, box = conjured_mask(gs[k], gs[k + 1])
            frac, blob = largest_blob_frac(m)
            series.append(frac)
            if frac > best_frac:
                best_frac, best_k, best_mask, best_box = frac, k + 1, blob, box
        tag = f"{mdl}_{scene}"
        ss = " ".join(f"{x:.2f}" for x in series)
        print(f"{tag:20s} max_blob={best_frac:.3f}  series=[{ss}]")
        if best_k > 0:
            def disp(i):
                H = int(WORK_W * frames[i].shape[0] / frames[i].shape[1])
                return cv2.cvtColor(cv2.resize(frames[i], (WORK_W, H)), cv2.COLOR_RGB2BGR)
            prev, now = disp(best_k - 1), disp(best_k)
            x0, y0, x1, y1 = best_box
            ov = now.copy()
            cv2.rectangle(ov, (x0, y0), (x1, y1), (200, 200, 0), 1)
            if best_mask is not None:
                full = np.zeros(now.shape[:2], np.uint8); full[y0:y1, x0:x1] = best_mask * 255
                cont, _ = cv2.findContours(full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(ov, cont, -1, (60, 60, 255), 2)
            def L(img, t):
                b = np.full((18, img.shape[1], 3), 30, np.uint8)
                cv2.putText(b, t, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1)
                return np.concatenate([b, img], 0)
            panel = np.concatenate([L(prev, f"prev {best_k-1}"), L(now, f"now {best_k}"),
                                    L(ov, f"conjured {best_frac:.2f} (box=central 1/4)")], 1)
            cv2.imwrite(os.path.join(OUTD, f"{tag}.png"), panel)
    print("viz ->", OUTD)


if __name__ == "__main__":
    main()
