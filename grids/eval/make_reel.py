"""Make a labelled filmstrip (REFERENCE + N generated frames across the 6s) for a
fleet rollout, so p_novel firings can be visually checked. Usage:
  python make_reel.py <model> <scene> <p_novel> <outdir>
"""
import os, sys
import cv2, numpy as np
import fleet_common as fc

H = 260   # panel height


def panel(img_rgb, label):
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    bgr = cv2.resize(bgr, (int(w * H / h), H))
    hdr = np.full((26, bgr.shape[1], 3), 32, np.uint8)
    cv2.putText(hdr, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return np.concatenate([hdr, bgr], 0)


def main():
    model, scene, pnov, outdir = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    os.makedirs(outdir, exist_ok=True)
    n, fps = fc.meta(scene, model); ctx = fc.ctx_of(model)
    ref_i = max(0, ctx - 1)
    gidx = list(np.linspace(ctx, n - 1, 5).round().astype(int))
    frames = fc.frames_at(scene, model, [ref_i] + gidx)
    panels = [panel(frames[0], "REFERENCE (last real)")]
    for j, i in enumerate(gidx):
        panels.append(panel(frames[1 + j], f"GEN {i}"))
    strip = np.concatenate(panels, 1)
    band = np.full((30, strip.shape[1], 3), 20, np.uint8)
    cv2.putText(band, f"{model} {scene}   p_novel={pnov}", (10, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 255, 120), 1)
    strip = np.concatenate([band, strip], 0)
    out = os.path.join(outdir, f"{model}_{scene}_pnovel{pnov}.png")
    cv2.imwrite(out, strip)
    print(out)


if __name__ == "__main__":
    main()
