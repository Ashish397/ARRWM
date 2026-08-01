"""Run the deployed conjuration detector over the entire 256x13 fleet.

Each rollout is truncated to its context plus SPAN seconds of generation (default 6 s), matching
the window used by the other quality axes, so models with very different clip lengths and frame
rates are judged over the same amount of generated time. Decoding stops at the cap too, which
is where most of the saving comes from for the longer baselines.

Outputs (out/popin_fleet_all/):
  popin_fleet_all.csv           one row per rollout: flag, score, class, birth, evidence
  summary.txt                   per-model flag counts and rates
  reel_<model>_<scene>.png      filmstrip for every flagged rollout
  popin_fleet_all_<model>.png   that model's flagged rollouts stacked

Usage: python popin_fleet_all.py [--span 6] [--backend rtdetr] [--limit N]
"""
import os, sys, json, time
import numpy as np, cv2, imageio, pandas as pd

import popin_detect as P
import popin_backends as B
import fleet_common as fc

HERE = os.path.dirname(os.path.abspath(__file__))
# AF_POPIN_OUT keeps a new ablation out of the shipped reference artefact.
OUTD = os.environ.get("AF_POPIN_OUT", os.path.join(HERE, "out", "popin_fleet_all"))


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def load(scene, model, span):
    """Decode a rollout up to ctx + span seconds only. ours_* tiles are cropped out of the
    3328x960 grid as it streams, so the full grid is never held in memory."""
    path = fc._path(scene, model)
    rd = imageio.get_reader(path)
    fps = float(rd.get_meta_data().get("fps", 16) or 16)
    ctx = fc.ctx_of(model)
    need = ctx + int(round(span * fps))
    # Variants rendered after the grid was built ship as standalone tiles and
    # are already cropped, so they need no de-tiling window.
    variant = model[len("ours_"):] if model.startswith("ours_") else None
    tile = fc.POS.get(variant) if variant else None
    frames = []
    for i, f in enumerate(rd.iter_data()):
        if i > need:
            break
        a = np.asarray(f)
        frames.append(a[tile[1] + 32:tile[1] + 480, tile[0]:tile[0] + 832] if tile else a)
    rd.close()
    return np.stack(frames), fps, ctx


def reel(vid, fps, f, title, path):
    b, n = f["birth"], len(vid)
    x0, y0, x1, y1 = [int(v) for v in f["box"]]
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    s = int(max(x1 - x0, y1 - y0) * 1.8)
    H, W = vid.shape[1:3]
    TW, TH = 300, 172
    full, zoom = [], []
    for o in [-1.0, -0.5, -0.15, 0.0, 0.5, 1.5, 3.0]:
        k = min(n - 1, max(0, b + int(round(o * fps))))
        fr = vid[k].copy()
        cv2.rectangle(fr, (x0, y0), (x1, y1), (60, 225, 60), 3)
        fr = cv2.resize(fr, (TW, TH))
        cv2.putText(fr, f"{o:+.1f}s", (4, 15), 0, 0.45, (255, 255, 0), 1)
        full.append(fr)
        cr = vid[k][max(0, cy - s):min(H, cy + s), max(0, cx - s):min(W, cx + s)]
        cr = cv2.resize(cr, (TH, TH), interpolation=cv2.INTER_NEAREST) if cr.size else np.zeros((TH, TH, 3), np.uint8)
        zoom.append(cv2.copyMakeBorder(cr, 0, 0, (TW - TH) // 2, TW - TH - (TW - TH) // 2,
                                       cv2.BORDER_CONSTANT, value=(18, 18, 18)))
    body = np.concatenate([np.concatenate(full, 1), np.concatenate(zoom, 1)], 0)
    lab = np.full((28, body.shape[1], 3), 16, np.uint8)
    cv2.putText(lab, title, (5, 20), 0, 0.48, (255, 255, 255), 1)
    cv2.imwrite(path, cv2.cvtColor(np.concatenate([lab, body], 0), cv2.COLOR_RGB2BGR))


def main():
    os.makedirs(OUTD, exist_ok=True)
    span = float(arg("--span", 6))
    backend = arg("--backend", "rtdetr")
    limit = int(arg("--limit", 0))
    dense, crop = B.build(backend)
    P.set_detector(crop)

    idx = fc.fleet_index()
    # AF_POPIN_MODELS scores a subset, so a new ablation does not put the whole
    # fleet back through the detector.
    _only = {m.strip() for m in os.environ.get("AF_POPIN_MODELS", "").split(",") if m.strip()}
    if _only:
        idx = [(s, m) for s, m in idx if m in _only or m.replace("ours_", "") in _only]
    if limit:
        idx = idx[:limit]
    csv_path = os.path.join(OUTD, "popin_fleet_all.csv")
    rows, hits, t0 = [], {}, time.time()
    print(f"{len(idx)} rollouts, {backend}, capped at ctx + {span:g}s of generation\n", flush=True)

    for i, (scene, model) in enumerate(idx, 1):
        uid = f"{model}_{scene}"
        try:
            vid, fps, ctx = load(scene, model, span)
        except Exception as e:
            print(f"[{i:4d}/{len(idx)}] {uid:<26} SKIP ({e})", flush=True)
            continue
        P.set_fps(fps)
        ev = P.analyse(vid, {"n": len(vid), "w": vid.shape[2], "h": vid.shape[1], "dets": dense(vid)}, ctx)
        pos = [f for f in ev if f["score"] > 0]
        top = ev[0] if ev else None
        rows.append(dict(uid=uid, model=model, scene=scene, n=len(vid), fps=fps, ctx=ctx,
                         flag=int(bool(pos)), n_events=len(pos),
                         score=top["score"] if top else None,
                         cls=top["cls"] if top else None,
                         birth=top["birth"] if top else None,
                         birth_s=round((top["birth"] - ctx) / fps, 2) if top else None,
                         branch=top.get("branch") if top else None,
                         onset=top["onset"] if top else None,
                         zprobe=top["zprobe"] if top else None,
                         bncc=top["bncc"] if top else None,
                         box=json.dumps([round(v, 1) for v in top["box"]]) if top else None))
        if pos:
            f = pos[0]
            title = (f"{model} {scene}   score={f['score']:+.2f}  [{f['cls']}]  "
                     f"birth={(f['birth']-ctx)/fps:.1f}s into generation   "
                     f"onset={f['onset']} zprobe={f['zprobe']} bncc={f['bncc']}")
            p = os.path.join(OUTD, f"reel_{model}_{scene}.png")
            reel(vid, fps, f, title, p)
            hits.setdefault(model, []).append(p)
        if i % 25 == 0 or pos:
            pd.DataFrame(rows).to_csv(csv_path, index=False)
        mark = f"HIT {top['score']:+.2f}" if pos else ("-   " + (f"{top['score']:+.2f}" if top else "     "))
        print(f"[{i:4d}/{len(idx)}] {uid:<26} n={len(vid):4d} {mark}   {time.time()-t0:.0f}s", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(csv_path, index=False)
    lines = [f"conjuration detector ({backend}) on {len(d)} fleet rollouts, "
             f"capped at ctx + {span:g}s of generation", ""]
    lines.append(f"{'model':<16}{'clips':>6}{'flagged':>9}{'rate':>7}   scenes")
    for m, g in d.groupby("model"):
        fl = sorted(g.loc[g["flag"] == 1, "scene"].tolist())
        lines.append(f"{m:<16}{len(g):>6}{int(g['flag'].sum()):>9}{g['flag'].mean():>7.3f}   "
                     + (", ".join(fl[:12]) + (" ..." if len(fl) > 12 else "")))
    lines.append("")
    lines.append(f"TOTAL flagged: {int(d['flag'].sum())} / {len(d)}  ({d['flag'].mean():.3f})")
    txt = "\n".join(lines)
    open(os.path.join(OUTD, "summary.txt"), "w").write(txt + "\n")
    print("\n" + txt)

    for m, ps in hits.items():
        ims = [cv2.imread(p) for p in ps]
        w = max(i.shape[1] for i in ims)
        ims = [cv2.copyMakeBorder(i, 0, 6, 0, w - i.shape[1], cv2.BORDER_CONSTANT, value=(18, 18, 18)) for i in ims]
        cv2.imwrite(os.path.join(OUTD, f"popin_fleet_all_{m}.png"), np.concatenate(ims, 0))
    print(f"\nwrote {OUTD}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
