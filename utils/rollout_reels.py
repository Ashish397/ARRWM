"""Example-rollout filmstrips for the appendix.

For each phase-A rollout of a model, score it on (a) action following and
(b) visual quality, rank, and render the best as filmstrips (6 evenly-spaced
frames side by side). Also builds a contact sheet of the top-K so a human can
cherry-pick.

Following score: mean |corr| of realized vs commanded on the axis the command
drives (corr_z2 for throttle dirs, corr_z7 for steer dirs, both for diagonals),
from the inject_eval JSONL.
Quality score: end-of-rollout Laplacian sharpness relative to the seed (no
collapse) + absolute sharpness (crisp).

Env: RR_MODELS (colon list, def "16node:pca8_8node"), RR_TOPK (def 24),
RR_DIRS (def all), RR_OUT (def analysis/reels).
"""
import os, json, glob
import numpy as np
import av
from PIL import Image, ImageDraw

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
MODELS = os.environ.get("RR_MODELS", "16node:pca8_8node").split(":")
TOPK = int(os.environ.get("RR_TOPK", "24"))
DIR_FILTER = set(os.environ.get("RR_DIRS", "").split(",")) - {""}
OUT = os.environ.get("RR_OUT", f"{ARR}/analysis/reels")
os.makedirs(OUT, exist_ok=True)
NFB_PX = 12
DIR_AXIS = {"F": "t", "B": "t", "R": "s", "L": "s",
            "FR": "b", "FL": "b", "BR": "b", "BL": "b"}
FRAME_PICKS = [6, 24, 48, 72, 96, 107]   # seed + chunks 1,2,4,6,8


def read_frames(path, keep=None):
    c = av.open(path)
    out = []
    for i, f in enumerate(c.decode(c.streams.video[0])):
        if keep is None or i in keep:
            out.append((i, np.asarray(f.to_image())))
    if keep is None:
        return [a for _, a in out]
    return {i: a for i, a in out}


def following(rec, d):
    # phase-A commands are CONSTANT, so correlation is undefined; use realized
    # motion projected onto the (constant) commanded direction over settled chunks.
    tz2 = np.array(rec["tz2"][1:]); tz7 = np.array(rec["tz7"][1:])
    cz2 = np.mean(rec["cz2"]); cz7 = np.mean(rec["cz7"])
    n = np.hypot(cz2, cz7) + 1e-6
    proj = (tz2 * cz2 + tz7 * cz7) / n           # realized speed along command
    return float(proj.mean())


def _sharp(f):
    g = np.asarray(Image.fromarray(f).convert("L"), float)
    return np.abs(np.diff(g, 2, 0)).var()


QUAL_IDX = [2, 6, 9, 96, 99, 102, 105, 107]   # seed + last chunk, for scoring


def quality(fd):
    seed = np.mean([_sharp(fd[i]) for i in (2, 6, 9) if i in fd])
    endk = [i for i in (96, 99, 102, 105, 107) if i in fd]
    end = np.mean([_sharp(fd[i]) for i in endk])
    keep = end / max(seed, 1e-6)               # 1 = no sharpness lost (unclamped)
    return keep, end


def filmstrip(fd, label):
    picks = [fd[i] for i in FRAME_PICKS if i in fd]
    h = picks[0].shape[0]
    pad = 4
    strip = np.full((h, sum(p.shape[1] for p in picks) + pad * (len(picks) - 1), 3), 255, np.uint8)
    x = 0
    for p in picks:
        strip[:, x:x + p.shape[1]] = p
        x += p.shape[1] + pad
    im = Image.fromarray(strip)
    ImageDraw.Draw(im).text((6, 6), label, fill=(255, 255, 0))
    return im


def main():
    scored = []
    for m in MODELS:
        vids = sorted(glob.glob(f"{ARR}/logs/eval_final/A/{m}/control_test/step05000_r*_*_raw.mp4"))
        rec_by = {}
        for jf in glob.glob(f"{ARR}/logs/eval_final/A/{m}/control_test/metrics_r*.jsonl"):
            for ln in open(jf):
                j = json.loads(ln)
                for k, v in j.items():
                    if isinstance(v, dict) and "corr_z2" in v:
                        rec_by[(j["rank"], k)] = v
        for p in vids:
            base = os.path.basename(p)
            wi = int(base.split("_r")[1].split("_")[0])
            d = base.split("_")[2]
            if DIR_FILTER and d not in DIR_FILTER:
                continue
            rec = rec_by.get((wi, d))
            if rec is None:
                continue
            fd = read_frames(p, keep=set(QUAL_IDX))
            fscore = following(rec, d)
            keep, end = quality(fd)
            # collapse/haze filter: drop rollouts that lost >25% sharpness or
            # ended soft; among survivors rank by following strength.
            # keep in [0.8, 2.0]: <0.8 lost sharpness (haze); >2.0 means the end
            # is far busier than the seed (spurious detail / artifact), not clean.
            if not (0.8 <= keep <= 2.0) or end < 300:
                continue
            scored.append((fscore, keep, end, m, wi, d, p))
    scored.sort(key=lambda x: -x[0])
    # dedupe by held-out scene (window): each distinct scene shows once, at its
    # best-following model+direction, for a diverse gallery.
    if os.environ.get("RR_DEDUPE", "1") == "1":
        seen, dedup = set(), []
        for row in scored:
            wi = row[4]
            if wi in seen:
                continue
            seen.add(wi); dedup.append(row)
        scored = dedup
    top = scored[:TOPK]
    print(f"[reels] {len(scored)} passed quality filter, rendering top {len(top)}")
    thumbs = []
    for rank, (f, keep, end, m, wi, d, p) in enumerate(top):
        lab = f"{m}  w{wi:02d}  {d}  follow={f:+.2f} keep={keep:.2f}"
        fd = read_frames(p, keep=set(FRAME_PICKS))
        im = filmstrip(fd, lab)
        fn = f"{OUT}/reel_{rank:02d}_{m}_w{wi:02d}_{d}.png"
        im.save(fn)
        thumbs.append((im, lab))
        print(f"  {rank:2d}  {lab}")
    # contact sheet: stack all filmstrips vertically at reduced width
    W = 1100
    rows = [t.resize((W, int(W * t.height / t.width))) for t, _ in thumbs]
    sheet = np.full((sum(r.height + 6 for r in rows), W, 3), 255, np.uint8)
    y = 0
    for r in rows:
        a = np.asarray(r)
        sheet[y:y + a.shape[0]] = a
        y += a.shape[0] + 6
    Image.fromarray(sheet).save(f"{OUT}/contact_sheet.png")
    print(f"[reels] contact sheet -> {OUT}/contact_sheet.png")


if __name__ == "__main__":
    main()
