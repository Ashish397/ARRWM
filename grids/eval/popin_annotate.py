"""Build human-annotation material for pop-in on the blind100 set, and score it once filled in.

Annotating pop-in needs time, not a single frame -- so this produces two things:

  popin_annot_index.png        one 10x10 contact sheet, one tile per clip (last frame), for
                               orientation and for finding a clip by number
  popin_annot_strip_NN.png     ten sheets of ten clips, each clip an 8-frame filmstrip spanning
                               context -> end, which is what you actually judge from
  popin_annotations.csv        blank sheet to fill in

Deliberately unlabelled: no model name and no detector verdict appears anywhere in the
material, so the annotation stays blind and is not anchored by what any backbone decided.

Fill the `popin` column with 1 (an object pops into existence) or 0 (it does not), and leave
`unsure` set to 1 for anything genuinely ambiguous so it can be reported separately rather than
silently forced into a class. Then:

  python popin_annotate.py --score    # precision/recall/F1 per backbone against your labels

Usage:
  python popin_annotate.py            # build the sheets + blank csv
  python popin_annotate.py --score    # score the backbones once the csv is filled
"""
import os, sys
import numpy as np, cv2, imageio.v3 as iio, pandas as pd

import blind100_common as BC

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, "out", "popin_annot")
CSV = os.path.join(OUTD, "popin_annotations.csv")
PER_SHEET = 10
NFR = 8


def build():
    os.makedirs(OUTD, exist_ok=True)
    refs = sorted(BC.refs(), key=lambda r: r["blind_id"])
    TW, TH = 300, 168
    tiles, rows = [], []

    for i, r in enumerate(refs):
        vid = iio.imread(r["path"], plugin="pyav")
        n, ctx = len(vid), r["ctx"]
        # frames spread from the context boundary to the end: pop-in can only happen after ctx,
        # and the two context frames anchor what the scene really contained.
        ks = [0, max(0, ctx - 1)] + [min(n - 1, ctx + int(round((n - 1 - ctx) * f))) for f in
                                     (0.08, 0.2, 0.36, 0.55, 0.77, 1.0)]
        ks = ks[:NFR]
        strip = []
        for j, k in enumerate(ks):
            f = cv2.resize(vid[k], (TW, TH))
            tag = "ctx" if j <= 1 else f"f{k}"
            cv2.putText(f, tag, (4, 15), 0, 0.45, (0, 255, 0), 1)
            strip.append(f)
        body = np.concatenate(strip, 1)
        lab = np.full((26, body.shape[1], 3), 18, np.uint8)
        cv2.putText(lab, f"{r['blind_id']}    ctx={ctx}  n={n}", (6, 18), 0, 0.55, (255, 255, 255), 1)
        rows.append(np.concatenate([lab, body], 0))

        t = cv2.resize(vid[-1], (TW, TH))
        t = cv2.copyMakeBorder(t, 24, 3, 3, 3, cv2.BORDER_CONSTANT, value=(60, 60, 60))
        cv2.putText(t, r["blind_id"], (7, 17), 0, 0.5, (255, 255, 255), 1)
        tiles.append(t)
        print(f"{i+1:3d}/{len(refs)} {r['blind_id']}", flush=True)

    for s in range(0, len(rows), PER_SHEET):
        sheet = np.concatenate(rows[s:s + PER_SHEET], 0)
        p = os.path.join(OUTD, f"popin_annot_strip_{s//PER_SHEET+1:02d}.png")
        cv2.imwrite(p, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        print("wrote", p)

    W = 10
    grid = []
    blank = np.full_like(tiles[0], 18)
    for i in range(0, len(tiles), W):
        r = tiles[i:i + W]
        grid.append(np.concatenate(r + [blank] * (W - len(r)), 1))
    p = os.path.join(OUTD, "popin_annot_index.png")
    cv2.imwrite(p, cv2.cvtColor(np.concatenate(grid, 0), cv2.COLOR_RGB2BGR))
    print("wrote", p)

    if not os.path.exists(CSV):
        pd.DataFrame({"blind_id": [r["blind_id"] for r in refs],
                      "popin": "", "unsure": "", "notes": ""}).to_csv(CSV, index=False)
        print("wrote", CSV)
    else:
        print("kept existing", CSV)


def score():
    ann = pd.read_csv(CSV)
    ann = ann[ann["popin"].notna() & (ann["popin"].astype(str).str.strip() != "")]
    ann["popin"] = ann["popin"].astype(int)
    unsure = ann.get("unsure")
    if unsure is not None:
        ann["unsure"] = pd.to_numeric(ann["unsure"], errors="coerce").fillna(0).astype(int)
    else:
        ann["unsure"] = 0
    res = pd.read_csv(os.path.join(HERE, "out", "popin_backend_blind100.csv"))
    df = res.merge(ann, left_on="uid", right_on="blind_id", how="inner")
    sure = df[df["unsure"] == 0]
    print(f"annotated={len(df)}  usable(excl. unsure)={len(sure)}  positives={int(sure['popin'].sum())}\n")
    backends = [c[:-5] for c in df.columns if c.endswith("_flag")]
    print(f"{'backend':<12}{'TP':>4}{'FP':>4}{'FN':>4}{'TN':>4}{'prec':>7}{'rec':>7}{'F1':>7}")
    for b in backends:
        p = sure[f"{b}_flag"].astype(int); y = sure["popin"]
        tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
        fn = int(((p == 0) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
        pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
        f1 = 2 * pr * rc / max(1e-9, pr + rc)
        print(f"{b:<12}{tp:>4}{fp:>4}{fn:>4}{tn:>4}{pr:>7.2f}{rc:>7.2f}{f1:>7.2f}")
    print("\ndisagreements worth eyeballing:")
    for _, r in sure.iterrows():
        v = [int(r[f"{b}_flag"]) for b in backends]
        if len(set(v)) > 1 or v[0] != int(r["popin"]):
            print(f"  {r['uid']:<8} human={int(r['popin'])}  " +
                  "  ".join(f"{b}={int(r[f'{b}_flag'])}" for b in backends))


if __name__ == "__main__":
    score() if "--score" in sys.argv else build()
