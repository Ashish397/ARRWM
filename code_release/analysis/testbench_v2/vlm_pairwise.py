"""VLM pairwise judge — the ONLY VLM configuration that worked in V1:

  - Two videos rendered SIDE-BY-SIDE per timestep (A left / B right),
    ~24 composites sampled at matching wall-clock fractions of each video's
    generated span, plus 3 |frame-diff| change-map composites.
  - A priority-rules prompt (structural corruption dominates; sharpness does
    not excuse mangling; haze/softness is minor — measured elsewhere;
    hallucinated content is fine if well-formed).
  - Logit scoring: force the reply prefix '{"overall": "' and read
    P(A)/P(B) at the next token; average over both left/right orders.
    [V1 8B judges: InternVL3.5 68.4%, Qwen3-VL 67.8% — below the ~80%
    composite. V2 upgrade 1 = same protocol with 30B-72B judges in bf16;
    adopt only if it clearly beats the composite.]

Configurations evaluated and rejected for <=8B judges: sequential non-side-by-side
frames (89% position bias), generated verdict letters (collapse under prompt
edits), 7-way mosaics (chance), triplet/quadruplet designs (54-61%),
absolute 1-5 ratings (saturate), severity ratings (saturate).

Modes:
  --mode pair  : judge pairs from a CSV (vid_a,vid_b paths or vids)
  --mode style : per-video first-second vs last-second style question
                 (instrument c — untrusted until validated on a restyled model)

Judge model via TB2_JUDGE (default Qwen2.5-VL-7B locally cached; for V2
upgrade 1 point it at Qwen2.5-VL-32B/72B-Instruct or Qwen3-VL, bf16,
device_map=auto across GPUs).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fleet

N_STEPS = 24            # side-by-side composites per pair
N_CHANGE = 3            # |frame-diff| change-map composites
TILE_W, TILE_H = 416, 224
GAP = 8

PAIR_PROMPT = (
    "You are judging two AI-generated driving videos, shown side-by-side in "
    "each image: video A is always the LEFT half, video B is always the RIGHT "
    "half. The image sequence follows both videos through time; the final "
    f"{N_CHANGE} images are |frame difference| change maps (bright = motion/"
    "change).\n"
    "Decide which video is the more plausible, better-formed continuation.\n"
    "Priority rules:\n"
    "1. Structural corruption dominates: melted, warped, smeared or "
    "geometrically impossible buildings, roads, vehicles or people are the "
    "worst failure.\n"
    "2. Sharpness does not excuse mangling: a sharp but structurally corrupt "
    "video loses to a soft but coherent one.\n"
    "3. Haze or softness is a minor issue — it is measured elsewhere; ignore "
    "it here.\n"
    "4. Hallucinated new content is fine if it is well-formed and plausible.\n"
    'Answer with JSON: {"overall": "A"} or {"overall": "B"}.'
)
REPLY_PREFIX = '{"overall": "'

STYLE_PROMPT = (
    "The first row of images is the FIRST second of a driving video; the "
    "second row is the LAST second of the same video. Ignore scene content "
    "changes (the camera moved). Question: does the LAST second's visual "
    "STYLE depart from the FIRST second's style — e.g. photoreal footage "
    "turning into a video-game / cartoon / painterly / synthetic look, a "
    "global palette or texture change not explained by camera motion?\n"
    'Answer with JSON: {"style_shift": "yes"} or {"style_shift": "no"}.'
)
STYLE_PREFIX = '{"style_shift": "'


# ------------------------------------------------------------------ imaging
def _label(img, text):
    import cv2
    img = img.copy()
    cv2.rectangle(img, (0, 0), (30, 24), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)
    return img


def _sbs(left, right):
    import cv2
    l = cv2.resize(left, (TILE_W, TILE_H))
    r = cv2.resize(right, (TILE_W, TILE_H))
    gap = np.full((TILE_H, GAP, 3), 255, np.uint8)
    return np.concatenate([_label(l, "A"), gap, _label(r, "B")], axis=1)


def pair_images(va, vb):
    """(frames,times) tuples -> list of uint8 composites (A left, B right)."""
    fa, ta = va
    fb, tb = vb
    tsa = fleet.gen_fraction_times(ta, N_STEPS)
    tsb = fleet.gen_fraction_times(tb, N_STEPS)
    imgs = [_sbs(fleet.frame_at(fa, ta, x), fleet.frame_at(fb, tb, y))
            for x, y in zip(tsa, tsb)]
    # change maps at early / mid / late generation
    for frac in np.linspace(0.15, 0.85, N_CHANGE):
        def cm(f, t, ts):
            i = int(frac * (len(ts) - 2))
            a = fleet.frame_at(f, t, ts[i]).astype(np.int16)
            b = fleet.frame_at(f, t, ts[i + 1]).astype(np.int16)
            d = np.abs(a - b)
            return (np.clip(d * (255.0 / max(1, d.max())), 0, 255)).astype(np.uint8)
        imgs.append(_sbs(cm(fa, ta, tsa), cm(fb, tb, tsb)))
    return imgs


def style_images(frames, times, n=8):
    """First-second strip stacked above last-second strip."""
    import cv2
    ctx, _, end = fleet.windows(times)
    def strip(idx):
        pick = idx[np.linspace(0, len(idx) - 1, min(n, len(idx))).astype(int)]
        tiles = [cv2.resize(frames[i], (TILE_W // 2, TILE_H // 2)) for i in pick]
        return np.concatenate(tiles, axis=1)
    top, bot = strip(ctx), strip(end)
    w = min(top.shape[1], bot.shape[1])
    gap = np.full((GAP, w, 3), 255, np.uint8)
    return [np.concatenate([top[:, :w], gap, bot[:, :w]], axis=0)]


# -------------------------------------------------------------------- judge
class HFJudge:
    """Any HF image-text-to-text judge: Qwen2.5-VL / Qwen3-VL (incl. the
    30B-A3B MoE) and InternVL3.5 *-HF repos all load via AutoProcessor +
    AutoModelForImageTextToText. bf16, device_map=auto so 30B-72B judges
    shard across GPUs. InternVL note: dynamic tiling has spiky memory —
    run with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True."""

    def __init__(self, model_id):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="auto")
        self.model.eval()

    def _tok_ids(self, variants):
        ids = []
        for v in variants:
            t = self.proc.tokenizer.encode(v, add_special_tokens=False)
            if len(t) >= 1:
                ids.append(t[0])
        return sorted(set(ids))

    @torch.no_grad()
    def choice_probs(self, images, prompt, prefix, options):
        """P(option) at the token right after the forced reply prefix."""
        from PIL import Image
        content = [{"type": "image", "image": Image.fromarray(im)} for im in images]
        content.append({"type": "text", "text": prompt})
        msgs = [{"role": "user", "content": content}]
        text = self.proc.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True) + prefix
        inputs = self.proc(text=[text],
                           images=[Image.fromarray(im) for im in images],
                           return_tensors="pt").to(self.model.device)
        logits = self.model(**inputs).logits[0, -1].float()
        lse = {o: torch.logsumexp(logits[self._tok_ids([o, o.lower(), " " + o])], 0)
               for o in options}
        z = torch.stack(list(lse.values()))
        p = torch.softmax(z, 0)
        return {o: float(p[i]) for i, o in enumerate(lse)}


def judge_pair(judge, va, vb):
    """Both-order averaged P(first arg wins)."""
    p1 = judge.choice_probs(pair_images(va, vb), PAIR_PROMPT,
                            REPLY_PREFIX, ("A", "B"))["A"]
    p2 = judge.choice_probs(pair_images(vb, va), PAIR_PROMPT,
                            REPLY_PREFIX, ("A", "B"))["B"]
    return 0.5 * (p1 + p2)


# ---------------------------------------------------------------------- CLI
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("pair", "style"), required=True)
    ap.add_argument("--pairs", default=os.path.join(here, "out", "judge_pairs.csv"),
                    help="pair mode: CSV with columns path_a,path_b,vid_a,vid_b")
    ap.add_argument("--models", default="", help="style mode: comma model filter")
    ap.add_argument("--out", default="")
    ap.add_argument("--shard", type=int, default=int(os.environ.get("TB2_SHARD", 0)))
    ap.add_argument("--nshard", type=int, default=int(os.environ.get("TB2_NSHARD", 1)))
    args = ap.parse_args()

    model_id = os.environ.get("TB2_JUDGE", "Qwen/Qwen2.5-VL-7B-Instruct")
    judge = HFJudge(model_id)
    out = args.out or os.path.join(
        here, "out", f"vlm_{args.mode}.csv" if args.nshard == 1 else
        f"vlm_{args.mode}.shard{args.shard}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []

    if args.mode == "pair":
        pairs = pd.read_csv(args.pairs)
        pairs = pairs.iloc[args.shard::args.nshard]
        cache = {}
        def load(p):
            if p not in cache:
                if len(cache) > 8:
                    cache.clear()
                f, t, _ = fleet.load_video(p)
                cache[p] = (f, t)
            return cache[p]
        for _, r in pairs.iterrows():
            try:
                p_a = judge_pair(judge, load(r.path_a), load(r.path_b))
                rows.append(dict(vid_a=r.vid_a, vid_b=r.vid_b, p_a=p_a,
                                 judge=model_id))
                print(f"{r.vid_a} vs {r.vid_b}: P(A)={p_a:.3f}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[skip] {r.vid_a} vs {r.vid_b}: {e}", flush=True)
            if len(rows) % 10 == 0:
                pd.DataFrame(rows).to_csv(out, index=False)
    else:
        models = [m for m in args.models.split(",") if m]
        refs = fleet.discover_fleet(models or None)
        refs = refs[args.shard::args.nshard]
        for ref in refs:
            try:
                f, t, _ = fleet.load_video(ref.path)
                p = judge.choice_probs(style_images(f, t), STYLE_PROMPT,
                                       STYLE_PREFIX, ("yes", "no"))["yes"]
                rows.append(dict(vid=ref.vid, model=ref.model, scene=ref.scene,
                                 direction=ref.direction, vlm_style_pyes=p,
                                 judge=model_id))
            except Exception as e:  # noqa: BLE001
                print(f"[skip] {ref.vid}: {e}", flush=True)
            if len(rows) % 20 == 0:
                print(f"{len(rows)} done", flush=True)
                pd.DataFrame(rows).to_csv(out, index=False)

    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
