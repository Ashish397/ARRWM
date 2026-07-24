"""Qwen3-VL-8B pairwise judge with LOGIT-based debiased scoring.

For each pair (a,b): run both presentation orders; in each, read the probability of
"A"/"B"/"T" at the verdict token. Debiased score for video x:
  p(x) = 0.5 * [P(A | x on left) + P(B | x on right)]
Continuous scores -> no forced ties; winner = argmax, tie iff |p(a)-p(b)| < eps.

Reuses PROMPTS and make_sxs_pack from vlm_judge.py.
Appends to vlm_judge_results.jsonl with model=qwen3vl8b and verdict {"overall", "p_a", "p_b"}.
"""
import argparse, itertools, json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from vlm_judge import PROMPTS, RULES, RULES_SXS, make_sxs_pack, read_frames, TILE_DIR

OUT = os.path.join(HERE, "vlm_judge_results.jsonl")


class Qwen3VL:
    name = "qwen3vl8b"

    def __init__(self):
        from transformers import AutoProcessor
        from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
        self.proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct", dtype=torch.bfloat16, device_map="cuda").eval()
        tok = self.proc.tokenizer
        self.ids = {c: tok.encode(c, add_special_tokens=False)[0] for c in "ABT"}

    @torch.no_grad()
    def probs(self, ims, prompt_text):
        """Return P(A), P(B), P(T) at the verdict position."""
        content = [{"type": "image", "image": im} for im in ims] + [
            {"type": "text", "text": prompt_text + '\n\nBegin your answer with exactly {"overall": "'}]
        messages = [{"role": "user", "content": content}]
        text = self.proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        # force the JSON prefix into the assistant turn so next token is the verdict letter
        text += '{"overall": "'
        inputs = self.proc(text=[text], images=ims, return_tensors="pt").to("cuda")
        logits = self.model(**inputs).logits[0, -1]
        sel = torch.tensor([self.ids["A"], self.ids["B"], self.ids["T"]], device=logits.device)
        p = torch.softmax(logits[sel], dim=0).float().cpu().numpy()
        return p  # [pA, pB, pT]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="direct_json")
    ap.add_argument("--pack", default="uniform24")
    ap.add_argument("--grids", default=None)
    ap.add_argument("--eps", type=float, default=0.05)
    args = ap.parse_args()

    gt = json.load(open(os.path.join(HERE, "gt.json")))["grids"]
    grids = args.grids.split(",") if args.grids else list(gt.keys())
    judge = Qwen3VL()

    prompt_text = PROMPTS[args.prompt]
    if prompt_text.startswith(RULES):
        prompt_text = RULES_SXS + prompt_text[len(RULES):]
    prompt_tag = f"{args.prompt}_sxs_logit"

    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            r = json.loads(line)
            done.add((r["model"], r["prompt"], r["pack"], r["grid"], r["a"], r["b"]))

    fout = open(OUT, "a")
    for grid in grids:
        variants = sorted(gt[grid].keys())
        raw = {}
        for v in variants:
            fp = os.path.join(TILE_DIR, f"{grid}__{v}.mp4")
            if not os.path.exists(fp):
                fp = os.path.join(HERE, "tiles_new", f"{grid}__{v}.mp4")
            raw[v] = read_frames(fp)
        for a, b in itertools.combinations(variants, 2):
            key = (judge.name, prompt_tag, args.pack, grid, a, b)
            if key in done:
                continue
            try:
                p1 = judge.probs(make_sxs_pack(raw[a], raw[b], args.pack), prompt_text)  # a left
                p2 = judge.probs(make_sxs_pack(raw[b], raw[a], args.pack), prompt_text)  # b left
            except Exception as e:
                print(f"ERR {grid} {a}v{b}: {e}", flush=True)
                continue
            pa = 0.5 * (p1[0] + p2[1])
            pb = 0.5 * (p1[1] + p2[0])
            overall = "T" if abs(pa - pb) < args.eps else ("A" if pa > pb else "B")
            rec = {"model": judge.name, "prompt": prompt_tag, "pack": args.pack,
                   "grid": grid, "a": a, "b": b,
                   "verdict": {"overall": overall, "p_a": round(float(pa), 4), "p_b": round(float(pb), 4)},
                   "raw": f"o1={np.round(p1,3).tolist()} o2={np.round(p2,3).tolist()}"}
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            print(f"{grid} {a} vs {b}: {overall} (pa={pa:.3f} pb={pb:.3f})", flush=True)
    fout.close()


ABS_PROMPT = """You are judging an AI-generated first-person driving video. The images are frames in temporal order; the final images labelled "CHANGE MAP" show frame-to-frame pixel change (widespread bright speckle = flickering/shimmering, a severe defect; edge-only brightness = normal motion).

Rate how plausible this video is as real driving footage, applying these rules:
- Structural corruption is the worst failure: vehicles collapsing into boxes, melted/mangled walls, jumbled chaotic texture, impossible geometry.
- Widespread shimmer is severe.
- Haze, fog, washed-out colour, softness are MINOR flaws.
- Hallucinated new objects are fine if well-formed.

Rate 1-5: 1 = severely corrupted, 2 = major structural problems, 3 = noticeable problems but scene mostly holds, 4 = minor flaws only (may be hazy/soft), 5 = plausible real footage."""


def absolute_main(pack="uniform24+diff"):
    import cv2
    from vlm_judge import make_pack
    gt = json.load(open(os.path.join(HERE, "gt.json")))["grids"]
    judge = Qwen3VL()
    tok = judge.proc.tokenizer
    rate_ids = [tok.encode(str(i), add_special_tokens=False)[0] for i in range(1, 6)]

    rows = []
    for grid, variants in gt.items():
        for v in sorted(variants):
            fp = os.path.join(TILE_DIR, f"{grid}__{v}.mp4")
            if not os.path.exists(fp):
                fp = os.path.join(HERE, "tiles_new", f"{grid}__{v}.mp4")
            frames = read_frames(fp)
            # single-video pack: reuse sxs composer against itself? no - plain frames + diffs
            n = len(frames)
            idx = np.linspace(0, n - 1, 24).round().astype(int)
            from PIL import Image
            ims = [Image.fromarray(cv2.resize(frames[i], (640, 352))) for i in idx]
            for t in [n // 2, 3 * n // 4, n - 2]:
                d = np.clip(np.abs(frames[t + 1].astype(np.int16) - frames[t].astype(np.int16)) * 3, 0, 255).astype(np.uint8)
                hdr = np.full((28, 640, 3), 32, np.uint8)
                cv2.putText(hdr, "CHANGE MAP", (240, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                ims.append(Image.fromarray(np.concatenate([hdr, cv2.resize(d, (640, 352))], axis=0)))
            content = [{"type": "image", "image": im} for im in ims] + [
                {"type": "text", "text": ABS_PROMPT + '\n\nAnswer with ONLY {"score": N}. Begin: {"score": '}]
            text = judge.proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False)
            text += '{"score": '
            inputs = judge.proc(text=[text], images=ims, return_tensors="pt").to("cuda")
            with torch.no_grad():
                logits = judge.model(**inputs).logits[0, -1]
            sel = torch.tensor(rate_ids, device=logits.device)
            p = torch.softmax(logits[sel], dim=0).float().cpu().numpy()
            score = float((p * np.arange(1, 6)).sum())
            rows.append({"model": judge.name, "prompt": "absolute_rating", "pack": pack,
                         "grid": grid, "a": v, "b": None,
                         "verdict": {"overall": None, "score": round(score, 3), "dist": np.round(p, 3).tolist()}})
            print(f"{grid} {v}: {score:.2f} {np.round(p,2).tolist()}", flush=True)
    with open(OUT, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")





REAL_PROMPT = """Look at this driving video (frames in temporal order; the first moments come from real footage, so pay most attention to how the scene evolves later). Does anything seem off to you? Is this the real world?

Check whether scene content stays believable: vehicles, buildings, people, vegetation, road geometry. IMPORTANT: fog, haze, washed-out colours and general softness are measured by a separate system - completely IGNORE them. Judge only whether the CONTENT could be real.

Could this be real dashcam footage? Answer Yes or No."""

MISTAKE_PROMPT = """This is an AI-generated driving video (frames in temporal order). There is likely at least one generation mistake somewhere in this video. Find the single worst mistake and describe it in one or two short sentences (what and where). IMPORTANT: fog, haze, washed-out colours and softness are handled by a separate system - do NOT report them; look for content errors: deformed vehicles or people, broken/melted structures, impossible geometry, objects changing identity.

After your description, rate how bad the mistake is on this scale:
1 = no real mistake found / trivial, 2 = small cosmetic error, 3 = clear error but scene still believable, 4 = major error, obviously artificial, 5 = reality-breaking corruption."""


def single_video_pack(frames, n_frames=24, size=(640, 352)):
    import cv2
    from PIL import Image
    idx = np.linspace(0, len(frames) - 1, n_frames).round().astype(int)
    return [Image.fromarray(cv2.resize(frames[i], size)) for i in idx]


def detection_main(mode):
    gt = json.load(open(os.path.join(HERE, "gt.json")))["grids"]
    judge = Qwen3VL()
    tok = judge.proc.tokenizer
    yes_id = tok.encode("Yes", add_special_tokens=False)[0]
    no_id = tok.encode("No", add_special_tokens=False)[0]
    rate_ids = [tok.encode(str(i), add_special_tokens=False)[0] for i in range(1, 6)]

    rows = []
    for grid, variants in gt.items():
        for v in sorted(variants):
            fp = os.path.join(TILE_DIR, f"{grid}__{v}.mp4")
            if not os.path.exists(fp):
                fp = os.path.join(HERE, "tiles_new", f"{grid}__{v}.mp4")
            ims = single_video_pack(read_frames(fp))
            if mode == "real_check":
                content = [{"type": "image", "image": im} for im in ims] + [
                    {"type": "text", "text": REAL_PROMPT + '\n\nAnswer with ONLY {"real": "Yes"} or {"real": "No"}.'}]
                text = judge.proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False)
                text += '{"real": "'
                inputs = judge.proc(text=[text], images=ims, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    logits = judge.model(**inputs).logits[0, -1]
                p = torch.softmax(logits[torch.tensor([yes_id, no_id], device=logits.device)], 0).float().cpu().numpy()
                score = float(p[0])  # P(real)
                extra = {"p_real": round(score, 4)}
                raw = ""
            else:  # find_mistake: generate description, then severity logits
                content = [{"type": "image", "image": im} for im in ims] + [{"type": "text", "text": MISTAKE_PROMPT}]
                text = judge.proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False)
                inputs = judge.proc(text=[text], images=ims, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    gen = judge.model.generate(**inputs, max_new_tokens=120, do_sample=False)
                desc = judge.proc.tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                text2 = text + desc + '\n\nSeverity rating: {"severity": '
                inputs2 = judge.proc(text=[text2], images=ims, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    logits = judge.model(**inputs2).logits[0, -1]
                p = torch.softmax(logits[torch.tensor(rate_ids, device=logits.device)], 0).float().cpu().numpy()
                sev = float((p * np.arange(1, 6)).sum())
                score = 6 - sev  # higher = more plausible
                extra = {"severity": round(sev, 3)}
                raw = desc[:300]
            rows.append({"model": judge.name, "prompt": mode, "pack": "det24",
                         "grid": grid, "a": v, "b": None,
                         "verdict": {"overall": None, "score": round(score, 4), **extra}, "raw": raw})
            print(f"{grid} {v}: {score:.3f} {extra} {raw[:80]}", flush=True)
    with open(OUT, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")





REF_PROMPT = """The first {nref} images are the REFERENCE: real dashcam footage that this AI world model was started from. The reference is by definition a 10/10 real video. The remaining images are the model's GENERATED CONTINUATION of that exact scene, in temporal order.

Judge how far the continuation departs from the reference in CONTENT PLAUSIBILITY. Fog, haze, washed-out colours and softness are measured elsewhere - completely IGNORE them. Look only at whether the content stays as believable as the reference: vehicles, buildings, people, vegetation, road geometry keeping real, coherent shapes.

Every generated continuation deviates from its reference at least slightly - there is always something. Scan carefully (including image edges and the final frames) and name the single worst content deviation in one or two short sentences: what object, where, and what happens to it."""

PROBES = [
    ("notice", "Would a casual viewer, glancing at the continuation for one second, notice something visually wrong or unreal? Answer Yes or No."),
    ("major", "Does any major object (vehicle, building, person) lose its believable shape at any point in the continuation? Answer Yes or No."),
    ("persist", "Is the worst deviation persistent or growing over time (rather than a brief momentary glitch)? Answer Yes or No."),
]


def ref_anchor_main():
    import cv2
    from PIL import Image
    gt = json.load(open(os.path.join(HERE, "gt.json")))["grids"]
    judge = Qwen3VL()
    tok = judge.proc.tokenizer
    yes_id = tok.encode("Yes", add_special_tokens=False)[0]
    no_id = tok.encode("No", add_special_tokens=False)[0]

    rows = []
    for grid, variants in gt.items():
        for v in sorted(variants):
            fp = os.path.join(TILE_DIR, f"{grid}__{v}.mp4")
            if not os.path.exists(fp):
                fp = os.path.join(HERE, "tiles_new", f"{grid}__{v}.mp4")
            frames = read_frames(fp)
            n = len(frames)
            ref_idx = np.linspace(0, 15, 6).round().astype(int)
            gen_idx = np.linspace(16, n - 1, 20).round().astype(int)
            def mk(i, label):
                img = cv2.resize(frames[i], (640, 352))
                hdr = np.full((26, 640, 3), 32, np.uint8)
                cv2.putText(hdr, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                return Image.fromarray(np.concatenate([hdr, img], axis=0))
            ims = [mk(i, "REFERENCE") for i in ref_idx] + [mk(i, "GENERATED") for i in gen_idx]
            content = [{"type": "image", "image": im} for im in ims] + [
                {"type": "text", "text": REF_PROMPT.format(nref=len(ref_idx))}]
            base_text = judge.proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False)
            inputs = judge.proc(text=[base_text], images=ims, return_tensors="pt").to("cuda")
            with torch.no_grad():
                gen = judge.model.generate(**inputs, max_new_tokens=90, do_sample=False)
            desc = judge.proc.tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            probs = {}
            for key, q in PROBES:
                t2 = base_text + desc + f"\n\n{q}\nAnswer: {{\"answer\": \""
                i2 = judge.proc(text=[t2], images=ims, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    lg = judge.model(**i2).logits[0, -1]
                p = torch.softmax(lg[torch.tensor([yes_id, no_id], device=lg.device)], 0).float().cpu().numpy()
                probs[key] = float(p[0])
            # deviation-calibrated score: start at 10, subtract behavioral evidence
            score = 10 - (4.5 * probs["notice"] + 3.0 * probs["major"] + 1.5 * probs["persist"])
            rows.append({"model": judge.name, "prompt": "ref_anchor2", "pack": "ref6+gen20",
                         "grid": grid, "a": v, "b": None,
                         "verdict": {"overall": None, "score": round(score, 3), **{f"p_{k}": round(pv, 3) for k, pv in probs.items()}},
                         "raw": desc[:300]})
            print(f"{grid} {v}: score={score:.2f} probes={ {k: round(pv,2) for k,pv in probs.items()} } :: {desc[:80]}", flush=True)
    with open(OUT, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")





GRID_PROMPT = """These images show SEVEN AI-generated driving videos (labelled {labels}) playing simultaneously, one panel each, in temporal order. All seven were generated from the SAME real context frames and the SAME action sequence, so they should show the same scene evolving.

Compare them RELATIONALLY and score each video's plausibility as real driving footage, 1-10, using the full range (they differ in quality - do not give them all similar scores). Judge content plausibility: vehicles, buildings, people and road geometry keeping real, coherent shapes; no melting, mangling, boxification, jumbled textures, or impossible geometry. IMPORTANT: fog, haze, washed-out colours and softness are measured elsewhere - completely IGNORE them.

Answer with ONLY a JSON object scoring every video, e.g.:
{{"A": 8, "B": 3, "C": 6, "D": 9, "E": 2, "F": 7, "G": 5}}"""


def grid_rank_main(n_perm=2, n_t=14):
    import cv2, random, re
    from PIL import Image
    gt = json.load(open(os.path.join(HERE, "gt.json")))["grids"]
    judge = Qwen3VL()
    rng = random.Random(7)
    letters = list("ABCDEFG")

    rows = []
    for grid, variants in gt.items():
        vs = sorted(variants)
        raws = {}
        for v in vs:
            fp = os.path.join(TILE_DIR, f"{grid}__{v}.mp4")
            if not os.path.exists(fp):
                fp = os.path.join(HERE, "tiles_new", f"{grid}__{v}.mp4")
            raws[v] = read_frames(fp)
        n = min(len(r) for r in raws.values())
        idx = np.linspace(0, n - 1, n_t).round().astype(int)
        scores_acc = {v: [] for v in vs}
        for perm_i in range(n_perm):
            order = vs[:]
            rng.shuffle(order)
            mapping = dict(zip(letters, order))
            ims = []
            for t in idx:
                panels = []
                for L in letters:
                    p = cv2.resize(raws[mapping[L]][t], (392, 210))
                    hdr = np.full((30, 392, 3), 32, np.uint8)
                    cv2.putText(hdr, L, (180, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                    panels.append(np.concatenate([hdr, p], axis=0))
                row1 = np.concatenate(panels[:4], axis=1)
                row2 = np.concatenate(panels[4:] + [np.zeros_like(panels[0])], axis=1)
                ims.append(Image.fromarray(np.concatenate([row1, row2], axis=0)))
            prompt = GRID_PROMPT.format(labels=", ".join(letters))
            content = [{"type": "image", "image": im} for im in ims] + [{"type": "text", "text": prompt}]
            text = judge.proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False)
            inputs = judge.proc(text=[text], images=ims, return_tensors="pt").to("cuda")
            with torch.no_grad():
                gen = judge.model.generate(**inputs, max_new_tokens=120, do_sample=False)
            ans = judge.proc.tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            m = re.search(r"\{[^{}]*\}", ans, re.S)
            if not m:
                print(f"{grid} perm{perm_i}: PARSE FAIL {ans[:120]}", flush=True)
                continue
            try:
                d = json.loads(m.group(0))
            except json.JSONDecodeError:
                print(f"{grid} perm{perm_i}: JSON FAIL {ans[:120]}", flush=True)
                continue
            for L, s in d.items():
                if L in mapping:
                    scores_acc[mapping[L]].append(float(s))
            print(f"{grid} perm{perm_i}: " + " ".join(f"{mapping[L]}={d.get(L)}" for L in letters), flush=True)
        for v, ss in scores_acc.items():
            if ss:
                rows.append({"model": judge.name, "prompt": "grid_rank", "pack": f"mosaic{n_t}",
                             "grid": grid, "a": v, "b": None,
                             "verdict": {"overall": None, "score": round(float(np.mean(ss)), 3), "n": len(ss)},
                             "raw": ""})
    with open(OUT, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "absolute":
        absolute_main()
    elif len(sys.argv) > 1 and sys.argv[1] in ("real_check", "find_mistake"):
        detection_main(sys.argv[1])
    elif len(sys.argv) > 1 and sys.argv[1] == "ref_anchor":
        ref_anchor_main()
    elif len(sys.argv) > 1 and sys.argv[1] == "grid_rank":
        grid_rank_main()
    else:
        main()
