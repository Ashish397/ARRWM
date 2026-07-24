"""Triplet / quadruplet VLM judging for the low-frequency (content plausibility) axis.

Efficiency designs over 7 variants:
  triplet: Fano plane (7 blocks, every pair covered exactly once)  -> 7 queries/grid
  quad:    greedy covering design (~5 blocks, all 21 pairs covered) -> 5 queries/grid
Each query shows the k videos side-by-side per timestep (labels A/B/C[/D]), asks for a
full plausibility ranking, and is repeated with a second label permutation; a pairwise
relation is kept only if both permutations agree, else tie.

Usage: multi_judge.py triplet|quad [n_frames]
Appends to vlm_judge_results.jsonl as model=qwen3vl8b prompt=triplet_rank|quad_rank
(one record per derived pair, verdict {"overall": A|B|T} keyed like pairwise records).
"""
import itertools, json, os, random, re, sys
import cv2
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from vlm_judge import read_frames, TILE_DIR

OUT = os.path.join(HERE, "vlm_judge_results.jsonl")

FANO = [(0, 1, 3), (1, 2, 4), (2, 3, 5), (3, 4, 6), (4, 5, 0), (5, 6, 1), (6, 0, 2)]


def quad_cover(n=7):
    """Greedy covering of all pairs by 4-subsets."""
    pairs = set(itertools.combinations(range(n), 2))
    blocks = []
    while pairs:
        best, bestcov = None, -1
        for c in itertools.combinations(range(n), 4):
            cov = sum(1 for p in itertools.combinations(c, 2) if p in pairs)
            if cov > bestcov:
                best, bestcov = c, cov
        blocks.append(best)
        for p in itertools.combinations(best, 2):
            pairs.discard(p)
    return blocks


PROMPT = """These images show {k} AI-generated driving videos (labelled {labels}) playing simultaneously, one panel each, in temporal order. All were generated from the SAME real context frames and the SAME action sequence.

Rank them by plausibility as real driving footage, judging CONTENT ONLY: vehicles, buildings, people and road geometry keeping real, coherent shapes; no melting, mangling, boxification, jumbled textures, or impossible geometry. Fog, haze, washed-out colours and softness are measured elsewhere - completely IGNORE them. A hallucinated but well-formed object is fine; a crisp but structurally corrupted object is a major failure.

Additionally, for each video judge whether a STYLE SHIFT occurs within it: does the visual style of its final second (colour palette, tone, texture character, art style) depart from the style of its first second? List the labels of videos where you perceive a style shift (empty list if none).

Answer with ONLY JSON, e.g.:
{{"ranking": [{example}], "style_shift": []}}"""


class Qwen3:
    def __init__(self):
        from transformers import AutoProcessor
        from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
        self.proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct", dtype=torch.bfloat16, device_map="cuda").eval()

    @torch.no_grad()
    def rank(self, ims, prompt):
        content = [{"type": "image", "image": im} for im in ims] + [{"type": "text", "text": prompt}]
        text = self.proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False)
        inputs = self.proc(text=[text], images=ims, return_tensors="pt").to("cuda")
        gen = self.model.generate(**inputs, max_new_tokens=60, do_sample=False)
        return self.proc.tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def compose(frame_list, labels, size=(432, 232)):
    panels = []
    for f, L in zip(frame_list, labels):
        img = cv2.resize(f, size)
        hdr = np.full((30, size[0], 3), 32, np.uint8)
        cv2.putText(hdr, L, (size[0] // 2 - 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        panels.append(np.concatenate([hdr, img], axis=0))
    if len(panels) <= 3:
        return Image.fromarray(np.concatenate(panels, axis=1))
    row1 = np.concatenate(panels[:2], axis=1)
    row2 = np.concatenate(panels[2:4], axis=1)
    return Image.fromarray(np.concatenate([row1, row2], axis=0))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "triplet"
    n_t = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    k = 3 if mode == "triplet" else 4
    blocks = FANO if mode == "triplet" else quad_cover()
    letters = list("ABCD")[:k]
    tag = f"{mode}_rank"

    gt = json.load(open(os.path.join(HERE, "gt.json")))["grids"]
    judge = Qwen3()
    rng = random.Random(11)

    fout = open(OUT, "a")
    for grid, variants in gt.items():
        vs = sorted(variants)
        raw = {}
        for v in vs:
            fp = os.path.join(TILE_DIR, f"{grid}__{v}.mp4")
            if not os.path.exists(fp):
                fp = os.path.join(HERE, "tiles_new", f"{grid}__{v}.mp4")
            raw[v] = read_frames(fp)
        n = min(len(r) for r in raw.values())
        idx = np.linspace(0, n - 1, n_t).round().astype(int)
        # collect relations: (a,b) -> list of outcomes over perms/blocks
        rel = {}
        style_votes = {}
        n_calls = 0
        for block in blocks:
            members = [vs[i] for i in block]
            perm_orders = []
            for perm_i in range(2):
                order = members[:]
                rng.shuffle(order)
                mapping = dict(zip(letters, order))
                ims = [compose([raw[mapping[L]][t] for L in letters], letters) for t in idx]
                prompt = PROMPT.format(k=k, labels="/".join(letters),
                                       example=", ".join(f'"{L}"' for L in letters))
                try:
                    ans = judge.rank(ims, prompt)
                    n_calls += 1
                except Exception as e:
                    print(f"ERR {grid} {block}: {e}", flush=True)
                    continue
                mr = re.search(r'"ranking"\s*:\s*\[([^\]]*)\]', ans)
                if not mr:
                    print(f"{grid} {block} perm{perm_i}: PARSE FAIL {ans[:80]}", flush=True)
                    continue
                ranked = [mapping[L] for L in re.findall(r'[A-D]', mr.group(1)) if L in mapping]
                if len(set(ranked)) != k:
                    print(f"{grid} {block} perm{perm_i}: BAD RANK {ans[:80]}", flush=True)
                    continue
                perm_orders.append(ranked)
                ms = re.search(r'"style_shift"\s*:\s*\[([^\]]*)\]', ans)
                shifted = [mapping[L] for L in re.findall(r'[A-D]', ms.group(1)) if L in mapping] if ms else []
                for v_m in members:
                    style_votes.setdefault(v_m, []).append(1 if v_m in shifted else 0)
                print(f"{grid} {members} perm{perm_i}: {ranked} shift={shifted}", flush=True)
            # derive pair relations agreed by both perms
            for a, b in itertools.combinations(members, 2):
                outs = []
                for ranked in perm_orders:
                    outs.append(a if ranked.index(a) < ranked.index(b) else b)
                if not outs:
                    continue
                w = outs[0] if len(outs) == 1 or outs[0] == outs[1] else "T"
                key = tuple(sorted((a, b)))
                rel.setdefault(key, w)
        for (a, b), w in rel.items():
            verdict = {"overall": "A" if w == a else ("B" if w == b else "T")}
            fout.write(json.dumps({"model": "qwen3vl8b", "prompt": tag, "pack": f"nup{n_t}",
                                   "grid": grid, "a": a, "b": b, "verdict": verdict, "raw": ""}) + "\n")
        for v, votes in style_votes.items():
            fout.write(json.dumps({"model": "qwen3vl8b", "prompt": f"{tag}_style", "pack": f"nup{n_t}",
                                   "grid": grid, "a": v, "b": None,
                                   "verdict": {"overall": None, "score": round(float(np.mean(votes)), 3), "n": len(votes)},
                                   "raw": ""}) + "\n")
        fout.flush()
        print(f"{grid}: {len(rel)} pair relations from {n_calls} calls", flush=True)
    fout.close()


if __name__ == "__main__":
    main()
