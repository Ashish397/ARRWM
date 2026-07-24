"""InternVL3.5-8B pairwise judge, same protocol as qwen3_judge:
side-by-side composites + change maps, rules_v2 prompt, logit A/B scoring
debiased over both presentation orders.

Appends to vlm_judge_results.jsonl with model=internvl35_8b.
"""
import itertools, json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from vlm_judge import PROMPTS, RULES, RULES_SXS, make_sxs_pack, read_frames, TILE_DIR

OUT = os.path.join(HERE, "vlm_judge_results.jsonl")
MODEL = "OpenGVLab/InternVL3_5-8B-HF"


class InternVL:
    name = "internvl35_8b"

    def __init__(self):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.proc = AutoProcessor.from_pretrained(MODEL)
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
        tok = self.proc.tokenizer
        self.ids = [tok.encode(c, add_special_tokens=False)[0] for c in "ABT"]

    @torch.no_grad()
    def probs(self, ims, prompt_text):
        content = [{"type": "image", "image": im} for im in ims] + [
            {"type": "text", "text": prompt_text + '\n\nBegin your answer with exactly {"overall": "'}]
        messages = [{"role": "user", "content": content}]
        text = self.proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        text += '{"overall": "'
        inputs = self.proc(text=[text], images=ims, return_tensors="pt").to("cuda")
        logits = self.model(**inputs).logits[0, -1]
        sel = torch.tensor(self.ids, device=logits.device)
        return torch.softmax(logits[sel], dim=0).float().cpu().numpy()


def main():
    prompt_name = sys.argv[1] if len(sys.argv) > 1 else "rules_v2"
    pack = sys.argv[2] if len(sys.argv) > 2 else "uniform24+diff"
    gt = json.load(open(os.path.join(HERE, "gt.json")))["grids"]
    judge = InternVL()
    prompt_text = PROMPTS[prompt_name]
    if prompt_text.startswith(RULES):
        prompt_text = RULES_SXS + prompt_text[len(RULES):]
    prompt_tag = f"{prompt_name}_sxs_logit"

    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            r = json.loads(line)
            done.add((r["model"], r["prompt"], r["pack"], r["grid"], r["a"], r["b"]))

    fout = open(OUT, "a")
    for grid, variants in gt.items():
        vs = sorted(variants)
        raw = {}
        for v in vs:
            fp = os.path.join(TILE_DIR, f"{grid}__{v}.mp4")
            if not os.path.exists(fp):
                fp = os.path.join(HERE, "tiles_new", f"{grid}__{v}.mp4")
            raw[v] = read_frames(fp)
        for a, b in itertools.combinations(vs, 2):
            key = (judge.name, prompt_tag, pack, grid, a, b)
            if key in done:
                continue
            try:
                p1 = judge.probs(make_sxs_pack(raw[a], raw[b], pack), prompt_text)
                p2 = judge.probs(make_sxs_pack(raw[b], raw[a], pack), prompt_text)
            except Exception as e:
                print(f"ERR {grid} {a}v{b}: {e}", flush=True)
                continue
            pa = 0.5 * (p1[0] + p2[1])
            pb = 0.5 * (p1[1] + p2[0])
            overall = "T" if abs(pa - pb) < 0.05 else ("A" if pa > pb else "B")
            fout.write(json.dumps({"model": judge.name, "prompt": prompt_tag, "pack": pack,
                                   "grid": grid, "a": a, "b": b,
                                   "verdict": {"overall": overall, "p_a": round(float(pa), 4), "p_b": round(float(pb), 4)},
                                   "raw": ""}) + "\n")
            fout.flush()
            print(f"{grid} {a} vs {b}: {overall} (pa={pa:.3f} pb={pb:.3f})", flush=True)
    fout.close()


if __name__ == "__main__":
    main()
