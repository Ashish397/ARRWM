"""VLM mangle-judge TESTBENCH.

Registry of VLM judges scored on the standard eval set (r08_B, r08_BL, r01_R x 8
models + real refs) against by-eye ground truth. Each judge = adapter returning a
per-video mangle score (higher = more mangled). Standard protocol: sample gen
frames, upper-region focus where applicable, P(Yes)-style logprob scoring for QA
judges (numeric rubrics mode-collapse).

Env: VB_JUDGES colon list from {qwen7b, sa2va, internvl3, minicpm}; VB_RUNS models.
Appends/updates analysis/eval_final/vlm_bench.csv; prints validation vs anchors.
"""
import os, sys, gc
import numpy as np, pandas as pd, imageio, torch
import torch.nn.functional as F
from PIL import Image

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
DEV = "cuda"
RUNS = os.environ.get("VB_RUNS", "pca8_8node,pca4,pca2,16node,4node,noatok,noadaln,minwm").replace(":", ",").split(",")
JUDGES = os.environ.get("VB_JUDGES", "sa2va,internvl3").replace(":", ",").split(",")
TARGETS = [(8, "B"), (8, "BL"), (1, "R")]
GEN_IDX = [24, 40, 56, 72, 88, 104]
CLEAN_BL = {"pca8_8node", "4node"}
_MWSW = {"L": "R", "R": "L", "FL": "FR", "FR": "FL", "BL": "BR", "BR": "BL"}

MELT_Q = ("Are any buildings, walls or structures in this image melted, warped, smeared "
          "or geometrically impossible, like a corrupted AI-generated image? "
          "Answer only Yes or No.")


def vid_path(run, rank, br):
    if run == "minwm":
        return f"{ARR}/logs/eval_final/A_minwm/minwm_r{rank:02d}_{_MWSW.get(br, br)}.mp4"
    return f"{ARR}/logs/eval_final/A/{run}/control_test/step05000_r{rank:02d}_{br}_raw.mp4"


def frames_of(path, idxs):
    r = imageio.get_reader(path)
    out = []
    for i in idxs:
        try:
            out.append(np.asarray(r.get_data(i)))
        except Exception:
            break
    r.close(); return out


def upper_crops(img):
    H, W = img.shape[:2]
    h = int(H * 0.6); w2 = W // 2
    return [img[:h, :w2], img[:h, w2:]]


# ---------------- judge adapters (return per-video score) ----------------
def make_qwen7b():
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    mid = "Qwen/Qwen2.5-VL-7B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(mid, dtype=torch.bfloat16, device_map=DEV)
    proc = AutoProcessor.from_pretrained(mid)
    yes = [proc.tokenizer.encode(t, add_special_tokens=False)[0] for t in ("Yes", " Yes", "yes")]
    no = [proc.tokenizer.encode(t, add_special_tokens=False)[0] for t in ("No", " No", "no")]

    def score(frames):
        vals = []
        for f in frames:
            for c in upper_crops(f):
                im = Image.fromarray(c)
                msgs = [{"role": "user", "content": [{"type": "image", "image": im},
                                                     {"type": "text", "text": MELT_Q}]}]
                text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inputs = proc(text=[text], images=[im], return_tensors="pt").to(DEV)
                with torch.no_grad():
                    lg = model(**inputs).logits[0, -1]
                vals.append(float(torch.sigmoid(torch.logsumexp(lg[yes], 0) - torch.logsumexp(lg[no], 0)).item()))
        return float(np.mean(sorted(vals)[-4:]))
    return score


def make_internvl3():
    from transformers import AutoModel, AutoTokenizer
    mid = os.environ.get("VB_INTERNVL", "OpenGVLab/InternVL3-8B")
    model = AutoModel.from_pretrained(mid, torch_dtype=torch.bfloat16, trust_remote_code=True,
                                      low_cpu_mem_usage=False).to(DEV).eval()
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    import torchvision.transforms as T
    tf = T.Compose([T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def score(frames):
        vals = []
        for f in frames:
            for c in upper_crops(f):
                im = Image.fromarray(c).resize((448, 448))
                px = tf(im).unsqueeze(0).to(DEV, torch.bfloat16)
                q = "<image>\n" + MELT_Q
                with torch.no_grad():
                    ans = model.chat(tok, px, q, dict(max_new_tokens=4, do_sample=False))
                vals.append(1.0 if "yes" in ans.lower() else 0.0)
        return float(np.mean(sorted(vals)[-4:]))
    return score


def make_sa2va():
    from transformers import AutoModel, AutoTokenizer
    mid = os.environ.get("VB_SA2VA", "ByteDance/Sa2VA-4B")
    model = AutoModel.from_pretrained(mid, torch_dtype=torch.bfloat16, trust_remote_code=True,
                                      low_cpu_mem_usage=False).to(DEV).eval()
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    PROMPT = ("<image>Please segment the regions that look melted, warped, distorted "
              "or corrupted by AI generation artifacts.")

    def score(frames):
        fracs = []
        for f in frames:
            im = Image.fromarray(f)
            with torch.no_grad():
                ret = model.predict_forward(image=im, text=PROMPT, tokenizer=tok)
            masks = ret.get("prediction_masks", [])
            fr = 0.0
            for mk in masks:
                m = np.asarray(mk).astype(bool)
                if m.ndim == 3:
                    m = m[0]
                H = m.shape[0]
                fr = max(fr, float(m[: int(H * 0.6)].mean()))
            fracs.append(fr)
        return float(np.mean(sorted(fracs)[-3:]))
    return score


def make_minicpm():
    from transformers import AutoModel, AutoTokenizer
    mid = "openbmb/MiniCPM-V-2_6"
    model = AutoModel.from_pretrained(mid, torch_dtype=torch.bfloat16, trust_remote_code=True).to(DEV).eval()
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)

    def score(frames):
        vals = []
        for f in frames:
            for c in upper_crops(f):
                im = Image.fromarray(c)
                msgs = [{"role": "user", "content": [im, MELT_Q]}]
                with torch.no_grad():
                    ans = model.chat(image=None, msgs=msgs, tokenizer=tok, max_new_tokens=4, sampling=False)
                vals.append(1.0 if "yes" in str(ans).lower() else 0.0)
        return float(np.mean(sorted(vals)[-4:]))
    return score


REGISTRY = {"qwen7b": make_qwen7b, "internvl3": make_internvl3, "sa2va": make_sa2va, "minicpm": make_minicpm}


def main():
    import glob
    vids = []
    for rank, br in TARGETS:
        for run in RUNS:
            p = vid_path(run, rank, br)
            if os.path.exists(p):
                vids.append((f"r{rank:02d}{br}", run, p))
    for p in sorted(glob.glob(f"{ARR}/analysis/eval_final/real_refs/*.mp4"))[:6]:
        vids.append(("REAL", os.path.basename(p)[:18], p))

    out_csv = f"{ARR}/analysis/eval_final/vlm_bench.csv"
    old = pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame(columns=["judge", "window", "run", "score"])
    for j in JUDGES:
        try:
            scorer = REGISTRY[j]()
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[{j}] LOAD FAILED: {str(e)[:150]}", flush=True); continue
        rows = []
        for w, run, p in vids:
            try:
                s = scorer(frames_of(p, GEN_IDX))
            except Exception as e:
                print(f"[{j}] {w} {run} failed: {str(e)[:100]}", flush=True); s = np.nan
            rows.append(dict(judge=j, window=w, run=run, score=round(s, 4) if s == s else np.nan))
            print(f"[{j}] {w} {run}: {s:.4f}" if s == s else f"[{j}] {w} {run}: nan", flush=True)
        old = old[old.judge != j]
        old = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
        old.to_csv(out_csv, index=False)
        del scorer; gc.collect(); torch.cuda.empty_cache()

    print("\n=== VALIDATION (higher = mangled) ===")
    for j in sorted(old.judge.unique()):
        for w in ["r08BL", "r08B", "r01R", "REAL"]:
            s = old[(old.judge == j) & (old.window == w)].sort_values("score", ascending=False)
            line = " ".join(f"{r}:{x}" for r, x in zip(s.run, s.score))
            print(f" [{j} {w}] {line}")


if __name__ == "__main__":
    main()
