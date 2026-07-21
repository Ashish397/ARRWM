"""VLM-as-judge mangle detectors (tier 1 of the mangle hunt).

A) qalign (pyiqa): mPLUG-Owl2-7B trained quality scorer, per-frame, video score =
   min over sampled gen frames (worst moment; lower qalign = worse quality).
B) qwen_melt: Qwen2.5-VL-7B-Instruct asked directly, per gen frame:
   'severity 0-10 of melted/warped/impossible structures'; video score = mean of
   top-3 frame severities (melt is judged at its worst moments).

Videos: r08 B + BL (ground-truth anchors), r01_R (blind), 8 real refs (calibration).
PASS = BL: clean {pca8_8node,4node} lowest melt; B: 16node worst + pca8 mid/low;
refs ~0. Saves analysis/eval_final/vlm_judge_scores.csv
"""
import os, re, glob
import numpy as np, pandas as pd, imageio, torch

DEV = "cuda"
RUNS = os.environ.get("VJ_RUNS", "pca8_8node,pca4,pca2,16node,4node,noatok").replace(":", ",").split(",")
TARGETS = [(8, "B"), (8, "BL"), (1, "R")]
GEN_IDX = [24, 40, 56, 72, 88, 104]          # generated portion (skip seed)
CLEAN_BL = {"pca8_8node", "4node"}



_MWSW = {"L": "R", "R": "L", "FL": "FR", "FR": "FL", "BL": "BR", "BR": "BL"}
def _vid_path(run, rank, br, arr="."):
    if run == "minwm":   # disk labels are yaw-sign-flipped; swap to reach TRUE direction
        return f"{arr}/logs/eval_final/A_minwm/minwm_r{rank:02d}_{_MWSW.get(br, br)}.mp4"
    return f"{arr}/logs/eval_final/A/{run}/control_test/step05000_r{rank:02d}_{br}_raw.mp4"

def frames_of(path, idxs):
    r = imageio.get_reader(path)
    out = []
    for i in idxs:
        try:
            out.append(np.asarray(r.get_data(i)))
        except Exception:
            break
    r.close(); return out


def main():
    vids = []
    for rank, br in TARGETS:
        for run in RUNS:
            vids.append((f"r{rank:02d}{br}", run, _vid_path(run, rank, br)))
    for p in sorted(glob.glob("analysis/eval_final/real_refs/*.mp4"))[:8]:
        vids.append(("REAL", os.path.basename(p)[:18], p))

    rows = {(w, r): {} for w, r, _ in vids}

    # ---------- A) qalign ----------
    try:
        import pyiqa
        qa = pyiqa.create_metric("qalign", device=DEV)
        for w, run, p in vids:
            fr = frames_of(p, GEN_IDX)
            vals = []
            for f in fr:
                t = torch.tensor(f).permute(2, 0, 1).unsqueeze(0).float().to(DEV) / 255.
                vals.append(float(qa(t).item()))
            rows[(w, run)]["qalign_min"] = round(min(vals), 3)
            rows[(w, run)]["qalign_mean"] = round(float(np.mean(vals)), 3)
            print(f"[qalign] {w} {run}: min={min(vals):.2f} mean={np.mean(vals):.2f}", flush=True)
        del qa; torch.cuda.empty_cache()
    except Exception as e:
        print(f"[qalign] FAILED: {str(e)[:200]}")

    # ---------- B) Qwen2.5-VL melt judge ----------
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from PIL import Image
        mid = "Qwen/Qwen2.5-VL-7B-Instruct"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            mid, dtype=torch.bfloat16, device_map=DEV)
        proc = AutoProcessor.from_pretrained(mid)
        PROMPT = ("This is a crop from a street-scene photo. Are any buildings, walls or "
                  "structures in it melted, warped, or geometrically impossible, like a "
                  "corrupted AI-generated image? Answer only Yes or No.")
        yes_ids = [proc.tokenizer.encode(t, add_special_tokens=False)[0] for t in ("Yes", " Yes", "yes")]
        no_ids = [proc.tokenizer.encode(t, add_special_tokens=False)[0] for t in ("No", " No", "no")]

        def p_yes(img):
            im = Image.fromarray(img)
            msgs = [{"role": "user", "content": [{"type": "image", "image": im},
                                                 {"type": "text", "text": PROMPT}]}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = proc(text=[text], images=[im], return_tensors="pt").to(DEV)
            with torch.no_grad():
                logits = model(**inputs).logits[0, -1]
            py = torch.logsumexp(logits[yes_ids], 0)
            pn = torch.logsumexp(logits[no_ids], 0)
            return float(torch.sigmoid(py - pn).item())

        def crops(img):
            H, W = img.shape[:2]
            h = int(H * 0.6); w2 = W // 2
            return [img[:h, :w2], img[:h, w2:]]   # upper-region crops only (road excluded)

        for w, run, p in vids:
            fr = frames_of(p, GEN_IDX)
            vals = []
            for f in fr:
                for c in crops(f):
                    vals.append(p_yes(c))
            top4 = sorted(vals)[-4:] if vals else [np.nan]
            rows[(w, run)]["qwen_pyes_top4"] = round(float(np.mean(top4)), 4)
            rows[(w, run)]["qwen_pyes_mean"] = round(float(np.mean(vals)), 4) if vals else np.nan
            print(f"[qwen] {w} {run}: top4={np.mean(top4):.3f} mean={np.mean(vals):.3f}", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[qwen] FAILED: {str(e)[:200]}")

    df = pd.DataFrame([dict(window=w, run=r, **v) for (w, r), v in rows.items()])
    df.to_csv("analysis/eval_final/vlm_judge_scores.csv", index=False)

    print("\n=== VALIDATION ===")
    for col, lower_is_mangled in [("qalign_min", True), ("qalign_mean", True),
                                  ("qwen_pyes_top4", False), ("qwen_pyes_mean", False)]:
        if col not in df.columns:
            continue
        print(f" [{col}]")
        for w in ["r08BL", "r08B", "r01R", "REAL"]:
            s = df[df.window == w].sort_values(col, ascending=lower_is_mangled)
            order = list(s.run)   # first = most mangled per metric
            line = " ".join(f"{r}:{x}" for r, x in zip(s.run, s[col]))
            verdict = ""
            if w == "r08BL" and len(order) == 6:
                verdict = " -> PASS" if set(order[-2:]) == CLEAN_BL else " -> fail"
            if w == "r08B" and len(order) == 6:
                verdict = " -> PASS" if (order[0] == "16node" and order.index("pca8_8node") >= 2) else " -> fail"
            print(f"   {w}: {line}{verdict}")


if __name__ == "__main__":
    main()
