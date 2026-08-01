"""VLM discriminator for external models vs our good variants (pca8/16node).

Per video: REFERENCE = first real second (6 labeled frames), GENERATED = 16 frames
import os
from the remainder (sampled by wall-clock time, native fps). Three yes/no logit
probes targeting the failure modes seen in the external fleet:
  p_style   style departs from reference (game-like/painted/cartoon counts; lighting doesn't)
  p_scene   scene changes to a different place / layout transforms
  p_uncanny reality-breaking content (impossible geometry, abstract surfaces)
Writes results_external_vlm.csv.
"""
import json, os, sys
import cv2
import numpy as np
import torch
import pandas as pd
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
# Stationary mode: score the no-op set, a flat directory of <model>_<scene>.mp4.
STATIONARY_DIR = os.environ.get("AF_STATIONARY_DIR", "")
STATIONARY = bool(STATIONARY_DIR) and os.path.isdir(STATIONARY_DIR)
BASE_DIR = os.path.join(os.environ.get("AF_FLEET_DIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "grids")), "baselines")
MODELS = ["astra", "matrixgame", "minwm", "worldcam", "worldplay", "yume"]
# AF_VLM_MODELS restricts the run, so a new ablation can be scored without
# putting the whole fleet through the VLM again.
if os.environ.get("AF_VLM_MODELS"):
    _want = {m.strip() for m in os.environ["AF_VLM_MODELS"].split(",") if m.strip()}
    MODELS = [m for m in MODELS if m in _want]
OURS = ["pca8", "pca4", "pca2", "16node", "4node", "noatok", "noadaln"]
# Ablations rendered after the grid was built, supplied as standalone tiles.
OURS += [v.strip() for v in os.environ.get("AF_EXTRA_VARIANTS", "").split(",") if v.strip()]
if os.environ.get("AF_VLM_MODELS"):
    _w = {m.strip() for m in os.environ["AF_VLM_MODELS"].split(",") if m.strip()}
    OURS = [v for v in OURS if v in _w or f"ours_{v}" in _w]
# AF_VLM_OUT keeps a new ablation out of the shipped reference artefact, which
# the paper's geometry column is computed from.
OUT = os.environ.get("AF_VLM_OUT", os.path.join(HERE, "results_external_vlm.csv"))

INTRO = """The first four images labelled REFERENCE are real frames of a driving video, in temporal order - the true scene and style. The remaining images labelled GENERATED are an AI world model's continuation of that exact scene, in temporal order. The model was supposed to continue the SAME scene in the SAME visual style with plausible content."""

# real-context frame counts in each model's saved video; generation starts after these
CTX_FRAMES = {"astra": 4, "matrixgame": 1, "minwm": 13, "worldcam": 65,
              "worldplay": 1, "yume": 1}
OURS_CTX = 12

# Directories holding <scene>__<variant>.mp4; AF_TILES_DIR may add more.
TILE_DIRS = [os.path.join(HERE, "tiles"), os.path.join(HERE, "tiles_new")]
TILE_DIRS += [d for d in os.environ.get("AF_TILES_DIR", "").split(os.pathsep) if d]

PROBES = [
    ("style", """Does the visual STYLE of the generated frames depart from the reference - e.g. becoming painted, game-like, cartoonish, watercolour, oversaturated, or otherwise a different rendering style? Ordinary lighting or exposure changes do NOT count. Answer Yes or No."""),
    ("novel", """Does any NEW OBJECT appear in the generated frames that was not visible in the reference - a vehicle, person, structure or large item that the model introduced? Newly revealed parts of the existing scene (from driving) do NOT count; only genuinely new objects. Answer Yes or No."""),
    ("uncanny", """Is there a SIGNIFICANT uncanny or reality-breaking failure in the generated frames - impossible geometry, surfaces dissolving into abstract patterns, large corrupted regions - that a casual viewer would notice within one second? Ignore small local artifacts and minor blur. Answer Yes or No."""),
]


def read_video(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 16
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (640, 352)))
    cap.release()
    return np.stack(frames), fps


def label_img(fr, label):
    hdr = np.full((26, fr.shape[1], 3), 32, np.uint8)
    cv2.putText(hdr, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return Image.fromarray(np.concatenate([hdr, fr], axis=0))


def main():
    scenes = sys.argv[1:] or list(json.load(open(os.path.join(HERE, "gt.json")))["grids"].keys())
    from transformers import AutoProcessor
    from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
    proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    model = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", dtype=torch.bfloat16, device_map="cuda").eval()
    tok = proc.tokenizer
    yes_id = tok.encode("Yes", add_special_tokens=False)[0]
    no_id = tok.encode("No", add_special_tokens=False)[0]

    @torch.no_grad()
    def p_yes(ims, question, nref):
        content = [{"type": "image", "image": im} for im in ims] + [
            {"type": "text", "text": INTRO.format(nref=nref) + "\n\n" + question + '\n\nAnswer with ONLY {"answer": "Yes"} or {"answer": "No"}.'}]
        text = proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False)
        text += '{"answer": "'
        inputs = proc(text=[text], images=ims, return_tensors="pt").to("cuda")
        logits = model(**inputs).logits[0, -1]
        p = torch.softmax(logits[torch.tensor([yes_id, no_id], device=logits.device)], 0).float().cpu().numpy()
        return float(p[0])

    done = set()
    if os.path.exists(OUT) and os.path.getsize(OUT) > 0:
        prev = pd.read_csv(OUT)
        done = set(zip(prev.scene, prev.model))
    write_header = not os.path.exists(OUT) or os.path.getsize(OUT) == 0
    fout = open(OUT, "a")
    rows = []
    for scene in scenes:
        vids = {}
        for m in MODELS:
            fp = (os.path.join(STATIONARY_DIR, f"{m}_{scene}.mp4") if STATIONARY
                  else os.path.join(BASE_DIR, f"A_{m}", f"{m}_{scene}.mp4"))
            if os.path.exists(fp):
                vids[m] = fp
        for v in OURS:
            if STATIONARY:
                fp = os.path.join(STATIONARY_DIR, f"{v}_{scene}.mp4")
                if os.path.exists(fp):
                    vids[f"ours_{v}"] = fp
                continue
            for tdir in TILE_DIRS:
                fp = os.path.join(HERE, tdir, f"{scene}__{v}.mp4")
                if os.path.exists(fp):
                    vids[f"ours_{v}"] = fp
                    break
        # shared REFERENCE: 4 real frames spanning the 0.75s context (frames 2,5,8,11 of our tile)
        ref_imgs = None
        # The stationary set ships the real rollout itself, so the reference
        # frames come from that rather than from our tile's context span.
        ref_candidates = ([os.path.join(STATIONARY_DIR, f"real_{scene}.mp4"),
                           os.path.join(STATIONARY_DIR, f"pca8_{scene}.mp4")] if STATIONARY
                          else [os.path.join(HERE, td, f"{scene}__pca8.mp4") for td in TILE_DIRS])
        for fp0 in ref_candidates:
            if os.path.exists(fp0):
                fr0, _ = read_video(fp0)
                ref_imgs = [label_img(fr0[i], "REFERENCE") for i in (2, 5, 8, 11)]
                break
        if ref_imgs is None:
            print(f"SKIP {scene}: no real reference tile", flush=True)
            continue
        for name, fp in vids.items():
            if (scene, name) in done:
                continue
            frames, fps = read_video(fp)
            n1 = OURS_CTX if name.startswith("ours_") else CTX_FRAMES[name]
            if len(frames) <= n1 + int(fps):
                continue
            # matched horizon: generation start + 6s of generated content, rest discarded
            n_end = min(len(frames) - 1, n1 + int(round(6.0 * fps)))
            gen_idx = np.linspace(n1, n_end, 16).round().astype(int)
            ims = ref_imgs + [label_img(frames[i], "GENERATED") for i in gen_idx]
            r = {"scene": scene, "model": name}
            for key, q in PROBES:
                r[f"p_{key}"] = round(p_yes(ims, q, 1), 4)
            rows.append(r)
            if write_header:
                fout.write("scene,model,p_style,p_novel,p_uncanny\n"); write_header = False
            fout.write(f"{r['scene']},{r['model']},{r['p_style']},{r['p_novel']},{r['p_uncanny']}\n"); fout.flush()
            print(f"{scene} {name}: style={r['p_style']:.2f} novel={r['p_novel']:.2f} uncanny={r['p_uncanny']:.2f}", flush=True)

    fout.close()
    print(f"appended {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
