"""Stall-proof HF weight fetcher: lists repo files via API, downloads each with
curl (-C - resume, --speed-time stall abort), prints a progress line per file.
Connections that stall >20s at <500KB/s are killed and resumed within seconds.

Plan format: (repo, include_prefixes_or_names, local_root, strip_prefix)
"""
import os, subprocess, sys, fnmatch, time
from huggingface_hub import HfApi

T = "/scratch/u6ex/as1748.u6ex/ARRWM/third_party"
HF = "/scratch/u6ex/as1748.u6ex/frodobots/hf_cache"

PLAN = [
    ("Skywork/Matrix-Game-2.0",
     ["base_distilled_model/*", "base_model/*", "Wan2.1_VAE.pth", "models_clip*", "xlm-roberta-large/*", "*.json"],
     f"{T}/Matrix-Game/Matrix-Game-2/Matrix-Game-2.0", ""),
    ("OpenDriveLab/Vista", ["vista.safetensors"], f"{T}/Vista/ckpts", ""),
    ("tencent/HunyuanVideo-1.5", ["vae/*", "scheduler/*", "transformer/480p_i2v/*"],
     f"{T}/HY-WorldPlay/weights/HunyuanVideo-1.5", ""),
    ("google/byt5-small", ["*"], f"{T}/HY-WorldPlay/weights/HunyuanVideo-1.5/text_encoder/byt5-small", ""),
    ("tencent/HY-WorldPlay", ["ar_rl_model/*"], f"{T}/HY-WorldPlay/weights", ""),
    ("google/siglip-so400m-patch14-384", ["*"], f"{T}/HY-WorldPlay/weights/_siglip_raw", ""),
    ("stdstu123/Yume-5B-720P", ["*"], f"{T}/YUME/Yume-5B-720P", ""),
    ("stdstu123/Yume-I2V-540P", ["*"], f"{T}/YUME/Yume-I2V-540P", ""),
    ("OpenGVLab/InternVL3-2B-Instruct", ["*"], f"{T}/YUME/InternVL3-2B-Instruct", ""),
    ("OpenGVLab/InternVL3-8B", ["*"], f"{T}/_vlm/InternVL3-8B", ""),
    ("ByteDance/Sa2VA-4B", ["*"], f"{T}/_vlm/Sa2VA-4B", ""),
]
EXCLUDE = ["demo.mp4", "*.git*", "*.md"]


def fetch(url, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for attempt in range(120):
        r = subprocess.run(
            ["curl", "-fsSL", "-C", "-", "--speed-limit", "500000", "--speed-time", "20",
             "--connect-timeout", "15", "-o", out, url],
            capture_output=True, text=True)
        if r.returncode == 0:
            return True
        # 416 = already complete
        if "416" in (r.stderr or ""):
            return True
        time.sleep(2)
    return False


def main():
    api = HfApi()
    for repo, pats, root, strip in PLAN:
        try:
            files = [s.rfilename for s in api.model_info(repo, files_metadata=True).siblings]
        except Exception as e:
            print(f"[fetch] LIST FAIL {repo}: {str(e)[:120]}", flush=True); continue
        todo = [f for f in files
                if any(fnmatch.fnmatch(f, p) or f.startswith(p.rstrip("*")) for p in pats)
                and not any(fnmatch.fnmatch(os.path.basename(f), e) for e in EXCLUDE)]
        sizes = {s.rfilename: (s.size or 0) for s in api.model_info(repo, files_metadata=True).siblings}
        for f in todo:
            out = os.path.join(root, f[len(strip):] if strip and f.startswith(strip) else f)
            want = sizes.get(f, 0)
            if os.path.exists(out) and want and os.path.getsize(out) == want:
                continue
            t0 = time.time()
            url = f"https://huggingface.co/{repo}/resolve/main/{f}"
            ok = fetch(url, out)
            if ok and want and os.path.exists(out) and os.path.getsize(out) != want:
                os.remove(out)                      # corrupt/mismatched: refetch fresh
                ok = fetch(url, out)
                ok = ok and os.path.getsize(out) == want
            mb = os.path.getsize(out) / 1e6 if os.path.exists(out) else 0
            print(f"[fetch] {'OK' if ok else 'FAIL'} {repo}/{f} {mb:.0f}MB {time.time()-t0:.0f}s", flush=True)
        print(f"[fetch] REPO DONE {repo}", flush=True)
    print("[fetch] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
