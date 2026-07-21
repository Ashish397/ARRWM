"""Find an IMAGE-quality metric that detects generative 'mangling' of newly-generated
content, measured PER MODEL as the degradation from its own clean first frame (the
real decoded seed) to its generated last frame. A good mangle metric shows a large
quality drop for 16node (mangles) and ~0 for noatok/pca2 (stay clean) on r08 B.

For each model: q(first) vs q(last); delta = signed degradation. Rank models by delta
per metric; the winner cleanly separates 16node (worst) from the clean models.

Env: MM_RANK (default 8), MM_BRANCH (default B).
"""
import os
import numpy as np, imageio, torch, gc
import pyiqa

RANK = int(os.environ.get("MM_RANK", "8"))
BR = os.environ.get("MM_BRANCH", "B")
ORDER = ["noatok", "pca2", "pca4", "pca8_8node", "4node", "16node"]   # perceived clean -> mangled
FIRST = [0, 3, 6, 9]        # clean real seed frames
LAST = [96, 99, 102, 105]   # generated frames where mangle shows
LOWER_BETTER = {"brisque", "niqe"}
METRICS = ["clipiqa", "clipiqa+", "brisque", "musiq", "niqe", "maniqa", "topiq_nr",
           "dbcnn", "paq2piq", "hyperiqa", "nima", "cnniqa", "tres"]


def load(run):
    r = imageio.get_reader(f"logs/eval_final/A/{run}/control_test/step05000_r{RANK:02d}_{BR}_raw.mp4")
    def t(i):
        f = np.asarray(r.get_data(i))
        return torch.tensor(f).permute(2, 0, 1).unsqueeze(0).float() / 255.
    first = [t(i) for i in FIRST]; last = [t(i) for i in LAST]
    r.close(); return first, last


def main():
    imgs = {run: load(run) for run in ORDER}
    print(f"r{RANK:02d} {BR}: per-model quality degradation first{FIRST} -> last{LAST}\n")
    for name in METRICS:
        try:
            m = pyiqa.create_metric(name, device="cuda" if torch.cuda.is_available() else "cpu")
            lb = name in LOWER_BETTER
            rows = {}
            for run in ORDER:
                first, last = imgs[run]
                q0 = np.mean([float(m(x).item()) for x in first])
                qT = np.mean([float(m(x).item()) for x in last])
                drop = (qT - q0) if lb else (q0 - qT)    # +ve = degraded
                rows[run] = (q0, qT, drop)
            del m; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
            clean = np.mean([rows[r][2] for r in ("noatok", "pca2", "pca4")])
            mang = rows["16node"][2]
            sep = mang - clean                            # how much MORE 16node degrades than clean
            worst = max(ORDER, key=lambda r: rows[r][2])
            flag = "*** FLAGS (16node worst-degraded)" if worst == "16node" and sep > 0 else ("misses" if worst != "16node" else "")
            print(f"{name:9} drop: " + " ".join(f"{r[:6]}:{rows[r][2]:+5.2f}" for r in ORDER)
                  + f"  | 16node-clean sep={sep:+.2f}  worst={worst}  {flag}")
        except Exception as e:
            print(f"{name:9} SKIP ({str(e)[:60]})")


if __name__ == "__main__":
    main()
