"""Match real reference clips to scene ranks by their context frames.

Every rollout's first ~1s is the REAL context of its scene, and the real
refs are the real continuations of those same contexts — so the real ref
whose early frames match a scene's rollout context IS that scene's sibling.
Match on downsampled grayscale frames at a few context timestamps, solve
the assignment with the Hungarian algorithm, and sanity-check the margin.

Writes out/real_scene_map.json (basename -> scene int).
"""
import json
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fleet

MATCH_TS = (0.30, 0.60, 0.90)      # context timestamps to compare
OFFSETS = np.arange(0.0, 5.0, 0.1)  # real clips may be trimmed differently
THUMB = (64, 36)
# context comes from rollouts of these models (any clean-context model works)
CONTEXT_MODELS = ("pca8", "matrixgame")
CONTEXT_DIR = "F"


def thumbs(path, ts, max_seconds):
    frames, times, _ = fleet.load_video(path, max_seconds=max_seconds)
    out = []
    for t in ts:
        g = cv2.cvtColor(fleet.frame_at(frames, times, t), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, THUMB, interpolation=cv2.INTER_AREA).astype(np.float32)
        out.append((g - g.mean()) / (g.std() + 1e-6))
    return np.stack(out)


def main():
    refs = fleet.discover_fleet(real_refs=False)
    scene_ctx = {}
    for m in CONTEXT_MODELS:
        for r in refs:
            if r.model == m and r.direction == CONTEXT_DIR and r.scene not in scene_ctx:
                scene_ctx[r.scene] = thumbs(r.path, MATCH_TS, max_seconds=1.2)
    scenes = sorted(scene_ctx)
    print(f"context thumbs for {len(scenes)} scenes")

    real = [r for r in fleet.discover_fleet(models=["real"], ablation=False)]
    # dense real-side thumbs so a trim offset can be searched
    dense_ts = np.arange(0.1, 6.2, 0.1)
    real_th = {os.path.basename(r.path): thumbs(r.path, dense_ts, max_seconds=6.5)
               for r in real}
    names = sorted(real_th)

    def offset_cost(dense, ctx):
        best = np.inf
        for d in OFFSETS:
            ii = [int(round((t + d - 0.1) / 0.1)) for t in MATCH_TS]
            if ii[-1] >= len(dense):
                break
            c = float(((dense[ii] - ctx) ** 2).mean())
            best = min(best, c)
        return best

    cost = np.zeros((len(names), len(scenes)))
    for i, nm in enumerate(names):
        for j, s in enumerate(scenes):
            cost[i, j] = offset_cost(real_th[nm], scene_ctx[s])

    from scipy.optimize import linear_sum_assignment
    ri, ci = linear_sum_assignment(cost)
    mapping, weak = {}, []
    for i, j in zip(ri, ci):
        assigned = cost[i, j]
        runner_up = np.partition(cost[i], 1)[1]
        margin = runner_up / (assigned + 1e-9)
        mapping[names[i]] = int(scenes[j])
        flag = "" if margin > 1.5 else "  <-- WEAK MATCH"
        if margin <= 1.5:
            weak.append(names[i])
        print(f"{names[i]} -> r{scenes[j]:02d}  cost={assigned:.3f} "
              f"margin={margin:.2f}{flag}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "out", "real_scene_map.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(mapping, open(out, "w"), indent=1)
    print(f"\nwrote {out}; {len(weak)} weak matches" +
          (f" ({', '.join(weak)}) — verify manually" if weak else ""))


if __name__ == "__main__":
    main()
