"""Melt VLM component on blind100 (Qwen2.5-VL-7B + Cosmos-Reason1-7B), rerun to
validate against the reference CSV's qwen_melt_pyes and to add the Cosmos vote.

Same protocol as melt_vlm_bench (top-60% L/R crops, top-4 mean logit P(Yes)); generated
frames sampled per-video: linspace(ctx, min(n-1, ctx+6s), 6) so mixed-fps/context
externals are handled. Env MELT_MODEL (qwen25vl7b|cosmos_reason1_7b).
Writes out/blind_melt_<model>.csv.
"""
import os
import numpy as np, imageio, torch, pandas as pd
import blind100_common as bc
from melt_vlm_bench import load_judge, MELT_MODEL

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", f"blind_melt_{MELT_MODEL}.csv")


def gen_frames(path, ctx):
    r = imageio.get_reader(path)
    n = r.count_frames()
    fps = r.get_meta_data().get("fps", 16) or 16
    idx = np.linspace(ctx, min(n - 1, ctx + int(round(6.0 * fps))), 6).round().astype(int)
    out = [np.asarray(r.get_data(int(i))) for i in idx]
    r.close()
    return out


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    print(f"[melt] blind100 model={MELT_MODEL}", flush=True)
    model, scorer = load_judge(MELT_MODEL)
    rows = []
    for r in bc.refs():
        try:
            p = scorer(gen_frames(r["path"], r["ctx"]))
        except Exception as e:
            print(f"[melt] {r['blind_id']} failed: {str(e)[:80]}", flush=True); p = np.nan
        rows.append(dict(blind_id=r["blind_id"], vid=r["vid"], model=r["model"],
                         scene=r["scene"], melt_pyes=p))
        print(f"[melt] {r['blind_id']} {r['vid']:20s} melt={p:.3f}", flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"[melt] wrote {OUT}")


if __name__ == "__main__":
    main()
