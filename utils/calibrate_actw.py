"""Calibrate the action-error weight against the TEACHER's own command-following.

The weight w = 1 + alpha*min(err/err_ref, max) is only meaningful if err_ref is
the scale of a GOOD chain's error. The teacher is the definition of good here,
so measure ||z_realised - z_commanded|| on the teacher's OWN committed chains
(the exact data the student is trained to reproduce) and report the
distribution. err_ref should be the teacher's median: a student that follows
commands as well as the teacher then sits at w = 1 + alpha, and only worse-than-
teacher chains are pushed above that.

Env: CA_N contexts (def 40), CA_OUT csv.
"""
import os, glob, re, json
import numpy as np
import torch

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
D = "/projects/u6ex/fbots/frodobots_lmdb/v14e_pilot_dir8n"
N = int(os.environ.get("CA_N", "40"))
OUT = os.environ.get("CA_OUT", f"{ARR}/analysis/eval_final/flow_viz/actw_calibration.json")
DIRS = ["cF", "cFR", "cR", "cBR", "cB", "cBL", "cL", "cFL", "cN"]


def main():
    import sys; sys.path.insert(0, ARR); sys.path.insert(0, f"{ARR}/action-forcing")
    from utils.wan_wrapper import WanVAEWrapper
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "aw", f"{ARR}/action-forcing/af_model/action_weight.py")
    aw = importlib.util.module_from_spec(spec); spec.loader.exec_module(aw)

    dev = "cuda"
    vae = WanVAEWrapper().to(dev).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    ct = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(dev).eval()
    ck = torch.load(f"{ARR}/action_query/checkpoints/ss_vae_8free.pt",
                    map_location="cpu", weights_only=False)
    pm = torch.tensor(np.asarray(ck["pca_mean"]), dtype=torch.float32, device=dev)
    pc = torch.tensor(np.asarray(ck["pca_comp"]).T, dtype=torch.float32, device=dev)
    # NOT ck["pca_scales"] -- that key does not exist, so the old fallback
    # silently used 1.0 and saturated every tanh. These are the teacher's
    # pca_raw_scales (2.5*std per component).
    ps = torch.tensor([93.7, 57.7, 22.5, 21.2, 18.1, 14.5, 12.6, 10.8], dtype=torch.float32, device=dev)

    ctxs = sorted({re.sub(r"_c[A-Z]+\.pt$", "", os.path.basename(f))
                   for f in glob.glob(f"{D}/*_cF.pt")})[:N]
    rows = []
    for i, ctx in enumerate(ctxs):
        for d in DIRS:
            f = f"{D}/{ctx}_{d}.pt"
            if not os.path.exists(f):
                continue
            pt = torch.load(f, map_location="cpu", weights_only=False)
            chain = pt["trajectory"][:, -1].reshape(1, -1, 16, 60, 104).to(dev).float()
            z = pt["z"].float()
            rz = aw.realised_action_z(chain_latents=chain, frozen_vae=vae, cotracker=ct,
                                      pca_mean=pm, pca_comp_T=pc, pca_scales=ps)
            if rz is None:
                continue
            got = rz[:, :2].mean(0).cpu()
            want = z[-chain.shape[1]:, :2].mean(0)
            dx = float(abs(got[0] - want[0])); dy = float(abs(got[1] - want[1]))
            # user formula: each dim / 4 (actions span [-1,1] so max err 2 per
            # dim => the sum spans [0,1] and the weight spans [1,0]).
            w_valid = max(0.0, min(1.0, 1.0 - (dx / 4.0 + dy / 4.0)))
            rows.append({"ctx": ctx, "dir": d,
                         "err": float(torch.linalg.vector_norm(got - want)),
                         "dx": dx, "dy": dy, "w_valid": w_valid,
                         "got": got.tolist(), "want": want.tolist()})
        if i % 10 == 0:
            print(f"[ca] {i+1}/{len(ctxs)} contexts, {len(rows)} chains", flush=True)

    # PER-DIRECTION residual: which commanded directions does the TEACHER
    # itself follow well, and which does it botch? Those are the ones whose
    # targets should be down-weighted or zeroed.
    per_dir = {}
    for d in DIRS:
        sub = [r for r in rows if r["dir"] == d]
        if sub:
            per_dir[d] = {
                "n": len(sub),
                "mean_err": float(np.mean([r["err"] for r in sub])),
                "mean_w_valid": float(np.mean([r["w_valid"] for r in sub])),
                "frac_w_below_0.5": float(np.mean([r["w_valid"] < 0.5 for r in sub])),
            }
    print("\nPER-DIRECTION teacher command-following:")
    print(f"  {'dir':>4} {'n':>4} {'mean_err':>9} {'mean_w':>7} {'frac w<0.5':>11}")
    for d, v in per_dir.items():
        print(f"  {d:>4} {v['n']:4d} {v['mean_err']:9.4f} "
              f"{v['mean_w_valid']:7.3f} {v['frac_w_below_0.5']:11.2f}")
    wv = np.array([r["w_valid"] for r in rows])
    print(f"\nTEACHER-VALIDITY WEIGHT over {len(wv)} chains: "
          f"mean {wv.mean():.3f}  median {np.median(wv):.3f}  "
          f"frac<0.5 {float((wv < 0.5).mean()):.3f}  frac==0 {float((wv <= 0).mean()):.3f}")
    e = np.array([r["err"] for r in rows])
    q = {f"p{p}": float(np.percentile(e, p)) for p in (10, 25, 50, 75, 90)}
    summary = {"n_chains": len(rows), "mean": float(e.mean()), **q,
               "suggested_err_ref_median": q["p50"]}
    print("\nTEACHER's OWN realised-vs-commanded action error:")
    for k, v in summary.items():
        print(f"  {k:26} {v}")
    print(f"\n=> err_ref was GUESSED at 0.25; the teacher's median is {q['p50']:.4f}")
    summary["per_direction"] = per_dir
    summary["w_valid_mean"] = float(wv.mean())
    summary["w_valid_median"] = float(np.median(wv))
    json.dump({"summary": summary, "rows": rows}, open(OUT, "w"), indent=1)
    print(f"[ca] wrote {OUT}")


if __name__ == "__main__":
    main()
