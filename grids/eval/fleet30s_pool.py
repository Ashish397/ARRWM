"""Pool the 30 s fleet eval outputs from both machines into one table per horizon and apply the flag rules.
Inputs per horizon H (all bit-exact decode): local out30s_bx/h<H> (six local fleets), out30s_bx/minwm_ode_h<H>,
u6qf out30s_u6qf_ours/h<H> or h<H>_a + h<H>_b (seven DMD arms).
Recomputed over the POOLED set of models at each scene:
  B = -(d_blur - median d_blur of all models at the scene)                     (fleet_hf.py's sibling-relative haze)
  consensus_all = max ORB inliers (fleet30s_orb.py matcher) between this model's horizon frame and every other
                  model's horizon frame at the scene, plus the real reference frame
  consensus_xfam = same, but peers from the same family are excluded (ours_* incl. ODE arms = one family,
                  minwm + minwm_ode = one family), so our seven sibling arms cannot vouch for each other
Flags: ctrl_fail = cos < 0.5 or static_inl > 500 (non-no-op commands); relocated = reloc_inl < scene_reloc_threshold;
       no_consensus_all / no_consensus_xfam = consensus < 50; haze = B > 150.
Writes out30s_final/h<H>/pooled.csv and prints per-model rates. Usage: fleet30s_pool.py H [H ...]
"""
import sys, os, glob, numpy as np, pandas as pd, cv2
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); THR = int(open(f"{HERE}/scene_reloc_threshold.txt").read())
RENAME = {"ours_base_v2": "ours_base_v2_BROKEN"}   # base_v2 marked BROKEN on 2026-09-17 (flagship is recovery_base)
def fam(m): return "ours" if m.startswith("ours_") else ("minwm" if m.startswith("minwm") else m)
def dirs(H):
    d = [f"{HERE}/out30s_bx/h{H}", f"{HERE}/out30s_bx/minwm_ode_h{H}"]
    u = f"{HERE}/out30s_u6qf_ours"; d += [f"{u}/h{H}"] if os.path.isdir(f"{u}/h{H}") else [f"{u}/h{H}_a", f"{u}/h{H}_b"]
    d += [f"{u}/flagship/h{H}"]   # recovery_base flagship (2026-09-17)
    return [x for x in d if os.path.isdir(x)]
def cat(H, name):
    fs = [f"{d}/{name}" for d in dirs(H) if os.path.exists(f"{d}/{name}")]
    if not fs: return None
    t = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True); t["model"] = t.model.replace(RENAME)
    return t.drop_duplicates(["scene", "model"])
bf = None
def load(p): z = np.load(p); return z["pts"], z["desc"]
def inl(a, b):
    global bf
    if bf is None: bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    (p0, d0), (p1, d1) = a, b
    if len(d0) == 0 or len(d1) == 0 or len(p0) < 8 or len(p1) < 8: return 0
    ms = bf.match(d0, d1)
    if len(ms) < 8: return 0
    src = np.float32([p0[m.queryIdx] for m in ms]); dst = np.float32([p1[m.trainIdx] for m in ms])
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0); return int(mask.sum()) if mask is not None else 0
def scene_cons(job):
    sc, clips, ref = job; kd = {m: load(p) for m, p in clips.items()}; kr = load(ref); out = []
    for m in kd:
        pr = {o: inl(kd[m], kd[o]) for o in kd if o != m}; r = inl(kd[m], kr)
        out.append(dict(scene=sc, model=m, consensus_all=max(list(pr.values()) + [r]),
                        consensus_xfam=max([v for o, v in pr.items() if fam(o) != fam(m)] + [r]), n_models=len(kd)))
    return out
COT_COLS = ["valid_frac", "survival_mean", "survival_min", "survival_last", "pan_dx", "pan_dy", "forward", "roll", "path_px",
            "motion_per_s", "motion_early", "motion_late", "late_early_ratio", "status"]
def cot30w(H):
    """Windowed CoTracker (fleet30s_cotracker.py): local out30s_bx/cot30w + u6qf out30s_u6qf_ours/cot30w/g*."""
    fs = glob.glob(f"{HERE}/out30s_bx/cot30w/fleet30s_cotracker_h{H}.csv") + glob.glob(f"{HERE}/out30s_u6qf_ours/cot30w/g*/fleet30s_cotracker_h{H}.csv") + glob.glob(f"{HERE}/out30s_u6qf_ours/flagship/cot30w/fleet30s_cotracker_h{H}.csv")
    t = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True); t["model"] = t.model.replace(RENAME)
    t = t.drop_duplicates(["scene", "model"])
    return t.rename(columns={"forward": "cot_forward", "roll": "cot_roll", "status": "cot_status"})
def main(H):
    clips, refs = {}, {}
    for d in dirs(H):
        for p in glob.glob(f"{d}/orb_desc_h{H}/*.npz"):
            b = os.path.basename(p)[:-4]
            if b.startswith("ref__"): refs.setdefault(b[5:], p)
            else: sc, m = b.split("__", 1); clips.setdefault(sc, {}).setdefault(RENAME.get(m, m), p)
    jobs = [(sc, c, refs[sc.rsplit("_", 1)[0]]) for sc, c in sorted(clips.items())]
    with Pool(int(os.environ.get("POOL_PROCS", "12"))) as pool: cons = pd.DataFrame([r for rs in pool.map(scene_cons, jobs, chunksize=4) for r in rs])
    df = cat(H, f"fleet30s_orb_h{H}.csv")[["scene", "model", "static_inl", "reloc_inl"]].merge(cons, on=["scene", "model"], how="outer")
    hf = cat(H, "fleet_hf.csv")
    if hf is not None:
        hf["B"] = -(hf.d_blur - hf.groupby("scene").d_blur.transform("median")); df = df.merge(hf[["scene", "model", "d_blur", "B"]], on=["scene", "model"], how="left")
    for name, cols in [("fleet_dino.csv", ["dino_drift"]), (f"fleet30s_pca_h{H}.csv", ["z0", "z1", "mag", "cos"]), (f"fleet30s_cotracker_h{H}.csv", COT_COLS)]:
        t = cat(H, name) if not name.startswith("fleet30s_cotracker") else cot30w(H)
        if t is not None:
            cols = [{"forward": "cot_forward", "roll": "cot_roll", "status": "cot_status"}.get(c, c) for c in cols]
            df = df.merge(t[["scene", "model"] + cols], on=["scene", "model"], how="left")
    df["dir"] = df.scene.str.rsplit("_", n=1).str[1]; df["horizon_s"] = H
    nn = df.dir != "N"
    df["ctrl_fail"] = np.where(nn, ((df.get("cos", np.nan) < 0.5) | (df.static_inl > 500)).astype(float), np.nan)
    df.loc[nn & df.get("cos", pd.Series(np.nan, index=df.index)).isna(), "ctrl_fail"] = np.nan
    df["relocated"] = (df.reloc_inl < THR).astype(int); df["no_consensus_all"] = (df.consensus_all < 50).astype(int)
    df["no_consensus_xfam"] = (df.consensus_xfam < 50).astype(int)
    if "B" in df: df["haze"] = (df.B > 150).astype(int)
    os.makedirs(f"{HERE}/out30s_final/h{H}", exist_ok=True); df.to_csv(f"{HERE}/out30s_final/h{H}/pooled.csv", index=False)
    agg = {"n": ("scene", "size"), "ctrl_fail": ("ctrl_fail", "mean"), "cos_med": ("cos", "median"), "relocated": ("relocated", "mean"),
           "no_cons_all": ("no_consensus_all", "mean"), "no_cons_xfam": ("no_consensus_xfam", "mean"), "haze": ("haze", "mean"), "dino_med": ("dino_drift", "median"), "cot_survival": ("survival_mean", "median"), "cot_late_early": ("late_early_ratio", "median")}
    agg = {k: v for k, v in agg.items() if v[0] in df}
    s = df.groupby("model").agg(**agg).round(3); s.to_csv(f"{HERE}/out30s_final/h{H}/summary.csv")
    pd.set_option("display.width", 220); print(f"== horizon {H}s: {len(df)} rows, {df.scene.nunique()} scenes\n{s}", flush=True)
if __name__ == "__main__":
    for H in sys.argv[1:]: main(int(H))
