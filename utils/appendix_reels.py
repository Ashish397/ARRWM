"""Appendix filmstrip reels: successes, failures, and interesting cases.

Three sets, all from OUR models (no baselines):
  success/      16node + pca8_8node only, stratified over the 8 commands and
                over distinct held-out scenes; strong following, sharpness
                preserved, scene identity retained.
  failure/      any of our ablations; stratified over FAILURE MODE
                (anti-following, haze collapse, scene relocation).
  interesting/  hand-named cases plus mined phenomena (roll coupling,
                collision-like stalls, interior spawns).

Two-stage for speed: stage 1 scores all rollouts from the JSONL teacher reads
and the head-to-head CoTracker-PCA table (no decoding); stage 2 decodes only
the shortlisted candidates.

Env: AR_OUT (default analysis/reels_appendix), AR_N (per-set size, default 20).
"""
import os, json, glob
import numpy as np
import pandas as pd
import av, cv2
from PIL import Image, ImageDraw

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
OUT = os.environ.get("AR_OUT", f"{ARR}/analysis/reels_appendix")
N_SET = int(os.environ.get("AR_N", "20"))
OURS = ["16node", "pca8_8node", "4node", "pca4", "pca2", "noatok", "noadaln"]
BEST = ["16node", "pca8_8node"]
DIRS = ["F", "FR", "R", "BR", "B", "BL", "L", "FL"]
NICE = {"16node": "batch 64", "pca8_8node": "batch 32", "4node": "batch 16",
        "pca4": "PCA4", "pca2": "PCA2", "noatok": "no action tokens",
        "noadaln": "no AdaLN"}
DNAME = {"F": "Forward", "FR": "Forward-Right", "R": "Right", "BR": "Back-Right",
         "B": "Backward", "BL": "Back-Left", "L": "Left", "FL": "Forward-Left"}
PICKS = [6, 24, 45, 66, 87, 107]          # seed + 5 across the generated span
SEED_F, END_F = [2, 6, 9], [96, 100, 104, 107]


def vpath(m, wi, d):
    return f"{ARR}/logs/eval_final/A/{m}/control_test/step05000_r{wi:02d}_{d}_raw.mp4"


# ---------------------------------------------------------------- stage 1
def stage1():
    """Per-rollout scores that need no video decoding."""
    rows = []
    hh = [pd.read_csv(f"{ARR}/analysis/eval_final/headtohead_motion.csv")]
    for f in sorted(glob.glob(f"{ARR}/analysis/eval_final/headtohead_*.csv")):
        if os.path.basename(f) != "headtohead_motion.csv":
            hh.append(pd.read_csv(f))
    hh = pd.concat(hh, ignore_index=True)
    hh["wi"] = hh["window"].astype(int)
    gcols = [f"g{i}" for i in range(8)]
    hidx = {(r.model, int(r.wi), r["dir"]): r for _, r in hh.iterrows()}

    for m in OURS:
        for jf in glob.glob(f"{ARR}/logs/eval_final/A/{m}/control_test/metrics_r*.jsonl"):
            for ln in open(jf):
                try:
                    j = json.loads(ln)
                except Exception:
                    continue
                wi = int(j["rank"])
                for d, v in j.items():
                    if not (isinstance(v, dict) and "tz2" in v):
                        continue
                    tz2 = np.array(v["tz2"], float); tz7 = np.array(v["tz7"], float)
                    c2, c7 = float(np.mean(v["cz2"])), float(np.mean(v["cz7"]))
                    n = np.hypot(c2, c7) + 1e-9
                    proj = (tz2 * c2 + tz7 * c7) / n            # per-chunk speed along command
                    settled = proj[1:]
                    h = hidx.get((m, wi, d))
                    # collision-like stall: strong early motion that dies and stays dead
                    early, late = settled[:3].mean(), settled[-3:].mean()
                    stall = float(early - late) if early > 0.25 else 0.0
                    rows.append(dict(
                        model=m, wi=wi, dir=d,
                        follow=float(settled.mean()), follow_min=float(settled.min()),
                        follow_sd=float(settled.std()), stall=stall,
                        g6=float(h["g6"]) if h is not None else np.nan,
                        g0=float(h["g0"]) if h is not None else np.nan,
                        g1=float(h["g1"]) if h is not None else np.nan,
                        rot=float(h["rot"]) if h is not None else np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/scores_stage1.csv", index=False)
    print(f"[s1] {len(df)} rollouts scored (no decode)")
    return df


# ---------------------------------------------------------------- stage 2
def frames_at(path, idxs):
    want = set(idxs)
    got = {}
    c = av.open(path)
    for i, f in enumerate(c.decode(c.streams.video[0])):
        if i in want:
            got[i] = np.asarray(f.to_image())
        if len(got) == len(want):
            break
    c.close()
    return got


def _sharp(a):
    g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def _orb_inliers(a, b):
    """RANSAC-verified ORB matches: scene identity between two frames."""
    orb = cv2.ORB_create(nfeatures=2000)
    ga, gb = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    ka, da = orb.detectAndCompute(ga, None)
    kb, db = orb.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    good = [mn[0] for mn in bf.knnMatch(da, db, k=2)
            if len(mn) == 2 and mn[0].distance < 0.75 * mn[1].distance]
    if len(good) < 8:
        return 0
    src = np.float32([ka[g.queryIdx].pt for g in good]).reshape(-1, 1, 2)
    dst = np.float32([kb[g.trainIdx].pt for g in good]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else 0


def decode_feats(m, wi, d):
    p = vpath(m, wi, d)
    if not os.path.exists(p):
        return None
    fr = frames_at(p, set(SEED_F + END_F + [6, 24, 45, 66, 87]))
    if not fr:
        return None
    s_seed = np.mean([_sharp(fr[i]) for i in SEED_F if i in fr])
    s_end = np.mean([_sharp(fr[i]) for i in END_F if i in fr])
    inl = _orb_inliers(fr[SEED_F[-1]], fr[max(k for k in fr)])
    # Seed-vs-end overlap is NOT a scene-identity test: a sustained yaw sweeps
    # the view off the seed legitimately (Left rollouts median ~6 inliers while
    # being perfectly good). Scene identity is instead continuity between
    # ADJACENT sampled frames -- a teleport breaks it, a turn does not.
    adj_keys = [k for k in (6, 24, 45, 66, 87, 107) if k in fr]
    adj = [_orb_inliers(fr[a], fr[b]) for a, b in zip(adj_keys, adj_keys[1:])]
    inl_adj = int(min(adj)) if adj else 0
    # interior novelty jump (spawn proxy): biggest step change in the centre crop
    keys = sorted(k for k in fr if k >= 6)
    cen = []
    for k in keys:
        a = fr[k]
        h, w = a.shape[:2]
        cen.append(cv2.cvtColor(a[h // 4:3 * h // 4, w // 4:3 * w // 4], cv2.COLOR_RGB2GRAY).astype(float))
    jumps = [np.abs(cen[i + 1] - cen[i]).mean() for i in range(len(cen) - 1)] or [0]
    return dict(sharp_seed=s_seed, sharp_end=s_end,
                keep=s_end / max(s_seed, 1e-6), inliers=inl, inl_adj=inl_adj,
                center_jump=float(max(jumps)))


# ---------------------------------------------------------------- rendering
def filmstrip(m, wi, d, caption, tw=832):
    p = vpath(m, wi, d)
    fr = frames_at(p, set(PICKS))
    ks = [k for k in PICKS if k in fr]
    tiles = []
    for k in ks:
        a = fr[k]
        im = Image.fromarray(a).resize((tw, int(tw * a.shape[0] / a.shape[1])))
        tiles.append(np.asarray(im))
    h = tiles[0].shape[0]
    pad = 4
    strip = np.full((h + 52, sum(t.shape[1] for t in tiles) + pad * (len(tiles) - 1), 3), 255, np.uint8)
    x = 0
    for t in tiles:
        strip[52:52 + h, x:x + t.shape[1]] = t
        x += t.shape[1] + pad
    im = Image.fromarray(strip)
    ImageDraw.Draw(im).text((8, 14), caption, fill=(0, 0, 0))
    return im


def render_set(name, picks):
    d = f"{OUT}/{name}"
    os.makedirs(d, exist_ok=True)
    sheet = []
    for i, (m, wi, di, cap) in enumerate(picks):
        im = filmstrip(m, wi, di, cap)
        im.save(f"{d}/{i:02d}_{m}_r{wi:02d}_{di}.png")
        sheet.append(im)
        print(f"  [{name}] {i:02d} {cap}")
    W = 3000
    rs = [s.resize((W, int(W * s.height / s.width))) for s in sheet]
    canvas = np.full((sum(r.height + 8 for r in rs), W, 3), 255, np.uint8)
    y = 0
    for r in rs:
        a = np.asarray(r.convert("RGB"))
        canvas[y:y + a.shape[0]] = a
        y += a.shape[0] + 8
    Image.fromarray(canvas).save(f"{OUT}/contact_{name}.png")
    print(f"[{name}] {len(picks)} reels -> {d}  (+ contact_{name}.png)")


# ---------------------------------------------------------------- selection
NAMED = [   # cases flagged by hand, always included in `interesting`
    ("4node", 26, "BR", "a giant human figure spawns mid-rollout"),
    ("noadaln", 26, "F", "a collision is registered though no contact signal is ever given"),
    ("noatok", 30, "FR", "yaw-induced roll rendered at the correct rate"),
]


def enrich(df, mask, tag):
    """Decode features for a shortlist and merge them in."""
    sub = df[mask].copy()
    feats = []
    for _, r in sub.iterrows():
        f = decode_feats(r.model, int(r.wi), r["dir"])
        feats.append(f or {})
    for k in ["sharp_seed", "sharp_end", "keep", "inliers", "inl_adj", "center_jump"]:
        sub[k] = [f.get(k, np.nan) for f in feats]
    print(f"[s2] decoded {len(sub)} candidates for {tag}")
    return sub


BLOCK = {("16node", 6, "BL"), ("pca8_8node", 26, "B")}


def pick_success(df):
    # Rank WITHIN each direction: backward realises ~half the magnitude of
    # forward by construction, so an absolute follow floor would erase the
    # backward commands from a "stratified over directions" gallery.
    cand = (df[df.model.isin(BEST)]
            .sort_values("follow", ascending=False)
            .groupby(["model", "dir"]).head(8))
    cand = enrich(df, df.index.isin(cand.index), "success")
    ok = cand[(cand.keep.between(0.80, 1.8)) & (cand.sharp_end > 250)
              & (cand.inl_adj >= 20) & (cand.follow > 0.10)]
    ok = ok.sort_values("follow", ascending=False)
    # Geometric mangle is invisible to sharpness/IQA (whole-frame features score
    # at chance on it), so visually-rejected rollouts are excluded by hand.
    ok = ok[~ok.apply(lambda r: (r.model, int(r.wi), r["dir"]) in BLOCK, axis=1)]

    picks, chosen, scene_use = [], set(), {}
    per_dir = {d: 0 for d in DIRS}

    def take(r, cap_scene, cap_dir):
        key = (r.model, int(r.wi), r["dir"])
        if key in chosen or per_dir[r["dir"]] >= cap_dir:
            return False
        if scene_use.get(int(r.wi), 0) >= cap_scene:
            return False
        picks.append((r.model, int(r.wi), r["dir"],
                      f"{NICE[r.model]}  scene {int(r.wi):02d}  command {DNAME[r['dir']]}"
                      f"   following {r.follow:+.2f}  sharpness kept {r.keep:.2f}"))
        chosen.add(key)
        scene_use[int(r.wi)] = scene_use.get(int(r.wi), 0) + 1
        per_dir[r["dir"]] += 1
        return True

    # pass 1: two per command. Direction coverage outranks scene-uniqueness --
    # otherwise a later command (e.g. Back-Left) finds all of its scenes already
    # consumed by earlier commands and drops out of the gallery entirely.
    for d in DIRS:
        for cap_s in (1, 2, 3):
            for _, r in ok[ok["dir"] == d].iterrows():
                if per_dir[d] >= 2:
                    break
                take(r, cap_scene=cap_s, cap_dir=2)
            if per_dir[d] >= 2:
                break
    # passes 2-3: fill to N_SET, progressively allowing a scene to recur under
    # a different command
    for cap_s, cap_d in ((2, 3), (3, 4)):
        for _, r in ok.iterrows():
            if len(picks) >= N_SET:
                break
            take(r, cap_scene=cap_s, cap_dir=cap_d)
    return picks[:N_SET]


def pick_failure(df):
    """Four cheaply-detectable failure modes. Geometric mangle is deliberately
    NOT mined here: whole-frame sharpness/IQA features score at chance on it,
    so it cannot be selected automatically."""
    cand = pd.concat([
        df[df.follow < -0.10].sort_values("follow").head(70),
        df[df.follow.abs() < 0.08].head(70),
        df.sort_values("follow").head(40),
        df[df.model.isin(BEST)].sort_values("follow").head(30),
    ]).drop_duplicates(subset=["model", "wi", "dir"])
    cand = enrich(df, df.index.isin(cand.index), "failure")
    cand["mode"] = np.where(cand.keep < 0.55, "sharpness collapse",
                    np.where(cand.inl_adj < 6, "scene relocation",
                     np.where(cand.follow < -0.10, "anti-following",
                      np.where(cand.follow.abs() < 0.08, "no response", "other"))))
    picks, used, per_model = [], set(), {}
    MODE_CAP, MODEL_CAP = 6, 6
    for mode in ["anti-following", "no response", "sharpness collapse", "scene relocation"]:
        sub = cand[cand["mode"] == mode]
        if mode == "anti-following":
            sub = sub.sort_values("follow")
        elif mode == "sharpness collapse":
            sub = sub.sort_values("keep")
        elif mode == "scene relocation":
            sub = sub.sort_values("inl_adj")
        else:
            sub = sub.reindex(sub.follow.abs().sort_values().index)
        n = 0
        for _, r in sub.iterrows():
            key = (r.model, int(r.wi))
            if key in used or n >= MODE_CAP or len(picks) >= N_SET:
                continue
            if per_model.get(r.model, 0) >= MODEL_CAP:
                continue
            det = (f"following {r.follow:+.2f}" if mode in ("anti-following", "no response")
                   else f"sharpness kept {r.keep:.2f}" if mode == "sharpness collapse"
                   else f"{int(r.inl_adj)} frame-to-frame inliers")
            picks.append((r.model, int(r.wi), r["dir"],
                          f"{NICE[r.model]}  scene {int(r.wi):02d}  command {DNAME[r['dir']]}"
                          f"   {mode.upper()}: {det}"))
            used.add(key); n += 1
            per_model[r.model] = per_model.get(r.model, 0) + 1
    # top up if any mode was thin
    for _, r in cand[cand["mode"] != "other"].sort_values("follow").iterrows():
        if len(picks) >= N_SET:
            break
        key = (r.model, int(r.wi))
        if key in used or per_model.get(r.model, 0) >= MODEL_CAP + 2:
            continue
        picks.append((r.model, int(r.wi), r["dir"],
                      f"{NICE[r.model]}  scene {int(r.wi):02d}  command {DNAME[r['dir']]}"
                      f"   {str(r['mode']).upper()}: following {r.follow:+.2f}"))
        used.add(key); per_model[r.model] = per_model.get(r.model, 0) + 1
    return picks[:N_SET]


def pick_interesting(df):
    picks = [(m, w, d, f"{NICE[m]}  scene {w:02d}  command {DNAME[d]}   {note}")
             for m, w, d, note in NAMED]
    taken = {(m, w, d) for m, w, d, _ in NAMED}
    idx = df.set_index(["model", "wi", "dir"])

    roll = df[(df.g6.abs() > 0.28) & (df.follow > 0.15)].sort_values("g6", key=abs, ascending=False)
    stall = df[(df.stall > 0.30)].sort_values("stall", ascending=False)
    cand = pd.concat([roll.head(45), stall.head(45)]).drop_duplicates(subset=["model", "wi", "dir"])
    cand = enrich(df, df.index.isin(cand.index), "interesting")
    spawn = cand.sort_values("center_jump", ascending=False)

    per_model = {}
    for m, _w, _d, _n in NAMED:
        per_model[m] = per_model.get(m, 0) + 1
    MODEL_CAP = 5

    per_scene = {}
    for _m, w, _d, _n in NAMED:
        per_scene[w] = per_scene.get(w, 0) + 1
    SCENE_CAP = 2       # one degenerate seed must not dominate the gallery

    def add(rows, note_fn, cap):
        n = 0
        for _, r in rows.iterrows():
            k = (r.model, int(r.wi), r["dir"])
            if k in taken or len(picks) >= N_SET or n >= cap:
                continue
            if per_model.get(r.model, 0) >= MODEL_CAP:
                continue
            if per_scene.get(int(r.wi), 0) >= SCENE_CAP:
                continue
            picks.append((r.model, int(r.wi), r["dir"],
                          f"{NICE[r.model]}  scene {int(r.wi):02d}  command {DNAME[r['dir']]}"
                          f"   {note_fn(r)}"))
            taken.add(k); n += 1
            per_model[r.model] = per_model.get(r.model, 0) + 1
            per_scene[int(r.wi)] = per_scene.get(int(r.wi), 0) + 1

    add(spawn[spawn.center_jump > 12], lambda r: f"abrupt content appears mid-rollout (jump {r.center_jump:.0f})", 7)
    add(cand[cand.stall > 0.30].sort_values("stall", ascending=False),
        lambda r: f"motion arrests mid-rollout despite a constant command (stall {r.stall:.2f})", 6)
    add(cand[cand.g6.abs() > 0.28].sort_values("g6", key=abs, ascending=False),
        lambda r: f"roll coupled to commanded yaw (roll {r.g6:+.2f})", 6)
    return picks[:N_SET]


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    cache = f"{OUT}/scores_stage1.csv"
    df = pd.read_csv(cache) if os.path.exists(cache) else stage1()
    which = os.environ.get("AR_SETS", "success,failure,interesting").split(",")
    if "success" in which:
        render_set("success", pick_success(df))
    if "failure" in which:
        render_set("failure", pick_failure(df))
    if "interesting" in which:
        render_set("interesting", pick_interesting(df))
