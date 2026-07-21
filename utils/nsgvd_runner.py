"""NSG-VD (NeurIPS'25) on our eval videos, using their components end-to-end.

Layout masquerade (their dataset asserts known generator names):
  custom_data/video_frames/fake/ModelScope/test/<id>/frame1..8.jpg   = FIRST-32-frame clips
  custom_data/video_frames/fake/Sora/test/<id>/frame1..8.jpg        = LAST-32-frame clips
  custom_data/video_frames/real/MSR-VTT/val/<id>/frame1..8.jpg      = 32 real reference clips

Pipeline: ScoreFeaturesDataset auto-extracts NSG (velocity) features via their
guided-diffusion score estimator (5 steps), then deep-MMD per-sample statistic vs
the real reference features (higher = more fake). Scores both SEINE ckpts (mp+d).
Run with cwd = third_party/NSG-VD. Saves ../../analysis/eval_final/nsgvd_scores.csv
"""
import os, sys, glob
import numpy as np, pandas as pd, imageio, torch
from PIL import Image

REPO = os.getcwd()                                   # third_party/NSG-VD
ARR = os.path.abspath(os.path.join(REPO, "..", ".."))
sys.path.insert(0, REPO)
DATA = os.path.join(REPO, "custom_data")
RUNS = os.environ.get("NV_RUNS", "pca8_8node,pca4,pca2,16node,4node,noatok").replace(":", ",").split(",")
TARGETS = [(1, "R"), (8, "B"), (8, "BL")]

# ---------- 1) frame prep ----------

_MWSW = {"L": "R", "R": "L", "FL": "FR", "FR": "FL", "BL": "BR", "BR": "BL"}
def _vid_path(run, rank, br, arr=ARR):
    if run == "minwm":   # disk labels are yaw-sign-flipped; swap to reach TRUE direction
        return f"{arr}/logs/eval_final/A_minwm/minwm_r{rank:02d}_{_MWSW.get(br, br)}.mp4"
    return f"{arr}/logs/eval_final/A/{run}/control_test/step05000_r{rank:02d}_{br}_raw.mp4"

def dump_frames(frames, outdir):
    os.makedirs(outdir, exist_ok=True)
    for i, f in enumerate(frames):
        Image.fromarray(f).save(os.path.join(outdir, f"frame{i+1}.jpg"), quality=95)

def prep():
    def _done(rank, br, run):
        return os.path.isdir(f"{DATA}/video_frames/fake/Sora/test/r{rank:02d}{br}_{run}")
    for rank, br in TARGETS:
        for run in RUNS:
            if _done(rank, br, run):
                continue
            p = _vid_path(run, rank, br)
            r = imageio.get_reader(p); fr = [np.asarray(f) for f in r]; r.close()
            n = len(fr)
            vid = f"r{rank:02d}{br}_{run}"
            first = [fr[i] for i in np.linspace(0, min(31, n - 1), 8).astype(int)]
            last = [fr[i] for i in np.linspace(max(0, n - 32), n - 1, 8).astype(int)]
            dump_frames(first, f"{DATA}/video_frames/fake/ModelScope/test/{vid}")
            dump_frames(last, f"{DATA}/video_frames/fake/Sora/test/{vid}")
    for p in sorted(glob.glob(f"{ARR}/analysis/eval_final/real_refs/*.mp4")):
        r = imageio.get_reader(p); fr = [np.asarray(f) for f in r]; r.close()
        n = len(fr)
        mid = [fr[i] for i in np.linspace(max(0, n // 2 - 16), min(n - 1, n // 2 + 16), 8).astype(int)]
        vid = os.path.basename(p).replace(".mp4", "")
        dump_frames(mid, f"{DATA}/video_frames/real/MSR-VTT/val/{vid}")
    open(os.path.join(DATA, ".prepped"), "w").write("ok")
    print("[prep] done")

def write_splits():
    """{DATA}/split/{label}/{gen}/{mode}_ids.txt — one '<id>.mp4' per line."""
    for label, gen, mode in [("fake", "ModelScope", "test"), ("fake", "Sora", "test"), ("real", "MSR-VTT", "val")]:
        fdir = f"{DATA}/video_frames/{label}/{gen}/{mode}"
        ids = sorted(os.listdir(fdir)) if os.path.isdir(fdir) else []
        sdir = f"{DATA}/split/{label}/{gen}"
        os.makedirs(sdir, exist_ok=True)
        with open(f"{sdir}/{mode}_ids.txt", "w") as f:
            f.write("\n".join(i + ".mp4" for i in ids) + "\n")
        print(f"[split] {label}/{gen}/{mode}: {len(ids)} ids")


# ---------- 2) NSG features + 3) scoring ----------
def main():
    prep()
    write_splits()
    from data.feature_dataset.score_feature_dataset import ScoreFeaturesDataset
    from models.deep_mmd import deep_MMD
    from models.tall import SingleSwinBlockDiscriminator
    from utils.mmd_utils import MMD_batch2

    def make_ds(gen, mode):
        return ScoreFeaturesDataset(
            data_path=DATA, dataset_name="GenVideo", generation_model=gen, mode=mode,
            num_frames=8, input_shape=(224, 224), diffuse_steps=5,
            score_config_path=f"{REPO}/libs/eps_ad/args.yml",
            score_args_path=f"{REPO}/libs/eps_ad/imagenet.yml",
            feature_type="velocity", load_len=200, resolution_size=224,
            process_batch_size=2)

    ds_ref = make_ds("MSR-VTT", "val")
    ds_first = make_ds("ModelScope", "test")
    ds_last = make_ds("Sora", "test")
    print(f"[ds] ref={len(ds_ref)} first={len(ds_first)} last={len(ds_last)}")

    def stack(ds):
        feats, ids = [], []
        for i in range(len(ds)):
            x, _ = ds[i]
            feats.append(torch.as_tensor(np.asarray(x)).float())
            ids.append(ds.video_ids[i] if hasattr(ds, "video_ids") else str(i))
        return torch.stack(feats), ids

    X_ref, id_ref = stack(ds_ref)
    X_first, id_first = stack(ds_first)
    X_last, id_last = stack(ds_last)

    rows = []
    for ck, is_yy_zero in [("standard-SEINE-mp.pth", True), ("standard-SEINE-d.pth", False)]:
        disc = SingleSwinBlockDiscriminator(num_features=300, duration=8)
        model = deep_MMD(discriminator=disc, sigma=1000, sigma0=0.1, epsilon=10,
                         img_size=224, is_yy_zero=is_yy_zero, is_smooth=True)
        model.load_state_dict(torch.load(f"{REPO}/ckpts/{ck}", map_location="cpu", weights_only=True))
        model = model.cuda().eval()
        net, sigma, sigma0_u, ep = model.net, model.sigma, model.sigma0_u, model.ep

        with torch.no_grad():
            def feats_of(X):
                out = []
                for i in range(0, len(X), 8):
                    b = X[i:i + 8].cuda()
                    r = net(b, out_feature=True)
                    f = r[1] if isinstance(r, tuple) else r
                    out.append(f)
                return torch.cat(out)
            F_ref = feats_of(X_ref).cuda()
            R_ref = X_ref.view(len(X_ref), -1).cuda()

            def mmd_scores(X, F):
                vals = []
                for i in range(len(X)):
                    f = F[i:i + 1].cuda()
                    d = X[i:i + 1].view(1, -1).cuda()
                    v = MMD_batch2(torch.cat([F_ref, f], 0), F_ref.shape[0],
                                   torch.cat([R_ref, d], 0), sigma, sigma0_u, ep,
                                   is_smooth=model.is_smooth)
                    vals.append(float(v.flatten()[-1].item()))
                return vals
            Ff = feats_of(X_first); Fl = feats_of(X_last)
            s_first = mmd_scores(X_first, Ff)
            s_last = mmd_scores(X_last, Fl)
            s_ref = mmd_scores(X_ref, F_ref)

        for i, vid in enumerate(id_first):
            rows.append(dict(ckpt=ck.replace(".pth", ""), video=vid, clip="first", score=round(s_first[i], 5)))
        for i, vid in enumerate(id_last):
            rows.append(dict(ckpt=ck.replace(".pth", ""), video=vid, clip="last", score=round(s_last[i], 5)))
        for i, vid in enumerate(id_ref):
            rows.append(dict(ckpt=ck.replace(".pth", ""), video=vid, clip="real_ref", score=round(s_ref[i], 5)))
        print(f"[{ck}] real_ref mean={np.mean(s_ref):.5f}  first mean={np.mean(s_first):.5f}  last mean={np.mean(s_last):.5f}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{ARR}/analysis/eval_final/nsgvd_scores.csv", index=False)
    print("\n=== per-window LAST-clip ranking (higher = more fake per NSG-VD) ===")
    for ck in df.ckpt.unique():
        s = df[(df.ckpt == ck) & (df.clip == "last")]
        for w in ["r01R", "r08B", "r08BL"]:
            ss = s[s.video.str.startswith(w)].sort_values("score")
            print(f" [{ck} {w}] " + " ".join(f"{v.split('_',1)[1]}:{x:.4f}" for v, x in zip(ss.video, ss.score)))


if __name__ == "__main__":
    main()
