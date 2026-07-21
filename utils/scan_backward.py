"""Ground-truth census of backward motion in the action DB.

For every ride we encode the full throttle track (PC0, tanh-squashed z0) and count,
at BOTH chunk grain (3 latents) and non-overlapping window grain (tot_f=27):
  forward  : mean z0 >  TH
  backward : mean z0 < -TH
  stationary: |mean z0| <= TH
at two thresholds (0.10 mild, 0.30 clear). Reports availability per data pool so we
can see whether backward is genuinely rare or was under-sampled when the 632-window
training manifest was built.

Env: SB_POOLS (comma names), SB_OUT (json).
"""
import os, json, glob
os.environ.setdefault("ARRWM_ACTION_ENCODER", "pca_raw")
os.environ.setdefault("WORLD_SIZE", "1"); os.environ.setdefault("RANK", "0"); os.environ.setdefault("LOCAL_RANK", "0")
import numpy as np, torch
from omegaconf import OmegaConf
from utils.zarr_dataset import build_ride_manifest, ZarrRideDataset

cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                      OmegaConf.load("configs/causal_lora_diffusion_teacher_v14e_16node.yaml"))
NFB = int(cfg.num_frame_per_block); GEN = int(cfg.get("control_gen_chunks", 8))
TOT_F = (1 + GEN) * NFB
CAP = str(cfg.caption_root); MOT = str(cfg.motion_root)
CK = "action_query/checkpoints/ss_vae_8free.pt"
OUT = os.environ.get("SB_OUT", "analysis/eval_final/backward_census.json")
THS = [0.10, 0.30]

POOLS = {
    "weunz_TRAIN_SOURCE": "/projects/u6ex/fbots/frodobots_encoded_weunz",
    "frodobots_encoded_FULL": "/projects/u6ex/fbots/frodobots_encoded",
    "weu_rear_CAMERA": "/projects/u6ex/fbots/frodobots_encoded_weu_rear",
}
want = os.environ.get("SB_POOLS")
if want:
    POOLS = {k: v for k, v in POOLS.items() if k in set(want.split(","))}


def census(root):
    rides = build_ride_manifest(root, CAP, motion_root=MOT,
                                cache_path=os.path.join("analysis/eval_final", ".mf_" + os.path.basename(root) + ".pt"))
    ds = ZarrRideDataset.from_manifest(rides_data=rides, motion_root=MOT, ss_vae_checkpoint=CK)
    res = {f"{th}": dict(win_f=0, win_b=0, win_s=0, ch_f=0, ch_b=0, ch_s=0, b_rides=set()) for th in THS}
    n_ok = 0
    for r in rides:
        zp, nlat = r["zarr_path"], int(r["n_latent_frames"])
        if nlat < TOT_F:
            continue
        try:
            z0 = ds.encode_z_actions_window(zp, nlat, 0, nlat).numpy()[:, 0]   # throttle track
        except Exception:
            continue
        n_ok += 1
        nch = nlat // NFB
        zc = z0[:nch * NFB].reshape(nch, NFB).mean(1)                          # per-chunk throttle
        wins = np.array([z0[o:o + TOT_F].mean() for o in range(0, nlat - TOT_F + 1, TOT_F)])
        for th in THS:
            R = res[f"{th}"]
            R["ch_f"] += int((zc > th).sum()); R["ch_b"] += int((zc < -th).sum()); R["ch_s"] += int((np.abs(zc) <= th).sum())
            R["win_f"] += int((wins > th).sum()); R["win_s"] += int((np.abs(wins) <= th).sum())
            nb = int((wins < -th).sum()); R["win_b"] += nb
            if nb:
                R["b_rides"].add(zp)
    for th in THS:
        res[f"{th}"]["b_rides"] = len(res[f"{th}"]["b_rides"])
    res["_n_rides"] = n_ok
    return res


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    allres = {}
    for name, root in POOLS.items():
        if not os.path.isdir(root):
            print(f"[skip] {name}: {root} absent"); continue
        print(f"\n===== {name}  ({root}) =====", flush=True)
        r = census(root); allres[name] = r
        print(f"  rides scanned: {r['_n_rides']}")
        for th in THS:
            R = r[f"{th}"]
            wt = R["win_f"] + R["win_b"] + R["win_s"]; ct = R["ch_f"] + R["ch_b"] + R["ch_s"]
            print(f"  TH={th}:  WINDOWS fwd={R['win_f']} back={R['win_b']} stat={R['win_s']} "
                  f"(back {100*R['win_b']/max(1,wt):.1f}%, {R['b_rides']} rides w/ backward)")
            print(f"           CHUNKS  fwd={R['ch_f']} back={R['ch_b']} stat={R['ch_s']} "
                  f"(back {100*R['ch_b']/max(1,ct):.1f}%)")
    json.dump(allres, open(OUT, "w"), indent=1)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
