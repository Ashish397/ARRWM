"""Replace the 8 NIGHT eval windows with fresh DAYTIME unseen windows (same slots).

Night (tod attr): phase A slots [0,1,4,10]; phase B slots [42(big),50,56,63(massive)].
Replacements must be: unseen ride (not train/eval/already-used), tod in
{morning,midday,afternoon}, non-moving (ego<=0.12), distinct rides; phase B must
match the removed scene-bins using the ORIGINAL fixed boundaries
(big: 0.29<=scene<0.67, massive: scene>=0.67).

Updates phaseA_windows.json / phaseB_windows.json IN PLACE (backups saved),
appends new rides to manifest_unseen.pt.
"""
import os, json, shutil
os.environ.setdefault("ARRWM_ACTION_ENCODER", "pca_raw")
import numpy as np, torch
from omegaconf import OmegaConf
from utils.zarr_dataset import ZarrRideDataset

OUT = "analysis/eval_final"
A_SLOTS = [int(x) for x in os.environ.get("RS_A_SLOTS", "0,1,4,10").replace(":", ",").split(",") if x.strip()]
_b = os.environ.get("RS_B_SLOTS", "42;big,50;massive,56;massive,63;massive").replace(":", ",").replace(";", ":")
B_SLOTS = {int(kv.split(":")[0]): kv.split(":")[1] for kv in _b.split(",") if ":" in kv}
BIN_LO = {"big": 0.29, "massive": 0.67}
BIN_HI = {"big": 0.67, "massive": 1e9}
EGO_HARD = 0.12
K_STARTS = 8

cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                      OmegaConf.load("configs/causal_lora_diffusion_teacher_v14e_16node.yaml"))
NFB = int(cfg.num_frame_per_block); GEN = int(cfg.get("control_gen_chunks", 8))
TOT_F = (1 + GEN) * NFB
CK = "action_query/checkpoints/ss_vae_8free.pt"


def main():
    attrs = json.load(open("paper_assets/v14d_ride_attrs.json"))
    tw = json.load(open("paper_assets/v14d_train_windows.json"))
    train = set(os.path.basename(x["zarr_path"]) for x in tw["windows"])
    evalr = set(cfg.get("eval_ride_zarrs", []) or [])
    wA = json.load(open(f"{OUT}/phaseA_windows.json"))
    wB = json.load(open(f"{OUT}/phaseB_windows.json"))
    used = {w["ride"] for w in wA} | {w["ride"] for w in wB}

    def daytime(zbase):
        return str(attrs.get(zbase.replace(".zarr", ""), {}).get("tod", "")).lower() in ("morning", "midday", "afternoon")

    print("[resample] loading full manifest cache ...", flush=True)
    cached = torch.load(f"{OUT}/.manifest_full.pt", map_location="cpu")
    rides = cached["rides"] if isinstance(cached, dict) else cached
    cand_rides = [r for r in rides
                  if os.path.basename(r["zarr_path"]) not in train
                  and os.path.basename(r["zarr_path"]) not in evalr
                  and os.path.basename(r["zarr_path"]) not in used
                  and daytime(os.path.basename(r["zarr_path"]))]
    print(f"[resample] {len(cand_rides)} candidate daytime unseen rides", flush=True)
    ds = ZarrRideDataset.from_manifest(rides_data=cand_rides, motion_root=str(cfg.motion_root), ss_vae_checkpoint=CK)

    cands = []
    for r in cand_rides:
        zp, nlat = r["zarr_path"], int(r["n_latent_frames"])
        if nlat < TOT_F + NFB:
            continue
        try:
            mag = ds.load_motion_magnitudes(zp, nlat)
        except Exception:
            continue
        for off in sorted(set(int(s) for s in np.linspace(0, nlat - TOT_F, K_STARTS).astype(int))):
            try:
                z = ds.encode_z_actions_window(zp, nlat, off, off + TOT_F).numpy()
            except Exception:
                continue
            ego = float(np.hypot(z[:, 0], z[:, 1]).mean())
            if ego > EGO_HARD:
                continue
            cands.append(dict(ride=os.path.basename(zp), zarr_path=zp, offset=int(off),
                              ego=round(ego, 4), scene=round(float(mag[off:off + TOT_F].mean()), 4)))
    print(f"[resample] {len(cands)} non-moving daytime candidate windows", flush=True)

    picked_rides = set()
    # Phase A: 4 most-non-moving, distinct rides
    newA = []
    for c in sorted(cands, key=lambda c: c["ego"]):
        if c["ride"] in picked_rides:
            continue
        picked_rides.add(c["ride"]); newA.append(c)
        if len(newA) == len(A_SLOTS):
            break
    # Phase B: match bins, distinct rides, most-non-moving within bin
    newB = {}
    for slot, b in B_SLOTS.items():
        pool = [c for c in cands if BIN_LO[b] <= c["scene"] < BIN_HI[b] and c["ride"] not in picked_rides]
        pool.sort(key=lambda c: c["ego"])
        assert pool, f"no candidate for bin {b}"
        newB[slot] = {**pool[0], "bin": b}
        picked_rides.add(pool[0]["ride"])

    for f in ("phaseA_windows.json", "phaseB_windows.json", "manifest_unseen.pt"):
        shutil.copy(f"{OUT}/{f}", f"{OUT}/{f}.night_backup")
    for slot, c in zip(A_SLOTS, newA):
        print(f"[A] slot r{slot:02d}: {wA[slot]['ride']} (night) -> {c['ride']} ego={c['ego']} scene={c['scene']}")
        wA[slot] = c
    for slot, c in newB.items():
        print(f"[B] slot r{slot:02d}: {wB[slot]['ride']} (night) -> {c['ride']} bin={c['bin']} ego={c['ego']} scene={c['scene']}")
        wB[slot] = c
    json.dump(wA, open(f"{OUT}/phaseA_windows.json", "w"), indent=1)
    json.dump(wB, open(f"{OUT}/phaseB_windows.json", "w"), indent=1)

    man = torch.load(f"{OUT}/manifest_unseen.pt", map_location="cpu")
    have = {r["zarr_path"] for r in man}
    need = {c["zarr_path"] for c in newA} | {c["zarr_path"] for c in newB.values()}
    add = [r for r in rides if r["zarr_path"] in need and r["zarr_path"] not in have]
    torch.save(man + add, f"{OUT}/manifest_unseen.pt")
    print(f"[resample] manifest_unseen.pt += {len(add)} rides. DONE")


if __name__ == "__main__":
    main()
