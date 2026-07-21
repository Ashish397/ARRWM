"""Mine held-out rides for windows with MODERATE-LEFT steer segments.

Purpose: the FLIP counterfactual steer curve (analysis/response_complete_steer)
is sparse at commanded +0.35..+0.8 because flipped commands there require real
moderate-LEFT turns, and the 4 training control rides barely contain any
(their lefts are sharp corners that saturate the tanh). This selects windows
from the 105-ride manifest_unseen pool whose settled generated chunks carry
steer in the (-0.8, -0.35) band, for a phase-FL (flip eval) inject_eval run.

Criterion per window (9 chunks = 27 latents; seed chunk 0, settled = chunks
2..8): >= MIN_CHUNKS settled chunks with steer command in the band. Windows
stride one chunk; at most PER_RIDE picks per ride (best-count first, no
overlap). Writes analysis/eval_final/left_windows.json.
"""
import os, json
os.environ.setdefault("ARRWM_ACTION_ENCODER", "pca_raw")
import numpy as np
import torch

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
NFB, N_CHUNKS = 3, 9
TOT_F = NFB * N_CHUNKS
SETTLE = 2                       # match the settled convention downstream
BAND = (-0.80, -0.35)            # moderate LEFT (real, pre-flip)
MIN_CHUNKS = 3
PER_RIDE = 2
CAP = 48
CK = "action_query/checkpoints/ss_vae_8free.pt"


def main():
    from omegaconf import OmegaConf
    from utils.zarr_dataset import ZarrRideDataset
    cfg = OmegaConf.merge(OmegaConf.load(f"{ARR}/configs/default_config.yaml"),
                          OmegaConf.load(f"{ARR}/configs/causal_lora_diffusion_teacher_v14e.yaml"))
    manifest = torch.load(f"{ARR}/analysis/eval_final/manifest_unseen.pt", map_location="cpu")
    ds = ZarrRideDataset.from_manifest(rides_data=manifest, motion_root=cfg.motion_root,
                                       ss_vae_checkpoint=CK)
    picks, stats = [], []
    for r in manifest:
        zp, n_lat = r["zarr_path"], int(r["n_latent_frames"])
        if n_lat < TOT_F:
            continue
        try:
            z = ds.encode_z_actions_window(zp, n_lat, 0, (n_lat // NFB) * NFB)
        except Exception as e:
            print(f"[mine] {os.path.basename(zp)} FAILED: {e}")
            continue
        steer = z[::NFB, 1].numpy()                     # one value per chunk
        nchunks = len(steer)
        cands = []
        for c0 in range(0, nchunks - N_CHUNKS + 1):     # window = chunks [c0, c0+9)
            settled = steer[c0 + 1 + SETTLE : c0 + N_CHUNKS]
            hits = int(((settled > BAND[0]) & (settled < BAND[1])).sum())
            if hits >= MIN_CHUNKS:
                cands.append((hits, c0))
        cands.sort(reverse=True)
        used = []
        for hits, c0 in cands:
            if any(abs(c0 - u) < N_CHUNKS for u in used):
                continue                                # no overlapping windows
            used.append(c0)
            picks.append({"zarr_path": zp, "offset": c0 * NFB, "hits": hits})
            if len(used) >= PER_RIDE:
                break
        if used:
            stats.append((os.path.basename(zp), len(used)))
    picks.sort(key=lambda p: -p["hits"])
    picks = picks[:CAP]
    out = f"{ARR}/analysis/eval_final/left_windows.json"
    json.dump(picks, open(out, "w"), indent=1)
    hits = [p["hits"] for p in picks]
    print(f"[mine] {len(picks)} windows from {len(set(p['zarr_path'] for p in picks))} rides "
          f"(hits per window: min {min(hits)}, median {int(np.median(hits))}, max {max(hits)})")
    print(f"[mine] saved {out}")


if __name__ == "__main__":
    main()
