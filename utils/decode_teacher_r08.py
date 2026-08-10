"""Decode the EXISTING 14e teacher 20-step compass recordings (window r08)
to mp4 — no generation: pure VAE decode of trajs_14e8s20_w8.npz committed
blocks (real 3-frame seed from zarr + generated blocks b0..b5 = 21 latent
frames, matching the student probe rollout length exactly).
Output: .motion_check/teacher14e8s20/r08_{DIR}_s0.mp4
"""
import os
import numpy as np
import torch

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
FV = f"{ARR}/analysis/eval_final/flow_viz"
DIRS = ["F", "FR", "R", "BR", "B", "BL", "L", "FL"]
NFB = 3


def main():
    import json, sys
    sys.path.insert(0, ARR)
    from utils.eval_causal_AR import ODEChainPipeline
    from utils.zarr_dataset import ZarrRideDataset
    import imageio

    pipe = ODEChainPipeline("cuda")
    pipe.build(config_path=os.environ.get(
        "DT_CONFIG", "configs/ar_eval_dmd_student.yaml"))

    w = json.load(open(f"{ARR}/analysis/eval_final/phaseA_windows.json"))[8]
    seed = np.asarray(ZarrRideDataset.load_latent_chunk(
        w["zarr_path"], int(w["offset"]), int(w["offset"]) + NFB),
        dtype=np.float32)                                # [3, C, H, W]

    z = np.load(f"{FV}/trajs_14e8s20_w8.npz")
    out_dir = f"{FV}/.motion_check/teacher14e8s20"
    os.makedirs(out_dir, exist_ok=True)
    for d in DIRS:
        blocks = [seed]
        for b in range(6):
            k = f"{d}_0" if b == 0 else f"b{b}_{d}_0"
            blocks.append(z[k][-1].astype(np.float32).reshape(NFB, 16, 60, 104))
        lat = torch.from_numpy(np.concatenate(blocks, 0)).unsqueeze(0).cuda()
        frames = pipe.decode_latents(lat)                # [T, H, W, 3] uint8
        p = f"{out_dir}/r08_{d}_s0.mp4"
        imageio.mimsave(p, frames, fps=5, quality=7)
        print(f"[dt] saved {p} ({frames.shape[0]}f)", flush=True)
    print("[dt] DONE", flush=True)


if __name__ == "__main__":
    main()
