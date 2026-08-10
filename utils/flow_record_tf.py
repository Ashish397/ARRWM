"""Teacher-forced 4-rung recording, window r08: chunk c is generated with
FULL GT context (real latents 0..3c) and the ride's own GT actions —
exposure-bias-free counterpart of the AR rollouts. One committed chunk per
call; 6 calls -> the 'TF line' for the flow trees.

Output: flow_viz/flow_{FR_RUN}/r08_TF_c{c}/steps.npz (same sdt format).
Env: FR_RUN (def tf_gt0), FR_CKPT (none = teacher-init via eval yaml),
FR_CONFIG, FR_SEED (def 1234).
"""
import os, json
os.environ.setdefault("WORLD_SIZE", "1"); os.environ.setdefault("RANK", "0"); os.environ.setdefault("LOCAL_RANK", "0")
os.environ.pop("AF_SNAPSHOT_STEPS", None); os.environ.pop("AF_EVAL_STEPS", None)
import torch

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
RUN = os.environ.get("FR_RUN", "tf_gt0")
CONFIG = os.environ.get("FR_CONFIG", "configs/ar_eval_dmd_student.yaml")
CKPT = os.environ.get("FR_CKPT", "none")
NFB, NCH = 3, 6


def main():
    from omegaconf import OmegaConf
    from utils.eval_causal_AR import ODEChainPipeline
    from utils.zarr_dataset import ZarrRideDataset
    from utils.infinity_rope import infinity_rope_active

    device = "cuda"
    pipe = ODEChainPipeline(device)
    pipe.build(config_path=CONFIG)
    if CKPT.lower() != "none":
        pipe.load_checkpoint(CKPT)
    if os.environ.get("FR_RUNGS"):
        _r = torch.tensor([float(x) for x in os.environ["FR_RUNGS"].split(",")],
                          dtype=torch.float32)
        pipe.denoising_step_list = _r
        pipe.ode_model.denoising_step_list = _r.clone()

    w = json.load(open(f"{ARR}/analysis/eval_final/phaseA_windows.json"))[8]
    zp, off = w["zarr_path"], int(w["offset"])
    manifest = torch.load(f"{ARR}/analysis/eval_final/manifest_unseen.pt", map_location="cpu")
    row = [r for r in manifest if r["zarr_path"] == zp][0]
    pe = row["prompt_embeds"].unsqueeze(0)

    cfg = OmegaConf.merge(OmegaConf.load(f"{ARR}/configs/default_config.yaml"),
                          OmegaConf.load(f"{ARR}/{CONFIG}" if not CONFIG.startswith("/") else CONFIG))
    z_ds = ZarrRideDataset.from_manifest(
        rides_data=[{"zarr_path": zp, "attrs": row["attrs"],
                     "prompt_embeds": row["prompt_embeds"],
                     "n_latent_frames": int(row["n_latent_frames"])}],
        motion_root=str(cfg.get("motion_root", "/projects/u6ex/fbots/frodobots_motion")),
        ss_vae_checkpoint=str(cfg.ss_vae_checkpoint),
        device="cpu", ss_vae_device=device)
    tot_f = NFB * (1 + NCH)
    z_full = z_ds.encode_z_actions_window(
        zp, int(row["n_latent_frames"]), off, off + tot_f)      # [21, zdim]
    dims = list(cfg.get("action_dims", [2, 7]))
    z_act = z_full[:, dims].to(torch.float32)                    # [21, 2]
    gt = ZarrRideDataset.load_latent_chunk(zp, off, off + tot_f).unsqueeze(0).to(device, torch.float32)

    base = pipe.wrapper.model
    if hasattr(base, "get_base_model"):
        base = base.get_base_model()
    for c in range(1, NCH + 1):
        dst = f"{ARR}/analysis/eval_final/flow_viz/flow_{RUN}/r08_TF_c{c}"
        if os.path.exists(f"{dst}/steps.npz"):
            print(f"[tf] {dst} exists, skipping", flush=True); continue
        ctx = gt[:, :NFB * c]
        z = z_act[:NFB * (c + 1)].unsqueeze(0).to(device)
        os.environ["ODE_FLOW_REC"] = dst
        os.environ["ODE_FLOW_SEED"] = os.environ.get("FR_SEED", "1234")
        try:
            with infinity_rope_active(True, base):
                pipe.generate_ar(prompt_embeds=pe, noisy_fa_full=z,
                                 initial_latents=ctx, num_gen_chunks=1,
                                 cache_chunks=6, ar_cache=False)
        finally:
            os.environ.pop("ODE_FLOW_REC", None)
        print(f"[tf] chunk {c} done (ctx={NFB*c} GT frames)", flush=True)
    print("[tf] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
