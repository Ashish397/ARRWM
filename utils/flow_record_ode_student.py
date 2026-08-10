"""Record few-step trajectories of an ODE-distilled student — window r08.

Same protocol as utils/flow_record.py's teacher recording (8 compass dirs x
FR_NSEEDS seeds, real 3-frame seed context, 2 generated chunks) but through
the student's own streaming sampler (ODEChainPipeline.generate_ar: per chunk,
fresh noise -> denoising_step_list [1000,750,500,250] with pred_x0 re-noised
between rungs). The env-gated hook in utils/eval_causal_AR.py pins the block
noise to the SAME generator stream as the teacher recorder (seed_base + block
index), so student chunk-0 initial noise is IDENTICAL to the teacher's
block-0 noise -> teacher-ODE vs student paths share their starting point.

Recorded per (dir, seed): flow_{run}/r{W}_{d}_s{sd}/steps.npz with
sdt=[(chunk, rung, t)] rows (rung=-1 initial noise; t>0 pred_x0 at rung;
t<0 re-noised input at |t|) + x{j} fp16 latents [1, 3, C, H, W].

Env: FR_RUN, FR_CONFIG (student action_ode_* yaml), FR_CKPT, FR_WINDOW
(def 8), FR_NSEEDS (def 4), FR_OUT, FR_TAG (suffix on the output dir, e.g.
"_det"), FR_CHUNKS (def 2), FR_DET=1 (deterministic straight-path re-noise
via the ODE_FLOW_DET hook), FR_VIDEO=1 (also decode the full rollout and
save an mp4 under .motion_check/ for realized-egomotion checks).
"""
import os, json
os.environ.setdefault("WORLD_SIZE", "1"); os.environ.setdefault("RANK", "0"); os.environ.setdefault("LOCAL_RANK", "0")
# Pilot TRAINING jobs export AF_SNAPSHOT_STEPS/AF_EVAL_STEPS (pinned 14e
# grid), but this recorder builds from the eval yaml whose random_steps
# assume the DEFAULT snapshot grid — the leaked env crashes the build
# ("step value 36 is not one of SNAPSHOT_STEPS"). Serving rungs are pinned
# separately via FR_RUNGS, so the training grid is irrelevant here: drop it.
os.environ.pop("AF_SNAPSHOT_STEPS", None); os.environ.pop("AF_EVAL_STEPS", None)
import torch

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
RUN = os.environ["FR_RUN"]
CONFIG = os.environ["FR_CONFIG"]
CKPT = os.environ["FR_CKPT"]
WINDOW = int(os.environ.get("FR_WINDOW", "8"))
NSEEDS = int(os.environ.get("FR_NSEEDS", "4"))
OUT = os.environ.get("FR_OUT", f"{ARR}/analysis/eval_final/flow_viz")
TAG = os.environ.get("FR_TAG", "")
VIDEO = bool(os.environ.get("FR_VIDEO"))
if os.environ.get("FR_DET"):
    os.environ["ODE_FLOW_DET"] = "1"

M = 0.5; Dv = M / (2 ** 0.5)
DIRS = {"F": (M, 0.0), "FR": (Dv, Dv), "R": (0.0, M), "BR": (-Dv, Dv),
        "B": (-M, 0.0), "BL": (-Dv, -Dv), "L": (0.0, -M), "FL": (Dv, -Dv)}
if os.environ.get("FR_NOOP"):          # stationary/no-op branch
    DIRS = {"N": (0.0, 0.0)}
   # (throttle, steer)
NFB = 3
GEN_CHUNKS = int(os.environ.get("FR_CHUNKS", "2"))
# EXPOSURE-BIAS TEST (audit finding A1): training ALWAYS supplies a
# 21-frame / 7-chunk clean context, but serving starts from ONE real
# chunk, so chunk k runs at context depth 3(k+1) = 3..18 and never
# reaches the trained 21 -- and all but the first chunk is the
# student's OWN output. FR_SEED_CHUNKS>1 seeds with more REAL context
# to separate "context too shallow" from "context is self-generated".
SEED_CHUNKS = int(os.environ.get("FR_SEED_CHUNKS", "1"))


def main():
    from utils.eval_causal_AR import ODEChainPipeline
    from utils.zarr_dataset import ZarrRideDataset

    device = "cuda"
    pipe = ODEChainPipeline(device)
    pipe.build(config_path=CONFIG)
    if CKPT.lower() == "none":
        step = "init"        # teacher-init: eval yaml's generator_ckpt only
    else:
        step = pipe.load_checkpoint(CKPT)
    if os.environ.get("FR_RUNGS"):                # pinned-grid override (14e pilots)
        _rungs = torch.tensor(
            [float(x) for x in os.environ["FR_RUNGS"].split(",")],
            dtype=torch.float32)
        pipe.denoising_step_list = _rungs
        pipe.ode_model.denoising_step_list = _rungs.clone()  # keep set_denoising_steps consistent
    print(f"[rec-ode] {RUN}: loaded {CKPT} (step {step}), "
          f"denoise steps {pipe.denoising_step_list.tolist()}", flush=True)

    windows = json.load(open(f"{ARR}/analysis/eval_final/phaseA_windows.json"))
    w = windows[WINDOW]
    zp, off = w["zarr_path"], int(w["offset"])
    seed = ZarrRideDataset.load_latent_chunk(
        zp, off, off + NFB * SEED_CHUNKS).unsqueeze(0).to(device, torch.float32)
    manifest = torch.load(f"{ARR}/analysis/eval_final/manifest_unseen.pt", map_location="cpu")
    pe = {r["zarr_path"]: r["prompt_embeds"] for r in manifest}[zp].unsqueeze(0)

    tot_f = NFB * (SEED_CHUNKS + GEN_CHUNKS)
    for dname, (thr, ste) in DIRS.items():
        for sd in range(NSEEDS):
            dst = f"{OUT}/flow_{RUN}{TAG}/r{WINDOW:02d}_{dname}_s{sd}"
            vid = f"{OUT}/.motion_check/{RUN}{TAG}/r{WINDOW:02d}_{dname}_s{sd}.mp4"
            if os.path.exists(f"{dst}/steps.npz") and (not VIDEO or os.path.exists(vid)):
                print(f"[rec-ode] {dst} exists, skipping", flush=True)
                continue
            z = torch.zeros(1, tot_f, 2, device=device, dtype=torch.float32)
            z[:, NFB * SEED_CHUNKS:, 0] = thr
            z[:, NFB * SEED_CHUNKS:, 1] = ste
            os.environ["ODE_FLOW_REC"] = dst
            os.environ["ODE_FLOW_SEED"] = str(1234 + sd * 7919)
            # CRITICAL: production AR evals wrap generate_ar in the
            # Infinity-RoPE context (un-roped K in cache); without it the
            # bare cached-RoPE path drifts ~1 token/frame per chunk
            # (staircase-left artifact). Match production exactly.
            from utils.infinity_rope import infinity_rope_active
            _base = pipe.wrapper.model
            if hasattr(_base, "get_base_model"):
                _base = _base.get_base_model()
            try:
                with infinity_rope_active(True, _base):
                    # cache_chunks=6 pins the attention span to the trained
                    # 21-frame window (= teacher local_attn_chunks=7)
                    full = pipe.generate_ar(prompt_embeds=pe, noisy_fa_full=z,
                                            initial_latents=seed,
                                            num_gen_chunks=GEN_CHUNKS,
                                            cache_chunks=6)
            finally:
                os.environ.pop("ODE_FLOW_REC", None)
            if VIDEO:
                import imageio
                os.makedirs(os.path.dirname(vid), exist_ok=True)
                frames = pipe.decode_latents(full.to(device))
                imageio.mimsave(vid, frames, fps=5, quality=7)
                print(f"[rec-ode] saved {vid} ({frames.shape[0]}f)", flush=True)
            print(f"[rec-ode] {RUN} {dname} seed{sd} done", flush=True)
    print(f"[rec-ode] {RUN} ALL DONE", flush=True)


if __name__ == "__main__":
    main()
