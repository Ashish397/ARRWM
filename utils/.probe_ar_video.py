"""Decoded, frame-synchronised AR comparison videos.

The only videos phase-3 produces today come from the TEACHER-FORCED 42f
head, where the teacher is handed the GT clean half of the very window it
scores — so it always looks excellent and says nothing about the AR
question. The AR head computes ``pred_real_ar`` and throws the tensor away
after reducing it to MAE. This script keeps the tensor and decodes it.

ROWS (all the SAME world frames, frame-synchronised, one tile each):
  GT                    the real latents
  student_AR            the student rolling its OWN chunks through its own
                        KV cache from a real 3-chunk seed (the baseline)
  teacher_AR_stuCTX     the SAME student chunk, noised to rung t, denoised
                        single-shot by the frozen v14e teacher served AR
                        through a KV cache committed with STUDENT content
                        (dmd_ar_head_commit="student" — what the AR head
                        actually trains against)
  teacher_AR_gtCTX      identical, except the teacher's cache is prefilled
                        AND committed with GT content (commit="gt")

Optional extra rows (--extra_gen): the teacher's OWN 20-step sampler
generating the chunk from noise on each of those two contexts.

The seed chunks are GT in every row (they are the shared real context), so
the first 3 chunks are identical by construction — divergence starts at the
first band chunk, which is the point.

Everything is read-only w.r.t. training code.
"""
import argparse
import importlib.util
import json
import os
import sys

os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")

import numpy as np  # noqa: E402
import torch  # noqa: E402

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
for _p in (ARR, f"{ARR}/action-forcing"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The edge2 probe owns the serving primitives; import them by path (its
# filename starts with a dot, so it is not a normal module name).
_spec = importlib.util.spec_from_file_location(
    "probe_ar_edge2", f"{ARR}/utils/.probe_ar_edge2.py")
E2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E2)

DEFAULT_TEACHER = f"{ARR}/logs/v14e_pca8_raw/causal_lora_step0005000.pt"
DEFAULT_MANIFEST = f"{ARR}/analysis/.dmd_weunz_manifest_min69.pt"
DEFAULT_POOL = f"{ARR}/paper_assets/v14e_train_windows_10k.json"
OUTDIR = f"{ARR}/analysis/ar_review"

TILE_W, TILE_H = 416, 240


def _label(img, text, sub=""):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, 7 * max(len(text), len(sub)) + 10, 30 if sub else 16],
                 fill=(0, 0, 0))
    dr.text((4, 2), text, fill=(255, 255, 0))
    if sub:
        dr.text((4, 16), sub, fill=(0, 255, 255))
    return np.asarray(im)


@torch.no_grad()
def _decode_chunked(vae, lat, device, tchunk):
    outs = []
    F = int(lat.shape[1])
    for s in range(0, F, tchunk):
        piece = lat[:, s:s + tchunk].to(device, torch.float32)
        px = vae.decode_to_pixel(piece, seed_first=True)
        v = (0.5 * (px.float() + 1.0)).clamp(0, 1)
        arr = (v[0].cpu().numpy() * 255).astype(np.uint8)
        if arr.shape[-1] != 3:
            arr = arr.transpose(0, 2, 3, 1)
        outs.append(arr)
        del px, v
        torch.cuda.empty_cache()
    return np.concatenate(outs, 0)


def decode(vae, lat, device, tchunk=9):
    """[1,F,C,H,W] latents -> uint8 [F_pix,H,W,3].

    Whole-clip decode first (no seams). Falls back to chunk-by-chunk on
    OOM; the fallback's seams land on npb=3 chunk boundaries and are
    IDENTICAL across every row, so the comparison stays fair either way.
    """
    try:
        return _decode_chunked(vae, lat, device, int(lat.shape[1]))
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print("[vid] whole-clip VAE decode OOM -> chunked", flush=True)
        return _decode_chunked(vae, lat, device, tchunk)


def resize_all(frames):
    from PIL import Image
    return [np.asarray(Image.fromarray(f).resize((TILE_W, TILE_H)))
            for f in frames]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student_ckpt", required=True)
    ap.add_argument("--teacher_ckpt", default=DEFAULT_TEACHER)
    ap.add_argument("--config",
                    default=f"{ARR}/configs/action_ode_distill_F.yaml")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--pool", default=DEFAULT_POOL)
    ap.add_argument("--windows", type=int, default=2)
    ap.add_argument("--chunks", type=int, default=6)
    ap.add_argument("--seed_chunks", type=int, default=3)
    ap.add_argument("--npb", type=int, default=3)
    ap.add_argument("--rungs", default="1000,625,357.142857,208.333333")
    ap.add_argument("--render_t", default="1000,625,208.333333",
                    help="rungs to render teacher-denoised rows at")
    ap.add_argument("--extra_gen", type=int, default=1,
                    help="also render the teacher's OWN 20-step sampler rows")
    ap.add_argument("--eval_steps", type=int, default=20)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--outdir", default=OUTDIR)
    args = ap.parse_args()

    import imageio.v2 as imageio
    from omegaconf import OmegaConf
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(args.outdir, exist_ok=True)

    cfg = OmegaConf.load(args.config)
    OmegaConf.set_struct(cfg, False)
    cfg.generator_ckpt = args.teacher_ckpt
    dtype = torch.bfloat16

    from utils.infinity_rope import install as _install_ir
    _install_ir(None)
    print("[vid] infinity-RoPE INSTALLED", flush=True)

    print("[vid] building TEACHER...", flush=True)
    tea = E2.build_ode(cfg, device)
    raw = torch.load(args.student_ckpt, map_location="cpu", weights_only=False)
    print(f"[vid] building STUDENT (step={raw.get('step', -1)})...", flush=True)
    stu = E2.build_ode(cfg, device, student_sd=raw)
    del raw

    from utils.wan_wrapper import WanVAEWrapper
    vae = WanVAEWrapper().to(device).eval()

    step_list = [float(x) for x in args.rungs.split(",") if x.strip()]
    render_ts = [float(x) for x in args.render_t.split(",") if x.strip()]
    print(f"[vid] rungs={step_list} render_t={render_ts}", flush=True)

    npb = args.npb
    seed_f = args.seed_chunks * npb
    tot_f = seed_f + args.chunks * npb
    print("[vid] loading manifest (21GB, minutes)...", flush=True)
    man = torch.load(args.manifest, map_location="cpu", weights_only=False)
    rides = man["rides"] if isinstance(man, dict) else man
    pe_by_zarr = {r["zarr_path"]: r["prompt_embeds"] for r in rides}
    nlat_by_zarr = {r["zarr_path"]: int(r["n_latent_frames"])
                    for r in rides if "n_latent_frames" in r}

    pool = json.load(open(args.pool))
    if isinstance(pool, dict):
        pool = pool["windows"]
    wins = []
    for w in pool:
        zp = w["zarr_path"]
        off = int(w.get("offset", w.get("start")))
        if zp not in pe_by_zarr or zp not in nlat_by_zarr:
            continue
        if off + tot_f > nlat_by_zarr[zp]:
            continue
        wins.append((zp, off))
        if len(wins) >= args.windows:
            break

    from utils.zarr_dataset import ZarrRideDataset
    zarrs = {zp for zp, _ in wins}
    z_ds = ZarrRideDataset.from_manifest(
        rides_data=[
            {"zarr_path": r["zarr_path"], "prompt_embeds": r["prompt_embeds"],
             "attrs": r.get("attrs", {}),
             "n_latent_frames": int(r["n_latent_frames"])}
            for r in rides
            if r["zarr_path"] in zarrs and "n_latent_frames" in r
        ],
        motion_root=str(cfg.get("motion_root",
                                "/projects/u6ex/fbots/frodobots_motion")),
        ss_vae_checkpoint=str(cfg.ss_vae_checkpoint),
        device="cpu", ss_vae_device=str(device),
    )
    del man, rides
    action_dims = list(cfg.get("action_dims", [0, 1]))
    fsl = 1560 + int(getattr(E2._base_dit(tea.generator),
                             "action_tokens_per_frame", 0))
    sched = stu.scheduler
    mae_log = []

    for wi, (zp, off) in enumerate(wins):
        gt = ZarrRideDataset.load_latent_chunk(
            zp, off, off + tot_f).unsqueeze(0).to(device, torch.float32)
        pe = pe_by_zarr[zp].unsqueeze(0).to(device, dtype)
        zw = z_ds.encode_z_actions_window(zp, nlat_by_zarr[zp], off,
                                          off + tot_f)
        z = zw[:, action_dims].float().to(device, dtype).view(1, tot_f, -1)
        z = z[..., :2]
        sh = (1, npb, gt.shape[2], gt.shape[3], gt.shape[4])
        tag = f"{os.path.basename(zp).replace('.zarr', '')}_o{off}"
        print(f"\n[vid] === window {wi} {tag} ===", flush=True)

        def mk(name, ode):
            return E2.Served(name, ode, pe, z, fsl, npb, tot_f, dtype,
                             device, sh)

        # THREE caches only. The 20-step generative rows run on the SAME
        # tea_stu / tea_gt caches: every forward at frame f0 overwrites the
        # same cache slots (is_recompute), and the t=0 commit that ends the
        # chunk rewrites them again, so the sampler leaves no residue — and
        # the history each row sees is exactly its own context policy.
        # (5 separate caches x 27 frames would peak ~86 GiB.)
        srv = {"stu": mk("stu", stu), "tea_stu": mk("tea_stu", tea),
               "tea_gt": mk("tea_gt", tea)}

        for f0 in range(0, seed_f, npb):
            for s in srv.values():
                s.commit(gt[:, f0:f0 + npb], f0)

        # accumulators (seed chunks are shared GT in every row)
        rows = {"student_AR": [gt[:, :seed_f]]}
        for t in render_ts:
            rows[f"teacherAR_stuCTX_t{int(t)}"] = [gt[:, :seed_f]]
            rows[f"teacherAR_gtCTX_t{int(t)}"] = [gt[:, :seed_f]]
        if args.extra_gen:
            rows["teacherGEN20_stuCTX"] = [gt[:, :seed_f]]
            rows["teacherGEN20_gtCTX"] = [gt[:, :seed_f]]

        for k in range(args.chunks):
            f0 = seed_f + k * npb
            gt_c = gt[:, f0:f0 + npb]
            sd0 = args.seed + 1000 * wi + k
            g = torch.Generator(device=device).manual_seed(sd0)
            stu_roll = E2.student_roll(srv["stu"], f0, step_list, g)
            rows["student_AR"].append(stu_roll)
            rec = {"window": tag, "chunk": k,
                   "mae_student_AR": E2.mae(stu_roll, gt_c)}

            for t in render_ts:
                xs = E2.noise_to(sched, stu_roll, t, sd0 + 7 * int(t), device)
                a = srv["tea_stu"].fwd(xs, f0, t)[1].float()
                b = srv["tea_gt"].fwd(xs, f0, t)[1].float()
                rows[f"teacherAR_stuCTX_t{int(t)}"].append(a)
                rows[f"teacherAR_gtCTX_t{int(t)}"].append(b)
                rec[f"mae_tea_stuCTX_t{int(t)}"] = E2.mae(a, gt_c)
                rec[f"mae_tea_gtCTX_t{int(t)}"] = E2.mae(b, gt_c)

            if args.extra_gen:
                g2 = torch.Generator(device=device).manual_seed(sd0)
                ga = E2.teacher_gen(srv["tea_stu"], f0, args.eval_steps, g2)
                g2 = torch.Generator(device=device).manual_seed(sd0)
                gb = E2.teacher_gen(srv["tea_gt"], f0, args.eval_steps, g2)
                rows["teacherGEN20_stuCTX"].append(ga)
                rows["teacherGEN20_gtCTX"].append(gb)
                rec["mae_teaGEN_stuCTX"] = E2.mae(ga, gt_c)
                rec["mae_teaGEN_gtCTX"] = E2.mae(gb, gt_c)

            mae_log.append(rec)
            print(f"  k={k} " + "  ".join(
                f"{kk.replace('mae_', '')}={vv:.4f}"
                for kk, vv in rec.items() if kk.startswith("mae_")),
                flush=True)

            # commits: the STUDENT's own chunk into the student-context
            # caches, GT into the gt-context ones.
            srv["stu"].commit(stu_roll, f0)
            srv["tea_stu"].commit(stu_roll, f0)
            srv["tea_gt"].commit(gt_c, f0)

        del srv
        torch.cuda.empty_cache()

        # ---- decode -------------------------------------------------
        seqs = {"GT": gt}
        for nm, parts in rows.items():
            seqs[nm] = torch.cat(parts, dim=1)
        dec = {}
        for nm, lat in seqs.items():
            dec[nm] = resize_all(decode(vae, lat, device))
            print(f"[vid] decoded {nm}: {len(dec[nm])} pixel frames",
                  flush=True)

        n_pix = min(len(v) for v in dec.values())
        seed_pix = int(round(n_pix * seed_f / float(tot_f)))

        # ---- per-rung 2x2 grid: GT | student | tea|stuCTX | tea|gtCTX
        for t in render_ts:
            order = ["GT", "student_AR",
                     f"teacherAR_stuCTX_t{int(t)}",
                     f"teacherAR_gtCTX_t{int(t)}"]
            names = ["GT (real)", "STUDENT AR roll",
                     f"TEACHER AR | student ctx  t={int(t)}",
                     f"TEACHER AR | GT ctx  t={int(t)}"]
            out = (f"{args.outdir}/ar_grid_{tag}_t{int(t)}_"
                   f"{args.chunks}chunks_{args.fps}fps.mp4")
            w = imageio.get_writer(out, fps=args.fps, quality=8,
                                   macro_block_size=16)
            for i in range(n_pix):
                sub = "SEED (real, shared)" if i < seed_pix else \
                    f"band f{i - seed_pix}"
                tiles = [_label(dec[o][i].copy(), nm,
                                sub if o != "GT" else "")
                         for o, nm in zip(order, names)]
                w.append_data(np.vstack([np.hstack(tiles[:2]),
                                         np.hstack(tiles[2:])]))
            w.close()
            print(f"[vid] WROTE {out}", flush=True)

        # ---- generative 2x2 grid (teacher's own 20-step sampler) -----
        if args.extra_gen:
            order = ["GT", "student_AR", "teacherGEN20_stuCTX",
                     "teacherGEN20_gtCTX"]
            names = ["GT (real)", "STUDENT AR roll (4 rungs)",
                     "TEACHER 20-step AR | student ctx",
                     "TEACHER 20-step AR | GT ctx"]
            out = (f"{args.outdir}/ar_grid_{tag}_teacherGEN20_"
                   f"{args.chunks}chunks_{args.fps}fps.mp4")
            w = imageio.get_writer(out, fps=args.fps, quality=8,
                                   macro_block_size=16)
            for i in range(n_pix):
                sub = "SEED (real, shared)" if i < seed_pix else \
                    f"band f{i - seed_pix}"
                tiles = [_label(dec[o][i].copy(), nm,
                                sub if o != "GT" else "")
                         for o, nm in zip(order, names)]
                w.append_data(np.vstack([np.hstack(tiles[:2]),
                                         np.hstack(tiles[2:])]))
            w.close()
            print(f"[vid] WROTE {out}", flush=True)

        # ---- singles (for anyone who wants one row full-res) ---------
        for nm, frames in dec.items():
            out = f"{args.outdir}/single_{tag}_{nm}_{args.fps}fps.mp4"
            imageio.mimsave(out, frames, fps=args.fps, quality=8,
                            macro_block_size=16)
        print(f"[vid] wrote {len(dec)} single-row mp4s for {tag}", flush=True)

    with open(f"{args.outdir}/ar_review_mae.json", "w") as f:
        json.dump(mae_log, f, indent=1)
    print(f"[vid] wrote {args.outdir}/ar_review_mae.json", flush=True)


if __name__ == "__main__":
    main()
