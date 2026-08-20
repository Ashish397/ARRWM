"""AR-teacher edge probe, v2 — falsification suite.

Extends utils/.probe_ar_teacher_edge.py (whose serving primitives this
file copies verbatim) with the three falsification axes and the GT-context
A/B:

  suite=main
    * teacher served on the STUDENT's context (commit="student")   [B-a/B-c]
    * teacher served on GT context           (commit="gt")         [B-b]
    * teacher served on its OWN rolled context (its own sampler)   [A-2]
    * a full t-sweep, with BOTH the student's roll and GT as the
      clean reference that gets noised to t                        [A-3]

  suite=control
    * a deliberately DEGRADED student (weight noise) — the probe MUST
      show a large edge here or the probe is broken                [A-1]
    * teacher-as-student null control — scorer ratio must be exactly
      1.000 (identical weights AND identical cache content)        [A-1]

NOTE ON B-c: the probe (like the trainer's _build_42f_scoring_inputs)
ALREADY prefills the context chunks with GT/real content. So "GT prefill
+ student band commits" IS variant (a). (a) and (b) are therefore
identical at band chunk k=0 and can only diverge at k>=1 — which is a
built-in consistency check, printed as such.

Everything here is read-only w.r.t. training code.
"""
import argparse
import json
import os
import sys

os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")

import torch  # noqa: E402

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
for _p in (ARR, f"{ARR}/action-forcing"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_TEACHER = f"{ARR}/logs/v14e_pca8_raw/causal_lora_step0005000.pt"
DEFAULT_MANIFEST = f"{ARR}/analysis/.dmd_weunz_manifest_min69.pt"
DEFAULT_POOL = f"{ARR}/paper_assets/v14e_train_windows_10k.json"


# ---------------------------------------------------------------- serving
def _base_dit(wrapper):
    m = wrapper.model
    return m.get_base_model() if hasattr(m, "get_base_model") else m


def _set_window(wrapper, local_attn_f, fsl, has_tokens):
    base = _base_dit(wrapper)
    base.block_mask = None
    target_max = local_attn_f * fsl
    base.local_attn_size = local_attn_f
    if hasattr(base, "max_attention_size"):
        base.max_attention_size = target_max
    for _, m in base.named_modules():
        if hasattr(m, "local_attn_size"):
            m.local_attn_size = local_attn_f
        if hasattr(m, "max_attention_size"):
            m.max_attention_size = target_max
        if hasattr(m, "action_tokens_per_frame"):
            m.cached_rope_action_aware = has_tokens
    return base


def _make_caches(base, kv_frames, fsl, dtype, device):
    n_blocks = len(base.blocks)
    blk0 = base.blocks[0]
    n_heads = int(getattr(blk0.self_attn, "num_heads",
                          getattr(base, "num_heads", 12)))
    head_dim = int(getattr(blk0.self_attn, "head_dim", base.dim // n_heads))
    text_len = int(getattr(base, "text_len", 512))
    kv_size = kv_frames * fsl
    kv, xa = [], []
    for _ in range(n_blocks):
        kv.append({
            "k": torch.zeros([1, kv_size, n_heads, head_dim],
                             dtype=dtype, device=device),
            "v": torch.zeros([1, kv_size, n_heads, head_dim],
                             dtype=dtype, device=device),
            "global_end_index": torch.tensor([0], dtype=torch.long,
                                             device=device),
            "local_end_index": torch.tensor([0], dtype=torch.long,
                                            device=device),
        })
        xa.append({
            "k": torch.zeros([1, text_len, n_heads, head_dim],
                             dtype=dtype, device=device),
            "v": torch.zeros([1, text_len, n_heads, head_dim],
                             dtype=dtype, device=device),
            "is_init": False,
        })
    return kv, xa


class Served:
    """One model + its OWN KV cache, driven frame-position-absolutely.

    Several Served objects may share one ODERegression (the weights are
    frozen and every cache is passed in explicitly), which is how the
    teacher gets three independent context policies at once.
    """

    def __init__(self, name, ode, pe, z, fsl, npb, kv_frames, dtype, device,
                 shape_hint):
        self.name = name
        self.ode = ode
        self.wrapper = ode.generator
        self.pe = pe
        self.z = z
        self.fsl = fsl
        self.npb = npb
        self.dtype = dtype
        self.device = device
        self.shape_hint = shape_hint
        has_tokens = ode.action_token_projection is not None
        self.base = _set_window(self.wrapper, 21, fsl, has_tokens)
        self.wrapper.adjust_seq_len_for_action_tokens(
            num_frames=npb, action_per_frame=1 if has_tokens else 0,
        )
        self.kv, self.xa = _make_caches(
            self.base, kv_frames, fsl, dtype, device,
        )

    def cond(self, f0):
        c = {"prompt_embeds": self.pe}
        zb = self.z[:, f0:f0 + self.npb]
        if self.ode.action_projection is not None:
            c["_action_modulation"] = self.ode.action_projection(
                zb, num_frames=self.npb,
            )
        if self.ode.action_token_projection is not None:
            c["_action_tokens"] = self.ode.action_token_projection(zb)
        return c

    @torch.no_grad()
    def fwd(self, lat, f0, t_val):
        ts = torch.full([1, self.npb], float(t_val),
                        device=self.device, dtype=torch.float32)
        with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
            out = self.wrapper(
                lat.to(self.dtype), self.cond(f0), ts,
                kv_cache=self.kv, crossattn_cache=self.xa,
                current_start=f0 * self.fsl,
            )
        if isinstance(out, (tuple, list)):
            return out[0], (out[1] if len(out) > 1 else out[0])
        return out, out

    @torch.no_grad()
    def commit(self, x0, f0):
        self.fwd(x0, f0, 0.0)


@torch.no_grad()
def student_roll(srv, f0, step_list, gen):
    """Distilled-student inference: rung ladder with re-noise."""
    lat = torch.randn(srv.shape_hint, device=srv.device, dtype=torch.float32,
                      generator=gen)
    x0 = None
    for i, t in enumerate(step_list):
        _, x0 = srv.fwd(lat, f0, float(t))
        x0 = x0.float()
        if i + 1 < len(step_list):
            nt = torch.full([srv.npb], float(step_list[i + 1]),
                            device=srv.device)
            lat = srv.ode.scheduler.add_noise(
                x0.flatten(0, 1), torch.randn_like(x0.flatten(0, 1)), nt,
            ).unflatten(0, x0.shape[:2])
    return x0


@torch.no_grad()
def teacher_gen(srv, f0, eval_steps, gen):
    """The teacher's OWN sampler: shift-5 flow-match chain."""
    from utils.scheduler import FlowMatchScheduler
    sched = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    sched.set_timesteps(num_inference_steps=eval_steps, denoising_strength=1.0)
    sched.sigmas = sched.sigmas.to(srv.device)
    lat = torch.randn(srv.shape_hint, device=srv.device, dtype=torch.float32,
                      generator=gen)
    for t in sched.timesteps:
        ts = t * torch.ones([1, srv.npb], device=srv.device,
                            dtype=torch.float32)
        flow, _ = srv.fwd(lat, f0, float(t))
        lat = sched.step(
            flow.float().flatten(0, 1), ts.flatten(0, 1),
            lat.flatten(0, 1),
        ).unflatten(0, flow.shape[:2])
    return lat.float()


def mae(a, b):
    return float((a.float() - b.float()).abs().mean().item())


def noise_to(sched, ref, t, seed, device):
    tv = torch.full([ref.shape[1]], float(t), device=device)
    g = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(ref.shape, device=device, dtype=torch.float32,
                        generator=g)
    return sched.add_noise(
        ref.float().flatten(0, 1), noise.flatten(0, 1), tv,
    ).unflatten(0, ref.shape[:2])


# ------------------------------------------------------------------ build
def build_ode(cfg, device, student_sd=None, degrade=0.0, degrade_seed=0):
    from af_model.ode_regression import ODERegression
    m = ODERegression(cfg, device=device).eval()
    m.use_motion_pipeline = False
    if student_sd is not None:
        m.generator.model.load_state_dict(student_sd["generator"], strict=True)
        for k, mod in (("action_projection", m.action_projection),
                       ("action_token_projection", m.action_token_projection)):
            if mod is not None and k in student_sd:
                mod.load_state_dict(student_sd[k])
    if degrade > 0.0:
        g = torch.Generator(device="cpu").manual_seed(degrade_seed)
        n_pert = 0
        with torch.no_grad():
            for p in m.generator.model.parameters():
                if p.dtype.is_floating_point and p.numel() > 1:
                    s = float(p.detach().float().std().item())
                    if s > 0:
                        p.add_((torch.randn(p.shape, generator=g,
                                            dtype=torch.float32) * (degrade * s)
                                ).to(p.device, p.dtype))
                        n_pert += 1
        print(f"[probe] DEGRADED {n_pert} tensors with sigma={degrade}"
              " x per-tensor std", flush=True)
    return m


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["main", "control"], default="main")
    ap.add_argument("--student_ckpt", required=True)
    ap.add_argument("--teacher_ckpt", default=DEFAULT_TEACHER)
    ap.add_argument("--config",
                    default=f"{ARR}/configs/action_ode_distill_F.yaml")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--pool", default=DEFAULT_POOL)
    ap.add_argument("--windows", type=int, default=3)
    ap.add_argument("--chunks", type=int, default=3)
    ap.add_argument("--seed_chunks", type=int, default=3)
    ap.add_argument("--npb", type=int, default=3)
    ap.add_argument("--probe_t",
                    default="1000,980,800,625,500,357,208,100,50")
    ap.add_argument("--rungs", default="",
                    help="student rung ladder; '' = the ODE config's. The "
                         "DMD arms override it to "
                         "1000,625,357.142857,208.333333 — pass that to be "
                         "like-for-like with sbatch/train_dmd10k_stat.sbatch.")
    ap.add_argument("--eval_steps", type=int, default=20)
    ap.add_argument("--degrade", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out_json", default="")
    ap.add_argument("--infinity_rope", type=int, default=1)
    args = ap.parse_args()

    from omegaconf import OmegaConf
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True

    cfg = OmegaConf.load(args.config)
    OmegaConf.set_struct(cfg, False)
    cfg.generator_ckpt = args.teacher_ckpt
    dtype = torch.bfloat16

    if args.infinity_rope:
        from utils.infinity_rope import install as _install_ir
        _install_ir(None)
        print("[probe] infinity-RoPE patch INSTALLED", flush=True)

    print("[probe] building TEACHER...", flush=True)
    tea = build_ode(cfg, device)

    raw = torch.load(args.student_ckpt, map_location="cpu", weights_only=False)
    if "generator" not in raw:
        raise SystemExit(f"{args.student_ckpt} has no 'generator' key")
    print(f"[probe] building STUDENT (step={raw.get('step', -1)})...",
          flush=True)
    stu = build_ode(cfg, device, student_sd=raw)

    deg = None
    if args.suite == "control":
        print("[probe] building DEGRADED student...", flush=True)
        deg = build_ode(cfg, device, student_sd=raw, degrade=args.degrade,
                        degrade_seed=7)
    del raw

    if args.rungs.strip():
        step_list = [float(x) for x in args.rungs.split(",") if x.strip()]
    else:
        step_list = [float(x) for x in stu.denoising_step_list.tolist()]
    probe_ts = [float(x) for x in args.probe_t.split(",") if x.strip()]
    print(f"[probe] rungs={step_list}", flush=True)
    print(f"[probe] probe_t={probe_ts}", flush=True)

    # ---- data ---------------------------------------------------------
    npb = args.npb
    seed_f = args.seed_chunks * npb
    tot_f = seed_f + args.chunks * npb
    print(f"[probe] loading manifest (21GB, minutes)...", flush=True)
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
    if not wins:
        raise SystemExit("no usable windows")

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

    fsl = 1560 + int(getattr(_base_dit(tea.generator),
                             "action_tokens_per_frame", 0))
    kv_frames = tot_f          # exactly what gets written; never rolls
    sched = stu.scheduler
    rows = []

    for wi, (zp, off) in enumerate(wins):
        gt = ZarrRideDataset.load_latent_chunk(
            zp, off, off + tot_f,
        ).unsqueeze(0).to(device, torch.float32)
        pe = pe_by_zarr[zp].unsqueeze(0).to(device, dtype)
        zw = z_ds.encode_z_actions_window(zp, nlat_by_zarr[zp], off,
                                          off + tot_f)
        z = zw[:, action_dims].float().to(device, dtype).view(1, tot_f, -1)
        z = z[..., :2]
        sh = (1, npb, gt.shape[2], gt.shape[3], gt.shape[4])

        def mk(name, ode):
            return Served(name, ode, pe, z, fsl, npb, kv_frames, dtype,
                          device, sh)

        if args.suite == "main":
            srv = {
                "stu":       mk("stu", stu),        # student, own commits
                "tea_stu":   mk("tea_stu", tea),    # teacher, STUDENT commits
                "tea_gt":    mk("tea_gt", tea),     # teacher, GT commits
                "tea_own":   mk("tea_own", tea),    # teacher, OWN-roll commits
            }
        else:
            srv = {
                "stu":       mk("stu", stu),
                "tea_stu":   mk("tea_stu", tea),
                "deg":       mk("deg", deg),
                "tea_deg":   mk("tea_deg", tea),
                "null":      mk("null", tea),       # teacher AS student
                "tea_null":  mk("tea_null", tea),
            }
        print(f"[probe] {len(srv)} caches x {kv_frames}f  "
              f"({torch.cuda.memory_allocated()/2**30:.1f} GiB alloc)",
              flush=True)

        # --- prefill: the SAME real/GT seed at t=0 into EVERY cache ----
        for f0 in range(0, seed_f, npb):
            for s in srv.values():
                s.commit(gt[:, f0:f0 + npb], f0)

        tag = f"{os.path.basename(zp).replace('.zarr', '')}_o{off}"
        print(f"\n[probe] === window {wi} {tag} ===", flush=True)

        for k in range(args.chunks):
            f0 = seed_f + k * npb
            gt_c = gt[:, f0:f0 + npb]
            sd0 = args.seed + 1000 * wi + k

            g = torch.Generator(device=device).manual_seed(sd0)
            stu_roll = student_roll(srv["stu"], f0, step_list, g)
            m_roll = mae(stu_roll, gt_c)
            base = {"window": tag, "chunk": k, "mae_stu_roll": m_roll}

            if args.suite == "main":
                # --- generative comparisons (teacher's own sampler) ----
                g2 = torch.Generator(device=device).manual_seed(sd0)
                tgen_stu = teacher_gen(srv["tea_stu"], f0, args.eval_steps, g2)
                g2 = torch.Generator(device=device).manual_seed(sd0)
                tgen_gt = teacher_gen(srv["tea_gt"], f0, args.eval_steps, g2)
                g2 = torch.Generator(device=device).manual_seed(sd0)
                tea_own_roll = teacher_gen(srv["tea_own"], f0,
                                           args.eval_steps, g2)
                base.update({
                    "mae_tea_gen_stuctx": mae(tgen_stu, gt_c),
                    "mae_tea_gen_gtctx": mae(tgen_gt, gt_c),
                    "mae_tea_own_roll": mae(tea_own_roll, gt_c),
                })
                print(f"  [k={k}] ROLLS  stu_roll={m_roll:.4f}  "
                      f"tea_gen|stuctx={base['mae_tea_gen_stuctx']:.4f}  "
                      f"tea_gen|gtctx={base['mae_tea_gen_gtctx']:.4f}  "
                      f"tea_own_roll={base['mae_tea_own_roll']:.4f}",
                      flush=True)

                for t in probe_ts:
                    r = dict(base); r["t"] = t
                    xs = noise_to(sched, stu_roll, t, sd0 + 7 * int(t), device)
                    xg = noise_to(sched, gt_c, t, sd0 + 7 * int(t), device)
                    xo = noise_to(sched, tea_own_roll, t, sd0 + 7 * int(t),
                                  device)
                    r["m_tea_stuctx_S"] = mae(
                        srv["tea_stu"].fwd(xs, f0, t)[1], gt_c)
                    r["m_tea_gtctx_S"] = mae(
                        srv["tea_gt"].fwd(xs, f0, t)[1], gt_c)
                    r["m_stu_S"] = mae(srv["stu"].fwd(xs, f0, t)[1], gt_c)
                    r["m_tea_stuctx_G"] = mae(
                        srv["tea_stu"].fwd(xg, f0, t)[1], gt_c)
                    r["m_tea_gtctx_G"] = mae(
                        srv["tea_gt"].fwd(xg, f0, t)[1], gt_c)
                    r["m_stu_G"] = mae(srv["stu"].fwd(xg, f0, t)[1], gt_c)
                    r["m_tea_ownctx_O"] = mae(
                        srv["tea_own"].fwd(xo, f0, t)[1], gt_c)
                    r["m_tea_ownctx_G"] = mae(
                        srv["tea_own"].fwd(xg, f0, t)[1], gt_c)
                    r["edge_a"] = m_roll / max(r["m_tea_stuctx_S"], 1e-8)
                    r["edge_b"] = m_roll / max(r["m_tea_gtctx_S"], 1e-8)
                    r["scorer_a"] = r["m_stu_S"] / max(r["m_tea_stuctx_S"],
                                                       1e-8)
                    r["scorer_b"] = r["m_stu_S"] / max(r["m_tea_gtctx_S"],
                                                       1e-8)
                    r["scorer_G"] = r["m_stu_G"] / max(r["m_tea_gtctx_G"],
                                                       1e-8)
                    rows.append(r)
                    print(f"    t={t:>6.0f} | S: tea|stuctx="
                          f"{r['m_tea_stuctx_S']:.4f} tea|gtctx="
                          f"{r['m_tea_gtctx_S']:.4f} stu={r['m_stu_S']:.4f}"
                          f" || G: tea|stuctx={r['m_tea_stuctx_G']:.4f} "
                          f"tea|gtctx={r['m_tea_gtctx_G']:.4f} "
                          f"stu={r['m_stu_G']:.4f} || OWN: "
                          f"tea|ownctx(O)={r['m_tea_ownctx_O']:.4f} "
                          f"tea|ownctx(G)={r['m_tea_ownctx_G']:.4f}"
                          f" || EDGE_a={r['edge_a']:.3f} "
                          f"EDGE_b={r['edge_b']:.3f}", flush=True)

                for nm, x in (("stu", stu_roll), ("tea_stu", stu_roll),
                              ("tea_gt", gt_c), ("tea_own", tea_own_roll)):
                    srv[nm].commit(x, f0)

            else:  # control
                g = torch.Generator(device=device).manual_seed(sd0)
                deg_roll = student_roll(srv["deg"], f0, step_list, g)
                g = torch.Generator(device=device).manual_seed(sd0)
                null_roll = student_roll(srv["null"], f0, step_list, g)
                base.update({
                    "mae_deg_roll": mae(deg_roll, gt_c),
                    "mae_null_roll": mae(null_roll, gt_c),
                })
                print(f"  [k={k}] ROLLS  stu={m_roll:.4f}  "
                      f"deg={base['mae_deg_roll']:.4f}  "
                      f"null(teacher-as-student)={base['mae_null_roll']:.4f}",
                      flush=True)
                for t in probe_ts:
                    r = dict(base); r["t"] = t
                    sd = sd0 + 7 * int(t)
                    xs = noise_to(sched, stu_roll, t, sd, device)
                    xd = noise_to(sched, deg_roll, t, sd, device)
                    xn = noise_to(sched, null_roll, t, sd, device)
                    r["m_tea_S"] = mae(srv["tea_stu"].fwd(xs, f0, t)[1], gt_c)
                    r["m_stu_S"] = mae(srv["stu"].fwd(xs, f0, t)[1], gt_c)
                    r["m_tea_D"] = mae(srv["tea_deg"].fwd(xd, f0, t)[1], gt_c)
                    r["m_deg_D"] = mae(srv["deg"].fwd(xd, f0, t)[1], gt_c)
                    r["m_tea_N"] = mae(srv["tea_null"].fwd(xn, f0, t)[1], gt_c)
                    r["m_null_N"] = mae(srv["null"].fwd(xn, f0, t)[1], gt_c)
                    # DISCRIMINATIVE column: the SAME GT-noised input into
                    # each model (on its own cache). If EDGE cannot separate
                    # stu from deg but THIS can, the EDGE metric — not the
                    # teacher — is what is null.
                    xg = noise_to(sched, gt_c, t, sd, device)
                    r["g_stu"] = mae(srv["stu"].fwd(xg, f0, t)[1], gt_c)
                    r["g_deg"] = mae(srv["deg"].fwd(xg, f0, t)[1], gt_c)
                    r["g_tea"] = mae(srv["tea_stu"].fwd(xg, f0, t)[1], gt_c)
                    r["edge_stu"] = m_roll / max(r["m_tea_S"], 1e-8)
                    r["edge_deg"] = base["mae_deg_roll"] / max(r["m_tea_D"],
                                                               1e-8)
                    r["edge_null"] = base["mae_null_roll"] / max(r["m_tea_N"],
                                                                 1e-8)
                    r["scorer_stu"] = r["m_stu_S"] / max(r["m_tea_S"], 1e-8)
                    r["scorer_deg"] = r["m_deg_D"] / max(r["m_tea_D"], 1e-8)
                    r["scorer_null"] = r["m_null_N"] / max(r["m_tea_N"], 1e-8)
                    rows.append(r)
                    print(f"    t={t:>6.0f} | STU tea={r['m_tea_S']:.4f} "
                          f"stu={r['m_stu_S']:.4f} edge={r['edge_stu']:.3f} "
                          f"scorer={r['scorer_stu']:.3f} | DEG "
                          f"tea={r['m_tea_D']:.4f} deg={r['m_deg_D']:.4f} "
                          f"edge={r['edge_deg']:.3f} "
                          f"scorer={r['scorer_deg']:.3f} | NULL "
                          f"tea={r['m_tea_N']:.4f} null={r['m_null_N']:.4f} "
                          f"edge={r['edge_null']:.3f} "
                          f"scorer={r['scorer_null']:.6f}", flush=True)
                for nm, x in (("stu", stu_roll), ("tea_stu", stu_roll),
                              ("deg", deg_roll), ("tea_deg", deg_roll),
                              ("null", null_roll), ("tea_null", null_roll)):
                    srv[nm].commit(x, f0)

        del srv
        torch.cuda.empty_cache()

    # ---- verdict ------------------------------------------------------
    def _mean(key, pred=lambda r: True):
        v = [r[key] for r in rows
             if pred(r) and key in r and r[key] == r[key]]
        return sum(v) / max(len(v), 1)

    print("\n[probe] ================ VERDICT (suite=%s) ================"
          % args.suite)
    if args.suite == "main":
        print("  per-t (avg over all windows/chunks):")
        print("    t     tea|stuCTX  tea|gtCTX   stu     EDGE_a EDGE_b "
              "| GTref: tea|gtCTX  stu   scorer_G | ownCTX(own) ownCTX(gt)")
        for t in probe_ts:
            sel = (lambda r, _t=t: r["t"] == _t)
            print(f"  {t:>6.0f}  {_mean('m_tea_stuctx_S', sel):9.4f} "
                  f"{_mean('m_tea_gtctx_S', sel):10.4f} "
                  f"{_mean('m_stu_S', sel):8.4f} "
                  f"{_mean('edge_a', sel):6.3f} {_mean('edge_b', sel):6.3f} | "
                  f"{_mean('m_tea_gtctx_G', sel):9.4f} "
                  f"{_mean('m_stu_G', sel):7.4f} "
                  f"{_mean('scorer_G', sel):8.3f} | "
                  f"{_mean('m_tea_ownctx_O', sel):10.4f} "
                  f"{_mean('m_tea_ownctx_G', sel):10.4f}")
        print("  per-chunk EDGE (a=student ctx, b=GT ctx); k=0 must MATCH "
              "(identical GT-only context):")
        for k in range(args.chunks):
            sel = (lambda r, _k=k: r["chunk"] == _k)
            print(f"    k={k}  EDGE_a={_mean('edge_a', sel):.3f}  "
                  f"EDGE_b={_mean('edge_b', sel):.3f}  "
                  f"scorer_a={_mean('scorer_a', sel):.3f}  "
                  f"scorer_b={_mean('scorer_b', sel):.3f}")
        print("  ROLL quality (MAE vs GT):")
        print(f"    stu 4-rung roll        = {_mean('mae_stu_roll'):.4f}")
        print(f"    tea 20-step | stu ctx  = "
              f"{_mean('mae_tea_gen_stuctx'):.4f}")
        print(f"    tea 20-step | gt  ctx  = {_mean('mae_tea_gen_gtctx'):.4f}")
        print(f"    tea 20-step | OWN ctx  = {_mean('mae_tea_own_roll'):.4f}")
    else:
        print("    t      EDGE_stu EDGE_deg EDGE_null | scorer_stu "
              "scorer_deg scorer_null")
        for t in probe_ts:
            sel = (lambda r, _t=t: r["t"] == _t)
            print(f"  {t:>6.0f}   {_mean('edge_stu', sel):7.3f} "
                  f"{_mean('edge_deg', sel):8.3f} "
                  f"{_mean('edge_null', sel):9.3f} | "
                  f"{_mean('scorer_stu', sel):10.3f} "
                  f"{_mean('scorer_deg', sel):10.3f} "
                  f"{_mean('scorer_null', sel):11.6f}")
        print("  DISCRIMINATIVE (same GT-noised input, MAE vs GT):")
        print("    t        stu      deg      tea   | deg/stu  stu/tea")
        for t in probe_ts:
            sel = (lambda r, _t=t: r["t"] == _t)
            gs, gd, gt_ = (_mean('g_stu', sel), _mean('g_deg', sel),
                           _mean('g_tea', sel))
            print(f"  {t:>6.0f}  {gs:8.4f} {gd:8.4f} {gt_:8.4f} | "
                  f"{gd / max(gs, 1e-8):7.3f} {gs / max(gt_, 1e-8):8.3f}")
        print(f"  ROLL MAE: stu={_mean('mae_stu_roll'):.4f}  "
              f"deg={_mean('mae_deg_roll'):.4f}  "
              f"null={_mean('mae_null_roll'):.4f}")
        print("  scorer_null MUST be 1.000000 (identical weights AND "
              "identical cache content) -> probe plumbing is sound.")
        print("  scorer_deg MUST be >> 1 -> the probe CAN detect a real "
              "quality gap.")
    print("[probe] ====================================================")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(rows, f, indent=1)
        print(f"[probe] wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
