"""GO/NO-GO probe for the DMD AR-teacher head (dmd_ar_head_weight).

THE QUESTION
------------
The AR head's entire premise is that the frozen v14e teacher, served
AUTOREGRESSIVELY through a KV cache on the STUDENT's own rolled context,
is a better oracle than the student itself. Inside the DMD machinery
that claim is entangled with the critic, the eq.-8 normalizer, the MAE
gate and the 42f window layout. This script answers it OUTSIDE all of
that:

    roll the student N chunks from a REAL seed, exactly as at inference;
    then, from the SAME cache state, have the teacher and the student
    each denoise the SAME noised chunk, and compare both to GT.

If the teacher wins clearly, the AR-head design is sound and only its
supervision layout is in question. If it does not, no supervision
layout rescues it — the head should be dropped.

WHAT IT PRINTS (per chunk, per probe rung)
------------------------------------------
  mae_stu_roll   the student's OWN rolled chunk vs GT
                 (== gen/dmd_mae_gate_m_fake in training)
  mae_tea_score  teacher single-shot x0 from the noised chunk vs GT
                 (== gen/dmd_ar_mae_vs_gt in training)
  mae_stu_score  student single-shot x0 from the SAME noised chunk vs GT
                 (the student's own version of the same measurement)
  mae_tea_gen    teacher's FULL sampler (20-step shift-5 chain) generating
                 the chunk from pure noise on the student's context vs GT
                 — this is the regime the comparison grids are made in
  ratio_edge     mae_stu_roll / mae_tea_score   <- the DMD-relevant edge
  ratio_scorer   mae_stu_score / mae_tea_score  <- scorer-vs-scorer; this
                 is the one that pins to ~1.0 by construction (both
                 denoise the SAME input from the SAME conditional), and
                 it is what gen/dmd_ar_fake_mae_vs_gt is really measuring
  ratio_gen      mae_stu_roll / mae_tea_gen     <- full-sampler AR edge

SERVING CONTRACT (mirrors sbatch/train_dmd10k_stat.sbatch exactly)
------------------------------------------------------------------
  * npb=3, 3 real seed chunks written at t=0 (seed_prefill_mode=real),
    local_attn_size=21, infinity_rope ON.
  * current_start = frame_index * frame_seq_length — NO offset, the
    convention utils/causal_chain_rollout.py::stream_causal_chain uses
    (the teacher's own generation sampler). Under infinity-RoPE the
    offset is inert anyway; this keeps the probe free of that argument.
  * between chunks the STUDENT's own x0 is committed at t=0 into BOTH
    caches — the same commit pipeline/action_forcing_training.py Step
    3.4 performs, and the commit mode the AR head now defaults to.

Both models are built from the SAME ODERegression config: the teacher is
an ODERegression left at its ``generator_ckpt`` (= the v14e LoRA merged
into the base = the teacher), the student is a second ODERegression with
its ``generator`` state-dict overlaid.

Run it via ``utils/.probe_ar_teacher_edge.sh`` (sets PYTHONPATH, the
action encoder and the conda env).
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


# ----------------------------------------------------------------------
# serving primitives (local copies of causal_chain_rollout's, so this
# probe never mutates the training path)
# ----------------------------------------------------------------------
def _base_dit(wrapper):
    m = wrapper.model
    return m.get_base_model() if hasattr(m, "get_base_model") else m


def _set_window(wrapper, local_attn_f, fsl, has_tokens):
    """Pin the attention window / action-aware cached RoPE, like
    stream_causal_chain does. Not restored — this process is a probe."""
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
    """One model + its own KV cache, driven frame-position-absolutely."""

    def __init__(self, name, ode, pe, z, fsl, npb, kv_frames, dtype, device):
        self.name = name
        self.ode = ode
        self.wrapper = ode.generator
        self.pe = pe
        self.z = z
        self.fsl = fsl
        self.npb = npb
        self.dtype = dtype
        self.device = device
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
        """One cached forward at absolute frame f0. Returns (flow, x0)."""
        ts = torch.full([1, self.npb], float(t_val),
                        device=self.device, dtype=torch.float32)
        # The DiT weights live in fp32 (ODERegression only does
        # ``wrapper.to(device)``); every serving path in the codebase
        # runs the forward under autocast — see
        # utils/causal_chain_rollout.py::fwd. Without it conv3d sees a
        # bf16 input against an fp32 bias and raises.
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
        """t=0 same-position overwrite (is_recompute in causal_model)."""
        self.fwd(x0, f0, 0.0)


# ----------------------------------------------------------------------
@torch.no_grad()
def student_roll(srv, f0, step_list, gen):
    """Distilled-student inference: rung ladder with re-noise between
    rungs (pipeline/action_forcing_training.py Step 3.2)."""
    shape = srv.shape_hint
    lat = torch.randn(shape, device=srv.device, dtype=torch.float32,
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
    """The teacher's OWN sampler: 20-step shift-5 flow-match chain,
    exactly utils/causal_chain_rollout.py::stream_causal_chain."""
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


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student_ckpt", required=True,
                    help="action_ode_step*.pt (the DMD arms' ODE init).")
    ap.add_argument("--teacher_ckpt", default=DEFAULT_TEACHER)
    ap.add_argument("--config", default=f"{ARR}/configs/action_ode_distill_F.yaml")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--pool", default=DEFAULT_POOL)
    ap.add_argument("--windows", type=int, default=4,
                    help="how many real contexts to probe")
    ap.add_argument("--chunks", type=int, default=3,
                    help="N student chunks to roll (the AR head's band)")
    ap.add_argument("--seed_chunks", type=int, default=3,
                    help="real seed chunks (=dmd_context_clean_frames/npb)")
    ap.add_argument("--npb", type=int, default=3)
    ap.add_argument("--probe_t", default="1000,625,357,208",
                    help="rungs at which the single-shot comparison runs")
    ap.add_argument("--noise_source", choices=["student", "gt"],
                    default="student",
                    help="what gets noised to probe_t. 'student' == exactly "
                         "what the DMD AR head feeds its scorers.")
    ap.add_argument("--eval_steps", type=int, default=20,
                    help="teacher full-sampler chain length (0 = skip)")
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
        _install_ir(None)   # class-level patch, both models (as in phase-3)
        print("[probe] infinity-RoPE patch INSTALLED", flush=True)

    print("[probe] building TEACHER (v14e LoRA merged into base)...",
          flush=True)
    from af_model.ode_regression import ODERegression
    tea = ODERegression(cfg, device=device).eval()
    tea.use_motion_pipeline = False

    print("[probe] building STUDENT (same base, student sd overlaid)...",
          flush=True)
    stu = ODERegression(cfg, device=device).eval()
    stu.use_motion_pipeline = False
    raw = torch.load(args.student_ckpt, map_location="cpu",
                     weights_only=False)
    if "generator" not in raw:
        raise SystemExit(
            f"{args.student_ckpt} has no 'generator' key "
            f"(keys={list(raw)[:8]}) — expected an action_ode_step*.pt."
        )
    stu.generator.model.load_state_dict(raw["generator"], strict=True)
    for k, mod in (("action_projection", stu.action_projection),
                   ("action_token_projection", stu.action_token_projection)):
        if mod is not None and k in raw:
            mod.load_state_dict(raw[k])
    print(f"[probe] student step={raw.get('step', -1)}", flush=True)

    step_list = [float(x) for x in stu.denoising_step_list.tolist()]
    probe_ts = [float(x) for x in args.probe_t.split(",") if x.strip()]
    print(f"[probe] student rungs={step_list}  probe_t={probe_ts}", flush=True)

    # ---- data ---------------------------------------------------------
    npb = args.npb
    seed_f = args.seed_chunks * npb
    tot_f = seed_f + args.chunks * npb
    print(f"[probe] loading manifest {args.manifest} (large — minutes)...",
          flush=True)
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
        raise SystemExit("no usable windows (pool/manifest disjoint?)")

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
    action_dims = list(cfg.get("action_dims", [0, 1]))

    fsl = 1560 + int(getattr(_base_dit(tea.generator),
                             "action_tokens_per_frame", 0))
    kv_frames = tot_f + 2 * npb
    rows = []

    for wi, (zp, off) in enumerate(wins):
        gt = ZarrRideDataset.load_latent_chunk(
            zp, off, off + tot_f,
        ).unsqueeze(0).to(device, torch.float32)          # [1, tot_f, C,H,W]
        pe = pe_by_zarr[zp].unsqueeze(0).to(device, dtype)
        zw = z_ds.encode_z_actions_window(
            zp, nlat_by_zarr[zp], off, off + tot_f,
        )
        z = zw[:, action_dims].float().to(device, dtype).view(1, tot_f, -1)
        z = z[..., :2]

        s_tea = Served("teacher", tea, pe, z, fsl, npb, kv_frames,
                       dtype, device)
        s_stu = Served("student", stu, pe, z, fsl, npb, kv_frames,
                       dtype, device)
        for s in (s_tea, s_stu):
            s.shape_hint = (1, npb, gt.shape[2], gt.shape[3], gt.shape[4])

        # --- prefill: the SAME real seed at t=0 into BOTH caches -------
        for f0 in range(0, seed_f, npb):
            for s in (s_tea, s_stu):
                s.commit(gt[:, f0:f0 + npb], f0)

        tag = f"{os.path.basename(zp).replace('.zarr','')}_o{off}"
        print(f"\n[probe] === window {wi} {tag} ===", flush=True)

        for k in range(args.chunks):
            f0 = seed_f + k * npb
            gt_c = gt[:, f0:f0 + npb]

            g = torch.Generator(device=device).manual_seed(
                args.seed + 1000 * wi + k)
            stu_roll = student_roll(s_stu, f0, step_list, g)
            m_roll = mae(stu_roll, gt_c)

            m_gen = float("nan")
            if args.eval_steps > 0:
                g2 = torch.Generator(device=device).manual_seed(
                    args.seed + 1000 * wi + k)
                tea_gen_x0 = teacher_gen(s_tea, f0, args.eval_steps, g2)
                m_gen = mae(tea_gen_x0, gt_c)

            ref = stu_roll if args.noise_source == "student" else gt_c
            for t in probe_ts:
                tv = torch.full([npb], float(t), device=device)
                noise = torch.randn(
                    ref.shape, device=device, dtype=torch.float32,
                    generator=torch.Generator(device=device).manual_seed(
                        args.seed + 7 * int(t) + k),
                )
                x_t = stu.scheduler.add_noise(
                    ref.float().flatten(0, 1), noise.flatten(0, 1), tv,
                ).unflatten(0, ref.shape[:2])
                _, tea_x0 = s_tea.fwd(x_t, f0, t)
                _, stu_x0 = s_stu.fwd(x_t, f0, t)
                m_tea = mae(tea_x0, gt_c)
                m_stu = mae(stu_x0, gt_c)
                row = {
                    "window": tag, "chunk": k, "t": t,
                    "mae_stu_roll": m_roll,
                    "mae_tea_score": m_tea,
                    "mae_stu_score": m_stu,
                    "mae_tea_gen": m_gen,
                    "ratio_edge": m_roll / max(m_tea, 1e-8),
                    "ratio_scorer": m_stu / max(m_tea, 1e-8),
                    "ratio_gen": m_roll / max(m_gen, 1e-8),
                }
                rows.append(row)
                print(
                    f"  chunk={k} t={t:>7.1f} | stu_roll={m_roll:.4f} "
                    f"tea_score={m_tea:.4f} stu_score={m_stu:.4f} "
                    f"tea_gen={m_gen:.4f} || EDGE={row['ratio_edge']:.3f} "
                    f"scorer={row['ratio_scorer']:.3f} "
                    f"gen={row['ratio_gen']:.3f}",
                    flush=True,
                )

            # commit the STUDENT's own chunk at t=0 into BOTH caches —
            # the conditional the AR head now trains on.
            for s in (s_tea, s_stu):
                s.commit(stu_roll, f0)

    # ---- verdict ------------------------------------------------------
    def _mean(key, pred=lambda r: True):
        v = [r[key] for r in rows if pred(r) and r[key] == r[key]]
        return sum(v) / max(len(v), 1)

    print("\n[probe] ===================== VERDICT =====================")
    for t in probe_ts:
        sel = (lambda r, _t=t: r["t"] == _t)
        print(f"  t={t:>7.1f}  EDGE(stu_roll/tea_score)="
              f"{_mean('ratio_edge', sel):.3f}   "
              f"scorer(stu_score/tea_score)={_mean('ratio_scorer', sel):.3f}")
    print(f"  full-sampler AR edge (stu_roll/tea_gen) = "
          f"{_mean('ratio_gen'):.3f}")
    print("  EDGE > ~1.5 at the rungs DMD samples => the AR teacher is a "
          "real oracle; the head is worth fixing.")
    print("  EDGE ~ 1.0 => the AR-served teacher is no better than the "
          "student on its own context; NO supervision layout rescues it.")
    print("  'scorer' ~ 1.0 is EXPECTED and is why gen/dmd_ar_fake_mae_vs_gt "
          "looked like near-parity.")
    print("[probe] =====================================================")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(rows, f, indent=1)
        print(f"[probe] wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
