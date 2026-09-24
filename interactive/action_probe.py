"""Checkpoint-level action-response probe (independent of WorldModelEngine).

Question: does the checkpoint respond to actions at all?  The interactive
player reports "the model does what it wants regardless of input", which is
either (a) the actions never made it into the weights, or (b) engine/UI
plumbing.  This harness removes (b) from the picture: it drives
``utils.play_world_model.WorldModelPlayer`` directly -- the validated build
path -- with no engine, no threads, no UI.

Method
------
Prefill the three ground-truth seed chunks, then roll ``--chunks`` chunks
under each of several CONSTANT action arms.  Every arm starts from an
identically rebuilt post-seed state and consumes the SAME pre-generated noise
tensors, so any divergence between arms is attributable to the action alone.

Arms (values are the two raw action slots, i.e. what the model actually eats):

    neutral      (0.0,  0.0)
    thr+         (+0.8, 0.0)
    thr-         (-0.8, 0.0)
    steer+       (0.0, +0.8)
    steer-       (0.0, -0.8)

Reported per chunk:
  * absolute L2 between arms' latents, and a RELATIVE divergence
    ||a-b|| / (0.5(||a||+||b||)) so "weakened" is distinguishable from "dead";
  * a same-arm noise floor (an arm against itself, re-run) for scale.
  * fingerprints (mean/std) of the conditioning tensors ``_action_modulation``
    and ``_action_tokens`` per arm, which proves the arms differ BEFORE the
    DiT -- if those are identical the harness is broken, not the checkpoint.

Action convention
-----------------
Both `action_forcing_phase1.yaml` (inherited by phase3) and
`action_ode_distill.yaml` declare ``raw_action_dim: 2`` and
``action_dims: [2, 7]`` -- the two slots are PCA components 2 and 7.  The
probe varies slot 0 and slot 1 and is agnostic about which physical control
they name; ``engine.py`` labels slot 0 throttle and slot 1 steer.

Usage
-----
    python -m interactive.action_probe --ckpt <ckpt.pt> --config <cfg.yaml> \
        --seed_zarr ~/20240224003808.zarr --tag B_phase3 --chunks 4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from interactive.engine_api import (  # noqa: E402
    LATENT_C, LATENT_H, LATENT_W, NUM_FRAME_PER_BLOCK, SEED_PREFILL_CHUNKS,
)

log = logging.getLogger("action_probe")

DEFAULT_ARMS: List[Tuple[str, float, float]] = [
    ("neutral", 0.0, 0.0),
    ("thr+", +0.8, 0.0),
    ("thr-", -0.8, 0.0),
    ("steer+", 0.0, +0.8),
    ("steer-", 0.0, -0.8),
]


# ---------------------------------------------------------------------------
# Noise book: identical across arms
# ---------------------------------------------------------------------------
class NoiseBook:
    """Pre-generated noise so every arm walks an identical stochastic path."""

    def __init__(self, n_chunks: int, n_rungs: int, npb: int,
                 device, dtype, seed: int = 1234):
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        shape = (1, npb, LATENT_C, LATENT_H, LATENT_W)
        self.init = [torch.randn(shape, generator=g, dtype=torch.float32).to(device, dtype)
                     for _ in range(n_chunks)]
        # One re-noise draw per rung transition, flattened like the ladder uses it.
        flat_shape = (npb, LATENT_C, LATENT_H, LATENT_W)
        self.rung = [[torch.randn(flat_shape, generator=g, dtype=torch.float32).to(device)
                      for _ in range(max(0, n_rungs - 1))]
                     for _ in range(n_chunks)]


# ---------------------------------------------------------------------------
# Deterministic rollout
# ---------------------------------------------------------------------------
@torch.no_grad()
def commit_gt_chunk(p, latents: torch.Tensor, z2: float, z7: float) -> None:
    """Commit one ground-truth chunk into the KV cache at context_noise."""
    cond = p._action_cond(p._make_action_fa(z2, z7))
    t = torch.full([1, p.num_frame_per_block], p.context_noise,
                   device=p.device, dtype=torch.float32)
    with torch.amp.autocast("cuda", dtype=p.dtype):
        p.generator(
            noisy_image_or_video=latents.to(p.device, p.dtype),
            conditional_dict=cond,
            timestep=t,
            kv_cache=p.kv_cache,
            crossattn_cache=p.crossattn_cache,
            current_start=p.current_start_frame * p.frame_seq_length,
        )
    p.current_start_frame += p.num_frame_per_block


@torch.no_grad()
def step_fixed_noise(p, z2: float, z7: float, book: NoiseBook,
                     chunk_idx: int) -> torch.Tensor:
    """One chunk under (z2, z7) using the book's noise. Returns pred_x0."""
    cond = p._action_cond(p._make_action_fa(z2, z7))
    x = book.init[chunk_idx].clone()
    ts = p.denoising_step_list
    n = int(ts.shape[0])
    pred_x0 = None
    for d in range(n):
        tt = torch.full([1, p.num_frame_per_block], float(ts[d].item()),
                        device=p.device, dtype=torch.float32)
        with torch.amp.autocast("cuda", dtype=p.dtype):
            out = p.generator(
                noisy_image_or_video=x,
                conditional_dict=cond,
                timestep=tt,
                kv_cache=p.kv_cache,
                crossattn_cache=p.crossattn_cache,
                current_start=p.current_start_frame * p.frame_seq_length,
            )
        pred_x0 = out[1]
        if d < n - 1:
            next_t = float(ts[d + 1].item())
            flat = pred_x0.flatten(0, 1).float()
            flat_t = torch.full((flat.shape[0],), next_t,
                                device=p.device, dtype=torch.float32)
            noise = book.rung[chunk_idx][d]
            x = (p.scheduler.add_noise(flat, noise, flat_t)
                 .view(1, p.num_frame_per_block, LATENT_C, LATENT_H, LATENT_W)
                 .to(p.dtype))
    assert pred_x0 is not None

    # Clean cache-refresh forward, exactly as the live path does.
    refresh_t = torch.full([1, p.num_frame_per_block], p.context_noise,
                           device=p.device, dtype=torch.float32)
    with torch.amp.autocast("cuda", dtype=p.dtype):
        p.generator(
            noisy_image_or_video=pred_x0,
            conditional_dict=cond,
            timestep=refresh_t,
            kv_cache=p.kv_cache,
            crossattn_cache=p.crossattn_cache,
            current_start=p.current_start_frame * p.frame_seq_length,
        )
    p.current_start_frame += p.num_frame_per_block
    return pred_x0.float().cpu()


@torch.no_grad()
def run_arm(p, seed_latents: torch.Tensor, prompt_embeds: torch.Tensor,
            z2: float, z7: float, book: NoiseBook, n_chunks: int
            ) -> List[torch.Tensor]:
    """Rebuild the post-seed state, then roll n_chunks under one action."""
    npb = p.num_frame_per_block
    # reset() commits seed chunk 0 and rebuilds caches/rope/RNG-independent state.
    p.reset(seed_latents[:, :npb], prompt_embeds, neutral_action=(0.0, 0.0))
    # Remaining GT seed chunks, committed neutrally like engine.reset() does.
    for c in range(1, SEED_PREFILL_CHUNKS):
        commit_gt_chunk(p, seed_latents[:, c * npb:(c + 1) * npb], 0.0, 0.0)
    return [step_fixed_noise(p, z2, z7, book, c) for c in range(n_chunks)]


# ---------------------------------------------------------------------------
# Conditioning fingerprints
# ---------------------------------------------------------------------------
@torch.no_grad()
def cond_fingerprints(p, arms) -> Dict[str, Dict[str, float]]:
    """mean/std of the conditioning tensors each arm actually feeds the DiT."""
    out: Dict[str, Dict[str, float]] = {}
    for name, z2, z7 in arms:
        cond = p._action_cond(p._make_action_fa(z2, z7))
        rec: Dict[str, float] = {}
        for key in ("_action_modulation", "_action_tokens"):
            t = cond.get(key)
            if t is None:
                rec[f"{key}.mean"] = float("nan")
                rec[f"{key}.std"] = float("nan")
                continue
            tf = t.float()
            rec[f"{key}.mean"] = float(tf.mean())
            rec[f"{key}.std"] = float(tf.std())
            rec[f"{key}.absmax"] = float(tf.abs().max())
        out[name] = rec
    return out


# ---------------------------------------------------------------------------
# Divergence metrics
# ---------------------------------------------------------------------------
def divergence(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    """(absolute L2, relative divergence) between two latent chunks."""
    d = float(torch.linalg.vector_norm(a - b))
    na = float(torch.linalg.vector_norm(a))
    nb = float(torch.linalg.vector_norm(b))
    denom = 0.5 * (na + nb)
    return d, (d / denom if denom > 0 else float("nan"))


# ---------------------------------------------------------------------------
# Action magnitude realism
# ---------------------------------------------------------------------------
def report_action_distribution(seed_zarr: str) -> Optional[dict]:
    """Distribution of tanh-squashed PCA actions in a real ride, if reachable."""
    try:
        from utils.zarr_dataset import _PCA_RAW_SCALES, _encode_motion_pca_raw
    except Exception as exc:
        log.warning("action distribution unavailable (import): %s", exc)
        return None
    for root in ("/home/ashish/frodobots/frodobots_data",):
        try:
            ride = Path(seed_zarr).stem
            motion = _encode_motion_pca_raw(Path(root), ride)  # type: ignore
            if motion is None:
                continue
            sq = np.tanh(np.asarray(motion) / np.asarray(_PCA_RAW_SCALES))
            return {
                "ride": ride,
                "n": int(sq.shape[0]),
                "slot0": {"mean": float(sq[:, 0].mean()), "std": float(sq[:, 0].std()),
                          "p05": float(np.percentile(sq[:, 0], 5)),
                          "p95": float(np.percentile(sq[:, 0], 95)),
                          "absmax": float(np.abs(sq[:, 0]).max())},
                "slot1": {"mean": float(sq[:, 1].mean()), "std": float(sq[:, 1].std()),
                          "p05": float(np.percentile(sq[:, 1], 5)),
                          "p95": float(np.percentile(sq[:, 1], 95)),
                          "absmax": float(np.abs(sq[:, 1]).max())},
            }
        except Exception as exc:
            log.warning("action distribution unavailable (%s): %s", root, exc)
            return None
    return None


# ---------------------------------------------------------------------------
# Video / strip output
# ---------------------------------------------------------------------------
@torch.no_grad()
def decode_arms(arm_latents: Dict[str, List[torch.Tensor]], out_dir: Path,
                device: str, fps: float = 16.0) -> Dict[str, np.ndarray]:
    """Decode every arm with a standalone taew2_1 and write one mp4 each."""
    from interactive.decoders import build_decoder
    dec = build_decoder("taew2_1", device=device, dtype=torch.bfloat16)
    frames: Dict[str, np.ndarray] = {}
    for name, chunks in arm_latents.items():
        dec.reset()
        vid = [dec.decode_chunk(c.to(device=device, dtype=torch.bfloat16)).cpu().numpy()
               for c in chunks]
        arr = np.concatenate(vid, axis=0)
        frames[name] = arr
        _write_mp4(arr, out_dir / f"{name}.mp4", fps)
    return frames


def _write_mp4(frames: np.ndarray, path: Path, fps: float) -> None:
    import subprocess
    h, w = frames.shape[1:3]
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    proc.stdin.write(frames.astype(np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()
    print(f"[probe] wrote {path}")


def save_strip(frames: Dict[str, np.ndarray], out_png: Path,
               frame_idx: int) -> None:
    """Stack one frame per arm into a labelled vertical comparison strip."""
    from PIL import Image, ImageDraw
    names = list(frames)
    idx = min(frame_idx, min(len(frames[n]) for n in names) - 1)
    tiles = []
    for n in names:
        img = Image.fromarray(frames[n][idx]).convert("RGB")
        bar = Image.new("RGB", (img.width, 22), (0, 0, 0))
        ImageDraw.Draw(bar).text((6, 5), f"{n}   frame {idx}", fill=(255, 255, 255))
        cell = Image.new("RGB", (img.width, img.height + 22))
        cell.paste(bar, (0, 0))
        cell.paste(img, (0, 22))
        tiles.append(cell)
    W = tiles[0].width
    H = sum(t.height for t in tiles)
    strip = Image.new("RGB", (W, H))
    y = 0
    for t in tiles:
        strip.paste(t, (0, y))
        y += t.height
    out_png.parent.mkdir(parents=True, exist_ok=True)
    strip.save(out_png)
    print(f"[probe] wrote {out_png}")


# ---------------------------------------------------------------------------
# Config normalisation
# ---------------------------------------------------------------------------
def _normalize_config(config_path: str, out_dir: Path) -> str:
    """Bridge a config-key rename so the ODE lineage still builds.

    ``WorldModelPlayer._build_generator`` reads
    ``action_modulation_activation``.  The phase-1/3 configs set it; the older
    ``action_ode_distill.yaml`` still uses the pre-rename key
    ``action_activation`` (same value, "silu"), so the builder passes
    ``activation=None`` and ``_get_activation`` dies on ``None.lower()``.
    Rather than patch a shared file, resolve the config here and write a
    normalised copy for this run only.
    """
    from omegaconf import OmegaConf
    from utils.play_world_model import _load_config_with_extends

    cfg = _load_config_with_extends(config_path)
    OmegaConf.set_struct(cfg, False)
    if cfg.get("action_modulation_activation", None) is None:
        alias = cfg.get("action_activation", None)
        if alias is not None:
            cfg["action_modulation_activation"] = alias
            out = out_dir / "resolved_config.yaml"
            OmegaConf.save(cfg, str(out))
            print(f"[probe] config: aliased action_activation={alias!r} -> "
                  f"action_modulation_activation; using {out}")
            return str(out)
        print("[probe] WARNING: no action activation in config; builder will "
              "likely fail.")
    return config_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="configs/action_forcing_phase3_dmd.yaml")
    ap.add_argument("--wan_model_path", default="/home/ashish/Wan2.1/Wan2.1-T2V-1.3B/")
    ap.add_argument("--seed_zarr", default=str(Path.home() / "20240224003808.zarr"))
    ap.add_argument("--tag", default="probe")
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--denoising_steps", type=int, default=4)
    ap.add_argument("--kv_cache_chunks", type=int, default=7)
    ap.add_argument("--magnitude", type=float, default=0.8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--noise_seed", type=int, default=1234)
    ap.add_argument("--out_dir", default=str(_REPO_ROOT / "interactive" / "probe_out"))
    ap.add_argument("--strip_frame", type=int, default=-1,
                    help="frame index for the comparison strip (-1 = last)")
    ap.add_argument("--no_video", action="store_true")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [probe] %(levelname)s | %(message)s")

    out_dir = Path(a.out_dir) / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    m = float(a.magnitude)
    arms = [("neutral", 0.0, 0.0), ("thr+", +m, 0.0), ("thr-", -m, 0.0),
            ("steer+", 0.0, +m), ("steer-", 0.0, -m)]

    from utils.play_world_model import WorldModelPlayer, load_seed_from_zarr
    # WorldModelPlayer appends the model name itself, so it wants the PARENT
    # dir; passing .../Wan2.1-T2V-1.3B/ doubles it. Reuse the engine's
    # normalizer, which accepts either form.
    from interactive.engine import _normalize_wan_path

    cfg_path = _normalize_config(a.config, out_dir)

    p = WorldModelPlayer(
        config_path=cfg_path,
        ckpt_path=a.ckpt,
        device=torch.device(a.device),
        wan_model_path=_normalize_wan_path(a.wan_model_path),
        use_ema=False,
        denoising_steps=a.denoising_steps,
        kv_cache_chunks=a.kv_cache_chunks,
        infinity_rope=None,
        weights_dtype="bf16",
    )
    print(f"[probe] npb={p.num_frame_per_block} raw_action_dim={p.raw_action_dim} "
          f"ladder={[round(float(x),1) for x in p.denoising_step_list.tolist()]}")

    # --- seed + prompt ------------------------------------------------------
    need = SEED_PREFILL_CHUNKS * p.num_frame_per_block
    seed = load_seed_from_zarr(a.seed_zarr, need)[:, :need]
    print(f"[probe] seed {a.seed_zarr} -> {tuple(seed.shape)}")
    prompt = p.encode_prompt(
        "a first-person view driving forward along a city sidewalk",
        encode_device="cpu")
    # _action_cond reads self.prompt_embeds, so it must be live before any
    # conditioning is built (reset() would also set it, but fingerprints run
    # first).
    p.prompt_embeds = prompt.to(p.device, p.dtype)

    # --- conditioning fingerprints (proves the arms differ pre-DiT) --------
    fp = cond_fingerprints(p, arms)
    print("\n[probe] ==== conditioning fingerprints (per arm) ====")
    for name in fp:
        bits = "  ".join(f"{k}={v:+.5f}" for k, v in fp[name].items())
        print(f"[probe]   {name:8s} {bits}")
    mods = {n: fp[n].get("_action_modulation.std", float('nan')) for n in fp}
    if len({round(v, 8) for v in mods.values()}) == 1:
        print("[probe]   !! all arms share one _action_modulation std -- "
              "conditioning is NOT varying; harness/model wiring bug.")

    book = NoiseBook(a.chunks, int(p.denoising_step_list.shape[0]),
                     p.num_frame_per_block, p.device, p.dtype, seed=a.noise_seed)

    # --- arms ---------------------------------------------------------------
    arm_latents: Dict[str, List[torch.Tensor]] = {}
    for name, z2, z7 in arms:
        print(f"[probe] --- arm {name} ({z2:+.2f}, {z7:+.2f}) ---")
        arm_latents[name] = run_arm(p, seed, prompt, z2, z7, book, a.chunks)

    # Same-arm repeat = the determinism floor these numbers sit on.
    floor = run_arm(p, seed, prompt, 0.0, 0.0, book, a.chunks)

    # --- divergences --------------------------------------------------------
    print("\n[probe] ==== per-chunk divergence (abs L2 / relative) ====")
    print(f"[probe] {'pair':22s} " + "  ".join(f"chunk{c}" for c in range(a.chunks)))
    results = {}
    pairs = [("neutral", "thr+"), ("neutral", "thr-"), ("thr+", "thr-"),
             ("neutral", "steer+"), ("neutral", "steer-"), ("steer+", "steer-")]
    for x, y in pairs:
        rows = [divergence(arm_latents[x][c], arm_latents[y][c]) for c in range(a.chunks)]
        results[f"{x}|{y}"] = rows
        cells = "  ".join(f"{ab:7.2f}/{rel:6.4f}" for ab, rel in rows)
        print(f"[probe] {x + ' vs ' + y:22s} {cells}")
    floor_rows = [divergence(arm_latents["neutral"][c], floor[c]) for c in range(a.chunks)]
    results["floor(neutral|neutral)"] = floor_rows
    cells = "  ".join(f"{ab:7.2f}/{rel:6.4f}" for ab, rel in floor_rows)
    print(f"[probe] {'FLOOR neutral vs itself':22s} {cells}")

    # --- verdict ------------------------------------------------------------
    last = a.chunks - 1
    steer_rel = results["steer+|steer-"][last][1]
    thr_rel = results["thr+|thr-"][last][1]
    floor_rel = floor_rows[last][1]
    print("\n[probe] ==== verdict ====")
    print(f"[probe] final-chunk relative divergence: "
          f"steer+/steer- = {steer_rel:.4f}, thr+/thr- = {thr_rel:.4f}, "
          f"determinism floor = {floor_rel:.4f}")
    ratio = (steer_rel / floor_rel) if floor_rel > 0 else float("inf")
    print(f"[probe] steer divergence is {ratio:.1f}x the floor")
    if steer_rel <= max(floor_rel * 3, 1e-4):
        print("[probe] VERDICT: NOT action-responsive (at/near determinism floor).")
    elif steer_rel < 0.05:
        print("[probe] VERDICT: WEAKLY action-responsive.")
    else:
        print("[probe] VERDICT: action-responsive.")

    # --- action realism -----------------------------------------------------
    dist = report_action_distribution(a.seed_zarr)
    if dist:
        print(f"\n[probe] real-ride squashed action distribution ({dist['ride']}, "
              f"n={dist['n']}):")
        for k in ("slot0", "slot1"):
            d = dist[k]
            print(f"[probe]   {k}: mean={d['mean']:+.3f} std={d['std']:.3f} "
                  f"p05={d['p05']:+.3f} p95={d['p95']:+.3f} absmax={d['absmax']:.3f}")
        print(f"[probe]   -> probe magnitude +-{m} vs p95 "
              f"{max(abs(dist['slot0']['p95']), abs(dist['slot1']['p95'])):.3f}")
    else:
        print("\n[probe] real-ride action distribution unavailable "
              "(motion_root not reachable) -- magnitude realism unverified.")

    # --- artefacts ----------------------------------------------------------
    summary = {
        "tag": a.tag, "ckpt": a.ckpt, "config": a.config,
        "chunks": a.chunks, "magnitude": m,
        "denoising_steps": int(p.denoising_step_list.shape[0]),
        "fingerprints": fp,
        "divergence": {k: [{"abs": ab, "rel": rel} for ab, rel in v]
                       for k, v in results.items()},
        "action_distribution": dist,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[probe] wrote {out_dir / 'summary.json'}")

    if not a.no_video:
        frames = decode_arms(arm_latents, out_dir, a.device)
        n = min(len(v) for v in frames.values())
        save_strip(frames, out_dir / "compare_strip.png",
                   a.strip_frame if a.strip_frame >= 0 else n - 1)

    p.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
